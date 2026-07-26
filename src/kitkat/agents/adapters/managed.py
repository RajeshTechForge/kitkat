"""ManagedModelAdapter: PydanticAI Model backed by LLMService.

This adapter translates PydanticAI's ``Model`` protocol (v2.x API) into calls
to :class:`~kitkat.service.managed.LLMService`, preserving all provider fidelity
— error mapping, retry policy, and thinking tokens — without any changes to the
provider layer.

Usage::

    from kitkat.agents.adapters.managed import ManagedModelAdapter
    from kitkat.core.enums import ProviderType
    from pydantic_ai import Agent

    adapter = ManagedModelAdapter(
        service=llm_service,
        provider_type=ProviderType.ANTHROPIC,
        default_model="claude-sonnet-4-5",
    )
    agent = Agent(model=adapter, deps_type=MyContext)
    result = await agent.run("Hello!")
"""

from __future__ import annotations

import dataclasses
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

try:
    from pydantic_ai.messages import (
        InstructionPart,
        ModelMessage,
        SystemPromptPart,
        TextPart,
        UserPromptPart,
    )
    from pydantic_ai.messages import (
        ModelResponse as PydanticModelResponse,
    )
    from pydantic_ai.models import (
        Model,
        ModelRequestParameters,
        ModelResponse,
        ModelResponseStreamEvent,
        ModelSettings,
        StreamedResponse,
    )
    from pydantic_ai.usage import RequestUsage
except ImportError as exc:
    raise ImportError(
        "Agent adapters require the 'agents' extra. Install with: pip install kitkat[agents]"
    ) from exc

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator

    from ...core.models import LLMRequest, Message, TokenUsage
    from ...service.managed import LLMService

from ...core.enums import ProviderType, Role


def _to_llm_request(
    messages: list[ModelMessage],
    settings: ModelSettings | None,
    stream: bool = False,
) -> LLMRequest:
    """Translate a PydanticAI message list into a kitkat :class:`~kitkat.core.models.LLMRequest`.

    Handles the four part types a kitkat agent will encounter in practice:
    ``SystemPromptPart``, ``InstructionPart``, ``UserPromptPart`` (string
    content only), and ``TextPart`` (assistant turns).  Unknown part types are
    silently skipped to avoid crashing on pydantic-ai tool/result parts that
    the adapter does not forward to the provider.

    Args:
        messages: Ordered list of :class:`pydantic_ai.messages.ModelMessage` objects
            as supplied by pydantic-ai's run loop.
        settings: Optional per-call model settings (max_tokens, temperature, etc.).
            If ``None``, kitkat provider defaults are used.
        stream: Whether to request a streaming response.

    Returns:
        A populated :class:`~kitkat.core.models.LLMRequest` ready for the service layer.
    """
    from ...core.models import LLMRequest, Message

    domain_messages: list[Message] = []
    for msg in messages:
        for part in msg.parts:
            if isinstance(part, (SystemPromptPart, InstructionPart)):
                domain_messages.append(Message(role=Role.SYSTEM, content=part.content))
            elif isinstance(part, UserPromptPart) and isinstance(part.content, str):
                domain_messages.append(Message(role=Role.USER, content=part.content))
            elif isinstance(part, TextPart):
                role = Role.ASSISTANT if isinstance(msg, PydanticModelResponse) else Role.USER
                domain_messages.append(Message(role=role, content=part.content))

    return LLMRequest(
        messages=domain_messages,
        model=settings.get("model", "") if settings else "",  # type: ignore[typeddict-item]
        max_tokens=settings.get("max_tokens", 2048) if settings else 2048,  # type: ignore[typeddict-item]
        temperature=settings.get("temperature", 0.1) if settings else 0.1,  # type: ignore[typeddict-item]
        stream=stream,
    )


def _to_request_usage(usage: TokenUsage) -> RequestUsage:
    """Map a kitkat :class:`~kitkat.core.models.TokenUsage` to ``pydantic_ai.usage.RequestUsage``.

    Args:
        usage: Token usage reported by the kitkat provider.

    Returns:
        A ``RequestUsage`` instance with ``input_tokens``, ``output_tokens``,
        and, when non-zero, ``thinking_tokens`` stored under ``details``.
    """
    details: dict[str, int] = {}
    if usage.thinking_tokens:
        details["thinking_tokens"] = usage.thinking_tokens
    ru = RequestUsage()
    ru.input_tokens = usage.prompt_tokens
    ru.output_tokens = usage.completion_tokens
    if details:
        ru.details = details
    return ru


@dataclass
class KitkatStreamedResponse(StreamedResponse):
    """PydanticAI ``StreamedResponse`` backed by a kitkat provider stream.

    Bridges the kitkat :class:`~kitkat.core.models.StreamChunk` async iterator
    into pydantic-ai's event protocol by implementing ``_get_event_iterator()``.
    Text deltas are forwarded through
    :meth:`~pydantic_ai.models.ModelResponsePartsManager.handle_text_delta`
    so that downstream pydantic-ai machinery (result extraction, FinalResultEvent)
    works correctly.

    Usage is populated on the final chunk and written into ``self._usage`` so
    the agent's ``RunResult.usage`` is accurate.

    Args:
        model_request_parameters: Passed to ``StreamedResponse.__init__`` for
            tool-call promotion and result-event wiring.
        _kitkat_chunks: Async iterator of :class:`~kitkat.core.models.StreamChunk`
            objects from the kitkat provider.
        _kitkat_model_name: Provider model identifier to surface via ``model_name``.
        _kitkat_provider_name: Human-readable provider label (e.g., ``"anthropic"``).
        _kitkat_provider_url: Optional provider API base URL; ``None`` is valid.
        _kitkat_timestamp: Datetime stamp for the response; defaults to now (UTC).
    """

    _kitkat_chunks: AsyncIterator[Any]
    _kitkat_model_name: str
    _kitkat_provider_name: str
    _kitkat_provider_url: str | None = None
    _kitkat_timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def model_name(self) -> str:
        """Model identifier returned with the streamed response."""
        return self._kitkat_model_name

    @property
    def provider_name(self) -> str:
        """Human-readable provider label (e.g. ``"anthropic"``)."""
        return self._kitkat_provider_name

    @property
    def provider_url(self) -> str | None:
        """Provider API base URL, or ``None`` if not applicable."""
        return self._kitkat_provider_url

    @property
    def timestamp(self) -> datetime:
        """UTC timestamp when the stream was initiated."""
        return self._kitkat_timestamp

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        """Translate kitkat stream chunks into pydantic-ai events.

        For each non-empty text delta, ``_parts_manager.handle_text_delta()``
        emits either a ``PartStartEvent`` (first delta) or a ``PartDeltaEvent``
        (subsequent deltas).  On the final chunk, ``self._usage`` is populated
        from the kitkat ``TokenUsage``.
        """
        async for chunk in self._kitkat_chunks:
            if chunk.delta:
                for event in self._parts_manager.handle_text_delta(
                    vendor_part_id=None,
                    content=chunk.delta,
                ):
                    yield event
            if chunk.is_final and chunk.usage is not None:
                self._usage.input_tokens = chunk.usage.prompt_tokens
                self._usage.output_tokens = chunk.usage.completion_tokens
                if chunk.usage.thinking_tokens:
                    self._usage.details = {"thinking_tokens": chunk.usage.thinking_tokens}

    async def close_stream(self) -> None:
        """No-op: the kitkat async iterator has no underlying connection to close."""


@dataclass
class ManagedModelAdapter(Model):
    """PydanticAI ``Model`` backed by the server-side :class:`~kitkat.service.managed.LLMService`.

    Translates pydantic-ai's ``request()`` / ``request_stream()`` protocol into
    ``LLMService.complete()`` / ``LLMService.stream()`` calls, preserving all
    provider semantics — retry policy, error mapping, thinking tokens — without
    changes to the provider layer.

    Usage::

        adapter = ManagedModelAdapter(
            service=llm_service,
            provider_type=ProviderType.ANTHROPIC,
            default_model="claude-sonnet-4-5",
        )
        agent = Agent(model=adapter, deps_type=MyContext)

    Attributes:
        service: The initialized :class:`~kitkat.service.managed.LLMService`
            instance.  Must have been initialized before use
            (i.e., ``await service.initialize()`` called).
        provider_type: Which registered provider to route requests to.
        default_model: Model identifier to use when the request's ``model`` field
            is empty.  Falls back to the provider's ``DEFAULT_MODEL`` when also
            empty.
    """

    service: LLMService
    provider_type: ProviderType
    default_model: str = ""

    @property
    def system(self) -> str:
        """Human-readable system/provider identifier required by the Model protocol."""
        return self.provider_type.value

    @property
    def model_name(self) -> str:
        """Identifier surfaced to pydantic-ai's run result and usage reporting."""
        return self.default_model or self.provider_type.value

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Execute a single non-streaming completion.

        Args:
            messages: Full conversation history from pydantic-ai's run loop.
            model_settings: Per-request overrides (max_tokens, temperature, etc.).
            model_request_parameters: Tool and output schema context from pydantic-ai.

        Returns:
            A ``ModelResponse`` containing one ``TextPart`` and embedded usage.

        Raises:
            Any :class:`~kitkat.core.exceptions.LLMError` subclass raised by the
            provider passes through unmodified — pydantic-ai will surface it as
            an agent run error.
        """
        req = _to_llm_request(messages, model_settings, stream=False)
        if self.default_model and not req.model:
            req = dataclasses.replace(req, model=self.default_model)

        response = await self.service.complete(req, self.provider_type)
        return ModelResponse(
            parts=[TextPart(content=response.content)],
            usage=_to_request_usage(response.usage),
            model_name=response.model or self.model_name,
            provider_name=self.provider_type.value,
        )

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: Any = None,
    ) -> AsyncGenerator[StreamedResponse]:
        """Execute a streaming completion.

        Yields a :class:`KitkatStreamedResponse` context manager that bridges
        the kitkat stream into pydantic-ai's event protocol.  The caller
        (pydantic-ai's run loop) iterates the response via ``async for``.

        Args:
            messages: Full conversation history from pydantic-ai's run loop.
            model_settings: Per-request overrides.
            model_request_parameters: Tool and output schema context.
            run_context: Unused; accepted for protocol compatibility.

        Yields:
            A :class:`KitkatStreamedResponse` ready to be iterated.
        """
        req = _to_llm_request(messages, model_settings, stream=True)
        if self.default_model and not req.model:
            req = dataclasses.replace(req, model=self.default_model)

        chunk_iter = self.service.stream(req, self.provider_type)
        yield KitkatStreamedResponse(
            model_request_parameters=model_request_parameters,
            _kitkat_chunks=chunk_iter,
            _kitkat_model_name=self.default_model or self.provider_type.value,
            _kitkat_provider_name=self.provider_type.value,
        )
