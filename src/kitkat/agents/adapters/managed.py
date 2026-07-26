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
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

try:
    from pydantic_ai.messages import (
        InstructionPart,
        ModelMessage,
        ModelResponse,
        ModelResponseStreamEvent,
        SystemPromptPart,
        TextPart,
        ToolCallPart,
        ToolReturnPart,
        UserPromptPart,
    )
    from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
    from pydantic_ai.usage import RequestUsage

except ImportError as exc:
    raise ImportError(
        "Agent adapters require the 'agents' extra. Install with: pip install kitkat[agents]"
    ) from exc

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator

    from pydantic_ai.settings import ModelSettings

    from ...core.models import LLMRequest, Message, TokenUsage
    from ...service.managed import LLMService

from ...core.enums import ProviderType, Role


# TODO: This is a text-based fallback. The ideal long-term fix is to extend
# LLMRequest / Message to carry structured tool-call content natively
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

            elif isinstance(part, UserPromptPart):
                content = part.content if isinstance(part.content, str) else str(part.content)
                domain_messages.append(Message(role=Role.USER, content=content))

            elif isinstance(part, TextPart):
                role = Role.ASSISTANT if isinstance(msg, ModelResponse) else Role.USER
                domain_messages.append(Message(role=role, content=part.content))

            elif isinstance(part, ToolCallPart):
                tool_name = part.tool_name
                args = part.args if isinstance(part.args, str) else json.dumps(part.args)
                domain_messages.append(
                    Message(role=Role.ASSISTANT, content=f"[tool_call:{tool_name}({args})]")
                )
            elif isinstance(part, ToolReturnPart):
                result = part.content if isinstance(part.content, str) else json.dumps(part.content)
                domain_messages.append(
                    Message(role=Role.USER, content=f"[tool_result:{part.tool_name}] {result}")
                )

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
    return RequestUsage(
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        details=details or None,
    )


class KitkatStreamedResponse(StreamedResponse):
    """PydanticAI StreamedResponse backed by a kitkat provider stream."""

    def __init__(
        self,
        *,
        model_request_parameters: ModelRequestParameters,
        chunks: AsyncIterator[Any],
        model_name: str,
        provider_name: str,
        provider_url: str | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        super().__init__(model_request_parameters=model_request_parameters)
        self._chunks = chunks
        self._model_name = model_name
        self._provider_name = provider_name
        self._provider_url = provider_url
        self._timestamp = timestamp or datetime.now(UTC)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def provider_url(self) -> str | None:
        return self._provider_url

    @property
    def timestamp(self) -> datetime:
        return self._timestamp

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        async for chunk in self._chunks:
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
        pass


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
            chunks=chunk_iter,
            model_name=self.default_model or self.provider_type.value,
            provider_name=self.provider_type.value,
        )
