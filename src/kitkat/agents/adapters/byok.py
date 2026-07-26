"""BYOKModelAdapter: PydanticAI Model backed by BYOKLLMService.

The :class:`~kitkat.service.byok.BYOKLLMService` **must** be entered via its
async context manager (``__aenter__`` called) before constructing this adapter.
The adapter borrows the service for the duration of the agent run — it does not
manage the service's lifecycle.

Usage::

    from kitkat.agents.adapters.byok import BYOKModelAdapter
    from kitkat.service.byok import BYOKLLMService
    from kitkat.core.enums import ProviderType
    from pydantic_ai import Agent

    async with BYOKLLMService(ProviderType.OPENAI, user_api_key, model) as byok:
        adapter = BYOKModelAdapter(byok_service=byok)
        agent = Agent(model=adapter, deps_type=UserContext)
        result = await agent.run(user_message, deps=ctx)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

try:
    from pydantic_ai.messages import ModelMessage, TextPart
    from pydantic_ai.models import (
        Model,
        ModelRequestParameters,
        ModelResponse,
        ModelSettings,
        StreamedResponse,
    )
except ImportError as exc:
    raise ImportError(
        "Agent adapters require the 'agents' extra. Install with: pip install kitkat[agents]"
    ) from exc

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from ...service.byok import BYOKLLMService

from .managed import KitkatStreamedResponse, _to_llm_request, _to_request_usage


@dataclass
class BYOKKitkatStreamedResponse(KitkatStreamedResponse):
    """Streaming response for the BYOK adapter.

    Identical to :class:`~kitkat.agents.adapters.managed.KitkatStreamedResponse`
    — exists as a named type so that callers can distinguish between managed
    and BYOK stream responses in logs and traces.
    """


@dataclass
class BYOKModelAdapter(Model):
    """PydanticAI ``Model`` backed by a per-request :class:`~kitkat.service.byok.BYOKLLMService`.

    Unlike :class:`~kitkat.agents.adapters.managed.ManagedModelAdapter`, this
    adapter does **not** own the service lifecycle.  The caller must enter the
    ``BYOKLLMService`` context manager before constructing the adapter and exit
    it after the agent run completes.

    Auth failures (invalid API key) surface on the first inference call rather
    than at adapter construction time — this is intentional.  The BYOK path
    avoids credential probes to reduce latency and billable overhead.

    Usage::

        async with BYOKLLMService(ProviderType.OPENAI, key, model) as byok:
            adapter = BYOKModelAdapter(byok_service=byok)
            result = await Agent(model=adapter).run("Hello!")

    Attributes:
        byok_service: An already-initialized :class:`~kitkat.service.byok.BYOKLLMService`.
    """

    byok_service: BYOKLLMService

    @property
    def system(self) -> str:
        """Human-readable system/provider identifier required by the Model protocol."""
        return repr(self.byok_service)

    @property
    def model_name(self) -> str:
        """Identifier constructed from the BYOK service's provider and model."""
        return repr(self.byok_service)

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Execute a single non-streaming BYOK completion.

        Args:
            messages: Full conversation history from pydantic-ai's run loop.
            model_settings: Per-request overrides (max_tokens, temperature, etc.).
            model_request_parameters: Tool and output schema context from pydantic-ai.

        Returns:
            A ``ModelResponse`` containing one ``TextPart`` and embedded usage.

        Raises:
            Any :class:`~kitkat.core.exceptions.LLMError` subclass raised by the
            provider passes through unmodified.
        """
        req = _to_llm_request(messages, model_settings, stream=False)
        response = await self.byok_service.complete(req)
        return ModelResponse(
            parts=[TextPart(content=response.content)],
            usage=_to_request_usage(response.usage),
            model_name=response.model or self.model_name,
            provider_name=response.provider.value,
        )

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: Any = None,
    ) -> AsyncGenerator[StreamedResponse]:
        """Execute a streaming BYOK completion.

        Yields a :class:`BYOKKitkatStreamedResponse` that bridges the kitkat
        stream into pydantic-ai's event protocol.

        Args:
            messages: Full conversation history from pydantic-ai's run loop.
            model_settings: Per-request overrides.
            model_request_parameters: Tool and output schema context.
            run_context: Unused; accepted for protocol compatibility.

        Yields:
            A :class:`BYOKKitkatStreamedResponse` ready to be iterated.
        """
        req = _to_llm_request(messages, model_settings, stream=True)
        chunk_iter = self.byok_service.stream(req)
        yield BYOKKitkatStreamedResponse(
            model_request_parameters=model_request_parameters,
            _kitkat_chunks=chunk_iter,
            _kitkat_model_name=self.model_name,
            _kitkat_provider_name=repr(self.byok_service),
        )
