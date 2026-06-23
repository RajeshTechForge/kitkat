"""The LLMProvider abstract base class.

Every concrete provider (Anthropic, OpenAI, Gemini, or a custom third-party
provider) must sub-class :class:`LLMProvider` and implement all abstract
methods.  The library's service layer works exclusively with this ABC — it
never imports concrete provider classes directly.

Implementing a custom provider::

    from kitkat.abc import LLMProvider
    from kitkat.core import (
        LLMRequest, LLMResponse, ProviderCapabilities, ProviderType,
        RetryPolicy, StreamChunk,
    )
    from collections.abc import AsyncIterator

    class MyProvider(LLMProvider):
        PROVIDER_TYPE = ProviderType.OPENAI        # reuse an existing slot …
        DEFAULT_MODEL = "my-model-v1"
        CAPABILITIES = ProviderCapabilities(
            supports_streaming=True,
            supports_thinking=False,
            max_context_tokens=32_768,
            provider_type=ProviderType.OPENAI,
        )

        async def initialize(self) -> None:
            self._client = MySDKClient(api_key=self._config["api_key"])
            self._initialized = True

        async def shutdown(self) -> None:
            await self._client.aclose()
            self._initialized = False

        async def _init_client_only(self) -> None:
            if self._initialized:
                return
            self._client = MySDKClient(api_key=self._config["api_key"])
            self._initialized = True

        async def complete(self, request: LLMRequest) -> LLMResponse:
            ...

        async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
            ...

        async def health_check(self) -> bool:
            ...

        def count_tokens(self, text: str) -> int:
            from kitkat._internal.tokenizers import count_tokens_tiktoken
            return count_tokens_tiktoken(text)
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ..core.enums import ProviderType

from .._internal.retry import execute_with_retry
from ..core.models import (
    LLMRequest,
    LLMResponse,
    Message,
    ProviderCapabilities,
    RetryPolicy,
    StreamChunk,
)

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for all LLM provider implementations.

    Concrete providers inherit from this class and implement the five
    abstract methods below.  The shared helpers (:meth:`complete_with_retry`,
    :meth:`run_sync`, :meth:`_assert_initialized`, …) are provided here so
    providers don't duplicate boilerplate.

    Lifecycle::

        async with MyProvider(config) as provider:
            response = await provider.complete(request)

    Or explicitly::

        provider = MyProvider(config)
        await provider.initialize()
        try:
            response = await provider.complete(request)
        finally:
            await provider.shutdown()
    """

    # -- Class-level attributes providers MUST declare --------------------

    PROVIDER_TYPE: ProviderType
    """Canonical enum value identifying this provider."""

    DEFAULT_MODEL: str
    """Default model identifier used when :attr:`LLMRequest.model` is empty."""

    CAPABILITIES: ProviderCapabilities
    """Static feature-flag descriptor queried by the service layer."""

    RETRY_POLICY: RetryPolicy = RetryPolicy()
    """Default retry policy; concrete providers may override at class level."""

    # -- Constructor -------------------------------------------------------

    def __init__(self, config: dict[str, Any]) -> None:
        """Create the provider with a raw configuration dictionary.

        Args:
            config: Provider-specific key/value pairs (API key, model, etc.).
                Concrete providers typically accept a typed ``*Config``
                dataclass and call ``super().__init__(config.__dict__)``.
        """
        self._config = config
        self._initialized = False
        logger.debug("%s provider created.", self.__class__.__name__)

    # -- Lifecycle (abstract) ---------------------------------------------

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider: create the HTTP client and probe credentials.

        This is the *full* initialization path. Callers using managed keys
        should always prefer this over ``_init_client_only``.

        Raises:
            LLMProviderInitError: If the provider fails to start due to
                configuration or credential errors.
        """

    @abstractmethod
    async def shutdown(self) -> None:
        """Gracefully release all resources associated with the provider."""

    @abstractmethod
    async def _init_client_only(self) -> None:
        """Create the HTTP client *without* running a credential probe.

        This lightweight initialization path is used by
        :class:`~kitkat.service.byok.BYOKLLMService` for BYOK requests.
        Auth failures surface on the first inference call rather than a
        pre-flight probe, avoiding extra latency and billable requests per
        user key.

        Concrete implementations must:

        1. Guard against double-initialization (idempotent — return early if
           ``self._initialized`` is already ``True``).
        2. Instantiate the provider-specific async HTTP client.
        3. Set ``self._initialized = True`` after successful client creation.

        Raises:
            LLMProviderInitError: If the underlying client cannot be created.
        """

    # -- Async context manager support ------------------------------------

    async def __aenter__(self) -> LLMProvider:
        """Initialize the provider upon context entry."""
        await self.initialize()
        return self

    async def __aexit__(self) -> None:
        """Ensure provider shutdown on context manager exit."""
        await self.shutdown()

    # -- Core inference (abstract) ----------------------------------------

    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Execute a single non-streaming completion attempt.

        This method does **not** apply retry logic. For retry-wrapped
        completion use :meth:`complete_with_retry`.

        Args:
            request: The generation request.

        Returns:
            The completed response from the provider.

        Raises:
            LLMTimeoutError: If the request exceeds the configured timeout.
            LLMRateLimitError: On HTTP 429.
            LLMTokenLimitError: If the prompt exceeds the context window.
            LLMProviderError: On any other provider-side failure.
        """

    @abstractmethod
    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        """Yield token deltas as an async stream.

        Args:
            request: The streaming generation request.

        Yields:
            :class:`~kitkat.core.models.StreamChunk` objects — one per
            token delta.  The final chunk has ``is_final=True`` and
            carries aggregated ``usage``, ``model``, ``provider``,
            ``finish_reason``, and ``latency_ms``.

        Raises:
            LLMTimeoutError: If the stream connection times out.
            LLMRateLimitError: If rate-limited mid-stream.
            LLMTokenLimitError: If the context window is exceeded.
            LLMProviderError: On any other streaming error.
        """
        # The ``yield`` below satisfies the type-checker's requirement that an
        # ``@abstractmethod`` decorated as ``AsyncIterator`` is a generator.
        # Concrete providers should replace the entire body.
        raise NotImplementedError  # pragma: no cover
        yield  # type: ignore[misc]  # makes this an async generator

    # -- Health & introspection (abstract) --------------------------------

    @abstractmethod
    async def health_check(self) -> bool:
        """Perform a lightweight liveness probe.

        Returns:
            ``True`` if the provider is reachable and credentials are valid.
        """

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Estimate the token count for a piece of text.

        Providers should delegate to
        :func:`~kitkat._internal.tokenizers.count_tokens_tiktoken`
        or their SDK's native token counter.

        Args:
            text: The text to evaluate.

        Returns:
            Estimated token count (≥ 1 for non-empty input).
        """

    # -- Shared helpers ---------------------------------------------------

    def count_prompt_tokens(self, messages: list[Message]) -> int:
        """Estimate total token count for a list of messages.

        Concatenates all message contents with a single-space separator and
        delegates to :meth:`count_tokens`.

        Args:
            messages: The conversation messages to estimate.

        Returns:
            Estimated token count, or 0 for an empty list.
        """
        if not messages:
            return 0
        return self.count_tokens(" ".join(m.content for m in messages))

    def _assert_initialized(self) -> None:
        """Raise if the provider has not been initialized.

        Raises:
            RuntimeError: If :meth:`initialize` (or :meth:`_init_client_only`)
                has not been successfully called.
        """
        if not self._initialized:
            raise RuntimeError(
                f"{self.__class__.__name__}.initialize() must be called "
                "before making inference requests. Use the async context manager."
            )

    def _build_base_response_kwargs(
        self,
        request: LLMRequest,  # noqa: ARG002  (kept for API compatibility)
        start_time: float,
    ) -> dict[str, Any]:
        """Build common tracing fields for every response.

        Args:
            request: The original :class:`~kitkat.core.models.LLMRequest`.
            start_time: Monotonic clock value recorded before the API call.

        Returns:
            Dict with ``provider`` and ``latency_ms`` keys ready to unpack
            into :class:`~kitkat.core.models.LLMResponse`.
        """
        return {
            "provider": self.PROVIDER_TYPE,
            "latency_ms": (time.monotonic() - start_time) * 1_000,
        }

    async def complete_with_retry(
        self,
        request: LLMRequest,
        *,
        policy: RetryPolicy | None = None,
    ) -> LLMResponse:
        """Execute a completion request with exponential back-off retry.

        Delegates to :func:`~kitkat._internal.retry.execute_with_retry`,
        which handles non-retriable errors (auth, token limit, content
        filter) by re-raising immediately.

        Args:
            request: The completion request.
            policy: Override the provider's class-level ``RETRY_POLICY``.

        Returns:
            The completed response after a successful attempt.

        Raises:
            LLMTimeoutError: If all retries time out.
            LLMRateLimitError: If all rate-limit retries are exhausted.
            LLMProviderError: On unrecoverable provider errors.
        """
        p = policy or getattr(self, "RETRY_POLICY", RetryPolicy())
        return await execute_with_retry(
            func=lambda: self.complete(request),
            policy=p,
            provider_name=self.__class__.__name__,
        )

    def run_sync(self, request: LLMRequest) -> LLMResponse:
        """Execute a completion synchronously (blocks the calling thread).

        Useful for scripts and tests that do not run inside an asyncio event
        loop.  **Do not call from within a running loop** — use
        ``await provider.complete(request)`` instead.

        Args:
            request: The request to send to the provider.

        Returns:
            The provider response.

        Raises:
            RuntimeError: If called from within a running asyncio event loop.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass  # No running loop — safe to proceed
        else:
            raise RuntimeError(
                "run_sync() cannot be called from within a running event loop. "
                "Use 'await provider.complete(request)' instead."
            )
        return asyncio.run(self.complete(request))

    # -- Representation ---------------------------------------------------

    def __repr__(self) -> str:
        status = "ready" if self._initialized else "uninitialised"
        provider_type = getattr(self, "PROVIDER_TYPE", "unknown")
        model = getattr(self, "DEFAULT_MODEL", "unknown")
        return (
            f"<{self.__class__.__name__} "
            f"provider={getattr(provider_type, 'value', provider_type)!r} "
            f"model={model!r} "
            f"status={status}>"
        )
