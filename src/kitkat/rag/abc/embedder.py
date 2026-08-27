"""Abstract foundational contract for embedding providers."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from kitkat.rag.core.models import EmbeddingRequest, EmbeddingResult

if TYPE_CHECKING:
    from kitkat.rag.core.enums import EmbeddingProviderType


class EmbeddingProvider(ABC):
    """Abstract contract orchestrating standard remote embedding calls synchronously.

    Attributes:
        PROVIDER_TYPE: Static enum distinguishing vendor origin.
        DEFAULT_MODEL: Known functional default model internally assigned.
        RETRY_POLICY: Vendor-specific execution resilience behavior limits.
    """

    PROVIDER_TYPE: EmbeddingProviderType
    DEFAULT_MODEL: str

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._initialized = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    async def initialize(self) -> None:
        """Validates internal configuration establishing underlying HTTP pools.

        Raises:
            EmbeddingProviderInitError: Underlying resources or dependencies failed.
        """

    @abstractmethod
    async def shutdown(self) -> None:
        """Dispose resources proactively terminating external background tasks cleanly."""

    async def __aenter__(self) -> EmbeddingProvider:
        """Initialize the provider on context entry."""
        await self.initialize()
        return self

    async def __aexit__(self, *_: Any) -> None:
        """Shut down the provider on context exit."""
        await self.shutdown()

    # ------------------------------------------------------------------
    # Core — implement this in every provider
    # ------------------------------------------------------------------

    @abstractmethod
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        """Execute requested external embeddings adhering rigorously inside timeouts.

        Args:
            request: Payload strictly enclosing exact contents dynamically processed.

        Returns:
            Resolved identically aligned multidimensional embeddings completely verified.

        Raises:
            EmbeddingTimeoutError: Network limits reliably tripped natively stopping execution.
            EmbeddingRateLimitError: Retries failed mitigating explicit external concurrency limits.
            EmbeddingAuthError: Invalid key immediately flagged locally terminating execution.
            EmbeddingDimensionError: Upstream payload width randomly natively mismatched.
            EmbeddingProviderError: Arbitrary provider failures randomly reported upstream.
        """

    # ------------------------------------------------------------------
    # Health & introspection
    # ------------------------------------------------------------------

    @abstractmethod
    async def health_check(self) -> bool:
        """Validate backend integrity quickly via isolated non-failing probing.

        Returns:
            True if the backend is healthy, False otherwise.
        """

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the static upstream dimensionality.

        Returns:
            The number of dimensions produced by this provider.
        """

    # ------------------------------------------------------------------
    # Convenience wrappers (concrete — do not override)
    # ------------------------------------------------------------------

    async def embed_query(
        self,
        text: str,
        *,
        timeout: float | None = None,
    ) -> list[float]:
        """Embed an isolated string explicitly optimized for retrieval queries natively.

        Args:
            text: Singular isolated lookup query conditionally routed inline.
            timeout: Optional isolated execution limit strictly restricting payload.

        Returns:
            Vectorized mathematical signature precisely isolating query dimensions logically.
        """
        self._assert_initialized()
        req = EmbeddingRequest(
            texts=[text],
            is_query=True,
            timeout=timeout,
        )
        result = await self.embed(req)
        return result.vectors[0]

    async def embed_documents(
        self,
        texts: list[str],
        *,
        timeout: float | None = None,
    ) -> list[list[float]]:
        """Resolve clustered documents synchronously processing structural embeddings linearly.

        Args:
            texts: Block chunks cleanly mapped identically via isolated processing.
            timeout: Granular operation ceiling optionally guarding background logic.

        Returns:
            Resolved sequential float arrays directly mirroring injected textual positions.
        """
        self._assert_initialized()
        req = EmbeddingRequest(
            texts=texts,
            is_query=False,
            timeout=timeout,
        )
        result = await self.embed(req)
        return result.vectors

    # ------------------------------------------------------------------
    # Shared helpers (concrete — available to all providers)
    # ------------------------------------------------------------------

    def _assert_initialized(self) -> None:
        """Assert that the provider has been initialized.

        Raises:
            RuntimeError: If initialize() was not called.
        """
        if not self._initialized:
            raise RuntimeError(
                f"{self.__class__.__name__}.initialize() must be called before "
                "embedding. Use the async context manager."
            )

    def _elapsed_ms(self, start: float) -> float:
        """Calculate elapsed milliseconds since the start time.

        Args:
            start: The start time in monotonic seconds.

        Returns:
            Elapsed time in milliseconds.
        """
        return (time.monotonic() - start) * 1_000

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a string representation of the provider."""
        status = "ready" if self._initialized else "uninitialised"
        return (
            f"<{self.__class__.__name__} "
            f"provider={self.PROVIDER_TYPE.value!r} "
            f"model={self.DEFAULT_MODEL!r} "
            f"dimensions={self.dimensions} "
            f"status={status}>"
        )
