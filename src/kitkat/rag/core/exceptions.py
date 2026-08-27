"""
Exception hierarchy for the kitkat RAG subsystem.

All RAG-specific exceptions descend from RAGError, which itself inherits
from kitkat.core.exceptions.KitkatError. This ensures that catching
KitkatError will also catch RAG errors, but catching RAGError will not
accidentally catch LLMError or other core errors.
"""

from __future__ import annotations

from typing import Any

from kitkat.core.exceptions import KitkatError


class RAGError(KitkatError):
    """Base class for all RAG-specific errors."""

    pass


# ──────────────────────────────────────────────────────────
# Embedding Errors
# ──────────────────────────────────────────────────────────


class EmbeddingError(RAGError):
    """Base class for all embedding-related errors.

    Attributes:
        provider: The name of the embedding provider.
        details: Additional context or metadata about the error.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.details = details or {}


class EmbeddingProviderInitError(EmbeddingError):
    """Raised when an embedding provider fails to initialize (bad key, network, config)."""


class EmbeddingProviderError(EmbeddingError):
    """Raised for generic runtime embedding failures.

    Attributes:
        status_code: The HTTP status code returned by the provider, if any.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 0,
        provider: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, provider=provider, details=details)
        self.status_code = status_code


class EmbeddingRateLimitError(EmbeddingError):
    """Raised when the embedding provider returns a rate limit error (429).

    Attributes:
        retry_after_s: Suggested wait time in seconds before retrying.
    """

    def __init__(
        self,
        message: str,
        *,
        retry_after_s: float | None = None,
        provider: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, provider=provider, details=details)
        self.retry_after_s = retry_after_s


class EmbeddingTimeoutError(EmbeddingError):
    """Raised when an embedding request exceeds the configured timeout.

    Attributes:
        elapsed_s: The actual time elapsed before timeout.
    """

    def __init__(
        self,
        message: str,
        *,
        elapsed_s: float = 0.0,
        provider: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, provider=provider, details=details)
        self.elapsed_s = elapsed_s


class EmbeddingAuthError(EmbeddingError):
    """Raised when authentication fails for an embedding provider."""


class EmbeddingDimensionError(EmbeddingError):
    """Raised when a provider returns vectors with unexpected dimensions."""


# ──────────────────────────────────────────────────────────
# Chunking Errors
# ──────────────────────────────────────────────────────────


class ChunkingError(RAGError):
    """Base class for chunking-related errors."""


class ChunkingConfigError(ChunkingError):
    """Raised when a chunker is configured with invalid parameters."""


# ──────────────────────────────────────────────────────────
# Vector Store Errors
# ──────────────────────────────────────────────────────────


class VectorStoreError(RAGError):
    """Base class for vector store errors."""


class VectorStoreConnectionError(VectorStoreError):
    """Raised when a connection to the vector store backend fails."""


class VectorStoreOperationError(VectorStoreError):
    """Raised when a vector store operation (add/search/delete) fails."""


class VectorStoreSchemaError(VectorStoreError):
    """Raised when there is a collection/schema mismatch or missing schema."""


# ──────────────────────────────────────────────────────────
# Retrieval Errors
# ──────────────────────────────────────────────────────────


class RetrievalError(RAGError):
    """Base class for retrieval-related errors."""


class RerankingError(RetrievalError):
    """Raised when the reranking step fails."""


# ──────────────────────────────────────────────────────────
# Ingestion Errors
# ──────────────────────────────────────────────────────────


class IngestionError(RAGError):
    """Base class for ingestion pipeline errors."""


class DocumentLoadError(IngestionError):
    """Raised when a document fails to load (e.g., file not found, parse error).

    Attributes:
        source: The source path or URL of the document.
    """

    def __init__(self, message: str, *, source: str = "") -> None:
        super().__init__(message)
        self.source = source


class IngestionPipelineError(IngestionError):
    """Raised when the ingestion pipeline orchestration fails."""
