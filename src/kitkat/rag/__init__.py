"""kitkat RAG subsystem: Retrieval-Augmented Generation support."""

from __future__ import annotations

# ── ABC ────────────────────────────
from .abc.chunker import Chunker
from .abc.embedder import EmbeddingProvider
from .abc.reranker import Reranker
from .abc.retriever import Retriever
from .abc.vector_store import VectorStore

# ── Core ────────────────────────────
from .core.enums import (
    ChunkingStrategy,
    DistanceMetric,
    EmbeddingProviderType,
    RerankerType,
    RetrievalStrategy,
    VectorBackendType,
)
from .core.exceptions import (
    ChunkingConfigError,
    ChunkingError,
    DocumentLoadError,
    EmbeddingAuthError,
    EmbeddingDimensionError,
    EmbeddingError,
    EmbeddingProviderError,
    EmbeddingProviderInitError,
    EmbeddingRateLimitError,
    EmbeddingTimeoutError,
    IngestionError,
    IngestionPipelineError,
    RerankingError,
    RetrievalError,
    VectorStoreConnectionError,
    VectorStoreError,
    VectorStoreOperationError,
    VectorStoreSchemaError,
)
from .core.models import (
    Chunk,
    Document,
    EmbeddingRequest,
    EmbeddingResult,
    RetrievalResult,
)

__all__ = [
    # abc
    "Chunker",
    "EmbeddingProvider",
    "Reranker",
    "Retriever",
    "VectorStore",
    # core
    "ChunkingStrategy",
    "DistanceMetric",
    "EmbeddingProviderType",
    "RerankerType",
    "RetrievalStrategy",
    "VectorBackendType",
    # Exceptions
    "ChunkingConfigError",
    "ChunkingError",
    "DocumentLoadError",
    "EmbeddingAuthError",
    "EmbeddingDimensionError",
    "EmbeddingError",
    "EmbeddingProviderError",
    "EmbeddingProviderInitError",
    "EmbeddingRateLimitError",
    "EmbeddingTimeoutError",
    "IngestionError",
    "IngestionPipelineError",
    "RerankingError",
    "RetrievalError",
    "VectorStoreConnectionError",
    "VectorStoreError",
    "VectorStoreOperationError",
    "VectorStoreSchemaError",
    # Models
    "Chunk",
    "Document",
    "EmbeddingRequest",
    "EmbeddingResult",
    "RetrievalResult",
]
