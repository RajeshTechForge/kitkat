"""Domain models for the RAG subsystem (dataclasses)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from kitkat.core.models import RetryPolicy

__all__ = ["RetryPolicy"]


# ──────────────────────────────────────────────────────────
# RAG Domain Models
# ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Document:
    """A source document before chunking.

    Attributes:
        id: Unique identifier.
        content: The text content of the document.
        metadata: Arbitrary metadata associated with the document.
        source: File path, URL, or identifier.
        content_type: MIME type of the content.
        created_at: UTC timestamp of creation.
    """

    content: str
    id: str = field(default_factory=lambda: str(uuid4()))
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)
    source: str = ""
    content_type: str = "text/plain"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("Document.content must not be empty or whitespace-only.")


@dataclass(frozen=True)
class Chunk:
    """A semantically coherent piece of a document.

    Attributes:
        id: Unique identifier.
        document_id: ID of the parent Document.
        content: The text content of the chunk.
        embedding: Vector representation, populated after embedding.
        metadata: Arbitrary metadata.
        chunk_index: Ordinal position within parent document.
        start_char: Character offset in source document.
        end_char: End character offset.
        token_count: Estimated token count.
    """

    document_id: str
    content: str
    id: str = field(default_factory=lambda: str(uuid4()))
    embedding: list[float] | None = None
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)
    chunk_index: int = 0
    start_char: int = 0
    end_char: int = 0
    token_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def has_embedding(self) -> bool:
        """Check if the chunk has been embedded."""
        return self.embedding is not None


@dataclass(frozen=True)
class RetrievalResult:
    """A single retrieved chunk with relevance scoring."""

    chunk: Chunk
    score: float
    rank: int
    retrieved_by: str
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────
# Embedding Models (adapted from base.py)
# ──────────────────────────────────────────────────────────


@dataclass
class EmbeddingRequest:
    """Consolidated payload configuring a single embedding operation.

    Attributes:
        texts: List of strings to embed.
        is_query: True if texts are search queries, False if documents.
        model: Model identifier to use.
        metadata: Arbitrary metadata.
        timeout: Request timeout in seconds.
    """

    texts: list[str]
    is_query: bool = False
    model: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    timeout: float | None = 30.0

    def __post_init__(self) -> None:
        if not self.texts:
            raise ValueError("EmbeddingRequest.texts must contain at least one string.")
        if any(not isinstance(t, str) for t in self.texts):
            raise TypeError("EmbeddingRequest.texts must be a list of str.")
        empty = [i for i, t in enumerate(self.texts) if not t.strip()]
        if empty:
            raise ValueError(
                f"EmbeddingRequest.texts contains empty/whitespace-only strings at indices: {empty}"
            )
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError(
                f"EmbeddingRequest.timeout must be positive or None, got {self.timeout}"
            )


@dataclass
class EmbeddingResult:
    """Standardized payload encapsulating embedding outputs unconditionally.

    Attributes:
        vectors: Produced float vectors ordered consistently with the request.
        model: Exact string identifier returned by the provider.
        provider: Resolved source vendor.
        prompt_tokens: Total tokens requested.
        latency_ms: Round-trip latency in milliseconds.
    """

    vectors: list[list[float]]
    model: str
    provider: str
    prompt_tokens: int = 0
    latency_ms: float = 0.0

    @property
    def dimensions(self) -> int:
        """Length of the internal embedding vectors."""
        return len(self.vectors[0]) if self.vectors else 0

    @property
    def count(self) -> int:
        """Total vectors collected."""
        return len(self.vectors)

    def __post_init__(self) -> None:
        if not self.vectors:
            raise ValueError("EmbeddingResult.vectors must not be empty.")
        first_dim = len(self.vectors[0])
        for i, vec in enumerate(self.vectors[1:], start=1):
            if len(vec) != first_dim:
                raise ValueError(
                    f"EmbeddingResult: vector at index {i} has {len(vec)} dimensions, "
                    f"expected {first_dim}. Provider returned inconsistent dimensions."
                )
