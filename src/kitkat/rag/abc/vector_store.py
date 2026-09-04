# src/kitkat/rag/abc/vector_store.py
"""Abstract contract for vector storage backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kitkat.rag.core.enums import DistanceMetric
    from kitkat.rag.core.models import Chunk


class VectorStore(ABC):
    """Abstract contract for vector storage backends.

    All operations are async. Backends must handle their own connection
    pooling, retries, and error mapping to VectorStoreError subclasses.
    """

    BACKEND_TYPE: str  # e.g. "in_memory", "qdrant", "pgvector"
    SUPPORTED_METRICS: frozenset[DistanceMetric]

    @abstractmethod
    async def initialize(self) -> None:
        """Create connection, ensure schema/collection exists."""

    @abstractmethod
    async def shutdown(self) -> None:
        """Close connections, release resources."""

    @abstractmethod
    async def add(self, chunks: list[Chunk], *, collection: str = "default") -> int:
        """Insert chunks with embeddings.

        Args:
            chunks: List of Chunk objects. Must have `embedding` populated.
            collection: Target collection/namespace name.

        Returns:
            The number of chunks successfully inserted.

        Raises:
            VectorStoreOperationError: If the operation fails.
            EmbeddingDimensionError: If embedding dims mismatch collection.
        """

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        *,
        collection: str = "default",
        top_k: int = 5,
        filter: dict[str, str | int | float | bool] | None = None,
        metric: DistanceMetric | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Search by vector similarity.

        Args:
            query_embedding: The vector to search for.
            collection: Target collection/namespace name.
            top_k: Maximum number of results to return.
            filter: Optional metadata filter (exact match).
            metric: Distance metric override. Falls back to default.

        Returns:
            List of (chunk, score) tuples sorted by relevance (highest first).
        """

    @abstractmethod
    async def delete(self, chunk_ids: list[str], *, collection: str = "default") -> int:
        """Delete chunks by ID.

        Args:
            chunk_ids: List of chunk UUIDs to delete.
            collection: Target collection/namespace name.

        Returns:
            The number of chunks successfully deleted.
        """

    @abstractmethod
    async def get(self, chunk_ids: list[str], *, collection: str = "default") -> list[Chunk]:
        """Retrieve chunks by ID without vector search.

        Args:
            chunk_ids: List of chunk UUIDs to retrieve.
            collection: Target collection/namespace name.

        Returns:
            List of retrieved Chunk objects.
        """

    @abstractmethod
    async def count(self, *, collection: str = "default") -> int:
        """Return the total number of chunks in a collection."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Lightweight liveness probe for the backend."""

    async def __aenter__(self) -> VectorStore:
        """Initialize the store on context entry."""
        await self.initialize()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Shut down the store on context exit."""
        await self.shutdown()
