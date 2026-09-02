# src/kitkat/rag/stores/in_memory.py
"""Zero-dependency in-process vector store."""

from __future__ import annotations

import asyncio
import logging

from kitkat.rag._internal.similarity import (
    cosine_similarity,
    dot_product,
    euclidean_distance,
)
from kitkat.rag.abc.vector_store import VectorStore
from kitkat.rag.core.enums import DistanceMetric
from kitkat.rag.core.exceptions import (
    EmbeddingDimensionError,
    VectorStoreOperationError,
)
from kitkat.rag.core.models import Chunk

logger = logging.getLogger(__name__)


class InMemoryVectorStore(VectorStore):
    """Pure-Python in-memory vector store.

    Suitable for development, unit tests, and small datasets (<50k chunks).
    Uses an exhaustive O(n) scan for similarity search.
    """

    BACKEND_TYPE = "in_memory"
    SUPPORTED_METRICS = frozenset(
        {DistanceMetric.COSINE, DistanceMetric.DOT, DistanceMetric.EUCLIDEAN}
    )

    def __init__(self, *, default_metric: DistanceMetric = DistanceMetric.COSINE) -> None:
        self._default_metric = default_metric
        self._collections: dict[str, dict[str, Chunk]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the in-memory store."""
        self._initialized = True
        logger.debug("InMemoryVectorStore initialized.")

    async def shutdown(self) -> None:
        """Clear memory and mark as uninitialized."""
        self._collections.clear()
        self._locks.clear()
        self._initialized = False
        logger.debug("InMemoryVectorStore shut down.")

    def _get_lock(self, collection: str) -> asyncio.Lock:
        if collection not in self._locks:
            self._locks[collection] = asyncio.Lock()
        return self._locks[collection]

    async def add(self, chunks: list[Chunk], *, collection: str = "default") -> int:
        """Insert chunks into the in-memory collection."""
        if not self._initialized:
            raise VectorStoreOperationError("Store not initialized. Use async with.")

        if not chunks:
            return 0

        # Validate dimensions based on the first chunk
        first_embedding = chunks[0].embedding
        if first_embedding is None:
            raise EmbeddingDimensionError(
                "Chunks must have embeddings before adding to store.",
                provider="in_memory",
            )
        expected_dim = len(first_embedding)

        for chunk in chunks[1:]:
            if chunk.embedding is None or len(chunk.embedding) != expected_dim:
                raise EmbeddingDimensionError(
                    f"Chunk {chunk.id} has missing or mismatched dimensions.",
                    provider="in_memory",
                )

        lock = self._get_lock(collection)
        async with lock:
            if collection not in self._collections:
                self._collections[collection] = {}
            col = self._collections[collection]
            for chunk in chunks:
                col[chunk.id] = chunk

        return len(chunks)

    async def search(
        self,
        query_embedding: list[float],
        *,
        collection: str = "default",
        top_k: int = 5,
        filter: dict[str, str | int | float | bool] | None = None,
        metric: DistanceMetric | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Search for similar chunks using pure-Python similarity functions."""
        if not self._initialized:
            raise VectorStoreOperationError("Store not initialized. Use async with.")

        col = self._collections.get(collection, {})
        if not col:
            return []

        selected_metric = metric or self._default_metric
        scored_chunks: list[tuple[Chunk, float]] = []

        for chunk in col.values():
            # Apply metadata filter
            if filter:
                match = all(chunk.metadata.get(k) == v for k, v in filter.items())
                if not match:
                    continue

            if chunk.embedding is None:
                continue

            if selected_metric == DistanceMetric.COSINE:
                score = cosine_similarity(query_embedding, chunk.embedding)
            elif selected_metric == DistanceMetric.DOT:
                score = dot_product(query_embedding, chunk.embedding)
            elif selected_metric == DistanceMetric.EUCLIDEAN:
                # For euclidean, lower distance is better. We negate so higher score = better.
                dist = euclidean_distance(query_embedding, chunk.embedding)
                score = -dist
            else:
                raise VectorStoreOperationError(f"Unsupported metric: {selected_metric}")

            scored_chunks.append((chunk, score))

        # Sort by score descending (highest similarity first)
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]

    async def delete(self, chunk_ids: list[str], *, collection: str = "default") -> int:
        """Delete chunks by ID."""
        col = self._collections.get(collection, {})
        if not col:
            return 0

        lock = self._get_lock(collection)
        async with lock:
            deleted_count = 0
            for chunk_id in chunk_ids:
                if chunk_id in col:
                    del col[chunk_id]
                    deleted_count += 1
            return deleted_count

    async def get(self, chunk_ids: list[str], *, collection: str = "default") -> list[Chunk]:
        """Retrieve chunks by ID."""
        col = self._collections.get(collection, {})
        return [col[chunk_id] for chunk_id in chunk_ids if chunk_id in col]

    async def count(self, *, collection: str = "default") -> int:
        """Return the number of chunks in a collection."""
        col = self._collections.get(collection, {})
        return len(col)

    async def health_check(self) -> bool:
        """Always returns True if initialized."""
        return self._initialized
