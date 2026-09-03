# src/kitkat/rag/retrieval/vector.py
"""Dense vector retrieval strategy."""

from __future__ import annotations

import logging

from kitkat.rag.abc.embedder import EmbeddingProvider
from kitkat.rag.abc.retriever import Retriever
from kitkat.rag.abc.vector_store import VectorStore
from kitkat.rag.core.enums import DistanceMetric
from kitkat.rag.core.exceptions import RetrievalError
from kitkat.rag.core.models import RetrievalResult

logger = logging.getLogger(__name__)


class VectorRetriever(Retriever):
    """Retrieves chunks by embedding the query and searching the vector store.

    Args:
        embedder: The embedding provider to generate query vectors.
        store: The vector store backend to search.
    """

    STRATEGY = "vector"

    def __init__(self, embedder: EmbeddingProvider, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        filter: dict[str, str | int | float | bool] | None = None,
    ) -> list[RetrievalResult]:
        """Embed the query and search the vector store."""
        try:
            query_embedding = await self._embedder.embed_query(query)
            results = await self._store.search(
                query_embedding,
                top_k=top_k,
                filter=filter,
                metric=DistanceMetric.COSINE,
            )

            return [
                RetrievalResult(
                    chunk=chunk,
                    score=score,
                    rank=idx + 1,
                    retrieved_by=self.STRATEGY,
                )
                for idx, (chunk, score) in enumerate(results)
            ]
        except Exception as exc:
            raise RetrievalError(f"Vector retrieval failed: {exc}") from exc
