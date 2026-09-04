"""Abstract contract for post-retrieval rerankers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kitkat.rag.core.models import RetrievalResult


class Reranker(ABC):
    """Abstract contract for post-retrieval reranking."""

    RERANKER_TYPE: str  # "cross_encoder", "llm"

    @abstractmethod
    async def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        *,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Re-score and re-order retrieval results.

        Args:
            query: Original user query.
            results: Initial retrieval results (already ranked).
            top_k: If provided, return only the top_k reranked results.

        Returns:
            Re-ordered list with updated scores and ranks.

        Raises:
            RerankingError: If reranking fails.
        """
