# src/kitkat/rag/abc/retriever.py
"""Abstract contract for retrieval strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kitkat.rag.core.models import RetrievalResult


class Retriever(ABC):
    """Abstract contract for retrieval strategies."""

    STRATEGY: str  # "vector", "keyword", "hybrid"

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        filter: dict[str, str | int | float | bool] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve relevant chunks for a query.

        Args:
            query: Natural language query text.
            top_k: Maximum number of results to return.
            filter: Optional metadata filter (backend-specific semantics).

        Returns:
            Ranked list of RetrievalResult objects.

        Raises:
            RetrievalError: If retrieval fails.
        """
