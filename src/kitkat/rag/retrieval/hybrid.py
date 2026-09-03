# src/kitkat/rag/retrieval/hybrid.py
"""Hybrid retrieval combining vector and keyword search via Reciprocal Rank Fusion."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict

from kitkat.rag.abc.retriever import Retriever
from kitkat.rag.core.exceptions import RetrievalError
from kitkat.rag.core.models import RetrievalResult
from kitkat.rag.retrieval.keyword import KeywordRetriever
from kitkat.rag.retrieval.vector import VectorRetriever

logger = logging.getLogger(__name__)


class HybridRetriever(Retriever):
    """Combines dense and sparse retrieval via Reciprocal Rank Fusion (RRF).

    Args:
        vector_retriever: Dense retrieval backend.
        keyword_retriever: Sparse retrieval backend.
        vector_weight: Weight for vector results (default: 0.5).
        keyword_weight: Weight for keyword results (default: 0.5).
        rrf_k: RRF constant (default: 60, standard value).
    """

    STRATEGY = "hybrid"

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        keyword_retriever: KeywordRetriever,
        *,
        vector_weight: float = 0.5,
        keyword_weight: float = 0.5,
        rrf_k: int = 60,
    ) -> None:
        self._vector_retriever = vector_retriever
        self._keyword_retriever = keyword_retriever
        self._vector_weight = vector_weight
        self._keyword_weight = keyword_weight
        self._rrf_k = rrf_k

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        filter: dict[str, str | int | float | bool] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve and fuse results from both retrievers concurrently."""
        # Fetch more results initially to give RRF a larger pool to fuse
        fetch_k = top_k * 3

        try:
            vector_results, keyword_results = await asyncio.gather(
                self._vector_retriever.retrieve(query, top_k=fetch_k, filter=filter),
                self._keyword_retriever.retrieve(query, top_k=fetch_k, filter=filter),
                return_exceptions=True,
            )

            if isinstance(vector_results, Exception):
                logger.warning("Vector retrieval failed in hybrid mode: %s", vector_results)
                vector_results = []
            if isinstance(keyword_results, Exception):
                logger.warning("Keyword retrieval failed in hybrid mode: %s", keyword_results)
                keyword_results = []

            return self._fuse_results(vector_results, keyword_results, top_k)
        except Exception as exc:
            raise RetrievalError(f"Hybrid retrieval failed: {exc}") from exc

    def _fuse_results(
        self,
        vector_results: list[RetrievalResult],
        keyword_results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Fuse results using Reciprocal Rank Fusion."""
        fused_scores: dict[str, float] = defaultdict(float)
        chunk_map: dict[str, RetrievalResult] = {}

        for res in vector_results:
            fused_scores[res.chunk.id] += self._vector_weight * (1.0 / (self._rrf_k + res.rank))
            # Keep the first encountered result (usually has embedding populated)
            if res.chunk.id not in chunk_map:
                chunk_map[res.chunk.id] = res

        for res in keyword_results:
            fused_scores[res.chunk.id] += self._keyword_weight * (1.0 / (self._rrf_k + res.rank))
            if res.chunk.id not in chunk_map:
                chunk_map[res.chunk.id] = res

        # Sort by fused score descending
        sorted_ids = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)

        final_results: list[RetrievalResult] = []
        for idx, chunk_id in enumerate(sorted_ids[:top_k]):
            original_res = chunk_map[chunk_id]
            final_results.append(
                RetrievalResult(
                    chunk=original_res.chunk,
                    score=fused_scores[chunk_id],
                    rank=idx + 1,
                    retrieved_by=self.STRATEGY,
                )
            )

        return final_results
