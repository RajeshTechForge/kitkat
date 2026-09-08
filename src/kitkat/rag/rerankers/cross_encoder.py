"""Cross-encoder reranker using sentence-transformers (requires rag-rerank extra)."""

from __future__ import annotations

import asyncio
import logging

from kitkat.rag._check import require_rag_extra
from kitkat.rag.abc.reranker import Reranker
from kitkat.rag.core.exceptions import RerankingError
from kitkat.rag.core.models import RetrievalResult

logger = logging.getLogger(__name__)

require_rag_extra("rag-rerank")


class CrossEncoderReranker(Reranker):
    """Reranks results using a HuggingFace cross-encoder model.

    Args:
        model_name: The HuggingFace model ID (default: ms-marco-MiniLM-L-6-v2).
    """

    RERANKER_TYPE = "cross_encoder"

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self._model_name = model_name
        self._model = None

    def _load_model(self) -> None:
        """Lazily load the cross-encoder model in a thread."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder

            logger.info("Loading CrossEncoder model: %s", self._model_name)
            self._model = CrossEncoder(self._model_name)
        except Exception as exc:
            raise RerankingError(f"Failed to load CrossEncoder model: {exc}") from exc

    async def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        *,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Rerank results using the cross-encoder."""
        if not results:
            return []

        try:
            # Load model in a thread to avoid blocking the event loop
            await asyncio.to_thread(self._load_model)

            # Prepare pairs
            pairs = [(query, res.chunk.content) for res in results]

            # Predict scores in a thread
            scores = await asyncio.to_thread(self._model.predict, pairs)

            # Combine and sort
            scored_results = list(zip(results, scores, strict=True))
            scored_results.sort(key=lambda x: x[1], reverse=True)

            limit = top_k if top_k is not None else len(scored_results)
            reranked: list[RetrievalResult] = []

            for idx, (res, score) in enumerate(scored_results[:limit]):
                reranked.append(
                    RetrievalResult(
                        chunk=res.chunk,
                        score=float(score),
                        rank=idx + 1,
                        retrieved_by=res.retrieved_by,
                        metadata={**res.metadata, "reranked_by": self.RERANKER_TYPE},
                    )
                )
            return reranked
        except Exception as exc:
            raise RerankingError(f"Cross-encoder reranking failed: {exc}") from exc
