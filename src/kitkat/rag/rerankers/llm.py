"""LLM-based reranker using kitkat's LLMService."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING

from kitkat.rag.abc.reranker import Reranker
from kitkat.rag.core.exceptions import RerankingError
from kitkat.rag.core.models import RetrievalResult

if TYPE_CHECKING:
    from kitkat.service.managed import LLMService
    from kitkat.service.router import LLMRouter

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """\
You are a relevance expert. Score how relevant the following passage is
to the query on a scale of 0 to 10. Respond with ONLY the number.

Query: {query}
Passage: {passage}
"""

# Extracts the first number found in the string
_SCORE_PATTERN = re.compile(r"\d+")


class LLMReranker(Reranker):
    """Reranks results by asking an LLM to score relevance (0-10).

    Args:
        llm_service: The kitkat LLMService or LLMRouter to use for scoring.
        model: The model name to use for scoring.
        max_concurrency: Max concurrent LLM calls for batch scoring.
    """

    RERANKER_TYPE = "llm"

    def __init__(
        self,
        llm_service: LLMService | LLMRouter,
        *,
        model: str = "gpt-4o-mini",
        max_concurrency: int = 5,
    ) -> None:
        self._llm = llm_service
        self._model = model
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def _score_chunk(self, query: str, passage: str) -> float:
        """Score a single chunk using the LLM."""
        from kitkat.core.enums import Role
        from kitkat.core.models import LLMRequest, Message

        prompt = _PROMPT_TEMPLATE.format(query=query, passage=passage)

        request = LLMRequest(
            messages=[
                Message(role=Role.SYSTEM, content="You are a helpful AI assistant."),
                Message(role=Role.USER, content=prompt),
            ],
            model=self._model,
            temperature=0.0,
            max_tokens=5,
        )

        async with self._semaphore:
            try:
                response = await self._llm.complete(request)
                match = _SCORE_PATTERN.search(response.content)
                if match:
                    return float(match.group())
                logger.warning("LLM reranker failed to parse score from: %s", response.content)
                return 0.0
            except Exception as exc:
                logger.warning("LLM reranker scoring failed: %s", exc)
                return 0.0

    async def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        *,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Rerank results by scoring each chunk concurrently."""
        if not results:
            return []

        try:
            # Score all chunks concurrently
            tasks = [self._score_chunk(query, res.chunk.content) for res in results]
            scores = await asyncio.gather(*tasks)

            # Combine and sort
            scored_results = list(zip(results, scores, strict=True))
            scored_results.sort(key=lambda x: x[1], reverse=True)

            limit = top_k if top_k is not None else len(scored_results)
            reranked: list[RetrievalResult] = []

            for idx, (res, score) in enumerate(scored_results[:limit]):
                reranked.append(
                    RetrievalResult(
                        chunk=res.chunk,
                        score=score / 10.0,  # Normalize 0-10 to 0.0-1.0
                        rank=idx + 1,
                        retrieved_by=res.retrieved_by,
                        metadata={**res.metadata, "reranked_by": self.RERANKER_TYPE},
                    )
                )
            return reranked
        except Exception as exc:
            raise RerankingError(f"LLM reranking failed: {exc}") from exc
