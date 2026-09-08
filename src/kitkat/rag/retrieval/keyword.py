"""Sparse keyword retrieval using a pure-Python BM25 implementation."""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict

from kitkat.rag.abc.retriever import Retriever
from kitkat.rag.core.exceptions import RetrievalError
from kitkat.rag.core.models import Chunk, RetrievalResult

logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Simple regex tokenizer for BM25 indexing."""
    return _TOKEN_PATTERN.findall(text.lower())


class KeywordRetriever(Retriever):
    """In-memory BM25 keyword retriever.

    Does not require an external service like Elasticsearch. Maintains an
    inverted index in memory. The index must be populated via `add_chunks()`
    during the ingestion pipeline.
    """

    STRATEGY = "keyword"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._chunks: dict[str, Chunk] = {}
        self._doc_tokens: dict[str, list[str]] = {}
        self._doc_lens: dict[str, int] = {}
        self._avg_doc_len: float = 0.0
        self._term_freqs: dict[str, dict[str, int]] = defaultdict(dict)
        self._doc_freq: dict[str, int] = defaultdict(int)
        self._num_docs: int = 0

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Add chunks to the BM25 index."""
        for chunk in chunks:
            if chunk.id in self._chunks:
                continue  # Skip duplicates

            tokens = _tokenize(chunk.content)
            self._chunks[chunk.id] = chunk
            self._doc_tokens[chunk.id] = tokens
            self._doc_lens[chunk.id] = len(tokens)

            unique_tokens = set(tokens)
            for token in unique_tokens:
                self._term_freqs[token][chunk.id] = tokens.count(token)
                self._doc_freq[token] += 1

            self._num_docs += 1

        total_len = sum(self._doc_lens.values())
        self._avg_doc_len = total_len / self._num_docs if self._num_docs > 0 else 0.0
        logger.debug("KeywordRetriever index updated. Total docs: %d", self._num_docs)

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        filter: dict[str, str | int | float | bool] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve chunks by BM25 score."""
        if self._num_docs == 0:
            return []

        try:
            query_tokens = _tokenize(query)
            scores: dict[str, float] = defaultdict(float)

            for token in query_tokens:
                if token not in self._term_freqs:
                    continue

                # IDF calculation
                df = self._doc_freq[token]
                idf = math.log(1 + (self._num_docs - df + 0.5) / (df + 0.5))

                for doc_id, tf in self._term_freqs[token].items():
                    doc_len = self._doc_lens[doc_id]
                    # BM25 score
                    numerator = tf * (self._k1 + 1)
                    denominator = tf + self._k1 * (
                        1 - self._b + self._b * (doc_len / self._avg_doc_len)
                    )
                    scores[doc_id] += idf * (numerator / denominator)

            # Apply metadata filter
            filtered_scores = scores
            if filter:
                filtered_scores = {
                    doc_id: score
                    for doc_id, score in scores.items()
                    if all(self._chunks[doc_id].metadata.get(k) == v for k, v in filter.items())
                }

            # Sort by score descending
            sorted_doc_ids = sorted(
                filtered_scores.keys(), key=lambda d: filtered_scores[d], reverse=True
            )

            results: list[RetrievalResult] = []
            for idx, doc_id in enumerate(sorted_doc_ids[:top_k]):
                if filtered_scores[doc_id] <= 0:
                    continue
                results.append(
                    RetrievalResult(
                        chunk=self._chunks[doc_id],
                        score=float(filtered_scores[doc_id]),
                        rank=idx + 1,
                        retrieved_by=self.STRATEGY,
                    )
                )

            return results
        except Exception as exc:
            raise RetrievalError(f"Keyword retrieval failed: {exc}") from exc
