# src/kitkat/rag/chunkers/token.py
"""Token-based text chunker."""

from __future__ import annotations

from typing import Any

from kitkat.rag._internal.tokenizers import count_tokens
from kitkat.rag.chunkers.base import BaseChunker


class TokenChunker(BaseChunker):
    """Splits text into chunks based on exact token counts."""

    STRATEGY = "token"

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    async def chunk(self, document) -> list:  # type: ignore[override]
        from kitkat.rag.core.models import Chunk, Document

        doc: Document = document

        # Simple word-based iteration to approximate token boundaries
        words = doc.content.split()
        final_chunks: list[Chunk] = []
        current_words: list[str] = []
        current_tokens = 0
        current_start = 0
        last_end = 0
        chunk_idx = 0

        for word in words:
            word_tokens = count_tokens(word + " ")
            if current_tokens + word_tokens > self._chunk_size and current_words:
                content = " ".join(current_words)
                final_chunks.append(
                    self._build_chunk(
                        doc, content, chunk_idx, current_start, current_start + len(content)
                    )
                )
                chunk_idx += 1

                # Apply overlap
                overlap_words = current_words[-self._chunk_overlap :]
                current_words = overlap_words
                current_tokens = count_tokens(" ".join(overlap_words) + " ")
                current_start = current_start + len(content) + 1  # +1 for space

            current_words.append(word)
            current_tokens += word_tokens

        if current_words:
            content = " ".join(current_words)
            final_chunks.append(
                self._build_chunk(
                    doc, content, chunk_idx, current_start, current_start + len(content)
                )
            )

        return final_chunks

    @property
    def config(self) -> dict[str, Any]:
        return {
            "chunk_size": self._chunk_size,
            "chunk_overlap": self._chunk_overlap,
        }
