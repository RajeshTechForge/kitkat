"""Sentence-based text chunker."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from kitkat.rag.chunkers.base import BaseChunker

if TYPE_CHECKING:
    from kitkat.rag.core.models import Chunk, Document

_SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")


class SentenceChunker(BaseChunker):
    """Splits text at sentence boundaries using regex."""

    STRATEGY = "sentence"

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    async def chunk(self, document) -> list:

        doc: Document = document

        sentences = _SENTENCE_PATTERN.split(doc.content)
        final_chunks: list[Chunk] = []
        current_content = ""
        current_start = 0
        chunk_idx = 0

        for sentence in sentences:
            if len(current_content) + len(sentence) <= self._chunk_size:
                current_content += sentence + " "
            else:
                if current_content.strip():
                    final_chunks.append(
                        self._build_chunk(
                            doc,
                            current_content.strip(),
                            chunk_idx,
                            current_start,
                            current_start + len(current_content),
                        )
                    )
                    chunk_idx += 1
                    # Apply overlap
                    overlap_start = max(0, len(current_content) - self._chunk_overlap)
                    current_content = current_content[overlap_start:] + sentence + " "
                    current_start = current_start + overlap_start
                else:
                    # Single sentence larger than chunk_size
                    current_content = sentence + " "

        if current_content.strip():
            final_chunks.append(
                self._build_chunk(
                    doc,
                    current_content.strip(),
                    chunk_idx,
                    current_start,
                    current_start + len(current_content),
                )
            )

        return final_chunks

    @property
    def config(self) -> dict[str, Any]:
        return {
            "chunk_size": self._chunk_size,
            "chunk_overlap": self._chunk_overlap,
        }
