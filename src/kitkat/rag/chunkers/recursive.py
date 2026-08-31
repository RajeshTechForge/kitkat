# src/kitkat/rag/chunkers/recursive.py
"""Recursive character text chunker."""

from __future__ import annotations

from typing import Any

from kitkat.rag.chunkers.base import BaseChunker


class RecursiveCharacterChunker(BaseChunker):
    """Splits text recursively by a list of separators.

    Tries to split on the first separator; if chunks are still too large,
    recursively splits with the next separator. Merges small chunks.
    """

    STRATEGY = "recursive"

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
    ) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._separators = separators or ["\n\n", "\n", ". ", " ", ""]

    async def chunk(self, document) -> list:  # type: ignore[override]
        from kitkat.rag.core.models import Chunk, Document

        doc: Document = document

        chunks_raw = self._split_text(doc.content, self._separators)

        # Merge small chunks and apply overlap
        final_chunks: list[Chunk] = []
        current_content = ""
        current_start = 0
        chunk_idx = 0

        for piece, start, end in chunks_raw:
            if len(current_content) + len(piece) <= self._chunk_size:
                current_content += piece
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
                current_content = current_content[overlap_start:] + piece
                current_start = current_start + overlap_start

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

    def _split_text(self, text: str, separators: list[str]) -> list[tuple[str, int, int]]:
        """Recursively split text into pieces with (piece, start_char, end_char)."""
        if len(text) <= self._chunk_size:
            return [(text, 0, len(text))]

        sep = separators[0]
        if sep == "":
            # Last resort: hard split
            return [
                (text[i : i + self._chunk_size], i, i + self._chunk_size)
                for i in range(0, len(text), self._chunk_size)
            ]

        pieces = text.split(sep)
        results: list[tuple[str, int, int]] = []
        current_pos = 0

        for i, piece in enumerate(pieces):
            piece_with_sep = piece + sep if i < len(pieces) - 1 else piece
            start = current_pos
            end = current_pos + len(piece_with_sep)
            current_pos = end

            if len(piece_with_sep) <= self._chunk_size:
                results.append((piece_with_sep, start, end))
            else:
                # Recursively split this large piece with the next separator
                sub_results = self._split_text(piece_with_sep, separators[1:])
                results.extend(sub_results)

        return results

    @property
    def config(self) -> dict[str, Any]:
        return {
            "chunk_size": self._chunk_size,
            "chunk_overlap": self._chunk_overlap,
            "separators": self._separators,
        }
