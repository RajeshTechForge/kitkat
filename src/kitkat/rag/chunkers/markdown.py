"""Markdown-aware text chunker."""

from __future__ import annotations

import re
from typing import Any

from kitkat.rag.chunkers.base import BaseChunker

_HEADER_pattern = re.compile(r"^#{1,6}\s+", re.MULTILINE)


class MarkdownChunker(BaseChunker):
    """Splits markdown by headers, preserving header metadata."""

    STRATEGY = "markdown"

    def __init__(self, max_chunk_size: int = 2000) -> None:
        self._max_chunk_size = max_chunk_size

    async def chunk(self, document) -> list:  # type: ignore[override]
        from kitkat.rag.core.models import Chunk, Document

        doc: Document = document

        sections = _header_pattern.split(doc.content)
        # The first element might be content before any headers
        final_chunks: list[Chunk] = []
        chunk_idx = 0
        current_pos = 0

        for section in sections:
            if not section.strip():
                continue
            # Reconstruct the header for the section if it was split
            header_match = re.match(r"^(#{1,6})\s+", section)
            metadata = doc.metadata.copy()
            if header_match:
                metadata["header_level"] = len(header_match.group(1))

            start = current_pos
            end = current_pos + len(section)
            current_pos = end

            if len(section) <= self._max_chunk_size:
                final_chunks.append(self._build_chunk(doc, section.strip(), chunk_idx, start, end))
                chunk_idx += 1
            else:
                # Fallback to recursive if a section is too large
                from kitkat.rag.chunkers.recursive import RecursiveCharacterChunker

                sub_chunker = RecursiveCharacterChunker(
                    chunk_size=self._max_chunk_size, chunk_overlap=200
                )
                sub_chunks = await sub_chunker.chunk(
                    Document(content=section, metadata=metadata, source=doc.source)
                )
                for sc in sub_chunks:
                    final_chunks.append(
                        Chunk(
                            document_id=doc.id,
                            content=sc.content,
                            metadata=sc.metadata,
                            chunk_index=chunk_idx,
                            start_char=start + sc.start_char,
                            end_char=start + sc.end_char,
                            token_count=sc.token_count,
                        )
                    )
                    chunk_idx += 1

        return final_chunks

    @property
    def config(self) -> dict[str, Any]:
        return {"max_chunk_size": self._max_chunk_size}
