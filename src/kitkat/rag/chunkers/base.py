"""Shared helpers for chunker implementations."""

from __future__ import annotations

from typing import Any

from kitkat.rag._internal.tokenizers import count_tokens
from kitkat.rag.abc.chunker import Chunker
from kitkat.rag.core.models import Chunk, Document


class BaseChunker(Chunker):
    """Base class providing shared chunking utilities."""

    def _estimate_tokens(self, text: str) -> int:
        """Estimate the token count for a string."""
        return count_tokens(text)

    def _build_chunk(
        self,
        document: Document,
        content: str,
        chunk_index: int,
        start_char: int,
        end_char: int,
    ) -> Chunk:
        """Construct a Chunk with metadata and token count."""
        return Chunk(
            document_id=document.id,
            content=content,
            metadata=document.metadata.copy(),
            chunk_index=chunk_index,
            start_char=start_char,
            end_char=end_char,
            token_count=self._estimate_tokens(content),
        )

    @property
    def config(self) -> dict[str, Any]:
        """Return an empty config dict. Subclasses should override."""
        return {}
