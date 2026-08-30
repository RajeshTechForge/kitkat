# src/kitkat/rag/abc/chunker.py
"""Abstract contract for document chunking strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

from kitkat.rag.core.models import Chunk, Document


class Chunker(ABC):
    """Abstract contract for document chunking strategies."""

    STRATEGY: str  # e.g. "recursive", "token", "sentence"

    @abstractmethod
    async def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into chunks.

        Args:
            document: The source document to chunk.

        Returns:
            Ordered list of Chunk objects with chunk_index, start_char,
            end_char, and token_count populated.

        Raises:
            ChunkingError: If chunking fails due to invalid input or
                strategy-specific constraints.
        """

    async def chunk_batch(self, documents: list[Document]) -> list[list[Chunk]]:
        """Chunk multiple documents. Default: sequential; override for parallelism."""
        results: list[list[Chunk]] = []
        for doc in documents:
            results.append(await self.chunk(doc))
        return results

    @property
    @abstractmethod
    def config(self) -> dict[str, object]:
        """Return the chunker's configuration as a serializable dict."""
