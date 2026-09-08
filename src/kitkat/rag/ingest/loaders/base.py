"""Abstract contract for document loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kitkat.rag.core.models import Document


class DocumentLoader(ABC):
    """Abstract contract for loading raw documents."""

    SUPPORTED_EXTENSIONS: frozenset[str] = frozenset()

    @abstractmethod
    async def load(self, source: str) -> Document:
        """Load a single document from a source path or URL.

        Args:
            source: File path or URL.

        Returns:
            A Document object.

        Raises:
            DocumentLoadError: If loading fails.
        """

    async def load_batch(self, sources: list[str]) -> list[Document]:
        """Load multiple documents. Default: sequential."""
        results: list[Document] = []
        for src in sources:
            results.append(await self.load(src))
        return results
