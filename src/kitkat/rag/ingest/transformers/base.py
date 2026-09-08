"""Abstract contract for document transformers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kitkat.rag.core.models import Document


class DocumentTransformer(ABC):
    """Abstract contract for transforming documents post-load."""

    @abstractmethod
    async def transform(self, document: Document) -> Document | None:
        """Transform a document or drop it by returning None.

        Args:
            document: The input document.

        Returns:
            The transformed Document, or None to drop the document.
        """
