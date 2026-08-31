# src/kitkat/rag/ingest/transformers/base.py
"""Abstract contract for document transformers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union

from kitkat.rag.core.models import Document


class DocumentTransformer(ABC):
    """Abstract contract for transforming documents post-load."""

    @abstractmethod
    async def transform(self, document: Document) -> Union[Document, None]:
        """Transform a document or drop it by returning None.

        Args:
            document: The input document.

        Returns:
            The transformed Document, or None to drop the document.
        """
