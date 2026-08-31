# src/kitkat/rag/ingest/transformers/metadata.py
"""Adds basic metadata to documents."""

from __future__ import annotations

from kitkat.rag.core.models import Document
from kitkat.rag.ingest.transformers.base import DocumentTransformer


class MetadataExtractor(DocumentTransformer):
    """Extracts basic stats like word count and character count into metadata."""

    async def transform(self, document: Document) -> Document:
        # Frozen dataclass requires object.__setattr__
        new_meta = document.metadata.copy()
        new_meta["char_count"] = len(document.content)
        new_meta["word_count"] = len(document.content.split())
        object.__setattr__(document, "metadata", new_meta)
        return document
