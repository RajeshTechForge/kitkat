# src/kitkat/rag/ingest/loaders/markdown.py
"""Markdown file loader."""

from __future__ import annotations

from kitkat.rag.core.models import Document
from kitkat.rag.ingest.loaders.text import TextLoader


class MarkdownLoader(TextLoader):
    """Loads .md files as plain text, marking content_type."""

    SUPPORTED_EXTENSIONS = frozenset({".md", ".markdown"})

    async def load(self, source: str) -> Document:
        doc = await super().load(source)
        # Frozen dataclass requires object.__setattr__ to modify
        object.__setattr__(doc, "content_type", "text/markdown")
        return doc
