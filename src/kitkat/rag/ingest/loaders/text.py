"""Plain text file loader."""

from __future__ import annotations

import asyncio
import logging
import os

from kitkat.rag.core.exceptions import DocumentLoadError
from kitkat.rag.core.models import Document
from kitkat.rag.ingest.loaders.base import DocumentLoader

logger = logging.getLogger(__name__)


class TextLoader(DocumentLoader):
    """Loads .txt and other plain text files."""

    SUPPORTED_EXTENSIONS = frozenset({".txt", ".text", ".log"})

    async def load(self, source: str) -> Document:
        if not os.path.exists(source):
            raise DocumentLoadError(f"File not found: {source}", source=source)

        try:
            # Run blocking file I/O in a thread
            content = await asyncio.to_thread(self._read_file, source)
            return Document(
                content=content,
                source=source,
                content_type="text/plain",
                metadata={"filename": os.path.basename(source)},
            )
        except Exception as exc:
            logger.error("Failed to load text file %s: %s", source, exc)
            raise DocumentLoadError(f"Failed to read file: {exc}", source=source) from exc

    def _read_file(self, source: str) -> str:
        with open(source, "r", encoding="utf-8") as f:
            return f.read()
