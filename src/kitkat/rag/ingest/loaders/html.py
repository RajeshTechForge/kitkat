# src/kitkat/rag/ingest/loaders/html.py
"""HTML file loader (requires rag-html extra)."""

from __future__ import annotations

import asyncio
import logging
import os

from kitkat.rag._check import require_rag_extra
from kitkat.rag.core.exceptions import DocumentLoadError
from kitkat.rag.core.models import Document
from kitkat.rag.ingest.loaders.base import DocumentLoader

logger = logging.getLogger(__name__)


class HTMLLoader(DocumentLoader):
    """Loads .html files, stripping tags via BeautifulSoup."""

    SUPPORTED_EXTENSIONS = frozenset({".html", ".htm"})

    def __init__(self) -> None:
        require_rag_extra("rag-html")
        from bs4 import BeautifulSoup  # type: ignore[import-not-found]

        self._bs = BeautifulSoup

    async def load(self, source: str) -> Document:
        if not os.path.exists(source):
            raise DocumentLoadError(f"File not found: {source}", source=source)

        try:

            def _read_and_parse():
                with open(source, "r", encoding="utf-8") as f:
                    raw = f.read()
                soup = self._bs(raw, "html.parser")
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                return soup.get_text(separator="\n", strip=True)

            content = await asyncio.to_thread(_read_and_parse)
            return Document(
                content=content,
                source=source,
                content_type="text/html",
                metadata={"filename": os.path.basename(source)},
            )
        except Exception as exc:
            logger.error("Failed to load HTML file %s: %s", source, exc)
            raise DocumentLoadError(f"Failed to parse HTML: {exc}", source=source) from exc
