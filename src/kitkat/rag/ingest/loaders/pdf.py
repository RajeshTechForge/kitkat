"""PDF file loader (requires rag-pdf extra)."""

from __future__ import annotations

import asyncio
import logging
import os

from pypdf import PdfReader

from kitkat.rag._check import require_rag_extra
from kitkat.rag.core.exceptions import DocumentLoadError
from kitkat.rag.core.models import Document
from kitkat.rag.ingest.loaders.base import DocumentLoader

logger = logging.getLogger(__name__)


class PDFLoader(DocumentLoader):
    """Loads .pdf files extracting text via pypdf."""

    SUPPORTED_EXTENSIONS = frozenset({".pdf"})

    def __init__(self) -> None:
        require_rag_extra("rag-pdf")

        self._reader_cls = PdfReader

    async def load(self, source: str) -> Document:
        if not os.path.exists(source):
            raise DocumentLoadError(f"File not found: {source}", source=source)

        try:

            def _read_pdf():
                reader = self._reader_cls(source)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()

            content = await asyncio.to_thread(_read_pdf)
            return Document(
                content=content,
                source=source,
                content_type="application/pdf",
                metadata={"filename": os.path.basename(source)},
            )
        except Exception as exc:
            logger.error("Failed to load PDF file %s: %s", source, exc)
            raise DocumentLoadError(f"Failed to parse PDF: {exc}", source=source) from exc
