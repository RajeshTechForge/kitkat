# src/kitkat/rag/ingest/loaders/directory.py
"""Batch loader for directories."""

from __future__ import annotations

import logging
import os

from kitkat.rag.core.exceptions import DocumentLoadError
from kitkat.rag.core.models import Document
from kitkat.rag.ingest.loaders.base import DocumentLoader

logger = logging.getLogger(__name__)


class DirectoryLoader(DocumentLoader):
    """Loads all supported files from a directory using child loaders."""

    def __init__(self, loaders: list[DocumentLoader]) -> None:
        self._loaders = loaders
        self._ext_map: dict[str, DocumentLoader] = {}
        for loader in loaders:
            for ext in loader.SUPPORTED_EXTENSIONS:
                self._ext_map[ext] = loader

    async def load(self, source: str) -> Document:
        # DirectoryLoader expects a directory path in `load_batch`
        raise NotImplementedError("Use load_batch() for DirectoryLoader.")

    async def load_batch(self, sources: list[str]) -> list[Document]:
        if len(sources) != 1:
            raise ValueError("DirectoryLoader expects exactly one directory path in sources.")
        directory = sources[0]
        if not os.path.isdir(directory):
            raise DocumentLoadError(f"Directory not found: {directory}", source=directory)

        docs: list[Document] = []
        for root, _, files in os.walk(directory):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in self._ext_map:
                    loader = self._ext_map[ext]
                    file_path = os.path.join(root, file)
                    try:
                        doc = await loader.load(file_path)
                        docs.append(doc)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Skipping file %s due to error: %s", file_path, exc)
        return docs
