"""Deduplicates documents by content hash."""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

from kitkat.rag.ingest.transformers.base import DocumentTransformer

if TYPE_CHECKING:
    from kitkat.rag.core.models import Document

logger = logging.getLogger(__name__)


class Deduplicator(DocumentTransformer):
    """Drops documents that have already been seen based on SHA-256 content hash."""

    def __init__(self) -> None:
        self._seen_hashes: set[str] = set()

    async def transform(self, document: Document) -> Document | None:
        content_hash = hashlib.sha256(document.content.encode("utf-8")).hexdigest()
        if content_hash in self._seen_hashes:
            logger.info("Skipping duplicate document: %s", document.source)
            return None

        self._seen_hashes.add(content_hash)

        new_meta = document.metadata.copy()
        new_meta["content_hash"] = content_hash
        object.__setattr__(document, "metadata", new_meta)

        return document
