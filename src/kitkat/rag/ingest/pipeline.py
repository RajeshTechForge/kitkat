# src/kitkat/rag/ingest/pipeline.py
"""Document ingestion pipeline orchestrator."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from kitkat.rag.core.exceptions import DocumentLoadError, IngestionPipelineError
from kitkat.rag.core.models import Document

if TYPE_CHECKING:
    from kitkat.rag.abc.chunker import Chunker
    from kitkat.rag.abc.embedder import EmbeddingProvider
    from kitkat.rag.abc.vector_store import VectorStore
    from kitkat.rag.ingest.loaders.base import DocumentLoader
    from kitkat.rag.ingest.transformers.base import DocumentTransformer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionResult:
    """Result of an ingestion operation."""

    documents_processed: int
    chunks_created: int
    chunks_embedded: int
    chunks_stored: int
    errors: list[str] = field(default_factory=list)
    latency_ms: float = 0.0


class IngestionPipeline:
    """Orchestrates the document ingestion workflow.

    Load -> Transform -> Chunk -> Embed -> Store.
    """

    def __init__(
        self,
        *,
        loaders: list["DocumentLoader"],
        transformers: list["DocumentTransformer"],
        chunker: "Chunker",
        embedder: "EmbeddingProvider",
        store: "VectorStore",
        batch_size: int = 100,
    ) -> None:
        self._loaders = {ext: loader for loader in loaders for ext in loader.SUPPORTED_EXTENSIONS}
        self._transformers = transformers
        self._chunker = chunker
        self._embedder = embedder
        self._store = store
        self._batch_size = batch_size

    async def ingest_sources(
        self,
        sources: list[str],
        *,
        collection: str = "default",
    ) -> IngestionResult:
        """Load from file paths/URLs, process, and store."""
        start_time = time.monotonic()
        all_docs: list[Document] = []
        errors: list[str] = []

        # 1. Load
        for source in sources:
            doc = await self._load_source(source, errors)
            if doc:
                all_docs.append(doc)

        if not all_docs:
            return IngestionResult(
                documents_processed=0,
                chunks_created=0,
                chunks_embedded=0,
                chunks_stored=0,
                errors=errors,
                latency_ms=(time.monotonic() - start_time) * 1000,
            )

        return await self.ingest_documents(all_docs, collection=collection, errors=errors)

    async def ingest_documents(
        self,
        documents: list[Document],
        *,
        collection: str = "default",
        errors: list[str] | None = None,
    ) -> IngestionResult:
        """Process pre-loaded Document objects."""
        start_time = time.monotonic()
        if errors is None:
            errors = []

        chunks_created = 0
        chunks_embedded = 0
        chunks_stored = 0

        # 2. Transform
        transformed_docs: list[Document] = []
        for doc in documents:
            current_doc = doc
            try:
                for transformer in self._transformers:
                    result = await transformer.transform(current_doc)
                    if result is None:
                        logger.info("Document dropped by transformer: %s", doc.source)
                        current_doc = None
                        break
                    current_doc = result
                if current_doc:
                    transformed_docs.append(current_doc)
            except Exception as exc:
                msg = f"Transformation failed for {doc.source}: {exc}"
                logger.error(msg)
                errors.append(msg)

        if not transformed_docs:
            return IngestionResult(
                documents_processed=len(documents),
                chunks_created=0,
                chunks_embedded=0,
                chunks_stored=0,
                errors=errors,
                latency_ms=(time.monotonic() - start_time) * 1000,
            )

        # 3. Chunk
        all_chunks = []
        for doc in transformed_docs:
            try:
                doc_chunks = await self._chunker.chunk(doc)
                all_chunks.extend(doc_chunks)
                chunks_created += len(doc_chunks)
            except Exception as exc:
                msg = f"Chunking failed for {doc.source}: {exc}"
                logger.error(msg)
                errors.append(msg)

        if not all_chunks:
            return IngestionResult(
                documents_processed=len(transformed_docs),
                chunks_created=0,
                chunks_embedded=0,
                chunks_stored=0,
                errors=errors,
                latency_ms=(time.monotonic() - start_time) * 1000,
            )

        # 4. Embed & 5. Store (Batched)
        for i in range(0, len(all_chunks), self._batch_size):
            batch = all_chunks[i : i + self._batch_size]
            try:
                texts = [c.content for c in batch]
                result = await self._embedder.embed_documents(texts)

                # Attach embeddings to chunks
                for j, chunk in enumerate(batch):
                    chunk = chunk  # type: ignore[assignment]
                    # Frozen dataclass workaround
                    object.__setattr__(chunk, "embedding", result[j])
                    chunks_embedded += 1

                stored_count = await self._store.add(batch, collection=collection)
                chunks_stored += stored_count
            except Exception as exc:
                msg = f"Embedding/Storage failed for batch {i // self._batch_size}: {exc}"
                logger.error(msg)
                errors.append(msg)

        return IngestionResult(
            documents_processed=len(transformed_docs),
            chunks_created=chunks_created,
            chunks_embedded=chunks_embedded,
            chunks_stored=chunks_stored,
            errors=errors,
            latency_ms=(time.monotonic() - start_time) * 1000,
        )

    async def _load_source(self, source: str, errors: list[str]) -> Document | None:
        """Load a single source using the appropriate loader."""
        import os

        ext = os.path.splitext(source)[1].lower()

        loader = self._loaders.get(ext)
        if not loader:
            msg = f"No loader found for extension '{ext}' (source: {source})"
            errors.append(msg)
            logger.warning(msg)
            return None

        try:
            return await loader.load(source)
        except DocumentLoadError as exc:
            errors.append(str(exc))
            return None
        except Exception as exc:
            msg = f"Unexpected error loading {source}: {exc}"
            errors.append(msg)
            logger.error(msg)
            return None
