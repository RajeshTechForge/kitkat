"""Qdrant vector store backend (requires rag-qdrant extra)."""

from __future__ import annotations

import logging
import uuid

from kitkat.rag._check import require_rag_extra
from kitkat.rag.abc.vector_store import VectorStore
from kitkat.rag.core.enums import DistanceMetric
from kitkat.rag.core.exceptions import (
    VectorStoreConnectionError,
    VectorStoreOperationError,
)
from kitkat.rag.core.models import Chunk

logger = logging.getLogger(__name__)

# Force import check
require_rag_extra("rag-qdrant")

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
)


class QdrantVectorStore(VectorStore):
    """Production-ready vector store using Qdrant.

    Uses the native async API of qdrant-client.
    """

    BACKEND_TYPE = "qdrant"
    SUPPORTED_METRICS = frozenset(
        {DistanceMetric.COSINE, DistanceMetric.DOT, DistanceMetric.EUCLIDEAN}
    )

    _METRIC_MAP = {
        DistanceMetric.COSINE: Distance.COSINE,
        DistanceMetric.DOT: Distance.DOT,
        DistanceMetric.EUCLIDEAN: Distance.EUCLID,
    }

    def __init__(
        self,
        *,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        vector_size: int,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        default_collection: str = "default",
    ) -> None:
        self._url = url
        self._api_key = api_key
        self._vector_size = vector_size
        self._default_metric = distance_metric
        self._default_collection = default_collection
        self._client: AsyncQdrantClient | None = None
        self._initialized = False
        self._ensured_collections: set[str] = set()

    async def initialize(self) -> None:
        """Initialize the Qdrant client and check connectivity."""
        try:
            self._client = AsyncQdrantClient(url=self._url, api_key=self._api_key)
            # Simple connectivity check
            await self._client.get_collections()
            self._initialized = True
            logger.info("QdrantVectorStore initialized at %s", self._url)
        except Exception as exc:
            raise VectorStoreConnectionError(
                f"Failed to connect to Qdrant at {self._url}: {exc}",
            ) from exc

    async def shutdown(self) -> None:
        """Close the Qdrant client connection."""
        if self._client:
            await self._client.close()
            self._client = None
            self._initialized = False
            logger.debug("QdrantVectorStore shut down.")

    async def _ensure_collection(self, collection: str) -> None:
        """Create a collection if it doesn't exist."""
        if collection in self._ensured_collections or not self._client:
            return

        try:
            collections = await self._client.get_collections()
            exists = any(c.name == collection for c in collections.collections)

            if not exists:
                await self._client.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(
                        size=self._vector_size,
                        distance=self._METRIC_MAP[self._default_metric],
                    ),
                )
                logger.info("Created Qdrant collection: %s", collection)

            self._ensured_collections.add(collection)
        except Exception as exc:
            raise VectorStoreOperationError(
                f"Failed to ensure collection '{collection}': {exc}",
            ) from exc

    async def add(self, chunks: list[Chunk], *, collection: str = "default") -> int:
        """Upsert chunks into a Qdrant collection."""
        if not self._client or not self._initialized:
            raise VectorStoreOperationError("Store not initialized. Use async with.")
        if not chunks:
            return 0

        await self._ensure_collection(collection)

        points = []
        for chunk in chunks:
            if chunk.embedding is None:
                raise VectorStoreOperationError(f"Chunk {chunk.id} is missing an embedding.")

            payload = {
                "document_id": chunk.document_id,
                "content": chunk.content,
                "metadata": chunk.metadata,
                "chunk_index": chunk.chunk_index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "token_count": chunk.token_count,
            }

            # Qdrant requires UUIDs or integers for point IDs
            try:
                point_id = str(uuid.UUID(chunk.id))
            except ValueError:
                # If not a valid UUID, use deterministic hash (less ideal but functional)
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.id))

            points.append(
                PointStruct(
                    id=point_id,
                    vector=chunk.embedding,
                    payload=payload,
                )
            )

        try:
            await self._client.upsert(collection_name=collection, points=points)
            return len(points)
        except Exception as exc:
            raise VectorStoreOperationError(
                f"Failed to upsert chunks to Qdrant: {exc}",
            ) from exc

    async def search(
        self,
        query_embedding: list[float],
        *,
        collection: str = "default",
        top_k: int = 5,
        filter: dict[str, str | int | float | bool] | None = None,
        metric: DistanceMetric | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Search Qdrant for similar vectors."""
        if not self._client or not self._initialized:
            raise VectorStoreOperationError("Store not initialized. Use async with.")

        await self._ensure_collection(collection)

        # Qdrant uses the metric defined at collection creation.
        # We ignore the `metric` override here for simplicity, but could switch collections.
        qdrant_filter = None
        if filter:
            conditions = [
                FieldCondition(key=f"metadata.{k}", match=MatchValue(value=v))
                for k, v in filter.items()
            ]
            qdrant_filter = Filter(must=conditions)

        try:
            results = await self._client.search(
                collection_name=collection,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=qdrant_filter,
                with_payload=True,
            )

            parsed_results: list[tuple[Chunk, float]] = []
            for res in results:
                payload = res.payload or {}
                chunk = Chunk(
                    id=str(res.id),
                    document_id=payload.get("document_id", ""),
                    content=payload.get("content", ""),
                    metadata=payload.get("metadata", {}),
                    chunk_index=payload.get("chunk_index", 0),
                    start_char=payload.get("start_char", 0),
                    end_char=payload.get("end_char", 0),
                    token_count=payload.get("token_count", 0),
                    embedding=None,  # Don't return the embedding on search to save memory
                )
                parsed_results.append((chunk, res.score))

            return parsed_results
        except Exception as exc:
            raise VectorStoreOperationError(
                f"Failed to search Qdrant: {exc}",
            ) from exc

    async def delete(self, chunk_ids: list[str], *, collection: str = "default") -> int:
        """Delete chunks from Qdrant by their IDs."""
        if not self._client or not self._initialized:
            raise VectorStoreOperationError("Store not initialized. Use async with.")

        # Convert IDs to UUID format if necessary (Qdrant is strict about this)
        uuid_ids = []
        for cid in chunk_ids:
            try:
                uuid_ids.append(str(uuid.UUID(cid)))
            except ValueError:
                uuid_ids.append(str(uuid.uuid5(uuid.NAMESPACE_DNS, cid)))

        try:
            await self._client.delete(
                collection_name=collection,
                points_selector=PointIdsList(points=uuid_ids),
            )
            return len(uuid_ids)
        except Exception as exc:
            raise VectorStoreOperationError(
                f"Failed to delete from Qdrant: {exc}",
            ) from exc

    async def get(self, chunk_ids: list[str], *, collection: str = "default") -> list[Chunk]:
        """Retrieve chunks by ID from Qdrant."""
        if not self._client or not self._initialized:
            raise VectorStoreOperationError("Store not initialized. Use async with.")

        uuid_ids = []
        for cid in chunk_ids:
            try:
                uuid_ids.append(str(uuid.UUID(cid)))
            except ValueError:
                uuid_ids.append(str(uuid.uuid5(uuid.NAMESPACE_DNS, cid)))

        try:
            points = await self._client.retrieve(
                collection_name=collection,
                ids=uuid_ids,
                with_payload=True,
                with_vectors=False,
            )

            chunks: list[Chunk] = []
            for p in points:
                payload = p.payload or {}
                chunks.append(
                    Chunk(
                        id=str(p.id),
                        document_id=payload.get("document_id", ""),
                        content=payload.get("content", ""),
                        metadata=payload.get("metadata", {}),
                        chunk_index=payload.get("chunk_index", 0),
                        start_char=payload.get("start_char", 0),
                        end_char=payload.get("end_char", 0),
                        token_count=payload.get("token_count", 0),
                        embedding=None,
                    )
                )
            return chunks
        except Exception as exc:
            raise VectorStoreOperationError(
                f"Failed to retrieve from Qdrant: {exc}",
            ) from exc

    async def count(self, *, collection: str = "default") -> int:
        """Return the exact count of points in a Qdrant collection."""
        if not self._client or not self._initialized:
            raise VectorStoreOperationError("Store not initialized. Use async with.")

        try:
            result = await self._client.count(collection_name=collection, exact=True)
            return result.count
        except Exception as exc:
            raise VectorStoreOperationError(
                f"Failed to count Qdrant collection: {exc}",
            ) from exc

    async def health_check(self) -> bool:
        """Check Qdrant connectivity."""
        if not self._client:
            return False
        try:
            await self._client.get_collections()
            return True
        except Exception:
            return False
