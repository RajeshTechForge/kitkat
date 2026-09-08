"""Shared helpers for vector store implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from kitkat.rag.core.exceptions import EmbeddingDimensionError

if TYPE_CHECKING:
    from kitkat.rag.core.models import Chunk


def validate_embedding_dimensions(chunks: list[Chunk], expected_dim: int) -> None:
    """Ensure all chunks have embeddings of the expected dimension.

    Args:
        chunks: Chunks to validate.
        expected_dim: The required vector dimension.

    Raises:
        EmbeddingDimensionError: If any chunk has a missing or mismatched embedding.
    """
    for chunk in chunks:
        if chunk.embedding is None:
            raise EmbeddingDimensionError(
                f"Chunk {chunk.id} is missing an embedding.",
                provider="vector_store",
            )
        if len(chunk.embedding) != expected_dim:
            raise EmbeddingDimensionError(
                f"Chunk {chunk.id} has dimension {len(chunk.embedding)}, expected {expected_dim}.",
                provider="vector_store",
            )
