"""Enums for the RAG subsystem."""

from __future__ import annotations

from enum import StrEnum


class EmbeddingProviderType(StrEnum):
    """Supported embedding provider types."""

    FAKE = "fake"
    OPENAI = "openai"
    GEMINI = "gemini"


class ChunkingStrategy(StrEnum):
    """Supported document chunking strategies."""

    RECURSIVE = "recursive"
    TOKEN = "token"
    SENTENCE = "sentence"
    MARKDOWN = "markdown"
    HTML = "html"


class RetrievalStrategy(StrEnum):
    """Supported retrieval strategies."""

    VECTOR = "vector"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class VectorBackendType(StrEnum):
    """Supported vector store backend types."""

    IN_MEMORY = "in_memory"
    QDRANT = "qdrant"
    PGVECTOR = "pgvector"
    CHROMA = "chroma"
    PINECONE = "pinecone"


class DistanceMetric(StrEnum):
    """Supported distance metrics for vector search."""

    COSINE = "cosine"
    DOT = "dot"
    EUCLIDEAN = "euclidean"


class RerankerType(StrEnum):
    """Supported reranker types."""

    CROSS_ENCODER = "cross_encoder"
    LLM = "llm"
