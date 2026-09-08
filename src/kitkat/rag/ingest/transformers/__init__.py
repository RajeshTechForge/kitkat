"""Transformers for processing documents in the RAG pipeline."""

from .base import DocumentTransformer
from .dedup import Deduplicator
from .metadata import MetadataExtractor

__all__ = ["DocumentTransformer", "Deduplicator", "MetadataExtractor"]
