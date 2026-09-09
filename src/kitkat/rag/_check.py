"""Guards for optional RAG dependencies."""

from __future__ import annotations

import importlib.util


def require_rag_extra(extra_name: str) -> None:
    """Verify that an optional RAG extra is installed."""

    match extra_name:
        case "rag-qdrant":
            if importlib.util.find_spec("qdrant-client") is None:
                raise ImportError(
                    "The 'rag-qdrant' extra is required for Qdrant vector store. "
                    "Please install it with 'pip install kitkat[rag-qdrant]'."
                )

        case "rag-rerank":
            if importlib.util.find_spec("sentence_transformers") is None:
                raise ImportError(
                    "The 'rag-rerank' extra is required for reranking. "
                    "Please install it with 'pip install kitkat[rag-rerank]'."
                )
        case _:
            pass
