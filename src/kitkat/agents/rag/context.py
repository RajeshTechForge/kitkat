# src/kitkat/agents/rag/context.py
"""Agent context for RAG-aware workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from kitkat.agents.context import BaseAgentContext

if TYPE_CHECKING:
    from kitkat.rag.pipeline import RAGPipeline


@dataclass
class RAGAgentContext(BaseAgentContext):
    """Extends BaseAgentContext with a reference to the RAGPipeline.

    Attributes:
        rag_pipeline: The initialized RAGPipeline instance.
        collection: The default vector store collection to use.
    """

    rag_pipeline: RAGPipeline
    collection: str = "default"
