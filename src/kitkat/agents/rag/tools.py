# src/kitkat/agents/rag/tools.py
"""Tools available to RAG-aware PydanticAI agents."""

from __future__ import annotations

import logging

from pydantic_ai import RunContext

from kitkat.agents.rag.context import RAGAgentContext

logger = logging.getLogger(__name__)


async def retrieve_documents(
    ctx: RunContext[RAGAgentContext],
    query: str,
    top_k: int = 5,
) -> str:
    """Retrieve relevant documents from the knowledge base.

    Use this tool when you need to find information to answer the user's question.

    Args:
        ctx: The run context containing dependencies.
        query: The natural language search query.
        top_k: The maximum number of documents to retrieve (default: 5).

    Returns:
        A formatted string of retrieved documents with their relevance scores,
        or an error message if retrieval fails.
    """
    try:
        results = await ctx.deps.rag_pipeline.retrieve(
            query,
            top_k=top_k,
            collection=ctx.deps.collection,
        )

        if not results:
            return "No relevant documents found in the knowledge base."

        formatted_chunks: list[str] = []
        for res in results:
            source = res.chunk.metadata.get("source", "unknown")
            formatted_chunks.append(
                f"[{res.rank}] (Score: {res.score:.4f}, Source: {source})\n{res.chunk.content}"
            )

        return "\n\n---\n\n".join(formatted_chunks)
    except Exception as exc:
        logger.error("RAG retrieve tool failed: %s", exc)
        return f"Error retrieving documents: {exc}"


async def ingest_document(
    ctx: RunContext[RAGAgentContext],
    source: str,
) -> str:
    """Ingest a document from a local file path into the knowledge base.

    Use this tool when the user asks to add or update a document in the system.

    Args:
        ctx: The run context containing dependencies.
        source: The local file path of the document to ingest.

    Returns:
        A summary of the ingestion result, or an error message.
    """
    try:
        result = await ctx.deps.rag_pipeline.ingest(
            [source],
            collection=ctx.deps.collection,
        )

        if result.errors:
            return (
                f"Ingestion partially failed. Processed: {result.documents_processed}, "
                f"Chunks created: {result.chunks_created}, Errors: {result.errors}"
            )

        return (
            f"Successfully ingested {result.documents_processed} documents. "
            f"Created and stored {result.chunks_stored} chunks."
        )
    except Exception as exc:
        logger.error("RAG ingest tool failed: %s", exc)
        return f"Error ingesting document: {exc}"
