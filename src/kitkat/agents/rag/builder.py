# src/kitkat/agents/rag/builder.py
"""Factory for building RAG-aware PydanticAI agents."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from pydantic_ai import Agent

from kitkat.agents._check import require_agents_extra
from kitkat.agents.rag.context import RAGAgentContext
from kitkat.agents.rag.tools import ingest_document, retrieve_documents

require_agents_extra()

if TYPE_CHECKING:
    from pydantic_ai.models import Model

# Type variable for context subclasses
T = TypeVar("T", bound=RAGAgentContext)

DEFAULT_SYSTEM_PROMPT = """\
You are an advanced AI assistant with access to a knowledge base via tools.
When asked a question, ALWAYS use the `retrieve_documents` tool to find relevant
information before answering. Cite your sources using [1], [2], etc. notation
based on the ranks provided in the retrieved documents. If the retrieved
documents do not contain enough information, state clearly that the answer
is not in the knowledge base.
"""


def build_rag_agent(
    *,
    model: Model,
    context_type: type[T],
    system_prompt: str | None = None,
) -> Agent[T]:
    """Build a PydanticAI agent equipped with RAG tools.

    Args:
        model: The PydanticAI Model instance (e.g., ManagedModelAdapter).
        context_type: The agent context class (must inherit from RAGAgentContext).
        system_prompt: Optional custom system prompt. Defaults to RAG instructions.

    Returns:
        A configured PydanticAI Agent instance.
    """
    agent: Agent[T] = Agent(
        model=model,
        deps_type=context_type,
        system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
    )

    # Register RAG tools
    agent.tool(retrieve_documents)
    agent.tool(ingest_document)

    return agent
