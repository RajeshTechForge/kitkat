"""RAG prompt construction."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kitkat.rag.context.assembler import ContextAssembler
    from kitkat.rag.core.models import RetrievalResult


class RAGPromptBuilder:
    """Builds complete RAG prompts from templates."""

    DEFAULT_SYSTEM_TEMPLATE = """\
You are a helpful assistant. Use the following context to answer the
user's question. If the context doesn't contain enough information,
say so clearly. Cite sources using [1], [2], etc. notation.

Context:
{context}
"""

    def __init__(
        self,
        *,
        system_template: str | None = None,
        context_assembler: ContextAssembler | None = None,
        include_citation_instructions: bool = True,
    ) -> None:
        self._template = system_template or self.DEFAULT_SYSTEM_TEMPLATE
        self._context_assembler = context_assembler
        self._include_citations = include_citation_instructions

    def build_messages(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> list[tuple[str, str]]:
        """Build (role, content) message pairs for the LLM.

        Args:
            query: The user's query.
            results: The retrieved chunks to use as context.

        Returns:
            List of (role, content) tuples, typically [("system", ...), ("user", ...)]
        """
        if self._context_assembler:
            context_str = self._context_assembler.assemble(results, query=query)
        else:
            # Fallback simple assembly if no assembler provided
            context_str = "\n\n".join([f"[{r.rank}] {r.chunk.content}" for r in results])

        system_prompt = self._template.format(context=context_str)

        if not self._include_citations:
            # Strip out the citation instruction if explicitly disabled
            system_prompt = system_prompt.replace("Cite sources using [1], [2], etc. notation.", "")

        return [
            ("system", system_prompt),
            ("user", query),
        ]
