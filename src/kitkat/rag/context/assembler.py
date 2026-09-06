"""Context assembly from retrieved chunks."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

from kitkat.rag._internal.tokenizers import count_tokens

if TYPE_CHECKING:
    from kitkat.rag.core.models import RetrievalResult

logger = logging.getLogger(__name__)


class ContextAssembler:
    """Assembles retrieved chunks into a coherent context string.

    Handles token budget management and formatting.
    """

    def __init__(
        self,
        *,
        max_context_tokens: int = 4096,
        tokenizer: Callable[[str], int] | None = None,
        format_style: str = "numbered",
        include_metadata: bool = True,
    ) -> None:
        self._max_tokens = max_context_tokens
        self._tokenizer = tokenizer or count_tokens
        self._format_style = format_style
        self._include_metadata = include_metadata

    def assemble(
        self,
        results: list[RetrievalResult],
        *,
        query: str | None = None,
    ) -> str:
        """Build a context string from retrieval results.

        Args:
            results: Ranked list of retrieval results.
            query: Optional original query (unused in default impl, but available).

        Returns:
            Formatted context string respecting the token budget.
        """
        if not results:
            return ""

        context_parts: list[str] = []
        current_tokens = 0

        for res in results:
            chunk_text = res.chunk.content
            header = ""

            if self._include_metadata:
                source = res.chunk.metadata.get("source", "unknown")
                header = f"(source: {source})\n"

            if self._format_style == "numbered":
                part = f"[{res.rank}] {header}{chunk_text}\n"
            elif self._format_style == "bulleted":
                part = f"* {header}{chunk_text}\n"
            elif self._format_style == "xml":
                part = f'<chunk rank="{res.rank}" source="{source}">\n{chunk_text}\n</chunk>\n'
            else:  # prose
                part = f"{chunk_text}\n\n"

            part_tokens = self._tokenizer(part)

            if current_tokens + part_tokens > self._max_tokens:
                logger.debug(
                    "Context budget exceeded (%d + %d > %d). Truncating context.",
                    current_tokens,
                    part_tokens,
                    self._max_tokens,
                )
                # Try to fit a truncated version of the current chunk
                remaining_budget = self._max_tokens - current_tokens
                if remaining_budget > 50:  # Only add if we have meaningful space left
                    truncated_text = chunk_text[: remaining_budget * 4]  # rough char estimate
                    part = f"[{res.rank}] {header}{truncated_text}... [truncated]\n"
                    context_parts.append(part)
                break

            context_parts.append(part)
            current_tokens += part_tokens

        return "".join(context_parts).strip()
