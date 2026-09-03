# src/kitkat/rag/context/citation.py
"""Citation extraction from LLM responses."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kitkat.rag.core.models import RetrievalResult

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


class CitationExtractor:
    """Parses [1], [2] citations from LLM responses and maps to sources."""

    @staticmethod
    def extract(
        text: str,
        sources: list["RetrievalResult"],
    ) -> list["RetrievalResult"]:
        """Extract cited sources from LLM text.

        Args:
            text: The LLM generated response text.
            sources: The list of RetrievalResults provided in the context.

        Returns:
            A list of unique RetrievalResult objects cited in the text.
        """
        if not text or not sources:
            return []

        # Create a map of rank -> RetrievalResult (1-based index as string)
        source_map = {str(res.rank): res for res in sources}

        cited_ranks = _CITATION_PATTERN.findall(text)

        cited_sources = []
        seen_ranks = set()

        for rank in cited_ranks:
            if rank in source_map and rank not in seen_ranks:
                cited_sources.append(source_map[rank])
                seen_ranks.add(rank)

        return cited_sources
