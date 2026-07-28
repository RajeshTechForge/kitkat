"""kitkat.workflows — LangGraph workflow layer.

Provides stateful, multi-step agentic workflows built on top of LangGraph.

Imports from this package require the ``workflows`` extra::

    pip install kitkat[workflows]

Canonical imports::

    from kitkat.workflows import BaseWorkflow, ResearchState, ResearchWorkflow
"""

from __future__ import annotations

import importlib.util

if importlib.util.find_spec("langgraph") is None:
    raise ImportError(
        "Workflow requires the 'langgraph' extra. Install with: pip install kitkat[langgraph]"
    )

from .base import BaseWorkflow
from .research import ResearchState, ResearchWorkflow

__all__ = [
    "BaseWorkflow",
    "ResearchState",
    "ResearchWorkflow",
]
