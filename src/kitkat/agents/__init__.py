"""kitkat.agents — PydanticAI adapter layer.

Provides the two ``Model`` adapters that bridge kitkat providers into
PydanticAI, along with context types, agent builders, and tool registration.

Imports from this package require the ``agents`` extra::

    pip install kitkat[agents]

"""

from __future__ import annotations

from .adapters.byok import BYOKModelAdapter
from .adapters.managed import ManagedModelAdapter
from .builders import build_chat_agent, build_structured_agent
from .context import BaseAgentContext, RoutingTier
from .tools.registry import ToolRegistry

__all__ = [
    "BYOKModelAdapter",
    "BaseAgentContext",
    "ManagedModelAdapter",
    "RoutingTier",
    "ToolRegistry",
    "build_chat_agent",
    "build_structured_agent",
]
