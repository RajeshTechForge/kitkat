"""kitkat.agents — PydanticAI adapter layer.

Provides the two ``Model`` adapters that bridge kitkat providers into
PydanticAI, along with context types, agent builders, and tool registration.

Imports from this package require the ``agents`` extra::

    pip install kitkat[agents]

Canonical imports::

    from kitkat.agents import ManagedModelAdapter, BYOKModelAdapter
    from kitkat.agents import BaseAgentContext, RoutingTier
    from kitkat.agents import build_chat_agent, build_structured_agent
    from kitkat.agents import ToolRegistry
"""

from __future__ import annotations

import importlib
from typing import Any

from .context import BaseAgentContext, RoutingTier

_LAZY_MAP: dict[str, str] = {
    "BYOKModelAdapter": "kitkat.agents.adapters.byok",
    "KitkatStreamedResponse": "kitkat.agents.adapters.managed",
    "ManagedModelAdapter": "kitkat.agents.adapters.managed",
    "ToolRegistry": "kitkat.agents.tools.registry",
    "build_chat_agent": "kitkat.agents.builders",
    "build_structured_agent": "kitkat.agents.builders",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_MAP:
        module_path = _LAZY_MAP[name]
        try:
            mod = importlib.import_module(module_path)
            return getattr(mod, name)
        except ImportError as exc:
            raise ImportError(
                f"'{name}' requires the 'agents' extra. Install with: pip install kitkat[agents]"
            ) from exc
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "BYOKModelAdapter",
    "BaseAgentContext",
    "ManagedModelAdapter",
    "RoutingTier",
    "ToolRegistry",
    "build_chat_agent",
    "build_structured_agent",
]
