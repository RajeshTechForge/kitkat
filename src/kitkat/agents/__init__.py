"""kitkat.agents — PydanticAI adapter layer.

Provides the two ``Model`` adapters that bridge kitkat providers into
PydanticAI, along with context types, agent builders, and tool registration.

Imports from this package require the ``agents`` extra::

    pip install kitkat[agents]

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .context import BaseAgentContext, RoutingTier

if TYPE_CHECKING:
    from .adapters.byok import BYOKModelAdapter
    from .adapters.managed import ManagedModelAdapter
    from .builders import build_chat_agent, build_structured_agent
    from .observability import configure_observability
    from .tools.registry import ToolRegistry

__all__ = [
    "BYOKModelAdapter",
    "BaseAgentContext",
    "ManagedModelAdapter",
    "RoutingTier",
    "ToolRegistry",
    "build_chat_agent",
    "build_structured_agent",
    "configure_observability",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "BYOKModelAdapter": (".adapters.byok", "BYOKModelAdapter"),
    "ManagedModelAdapter": (".adapters.managed", "ManagedModelAdapter"),
    "build_chat_agent": (".builders", "build_chat_agent"),
    "build_structured_agent": (".builders", "build_structured_agent"),
    "configure_observability": (".observability", "configure_observability"),
    "ToolRegistry": (".tools.registry", "ToolRegistry"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_path, attr_name = _LAZY_EXPORTS[name]
        import importlib

        mod = importlib.import_module(module_path, __package__)
        val = getattr(mod, attr_name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY_EXPORTS.keys()))
