"""ToolRegistry: programmatic tool registration for PydanticAI agents.

The library ships the *registration mechanism*, not domain-specific tools.
Application code registers tools using pydantic-ai's ``@agent.tool`` decorator
or via this registry for programmatic bulk registration on multiple agents.

Usage::

    from kitkat.agents.tools.registry import ToolRegistry

    registry = ToolRegistry()

    @registry.tool
    async def search_docs(ctx, query: str) -> str:
        return "..."

    # Register all tools on an agent at once
    registry.register_on(agent)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    import pydantic_ai as _  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "ToolRegistry requires the 'agents' extra. Install with: pip install kitkat[agents]"
    ) from exc

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic_ai import Agent


class ToolRegistry:
    """Collects tool callables and bulk-registers them onto PydanticAI agents.

    Useful when multiple agents should share the same tool set, or when tools
    are assembled dynamically (e.g. from a plugin system) before the agent is
    constructed.

    Usage::

        registry = ToolRegistry()

        @registry.tool
        async def summarise(ctx, text: str) -> str:
            return text[:200]

        @registry.tool
        async def translate(ctx, text: str, target_lang: str) -> str:
            return text  # stub

        agent = build_chat_agent(model=adapter)
        registry.register_on(agent)
    """

    def __init__(self) -> None:
        self._tools: list[Callable[..., Any]] = []

    def tool(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator: enqueue a callable for later registration on agents.

        Args:
            fn: A coroutine function compatible with pydantic-ai's tool protocol.
                Its first argument must be ``RunContext[ContextT]``.

        Returns:
            The original function, unmodified — the decorator is transparent.
        """
        self._tools.append(fn)
        return fn

    def register_on(self, agent: Agent) -> None:  # type: ignore[type-arg]
        """Register all collected tools on the given agent.

        Calls ``agent.tool(fn)`` for each function in the registry in the order
        they were decorated.  Idempotent per agent only if pydantic-ai's
        ``agent.tool()`` is idempotent (it raises on duplicate names).

        Args:
            agent: The PydanticAI ``Agent`` instance to register tools on.
        """
        for tool_fn in self._tools:
            agent.tool(tool_fn)

    @property
    def tools(self) -> list[Callable[..., Any]]:
        """Read-only view of the registered tool callables."""
        return list(self._tools)
