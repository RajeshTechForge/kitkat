"""ToolRegistry: programmatic tool registration for PydanticAI agents.

The library ships the *registration mechanism*, not domain-specific tools.
Application code registers tools using pydantic-ai's ``@agent.tool`` decorator
or via this registry for programmatic bulk registration across multiple agents.

Auth0-ready tool pattern:
    Every tool receives ``RunContext[UserContext]`` where ``UserContext`` carries
    a ``token_store``. Tools check ``token_store.get_token(service)`` before
    calling third-party APIs. When Auth0 Token Vault is wired in, that method
    returns real tokens — tool signatures don't change.

Usage::

    from kitkat.agents.tools.registry import ToolRegistry

    registry = ToolRegistry()

    @registry.tool
    async def search_docs(ctx, query: str) -> str:
        '''Search internal documentation.'''
        return "..."

    @registry.tool(
        name="translate",
        description="Translate text to a target language.",
    )
    async def translate_text(ctx, text: str, target_lang: str) -> str:
        return text

    agent = build_chat_agent(model=adapter)
    registry.register_on(agent)
"""

from __future__ import annotations

from .._check import require_agents_extra

require_agents_extra()

from typing import TYPE_CHECKING, Any, overload

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic_ai import Agent


class _RegisteredTool:
    """Internal container for a tool function and its metadata."""

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        prep: bool = False,
    ) -> None:
        self.fn = fn
        self.name = name
        self.description = description
        self.prep = prep


class ToolRegistry:
    """Collects tool callables and bulk-registers them onto PydanticAI agents.

    Useful when multiple agents should share the same tool set, or when tools
    are assembled dynamically (e.g. from a plugin system) before the agent is
    constructed.

    The registry supports both bare decorators and metadata-rich decorators::

        registry = ToolRegistry()

        @registry.tool
        async def simple_tool(ctx, x: int) -> int:
            return x + 1

        @registry.tool(name="custom_name", description="Does a thing.")
        async def annotated_tool(ctx, x: str) -> str:
            return x.upper()

    After collection, register all tools on an agent::

        agent = build_chat_agent(model=adapter)
        registry.register_on(agent)
    """

    def __init__(self) -> None:
        self._tools: list[_RegisteredTool] = []

    @overload
    def tool(self, fn: Callable[..., Any]) -> Callable[..., Any]: ...

    @overload
    def tool(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        prep: bool = False,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...

    def tool(
        self,
        fn: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        prep: bool = False,
    ) -> Any:
        """Decorator: enqueue a callable for later registration on agents.

        Can be used bare (``@registry.tool``) or with metadata
        (``@registry.tool(name=..., description=...)``).

        Args:
            fn: A coroutine function compatible with pydantic-ai's tool
                protocol. Its first argument must be ``RunContext[ContextT]``.
            name: Override the tool name exposed to the LLM. Defaults to the
                function name.
            description: Override the tool description. Defaults to the
                function's docstring.
            prep: If ``True``, the tool is registered with
                ``agent.tool(prepare=True)`` in v2.x — enabling pydantic-ai's
                tool preparation hook for dynamic tool definitions.

        Returns:
            The original function (bare form) or a decorator (metadata form).
        """

        def _register(target: Callable[..., Any]) -> Callable[..., Any]:
            self._tools.append(
                _RegisteredTool(target, name=name, description=description, prep=prep)
            )
            return target

        if fn is not None:
            # Bare decorator: @registry.tool
            return _register(fn)
        # Metadata form: @registry.tool(name=..., description=...)
        return _register

    def register_on(self, agent: Agent[Any, Any]) -> None:
        """Register all collected tools on the given agent.

        Calls ``agent.tool(fn, name=..., description=...)`` for each function
        in the registry, in registration order.

        Args:
            agent: The PydanticAI ``Agent`` instance to register tools on.
        """
        for entry in self._tools:
            kwargs: dict[str, Any] = {}
            if entry.name is not None:
                kwargs["name"] = entry.name
            if entry.description is not None:
                kwargs["description"] = entry.description
            if entry.prep:
                kwargs["prepare"] = True
            agent.tool(entry.fn, **kwargs)

    @property
    def tools(self) -> list[Callable[..., Any]]:
        """Read-only list of the registered tool callables (without metadata)."""
        return [entry.fn for entry in self._tools]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, fn: Callable[..., Any]) -> bool:
        return any(entry.fn is fn for entry in self._tools)
