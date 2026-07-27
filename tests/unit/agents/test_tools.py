"""Unit tests for ToolRegistry."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from kitkat.agents.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from pydantic_ai import RunContext

    from kitkat.agents.context import BaseAgentContext


class TestToolRegistry:
    def test_bare_decorator_registers_function(self) -> None:
        registry = ToolRegistry()

        @registry.tool
        async def my_tool(ctx: RunContext[BaseAgentContext], x: int) -> int:
            return x + 1

        assert len(registry) == 1
        assert my_tool in registry

    def test_metadata_decorator_registers_with_name_and_description(self) -> None:
        registry = ToolRegistry()

        @registry.tool(name="custom_name", description="A custom tool.")
        async def my_tool(ctx: RunContext[BaseAgentContext], x: str) -> str:
            return x.upper()

        assert len(registry) == 1
        assert my_tool in registry

    def test_register_on_calls_agent_tool(self) -> None:
        registry = ToolRegistry()
        mock_agent = MagicMock()

        @registry.tool
        async def tool_a(ctx: RunContext[BaseAgentContext], x: int) -> int:
            return x

        @registry.tool(name="tool_b")
        async def tool_b_fn(ctx: RunContext[BaseAgentContext], x: str) -> str:
            return x

        registry.register_on(mock_agent)

        assert mock_agent.tool.call_count == 2
        first_call = mock_agent.tool.call_args_list[0]
        assert first_call.args[0] is tool_a
        second_call = mock_agent.tool.call_args_list[1]
        assert second_call.kwargs.get("name") == "tool_b"

    def test_register_on_passes_description_and_prep_kwargs(self) -> None:
        registry = ToolRegistry()
        mock_agent = MagicMock()

        @registry.tool(description="Does something useful.", prep=True)
        async def my_tool(ctx: RunContext[BaseAgentContext]) -> str:
            return "ok"

        registry.register_on(mock_agent)
        call = mock_agent.tool.call_args_list[0]
        assert call.kwargs.get("description") == "Does something useful."
        assert call.kwargs.get("prepare") is True

    def test_tools_property_returns_copy(self) -> None:
        registry = ToolRegistry()

        @registry.tool
        async def my_tool(ctx: RunContext[BaseAgentContext]) -> str:
            return ""

        tools = registry.tools
        tools.clear()
        assert len(registry) == 1

    def test_contains_check(self) -> None:
        registry = ToolRegistry()

        @registry.tool
        async def my_tool(ctx: RunContext[BaseAgentContext]) -> str:
            return ""

        async def unregistered_tool(ctx: RunContext[BaseAgentContext]) -> str:
            return ""

        assert my_tool in registry
        assert unregistered_tool not in registry
