"""Unit tests for agent builder functions and ToolRegistry."""

from __future__ import annotations

from unittest.mock import MagicMock

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from kitkat.agents.builders import build_chat_agent, build_structured_agent
from kitkat.agents.context import BaseAgentContext
from kitkat.agents.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# build_chat_agent
# ---------------------------------------------------------------------------


class TestBuildChatAgent:
    def test_returns_agent_instance(self) -> None:
        agent = build_chat_agent(model=TestModel())
        assert isinstance(agent, Agent)

    def test_result_type_is_str(self) -> None:
        agent = build_chat_agent(model=TestModel())
        assert agent.output_type is str

    def test_deps_type_defaults_to_base_context(self) -> None:
        agent = build_chat_agent(model=TestModel())
        assert agent.deps_type is BaseAgentContext

    def test_custom_context_type_applied(self) -> None:
        from dataclasses import dataclass

        @dataclass
        class MyContext(BaseAgentContext):
            extra: str = ""

        agent = build_chat_agent(model=TestModel(), context_type=MyContext)
        assert agent.deps_type is MyContext

    def test_static_system_prompt_applied(self) -> None:
        agent = build_chat_agent(model=TestModel(), system_prompt="Be terse.")
        assert isinstance(agent, Agent)

    def test_empty_prompt_uses_dynamic(self) -> None:
        agent = build_chat_agent(model=TestModel(), system_prompt="")
        assert isinstance(agent, Agent)


# ---------------------------------------------------------------------------
# build_structured_agent
# ---------------------------------------------------------------------------


class _SampleResult(BaseModel):
    content: str
    confidence: float


class TestBuildStructuredAgent:
    def test_returns_agent_instance(self) -> None:
        agent = build_structured_agent(model=TestModel(), result_type=_SampleResult)
        assert isinstance(agent, Agent)

    def test_result_type_set_correctly(self) -> None:
        agent = build_structured_agent(model=TestModel(), result_type=_SampleResult)
        assert agent.output_type is _SampleResult

    def test_custom_context_type_applied(self) -> None:
        from dataclasses import dataclass

        @dataclass
        class MyCtx(BaseAgentContext):
            token: str = ""

        agent = build_structured_agent(
            model=TestModel(),
            result_type=_SampleResult,
            context_type=MyCtx,
        )
        assert agent.deps_type is MyCtx

    def test_custom_system_prompt_applied(self) -> None:
        agent = build_structured_agent(
            model=TestModel(),
            result_type=_SampleResult,
            system_prompt="Output valid JSON.",
        )
        assert isinstance(agent, Agent)


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


class TestToolRegistry:
    def test_tool_decorator_adds_function(self) -> None:
        registry = ToolRegistry()

        @registry.tool
        async def my_tool(ctx: BaseAgentContext, query: str) -> str:
            return query

        assert my_tool in registry.tools

    def test_tool_decorator_returns_original_function(self) -> None:
        registry = ToolRegistry()

        async def my_tool(ctx: BaseAgentContext) -> str:
            return "ok"

        result = registry.tool(my_tool)
        assert result is my_tool

    def test_multiple_tools_registered_in_order(self) -> None:
        registry = ToolRegistry()

        @registry.tool
        async def tool_a(ctx: BaseAgentContext) -> str:
            return "a"

        @registry.tool
        async def tool_b(ctx: BaseAgentContext) -> str:
            return "b"

        assert registry.tools == [tool_a, tool_b]

    def test_register_on_calls_agent_tool_for_each(self) -> None:
        registry = ToolRegistry()

        @registry.tool
        async def tool_one(ctx: BaseAgentContext) -> str:
            return "one"

        @registry.tool
        async def tool_two(ctx: BaseAgentContext) -> str:
            return "two"

        agent = MagicMock()
        registry.register_on(agent)

        assert agent.tool.call_count == 2
        agent.tool.assert_any_call(tool_one)
        agent.tool.assert_any_call(tool_two)

    def test_empty_registry_register_on_noop(self) -> None:
        registry = ToolRegistry()
        agent = MagicMock()
        registry.register_on(agent)
        agent.tool.assert_not_called()

    def test_tools_property_returns_copy(self) -> None:
        registry = ToolRegistry()

        @registry.tool
        async def tool_a(ctx: BaseAgentContext) -> str:
            return "a"

        copy = registry.tools
        copy.clear()
        assert len(registry.tools) == 1
