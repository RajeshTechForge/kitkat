"""Unit tests for agent builder factory functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

if TYPE_CHECKING:
    from unittest.mock import MagicMock

from kitkat.agents.builders import build_chat_agent, build_structured_agent
from kitkat.agents.context import BaseAgentContext


class TestBuildChatAgent:
    def test_returns_agent_instance(self) -> None:
        agent = build_chat_agent(model=TestModel())
        assert isinstance(agent, Agent)

    def test_output_type_is_str(self) -> None:
        agent = build_chat_agent(model=TestModel())
        assert agent.output_type is str

    def test_deps_type_defaults_to_base_context(self) -> None:
        agent = build_chat_agent(model=TestModel())
        assert agent.deps_type is BaseAgentContext

    def test_custom_context_type_applied(self) -> None:
        @dataclass
        class MyContext(BaseAgentContext):
            extra: str = ""

        agent = build_chat_agent(model=TestModel(), context_type=MyContext)
        assert agent.deps_type is MyContext

    def test_static_system_prompt_applied(self) -> None:
        agent = build_chat_agent(model=TestModel(), system_prompt="Be terse.")
        assert isinstance(agent, Agent)
        assert any("Be terse." in str(p) for p in agent._system_prompts)

    def test_empty_prompt_uses_dynamic(self) -> None:
        agent = build_chat_agent(model=TestModel(), system_prompt="")
        assert isinstance(agent, Agent)

    def test_output_retries_applied(self) -> None:
        agent = build_chat_agent(model=TestModel(), output_retries=3)
        assert isinstance(agent, Agent)


class _SampleResult(BaseModel):
    content: str
    confidence: float


class TestBuildStructuredAgent:
    def test_returns_agent_instance(self) -> None:
        agent = build_structured_agent(model=TestModel(), output_type=_SampleResult)
        assert isinstance(agent, Agent)

    def test_output_type_set_correctly(self) -> None:
        agent = build_structured_agent(model=TestModel(), output_type=_SampleResult)
        assert agent.output_type is _SampleResult

    def test_custom_context_type_applied(self) -> None:
        @dataclass
        class MyCtx(BaseAgentContext):
            token: str = ""

        agent = build_structured_agent(
            model=TestModel(),
            output_type=_SampleResult,
            context_type=MyCtx,
        )
        assert agent.deps_type is MyCtx

    def test_custom_system_prompt_applied(self) -> None:
        agent = build_structured_agent(
            model=TestModel(),
            output_type=_SampleResult,
            system_prompt="Output valid JSON.",
        )
        assert isinstance(agent, Agent)

    def test_custom_validator_registered(self) -> None:
        def my_validator(value: _SampleResult, ctx: MagicMock) -> _SampleResult:
            return value

        agent = build_structured_agent(
            model=TestModel(),
            output_type=_SampleResult,
            validator=my_validator,
        )
        assert isinstance(agent, Agent)
