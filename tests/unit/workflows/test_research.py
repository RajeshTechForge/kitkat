"""Unit tests for ResearchWorkflow and ResearchState."""

from __future__ import annotations

import pytest

from kitkat import BaseAgentContext
from kitkat.workflows import BaseWorkflow, ResearchState, ResearchWorkflow


class TestResearchState:
    def test_default_values(self) -> None:
        state = ResearchState()
        assert state.user_query == ""
        assert state.research_plan == []
        assert state.search_results == []
        assert state.retrieved_docs == []
        assert state.final_answer == ""
        assert state.agent_context is None
        assert state.pending_approval is False
        assert state.approval_action is None

    def test_custom_values(self) -> None:
        ctx = BaseAgentContext(user_id="user_123")
        state = ResearchState(
            user_query="Quantum Computing",
            agent_context=ctx,
            pending_approval=True,
            approval_action="ciba_request",
        )
        assert state.user_query == "Quantum Computing"
        assert state.agent_context == ctx
        assert state.pending_approval is True
        assert state.approval_action == "ciba_request"


class TestResearchWorkflow:
    def test_subclass_of_base_workflow(self) -> None:
        workflow = ResearchWorkflow()
        assert isinstance(workflow, BaseWorkflow)

    @pytest.mark.asyncio
    async def test_execution(self) -> None:
        workflow = ResearchWorkflow()
        initial_state = ResearchState(user_query="LLM Architectures")
        final_state = await workflow.run(initial_state)

        assert isinstance(final_state, ResearchState)
        assert len(final_state.research_plan) > 0
        assert len(final_state.search_results) > 0
        assert len(final_state.retrieved_docs) > 0
        assert "Based on research:" in final_state.final_answer

    @pytest.mark.asyncio
    async def test_execution_from_dict(self) -> None:
        workflow = ResearchWorkflow()
        final_state = await workflow.run({"user_query": "Python 3.11"})

        assert isinstance(final_state, ResearchState)
        assert final_state.user_query == "Python 3.11"
        assert final_state.final_answer != ""
