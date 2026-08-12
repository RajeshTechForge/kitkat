"""End-to-end tests for kitkat PydanticAI chat agent execution."""

from __future__ import annotations

import os

import pytest

from kitkat.agents import BaseAgentContext, ManagedModelAdapter, build_chat_agent
from kitkat.core import ProviderType
from kitkat.providers.anthropic import AnthropicConfig, AnthropicProvider
from kitkat.service import create_llm_service

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def check_e2e_requirements() -> None:
    """Ensure ANTHROPIC_API_KEY environment variable is present."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY required for E2E chat agent tests.")


@pytest.mark.asyncio
async def test_managed_chat_agent_e2e() -> None:
    """Verify full end-to-end PydanticAI agent run using ManagedModelAdapter."""
    provider = AnthropicProvider(AnthropicConfig(api_key=os.environ["ANTHROPIC_API_KEY"]))
    service = create_llm_service({ProviderType.ANTHROPIC: provider})
    await service.initialize()

    try:
        adapter = ManagedModelAdapter(
            service=service,
            provider_type=ProviderType.ANTHROPIC,
            default_model="claude-3-5-haiku-20241022",
        )
        agent = build_chat_agent(model=adapter, context_type=BaseAgentContext)
        ctx = BaseAgentContext(user_id="e2e-test-user")

        result = await agent.run("Reply with exactly: AGENT_OK", deps=ctx)

        assert result.output is not None
        assert "AGENT_OK" in result.output.upper()
    finally:
        await service.shutdown()
