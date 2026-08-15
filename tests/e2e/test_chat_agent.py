"""End-to-end tests for kitkat PydanticAI chat agent execution."""

from __future__ import annotations

import os

import pytest

from kitkat.agents import BaseAgentContext, ManagedModelAdapter, build_chat_agent
from kitkat.core import ProviderType
from kitkat.providers.openai import OpenAIConfig, OpenAIProvider
from kitkat.service import create_llm_service

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def check_e2e_requirements() -> None:
    """Ensure required API Key environment variable is present."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("API Key required for E2E chat agent tests.")


@pytest.mark.asyncio
async def test_managed_chat_agent_e2e() -> None:
    """Verify full end-to-end PydanticAI agent run using ManagedModelAdapter."""
    provider = OpenAIProvider(
        OpenAIConfig(api_key=os.environ["OPENAI_API_KEY"], base_url="https://openrouter.ai/api/v1")
    )
    service = create_llm_service({ProviderType.OPENAI: provider})
    await service.initialize()

    try:
        adapter = ManagedModelAdapter(
            service=service,
            provider_type=ProviderType.OPENAI,
            default_model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        )
        agent = build_chat_agent(model=adapter, context_type=BaseAgentContext)
        ctx = BaseAgentContext(user_id="e2e-test-user")

        result = await agent.run("Reply with exactly: AGENT_OK", deps=ctx)

        assert result.output is not None
        assert "AGENT_OK" in result.output.upper()
    finally:
        await service.shutdown()
