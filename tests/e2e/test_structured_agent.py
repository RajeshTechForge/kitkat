"""End-to-end tests for kitkat PydanticAI structured output agent execution."""

from __future__ import annotations

import os

import pytest
from pydantic import BaseModel, Field

from kitkat.agents import BaseAgentContext, ManagedModelAdapter, build_structured_agent
from kitkat.core import ProviderType
from kitkat.providers.anthropic import AnthropicConfig, AnthropicProvider
from kitkat.service import create_llm_service

pytestmark = pytest.mark.integration


class CapitalCityResult(BaseModel):
    """Pydantic model for structured capital city response."""

    country: str = Field(description="Name of the requested country")
    capital: str = Field(description="Capital city name")


@pytest.fixture(autouse=True)
def check_e2e_requirements() -> None:
    """Ensure ANTHROPIC_API_KEY environment variable is present."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY required for E2E structured agent tests.")


@pytest.mark.asyncio
async def test_managed_structured_agent_e2e() -> None:
    """Verify end-to-end PydanticAI structured agent execution returning validated BaseModel."""
    provider = AnthropicProvider(AnthropicConfig(api_key=os.environ["ANTHROPIC_API_KEY"]))
    service = create_llm_service({ProviderType.ANTHROPIC: provider})
    await service.initialize()

    try:
        adapter = ManagedModelAdapter(
            service=service,
            provider_type=ProviderType.ANTHROPIC,
            default_model="claude-3-5-haiku-20241022",
        )
        agent = build_structured_agent(
            model=adapter,
            result_type=CapitalCityResult,
            context_type=BaseAgentContext,
        )
        ctx = BaseAgentContext(user_id="e2e-structured-user")

        result = await agent.run("What is the capital of France?", deps=ctx)

        assert isinstance(result.output, CapitalCityResult)
        assert result.output.country.lower() == "france"
        assert result.output.capital.lower() == "paris"
    finally:
        await service.shutdown()
