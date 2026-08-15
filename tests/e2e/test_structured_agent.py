"""End-to-end tests for kitkat PydanticAI structured output agent execution."""

from __future__ import annotations

import os

import pytest
from pydantic import BaseModel, Field

from kitkat.agents import BaseAgentContext, ManagedModelAdapter, build_structured_agent
from kitkat.core import ProviderType
from kitkat.providers.google import GoogleConfig, GoogleProvider
from kitkat.service import create_llm_service

pytestmark = pytest.mark.integration


class CapitalCityResult(BaseModel):
    """Pydantic model for structured capital city response."""

    country: str = Field(description="Name of the requested country")
    capital: str = Field(description="Capital city name")


@pytest.fixture(autouse=True)
def check_e2e_requirements() -> None:
    """Ensure required API KEY environment variable is present."""
    if not os.getenv("GOOGLE_API_KEY"):
        pytest.skip("API Key required for E2E structured agent tests.")


@pytest.mark.asyncio
async def test_managed_structured_agent_e2e() -> None:
    """Verify end-to-end PydanticAI structured agent execution returning validated BaseModel."""
    provider = GoogleProvider(GoogleConfig(api_key=os.environ["GOOGLE_API_KEY"]))
    service = create_llm_service({ProviderType.GOOGLE: provider})
    await service.initialize()

    try:
        adapter = ManagedModelAdapter(
            service=service,
            provider_type=ProviderType.GOOGLE,
            default_model="gemini-3.1-flash-lite",
        )
        agent = build_structured_agent(
            model=adapter,
            output_type=CapitalCityResult,
            context_type=BaseAgentContext,
            output_retries=3,
        )
        ctx = BaseAgentContext(user_id="e2e-structured-user")

        result = await agent.run("What is the capital of France?", deps=ctx)

        assert isinstance(result.output, CapitalCityResult)
        assert result.output.country.lower() == "france"
        assert result.output.capital.lower() == "paris"
    finally:
        await service.shutdown()
