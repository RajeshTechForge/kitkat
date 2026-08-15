"""Live integration tests for GoogleProvider against real API endpoints.

Must be run with INTEGRATION_TESTS=1 GOOGLE_API_KEY=AIzaSy... pytest tests/integration
"""

from __future__ import annotations

import os

import pytest

from kitkat.core import (
    LLMAuthenticationError,
    LLMRequest,
    Message,
    Role,
)
from kitkat.providers.google import GoogleConfig, GoogleProvider

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def check_google_key() -> None:
    """Ensure GOOGLE_API_KEY environment variable is present."""
    if not os.getenv("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY not set in environment.")


@pytest.mark.asyncio
async def test_google_live_complete() -> None:
    """Verify live non-streaming completion call against Google API."""
    config = GoogleConfig(api_key=os.environ["GOOGLE_API_KEY"])
    async with GoogleProvider(config) as provider:
        request = LLMRequest(
            messages=[Message(role=Role.USER, content="Reply with: OK")],
            model="google-2.5-flash",
            max_tokens=10,
        )
        response = await provider.complete_with_retry(request)

        assert response.content is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.latency_ms > 0.0


@pytest.mark.asyncio
async def test_google_live_stream() -> None:
    """Verify live streaming token deltas against Google API."""
    config = GoogleConfig(api_key=os.environ["GOOGLE_API_KEY"])
    async with GoogleProvider(config) as provider:
        request = LLMRequest(
            messages=[Message(role=Role.USER, content="Count 1 to 3.")],
            model="google-2.5-flash",
            max_tokens=20,
        )
        chunks = []
        async for chunk in provider.stream(request):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert chunks[-1].is_final is True


@pytest.mark.asyncio
async def test_google_invalid_key_raises_auth_error() -> None:
    """Verify invalid API key raises LLMAuthenticationError."""
    config = GoogleConfig(api_key="AIzaSyInvalidTestKey123456789")
    provider = GoogleProvider(config)
    await provider._init_client_only()
    try:
        request = LLMRequest(
            messages=[Message(role=Role.USER, content="Hi")],
            max_tokens=5,
        )
        with pytest.raises(LLMAuthenticationError):
            await provider.complete(request)
    finally:
        await provider.shutdown()
