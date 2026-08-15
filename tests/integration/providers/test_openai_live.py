"""Live integration tests for OpenAIProvider using OpenAI & OpenAI-compatible endpoints.

Must be run with INTEGRATION_TESTS=1 OPENAI_API_KEY=nvapi-... pytest tests/integration
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
from kitkat.providers.openai import OpenAIConfig, OpenAIProvider

pytestmark = pytest.mark.integration

_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"


@pytest.fixture(autouse=True)
def check_openai_key() -> None:
    """Ensure OPENAI_API_KEY environment variable is present."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set in environment.")


def _get_config() -> OpenAIConfig:
    """Return OpenAIConfig configured for the target endpoint."""
    api_key = os.environ["OPENAI_API_KEY"]
    base_url = os.getenv("OPENAI_BASE_URL", _DEFAULT_BASE_URL)
    model = os.getenv("OPENAI_MODEL", _DEFAULT_MODEL)
    return OpenAIConfig(api_key=api_key, base_url=base_url, model=model)


@pytest.mark.asyncio
async def test_openai_live_complete() -> None:
    """Verify live non-streaming completion call against OpenAI-compatible endpoint."""
    config = _get_config()
    async with OpenAIProvider(config) as provider:
        request = LLMRequest(
            messages=[Message(role=Role.USER, content="Reply with: OK")],
            model=config.model,
            max_tokens=10,
        )
        response = await provider.complete_with_retry(request)

        assert response.content is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.latency_ms > 0.0


@pytest.mark.asyncio
async def test_openai_live_stream() -> None:
    """Verify live streaming token deltas against OpenAI-compatible endpoint."""
    config = _get_config()
    async with OpenAIProvider(config) as provider:
        request = LLMRequest(
            messages=[Message(role=Role.USER, content="Count 1 to 3.")],
            model=config.model,
            max_tokens=20,
        )
        chunks = []
        async for chunk in provider.stream(request):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert chunks[-1].is_final is True


@pytest.mark.asyncio
async def test_openai_invalid_key_raises_auth_error() -> None:
    """Verify invalid API key raises LLMAuthenticationError."""
    base_url = os.getenv("OPENAI_BASE_URL", _DEFAULT_BASE_URL)
    model = os.getenv("OPENAI_MODEL", _DEFAULT_MODEL)
    config = OpenAIConfig(
        api_key="nvapi-invalid-test-key-12345",
        base_url=base_url,
        model=model,
    )
    provider = OpenAIProvider(config)
    await provider._init_client_only()
    try:
        request = LLMRequest(
            messages=[Message(role=Role.USER, content="Hi")],
            model=model,
            max_tokens=5,
        )
        with pytest.raises(LLMAuthenticationError):
            await provider.complete(request)
    finally:
        await provider.shutdown()
