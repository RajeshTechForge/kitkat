"""Shared pytest fixtures for kitkat test suite."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from kitkat.core.enums import FinishReason, ProviderType, Role
from kitkat.core.models import (
    LLMRequest,
    LLMResponse,
    Message,
    ProviderCapabilities,
    TokenUsage,
)


@pytest.fixture
def sample_messages() -> list[Message]:
    """Return a sample conversation message sequence.

    Returns:
        List containing user and assistant Message instances.
    """
    return [
        Message(role=Role.USER, content="Hello! How are you?"),
        Message(role=Role.ASSISTANT, content="I am an AI assistant. How can I help you today?"),
    ]


@pytest.fixture
def sample_request(sample_messages: list[Message]) -> LLMRequest:
    """Return a sample LLMRequest object.

    Args:
        sample_messages: Fixture providing conversation messages.

    Returns:
        Configured LLMRequest instance.
    """
    return LLMRequest(
        messages=sample_messages,
        model="claude-sonnet-4-6",
        max_tokens=100,
        temperature=0.7,
    )


@pytest.fixture
def sample_response() -> LLMResponse:
    """Return a sample completed LLMResponse object.

    Returns:
        Populated LLMResponse instance.
    """
    return LLMResponse(
        content="Hello! How can I help you?",
        model="claude-sonnet-4-6",
        provider=ProviderType.ANTHROPIC,
        usage=TokenUsage(prompt_tokens=12, completion_tokens=8),
        finish_reason=FinishReason.STOP,
        latency_ms=350.0,
    )


@pytest.fixture
def mock_provider(sample_response: LLMResponse) -> AsyncMock:
    """Return a mock LLMProvider instance with standard methods stubbed.

    Args:
        sample_response: Fixture providing default LLMResponse.

    Returns:
        AsyncMock instance representing an LLMProvider subclass.
    """
    provider = AsyncMock()
    provider.complete_with_retry = AsyncMock(return_value=sample_response)
    provider.complete = AsyncMock(return_value=sample_response)
    provider.health_check = AsyncMock(return_value=True)
    provider.count_tokens = MagicMock(return_value=10)
    provider.CAPABILITIES = ProviderCapabilities(
        supports_streaming=True,
        supports_thinking=False,
        max_context_tokens=100_000,
        provider_type=ProviderType.ANTHROPIC,
    )
    return provider


@pytest.fixture
def require_integration(request: pytest.FixtureRequest) -> None:
    """Autouse fixture that skips integration/E2E tests unless INTEGRATION_TESTS=1.

    Args:
        request: Pytest fixture request object.
    """
    if "integration" in request.keywords and not os.getenv("INTEGRATION_TESTS"):
        pytest.skip("Integration tests skipped. Set INTEGRATION_TESTS=1 to run.")
