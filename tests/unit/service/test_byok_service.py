"""Unit tests for BYOKLLMService (service/byok.py).

Covers the async context manager lifecycle, inference delegation, error
propagation, and __repr__ — all without network calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest

from kitkat.core.enums import FinishReason, ProviderType, Role
from kitkat.core.models import (
    LLMRequest,
    LLMResponse,
    Message,
    StreamChunk,
    TokenUsage,
)
from kitkat.service.byok import BYOKLLMService

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request() -> LLMRequest:
    return LLMRequest(messages=[Message(role=Role.USER, content="hello")])


def _canned_response(provider_type: ProviderType = ProviderType.ANTHROPIC) -> LLMResponse:
    return LLMResponse(
        content="stub",
        finish_reason=FinishReason.STOP,
        usage=TokenUsage(prompt_tokens=2, completion_tokens=1, total_tokens=3),
        model="stub",
        provider=provider_type,
    )


# ---------------------------------------------------------------------------
# Context-manager lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aenter_calls_init_client_only_not_initialize() -> None:
    """BYOKLLMService must use _init_client_only, never initialize()."""
    with patch(
        "kitkat.service.byok.AnthropicProvider",
        autospec=True,
    ) as MockProvider:
        instance = MockProvider.return_value
        instance._init_client_only = AsyncMock()
        instance.shutdown = AsyncMock()

        svc = BYOKLLMService(ProviderType.ANTHROPIC, "sk-test", "claude-3-haiku")
        svc._provider = instance

        async with svc:
            instance._init_client_only.assert_awaited_once()
            instance.initialize.assert_not_called()


@pytest.mark.asyncio
async def test_shutdown_called_on_normal_exit() -> None:
    with patch("kitkat.service.byok.AnthropicProvider", autospec=True) as MockProvider:
        instance = MockProvider.return_value
        instance._init_client_only = AsyncMock()
        instance.shutdown = AsyncMock()

        svc = BYOKLLMService(ProviderType.ANTHROPIC, "sk-test", "claude-3-haiku")
        svc._provider = instance

        async with svc:
            pass

        instance.shutdown.assert_awaited_once()


@pytest.mark.asyncio
async def test_shutdown_called_on_exception() -> None:
    """Provider shutdown must run even if inference raises, preventing leaks."""
    with patch("kitkat.service.byok.AnthropicProvider", autospec=True) as MockProvider:
        instance = MockProvider.return_value
        instance._init_client_only = AsyncMock()
        instance.shutdown = AsyncMock()
        instance.complete_with_retry = AsyncMock(side_effect=RuntimeError("inference error"))

        svc = BYOKLLMService(ProviderType.ANTHROPIC, "sk-test", "claude-3-haiku")
        svc._provider = instance

        with pytest.raises(RuntimeError, match="inference error"):
            async with svc:
                await svc.complete(_make_request())

        instance.shutdown.assert_awaited_once()


# ---------------------------------------------------------------------------
# Inference delegation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_delegates_to_complete_with_retry() -> None:
    with patch("kitkat.service.byok.AnthropicProvider", autospec=True) as MockProvider:
        instance = MockProvider.return_value
        instance._init_client_only = AsyncMock()
        instance.shutdown = AsyncMock()
        instance.complete_with_retry = AsyncMock(
            return_value=_canned_response(ProviderType.ANTHROPIC)
        )

        svc = BYOKLLMService(ProviderType.ANTHROPIC, "sk-test", "claude-3-haiku")
        svc._provider = instance

        async with svc:
            response = await svc.complete(_make_request())

        assert response.content == "stub"
        instance.complete_with_retry.assert_awaited_once()


@pytest.mark.asyncio
async def test_stream_delegates_to_provider_stream() -> None:
    async def _fake_stream(_request: LLMRequest) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(delta="tok", is_final=False)
        yield StreamChunk(delta="", is_final=True, finish_reason=FinishReason.STOP)

    with patch("kitkat.service.byok.AnthropicProvider", autospec=True) as MockProvider:
        instance = MockProvider.return_value
        instance._init_client_only = AsyncMock()
        instance.shutdown = AsyncMock()
        instance.stream = _fake_stream

        svc = BYOKLLMService(ProviderType.ANTHROPIC, "sk-test", "claude-3-haiku")
        svc._provider = instance

        chunks: list[StreamChunk] = []
        async with svc:
            async for chunk in svc.stream(_make_request()):
                chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0].delta == "tok"
    assert chunks[-1].is_final


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


def test_repr_contains_provider_and_model() -> None:
    svc = BYOKLLMService(ProviderType.ANTHROPIC, "sk-ant", "claude-3-haiku")
    r = repr(svc)
    assert "anthropic" in r
    assert "claude-3-haiku" in r


def test_repr_format() -> None:
    svc = BYOKLLMService(ProviderType.OPENAI, "sk-open", "gpt-4o")
    assert repr(svc) == "<BYOKLLMService provider='openai' model='gpt-4o'>"


# ---------------------------------------------------------------------------
# Provider dispatch
# ---------------------------------------------------------------------------


def test_build_provider_anthropic() -> None:
    provider = BYOKLLMService._build_provider(ProviderType.ANTHROPIC, "sk-ant", "claude")
    from kitkat.providers.anthropic.provider import AnthropicProvider

    assert isinstance(provider, AnthropicProvider)


def test_build_provider_openai() -> None:
    provider = BYOKLLMService._build_provider(ProviderType.OPENAI, "sk-open", "gpt-4o")
    from kitkat.providers.openai.provider import OpenAIProvider

    assert isinstance(provider, OpenAIProvider)


def test_build_provider_gemini() -> None:
    provider = BYOKLLMService._build_provider(ProviderType.GEMINI, "AIza", "gemini-2")
    from kitkat.providers.gemini.provider import GeminiProvider

    assert isinstance(provider, GeminiProvider)


def test_build_provider_empty_api_key_raises() -> None:
    """Empty api_key must raise LLMProviderInitError from config validation."""
    from kitkat.core.exceptions import LLMProviderInitError

    with pytest.raises(LLMProviderInitError, match="api_key"):
        BYOKLLMService._build_provider(ProviderType.ANTHROPIC, "", "claude")
