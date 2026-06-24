"""Unit tests for LLMService (service/managed.py).

Covers provider registration, request routing, health checks, and token
counting without making any network calls.  All providers are replaced by
a minimal in-process stub.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from kitkat.abc import LLMProvider
from kitkat.core.enums import FinishReason, ProviderType, Role
from kitkat.core.exceptions import LLMProviderError
from kitkat.core.models import (
    LLMRequest,
    LLMResponse,
    Message,
    ProviderCapabilities,
    RetryPolicy,
    StreamChunk,
    TokenUsage,
)
from kitkat.service.factory import create_llm_service
from kitkat.service.managed import LLMService

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# ---------------------------------------------------------------------------
# Stub provider
# ---------------------------------------------------------------------------


class _StubProvider(LLMProvider):
    """Minimal LLMProvider stub that records calls and returns canned values."""

    PROVIDER_TYPE = ProviderType.ANTHROPIC
    DEFAULT_MODEL = "stub"
    CAPABILITIES = ProviderCapabilities(provider_type=ProviderType.ANTHROPIC)
    RETRY_POLICY = RetryPolicy(max_attempts=1, base_delay_s=0.0, max_delay_s=0.0, jitter=False)

    def __init__(
        self,
        provider_type: ProviderType = ProviderType.ANTHROPIC,
        health: bool = True,
        token_count: int = 7,
    ) -> None:
        super().__init__(config={"api_key": "stub"})
        self.PROVIDER_TYPE = provider_type
        self._health = health
        self._token_count = token_count
        self.initialize_called = False
        self.shutdown_called = False

    async def initialize(self) -> None:
        self.initialize_called = True
        self._initialized = True

    async def shutdown(self) -> None:
        self.shutdown_called = True
        self._initialized = False

    async def _init_client_only(self) -> None:
        self._initialized = True

    async def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            content="stub response",
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(prompt_tokens=5, completion_tokens=2, total_tokens=7),
            model="stub",
            provider=self.PROVIDER_TYPE,
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(delta="hi", is_final=False)
        yield StreamChunk(delta="", is_final=True, finish_reason=FinishReason.STOP)

    async def health_check(self) -> bool:
        return self._health

    def count_tokens(self, text: str) -> int:
        return self._token_count


def _make_request() -> LLMRequest:
    return LLMRequest(messages=[Message(role=Role.USER, content="hello")])


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------


class TestProviderRegistration:
    def test_register_single_provider(self) -> None:
        service = LLMService()
        provider = _StubProvider()
        service.register_provider(ProviderType.ANTHROPIC, provider)
        assert service.provider_count == 1

    def test_register_duplicate_raises(self) -> None:
        service = LLMService()
        service.register_provider(ProviderType.ANTHROPIC, _StubProvider())
        with pytest.raises(ValueError, match="anthropic"):
            service.register_provider(ProviderType.ANTHROPIC, _StubProvider())

    def test_providers_property_returns_copy(self) -> None:
        service = LLMService()
        service.register_provider(ProviderType.ANTHROPIC, _StubProvider())
        providers_copy = service.providers
        providers_copy.clear()
        assert service.provider_count == 1, "Modifying copy must not affect internal state"

    def test_resolve_unregistered_raises(self) -> None:
        service = LLMService()
        with pytest.raises(LLMProviderError, match="openai"):
            service._resolve(ProviderType.OPENAI)

    def test_resolve_unregistered_error_lists_available(self) -> None:
        service = LLMService()
        service.register_provider(ProviderType.ANTHROPIC, _StubProvider())
        try:
            service._resolve(ProviderType.OPENAI)
        except LLMProviderError as exc:
            assert "anthropic" in exc.message


# ---------------------------------------------------------------------------
# Lifecycle tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_initialize_calls_all_providers() -> None:
    service = LLMService()
    p1 = _StubProvider(provider_type=ProviderType.ANTHROPIC)
    p2 = _StubProvider(provider_type=ProviderType.OPENAI)
    service.register_provider(ProviderType.ANTHROPIC, p1)
    service.register_provider(ProviderType.OPENAI, p2)
    await service.initialize()
    assert p1.initialize_called
    assert p2.initialize_called


@pytest.mark.asyncio
async def test_shutdown_calls_all_providers() -> None:
    service = LLMService()
    p1 = _StubProvider(provider_type=ProviderType.ANTHROPIC)
    service.register_provider(ProviderType.ANTHROPIC, p1)
    await service.initialize()
    await service.shutdown()
    assert p1.shutdown_called
    assert service.provider_count == 0


@pytest.mark.asyncio
async def test_shutdown_tolerates_provider_error() -> None:
    """Shutdown must not propagate exceptions from individual providers."""

    class _FaultyProvider(_StubProvider):
        async def shutdown(self) -> None:
            raise RuntimeError("connection reset")

    service = LLMService()
    service.register_provider(ProviderType.ANTHROPIC, _FaultyProvider())
    await service.initialize()
    await service.shutdown()  # Must not raise
    assert service.provider_count == 0


# ---------------------------------------------------------------------------
# Inference routing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_routes_to_correct_provider() -> None:
    service = LLMService()
    provider = _StubProvider()
    await provider.initialize()
    service.register_provider(ProviderType.ANTHROPIC, provider)

    response = await service.complete(_make_request(), ProviderType.ANTHROPIC)
    assert response.content == "stub response"


@pytest.mark.asyncio
async def test_complete_unregistered_provider_raises() -> None:
    service = LLMService()
    with pytest.raises(LLMProviderError):
        await service.complete(_make_request(), ProviderType.OPENAI)


@pytest.mark.asyncio
async def test_stream_yields_chunks() -> None:
    service = LLMService()
    provider = _StubProvider()
    await provider.initialize()
    service.register_provider(ProviderType.ANTHROPIC, provider)

    chunks = [chunk async for chunk in service.stream(_make_request(), ProviderType.ANTHROPIC)]
    assert len(chunks) == 2
    assert chunks[0].delta == "hi"
    assert chunks[-1].is_final


# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_check_returns_true_for_healthy_provider() -> None:
    service = LLMService()
    service.register_provider(ProviderType.ANTHROPIC, _StubProvider(health=True))
    assert await service.health_check(ProviderType.ANTHROPIC) is True


@pytest.mark.asyncio
async def test_health_check_all_partial_failure() -> None:
    """health_check_all must return False for failing providers without raising."""
    service = LLMService()

    class _UnhealthyProvider(_StubProvider):
        async def health_check(self) -> bool:
            raise RuntimeError("probe failed")

    service.register_provider(ProviderType.ANTHROPIC, _StubProvider(health=True))
    service.register_provider(
        ProviderType.OPENAI, _UnhealthyProvider(provider_type=ProviderType.OPENAI)
    )

    results = await service.health_check_all()
    assert results[ProviderType.ANTHROPIC] is True
    assert results[ProviderType.OPENAI] is False


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


def test_count_tokens_delegates_to_provider() -> None:
    service = LLMService()
    service.register_provider(ProviderType.ANTHROPIC, _StubProvider(token_count=42))
    assert service.count_tokens(ProviderType.ANTHROPIC, "hello world") == 42


def test_count_prompt_tokens_empty_list_returns_zero() -> None:
    service = LLMService()
    service.register_provider(ProviderType.ANTHROPIC, _StubProvider())
    assert service.count_prompt_tokens(ProviderType.ANTHROPIC, []) == 0


def test_count_prompt_tokens_concatenates_messages() -> None:
    service = LLMService()
    service.register_provider(ProviderType.ANTHROPIC, _StubProvider(token_count=99))
    msgs = [
        Message(role=Role.USER, content="hello"),
        Message(role=Role.ASSISTANT, content="world"),
    ]
    assert service.count_prompt_tokens(ProviderType.ANTHROPIC, msgs) == 99


# ---------------------------------------------------------------------------
# get_capabilities
# ---------------------------------------------------------------------------


def test_get_capabilities_returns_provider_capabilities() -> None:
    service = LLMService()
    service.register_provider(ProviderType.ANTHROPIC, _StubProvider())
    caps = service.get_capabilities(ProviderType.ANTHROPIC)
    assert caps.provider_type == ProviderType.ANTHROPIC


# ---------------------------------------------------------------------------
# create_llm_service factory
# ---------------------------------------------------------------------------


def test_create_llm_service_registers_all_providers() -> None:
    p1 = _StubProvider(provider_type=ProviderType.ANTHROPIC)
    p2 = _StubProvider(provider_type=ProviderType.OPENAI)
    service = create_llm_service({ProviderType.ANTHROPIC: p1, ProviderType.OPENAI: p2})
    assert service.provider_count == 2


def test_create_llm_service_empty_dict() -> None:
    service = create_llm_service({})
    assert service.provider_count == 0
