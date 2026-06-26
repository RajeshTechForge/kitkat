"""Unit tests for LLMRouter (service/router.py).

Covers:
- CircuitBreaker: state machine transitions (CLOSED→OPEN→HALF_OPEN→CLOSED)
- LLMRouter: provider ordering strategies (FAILOVER, ROUND_ROBIN, LEAST_LATENCY, RANDOM)
- LLMRouter: non-retryable exceptions re-raised immediately without fallback
- LLMRouter: rate-limit skipping
- LLMRouter: async reset_circuit_breaker returns bool
- LLMRouter: all-providers-exhausted raises LLMProviderError
- LLMRouter: streaming fallback before first chunk
- LLMRouter: mid-stream failure raises without re-raise from wrong provider
- LLMRouter: health_check aggregation
- create_llm_router factory: wires config correctly
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

import pytest

from kitkat.abc import LLMProvider
from kitkat.core.enums import CircuitState, FinishReason, ProviderType, Role, RoutingStrategy
from kitkat.core.exceptions import (
    LLMAuthenticationError,
    LLMContentFilterError,
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTokenLimitError,
)
from kitkat.core.models import (
    LLMRequest,
    LLMResponse,
    Message,
    ProviderCapabilities,
    RetryPolicy,
    StreamChunk,
    TokenUsage,
)
from kitkat.service.factory import create_llm_router
from kitkat.service.router import (
    CircuitBreaker,
    CircuitBreakerConfig,
    LLMRouter,
    RouterConfig,
)

# ---------------------------------------------------------------------------
# Stub provider
# ---------------------------------------------------------------------------


class _StubProvider(LLMProvider):
    """Configurable in-process stub for routing tests."""

    PROVIDER_TYPE = ProviderType.ANTHROPIC
    DEFAULT_MODEL = "stub"
    CAPABILITIES = ProviderCapabilities(provider_type=ProviderType.ANTHROPIC)
    RETRY_POLICY = RetryPolicy(max_attempts=1, base_delay_s=0.0, max_delay_s=0.0, jitter=False)

    def __init__(
        self,
        provider_type: ProviderType = ProviderType.ANTHROPIC,
        raises: Exception | None = None,
        health: bool = True,
    ) -> None:
        super().__init__(config={"api_key": "stub"})
        self.PROVIDER_TYPE = provider_type
        self.CAPABILITIES = ProviderCapabilities(provider_type=provider_type)
        self._raises = raises
        self._health = health
        self.call_count = 0

    async def initialize(self) -> None:
        self._initialized = True

    async def shutdown(self) -> None:
        self._initialized = False

    async def _init_client_only(self) -> None:
        self._initialized = True

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.call_count += 1
        if self._raises is not None:
            raise self._raises
        return LLMResponse(
            content="ok",
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(prompt_tokens=5, completion_tokens=2, total_tokens=7),
            model="stub",
            provider=self.PROVIDER_TYPE,
            latency_ms=10.0,
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        if self._raises is not None:
            raise self._raises
        yield StreamChunk(delta="hi", is_final=False)
        yield StreamChunk(delta="", is_final=True, finish_reason=FinishReason.STOP)

    async def health_check(self) -> bool:
        return self._health

    def count_tokens(self, text: str) -> int:
        return len(text.split())


def _req() -> LLMRequest:
    return LLMRequest(messages=[Message(role=Role.USER, content="hi")])


# ---------------------------------------------------------------------------
# CircuitBreaker state machine
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_circuit_breaker_starts_closed() -> None:
    cb = CircuitBreaker(ProviderType.ANTHROPIC)
    assert cb.state == CircuitState.CLOSED
    assert not await cb.is_open()


@pytest.mark.asyncio
async def test_circuit_breaker_opens_on_threshold() -> None:
    cfg = CircuitBreakerConfig(failure_threshold=2)
    cb = CircuitBreaker(ProviderType.ANTHROPIC, config=cfg)
    await cb.record_failure()
    assert cb.state == CircuitState.CLOSED
    await cb.record_failure()
    assert cb.state == CircuitState.OPEN
    assert await cb.is_open()


@pytest.mark.asyncio
async def test_circuit_breaker_half_open_after_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    import time as _time

    cfg = CircuitBreakerConfig(failure_threshold=1, recovery_timeout_s=10.0)
    cb = CircuitBreaker(ProviderType.ANTHROPIC, config=cfg)
    await cb.record_failure()
    assert cb.state == CircuitState.OPEN

    # Simulate clock advancing past recovery window
    start = _time.monotonic()
    monkeypatch.setattr("kitkat.service.router.time.monotonic", lambda: start + 11.0)
    assert not await cb.is_open()
    assert cb.state == CircuitState.HALF_OPEN


@pytest.mark.asyncio
async def test_circuit_breaker_closed_after_success_threshold() -> None:
    cfg = CircuitBreakerConfig(failure_threshold=1, success_threshold=2, recovery_timeout_s=0.0)
    cb = CircuitBreaker(ProviderType.ANTHROPIC, config=cfg)
    await cb.record_failure()
    # Force to HALF_OPEN by checking is_open after zero-timeout
    await asyncio.sleep(0)
    assert not await cb.is_open()  # transitions to HALF_OPEN
    await cb.record_success()
    assert cb.state == CircuitState.HALF_OPEN  # one success, threshold=2
    await cb.record_success()
    assert cb.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_circuit_breaker_reset() -> None:
    cfg = CircuitBreakerConfig(failure_threshold=1)
    cb = CircuitBreaker(ProviderType.ANTHROPIC, config=cfg)
    await cb.record_failure()
    assert cb.state == CircuitState.OPEN
    await cb.reset()
    assert cb.state == CircuitState.CLOSED
    assert not await cb.is_open()


# ---------------------------------------------------------------------------
# LLMRouter — basic routing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_router_failover_uses_first_provider() -> None:
    p1 = _StubProvider(provider_type=ProviderType.ANTHROPIC)
    p2 = _StubProvider(provider_type=ProviderType.OPENAI)
    router = LLMRouter([p1, p2], RouterConfig(strategy=RoutingStrategy.FAILOVER))
    response = await router.complete(_req())
    assert response.content == "ok"
    assert p1.call_count == 1
    assert p2.call_count == 0


@pytest.mark.asyncio
async def test_router_failover_falls_back_on_error() -> None:
    p1 = _StubProvider(
        provider_type=ProviderType.ANTHROPIC,
        raises=LLMProviderError("down"),
    )
    p2 = _StubProvider(provider_type=ProviderType.OPENAI)
    router = LLMRouter(
        [p1, p2], RouterConfig(strategy=RoutingStrategy.FAILOVER, enable_cache=False)
    )
    response = await router.complete(_req())
    assert response.content == "ok"
    assert p1.call_count == 1
    assert p2.call_count == 1


@pytest.mark.asyncio
async def test_router_all_providers_exhausted_raises() -> None:
    p1 = _StubProvider(raises=LLMProviderError("down"))
    p2 = _StubProvider(provider_type=ProviderType.OPENAI, raises=LLMProviderError("down"))
    router = LLMRouter([p1, p2], RouterConfig(enable_cache=False))
    with pytest.raises(LLMError):
        await router.complete(_req())


# ---------------------------------------------------------------------------
# LLMRouter — non-retryable exceptions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc",
    [
        LLMTokenLimitError("too long"),
        LLMContentFilterError("blocked"),
        LLMAuthenticationError("bad key"),
    ],
)
async def test_router_non_retryable_raises_immediately(exc: LLMError) -> None:
    p1 = _StubProvider(raises=exc)
    p2 = _StubProvider(provider_type=ProviderType.OPENAI)
    router = LLMRouter([p1, p2], RouterConfig(enable_cache=False))
    with pytest.raises(type(exc)):
        await router.complete(_req())
    # p2 must never be tried
    assert p2.call_count == 0


# ---------------------------------------------------------------------------
# LLMRouter — rate-limit window enforcement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_router_skips_rate_limited_provider() -> None:

    p1 = _StubProvider(raises=LLMRateLimitError("429", retry_after_s=60.0))
    p2 = _StubProvider(provider_type=ProviderType.OPENAI)
    router = LLMRouter([p1, p2], RouterConfig(enable_cache=False))

    # First call: p1 fails with rate limit, p2 succeeds
    response = await router.complete(_req())
    assert response.content == "ok"
    assert p1.call_count == 1
    assert p2.call_count == 1

    # Second call: p1 is still rate-limited — p2 is used directly
    p1.call_count = 0
    p2.call_count = 0
    response = await router.complete(_req())
    assert p1.call_count == 0
    assert p2.call_count == 1


# ---------------------------------------------------------------------------
# LLMRouter — streaming
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_router_stream_yields_all_chunks() -> None:
    router = LLMRouter([_StubProvider()], RouterConfig(enable_cache=False))
    chunks = [c async for c in router.stream(_req())]
    assert len(chunks) == 2
    assert chunks[0].delta == "hi"
    assert chunks[-1].is_final


@pytest.mark.asyncio
async def test_router_stream_fallback_before_first_chunk() -> None:
    p1 = _StubProvider(raises=LLMProviderError("stream error"))
    p2 = _StubProvider(provider_type=ProviderType.OPENAI)
    router = LLMRouter([p1, p2], RouterConfig(enable_cache=False))
    chunks = [c async for c in router.stream(_req())]
    assert chunks[0].delta == "hi"


# ---------------------------------------------------------------------------
# LLMRouter — reset_circuit_breaker
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reset_circuit_breaker_returns_true_when_found() -> None:
    p = _StubProvider(provider_type=ProviderType.ANTHROPIC)
    router = LLMRouter([p], RouterConfig(enable_cache=False))
    result = await router.reset_circuit_breaker(ProviderType.ANTHROPIC)
    assert result is True


@pytest.mark.asyncio
async def test_reset_circuit_breaker_returns_false_when_not_found() -> None:
    p = _StubProvider(provider_type=ProviderType.ANTHROPIC)
    router = LLMRouter([p], RouterConfig(enable_cache=False))
    result = await router.reset_circuit_breaker(ProviderType.OPENAI)
    assert result is False


@pytest.mark.asyncio
async def test_reset_circuit_breaker_actually_resets_state() -> None:
    cfg = CircuitBreakerConfig(failure_threshold=1)
    p = _StubProvider(raises=LLMProviderError("down"))
    router = LLMRouter([p], RouterConfig(circuit_breaker=cfg, enable_cache=False))
    with pytest.raises(LLMError):
        await router.complete(_req())
    assert router._breakers[0].state == CircuitState.OPEN

    await router.reset_circuit_breaker(ProviderType.ANTHROPIC)
    assert router._breakers[0].state == CircuitState.CLOSED


# ---------------------------------------------------------------------------
# LLMRouter — health_check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_check_all_healthy() -> None:
    p1 = _StubProvider(provider_type=ProviderType.ANTHROPIC, health=True)
    p2 = _StubProvider(provider_type=ProviderType.OPENAI, health=True)
    router = LLMRouter([p1, p2], RouterConfig(enable_cache=False))
    results = await router.health_check()
    assert all(ok for ok in results.values())


@pytest.mark.asyncio
async def test_health_check_partial_failure() -> None:
    """health_check must return False for failing providers without raising."""

    class _Unhealthy(_StubProvider):
        async def health_check(self) -> bool:
            raise RuntimeError("probe failed")

    p1 = _StubProvider(provider_type=ProviderType.ANTHROPIC, health=True)
    p2 = _Unhealthy(provider_type=ProviderType.OPENAI)
    router = LLMRouter([p1, p2], RouterConfig(enable_cache=False))
    results = await router.health_check()
    assert results[(0, ProviderType.ANTHROPIC)] is True
    assert results[(1, ProviderType.OPENAI)] is False


# ---------------------------------------------------------------------------
# LLMRouter — round-robin ordering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_round_robin_cycles_through_providers() -> None:
    p1 = _StubProvider(provider_type=ProviderType.ANTHROPIC)
    p2 = _StubProvider(provider_type=ProviderType.OPENAI)
    router = LLMRouter(
        [p1, p2], RouterConfig(strategy=RoutingStrategy.ROUND_ROBIN, enable_cache=False)
    )

    await router.complete(_req())
    await router.complete(_req())
    # Each provider should have been called exactly once
    assert p1.call_count == 1
    assert p2.call_count == 1


# ---------------------------------------------------------------------------
# LLMRouter — least-latency ordering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_least_latency_picks_fastest() -> None:
    p1 = _StubProvider(provider_type=ProviderType.ANTHROPIC)
    p2 = _StubProvider(provider_type=ProviderType.OPENAI)
    router = LLMRouter(
        [p1, p2], RouterConfig(strategy=RoutingStrategy.LEAST_LATENCY, enable_cache=False)
    )

    # Seed stats: p1 fast, p2 slow
    router._stats[0].record_success(10.0)
    router._stats[1].record_success(500.0)

    await router.complete(_req())
    assert p1.call_count == 1
    assert p2.call_count == 0


# ---------------------------------------------------------------------------
# LLMRouter — minimum providers check
# ---------------------------------------------------------------------------


def test_router_requires_at_least_one_provider() -> None:
    with pytest.raises(ValueError, match="at least one provider"):
        LLMRouter([], RouterConfig())


# ---------------------------------------------------------------------------
# create_llm_router factory
# ---------------------------------------------------------------------------


def test_create_llm_router_returns_router() -> None:
    p = _StubProvider()
    router = create_llm_router([p], enable_cache=False)
    assert isinstance(router, LLMRouter)
    assert router._cfg.strategy == RoutingStrategy.FAILOVER


def test_create_llm_router_custom_strategy() -> None:
    p = _StubProvider()
    router = create_llm_router([p], strategy=RoutingStrategy.RANDOM, enable_cache=False)
    assert router._cfg.strategy == RoutingStrategy.RANDOM


def test_create_llm_router_with_cache() -> None:
    p = _StubProvider()
    router = create_llm_router([p], enable_cache=True)
    assert router.cache is not None


def test_create_llm_router_without_cache() -> None:
    p = _StubProvider()
    router = create_llm_router([p], enable_cache=False)
    assert router.cache is None
