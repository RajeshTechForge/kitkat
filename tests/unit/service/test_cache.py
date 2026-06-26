"""Unit tests for the LLM response cache system (service/cache.py).

Covers:
- CacheBackendType enum membership
- CacheConfig defaults and field types
- make_cache_key determinism and sensitivity to parameter changes
- InMemoryCache — LRU eviction, TTL expiry, hit/miss, clear_pattern
- LLMCache — skip caching on CONTENT_FILTER / ERROR finish reasons
- LLMCache — fail-safe: backend exceptions never propagate to callers
- LLMCache stats and hit_rate
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

import pytest

from kitkat.core.enums import CacheBackendType, FinishReason, ProviderType, Role
from kitkat.core.models import LLMRequest, LLMResponse, Message, TokenUsage
from kitkat.service.cache import (
    CacheConfig,
    InMemoryCache,
    LLMCache,
    _CacheEntry,
    make_cache_key,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(model: str = "claude-3", temperature: float = 0.1) -> LLMRequest:
    return LLMRequest(
        messages=[Message(role=Role.USER, content="hello")],
        model=model,
        temperature=temperature,
    )


def _make_response(
    finish_reason: FinishReason = FinishReason.STOP,
    provider: ProviderType = ProviderType.ANTHROPIC,
) -> LLMResponse:
    return LLMResponse(
        content="stub",
        finish_reason=finish_reason,
        usage=TokenUsage(prompt_tokens=5, completion_tokens=2, total_tokens=7),
        model="claude-3",
        provider=provider,
        latency_ms=42.0,
    )


# ---------------------------------------------------------------------------
# CacheBackendType enum
# ---------------------------------------------------------------------------


class TestCacheBackendType:
    def test_memory_value(self) -> None:
        assert CacheBackendType.MEMORY == "memory"

    def test_redis_value(self) -> None:
        assert CacheBackendType.REDIS == "redis"

    def test_is_str_subclass(self) -> None:
        assert isinstance(CacheBackendType.MEMORY, str)


# ---------------------------------------------------------------------------
# CacheConfig defaults
# ---------------------------------------------------------------------------


class TestCacheConfig:
    def test_default_backend_is_memory(self) -> None:
        cfg = CacheConfig()
        assert cfg.backend == CacheBackendType.MEMORY

    def test_default_key_prefix(self) -> None:
        cfg = CacheConfig()
        assert cfg.key_prefix == "kitkat:llm:"

    def test_custom_key_prefix(self) -> None:
        cfg = CacheConfig(key_prefix="myapp:llm:")
        assert cfg.key_prefix == "myapp:llm:"

    def test_default_ttl(self) -> None:
        assert CacheConfig().ttl_seconds == 3_600

    def test_default_max_memory_size(self) -> None:
        assert CacheConfig().max_memory_size == 1_000


# ---------------------------------------------------------------------------
# make_cache_key
# ---------------------------------------------------------------------------


class TestMakeCacheKey:
    def test_deterministic(self) -> None:
        req = _make_request()
        assert make_cache_key(req) == make_cache_key(req)

    def test_different_model_produces_different_key(self) -> None:
        r1 = _make_request(model="claude-3")
        r2 = _make_request(model="gpt-4o")
        assert make_cache_key(r1) != make_cache_key(r2)

    def test_different_temperature_produces_different_key(self) -> None:
        r1 = _make_request(temperature=0.0)
        r2 = _make_request(temperature=1.0)
        assert make_cache_key(r1) != make_cache_key(r2)

    def test_stop_sequences_order_invariant(self) -> None:
        """Sorted stop_sequences — cache key must not depend on list order."""
        r1 = LLMRequest(
            messages=[Message(role=Role.USER, content="hi")],
            stop_sequences=["a", "b"],
        )
        r2 = LLMRequest(
            messages=[Message(role=Role.USER, content="hi")],
            stop_sequences=["b", "a"],
        )
        assert make_cache_key(r1) == make_cache_key(r2)

    def test_returns_64_char_hex(self) -> None:
        key = make_cache_key(_make_request())
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)


# ---------------------------------------------------------------------------
# _CacheEntry round-trip
# ---------------------------------------------------------------------------


class TestCacheEntry:
    def test_from_response_round_trip(self) -> None:
        resp = _make_response()
        entry = _CacheEntry.from_response(resp)
        reconstructed = entry.to_response()
        assert reconstructed.content == resp.content
        assert reconstructed.finish_reason == resp.finish_reason
        assert reconstructed.usage.total_tokens == resp.usage.total_tokens
        assert reconstructed.model == resp.model
        assert reconstructed.provider == resp.provider

    def test_to_response_does_not_set_raw_response(self) -> None:
        """raw_response must be absent (default None) — not force-assigned."""
        entry = _CacheEntry.from_response(_make_response())
        resp = entry.to_response()
        assert resp.raw_response is None

    def test_to_dict_from_dict_round_trip(self) -> None:
        entry = _CacheEntry.from_response(_make_response())
        assert _CacheEntry.from_dict(entry.to_dict()).content == entry.content


# ---------------------------------------------------------------------------
# InMemoryCache
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_in_memory_cache_hit_miss() -> None:
    cache = InMemoryCache(max_size=10)
    entry = _CacheEntry.from_response(_make_response())
    await cache.set("key1", entry, ttl_seconds=60)
    assert await cache.get("key1") is not None
    assert await cache.get("missing") is None


@pytest.mark.asyncio
async def test_in_memory_cache_ttl_expiry() -> None:
    cache = InMemoryCache(max_size=10)
    entry = _CacheEntry.from_response(_make_response())
    await cache.set("key1", entry, ttl_seconds=0)  # expires immediately
    # Advance past expiry using monkeypatch-style: sleep is not needed,
    # just set cached_at far in the past by directly patching time.monotonic
    with patch("kitkat.service.cache.time") as mock_time:
        mock_time.monotonic.return_value = time.monotonic() + 100
        result = await cache.get("key1")
    assert result is None


@pytest.mark.asyncio
async def test_in_memory_cache_lru_eviction() -> None:
    cache = InMemoryCache(max_size=2)
    entry = _CacheEntry.from_response(_make_response())
    await cache.set("a", entry, ttl_seconds=3600)
    await cache.set("b", entry, ttl_seconds=3600)
    await cache.set("c", entry, ttl_seconds=3600)  # evicts "a"
    assert await cache.get("a") is None
    assert await cache.get("b") is not None
    assert await cache.get("c") is not None


@pytest.mark.asyncio
async def test_in_memory_cache_size() -> None:
    cache = InMemoryCache(max_size=10)
    entry = _CacheEntry.from_response(_make_response())
    assert await cache.size() == 0
    await cache.set("k", entry, ttl_seconds=3600)
    assert await cache.size() == 1


@pytest.mark.asyncio
async def test_in_memory_cache_delete() -> None:
    cache = InMemoryCache(max_size=10)
    entry = _CacheEntry.from_response(_make_response())
    await cache.set("k", entry, ttl_seconds=3600)
    await cache.delete("k")
    assert await cache.get("k") is None


@pytest.mark.asyncio
async def test_in_memory_cache_clear_pattern() -> None:
    cache = InMemoryCache(max_size=10)
    entry = _CacheEntry.from_response(_make_response())
    await cache.set("abc123", entry, ttl_seconds=3600)
    await cache.set("abc456", entry, ttl_seconds=3600)
    await cache.set("xyz789", entry, ttl_seconds=3600)
    deleted = await cache.clear_pattern("abc*")
    assert deleted == 2
    assert await cache.get("abc123") is None
    assert await cache.get("xyz789") is not None


@pytest.mark.asyncio
async def test_in_memory_cache_max_size_one() -> None:
    cache = InMemoryCache(max_size=1)
    entry = _CacheEntry.from_response(_make_response())
    await cache.set("a", entry, ttl_seconds=3600)
    await cache.set("b", entry, ttl_seconds=3600)
    assert await cache.size() == 1


def test_in_memory_cache_invalid_size_raises() -> None:
    with pytest.raises(ValueError, match="max_size"):
        InMemoryCache(max_size=0)


# ---------------------------------------------------------------------------
# LLMCache — skip caching for non-cacheable finish reasons
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_cache_skips_content_filter() -> None:
    cfg = CacheConfig(backend=CacheBackendType.MEMORY)
    cache = LLMCache(cfg)
    req = _make_request()
    resp = _make_response(finish_reason=FinishReason.CONTENT_FILTER)
    await cache.set(req, resp)
    assert await cache.get(req) is None


@pytest.mark.asyncio
async def test_llm_cache_skips_error_finish_reason() -> None:
    cfg = CacheConfig(backend=CacheBackendType.MEMORY)
    cache = LLMCache(cfg)
    req = _make_request()
    resp = _make_response(finish_reason=FinishReason.ERROR)
    await cache.set(req, resp)
    assert await cache.get(req) is None


@pytest.mark.asyncio
async def test_llm_cache_stores_stop_finish_reason() -> None:
    cfg = CacheConfig(backend=CacheBackendType.MEMORY)
    cache = LLMCache(cfg)
    req = _make_request()
    resp = _make_response(finish_reason=FinishReason.STOP)
    await cache.set(req, resp)
    cached = await cache.get(req)
    assert cached is not None
    assert cached.content == resp.content


# ---------------------------------------------------------------------------
# LLMCache — fail-safe: backend errors never propagate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_cache_get_fail_safe() -> None:
    """Backend get() exception must be swallowed and treated as a miss."""
    cfg = CacheConfig(backend=CacheBackendType.MEMORY)
    cache = LLMCache(cfg)
    cache._backend.get = AsyncMock(side_effect=RuntimeError("redis down"))
    result = await cache.get(_make_request())
    assert result is None


@pytest.mark.asyncio
async def test_llm_cache_set_fail_safe() -> None:
    """Backend set() exception must not propagate to callers."""
    cfg = CacheConfig(backend=CacheBackendType.MEMORY)
    cache = LLMCache(cfg)
    cache._backend.set = AsyncMock(side_effect=RuntimeError("disk full"))
    await cache.set(_make_request(), _make_response())  # Must not raise


# ---------------------------------------------------------------------------
# LLMCache hit_rate and stats
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_cache_hit_rate_zero_initially() -> None:
    cache = LLMCache()
    assert cache.hit_rate == 0.0


@pytest.mark.asyncio
async def test_llm_cache_hit_rate_after_hit() -> None:
    cache = LLMCache()
    req = _make_request()
    await cache.set(req, _make_response())
    await cache.get(req)  # HIT
    await cache.get(_make_request(model="other"))  # MISS
    assert cache.hit_rate == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_llm_cache_stats_backend_value_is_string() -> None:
    cache = LLMCache(CacheConfig(backend=CacheBackendType.MEMORY))
    stats = await cache.stats()
    assert stats["backend"] == "memory"


# ---------------------------------------------------------------------------
# CacheConfig — unknown backend raises
# ---------------------------------------------------------------------------


def test_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="Unknown cache backend"):
        LLMCache(CacheConfig(backend="turbo"))  # type: ignore[arg-type]
