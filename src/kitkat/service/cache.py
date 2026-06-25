"""Define the async LLM response cache system.

This module provides an orchestration layer to cache non-streaming LLM responses
using deterministic hashing. It includes both in-memory and Redis backend implementations.
All interactions with backends are fail-safe to prevent caching issues from breaking inference.

Cache key
---------
SHA-256 of the JSON-serialised tuple:
  (messages, model, max_tokens, temperature, top_p, stop_sequences_sorted)

Deliberately excluded from the key:
  - metadata    (arbitrary bag — no semantic impact on the response)
  - timeout     (infrastructure concern, not content)

Serialisation
-------------
LLMResponse is stored as a JSON dict (excluding `raw_response` which is not
serialisable). Deserialisation reconstructs the full domain object.

Both backends are async-safe:
  - InMemoryCache uses asyncio.Lock (not threading.Lock — single event loop)
  - RedisCache uses redis.asyncio connection pool
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from ..core.enums import CacheBackendType, FinishReason, ProviderType
from ..core.models import (
    LLMRequest,
    LLMResponse,
    TokenUsage,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Core data models
# ----------------------------------------------------------------------------


@dataclass
class _CacheEntry:
    """Serialisable snapshot of an LLMResponse."""

    content: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    provider: str
    latency_ms: float
    cached_at: float

    def to_response(self) -> LLMResponse:
        """Reconstruct a full LLMResponse domain object from this entry."""
        return LLMResponse(
            content=self.content,
            finish_reason=FinishReason(self.finish_reason),
            usage=TokenUsage(
                prompt_tokens=self.prompt_tokens,
                completion_tokens=self.completion_tokens,
                total_tokens=self.total_tokens,
            ),
            model=self.model,
            provider=ProviderType(self.provider),
            latency_ms=self.latency_ms,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary suitable for JSON serialization."""
        return {
            "content": self.content,
            "finish_reason": self.finish_reason,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "model": self.model,
            "provider": self.provider,
            "latency_ms": self.latency_ms,
            "cached_at": self.cached_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> _CacheEntry:
        """Deserialize from a dictionary representation.

        Args:
            d: The dictionary to construct the entry from.

        Returns:
            A new _CacheEntry instance.
        """
        return cls(
            content=d["content"],
            finish_reason=d["finish_reason"],
            prompt_tokens=int(d["prompt_tokens"]),
            completion_tokens=int(d["completion_tokens"]),
            total_tokens=int(d["total_tokens"]),
            model=d["model"],
            provider=d["provider"],
            latency_ms=float(d["latency_ms"]),
            cached_at=float(d.get("cached_at", time.monotonic())),
        )

    @classmethod
    def from_response(cls, resp: LLMResponse) -> _CacheEntry:
        """Build a _CacheEntry from a fresh LLMResponse.

        Args:
            resp: The original LLMResponse.

        Returns:
            A serializable cache entry based on the response.
        """
        return cls(
            content=resp.content,
            finish_reason=resp.finish_reason.value,
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
            total_tokens=resp.usage.total_tokens,
            model=resp.model,
            provider=resp.provider.value,
            latency_ms=resp.latency_ms,
            cached_at=time.monotonic(),
        )


# ===========================================================================
# Abstract base
# ===========================================================================


class CacheBackend(ABC):
    """Abstract async cache backend."""

    @abstractmethod
    async def get(self, key: str) -> _CacheEntry | None:
        """Return the cached entry for *key*, or "None" on miss / expired."""

    @abstractmethod
    async def set(self, key: str, entry: _CacheEntry, ttl_seconds: int) -> None:
        """Store *entry* under *key* with the given TTL (seconds)."""

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Remove a single key.  No-op if absent."""

    @abstractmethod
    async def clear_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching *pattern* (glob syntax).

        Returns the number of keys deleted (best-effort for Redis).
        """

    @abstractmethod
    async def size(self) -> int:
        """Return the current number of entries held by this backend."""

    @abstractmethod
    async def close(self) -> None:
        """Release any underlying connections / file handles / resources."""


# ===========================================================================
# In-memory backend (LRU with lazy TTL eviction)
# ===========================================================================


class InMemoryCache(CacheBackend):
    """Asyncio-safe LRU cache backed by OrderedDict."""

    def __init__(self, max_size: int = 1_000) -> None:
        """Initialize the in-memory cache."""
        if max_size < 1:
            raise ValueError(f"InMemoryCache.max_size must be ≥ 1, got {max_size}")
        self._max_size = max_size
        self._store: OrderedDict[str, tuple[_CacheEntry, float]] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> _CacheEntry | None:
        """Return the cached entry for *key*, or None on miss / expired."""
        async with self._lock:
            if key not in self._store:
                return None
            entry, expire_at = self._store[key]
            if time.monotonic() > expire_at:
                del self._store[key]
                return None

            self._store.move_to_end(key)
            return entry

    async def set(self, key: str, entry: _CacheEntry, ttl_seconds: int) -> None:
        """Store *entry* under *key* with the given TTL (seconds)."""
        async with self._lock:
            expire_at = time.monotonic() + ttl_seconds
            if key in self._store:
                del self._store[key]

            elif len(self._store) >= self._max_size:
                # Evict least-recently-used entry
                evicted_key, _ = self._store.popitem(last=False)
                logger.debug("InMemoryCache LRU eviction: key=%s", evicted_key)

            self._store[key] = (entry, expire_at)

    async def delete(self, key: str) -> None:
        """Remove a single key. No-op if absent."""
        async with self._lock:
            self._store.pop(key, None)

    async def clear_pattern(self, pattern: str) -> int:
        """Delete all keys matching *pattern* (glob syntax). Returns the number of keys deleted."""
        import fnmatch

        async with self._lock:
            matching = [k for k in self._store if fnmatch.fnmatch(k, pattern)]
            for k in matching:
                del self._store[k]
        return len(matching)

    async def size(self) -> int:
        """Return the current number of entries held by this backend."""
        async with self._lock:
            return len(self._store)

    async def close(self) -> None:
        """No-op: in-memory cache needs no teardown."""

    async def clear_all(self) -> None:
        """Flush all entries.  Used in tests and admin flush operations."""
        async with self._lock:
            self._store.clear()


# ===========================================================================
# Redis backend
# ===========================================================================


class RedisCache(CacheBackend):
    """Async Redis cache backend using redis.asyncio.

    Requires the ``redis`` extra::

        pip install kitkat[redis]
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        key_prefix: str = "kitkat:llm:",
    ) -> None:
        """Create a Redis-backed cache.

        Args:
            redis_url: Full Redis connection URL.  Supports ``redis://``,
                ``rediss://`` (TLS), and ``redis+sentinel://`` schemes.
                Example: ``"redis://:password@redis-host:6379/0"``
            key_prefix: Namespace prefix applied to every key stored in Redis.
                Override to isolate multiple apps sharing one Redis instance.
        """
        try:
            import redis.asyncio as aioredis
        except ImportError as exc:
            raise ImportError(
                "RedisCache requires the 'redis' extra. Install it with: pip install kitkat[redis]"
            ) from exc

        self._url = redis_url
        self._key_prefix = key_prefix
        self._pool = aioredis.ConnectionPool.from_url(
            redis_url,
            max_connections=20,
            decode_responses=True,  # return str, not bytes
        )
        self._client = aioredis.Redis(connection_pool=self._pool)

    def _full_key(self, key: str) -> str:
        """Return the namespaced Redis key for the given hash."""
        return f"{self._key_prefix}{key}"

    async def get(self, key: str) -> _CacheEntry | None:
        try:
            raw = await self._client.get(self._full_key(key))
            if raw is None:
                return None
            return _CacheEntry.from_dict(json.loads(raw))
        except Exception as exc:
            logger.warning("RedisCache.get failed for key %s: %s", key, exc)
            return None

    async def set(self, key: str, entry: _CacheEntry, ttl_seconds: int) -> None:
        try:
            payload = json.dumps(entry.to_dict(), ensure_ascii=False)
            await self._client.setex(
                name=self._full_key(key),
                time=ttl_seconds,
                value=payload,
            )
        except Exception as exc:
            logger.warning("RedisCache.set failed for key %s: %s", key, exc)

    async def delete(self, key: str) -> None:
        try:
            await self._client.delete(self._full_key(key))
        except Exception as exc:
            logger.warning("RedisCache.delete failed for key %s: %s", key, exc)

    async def clear_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching *pattern* using SCAN + batched DELETE.

        Uses "SCAN" (non-blocking, cursor-based) instead of "KEYS"
        to avoid stalling Redis while iterating large key spaces.  Keys are
        accumulated into batches of 100 and deleted with a single "DELETE"
        command per batch, reducing network round-trips from O(n) to O(n/100).
        """
        full_pattern = self._full_key(pattern)
        deleted = 0
        try:
            batch: list[str] = []
            async for key in self._client.scan_iter(match=full_pattern, count=100):
                batch.append(key)
                if len(batch) >= 100:
                    await self._client.delete(*batch)
                    deleted += len(batch)
                    batch = []
            # Flush remaining keys in the final incomplete batch
            if batch:
                await self._client.delete(*batch)
                deleted += len(batch)
        except Exception as exc:
            logger.warning("RedisCache.clear_pattern failed: %s", exc)
        return deleted

    async def size(self) -> int:
        """
        Approximate entry count — scans only our key prefix via SCAN.

        Uses "SCAN" (non-blocking) rather than "KEYS" to avoid stalling
        Redis for large data sets.
        """
        count = 0
        try:
            async for _ in self._client.scan_iter(match=f"{self._key_prefix}*", count=100):
                count += 1
        except Exception as exc:
            logger.warning("RedisCache.size failed: %s", exc)
        return count

    async def ping(self) -> bool:
        """Liveness check — returns "False" on any error."""
        try:
            return await self._client.ping()
        except Exception:
            return False

    async def close(self) -> None:
        """Close the Redis client and drain the connection pool."""
        try:
            await self._client.aclose()
            await self._pool.aclose()
        except Exception as exc:
            logger.warning("RedisCache.close error: %s", exc)


# ===========================================================================
# LLMCache — orchestrator
# ===========================================================================


def make_cache_key(request: LLMRequest) -> str:
    """Produce a deterministic, provider-agnostic SHA-256 cache key.

    Args:
        request: The context and parameters of the request.

    Returns:
        A 64-character lowercase hex string representing the SHA-256 digest.
    """
    key_data: dict[str, Any] = {
        "messages": [{"role": m.role.value, "content": m.content} for m in request.messages],
        "model": request.model,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "stop_sequences": sorted(request.stop_sequences),
    }
    serialised = json.dumps(key_data, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


@dataclass
class CacheConfig:
    """Configuration for the LLMCache orchestrator.

    Attributes:
        backend: Storage backend to use.  Defaults to in-process LRU memory.
        redis_url: Redis connection URL used when ``backend`` is
            :attr:`~kitkat.core.enums.CacheBackendType.REDIS`.
        ttl_seconds: Default entry lifetime in seconds.  Can be overridden
            per ``set()`` call.
        max_memory_size: Maximum entries held by :class:`InMemoryCache`
            before LRU eviction begins.  Ignored for the Redis backend.
        key_prefix: Namespace prefix applied to every Redis key.
            Override to isolate multiple apps sharing one Redis instance.
    """

    backend: CacheBackendType = CacheBackendType.MEMORY
    redis_url: str = "redis://localhost:6379/0"
    ttl_seconds: int = 3_600
    max_memory_size: int = 1_000
    key_prefix: str = "kitkat:llm:"


class LLMCache:
    """LLM response cache orchestrator."""

    def __init__(self, config: CacheConfig | None = None) -> None:
        cfg = config or CacheConfig()
        self._cfg = cfg
        self._backend: CacheBackend = self._build_backend(cfg)
        self._hits: int = 0
        self._misses: int = 0

    @staticmethod
    def _build_backend(cfg: CacheConfig) -> CacheBackend:
        if cfg.backend == CacheBackendType.REDIS:
            return RedisCache(redis_url=cfg.redis_url, key_prefix=cfg.key_prefix)
        if cfg.backend == CacheBackendType.MEMORY:
            return InMemoryCache(max_size=cfg.max_memory_size)
        raise ValueError(
            f"Unknown cache backend {cfg.backend!r}. "
            f"Valid values: {[b.value for b in CacheBackendType]}."
        )

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    async def get(self, request: LLMRequest) -> LLMResponse | None:
        """Look up a cached response for the specific request.

        Args:
            request: The LLMRequest whose key should be looked up.

        Returns:
            The cached LLMResponse, or None if a miss occurs.
        """
        key = make_cache_key(request)
        try:
            entry = await self._backend.get(key)
        except Exception as exc:
            logger.warning("LLMCache.get error (treating as miss): %s", exc)
            entry = None

        if entry is not None:
            self._hits += 1
            logger.debug("Cache HIT  key=%s", key[:12])
            resp = entry.to_response()
            return resp

        self._misses += 1
        logger.debug("Cache MISS key=%s", key[:12])
        return None

    async def set(
        self,
        request: LLMRequest,
        response: LLMResponse,
        *,
        ttl_seconds: int | None = None,
    ) -> None:
        """Store the response in the cache using the given request parameters.

        Args:
            request: The originating request used to derive the cache key.
            response: The LLMResponse to store.
            ttl_seconds: Optional per-call TTL override.
        """
        if response.finish_reason in (FinishReason.CONTENT_FILTER, FinishReason.ERROR):
            logger.debug(
                "Cache SKIP — finish_reason=%s",
                response.finish_reason,
            )
            return

        key = make_cache_key(request)
        entry = _CacheEntry.from_response(response)
        ttl = ttl_seconds if ttl_seconds is not None else self._cfg.ttl_seconds

        try:
            await self._backend.set(key, entry, ttl)
            logger.debug(
                "Cache SET  key=%s ttl=%ds",
                key[:12],
                ttl,
            )
        except Exception as exc:
            logger.warning("LLMCache.set error (non-fatal): %s", exc)

    async def invalidate(self, request: LLMRequest) -> None:
        """Remove the cache entry for a specific request (force-refresh)."""
        key = make_cache_key(request)
        try:
            await self._backend.delete(key)
        except Exception as exc:
            logger.warning("LLMCache.invalidate error: %s", exc)

    async def clear(self, pattern: str = "*") -> int:
        """
        Flush all entries, or only those matching glob *pattern*.

        Returns the number of keys deleted (best-effort for Redis).
        """
        try:
            return await self._backend.clear_pattern(pattern)
        except Exception as exc:
            logger.warning("LLMCache.clear error: %s", exc)
            return 0

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    @property
    def hits(self) -> int:
        """Total cache hits since this instance was created."""
        return self._hits

    @property
    def misses(self) -> int:
        """Total cache misses since this instance was created."""
        return self._misses

    @property
    def hit_rate(self) -> float:
        """Fraction of lookups that were cache hits (0.0–1.0)."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    async def stats(self) -> dict[str, Any]:
        """Return a stats dict compatible with "CacheStatsSchema"."""
        return {
            "backend": self._cfg.backend.value,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
            "size": await self._backend.size(),
            "max_size": (
                self._cfg.max_memory_size if self._cfg.backend == CacheBackendType.MEMORY else 0
            ),
            "ttl_seconds": self._cfg.ttl_seconds,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Release backend resources.  Call during application shutdown."""
        await self._backend.close()

    async def __aenter__(self) -> LLMCache:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()
