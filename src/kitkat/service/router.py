"""Define the multi-provider LLM router.

This module provides a router for selecting LLM providers based on configured
strategies such as round-robin, failover, least-latency, and random selection.
It implements per-provider circuit breaking and tracks provider health to
guarantee robust request execution without propagating failed state.

Routing strategies
------------------
  ROUND_ROBIN    Cycle through healthy providers in order.
  FAILOVER       Always try providers in priority order; only move on error.
  LEAST_LATENCY  Pick the provider with the lowest average response latency.
  RANDOM         Uniformly random pick from healthy providers.

Circuit breaker (per provider)
-------------------------------
  CLOSED    → Normal operation. Failures increment a counter.
  OPEN      → Provider is failing. Requests are rejected immediately and
              the next provider is tried. Recovery attempted after
              `recovery_timeout_s`.
  HALF_OPEN → Recovery probe: one request is allowed through.
              Success → CLOSED. Failure → OPEN (reset timer).

The circuit breaker is asyncio-safe: state transitions are protected by
asyncio.Lock so concurrent coroutines never observe inconsistent states.

Non-retryable exceptions (LLMTokenLimitError, LLMContentFilterError,
LLMAuthenticationError) are re-raised immediately without trying fallback
providers, because a different provider would produce the same outcome.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from ..core.enums import CircuitState, RoutingStrategy
from ..core.exceptions import (
    LLMAuthenticationError,
    LLMContentFilterError,
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMTokenLimitError,
)
from .cache import CacheConfig, LLMCache

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ..abc import LLMProvider
    from ..core.enums import ProviderType
    from ..core.models import (
        LLMRequest,
        LLMResponse,
        StreamChunk,
    )

logger = logging.getLogger(__name__)


@dataclass
class CircuitBreakerConfig:
    """Tunable parameters for the per-provider circuit breaker."""

    failure_threshold: int = 5
    recovery_timeout_s: float = 60.0
    half_open_max_calls: int = 1
    success_threshold: int = 2


class CircuitBreaker:
    """Per-provider circuit breaker with asyncio-safe state transitions."""

    def __init__(
        self,
        provider_type: ProviderType,
        config: CircuitBreakerConfig | None = None,
    ) -> None:
        self._provider = provider_type
        self._cfg = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count: int = 0
        self._success_count: int = 0  # Consecutive successes in HALF_OPEN
        self._half_open_calls: int = 0  # In-flight probes in HALF_OPEN
        self._opened_at: float | None = None
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Return the current circuit state."""
        return self._state

    async def is_open(self) -> bool:
        """Return True if the circuit should block the incoming call.

        Returns:
            A boolean indicating if the circuit is open.
        """
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return False

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self._cfg.half_open_max_calls:
                    return True  # Max probes in flight; block calls
                self._half_open_calls += 1
                return False  # Allow probe

            # OPEN: check recovery timeout
            assert self._opened_at is not None
            if time.monotonic() - self._opened_at >= self._cfg.recovery_timeout_s:
                logger.info(
                    "CircuitBreaker %s: OPEN → HALF_OPEN (recovery probe)",
                    self._provider.value,
                )
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 1
                self._success_count = 0
                return False  # Allow initial HALF_OPEN probe

            return True  # Await recovery window

    async def record_success(self) -> None:
        """Record a successful provider call."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._cfg.success_threshold:
                    logger.info(
                        "CircuitBreaker %s: HALF_OPEN → CLOSED (recovered)",
                        self._provider.value,
                    )
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._half_open_calls = 0
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0  # Reset streak on success

    async def record_failure(self) -> None:
        """Record a failed provider call."""
        async with self._lock:
            self._failure_count += 1

            if self._state == CircuitState.HALF_OPEN:
                logger.warning(
                    "CircuitBreaker %s: HALF_OPEN → OPEN (probe failed)",
                    self._provider.value,
                )
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                self._half_open_calls = 0
                self._success_count = 0
                return

            if (
                self._state == CircuitState.CLOSED
                and self._failure_count >= self._cfg.failure_threshold
            ):
                logger.warning(
                    "CircuitBreaker %s: CLOSED → OPEN (%d consecutive failures)",
                    self._provider.value,
                    self._failure_count,
                )
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()

    async def reset(self) -> None:
        """Manually force the circuit back to CLOSED."""
        async with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
            self._opened_at = None


# ===========================================================================
# LLM Router
# ===========================================================================


@dataclass
class ProviderStats:
    """Running statistics for a single provider slot in the router pool."""

    provider_type: ProviderType
    model: str
    total_requests: int = 0
    failed_requests: int = 0
    _total_latency_ms: float = field(default=0.0, repr=False)
    last_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        """Return the average response latency across all completed queries in milliseconds."""
        if self.total_requests == 0:
            return 0.0
        return self._total_latency_ms / self.total_requests

    @property
    def error_rate(self) -> float:
        """Return the fraction of requests that failed."""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests

    def record_success(self, latency_ms: float) -> None:
        self.total_requests += 1
        self._total_latency_ms += latency_ms
        self.last_latency_ms = latency_ms

    def record_failure(self, latency_ms: float) -> None:
        self.total_requests += 1
        self.failed_requests += 1
        self._total_latency_ms += latency_ms
        self.last_latency_ms = latency_ms


@dataclass
class RouterConfig:
    """Top-level configuration for LLMRouter."""

    strategy: RoutingStrategy = RoutingStrategy.FAILOVER
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    enable_cache: bool = True
    cache_on_truncated: bool = False


# Non-retryable exceptions
_NON_RETRYABLE = (
    LLMTokenLimitError,
    LLMContentFilterError,
    LLMAuthenticationError,
)


class LLMRouter:
    """Multi-provider LLM router."""

    def __init__(
        self,
        providers: list[LLMProvider],
        config: RouterConfig | None = None,
    ) -> None:
        if not providers:
            raise ValueError("LLMRouter requires at least one provider.")

        self._providers = providers
        self._cfg = config or RouterConfig()

        self._breakers: dict[int, CircuitBreaker] = {
            i: CircuitBreaker(p.PROVIDER_TYPE, self._cfg.circuit_breaker)
            for i, p in enumerate(providers)
        }
        self._stats: list[ProviderStats] = [
            ProviderStats(provider_type=p.PROVIDER_TYPE, model=p.DEFAULT_MODEL) for p in providers
        ]

        self._rate_limited_until: dict[int, float] = {}

        self._rr_index: int = 0
        self._rr_lock = asyncio.Lock()

        self._cache: LLMCache | None = LLMCache(self._cfg.cache) if self._cfg.enable_cache else None

    # ------------------------------------------------------------------
    # Lifecycle methods
    # ------------------------------------------------------------------

    @classmethod
    async def build(
        cls,
        providers: list[LLMProvider],
        config: RouterConfig | None = None,
    ) -> LLMRouter:
        """Initialize all providers concurrently and return a ready router.

        Args:
            providers: Un-initialised provider instances.
            config: Router configuration. Defaults to RouterConfig().

        Returns:
            An initialized LLMRouter instance.

        Raises:
            ValueError: If all providers fail to initialize.
        """
        live: list[LLMProvider] = []
        for p in providers:
            try:
                await p.initialize()
                live.append(p)
                logger.info("LLMRouter: initialised %r", p)
            except Exception as exc:
                logger.error(
                    "LLMRouter: skipping %s — init failed: %s",
                    p.__class__.__name__,
                    exc,
                )

        if not live:
            raise ValueError("All providers failed to initialise. Cannot build LLMRouter.")

        return cls(live, config)

    async def shutdown(self) -> None:
        """Shut down all providers and flush the cache."""
        for p in self._providers:
            try:
                await p.shutdown()
            except Exception as exc:
                logger.warning("LLMRouter: error shutting down %r: %s", p, exc)
        if self._cache is not None:
            await self._cache.close()
        logger.info("LLMRouter shut down.")

    async def __aenter__(self) -> LLMRouter:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.shutdown()

    # ------------------------------------------------------------------
    # Completion routing
    # ------------------------------------------------------------------

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Route a blocking completion through the provider pool.

        Args:
            request: The LLM request to complete.

        Returns:
            The successful LLMResponse.

        Raises:
            LLMTokenLimitError: If prompt is too large for any provider.
            LLMContentFilterError: If content is blocked.
            LLMAuthenticationError: If API credentials fail.
            LLMError: If all providers are exhausted.
        """
        use_cache = self._cache is not None and not request.stream

        if use_cache:
            cached = await self._cache.get(request)
            if cached is not None:
                logger.info("LLMRouter cache HIT")
                return cached

        response, _ = await self._route_complete(request)

        if use_cache:
            should_cache = self._cfg.cache_on_truncated or not response.was_truncated
            if should_cache:
                await self._cache.set(request, response)

        return response

    async def _route_complete(self, request: LLMRequest) -> tuple[LLMResponse, int]:
        """Attempt the request on each candidate provider in strategy order.

        Args:
            request: The LLM request.

        Returns:
            A tuple of the raw LLMResponse and the winning provider's index.
        """
        order = await self._provider_order()
        last_exc: LLMError | None = None

        for idx in order:
            provider = self._providers[idx]
            breaker = self._breakers[idx]

            if await breaker.is_open():
                logger.debug(
                    "LLMRouter: skipping %s — circuit OPEN",
                    provider.PROVIDER_TYPE.value,
                )
                continue

            until = self._rate_limited_until.get(idx)
            if until is not None:
                now = time.monotonic()
                if now < until:
                    logger.debug(
                        "LLMRouter: skipping %s — rate-limited for %.1fs",
                        provider.PROVIDER_TYPE.value,
                        until - now,
                    )
                    continue

            start = time.monotonic()
            try:
                response = await provider.complete(request)
                latency = (time.monotonic() - start) * 1_000

                self._stats[idx].record_success(latency)
                await breaker.record_success()
                logger.info(
                    "LLMRouter: complete via %s | %.0fms",
                    provider.PROVIDER_TYPE.value,
                    latency,
                )
                return response, idx

            except _NON_RETRYABLE as exc:
                latency = (time.monotonic() - start) * 1_000
                self._stats[idx].record_failure(latency)
                await breaker.record_failure()
                logger.warning(
                    "LLMRouter: non-retryable error from %s: %s",
                    provider.PROVIDER_TYPE.value,
                    exc,
                )
                raise

            except LLMRateLimitError as exc:
                latency = (time.monotonic() - start) * 1_000
                self._stats[idx].record_failure(latency)
                await breaker.record_failure()
                last_exc = exc

                if exc.retry_after_s:
                    self._rate_limited_until[idx] = time.monotonic() + exc.retry_after_s
                    logger.warning(
                        "LLMRouter: %s rate-limited for %.1fs (Retry-After)",
                        provider.PROVIDER_TYPE.value,
                        exc.retry_after_s,
                    )
                else:
                    logger.warning(
                        "LLMRouter: %s rate-limited — falling back",
                        provider.PROVIDER_TYPE.value,
                    )

            except (LLMProviderError, LLMTimeoutError) as exc:
                latency = (time.monotonic() - start) * 1_000
                self._stats[idx].record_failure(latency)
                await breaker.record_failure()
                last_exc = exc
                logger.warning(
                    "LLMRouter: %s failed (%s) — falling back",
                    provider.PROVIDER_TYPE.value,
                    type(exc).__name__,
                )

        raise (
            last_exc
            if last_exc is not None
            else LLMProviderError(
                "All providers exhausted without a successful response.",
            )
        )

    # ------------------------------------------------------------------
    # Stream routing
    # ------------------------------------------------------------------

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        """Route a streaming request through the provider pool.

        Args:
            request: The streaming LLM request.

        Returns:
            An async iterator yielding successive stream chunks.

        Raises:
            LLMError: If all providers fail.
        """
        order = await self._provider_order()
        last_exc: LLMError | None = None

        for idx in order:
            provider = self._providers[idx]
            breaker = self._breakers[idx]

            if await breaker.is_open():
                continue

            until = self._rate_limited_until.get(idx)
            if until is not None:
                now = time.monotonic()
                if now < until:
                    continue

            start = time.monotonic()
            first_chunk_yielded = False

            try:
                async for chunk in provider.stream(request):
                    first_chunk_yielded = True
                    yield chunk

                latency = (time.monotonic() - start) * 1_000
                self._stats[idx].record_success(latency)
                await breaker.record_success()
                logger.info(
                    "LLMRouter: stream via %s | %.0fms",
                    provider.PROVIDER_TYPE.value,
                    latency,
                )
                return

            except _NON_RETRYABLE:
                latency = (time.monotonic() - start) * 1_000
                self._stats[idx].record_failure(latency)
                await breaker.record_failure()
                raise

            except (LLMRateLimitError, LLMProviderError, LLMTimeoutError) as exc:
                latency = (time.monotonic() - start) * 1_000
                self._stats[idx].record_failure(latency)
                await breaker.record_failure()
                last_exc = exc

                if isinstance(exc, LLMRateLimitError) and exc.retry_after_s:
                    self._rate_limited_until[idx] = time.monotonic() + exc.retry_after_s

                if first_chunk_yielded:
                    logger.error(
                        "LLMRouter: %s stream failed mid-response (after first chunk). "
                        "Cannot fall back.",
                        provider.PROVIDER_TYPE.value,
                    )
                    raise

                logger.warning(
                    "LLMRouter: %s stream failed before first token — falling back",
                    provider.PROVIDER_TYPE.value,
                )

        raise (
            last_exc
            if last_exc is not None
            else LLMProviderError(
                "All providers exhausted for streaming request.",
            )
        )

    # ------------------------------------------------------------------
    # Provider ordering strategies
    # ------------------------------------------------------------------

    async def _provider_order(self) -> list[int]:
        """Return pool indices in the sorted execution routing order."""
        n = len(self._providers)
        if n == 1:
            return [0]

        if self._cfg.strategy == RoutingStrategy.FAILOVER:
            return list(range(n))

        if self._cfg.strategy == RoutingStrategy.ROUND_ROBIN:
            async with self._rr_lock:
                start = self._rr_index
                self._rr_index = (self._rr_index + 1) % n
            return [(start + i) % n for i in range(n)]

        if self._cfg.strategy == RoutingStrategy.LEAST_LATENCY:

            def latency_key(i: int) -> float:
                avg = self._stats[i].avg_latency_ms
                return avg if avg > 0 else -1.0

            return sorted(range(n), key=latency_key)

        if self._cfg.strategy == RoutingStrategy.RANDOM:
            indices = list(range(n))
            random.shuffle(indices)
            return indices

        return list(range(n))

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    async def health_check(self) -> dict[tuple[int, ProviderType], bool]:
        """Probe all providers concurrently and return their health status.

        Returns:
            A dictionary keyed by pool index and provider type indicating health.
        """
        results = await asyncio.gather(
            *[p.health_check() for p in self._providers],
            return_exceptions=True,
        )
        return {
            (i, p.PROVIDER_TYPE): (r is True)  # Handle exceptions from gather
            for i, (p, r) in enumerate(zip(self._providers, results))
        }

    async def status(self) -> dict[str, Any]:
        """Return a full status snapshot for the admin status endpoint.

        Returns:
            A dictionary containing comprehensive router status.
        """
        health = await self.health_check()
        providers_status = []

        now_mono = time.monotonic()
        now_wall = time.time()

        for idx, provider in enumerate(self._providers):
            breaker = self._breakers[idx]
            stats = self._stats[idx]
            rate_until = self._rate_limited_until.get(idx)

            rate_until_dt: datetime | None = None
            if rate_until is not None and now_mono < rate_until:
                wall_offset = rate_until - now_mono
                rate_until_dt = datetime.fromtimestamp(now_wall + wall_offset, tz=datetime.UTC)

            providers_status.append(
                {
                    "provider": provider.PROVIDER_TYPE,
                    "model": provider.DEFAULT_MODEL,
                    "healthy": health.get((idx, provider.PROVIDER_TYPE), False),
                    "circuit_state": breaker.state.value,
                    "total_requests": stats.total_requests,
                    "failed_requests": stats.failed_requests,
                    "avg_latency_ms": stats.avg_latency_ms,
                    "error_rate": stats.error_rate,
                    "rate_limited_until": rate_until_dt,
                }
            )

        return {
            "strategy": self._cfg.strategy.value,
            "provider_count": len(self._providers),
            "healthy_count": sum(1 for (_, _pt), ok in health.items() if ok),
            "providers": providers_status,
            "cache_enabled": self._cache is not None,
        }

    async def reset_circuit_breaker(self, provider_type: ProviderType) -> bool:
        """Manually reset a provider's circuit breaker to CLOSED.

        Args:
            provider_type: The :class:`~kitkat.core.enums.ProviderType` to reset.

        Returns:
            ``True`` if the provider was found and reset; ``False`` if
            no provider with that type is present in the pool.
        """
        for idx, provider in enumerate(self._providers):
            if provider_type == provider.PROVIDER_TYPE:
                await self._breakers[idx].reset()
                self._rate_limited_until.pop(idx, None)
                logger.info(
                    "LLMRouter: manually reset circuit breaker for %s",
                    provider_type.value,
                )
                return True

        logger.warning(
            "LLMRouter.reset_circuit_breaker: provider %s not found in pool.",
            provider_type.value,
        )
        return False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def providers(self) -> list[LLMProvider]:
        """Return a read-only snapshot of the provider pool."""
        return list(self._providers)

    @property
    def cache(self) -> LLMCache | None:
        """Return direct access to the LLMCache instance."""
        return self._cache
