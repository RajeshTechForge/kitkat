---
title: Service & Router
description: Reference documentation for Kitkat's LLMService, LLMRouter, BYOKLLMService, and LLMCache classes.
order: 2
---

This page documents `LLMService`, `LLMRouter`, `BYOKLLMService`, `RouterConfig`, `CacheConfig`, `LLMCache`, and `create_llm_service`.

**Import path:** `from kitkat.service import ...`

## `create_llm_service`

```python
from kitkat.service import create_llm_service
```

Factory function that builds an `LLMService` pre-loaded with the given providers.

### Signature

```python
def create_llm_service(
    providers: dict[ProviderType, LLMProvider],
) -> LLMService
```

### Parameters

| Parameter   | Type                              | Description                                                   |
| ----------- | --------------------------------- | ------------------------------------------------------------- |
| `providers` | `dict[ProviderType, LLMProvider]` | Mapping of canonical type to uninitialized provider instance. |

### Returns

`LLMService` — ready to call `await service.initialize()`.

### Example

```python
from kitkat.service import create_llm_service
from kitkat import ProviderType
from kitkat.providers.anthropic import AnthropicProvider, AnthropicConfig
import os

service = create_llm_service({
    ProviderType.ANTHROPIC: AnthropicProvider(
        AnthropicConfig(api_key=os.environ["ANTHROPIC_API_KEY"])
    )
})
await service.initialize()
```

## `LLMService`

```python
from kitkat.service import LLMService
```

Facade over all registered providers. Owns the provider lifecycle and exposes every inference operation through a single, provider-type-routed interface. Route handlers and agent adapters interact through this class; they never touch provider classes directly.

### Lifecycle Methods

| Method              | Signature                                                      | Description                                                                                                                  |
| ------------------- | -------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `register_provider` | `(provider_type: ProviderType, provider: LLMProvider) -> None` | Add a provider. Raises `ValueError` if the type is already registered.                                                       |
| `initialize`        | `async () -> None`                                             | Call `provider.initialize()` on each registered provider in insertion order. Raises `LLMProviderInitError` on first failure. |
| `shutdown`          | `async () -> None`                                             | Call `provider.shutdown()` on all providers. Errors are logged and swallowed so remaining providers still shut down.         |

### Properties

| Property         | Type                              | Description                                        |
| ---------------- | --------------------------------- | -------------------------------------------------- |
| `providers`      | `dict[ProviderType, LLMProvider]` | Read-only copy of the registered provider mapping. |
| `provider_count` | `int`                             | Number of registered providers.                    |

### Inference Methods

| Method     | Signature                                                                                | Description                                                                                |
| ---------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `complete` | `async (request: LLMRequest, provider_type: ProviderType) -> LLMResponse`                | Non-streaming completion with provider's `RetryPolicy` applied automatically.              |
| `stream`   | `async (request: LLMRequest, provider_type: ProviderType) -> AsyncIterator[StreamChunk]` | Streaming completion. Yields one `StreamChunk` per token; final chunk has `is_final=True`. |

**Raises** (both methods): `LLMProviderError` if `provider_type` is not registered; `LLMTimeoutError`, `LLMRateLimitError`, `LLMTokenLimitError` from the provider.

### Observability Methods

| Method                | Signature                                                       | Description                                                                                |
| --------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `health_check`        | `async (provider_type: ProviderType) -> bool`                   | Probe a single provider's liveness.                                                        |
| `health_check_all`    | `async () -> dict[ProviderType, bool]`                          | Probe all providers. Failures for individual providers are caught and recorded as `False`. |
| `count_tokens`        | `(provider_type: ProviderType, text: str) -> int`               | Local token estimate using the provider's tokenizer. Always ≥ 1 for non-empty text.        |
| `count_prompt_tokens` | `(provider_type: ProviderType, messages: list[Message]) -> int` | Token estimate across a conversation. Returns `0` for an empty message list.               |
| `get_capabilities`    | `(provider_type: ProviderType) -> ProviderCapabilities`         | Returns the provider's feature flags and context window size.                              |

## `LLMRouter`

```python
from kitkat.service import LLMRouter
```

Multi-provider router with per-provider circuit breakers, configurable routing strategies, and an optional LLM response cache. Use `LLMRouter` when you need automatic fallback across providers, latency-optimized selection, or response caching.

### Construction

Use the async `build()` class method — it initializes all providers concurrently and skips any that fail to start (logging the error) so a broken provider never prevents the application from starting.

```python
from kitkat.service import LLMRouter, RouterConfig
from kitkat.core.enums import RoutingStrategy

router = await LLMRouter.build(
    providers=[anthropic_provider, openai_provider],
    config=RouterConfig(strategy=RoutingStrategy.FAILOVER),
)
```

### `RouterConfig`

```python
from kitkat.service import RouterConfig
```

| Field                | Type                   | Default                  | Description                                  |
| -------------------- | ---------------------- | ------------------------ | -------------------------------------------- |
| `strategy`           | `RoutingStrategy`      | `FAILOVER`               | Provider selection algorithm                 |
| `circuit_breaker`    | `CircuitBreakerConfig` | `CircuitBreakerConfig()` | Per-provider circuit breaker parameters      |
| `cache`              | `CacheConfig`          | `CacheConfig()`          | Cache configuration                          |
| `enable_cache`       | `bool`                 | `True`                   | Enable/disable response caching              |
| `cache_on_truncated` | `bool`                 | `False`                  | Cache responses where `finish_reason=LENGTH` |

### `CircuitBreakerConfig`

```python
from kitkat.service.router import CircuitBreakerConfig
```

| Field                 | Type    | Default | Description                                             |
| --------------------- | ------- | ------- | ------------------------------------------------------- |
| `failure_threshold`   | `int`   | `5`     | Consecutive failures to trip CLOSED → OPEN              |
| `recovery_timeout_s`  | `float` | `60.0`  | Seconds OPEN before allowing a recovery probe           |
| `half_open_max_calls` | `int`   | `1`     | Maximum in-flight probes in HALF_OPEN state             |
| `success_threshold`   | `int`   | `2`     | Consecutive successes in HALF_OPEN to close the circuit |

### Lifecycle Methods

| Method                     | Signature                                     | Description                                                                                  |
| -------------------------- | --------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `build` _(classmethod)_    | `async (providers, config=None) -> LLMRouter` | Initialize providers concurrently; skip failures. Raises `ValueError` if all providers fail. |
| `shutdown`                 | `async () -> None`                            | Shut down all providers and flush the cache.                                                 |
| `__aenter__` / `__aexit__` | —                                             | Async context manager. `__aexit__` calls `shutdown()`.                                       |

### Inference Methods

| Method     | Signature                                                   | Description                                                                                                                                                                                                                                                     |
| ---------- | ----------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `complete` | `async (request: LLMRequest) -> LLMResponse`                | Route through the provider pool using the configured strategy. Returns a cached response on hit. Non-retryable exceptions (`LLMTokenLimitError`, `LLMContentFilterError`, `LLMAuthenticationError`) are re-raised immediately without trying further providers. |
| `stream`   | `async (request: LLMRequest) -> AsyncIterator[StreamChunk]` | Route a streaming request. If a provider fails **before** the first token, the next provider is tried. If it fails **mid-stream**, the error is re-raised immediately (partial output has already been sent).                                                   |

### Observability Methods

| Method                  | Signature                                          | Description                                                                                                                         |
| ----------------------- | -------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `health_check`          | `async () -> dict[tuple[int, ProviderType], bool]` | Probe all providers concurrently. Keyed by `(pool_index, ProviderType)`.                                                            |
| `status`                | `async () -> dict[str, Any]`                       | Full snapshot: strategy, provider count, per-provider circuit state, error rates, average latency, rate-limit expiry, cache status. |
| `reset_circuit_breaker` | `async (provider_type: ProviderType) -> bool`      | Manually force a provider's circuit to CLOSED and clear any rate-limit cooldown. Returns `True` if found, `False` otherwise.        |

### Properties

| Property    | Type                | Description                                                           |
| ----------- | ------------------- | --------------------------------------------------------------------- |
| `providers` | `list[LLMProvider]` | Read-only snapshot of the provider pool                               |
| `cache`     | `LLMCache \| None`  | Direct access to the cache instance; `None` when `enable_cache=False` |

## `BYOKLLMService`

```python
from kitkat.service import BYOKLLMService
```

Short-lived, per-request service for BYOK (Bring Your Own Key) mode. Each instance is bound to a single `(provider_type, api_key, model)` triple and must be used as an `async with` context manager.

Initialization calls `provider._init_client_only()` (no credential probe) — authentication errors surface on the first `complete()` or `stream()` call.

### Constructor

```python
BYOKLLMService(
    provider_type: ProviderType,
    api_key: str,
    model: str,
)
```

| Parameter       | Type           | Description                                                                                            |
| --------------- | -------------- | ------------------------------------------------------------------------------------------------------ |
| `provider_type` | `ProviderType` | Target provider. Supported: `ANTHROPIC`, `OPENAI`, `GEMINI`.                                           |
| `api_key`       | `str`          | Caller-supplied API key. Validated by the provider config; empty string raises `LLMProviderInitError`. |
| `model`         | `str`          | Model identifier. Empty string falls back to each provider's `DEFAULT_MODEL`.                          |

### Context Manager

| Phase        | Action                                                                                      |
| ------------ | ------------------------------------------------------------------------------------------- |
| `__aenter__` | Creates the provider HTTP client (`_init_client_only`). No network calls.                   |
| `__aexit__`  | Calls `provider.shutdown()` unconditionally — runs even when inference raised an exception. |

### Inference Methods

| Method     | Signature                                                   | Description                                                           |
| ---------- | ----------------------------------------------------------- | --------------------------------------------------------------------- |
| `complete` | `async (request: LLMRequest) -> LLMResponse`                | Non-streaming completion with retry policy applied.                   |
| `stream`   | `async (request: LLMRequest) -> AsyncIterator[StreamChunk]` | Streaming completion. Must be consumed inside the `async with` block. |

**Raises** (both methods): `LLMAuthenticationError`, `LLMRateLimitError`, `LLMTokenLimitError`, `LLMTimeoutError`, `LLMProviderError`.

### Example

```python
async with BYOKLLMService(
    provider_type=ProviderType.OPENAI,
    api_key=user_supplied_key,
    model="gpt-4o-mini",
) as svc:
    response = await svc.complete(request)
```

## `LLMCache`

```python
from kitkat.service.cache import LLMCache, CacheConfig
```

LLM response cache orchestrator. Wraps either `InMemoryCache` (LRU `OrderedDict`) or `RedisCache` (`redis.asyncio`).

**Cache key:** SHA-256 of `(messages, model, max_tokens, temperature, top_p, stop_sequences_sorted)`. Metadata and timeout are excluded.

**Skip conditions:** Responses with `finish_reason=CONTENT_FILTER` or `ERROR` are never cached. Responses with `finish_reason=LENGTH` (truncated) are cached only when `RouterConfig.cache_on_truncated=True`.

### `CacheConfig`

| Field             | Type               | Default                      | Description                                                   |
| ----------------- | ------------------ | ---------------------------- | ------------------------------------------------------------- |
| `backend`         | `CacheBackendType` | `MEMORY`                     | Storage backend                                               |
| `redis_url`       | `str`              | `"redis://localhost:6379/0"` | Redis connection URL. Supports `redis://`, `rediss://` (TLS). |
| `ttl_seconds`     | `int`              | `3600`                       | Default entry TTL                                             |
| `max_memory_size` | `int`              | `1000`                       | Maximum entries in `InMemoryCache` before LRU eviction        |
| `key_prefix`      | `str`              | `"kitkat:llm:"`              | Redis key namespace                                           |

### Methods

| Method       | Signature                                                                                        | Description                                                               |
| ------------ | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------- |
| `get`        | `async (request: LLMRequest) -> LLMResponse \| None`                                             | Return cached response or `None` on miss. Cache errors treated as misses. |
| `set`        | `async (request: LLMRequest, response: LLMResponse, *, ttl_seconds: int \| None = None) -> None` | Store response. Cache errors are non-fatal (logged as warnings).          |
| `invalidate` | `async (request: LLMRequest) -> None`                                                            | Remove specific cache entry by request key.                               |
| `clear`      | `async (pattern: str = "*") -> int`                                                              | Flush all or pattern-matched entries. Returns count deleted.              |
| `stats`      | `async () -> dict[str, Any]`                                                                     | Returns `{backend, hits, misses, hit_rate, size, max_size, ttl_seconds}`. |
| `close`      | `async () -> None`                                                                               | Release backend resources.                                                |

### Properties

| Property   | Type    | Description                                      |
| ---------- | ------- | ------------------------------------------------ |
| `hits`     | `int`   | Total cache hits since creation                  |
| `misses`   | `int`   | Total cache misses since creation                |
| `hit_rate` | `float` | Fraction of lookups that were hits (`0.0`–`1.0`) |

## Further Reading

- [Providers Overview](../providers.md) — `create_llm_service` patterns and provider config
- [Routing & Cache](../routing-cache.md) — `LLMRouter` usage guide with full code examples
- [BYOK Guide](../byok.md) — `BYOKLLMService` patterns and multi-tenant architecture
- [API Reference — Core](./core.md) — Data models and enumerations
