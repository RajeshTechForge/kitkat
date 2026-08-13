---
title: Routing & Cache
description: This page explains all routing strategies, the circuit breaker state machine, cache configuration for both in-memory and Redis backends, and the observability surface exposed by the router.
order: 2
---

This page explains how to use `LLMRouter` to distribute requests across multiple providers with automatic failover, circuit breaking, and response caching. It covers all four routing strategies, the circuit breaker state machine, cache configuration for both in-memory and Redis backends, and the observability surface exposed by the router.

## Why Use the Router?

`LLMService` is the right choice when you always want to target a specific provider. `LLMRouter` is the right choice when you want:

- **Automatic failover** — if Anthropic is down, fall back to OpenAI transparently.
- **Load distribution** — spread load across providers via round-robin or random selection.
- **Latency optimization** — always pick the provider that has been fastest recently.
- **Response caching** — avoid redundant API calls for identical requests.
- **Circuit breaking** — stop sending requests to a provider that is failing, give it time to recover, then test it with a single probe.

## Building a Router

### Using the factory

```python
import asyncio
import os

from kitkat.service.router import LLMRouter
from kitkat.core.enums import RoutingStrategy
from kitkat.providers.anthropic import AnthropicProvider, AnthropicConfig
from kitkat.providers.openai import OpenAIProvider, OpenAIConfig

async def main() -> None:
    # LLMRouter.build() initializes all providers concurrently and skips
    # any provider that fails to initialize rather than raising immediately.
    router = await LLMRouter.build(
        providers=[
            AnthropicProvider(AnthropicConfig(api_key=os.environ["ANTHROPIC_API_KEY"])),
            OpenAIProvider(OpenAIConfig(api_key=os.environ["OPENAI_API_KEY"])),
        ],
    )

    from kitkat import LLMRequest, Message, Role
    request = LLMRequest(messages=[Message(role=Role.USER, content="Hello!")])
    response = await router.complete(request)
    print(response.content)
    print(f"Answered by: {response.provider}")

    await router.shutdown()

asyncio.run(main())
```

### Using the convenience factory function

```python
from kitkat.service import create_llm_router
from kitkat.core.enums import RoutingStrategy, CacheBackendType

router = create_llm_router(
    providers=[anthropic_provider, openai_provider],
    strategy=RoutingStrategy.FAILOVER,
    enable_cache=True,
    cache_backend=CacheBackendType.MEMORY,
)
```

> **📝 Note:** `create_llm_router` does **not** call `initialize()` on providers. Use `await LLMRouter.build(providers)` for concurrent initialization, or call `provider.initialize()` manually before passing to the factory.

### Using as an async context manager

```python
async with await LLMRouter.build([anthropic_provider, openai_provider]) as router:
    response = await router.complete(request)
    # shutdown() is called automatically on exit
```

### `RouterConfig` — full configuration

```python
from kitkat.service.router import RouterConfig, CircuitBreakerConfig
from kitkat.service.cache import CacheConfig
from kitkat.core.enums import RoutingStrategy, CacheBackendType

config = RouterConfig(
    strategy=RoutingStrategy.FAILOVER,    # Default: FAILOVER
    circuit_breaker=CircuitBreakerConfig(
        failure_threshold=5,              # Failures before opening. Default: 5
        recovery_timeout_s=60.0,          # Seconds before HALF_OPEN probe. Default: 60.0
        half_open_max_calls=1,            # Max probes in HALF_OPEN state. Default: 1
        success_threshold=2,              # Consecutive successes to close. Default: 2
    ),
    enable_cache=True,                    # Attach an LLMCache to the router. Default: True
    cache_on_truncated=False,             # Cache responses even when was_truncated=True. Default: False
    cache=CacheConfig(
        backend=CacheBackendType.MEMORY,  # Default: MEMORY
        ttl_seconds=3600,                 # Entry lifetime in seconds. Default: 3600
        max_memory_size=1000,             # Max entries before LRU eviction. Default: 1000
        redis_url="redis://localhost:6379/0",
        key_prefix="kitkat:llm:",
    ),
)
```

## Routing Strategies

The router selects which provider to try first based on the configured `RoutingStrategy`. All strategies maintain a fallback list — if the first candidate fails (and the error is retryable), the next candidate in the ordered list is tried.

### `FAILOVER` (default)

Providers are always tried in insertion order. The second provider is only used if the first fails.

```python
from kitkat.core.enums import RoutingStrategy

# providers=[anthropic, openai] → always tries Anthropic first, OpenAI on failure
config = RouterConfig(strategy=RoutingStrategy.FAILOVER)
```

**Best for:** Production setups where you have a preferred primary provider and want cost-predictable behavior.

### `ROUND_ROBIN`

Cycles through healthy providers in a round-robin pattern. Each call advances a shared counter (protected by `asyncio.Lock` for safe concurrent access).

```python
config = RouterConfig(strategy=RoutingStrategy.ROUND_ROBIN)
# Request 1 → Anthropic, Request 2 → OpenAI, Request 3 → Anthropic, ...
```

**Best for:** Spreading load evenly across providers with similar capabilities.

### `LEAST_LATENCY`

Selects the provider with the lowest average response latency, calculated from all completed requests since startup. Providers with no request history are ranked first (assigned a virtual latency of `-1.0 ms`).

```python
config = RouterConfig(strategy=RoutingStrategy.LEAST_LATENCY)
# Over time, the faster provider gets more traffic automatically.
```

**Best for:** Latency-sensitive applications where you want the router to self-tune.

### `RANDOM`

Selects a provider uniformly at random for each request.

```python
config = RouterConfig(strategy=RoutingStrategy.RANDOM)
```

**Best for:** Rough load distribution when exact fairness is not required.

## Circuit Breaker

Each provider in the pool has its own independent circuit breaker. The circuit breaker prevents cascading failures by blocking requests to unhealthy providers until they have had time to recover.

### State machine

```
         failure_threshold
CLOSED ──────────────────────► OPEN
  ▲                               │
  │  success_threshold            │ recovery_timeout_s
  │  consecutive successes        ▼
HALF_OPEN ◄────────────────── (timer fires)
  │
  │ probe fails
  └──────────────────────────► OPEN
```

| State       | Behavior                                                                                                                                                                                        |
| ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CLOSED`    | Normal operation. Requests are forwarded to the provider. A successful call resets the failure counter.                                                                                         |
| `OPEN`      | Provider is considered unhealthy. All requests are blocked immediately without hitting the provider. After `recovery_timeout_s` seconds, transitions to `HALF_OPEN`.                            |
| `HALF_OPEN` | One test probe is allowed through. If the probe succeeds, the circuit moves toward `CLOSED` after `success_threshold` consecutive successes. If the probe fails, the circuit returns to `OPEN`. |

### Configuring the circuit breaker

```python
from kitkat.service.router import CircuitBreakerConfig

cb_config = CircuitBreakerConfig(
    failure_threshold=5,     # Default: 5 — open after 5 consecutive failures
    recovery_timeout_s=60.0, # Default: 60.0 — wait 60s before probing
    half_open_max_calls=1,   # Default: 1 — only one probe at a time
    success_threshold=2,     # Default: 2 — need 2 successes to close
)
```

### Non-retryable errors

The following exception types are **never** retried across providers, because a different provider would encounter the same issue:

| Exception                | Reason not retried                                                                                    |
| ------------------------ | ----------------------------------------------------------------------------------------------------- |
| `LLMAuthenticationError` | The request has invalid credentials — routing to another provider changes nothing                     |
| `LLMTokenLimitError`     | The prompt is too long — routing to another provider (unless it has a larger context) changes nothing |
| `LLMContentFilterError`  | The content was blocked by safety policy — the same content would be blocked elsewhere                |

### Manual circuit breaker reset

In some scenarios (e.g., you have confirmed a provider issue is resolved), you can manually reset a circuit breaker:

```python
# Returns True if the provider was found and reset, False otherwise.
was_reset = await router.reset_circuit_breaker(ProviderType.ANTHROPIC)
print(was_reset)  # True
```

## Rate Limit Handling

When a provider returns `LLMRateLimitError` with a `retry_after_s` value (from the `Retry-After` HTTP header), the router records that provider as rate-limited until the specified time has elapsed. During this window, the provider is skipped in the routing order without counting against the circuit breaker. This respects the provider's back-off hint and avoids piling on further requests that would also fail.

## Response Caching

The router integrates an `LLMCache` that stores completed responses and returns them for identical requests without calling the provider. Caching applies only to non-streaming requests.

### Cache key

The cache key is a **SHA-256 hash** of:

```
(messages, model, max_tokens, temperature, top_p, stop_sequences_sorted)
```

The following fields are deliberately excluded from the key because they do not affect the generated text:

- `timeout` — an infrastructure concern, not a content parameter
- `stream` — caching applies only to non-streaming requests

### In-memory cache (default)

```python
from kitkat.service.cache import CacheConfig
from kitkat.core.enums import CacheBackendType

config = RouterConfig(
    enable_cache=True,
    cache=CacheConfig(
        backend=CacheBackendType.MEMORY,
        max_memory_size=1000,   # Max entries before LRU eviction. Default: 1000
        ttl_seconds=3600,       # Entry lifetime in seconds. Default: 3600
    ),
)
```

The in-memory backend is an `asyncio`-safe LRU cache backed by `collections.OrderedDict`. When `max_memory_size` is reached, the least-recently-used entry is evicted. TTL expiry is enforced lazily on each `get()` call.

**Suitable for:** Single-process applications or development environments. Cache state is lost on process restart.

### Redis cache

```python
from kitkat.service.cache import CacheConfig
from kitkat.core.enums import CacheBackendType

config = RouterConfig(
    enable_cache=True,
    cache=CacheConfig(
        backend=CacheBackendType.REDIS,
        redis_url="redis://localhost:6379/0",  # Supports redis://, rediss:// (TLS)
        ttl_seconds=3600,
        key_prefix="kitkat:llm:",             # Namespace prefix. Change to avoid key collisions.
    ),
)
```

The Redis backend uses `redis.asyncio` with a connection pool of up to 20 connections. Entries are stored as JSON with native Redis TTL (`SETEX`). The backend uses `SCAN` (non-blocking, cursor-based) rather than `KEYS` to avoid stalling Redis on large keyspaces.

**Suitable for:** Multi-process or multi-instance deployments where cache state must be shared.

> **🔒 Security:** When using Redis over a network, use the `rediss://` (TLS) scheme and configure Redis authentication to prevent unauthorized access to cached LLM responses.

### Cache behaviour guarantees

- **Fail-safe:** Cache errors (network failures, serialization issues) are logged as warnings and treated as misses. A cache error never interrupts inference.
- **No caching of filtered content:** Responses with `FinishReason.CONTENT_FILTER` or `FinishReason.ERROR` are never cached.
- **Truncated responses:** By default, responses with `was_truncated=True` (i.e., `finish_reason=LENGTH`) are not cached. Enable caching of truncated responses with `RouterConfig(cache_on_truncated=True)`.

### Invalidating the cache

```python
# Remove a specific request from the cache.
await router.cache.invalidate(request)

# Clear all entries.
await router.cache.clear()

# Clear entries matching a glob pattern.
deleted_count = await router.cache.clear(pattern="some-prefix*")
```

### Cache statistics

```python
stats = await router.cache.stats()
# {
#   "backend": "memory",
#   "hits": 42,
#   "misses": 10,
#   "hit_rate": 0.807,
#   "size": 38,
#   "max_size": 1000,
#   "ttl_seconds": 3600,
# }
print(f"Hit rate: {router.cache.hit_rate:.1%}")
```

## Router API

#### `classmethod async LLMRouter.build(providers, config=None) -> LLMRouter`

Initializes all providers concurrently and returns a ready router. Providers that fail to initialize are logged and skipped. Raises `ValueError` if **all** providers fail.

#### `async complete(request) -> LLMResponse`

Routes a blocking completion through the provider pool. Checks the cache first, applies the routing strategy, respects circuit breakers and rate-limit windows, and writes the result to the cache on success.

#### `async stream(request) -> AsyncIterator[StreamChunk]`

Routes a streaming request. Streaming responses are not cached. If a stream fails **before** the first token is yielded, the router falls back to the next provider. If a stream fails **after** the first token, the router raises immediately (the partial response cannot be replayed).

#### `async health_check() -> dict[tuple[int, ProviderType], bool]`

Probes all providers concurrently (using `asyncio.gather`). Returns a dict keyed by `(pool_index, ProviderType)` → `bool`.

```python
health = await router.health_check()
for (idx, provider_type), healthy in health.items():
    print(f"  [{idx}] {provider_type.value}: {'OK' if healthy else 'FAILING'}")
```

#### `async status() -> dict[str, Any]`

Returns a comprehensive status snapshot suitable for an admin endpoint or dashboard:

```python
status = await router.status()
# {
#   "strategy": "failover",
#   "provider_count": 2,
#   "healthy_count": 2,
#   "cache_enabled": True,
#   "providers": [
#     {
#       "provider": ProviderType.ANTHROPIC,
#       "model": "claude-sonnet-4-6",
#       "healthy": True,
#       "circuit_state": "CLOSED",
#       "total_requests": 150,
#       "failed_requests": 2,
#       "avg_latency_ms": 843.2,
#       "error_rate": 0.013,
#       "rate_limited_until": None,
#     },
#     ...
#   ],
# }
```

#### `async reset_circuit_breaker(provider_type) -> bool`

Manually resets a provider's circuit breaker to `CLOSED` and clears its rate-limit window. Returns `True` if the provider was found, `False` otherwise.

#### `async shutdown()`

Shuts down all providers and flushes the cache.

#### Properties

| Property    | Type                | Description                                                           |
| ----------- | ------------------- | --------------------------------------------------------------------- |
| `providers` | `list[LLMProvider]` | Read-only snapshot of the provider pool                               |
| `cache`     | `LLMCache \| None`  | Direct access to the cache instance, or `None` if caching is disabled |

## Full Example: Failover with Redis Cache

```python
import asyncio
import os

from kitkat.service.router import LLMRouter, RouterConfig, CircuitBreakerConfig
from kitkat.service.cache import CacheConfig
from kitkat.core.enums import RoutingStrategy, CacheBackendType
from kitkat import LLMRequest, Message, Role, ProviderType
from kitkat.providers.anthropic import AnthropicProvider, AnthropicConfig
from kitkat.providers.openai import OpenAIProvider, OpenAIConfig

async def main() -> None:
    config = RouterConfig(
        strategy=RoutingStrategy.FAILOVER,
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout_s=30.0,
        ),
        enable_cache=True,
        cache=CacheConfig(
            backend=CacheBackendType.REDIS,
            redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
            ttl_seconds=1800,  # 30-minute cache
        ),
    )

    async with await LLMRouter.build(
        providers=[
            AnthropicProvider(AnthropicConfig(api_key=os.environ["ANTHROPIC_API_KEY"])),
            OpenAIProvider(OpenAIConfig(api_key=os.environ["OPENAI_API_KEY"])),
        ],
        config=config,
    ) as router:
        request = LLMRequest(
            messages=[Message(role=Role.USER, content="What is the capital of France?")],
            max_tokens=64,
            temperature=0.0,  # Deterministic — good candidate for caching
        )

        # First call — hits the provider
        response1 = await router.complete(request)
        print(f"Response 1 (from {response1.provider}): {response1.content}")

        # Second identical call — served from Redis cache
        response2 = await router.complete(request)
        print(f"Response 2 (cached): {response2.content}")

        stats = await router.cache.stats()
        print(f"Cache hit rate: {stats['hit_rate']:.1%}")

asyncio.run(main())
```

## Further Reading

- [Providers](./providers.md) — Configure each individual provider
- [BYOK](./byok.md) — Per-request user API keys
- [Error Handling](./error-handling.md) — The full exception hierarchy
- [API Reference — Service](./api-reference/service.md) — Complete API for `LLMRouter` and `LLMCache`
