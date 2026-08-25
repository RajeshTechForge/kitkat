# MASTER ARCHITECTURE

> **Audience:** Library contributors and AI assistants modifying or extending `kitkat`.
> This document is the single source of truth for the internal architecture.
> It is deliberately **not** user documentation — end-user docs live in `README.md` and `docs/`.

---

## Table of Contents

1. [Purpose & Scope](#1-purpose--scope)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Directory & Module Structure](#3-directory--module-structure)
4. [Core Abstractions](#4-core-abstractions)
5. [Data Flow](#5-data-flow)
6. [Key Implementation Details](#6-key-implementation-details)
7. [Extension Points](#7-extension-points)
8. [Dependencies & Constraints](#8-dependencies--constraints)
9. [Conventions](#9-conventions)

---

## 1. Purpose & Scope

### What the library does

`kitkat` is a **production-grade, async-first Python library** that provides a unified interface
for making inference calls to multiple LLM providers (Anthropic, OpenAI, Google Gemini, Vertex AI).
It handles:

- **Provider lifecycle** — connection pool creation, credential probing, graceful shutdown
- **Retry logic** — exponential back-off with jitter, provider-aware non-retriable error fast-fail
- **Multi-provider routing** — failover, round-robin, least-latency, and random strategies
- **Circuit breaking** — per-provider, asyncio-safe three-state (CLOSED → OPEN → HALF_OPEN) breakers
- **Response caching** — SHA-256-keyed, with in-memory LRU and Redis backends
- **BYOK (Bring Your Own Key)** — per-request ephemeral provider instances that skip credential
  probing
- **PydanticAI integration** — model adapters and agent builder factories for both managed and BYOK
  service paths
- **LangGraph workflows** — a typed base class for stateful multi-step agentic pipelines
- **Observability** — Logfire + Langfuse via OpenTelemetry, wired to PydanticAI automatically
- **Plugin system** — third-party providers ship via `entry_points` and are auto-discovered

### What it deliberately does NOT do

- **No HTTP framework** — kitkat does not include a web server or FastAPI route handlers. It is
  a library to be embedded in an application.
- **No auth/identity** — it accepts API keys as opaque strings; key management, rotation, and
  vault integration are the caller's responsibility.
- **No domain tools** — `ToolRegistry` provides the *mechanism* for registering tools on
  PydanticAI agents, but ships zero domain-specific tools.
- **No streaming transport** — SSE framing schemas (`StreamSSEEvent`, `StreamChunkEvent`,
  `StreamErrorEvent`) are defined in `core/schemas.py` but the actual SSE transport is the
  responsibility of the embedding application.
- **No configuration file parsing** — there is no `config.toml` reader. All configuration is
  passed in as typed dataclasses (`AnthropicConfig`, `OpenAIConfig`, etc.).
- **No model fine-tuning or embedding** — completion and streaming only.

---

## 2. High-Level Architecture

The library is organized into five horizontal layers. Each layer only imports downward; the
arrows below show allowed dependency direction.

```
┌─────────────────────────────────────────────────────────────────┐
│  Public API  (__init__.py — lazy re-exports, no business logic) │
└────────────────────────────┬────────────────────────────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
┌──────────────────┐ ┌──────────────┐ ┌──────────────────┐
│   agents/        │ │  workflows/  │ │    service/      │
│  PydanticAI      │ │  LangGraph   │ │  Routing facade  │
│  adapters,       │ │  stateful    │ │  (LLMService,    │
│  builders,       │ │  pipelines   │ │  LLMRouter,      │
│  tool registry,  │ │              │ │  BYOKLLMService, │
│  observability   │ │              │ │  LLMCache)       │
└────────┬─────────┘ └──────┬───────┘ └────────┬─────────┘
         │                  │                  │
         └──────────────────┼──────────────────┘
                            │
                            ▼
          ┌─────────────────────────────────┐
          │     abc/ + providers/           │
          │  LLMProvider ABC, concrete      │
          │  AnthropicProvider,             │
          │  OpenAIProvider,                │
          │  GeminiProvider,                │
          │  VertexAIProvider               │
          │  + _registry.py (entry-points)  │
          └─────────────────┬───────────────┘
                            │
                            ▼
          ┌─────────────────────────────────┐
          │         core/ + _internal/      │
          │  Enums, models (dataclasses),   │
          │  Pydantic schemas, exceptions,  │
          │  retry logic, tokenizers, HTTP  │
          └─────────────────────────────────┘
```

### Component interaction summary

| Component | Talks to | Purpose |
|---|---|---|
| `LLMService` | `LLMProvider` ABC | Provider registry and routing facade for managed (server-key) path |
| `LLMRouter` | `LLMProvider` ABC, `LLMCache` | Multi-provider strategy routing with circuit breaking and caching |
| `BYOKLLMService` | Concrete providers directly | Per-request ephemeral provider for user-supplied keys |
| `ManagedModelAdapter` | `LLMService` | Bridges PydanticAI `Model` protocol → `LLMService` |
| `BYOKModelAdapter` | `BYOKLLMService` | Bridges PydanticAI `Model` protocol → `BYOKLLMService` |
| `AnthropicProvider` / etc. | Vendor SDKs, `_internal/retry.py` | Translates domain requests to SDK calls |
| `plugins/loader.py` | `providers/_registry.py` | Public surface for `entry_points`-based provider discovery |

---

## 3. Directory & Module Structure

```
src/kitkat/
├── __init__.py            # Public API surface + lazy import machinery
├── py.typed               # PEP 561 marker — kitkat ships type stubs
│
├── core/                  # Zero-dependency domain types (enums, models, exceptions, schemas)
│   ├── enums.py           # StrEnum: Role, FinishReason, ProviderType, RoutingStrategy…
│   ├── models.py          # Frozen dataclasses: Message, LLMRequest, LLMResponse, StreamChunk…
│   ├── exceptions.py      # KitkatError → LLMError → per-condition subclasses
│   └── schemas.py         # Pydantic V2 schemas for the API boundary (validated input/output)
│
├── abc/
│   └── provider.py        # LLMProvider ABC — the contract every provider must satisfy
│
├── _internal/             # Private utilities (NOT part of public API)
│   ├── http.py            # httpx.AsyncClient factory with kitkat User-Agent
│   ├── retry.py           # execute_with_retry() — the shared retry engine
│   └── tokenizers.py      # count_tokens_tiktoken() with char-ratio fallback
│
├── providers/             # Concrete provider implementations
│   ├── _registry.py       # entry_points-based auto-discovery + get_provider_class()
│   ├── anthropic/
│   │   └── provider.py    # AnthropicProvider + AnthropicConfig
│   ├── openai/
│   │   └── provider.py    # OpenAIProvider + OpenAIConfig (also covers NVIDIA NIM)
│   └── google/
│       ├── _shared.py     # Shared helpers between Gemini and Vertex AI providers
│       ├── gemini.py      # GeminiProvider + GeminiConfig
│       └── vertex_ai.py   # VertexAIProvider + VertexAIConfig
│
├── service/               # Service layer — routing, caching, lifecycle management
│   ├── managed.py         # LLMService — provider registry facade (server-key path)
│   ├── byok.py            # BYOKLLMService — per-request ephemeral provider
│   ├── router.py          # LLMRouter — multi-provider strategy routing + circuit breakers
│   ├── cache.py           # LLMCache, InMemoryCache, RedisCache, CacheBackend ABC
│   └── factory.py         # create_llm_service(), create_llm_router() convenience factories
│
├── agents/                # PydanticAI integration (requires kitkat[agents])
│   ├── _check.py          # require_agents_extra() / require_observability_extra() guards
│   ├── context.py         # BaseAgentContext dataclass — deps_type for all agents
│   ├── builders.py        # build_chat_agent(), build_structured_agent()
│   ├── observability.py   # configure_observability() — Logfire + Langfuse OTel wiring
│   ├── adapters/
│   │   ├── managed.py     # ManagedModelAdapter, KitkatStreamedResponse
│   │   └── byok.py        # BYOKModelAdapter, BYOKKitkatStreamedResponse
│   └── tools/
│       └── registry.py    # ToolRegistry — bulk tool registration for PydanticAI agents
│
├── plugins/
│   └── loader.py          # Public API facade over providers/_registry.py
│
└── workflows/             # LangGraph integration (requires kitkat[workflows])
    ├── base.py            # BaseWorkflow ABC — typed, generic, async
    └── research.py        # ResearchWorkflow — reference implementation
```

### Key file responsibilities at a glance

| File | One-line responsibility |
|---|---|
| `__init__.py` | Exposes the public API; eagerly imports core types; lazily imports service/agents/workflows via `__getattr__` |
| `core/models.py` | The **lingua franca** of the library — all layers exchange these dataclasses |
| `core/schemas.py` | Pydantic validation at the API boundary; converts to/from domain dataclasses |
| `abc/provider.py` | Defines the five abstract methods every provider must implement; holds shared retry, sync-run, and token-counting helpers |
| `_internal/retry.py` | The single retry engine shared by every provider via `complete_with_retry()` |
| `service/router.py` | The most complex module — routing strategies + per-provider circuit breakers |
| `service/cache.py` | Two-backend LRU cache with SHA-256 keys; fail-safe (cache errors never break inference) |
| `agents/adapters/managed.py` | Translates PydanticAI's `Model` protocol into `LLMService` calls |
| `providers/_registry.py` | Auto-discovers providers from `entry_points` at import time |

---

## 4. Core Abstractions

### 4.1 `LLMProvider` — the central contract

[`abc/provider.py`](file:///home/rajesh/Desktop/kitkat/src/kitkat/abc/provider.py)

Every concrete provider (`AnthropicProvider`, `OpenAIProvider`, `GeminiProvider`,
`VertexAIProvider`, and any third-party provider) **must** subclass `LLMProvider` and implement
these five abstract methods:

| Abstract method | Purpose |
|---|---|
| `initialize()` | Create HTTP client + probe credentials. Called by `LLMService.initialize()` or `LLMRouter.build()`. |
| `shutdown()` | Release all resources (close HTTP connection pool). |
| `_init_client_only()` | Create HTTP client **without** credential probe. Used by `BYOKLLMService` to avoid per-user preflight calls. |
| `complete(request)` | Single non-streaming inference attempt. No retry logic here. |
| `stream(request)` | Async generator yielding `StreamChunk` objects. Final chunk has `is_final=True` with aggregated metadata. |
| `health_check()` | Lightweight liveness probe. Anthropic uses `count_tokens`; OpenAI uses `models.list`. |
| `count_tokens(text)` | Synchronous token estimate. Providers delegate to `_internal/tokenizers.py`. |

**Why is `complete()` separated from `complete_with_retry()`?**
The `complete()` method is a *single attempt*. The retry wrapper `complete_with_retry()` lives on
the base class and calls `_internal/retry.execute_with_retry()`. This separation means providers
focus only on the SDK integration, while retry logic is centralized and uniform.

**Class-level attributes providers MUST declare:**

```python
class MyProvider(LLMProvider):
    PROVIDER_TYPE: ProviderType  = ProviderType.OPENAI
    DEFAULT_MODEL: str           = "my-model-v1"
    CAPABILITIES: ProviderCapabilities = ProviderCapabilities(
        supports_streaming=True,
        max_context_tokens=32_768,
        provider_type=ProviderType.OPENAI,
    )
    RETRY_POLICY: RetryPolicy    = RetryPolicy(...)  # optional override
```

The service and router layers query `CAPABILITIES` to understand what the provider supports, and
use `PROVIDER_TYPE` as the routing key.

---

### 4.2 Domain types (`core/models.py`)

These are the **only** types that flow between the ABC, service, and provider layers.
They are plain Python `dataclass` objects — no Pydantic, no SDK dependencies.

| Type | Role |
|---|---|
| `Message(role, content)` | A single conversation turn |
| `LLMRequest` | Everything a provider needs to serve a request (messages, model, temperature, etc.) |
| `LLMResponse` | A completed response (content, finish_reason, usage, latency_ms, raw_response) |
| `StreamChunk` | One token delta from a streaming response; final chunk carries aggregated metadata |
| `TokenUsage` | prompt/completion/thinking/total token counts; supports `+` aggregation |
| `ProviderCapabilities` | Feature flags queried by the router (`supports_thinking`, `max_context_tokens`, etc.) |
| `RetryPolicy` | Back-off configuration; has `delay_for_attempt(n)` with jitter |
| `ThinkingConfig` | Provider-agnostic extended-thinking configuration |

**Why dataclasses not Pydantic?**
These types are allocated on the hot path (every request). Frozen dataclasses are faster to
instantiate and carry no validation overhead — validation happens once at the schema boundary
(`core/schemas.py`) before the domain objects are created.

---

### 4.3 Schema boundary (`core/schemas.py`)

Pydantic V2 models that validate **inbound** data (from API routes or external callers) and
convert it into domain dataclasses via `to_domain()`.

```
External caller ──► LLMRequestSchema.validate() ──► LLMRequest (dataclass)
                                                            │
                                                            ▼
                                                      LLMProvider.complete()
                                                            │
                                                            ▼
                                              LLMResponse (dataclass)
                                                            │
                                 LLMResponseSchema.from_domain() ──► JSON
```

Key validators enforced at this boundary:
- `system_message_at_most_one_and_first` — structural constraint on the message list
- `content_not_whitespace_only` — rejects blank messages
- `total_is_consistent` — auto-corrects `TokenUsage.total_tokens` when providers round differently

`raw_response` (the native SDK object) is intentionally dropped at the `to_domain()` boundary
on the response side — it is never serializable and belongs only to provider-level debugging.

---

### 4.4 Exception hierarchy

```
KitkatError
└── LLMError(status_code, provider)
    ├── LLMProviderInitError   — startup failures (bad API key, network)
    ├── LLMProviderError       — generic runtime failure
    ├── LLMRateLimitError      — HTTP 429; carries retry_after_s
    ├── LLMTimeoutError        — request exceeded timeout; carries elapsed_s
    ├── LLMTokenLimitError     — prompt exceeds context window
    ├── LLMContentFilterError  — blocked by safety policy
    └── LLMAuthenticationError — invalid/revoked credentials
```

**Non-retriable errors** (`LLMAuthenticationError`, `LLMTokenLimitError`,
`LLMContentFilterError`) are re-raised immediately by `execute_with_retry()` and by
`LLMRouter._route_complete()`. Retrying them is deterministically pointless.

---

### 4.5 Service layer classes

| Class | File | Role |
|---|---|---|
| `LLMService` | `service/managed.py` | Registry + lifecycle manager for server-side provider keys. Single entry point for managed inference. |
| `LLMRouter` | `service/router.py` | Multi-provider routing with strategy selection and per-provider circuit breakers. Wraps `LLMService` or used standalone. |
| `BYOKLLMService` | `service/byok.py` | Async context manager that builds, initializes (without probe), uses, and tears down a single provider per request. |
| `LLMCache` | `service/cache.py` | Orchestrator; wraps `InMemoryCache` or `RedisCache`. Keyed by SHA-256 of serialized request semantics. |

---

## 5. Data Flow

### Flow A: Managed path — blocking completion

This traces a request through `LLMService` (the most common server-side path).

```
Caller
  │
  │  await service.complete(request, ProviderType.ANTHROPIC)
  ▼
LLMService._resolve(ProviderType.ANTHROPIC)
  │   → looks up provider in self._providers dict
  │   → raises LLMProviderError if not registered
  ▼
AnthropicProvider.complete_with_retry(request)
  │   → delegates to _internal/retry.execute_with_retry(
  │         func=lambda: self.complete(request),
  │         policy=self.RETRY_POLICY,
  │     )
  ▼
execute_with_retry (loop: up to policy.max_attempts)
  │
  │── attempt 1 ──►  AnthropicProvider.complete(request)
  │                    │  _assert_initialized()
  │                    │  _split_messages()  → extract system prompt
  │                    │  _build_thinking_params()
  │                    │  asyncio.wait_for(
  │                    │      client.messages.create(...), timeout
  │                    │  )
  │                    │  _build_response(raw_sdk_msg)
  │                    └─► LLMResponse(content, usage, finish_reason, latency_ms, …)
  │
  │  On LLMRateLimitError: sleep(retry_after_s or backoff), retry
  │  On LLMAuthenticationError/LLMTokenLimitError/LLMContentFilterError: raise immediately
  │
  ▼
LLMResponse returned to caller
```

---

### Flow B: Router path — completion with failover and cache

This traces a request through `LLMRouter` (used when multi-provider resilience is needed).

```
Caller
  │
  │  await router.complete(request)
  ▼
LLMRouter.complete(request)
  │
  ├── Cache lookup: await self._cache.get(request)
  │       make_cache_key(request) → SHA-256 of serialized semantics
  │       InMemoryCache.get(key) or RedisCache.get(key)
  │       Cache HIT? → return cached LLMResponse immediately
  │
  └── Cache MISS → _route_complete(request)
          │
          │  _provider_order() → [0, 1, 2] (FAILOVER)
          │                      or round-robin rotated indices
          │                      or sorted by avg_latency_ms
          │
          │  for idx in order:
          │      CircuitBreaker.is_open()?   → skip if OPEN
          │      rate_limited_until[idx]?    → skip if still cooling down
          │
          │      try:
          │          response = await provider.complete(request)
          │          stats[idx].record_success(latency)
          │          breaker.record_success()
          │          return response, idx
          │
          │      except _NON_RETRYABLE: raise immediately (no fallback)
          │      except LLMRateLimitError: record_failure, set cooldown, try next
          │      except LLMProviderError/LLMTimeoutError: record_failure, try next
          │
          │  All providers exhausted? raise last_exc
          │
          ▼
      LLMResponse
          │
          └── Cache SET (unless was_truncated and cache_on_truncated=False,
                          or finish_reason is CONTENT_FILTER/ERROR)
              await self._cache.set(request, response)
```

---

### Flow C: BYOK path — per-request ephemeral provider

Used when each request carries the user's own API key.

```
Caller code:
    async with BYOKLLMService(ProviderType.OPENAI, user_key, model) as byok:
        response = await byok.complete(request)

BYOKLLMService.__init__:
    _build_provider(ProviderType.OPENAI, api_key, model)
    → OpenAIProvider(OpenAIConfig(api_key=api_key, model=model))
    # Config __post_init__ validates api_key is non-empty

BYOKLLMService.__aenter__:
    await self._provider._init_client_only()
    # Creates AsyncOpenAI client WITHOUT models.list credential probe
    # Auth failures surface on first complete() instead

BYOKLLMService.complete(request):
    await self._provider.complete_with_retry(request)
    # Same retry engine as managed path

BYOKLLMService.__aexit__:
    await self._provider.shutdown()
    # Closes HTTPX connection pool — no leaks on error paths
```

---

### Flow D: PydanticAI agent (managed adapter)

```
PydanticAI Agent.run("Hello!", deps=user_ctx)
  │
  │  pydantic-ai run loop assembles ModelMessage list
  ▼
ManagedModelAdapter.request(messages, model_settings, ...)
  │
  │  _to_llm_request(messages, model_settings)
  │    → iterates pydantic-ai message parts
  │    → maps SystemPromptPart/InstructionPart → Role.SYSTEM
  │    → maps UserPromptPart → Role.USER
  │    → maps TextPart on ModelResponse → Role.ASSISTANT
  │    → ToolCallPart / ToolReturnPart → text fallback (known limitation)
  │
  ▼
LLMService.complete(llm_request, provider_type)
  │   (same as Flow A from here)
  ▼
LLMResponse
  │
  ▼
ModelResponse(parts=[TextPart(response.content)], usage=_to_request_usage(response.usage))
  │
  ▼
pydantic-ai returns result.data to caller
```

---

## 6. Key Implementation Details

### 6.1 Lazy import machinery in `__init__.py`

Heavy optional dependencies (`pydantic-ai`, `langgraph`, provider SDKs) are not imported at
package load time. The `_LAZY_EXPORTS` dict maps public names to `(module_path, attr_name)` pairs.
`__getattr__` is implemented to perform the import on first access and cache the result in
`globals()`.

```python
# Works even if pydantic-ai is not installed — until you actually access it:
import kitkat
kitkat.ManagedModelAdapter  # only NOW does `agents.adapters.managed` get imported
```

This means `import kitkat` always succeeds with only the mandatory deps installed.

---

### 6.2 Two initialization paths: `initialize()` vs `_init_client_only()`

The `_init_client_only()` method is specifically designed for BYOK:

| | `initialize()` | `_init_client_only()` |
|---|---|---|
| **Used by** | `LLMService`, `LLMRouter.build()` | `BYOKLLMService.__aenter__()` |
| **Credential probe** | ✅ Yes (fails fast at startup) | ❌ No (avoids extra latency per user) |
| **Auth failure timing** | At startup | At first `complete()` / `stream()` call |
| **Idempotent** | Yes — checks `self._initialized` | Yes — checks `self._initialized` |

Do **not** call `_init_client_only()` for long-lived providers. The credential probe in
`initialize()` exists so that a misconfigured API key is caught at server startup, not on the
first user request in production.

---

### 6.3 Thinking mode — provider differences

The `ThinkingConfig` domain object is normalized and then provider-specific:

| Provider | Thinking toggle | Effort mapping | Token reporting |
|---|---|---|---|
| Anthropic | `thinking` param: `{type: "adaptive"}` + `output_config: {effort: "high"}` | `"low"/"medium"/"high"` or provider `effort` override | `output_tokens` includes thinking; `thinking_tokens` reported as 0 |
| OpenAI | No toggle — always reasoning-capable; `reasoning_effort` param | `"low"/"medium"/"high"/"xhigh"` | `completion_tokens_details.reasoning_tokens` extracted separately |
| Gemini | `thinking_config.thinking_budget` | Mapped from `"low"/"medium"/"high"` | Separate field |

**Critical Anthropic detail**: when thinking is enabled, `temperature` and `top_p` must be omitted
from the request (Anthropic rejects them). The provider sends `anthropic.omit` for those fields
when `thinking_param is not anthropic.omit`.

---

### 6.4 Circuit breaker — asyncio safety

All circuit-breaker state transitions are protected by `asyncio.Lock`. This is crucial because
multiple coroutines may be invoking the router concurrently.

State machine:
```
CLOSED (normal)
  │  failure_count >= failure_threshold (default 5)
  ▼
OPEN (reject all requests)
  │  recovery_timeout_s elapsed (default 60s)
  ▼
HALF_OPEN (let one probe through)
  │  success_threshold consecutive successes (default 2) → CLOSED
  │  any failure → OPEN (reset timer)
```

**Gotcha**: HALF_OPEN tracks `_half_open_calls` to limit concurrent probes. When
`_half_open_calls >= half_open_max_calls`, the circuit blocks incoming requests even in HALF_OPEN
state. This prevents a thundering herd of probes when the timer fires under high concurrency.

---

### 6.5 Router stream fallback limitation

For streaming, the router can only fall back to another provider **before the first chunk is
yielded**. Once streaming has started and a chunk has been delivered to the caller, falling back
mid-stream is impossible because the caller has already received partial content.

```python
if first_chunk_yielded:
    # Cannot fall back — partial content already delivered
    raise
```

This means streaming reliability depends more heavily on choosing a stable primary provider than
in the non-streaming path.

---

### 6.6 Cache key design

The SHA-256 key covers only semantically significant fields:

```python
key_data = {
    "messages":       [...],  # role+content pairs
    "model":          ...,
    "max_tokens":     ...,
    "temperature":    ...,
    "top_p":          ...,
    "stop_sequences": sorted(...),  # order-normalized
}
```

**Deliberately excluded**:
- `metadata` — arbitrary bag; no semantic impact on LLM output
- `timeout` — infrastructure concern, not content

**Responses that are never cached**:
- `finish_reason == CONTENT_FILTER` or `ERROR` — non-deterministic/policy-driven
- Truncated responses (`finish_reason == LENGTH`) — unless `RouterConfig.cache_on_truncated=True`

Both backends are fail-safe: any cache exception is caught and logged as a warning; inference
continues as a cache miss.

---

### 6.7 tiktoken fallback

Each provider initializes its tiktoken encoder lazily (on first `count_tokens()` call). In
air-gapped environments where tiktoken cannot download the BPE file, the sentinel object
`_TIKTOKEN_UNAVAILABLE` is stored so the expensive load is never retried. The fallback returns
`max(1, len(text) // 4)` — approximately 4 chars per token. This is good enough for budgeting
but should not be relied on for precise cost attribution.

---

### 6.8 PydanticAI adapter — tool call limitation

`_to_llm_request()` in `agents/adapters/managed.py` handles four pydantic-ai part types.
`ToolCallPart` and `ToolReturnPart` are serialized as **text fallbacks**:

```python
# ToolCallPart → "[tool_call:search_docs({"query": "..."})]"
# ToolReturnPart → "[tool_result:search_docs] result text"
```

This works for simple tool echoing but does not preserve native function-call semantics at the
provider level. This is a known limitation documented with a `# TODO` in the source. The correct
long-term fix is to extend `LLMRequest`/`Message` to carry structured tool content natively.

---

### 6.9 Provider `config.__dict__` pass-through

All concrete providers call `super().__init__(config.__dict__)` to satisfy `LLMProvider.__init__`
which expects a `dict[str, Any]`. The typed `*Config` dataclass is also stored directly as
`self._cfg` for idiomatic attribute access. This dual storage is intentional: the raw dict is
needed for the ABC contract; the typed config is used internally.

---

### 6.10 OpenAI uses `max_completion_tokens` not `max_tokens`

OpenAI reasoning models (`o1`, `o3`, etc.) require `max_completion_tokens` instead of the older
`max_tokens`. The `OpenAIProvider` always sends `max_completion_tokens=request.max_tokens` to
support both standard and reasoning models uniformly.

---

## 7. Extension Points

### 7.1 Adding a new LLM provider

This is the primary extension point. Three steps:

**Step 1 — Implement `LLMProvider`**

```python
# mypkg/provider.py
from kitkat.abc import LLMProvider
from kitkat.core.models import LLMRequest, LLMResponse, ProviderCapabilities, StreamChunk
from kitkat.core.enums import ProviderType
from collections.abc import AsyncIterator

class MyProvider(LLMProvider):
    PROVIDER_TYPE = ProviderType.OPENAI  # reuse nearest slot or extend ProviderType
    DEFAULT_MODEL = "my-model-v1"
    CAPABILITIES = ProviderCapabilities(
        supports_streaming=True,
        max_context_tokens=32_768,
        provider_type=ProviderType.OPENAI,
    )

    async def initialize(self) -> None: ...
    async def shutdown(self) -> None: ...
    async def _init_client_only(self) -> None: ...
    async def complete(self, request: LLMRequest) -> LLMResponse: ...
    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]: ...
    async def health_check(self) -> bool: ...
    def count_tokens(self, text: str) -> int: ...
```

**Step 2 — Register via entry point (for distributable packages)**

```toml
# mypkg/pyproject.toml
[project.entry-points."kitkat.providers"]
my-llm = "mypkg.provider:MyProvider"
```

After `pip install mypkg`, the provider is auto-discovered by `providers/_registry.py` at
import time.

**Step 3 — Or register programmatically (for in-repo providers)**

```python
from kitkat.plugins import register_provider
register_provider("my-llm", MyProvider)
```

> **Important**: `ProviderType` is a `StrEnum`. If you need a canonical value for a genuinely
> new provider (not one of the four built-in slots), open a PR to add it to `core/enums.py`.
> Do not invent string literals inline.

---

### 7.2 Adding a new routing strategy

1. Add a new value to `RoutingStrategy` in `core/enums.py`
2. Implement the ordering logic in `LLMRouter._provider_order()` in `service/router.py`

The method must return a `list[int]` of provider indices in the order they should be tried.
Circuit-breaker checks and rate-limit skipping happen *after* the order is determined.

---

### 7.3 Adding a new cache backend

1. Subclass `CacheBackend` (ABC in `service/cache.py`)
2. Implement all six abstract methods: `get`, `set`, `delete`, `clear_pattern`, `size`, `close`
3. Add a new `CacheBackendType` enum value in `core/enums.py`
4. Wire up the new backend in `LLMCache._build_backend()`

---

### 7.4 Adding a new workflow

Subclass `BaseWorkflow[StateT]` from `workflows/base.py`:

```python
from kitkat.workflows.base import BaseWorkflow
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel

class MyState(BaseModel):
    ...

class MyWorkflow(BaseWorkflow[MyState]):
    def __init__(self) -> None:
        self._graph = self.build_graph()

    def build_graph(self):
        g = StateGraph(MyState)
        # add nodes and edges
        return g.compile()

    async def run(self, initial_state: MyState | dict) -> MyState:
        result = await self._graph.ainvoke(initial_state)
        return MyState.model_validate(result)
```

Compile the graph once at module level (`__init__`); share the instance across requests.

---

### 7.5 Extending `BaseAgentContext`

Application code subclasses `BaseAgentContext` to carry app-specific fields:

```python
@dataclass
class UserContext(BaseAgentContext):
    conversation_id: str = ""
    byok_api_key: str | None = None
    # …any app-specific fields
```

Pass the subclass as `context_type` to `build_chat_agent()`. Library code only reads fields
declared on `BaseAgentContext`; all subclass fields are accessible to application tools through
`RunContext[UserContext].deps`.

---

### 7.6 Adding tools to agents

Use `ToolRegistry` for shared tools across multiple agents, or pydantic-ai's `@agent.tool`
decorator for agent-specific tools:

```python
from kitkat.agents.tools.registry import ToolRegistry

registry = ToolRegistry()

@registry.tool(name="web_search", description="Search the web.")
async def web_search(ctx: RunContext[UserContext], query: str) -> str:
    return "..."

agent = build_chat_agent(model=adapter, context_type=UserContext)
registry.register_on(agent)
```

---

## 8. Dependencies & Constraints

### Mandatory dependencies (always installed)

| Package | Why mandatory |
|---|---|
| `httpx >= 0.28.1` | Async HTTP client used by `_internal/http.py` for custom providers |
| `pydantic >= 2.13` | V2 schema validation at the API boundary; `BaseModel` in `core/schemas.py` |
| `pydantic-settings >= 2.13` | Env-var binding for provider config (available for consumers) |
| `tiktoken >= 0.13.0` | Token counting; mandatory even if no provider is installed, as it's a fallback |
| `redis >= 7.1.1` | `redis.asyncio` is imported directly in `service/cache.py` at module level |

> **Gotcha**: `redis` is listed as a mandatory dependency even though the Redis cache backend is
> optional in practice. This is because `service/cache.py` imports `redis.asyncio` unconditionally
> at the top of the module (not inside a conditional). If you want to make it truly optional, you
> would need to move the import inside `RedisCache.__init__` and add a guard in `_build_backend`.

### Optional extras (provider-gated)

| Extra | Package | Constraint notes |
|---|---|---|
| `anthropic` | `anthropic >= 0.76.0` | Claude 4.x adaptive thinking requires ≥ 0.76. `ThinkingConfigAdaptiveParam` type was added in that version. |
| `openai` | `openai >= 2.15` | `reasoning_effort` parameter requires ≥ 2.15; `max_completion_tokens` field also requires a recent version. |
| `google` | `google-genai >= 1.57.0` | Covers both Gemini and Vertex AI providers. |

### Feature extras

| Extra | Package | Notes |
|---|---|---|
| `agents` | `pydantic-ai >= 2.0` | v2.x broke the `Model` protocol; adapters target the v2 API specifically. `ModelRequestParameters`, `RequestUsage` are v2 types. |
| `workflows` | `langgraph >= 1.2.0` | `CompiledStateGraph` type signature changed in 1.2. |
| `observability` | `logfire >= 3.0`, `langfuse >= 3.0`, `opentelemetry-sdk >= 1.20`, `opentelemetry-exporter-otlp-proto-http >= 1.20` | All four must be present — `_check.py` verifies all three at once. |

### Python version support

`requires-python = ">=3.11"`. The codebase uses:
- `StrEnum` (3.11+)
- `from __future__ import annotations` (deferred evaluation — safe on 3.11+)
- `asyncio.timeout()` context manager (3.11+)
- `TypeVar` with `bound` constraints
- `X | Y` union syntax in type hints (3.10+)

Tests run against 3.11, 3.12, 3.13, and 3.14 via `just test-all`.

### Design constraints shaped by dependencies

1. **PydanticAI v2 protocol** — the `Model` protocol changed significantly between v1 and v2.
   Adapters use `ModelRequestParameters`, `StreamedResponse`, and `_get_event_iterator()` which
   are v2-only. Do not try to maintain v1 compatibility.

2. **Anthropic `temperature` + thinking incompatibility** — when thinking is enabled, `temperature`
   and `top_p` must be passed as `anthropic.omit`. This SDK constraint is encapsulated entirely
   inside `AnthropicProvider.complete()` and `AnthropicProvider.stream()`.

3. **Redis `asyncio` import** — `redis.asyncio` must not be used with `threading.Lock`. The
   `InMemoryCache` uses `asyncio.Lock` for the same reason. Never mix thread locks with async code
   in these modules.

---

## 9. Conventions

### 9.1 Coding style

- **Line length**: 100 characters (enforced by `ruff`)
- **Quote style**: double quotes
- **Ruff rules**: `E, F, I, UP, B, SIM, TCH` — this includes `TCH` (TYPE_CHECKING imports),
  so all imports used *only* for type hints must live inside `if TYPE_CHECKING:` blocks
- **`from __future__ import annotations`**: required in every module to enable deferred
  annotation evaluation
- **`assert` for internal invariants**: `assert self._client is not None` after
  `_assert_initialized()` — satisfies Pyright's narrowing without raising `RuntimeError`

### 9.2 Naming conventions

| Kind | Pattern | Example |
|---|---|---|
| Provider config classes | `{ProviderName}Config` | `AnthropicConfig`, `OpenAIConfig` |
| Provider classes | `{ProviderName}Provider` | `AnthropicProvider` |
| Private module-level constants | `_SCREAMING_SNAKE` | `_DEFAULT_MODEL`, `_TIKTOKEN_UNAVAILABLE` |
| Private instance helpers | `_snake_case` | `_build_response()`, `_split_messages()` |
| Schema classes | `{Domain}Schema` | `LLMRequestSchema`, `MessageSchema` |
| Internal modules | `_snake_case.py` prefix | `_internal/`, `_registry.py`, `_check.py` |

### 9.3 Module-level logger pattern

Every module that logs declares its logger at module scope:

```python
import logging
logger = logging.getLogger(__name__)
```

Never use `print()`. Levels:
- `DEBUG`: per-request details (model, token counts, cache key prefixes)
- `INFO`: lifecycle events (provider initialized, router shut down)
- `WARNING`: recoverable issues (circuit state transitions, cache errors, health check failures)
- `ERROR`: non-recoverable failures that still allow partial operation

### 9.4 Error handling conventions

- **Provider errors**: always map SDK-native exceptions to the `LLMError` hierarchy inside the
  provider's `_map_{provider}_error()` static method. Never let vendor SDK exceptions escape the
  provider layer.
- **Cache errors**: always catch and log as `WARNING`; never let a cache failure propagate to the
  inference path.
- **Shutdown errors**: catch per-provider in loops (`LLMService.shutdown`, `LLMRouter.shutdown`);
  log as `WARNING` and continue shutting down remaining providers.
- **Plugin discovery errors**: log as `WARNING` and skip; a broken third-party plugin must not
  prevent the library from loading.

### 9.5 Lifecycle pattern

All stateful objects follow the async context manager pattern:

```python
# Explicit lifecycle
provider = AnthropicProvider(config)
await provider.initialize()
try:
    response = await provider.complete(request)
finally:
    await provider.shutdown()

# Preferred: async context manager
async with AnthropicProvider(config) as provider:
    response = await provider.complete(request)
```

`LLMRouter` also supports `async with` via `__aenter__`/`__aexit__` (calls `shutdown()`).
`LLMCache` supports `async with` (calls `close()`).

### 9.6 Testing conventions

- **Test layout mirrors source layout**: `tests/unit/service/` tests `src/kitkat/service/`.
- **Default test suite**: `pytest tests/unit` (run by `just test-*`).
- **Integration tests**: marked with `@pytest.mark.integration`; skipped unless
  `INTEGRATION_TESTS=1` is set.
- **Mocking providers**: use `conftest.py`'s `mock_provider` fixture — an `AsyncMock` with
  `complete_with_retry`, `complete`, `health_check`, and `count_tokens` pre-stubbed, plus a
  valid `CAPABILITIES` descriptor.
- **asyncio mode**: `asyncio_mode = "auto"` in `pyproject.toml` — all `async def test_*`
  functions run automatically without `@pytest.mark.asyncio`.
- **No live API calls in unit tests**: provider tests mock the SDK client; integration tests in
  `tests/integration/` hit real endpoints.

### 9.7 Type checking

- **Primary type checker**: `mypy` with `strict = true`.
- **Secondary**: `basedpyright` (superset of pyright).
- **`TYPE_CHECKING` pattern**: expensive or optional imports that are only needed for type
  annotations go inside `if TYPE_CHECKING:` blocks. This is enforced by the `TCH` ruff rule.
- **`# type: ignore[...]`**: used sparingly and always with a specific error code. A comment
  explaining why is required if the reason isn't obvious from context.
- **`cast()`**: used for SDK return types where the SDK's own stubs are imprecise
  (e.g., `cast("Awaitable[AnthropicMessage]", client.messages.create(...))`).

### 9.8 Adding a new optional dependency

1. Add the extra to `[project.optional-dependencies]` in `pyproject.toml`.
2. Add a `require_{feature}_extra()` guard function in `agents/_check.py` (or a new `_check.py`
   in the relevant package).
3. Call `require_{feature}_extra()` at the **top of every module** in that feature group — before
   any imports that would fail if the package is absent.
4. Keep the import of the optional package inside the module body (not at `__init__.py` level)
   so `import kitkat` still works without it.

---

*Last updated: 2026-08-25 — reflects kitkat v0.8.0*
