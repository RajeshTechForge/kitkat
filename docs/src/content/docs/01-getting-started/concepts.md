---
title: Concepts
description: The fundamental building blocks of Kitkat, including request/response types, service paths, and provider capabilities.
order: 3
---

This page explains the fundamental building blocks of Kitkat: what each type is, why it exists, and how the pieces fit together at runtime. Reading this page is not required to follow the Quick Start, but it will make every other page in the documentation much easier to understand.

## The Two Service Paths

Every Kitkat application uses one of two service paths. Understanding the distinction upfront prevents confusion when choosing which classes to instantiate.

### Managed path

In the managed path, API keys live on your server. You configure one or more providers once, and all requests use those shared credentials.

```
Your code
  └─→  LLMService  ──→  Provider (Anthropic / OpenAI / Google)  ──→  LLM API
```

Use the managed path when:

- Your application calls LLMs on behalf of the user using your own API account.
- You want centralized rate-limit handling, caching, and routing.

### BYOK path

In the BYOK (Bring Your Own Key) path, each request carries its own API key supplied by the end user. No shared credentials exist on the server.

```
User request + user API key
  └─→  BYOKLLMService  ──→  Provider (ephemeral client)  ──→  LLM API
```

Use the BYOK path when:

- You are building a multi-tenant SaaS product where users connect their own LLM accounts.
- You must never store or share a user's API key beyond the request boundary.

## Core Types

This section covers every type exported from the top-level `kitkat` namespace. All of these are always available regardless of which extras are installed.

### `Role`

`Role` is a `StrEnum` that identifies the participant in a conversation turn. It has exactly three values:

| Value            | String        | Meaning                                                                       |
| ---------------- | ------------- | ----------------------------------------------------------------------------- |
| `Role.SYSTEM`    | `"system"`    | Instructions that condition the model's behaviour for the entire conversation |
| `Role.USER`      | `"user"`      | A message from the human participant                                          |
| `Role.ASSISTANT` | `"assistant"` | A message from the model (used when replying with prior conversation history) |

```python
from kitkat import Role

# All three values are plain strings due to StrEnum.
assert Role.USER == "user"
assert Role.SYSTEM == "system"
assert Role.ASSISTANT == "assistant"
```

`Role` has no dependencies and is importable without any extras installed.

### `Message`

`Message` is a frozen dataclass representing one turn in a conversation. It is immutable after construction.

```python
from kitkat import Message, Role

system_msg = Message(role=Role.SYSTEM, content="You are a helpful assistant.")
user_msg   = Message(role=Role.USER,   content="What is the capital of France?")
```

**Fields:**

| Field     | Type   | Description                           |
| --------- | ------ | ------------------------------------- |
| `role`    | `Role` | The participant role for this message |
| `content` | `str`  | The text content of the message       |

**Methods:**

`Message.to_dict() -> dict[str, str]` — Serializes the message to `{"role": "<value>", "content": "<text>"}`. Used internally by provider adapters but also useful for logging or debugging.

```python
user_msg.to_dict()
# {"role": "user", "content": "What is the capital of France?"}
```

> **📝 Note:** `Message` is frozen (`frozen=True`), which means its fields cannot be changed after construction and the object is hashable. This ensures that message lists passed to `LLMRequest` cannot be mutated by accident.

### `LLMRequest`

`LLMRequest` is the single object you build to describe everything a provider needs to fulfil a completion or streaming request. All providers accept exactly this type — you never construct a provider-specific request object directly.

```python
from kitkat import LLMRequest, Message, Role

request = LLMRequest(
    messages=[
        Message(role=Role.SYSTEM, content="Be concise."),
        Message(role=Role.USER,   content="Explain polymorphism."),
    ],
    model="claude-opus-4-5",    # Provider-specific model string; "" defers to provider default
    max_tokens=512,              # Maximum completion tokens (default: 2048)
    temperature=0.7,             # Sampling temperature in [0.0, 2.0] (default: 0.1)
    top_p=1.0,                   # Nucleus sampling probability (default: 1.0)
    stop_sequences=["\n\n"],     # Sequences that stop generation (default: [])
    stream=False,                # True to receive tokens incrementally (default: False)
    timeout=30.0,                # Per-request timeout in seconds; None = no timeout (default: 30.0)
    thinking=None,               # ThinkingConfig for extended reasoning (default: None)
)
```

**Validation rules enforced in `__post_init__`:**

- `messages` must contain at least one item. An empty list raises `ValueError`.
- `temperature` must be in the closed interval `[0.0, 2.0]`. Values outside this range raise `ValueError`.
- `max_tokens` must be ≥ 1. Zero or negative values raise `ValueError`.

**Why these defaults?**

- `max_tokens=2048` — A safe middle ground that prevents runaway token usage while allowing reasonably long answers.
- `temperature=0.1` — Low temperature produces focused, deterministic output by default. Raise it for creative tasks.
- `timeout=30.0` — A 30-second ceiling prevents indefinitely stalled requests from blocking event-loop tasks.
- `model=""` — An empty string signals "use the provider's default model", which each provider implementation resolves.

### `ThinkingConfig`

`ThinkingConfig` is a frozen dataclass that enables and configures extended reasoning for providers that support it (currently Anthropic Claude and OpenAI o-series models).

```python
from kitkat import LLMRequest, Message, Role, ThinkingConfig

request = LLMRequest(
    messages=[Message(role=Role.USER, content="Solve this step by step: 17 × 24")],
    thinking=ThinkingConfig(
        enabled=True,
        effort="medium",   # "low", "medium", or "high"
    ),
)
```

**Fields:**

| Field              | Type                                    | Default | Description                                                                                                            |
| ------------------ | --------------------------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------- |
| `enabled`          | `bool`                                  | `False` | Whether to activate extended reasoning for this request                                                                |
| `effort`           | `str \| None`                           | `None`  | Normalized effort level. Providers map this to their native vocabulary. `None` defers to the provider's default effort |
| `provider_options` | `dict[str, str \| int \| None] \| None` | `None`  | Raw provider-specific overrides. When set, takes precedence over `effort`                                              |

> **📝 Note:** When both `effort` and `provider_options` are set, `provider_options` takes precedence. Use `provider_options` only when you need settings that the normalized `effort` field cannot express.

### `LLMResponse`

`LLMResponse` is the object returned by every non-streaming `complete()` call. Its shape is identical regardless of which provider fulfilled the request.

```python
from kitkat import LLMResponse, FinishReason, ProviderType

# You never construct LLMResponse yourself; it comes back from service.complete().
response: LLMResponse = await service.complete(request, ProviderType.ANTHROPIC)

print(response.content)           # The model's text answer
print(response.thinking_content)  # Reasoning trace (empty string if thinking was off)
print(response.model)             # Model string reported by the provider
print(response.provider)          # ProviderType.ANTHROPIC
print(response.usage.total_tokens)
print(response.latency_ms)        # Wall-clock request time in milliseconds
print(response.was_truncated)     # True if finish_reason == FinishReason.LENGTH
```

**Fields:**

| Field              | Type           | Description                                               |
| ------------------ | -------------- | --------------------------------------------------------- |
| `content`          | `str`          | The model's answer text                                   |
| `finish_reason`    | `FinishReason` | Why the model stopped generating                          |
| `usage`            | `TokenUsage`   | Token counts for the request                              |
| `model`            | `str`          | Model identifier as reported by the provider              |
| `provider`         | `ProviderType` | Which provider fulfilled the request                      |
| `thinking_content` | `str`          | Extended-reasoning trace; `""` when thinking was disabled |
| `latency_ms`       | `float`        | Wall-clock request duration in milliseconds               |
| `raw_response`     | `Any`          | Unmodified SDK response object; excluded from `repr()`    |

**Properties:**

`was_truncated -> bool` — Returns `True` when `finish_reason` is `FinishReason.LENGTH`, indicating the output was cut short by the `max_tokens` limit.

### `StreamChunk`

`StreamChunk` is emitted by the `stream()` method one token at a time. Each chunk carries a delta (the new text fragment) and metadata.

```python
from kitkat import StreamChunk

async for chunk in service.stream(request, ProviderType.ANTHROPIC):
    # chunk.delta: the new text fragment (may be empty for the final sentinel)
    # chunk.is_thinking: True while extended-reasoning tokens are streaming
    # chunk.is_final: True on the last chunk; carries usage and finish_reason
    # chunk.finish_reason: why the model stopped (only meaningful on is_final=True)
    # chunk.usage: token counts (only meaningful on is_final=True)
    # chunk.latency_ms: request latency (only meaningful on is_final=True)
    pass
```

**Ordering contract:**

1. All thinking chunks (`is_thinking=True`) are emitted first.
2. Answer chunks (`is_thinking=False`) follow.
3. The transition from thinking to answer is one-way and never interleaved.
4. The very last chunk always has `is_final=True` and `is_thinking=False`.

### `TokenUsage`

`TokenUsage` tracks token consumption for a single provider call. Every `LLMResponse` and every final `StreamChunk` carries a `TokenUsage` instance.

```python
from kitkat import TokenUsage

usage = TokenUsage(
    prompt_tokens=120,
    completion_tokens=48,
    thinking_tokens=0,   # 0 when provider doesn't separate thinking tokens
    total_tokens=168,
)

# TokenUsage supports addition for aggregating across calls.
total = TokenUsage(prompt_tokens=100, completion_tokens=40, total_tokens=140)
total += usage
print(total.total_tokens)  # 308
```

> **📝 Note:** `completion_tokens` counts answer tokens only. It excludes thinking tokens even when extended reasoning is enabled. `total_tokens` always equals `prompt_tokens + completion_tokens + thinking_tokens`.

### `RetryPolicy`

`RetryPolicy` configures the exponential back-off strategy that the service layer applies before raising an exception on transient errors.

```python
from kitkat import RetryPolicy

policy = RetryPolicy(
    max_attempts=3,          # Total attempts (1 original + 2 retries). Default: 3
    base_delay_s=1.0,        # Seconds to wait before the first retry. Default: 1.0
    max_delay_s=60.0,        # Maximum wait cap. Default: 60.0
    exponential_base=2.0,    # Multiplier per attempt. Default: 2.0
    jitter=True,             # Add ±50% random variation to prevent thundering herd. Default: True
)
```

The delay before attempt `n` (0-indexed) is calculated as:

```
delay = min(base_delay_s × exponential_base^n, max_delay_s)
# With jitter=True: delay × random(0.5, 1.0)
```

The default retryable HTTP status codes are: `408`, `429`, `500`, `502`, `503`, `504`. Errors with other status codes (e.g. `401`, `400`) are not retried.

### `ProviderCapabilities`

`ProviderCapabilities` is a frozen dataclass that describes what a specific provider and model combination supports. The router queries capabilities when selecting a provider for a request.

```python
from kitkat import ProviderCapabilities, ProviderType

caps = ProviderCapabilities(
    supports_streaming=True,
    supports_system_prompt=True,
    supports_tool_calling=True,
    supports_vision=False,
    supports_thinking=True,
    max_context_tokens=200_000,
    provider_type=ProviderType.ANTHROPIC,
)
```

You typically read capabilities from an existing provider rather than constructing them yourself:

```python
caps = provider.get_capabilities()
if caps.supports_streaming:
    async for chunk in service.stream(request, ProviderType.ANTHROPIC):
        ...
```

## Enums Reference

Kitkat uses `StrEnum` throughout. All enum values are plain strings, so they serialize cleanly to JSON and compare equal to their string equivalents.

### `ProviderType`

| Member      | Value         | Meaning                                           |
| ----------- | ------------- | ------------------------------------------------- |
| `ANTHROPIC` | `"anthropic"` | Anthropic Claude models                           |
| `OPENAI`    | `"openai"`    | OpenAI GPT models and OpenAI-compatible endpoints |
| `GOOGLE`    | `"google"`    | Google Gemini models and Vertex AI                |

### `FinishReason`

| Member           | Value              | Meaning                                    |
| ---------------- | ------------------ | ------------------------------------------ |
| `STOP`           | `"stop"`           | Model reached a natural stopping point     |
| `LENGTH`         | `"length"`         | Output was truncated by `max_tokens`       |
| `TOOL_CALL`      | `"tool_call"`      | Model is requesting a tool execution       |
| `CONTENT_FILTER` | `"content_filter"` | Response was blocked by a safety filter    |
| `ERROR`          | `"error"`          | Provider-side generation failure           |
| `UNKNOWN`        | `"unknown"`        | Fallback for unmapped or unexpected values |

### `RoutingStrategy`

| Member          | Value             | Meaning                                                       |
| --------------- | ----------------- | ------------------------------------------------------------- |
| `FAILOVER`      | `"failover"`      | Always try providers in priority order; advance only on error |
| `ROUND_ROBIN`   | `"round_robin"`   | Cycle through healthy providers in insertion order            |
| `LEAST_LATENCY` | `"least_latency"` | Pick the provider with the lowest average response latency    |
| `RANDOM`        | `"random"`        | Uniformly random selection from the healthy provider pool     |

### `CircuitState`

| Member      | Value         | Meaning                                                          |
| ----------- | ------------- | ---------------------------------------------------------------- |
| `CLOSED`    | `"CLOSED"`    | Normal operation — requests are forwarded                        |
| `OPEN`      | `"OPEN"`      | Provider is considered unhealthy; requests are blocked           |
| `HALF_OPEN` | `"HALF_OPEN"` | One test probe is allowed to check if the provider has recovered |

### `CacheBackendType`

| Member   | Value      | Meaning                                                                |
| -------- | ---------- | ---------------------------------------------------------------------- |
| `MEMORY` | `"memory"` | In-process LRU cache — suitable for single-process deployments         |
| `REDIS`  | `"redis"`  | Async Redis — suitable for multi-process or multi-instance deployments |

### `RoutingTier`

| Member       | Value          | Meaning                                                          |
| ------------ | -------------- | ---------------------------------------------------------------- |
| `MANAGED`    | `"managed"`    | Agent uses the managed service path with server-side API keys    |
| `BYOK`       | `"byok"`       | Agent uses `BYOKLLMService` with a per-request user-supplied key |
| `ENTERPRISE` | `"enterprise"` | Managed path with a priority queue (reserved for future use)     |

## Exception Hierarchy

All Kitkat exceptions inherit from a single base so you can write catch-all handlers or narrow handlers as needed.

```
KitkatError
└── LLMError
    ├── LLMProviderInitError   # Provider failed to initialize
    ├── LLMProviderError       # Generic provider-side failure
    ├── LLMRateLimitError      # HTTP 429 or quota exceeded
    ├── LLMTimeoutError        # Request exceeded timeout
    ├── LLMTokenLimitError     # Request exceeds model context window
    ├── LLMContentFilterError  # Response blocked by safety policy
    └── LLMAuthenticationError # Invalid or missing API credentials
```

Every exception carries `.message`, `.status_code`, and `.provider` attributes. Some subclasses carry additional fields:

| Exception            | Extra fields                                                             |
| -------------------- | ------------------------------------------------------------------------ |
| `LLMRateLimitError`  | `retry_after_s: float \| None` — seconds to wait, if provided by the API |
| `LLMTimeoutError`    | `elapsed_s: float \| None` — how long the request ran before timing out  |
| `LLMTokenLimitError` | `token_count: int \| None`, `context_limit: int \| None`                 |

## The Request / Response Lifecycle

Here is the complete path a request takes from your code to the LLM API and back:

```
1. You build LLMRequest and call service.complete(request, provider_type)
   │
2. LLMService looks up the registered LLMProvider for that ProviderType
   │
3. LLMService applies RetryPolicy — wraps the call in a retry loop
   │
4. The LLMProvider translates LLMRequest to a native SDK call
   │
5. Provider makes the HTTP request via httpx
   │
6. Provider maps the SDK response to LLMResponse (or StreamChunk iterator)
   │
7. On transient error: sleep(delay) and retry up to max_attempts
   On non-retryable error: raise the appropriate LLMError subclass
   │
8. LLMResponse is returned to your code
```

For streaming, step 6 returns an `AsyncIterator[StreamChunk]` instead of a single `LLMResponse`. The caller pulls from that iterator token by token.

## Further Reading

- [Quick Start](./quickstart.md) — Practical examples for every path described above
- [Providers](./providers.md) — Provider-specific config and model strings
- [Routing & Cache](./routing-cache.md) — How `RoutingStrategy` and `CircuitState` work in practice
- [Error Handling](./error-handling.md) — Complete exception handling guide with retry configuration
- [API Reference — Core](./api-reference/core.md) — Full API surface with all parameters and return types
