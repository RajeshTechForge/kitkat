---
title: Core
description: This page documents every public type in `kitkat.core`. These types are the universal currency of the library — they flow between all layers (providers, service, agents) and carry no SDK dependencies.
order: 1
---

This page documents every public type in `kitkat.core`: data models, enumerations, and exceptions. These types are the universal currency of the library — they flow between all layers (providers, service, agents) and carry no SDK dependencies.

**Import path:** `from kitkat import ...` or `from kitkat.core.models import ...`

## Enumerations

All enumerations extend `StrEnum`, so their values are plain strings and can be compared directly with string literals.

### `ProviderType`

```python
from kitkat import ProviderType
```

| Member      | Value         | Description               |
| ----------- | ------------- | ------------------------- |
| `ANTHROPIC` | `"anthropic"` | Anthropic (Claude models) |
| `OPENAI`    | `"openai"`    | OpenAI (GPT models)       |
| `GEMINI`    | `"gemini"`    | Google Gemini             |

### `Role`

```python
from kitkat import Role
```

| Member      | Value         | Description                  |
| ----------- | ------------- | ---------------------------- |
| `SYSTEM`    | `"system"`    | System prompt / instructions |
| `USER`      | `"user"`      | End-user turn                |
| `ASSISTANT` | `"assistant"` | Model turn                   |

### `FinishReason`

```python
from kitkat.core.enums import FinishReason
```

| Member           | Value              | Description                           |
| ---------------- | ------------------ | ------------------------------------- |
| `STOP`           | `"stop"`           | Natural completion                    |
| `LENGTH`         | `"length"`         | Truncated at `max_tokens`             |
| `TOOL_CALL`      | `"tool_call"`      | Model requested tool execution        |
| `CONTENT_FILTER` | `"content_filter"` | Blocked by safety policy              |
| `ERROR`          | `"error"`          | Provider generation failure           |
| `UNKNOWN`        | `"unknown"`        | Fallback for unmapped provider values |

### `RoutingStrategy`

```python
from kitkat.core.enums import RoutingStrategy
```

| Member          | Value             | Description                                            |
| --------------- | ----------------- | ------------------------------------------------------ |
| `FAILOVER`      | `"failover"`      | Try providers in priority order; advance only on error |
| `ROUND_ROBIN`   | `"round_robin"`   | Cycle through healthy providers                        |
| `LEAST_LATENCY` | `"least_latency"` | Select the provider with lowest average latency        |
| `RANDOM`        | `"random"`        | Uniform random selection                               |

### `CircuitState`

```python
from kitkat.core.enums import CircuitState
```

| Member      | Value         | Description                                  |
| ----------- | ------------- | -------------------------------------------- |
| `CLOSED`    | `"CLOSED"`    | Normal operation                             |
| `OPEN`      | `"OPEN"`      | Blocking requests; awaiting recovery timeout |
| `HALF_OPEN` | `"HALF_OPEN"` | Allowing single recovery probe               |

### `CacheBackendType`

```python
from kitkat.core.enums import CacheBackendType
```

| Member   | Value      | Description                                           |
| -------- | ---------- | ----------------------------------------------------- |
| `MEMORY` | `"memory"` | In-process LRU (`OrderedDict`); no external deps      |
| `REDIS`  | `"redis"`  | `redis.asyncio` backend for multi-process deployments |

### `RoutingTier`

```python
from kitkat.core.enums import RoutingTier
```

Used inside `BaseAgentContext` to select the service path.

| Member       | Value          | Description                               |
| ------------ | -------------- | ----------------------------------------- |
| `MANAGED`    | `"managed"`    | Server-side key via `LLMService`          |
| `BYOK`       | `"byok"`       | Per-request user key via `BYOKLLMService` |
| `ENTERPRISE` | `"enterprise"` | Priority path; reserved for future use    |

## Data Models

### `Message`

```python
from kitkat import Message, Role

Message(role=Role.USER, content="Hello")
```

Frozen dataclass — immutable after creation.

| Field     | Type   | Description              |
| --------- | ------ | ------------------------ |
| `role`    | `Role` | Participant role         |
| `content` | `str`  | Text content of the turn |

**Method:** `to_dict() -> dict[str, str]` — serializes to `{"role": ..., "content": ...}`.

### `LLMRequest`

```python
from kitkat import LLMRequest, Message, Role

LLMRequest(messages=[Message(role=Role.USER, content="Hi")])
```

| Field            | Type                     | Default | Description                                                                    |
| ---------------- | ------------------------ | ------- | ------------------------------------------------------------------------------ |
| `messages`       | `list[Message]`          | —       | **Required.** Must contain at least one message.                               |
| `model`          | `str`                    | `""`    | Model identifier. Empty string uses provider's `DEFAULT_MODEL`.                |
| `max_tokens`     | `int`                    | `2048`  | Maximum completion tokens. Must be ≥ 1.                                        |
| `temperature`    | `float`                  | `0.1`   | Sampling temperature in `[0.0, 2.0]`.                                          |
| `top_p`          | `float`                  | `1.0`   | Nucleus sampling probability.                                                  |
| `stop_sequences` | `list[str]`              | `[]`    | Optional stop strings.                                                         |
| `stream`         | `bool`                   | `False` | Hint flag. Provider behaviour governed by calling `stream()` vs. `complete()`. |
| `timeout`        | `float \| None`          | `30.0`  | Per-request timeout in seconds. `None` = no timeout.                           |
| `thinking`       | `ThinkingConfig \| None` | `None`  | Extended reasoning config.                                                     |

**Raises `ValueError` at construction** if `messages` is empty, `temperature` is outside `[0.0, 2.0]`, or `max_tokens < 1`.

### `ThinkingConfig`

```python
from kitkat.core.models import ThinkingConfig
```

Frozen dataclass. Carries extended reasoning parameters through the domain layer.

| Field              | Type           | Default | Description                                                                           |
| ------------------ | -------------- | ------- | ------------------------------------------------------------------------------------- |
| `enabled`          | `bool`         | `False` | Activates thinking/reasoning mode                                                     |
| `effort`           | `str \| None`  | `None`  | Normalized effort: `"low"`, `"medium"`, `"high"`. Provider maps to native vocabulary. |
| `provider_options` | `dict \| None` | `None`  | Provider-specific overrides. Takes precedence over `effort` when both are set.        |

### `LLMResponse`

```python
from kitkat.core.models import LLMResponse
```

| Field              | Type           | Description                                                      |
| ------------------ | -------------- | ---------------------------------------------------------------- |
| `content`          | `str`          | Generated text                                                   |
| `finish_reason`    | `FinishReason` | Why generation stopped                                           |
| `usage`            | `TokenUsage`   | Token consumption breakdown                                      |
| `model`            | `str`          | Exact model version used (e.g. `"claude-opus-4-5-20251101"`)     |
| `provider`         | `ProviderType` | Which provider served the request                                |
| `thinking_content` | `str`          | Extended reasoning text. Empty string when thinking is disabled. |
| `latency_ms`       | `float`        | Wall-clock time from request dispatch to full response           |
| `raw_response`     | `Any`          | Unprocessed SDK response object. `None` for cached responses.    |

**Property:** `was_truncated: bool` — `True` when `finish_reason == FinishReason.LENGTH`.

### `StreamChunk`

```python
from kitkat.core.models import StreamChunk
```

One token delta from a streaming response.

| Field           | Type           | Default              | Description                                                                           |
| --------------- | -------------- | -------------------- | ------------------------------------------------------------------------------------- |
| `delta`         | `str`          | —                    | Token text. Empty string on the final sentinel chunk.                                 |
| `is_thinking`   | `bool`         | `False`              | `True` for extended reasoning tokens. All thinking chunks precede all content chunks. |
| `is_final`      | `bool`         | `False`              | `True` on the sentinel chunk that closes the stream.                                  |
| `finish_reason` | `FinishReason` | `UNKNOWN`            | Set on the final sentinel chunk.                                                      |
| `usage`         | `TokenUsage`   | `TokenUsage.empty()` | Aggregated usage. Set on the final sentinel chunk.                                    |
| `model`         | `str`          | `""`                 | Model version. Set on the final sentinel chunk.                                       |
| `provider`      | `ProviderType` | —                    | Provider that served the stream.                                                      |
| `latency_ms`    | `float`        | `0.0`                | Total stream duration. Set on the final sentinel chunk.                               |

**Ordering contract:** `is_thinking=True` chunks are always emitted before any `is_thinking=False` chunks.

### `TokenUsage`

```python
from kitkat.core.models import TokenUsage

total = usage_a + usage_b   # Addable
empty = TokenUsage.empty()  # Zero-valued instance
```

| Field               | Type  | Description                                                                  |
| ------------------- | ----- | ---------------------------------------------------------------------------- |
| `prompt_tokens`     | `int` | Input tokens                                                                 |
| `completion_tokens` | `int` | Output (answer) tokens only — excludes thinking tokens                       |
| `thinking_tokens`   | `int` | Extended reasoning tokens. `0` when not reported separately by the provider. |
| `total_tokens`      | `int` | `prompt + completion + thinking`                                             |

**Class method:** `TokenUsage.empty() -> TokenUsage` — returns a zero-valued instance.  
**Operator:** `__add__` aggregates two `TokenUsage` instances field-by-field.

### `RetryPolicy`

```python
from kitkat.core.models import RetryPolicy
```

| Field                    | Type             | Default                     | Description                                        |
| ------------------------ | ---------------- | --------------------------- | -------------------------------------------------- |
| `max_attempts`           | `int`            | `3`                         | Total number of attempts (including the first)     |
| `base_delay_s`           | `float`          | `1.0`                       | Initial backoff in seconds                         |
| `max_delay_s`            | `float`          | `60.0`                      | Maximum delay cap                                  |
| `exponential_base`       | `float`          | `2.0`                       | Backoff multiplier per attempt                     |
| `jitter`                 | `bool`           | `True`                      | Adds ±50% random jitter to prevent thundering herd |
| `retryable_status_codes` | `frozenset[int]` | `{408,429,500,502,503,504}` | HTTP codes that qualify for retry                  |

**Method:** `delay_for_attempt(attempt: int) -> float` — returns the sleep duration before the given attempt (0-indexed).

### `ProviderCapabilities`

```python
from kitkat.core.models import ProviderCapabilities
```

Frozen dataclass. Queried by the router when selecting providers.

| Field                    | Type           | Default | Description                             |
| ------------------------ | -------------- | ------- | --------------------------------------- |
| `supports_streaming`     | `bool`         | `True`  | Provider can stream token deltas        |
| `supports_system_prompt` | `bool`         | `True`  | Provider accepts system-role messages   |
| `supports_tool_calling`  | `bool`         | `False` | Provider supports function/tool calling |
| `supports_vision`        | `bool`         | `False` | Provider accepts image inputs           |
| `supports_thinking`      | `bool`         | `False` | Provider supports extended reasoning    |
| `max_context_tokens`     | `int`          | `8_192` | Maximum context window in tokens        |
| `provider_type`          | `ProviderType` | —       | Canonical provider identifier           |

## Exceptions

All exceptions are importable from the `kitkat` package top level.

```python
from kitkat import (
    KitkatError,
    LLMError,
    LLMProviderInitError,
    LLMProviderError,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMTokenLimitError,
    LLMContentFilterError,
)
```

### Hierarchy

```
KitkatError (message, code, details, status_code)
└── LLMError (message, provider, status_code)
    ├── LLMProviderInitError
    ├── LLMProviderError
    ├── LLMAuthenticationError
    ├── LLMRateLimitError      (+ retry_after_s)
    ├── LLMTimeoutError        (+ elapsed_s)
    ├── LLMTokenLimitError     (+ token_count, context_limit)
    └── LLMContentFilterError
```

### Exception Attributes

| Exception                | Extra Attributes                                     | HTTP Status | Retried?              |
| ------------------------ | ---------------------------------------------------- | ----------- | --------------------- |
| `KitkatError`            | `code: str`, `details: dict\|None`                   | 500         | —                     |
| `LLMError`               | `provider: str\|None`                                | 500         | —                     |
| `LLMProviderInitError`   | —                                                    | 500/401     | ❌ (startup only)     |
| `LLMProviderError`       | —                                                    | varies      | ✅ on retryable codes |
| `LLMAuthenticationError` | —                                                    | 401         | ❌ Never              |
| `LLMRateLimitError`      | `retry_after_s: float\|None`                         | 429         | ✅ Yes                |
| `LLMTimeoutError`        | `elapsed_s: float\|None`                             | 504         | ✅ Yes                |
| `LLMTokenLimitError`     | `token_count: int\|None`, `context_limit: int\|None` | 413         | ❌ Never              |
| `LLMContentFilterError`  | —                                                    | 400         | ❌ Never              |

## Further Reading

- [Error Handling](../error-handling.md) — Patterns for catching and mapping exceptions
- [Custom Providers](../custom-provider.md) — Mapping SDK errors to Kitkat exceptions
- [API Reference — Service](./service.md) — `LLMService`, `LLMRouter`, `BYOKLLMService`
