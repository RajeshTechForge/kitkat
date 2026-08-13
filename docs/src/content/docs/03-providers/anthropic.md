---
title: Anthropic
description: Complete reference for Kitkat's Anthropic provider, including installation, configuration, model selection, system prompt handling, streaming, extended thinking, token counting, error mapping, and retry policy.
order: 1
---

This page is the complete reference for Kitkat's Anthropic provider. It covers installation, every configuration field, model selection, system prompt handling, streaming, extended thinking in both adaptive and fixed-budget modes, exact token counting, the full error mapping, and a summary of the retry policy.

> **📝 Note:** This page assumes you have read [Concepts](../concepts.md) and understand `LLMRequest`, `LLMResponse`, and `StreamChunk`. If not, start there first.

## Installation

```bash
pip install kitkat[anthropic]
```

This installs the `anthropic` Python SDK (≥ 0.76.0) alongside Kitkat's core package.

## Quick Start

```python
import asyncio
import os

from kitkat.service import create_llm_service
from kitkat import ProviderType, LLMRequest, Message, Role
from kitkat.providers.anthropic import AnthropicProvider, AnthropicConfig

async def main() -> None:
    config = AnthropicConfig(api_key=os.environ["ANTHROPIC_API_KEY"])
    service = create_llm_service({ProviderType.ANTHROPIC: AnthropicProvider(config)})
    await service.initialize()

    response = await service.complete(
        LLMRequest(
            messages=[Message(role=Role.USER, content="Explain the GIL in one sentence.")],
            model="claude-opus-4-5",
            max_tokens=128,
        ),
        ProviderType.ANTHROPIC,
    )
    print(response.content)
    # The GIL (Global Interpreter Lock) is a mutex in CPython that allows only
    # one thread to execute Python bytecode at a time, preventing true parallelism
    # in CPU-bound multi-threaded programs.

asyncio.run(main())
```

## `AnthropicConfig`

`AnthropicConfig` is a dataclass that holds all configuration for the Anthropic provider. Every field is validated in `__post_init__` — invalid values raise `LLMProviderInitError` immediately at object construction, before any network calls are made.

```python
from kitkat.providers.anthropic import AnthropicConfig
import os

config = AnthropicConfig(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    model="claude-sonnet-4-6",
    base_url=None,
    max_retries=0,
    timeout_s=30.0,
    extra_headers={},
)
```

### Fields

| Field           | Type             | Default               | Description                                                                                                                                            |
| --------------- | ---------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `api_key`       | `str`            | —                     | **Required.** Your Anthropic API key. Must be a non-empty string.                                                                                      |
| `model`         | `str`            | `"claude-sonnet-4-6"` | The default model identifier. Used when `LLMRequest.model` is empty.                                                                                   |
| `base_url`      | `str \| None`    | `None`                | Override the API base URL. Useful for Anthropic-compatible proxies or enterprise gateways. When `None`, the SDK default (`api.anthropic.com`) is used. |
| `max_retries`   | `int`            | `0`                   | Number of SDK-level automatic retries. Keep at `0` so Kitkat's own `RetryPolicy` has exclusive control over the retry schedule.                        |
| `timeout_s`     | `float`          | `30.0`                | Per-request wall-clock timeout in seconds. Applied via `asyncio.wait_for`. Overridden per-request by `LLMRequest.timeout`.                             |
| `extra_headers` | `dict[str, str]` | `{}`                  | Arbitrary HTTP headers injected into every request. Useful for tracing IDs or custom gateway auth headers.                                             |

### Validation rules

- `api_key` must be a non-empty, non-whitespace string. An empty string raises `LLMProviderInitError("AnthropicConfig.api_key must be a non-empty string.")`.
- `timeout_s` must be positive (> 0). Zero or negative values raise `LLMProviderInitError`.

### Building from a dictionary

```python
config = AnthropicConfig.from_dict({
    "api_key": os.environ["ANTHROPIC_API_KEY"],
    "model": "claude-opus-4-5",
    "timeout_s": 60.0,
    "extra_headers": {"X-Request-ID": "my-trace-id"},
})
```

## `AnthropicProvider`

`AnthropicProvider` wraps `AnthropicConfig` and implements the `LLMProvider` ABC. It uses the official `anthropic.AsyncAnthropic` client internally.

```python
from kitkat.providers.anthropic import AnthropicProvider, AnthropicConfig
import os

provider = AnthropicProvider(
    AnthropicConfig(api_key=os.environ["ANTHROPIC_API_KEY"])
)
# Or pass a dict directly — it is coerced into AnthropicConfig automatically:
provider = AnthropicProvider({"api_key": os.environ["ANTHROPIC_API_KEY"]})
```

### Class-level attributes

| Attribute                             | Value                    |
| ------------------------------------- | ------------------------ |
| `PROVIDER_TYPE`                       | `ProviderType.ANTHROPIC` |
| `DEFAULT_MODEL`                       | `"claude-sonnet-4-6"`    |
| `CAPABILITIES.supports_streaming`     | `True`                   |
| `CAPABILITIES.supports_system_prompt` | `True`                   |
| `CAPABILITIES.supports_tool_calling`  | `True`                   |
| `CAPABILITIES.supports_vision`        | `True`                   |
| `CAPABILITIES.supports_thinking`      | `True`                   |
| `CAPABILITIES.max_context_tokens`     | `200_000`                |

## Lifecycle

### `async initialize()`

Opens the `AsyncAnthropic` HTTP client and validates credentials via a lightweight `messages.count_tokens` probe (sends `[{"role": "user", "content": "ping"}]`). The probe consumes no inference tokens and times out after 5 seconds.

- Raises `LLMAuthenticationError` if the API key is invalid.
- Raises `LLMProviderInitError` if the HTTP client cannot be created.
- Is idempotent: calling `initialize()` a second time on an already-initialized provider is a no-op.

```python
provider = AnthropicProvider(config)
await provider.initialize()
# Provider is now ready to serve requests.
```

### Using as an async context manager

```python
async with AnthropicProvider(config) as provider:
    response = await provider.complete(request)
# shutdown() is called automatically on exit.
```

### `async shutdown()`

Closes the `AsyncAnthropic` HTTP connection pool and marks the provider as uninitialized. Safe to call if the provider was never initialized — it is a no-op in that case.

## Completions

### Non-streaming

```python
import asyncio
import os

from kitkat.providers.anthropic import AnthropicProvider, AnthropicConfig
from kitkat import LLMRequest, Message, Role

async def main() -> None:
    async with AnthropicProvider(AnthropicConfig(api_key=os.environ["ANTHROPIC_API_KEY"])) as provider:
        request = LLMRequest(
            messages=[
                Message(role=Role.SYSTEM, content="Answer in exactly one sentence."),
                Message(role=Role.USER, content="What is a Python generator?"),
            ],
            model="claude-opus-4-5",
            max_tokens=128,
            temperature=0.2,
            top_p=1.0,
            stop_sequences=["\n"],  # Stop at the first newline
            timeout=20.0,
        )
        response = await provider.complete(request)

        print(response.content)
        print(f"Model: {response.model}")
        print(f"Finish reason: {response.finish_reason}")
        print(f"Prompt tokens: {response.usage.prompt_tokens}")
        print(f"Completion tokens: {response.usage.completion_tokens}")
        print(f"Total tokens: {response.usage.total_tokens}")
        print(f"Latency: {response.latency_ms:.0f} ms")
        print(f"Truncated: {response.was_truncated}")

asyncio.run(main())
```

> **📝 Note:** `provider.complete()` executes a single attempt with no retry. Use `provider.complete_with_retry(request)` or route through `LLMService` to apply the built-in `RetryPolicy`.

### Streaming

```python
import asyncio
import os

from kitkat.providers.anthropic import AnthropicProvider, AnthropicConfig
from kitkat import LLMRequest, Message, Role

async def main() -> None:
    async with AnthropicProvider(AnthropicConfig(api_key=os.environ["ANTHROPIC_API_KEY"])) as provider:
        request = LLMRequest(
            messages=[Message(role=Role.USER, content="Write a haiku about coroutines.")],
            model="claude-opus-4-5",
            max_tokens=64,
            stream=True,
        )

        answer_parts: list[str] = []

        async for chunk in provider.stream(request):
            if chunk.is_thinking:
                # Thinking chunks arrive first — skip them in this example.
                continue

            if not chunk.is_final:
                answer_parts.append(chunk.delta)
                print(chunk.delta, end="", flush=True)
            else:
                # Final sentinel chunk carries usage and latency.
                print()
                print(f"Finish reason: {chunk.finish_reason}")
                print(f"Total tokens: {chunk.usage.total_tokens}")
                print(f"Latency: {chunk.latency_ms:.0f} ms")

asyncio.run(main())
```

## System Prompt Handling

Anthropic's API treats the system prompt as a **separate top-level parameter** (`system=`) rather than an element in the messages list. Kitkat handles this automatically.

When you include `Message(role=Role.SYSTEM, ...)` objects in your message list, Kitkat:

1. Extracts all system messages from the list.
2. Concatenates their content with `\n\n---\n\n` as a separator (for multi-system-message scenarios).
3. Passes the result as the `system=` parameter to `messages.create()`.
4. Passes the remaining non-system messages as the `messages=` parameter.

```python
from kitkat import Message, Role

messages = [
    Message(role=Role.SYSTEM, content="You are a Python expert."),
    Message(role=Role.SYSTEM, content="Always show code examples."),
    Message(role=Role.USER, content="Explain list comprehensions."),
]

# What Kitkat sends to Anthropic:
# system="You are a Python expert.\n\n---\n\nAlways show code examples."
# messages=[{"role": "user", "content": "Explain list comprehensions."}]
```

> **💡 Tip:** While Anthropic supports multiple system messages via this concatenation, it is best practice to keep your system instructions in a single `Role.SYSTEM` message for clarity.

## Extended Thinking

Anthropic's Claude supports two thinking modes: **adaptive** and **enabled** (fixed budget). Both are controlled via `ThinkingConfig` in your `LLMRequest`.

### Adaptive mode (recommended)

In adaptive mode, the model decides when and how much to think based on the task. You control the thinking intensity with the `effort` field.

```python
from kitkat import LLMRequest, Message, Role, ThinkingConfig

request = LLMRequest(
    messages=[
        Message(role=Role.USER, content="What is 17 × 24 + 38 / 2? Show your work.")
    ],
    model="claude-opus-4-5",
    thinking=ThinkingConfig(
        enabled=True,
        effort="high",   # "low", "medium", or "high". Default: "high"
    ),
    max_tokens=1024,
)
```

Kitkat maps `effort` → Anthropic's `output_config: {effort: ...}` parameter.

### Fixed-budget mode

In fixed-budget mode, the model always thinks using exactly `budget_tokens` tokens. Use this when you want predictable thinking costs.

```python
request = LLMRequest(
    messages=[Message(role=Role.USER, content="Prove by induction that 1+2+...+n = n(n+1)/2.")],
    model="claude-opus-4-5",
    thinking=ThinkingConfig(
        enabled=True,
        provider_options={
            "thinking_type": "enabled",  # Force fixed-budget mode
            "budget_tokens": 8000,       # Exactly 8000 thinking tokens
        },
    ),
    max_tokens=2048,
)
```

Kitkat maps `provider_options` → Anthropic's `thinking: {type: "enabled", budget_tokens: ...}`.

### Accessing thinking output

Thinking content is available in `LLMResponse.thinking_content` for non-streaming requests and in `StreamChunk.delta` (with `is_thinking=True`) for streaming requests.

```python
response = await provider.complete(request)
if response.thinking_content:
    print("=== Thinking ===")
    print(response.thinking_content)
print("=== Answer ===")
print(response.content)
```

For streaming:

```python
thinking_buf: list[str] = []
answer_buf: list[str] = []

async for chunk in provider.stream(request):
    if chunk.is_final:
        break
    if chunk.is_thinking:
        thinking_buf.append(chunk.delta)
    else:
        answer_buf.append(chunk.delta)

print("Thinking:", "".join(thinking_buf))
print("Answer:", "".join(answer_buf))
```

### Thinking and temperature

> **⚠️ Warning:** Anthropic requires that `temperature` and `top_p` are **not sent** when extended thinking is enabled. Kitkat automatically omits these parameters from the API call when `ThinkingConfig.enabled=True`. You do not need to remove them from your `LLMRequest` manually.

### Token counting with thinking

Anthropic does not expose separate thinking-token and answer-token counts in its usage response. `output_tokens` covers both. Kitkat reflects this faithfully:

- `TokenUsage.thinking_tokens` is always `0` for Anthropic responses.
- `TokenUsage.completion_tokens` equals the total output tokens (thinking + answer combined).
- `TokenUsage.total_tokens` = `prompt_tokens` + `completion_tokens`.

## Token Counting

### Fast local estimate

`count_tokens(text)` uses tiktoken's `cl100k_base` BPE encoding locally, with no network call. This is a fast approximation — suitable for pre-flight budget checks.

```python
provider = AnthropicProvider(config)
await provider.initialize()

estimate = provider.count_tokens("Hello, world!")
print(estimate)  # 4

# Estimate a full conversation
from kitkat import Message, Role
messages = [
    Message(role=Role.SYSTEM, content="You are a helpful assistant."),
    Message(role=Role.USER, content="Explain the GIL."),
]
estimate = provider.count_prompt_tokens(messages)
print(estimate)  # ~15
```

**Fallback:** If tiktoken's BPE data cannot be downloaded (e.g., air-gapped environments), Kitkat falls back to a character-based estimate: `max(1, len(text) // 4)`.

### Exact count via Anthropic API

`async_count_tokens(request)` calls Anthropic's `messages.count_tokens` endpoint for the exact prompt token count. This makes a real API call but costs no inference tokens.

```python
from kitkat import LLMRequest, Message, Role

exact = await provider.async_count_tokens(
    LLMRequest(
        messages=[
            Message(role=Role.SYSTEM, content="You are a helpful assistant."),
            Message(role=Role.USER, content="Explain the GIL."),
        ]
    )
)
print(exact)  # Exact value from Anthropic
```

Use `async_count_tokens` when you need to gate requests against a strict token budget (e.g., reject requests that would exceed `max_context_tokens`).

## Health Check

`health_check()` sends a lightweight `messages.count_tokens("ping")` call to verify the provider is reachable. It times out after 5 seconds.

```python
is_healthy = await provider.health_check()
print(is_healthy)  # True or False
```

Returns `False` (rather than raising) on any error, including network failures and authentication errors. This makes it safe to use in monitoring loops.

## Retry Policy

The Anthropic provider's class-level `RETRY_POLICY` is:

| Parameter          | Value                            | Rationale                                                      |
| ------------------ | -------------------------------- | -------------------------------------------------------------- |
| `max_attempts`     | `3`                              | 1 original + 2 retries covers most transient blips             |
| `base_delay_s`     | `1.0`                            | Short initial delay for fast recovery on brief outages         |
| `max_delay_s`      | `60.0`                           | Cap prevents multi-minute waits in degraded scenarios          |
| `exponential_base` | `2.0`                            | Standard doubling: 1s → 2s → 4s (before jitter)                |
| `jitter`           | `True`                           | ±50% random variation prevents thundering-herd on shared infra |
| Retryable codes    | `{408, 429, 500, 502, 503, 504}` | Standard transient HTTP codes                                  |

The following errors bypass retry entirely and are raised immediately:

- `LLMAuthenticationError` (401, 403) — a different attempt will not fix the credentials.
- `LLMTokenLimitError` — the prompt is too long for any number of retries.
- `LLMContentFilterError` — content policy blocks do not resolve on retry.

## Error Mapping

Every Anthropic SDK exception is mapped to a specific Kitkat `LLMError` subclass:

| Anthropic SDK exception              | Kitkat exception         | Notes                                                  |
| ------------------------------------ | ------------------------ | ------------------------------------------------------ |
| `anthropic.AuthenticationError`      | `LLMAuthenticationError` | Invalid or revoked API key                             |
| `anthropic.PermissionDeniedError`    | `LLMAuthenticationError` | Key lacks permission for the operation                 |
| `anthropic.RateLimitError`           | `LLMRateLimitError`      | Parses `Retry-After` header into `retry_after_s`       |
| `anthropic.APITimeoutError`          | `LLMTimeoutError`        | SDK-level timeout                                      |
| `asyncio.TimeoutError`               | `LLMTimeoutError`        | `asyncio.wait_for` timeout (from `LLMRequest.timeout`) |
| `anthropic.APIConnectionError`       | `LLMProviderError`       | Network-level connection failure                       |
| `anthropic.NotFoundError`            | `LLMProviderError`       | Model or resource not found                            |
| `anthropic.BadRequestError`          | `LLMProviderError`       | Malformed request                                      |
| `anthropic.UnprocessableEntityError` | `LLMProviderError`       | Input format or parameter issue                        |
| `anthropic.InternalServerError`      | `LLMProviderError`       | Covers `OverloadedError` and `ServiceUnavailableError` |
| `anthropic.APIStatusError`           | `LLMProviderError`       | Catch-all for other HTTP status errors                 |

## `stop_reason` → `FinishReason` Mapping

| Anthropic `stop_reason` | `FinishReason` |
| ----------------------- | -------------- |
| `"end_turn"`            | `STOP`         |
| `"stop_sequence"`       | `STOP`         |
| `"max_tokens"`          | `LENGTH`       |
| `"tool_use"`            | `TOOL_CALL`    |
| `"pause_turn"`          | `UNKNOWN`      |
| `None`                  | `UNKNOWN`      |

## Further Reading

- [Providers Overview](../providers.md) — `LLMService` API and provider comparison table
- [Concepts](../concepts.md) — `LLMRequest`, `LLMResponse`, `ThinkingConfig` reference
- [Error Handling](../error-handling.md) — Full exception handling guide
- [API Reference — Providers](../api-reference/providers.md) — Complete API surface
