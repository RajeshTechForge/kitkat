---
title: OpenAI
description: Complete reference for KitKat's OpenAI provider, including installation, configuration, model selection, endpoints, streaming, extended thinking, token counting, error mapping, and retry policy.
order: 2
---

This page is the complete reference for KitKat's OpenAI provider. It covers installation, every configuration field, model selection, OpenAI-compatible endpoints, streaming, extended thinking for o-series models, token counting, the full error mapping, and a summary of the retry policy.

> **📝 Note:** This page assumes you have read [Concepts](../concepts.md). If not, start there first.

---

## Installation

```bash
pip install kitkat[openai]
```

This installs the `openai` Python SDK (≥ 2.15) alongside KitKat's core package.

---

## Quick Start

```python
import asyncio
import os

from kitkat.service import create_llm_service
from kitkat import ProviderType, LLMRequest, Message, Role
from kitkat.providers.openai import OpenAIProvider, OpenAIConfig

async def main() -> None:
    config = OpenAIConfig(api_key=os.environ["OPENAI_API_KEY"])
    service = create_llm_service({ProviderType.OPENAI: OpenAIProvider(config)})
    await service.initialize()

    response = await service.complete(
        LLMRequest(
            messages=[Message(role=Role.USER, content="What is a Python decorator?")],
            model="gpt-4o",
            max_tokens=256,
        ),
        ProviderType.OPENAI,
    )
    print(response.content)

asyncio.run(main())
```

---

## `OpenAIConfig`

`OpenAIConfig` is a dataclass that holds all configuration for the OpenAI provider. All fields are validated in `__post_init__` before any network calls are made.

```python
from kitkat.providers.openai import OpenAIConfig
import os

config = OpenAIConfig(
    api_key=os.environ["OPENAI_API_KEY"],
    model="gpt-4o-mini",
    base_url=None,
    max_retries=0,
    timeout_s=60.0,
    extra_headers={},
    organization=None,
)
```

### Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `api_key` | `str` | — | **Required.** Your OpenAI API key. Must be a non-empty string. |
| `model` | `str` | `"gpt-4o-mini"` | The default model identifier. Used when `LLMRequest.model` is empty. |
| `base_url` | `str \| None` | `None` | Override the API base URL. Use for NVIDIA NIM, Azure OpenAI, self-hosted vLLM, or any OpenAI-compatible proxy. When `None`, the SDK default (`api.openai.com`) is used. |
| `max_retries` | `int` | `0` | SDK-level automatic retries. Keep at `0` so KitKat's `RetryPolicy` has full control. |
| `timeout_s` | `float` | `60.0` | Per-request wall-clock timeout in seconds. The default is 60 s (higher than Anthropic's 30 s default) because OpenAI requests, especially with reasoning models, can take longer. Overridden per-request by `LLMRequest.timeout`. |
| `extra_headers` | `dict[str, str]` | `{}` | Arbitrary HTTP headers injected into every request. Useful for tracing (`X-Request-ID`) or gateway auth. |
| `organization` | `str \| None` | `None` | OpenAI organization ID. Ignored by non-OpenAI endpoints. |

### Validation rules

- `api_key` must be a non-empty, non-whitespace string.
- `timeout_s` must be positive.
- `max_retries` must be ≥ 0.

### Building from a dictionary

```python
config = OpenAIConfig.from_dict({
    "api_key": os.environ["OPENAI_API_KEY"],
    "model": "gpt-4o",
    "timeout_s": 90.0,
    "organization": "org-abc123",
})
```

---

## `OpenAIProvider`

`OpenAIProvider` wraps `OpenAIConfig` and implements the `LLMProvider` ABC using the official `openai.AsyncOpenAI` client.

```python
from kitkat.providers.openai import OpenAIProvider, OpenAIConfig
import os

provider = OpenAIProvider(OpenAIConfig(api_key=os.environ["OPENAI_API_KEY"]))
# Or pass a dict directly:
provider = OpenAIProvider({"api_key": os.environ["OPENAI_API_KEY"]})
```

### Class-level attributes

| Attribute | Value |
|---|---|
| `PROVIDER_TYPE` | `ProviderType.OPENAI` |
| `DEFAULT_MODEL` | `"gpt-4o-mini"` |
| `CAPABILITIES.supports_streaming` | `True` |
| `CAPABILITIES.supports_system_prompt` | `True` |
| `CAPABILITIES.supports_tool_calling` | `True` |
| `CAPABILITIES.supports_vision` | `True` |
| `CAPABILITIES.supports_thinking` | `True` (o-series models) |
| `CAPABILITIES.max_context_tokens` | `128_000` |

---

## OpenAI-Compatible Endpoints

The `OpenAIProvider` works with any endpoint that implements the OpenAI Chat Completions API specification. Set `base_url` in `OpenAIConfig` to target alternative endpoints.

### NVIDIA NIM

```python
from kitkat.providers.openai import OpenAIProvider, OpenAIConfig
import os

nvidia_provider = OpenAIProvider(OpenAIConfig(
    api_key=os.environ["NVIDIA_API_KEY"],
    base_url="https://integrate.api.nvidia.com/v1",
    model="nvidia/llama-3.1-70b-instruct",
    timeout_s=120.0,   # NIM can be slower for large models
))
```

### Azure OpenAI

```python
azure_provider = OpenAIProvider(OpenAIConfig(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    base_url="https://my-resource.openai.azure.com/openai/deployments/my-deployment",
    model="",   # Azure ignores the model field — it is baked into the endpoint URL
))
```

### Self-hosted vLLM

```python
vllm_provider = OpenAIProvider(OpenAIConfig(
    api_key="not-used",   # vLLM does not require authentication by default
    base_url="http://localhost:8000/v1",
    model="meta-llama/Llama-3-8b-instruct",
))
```

> **📝 Note:** The health check probe (`models.list()`) may not be supported by all OpenAI-compatible endpoints. If the probe fails during `initialize()`, a `LLMProviderInitError` is raised. You can work around this by calling `_init_client_only()` manually (used by `BYOKLLMService`), but this skips the credential validation.

---

## Lifecycle

### `async initialize()`

Opens the `AsyncOpenAI` HTTP client and probes credentials by calling `models.list()` with an 8-second timeout. This call lists available models and consumes no inference tokens.

- Raises `LLMAuthenticationError` if the API key is invalid (401 / 403).
- Raises `LLMProviderInitError` if the HTTP client cannot be created or the probe fails.
- Is idempotent: calling it twice is a no-op.

### Using as an async context manager

```python
async with OpenAIProvider(config) as provider:
    response = await provider.complete(request)
```

### `async shutdown()`

Closes the `AsyncOpenAI` HTTP connection pool and marks the provider as uninitialized.

---

## Completions

### Non-streaming

```python
import asyncio
import os

from kitkat.providers.openai import OpenAIProvider, OpenAIConfig
from kitkat import LLMRequest, Message, Role

async def main() -> None:
    async with OpenAIProvider(OpenAIConfig(api_key=os.environ["OPENAI_API_KEY"])) as provider:
        request = LLMRequest(
            messages=[
                Message(role=Role.SYSTEM, content="You are a concise assistant. Reply in bullet points."),
                Message(role=Role.USER, content="What are the benefits of async Python?"),
            ],
            model="gpt-4o",
            max_tokens=256,
            temperature=0.5,
            top_p=0.95,
            stop_sequences=["---"],
            timeout=45.0,
        )
        response = await provider.complete(request)

        print(response.content)
        print(f"Model: {response.model}")
        print(f"Finish reason: {response.finish_reason}")
        print(f"Prompt tokens: {response.usage.prompt_tokens}")
        print(f"Completion tokens: {response.usage.completion_tokens}")
        print(f"Thinking tokens: {response.usage.thinking_tokens}")  # 0 for non-reasoning models
        print(f"Latency: {response.latency_ms:.0f} ms")

asyncio.run(main())
```

> **📝 Note:** KitKat uses `max_completion_tokens` (rather than the older `max_tokens`) when calling the Chat Completions API. This is required by OpenAI's o-series reasoning models and is forward-compatible with standard GPT models.

### Streaming

```python
import asyncio
import os

from kitkat.providers.openai import OpenAIProvider, OpenAIConfig
from kitkat import LLMRequest, Message, Role

async def main() -> None:
    async with OpenAIProvider(OpenAIConfig(api_key=os.environ["OPENAI_API_KEY"])) as provider:
        request = LLMRequest(
            messages=[Message(role=Role.USER, content="Write a limerick about Python.")],
            model="gpt-4o",
            max_tokens=128,
            stream=True,
        )

        async for chunk in provider.stream(request):
            if not chunk.is_final:
                print(chunk.delta, end="", flush=True)
            else:
                print()
                print(f"Finish: {chunk.finish_reason}")
                print(f"Tokens: {chunk.usage.total_tokens}")
                print(f"Latency: {chunk.latency_ms:.0f} ms")

asyncio.run(main())
```

---

## System Prompt Handling

OpenAI supports system prompts inline as a standard `{"role": "system", "content": "..."}` message in the conversation list. KitKat serializes all `Message` objects verbatim using `Message.to_dict()` — no extraction or restructuring is performed.

```python
from kitkat import Message, Role

messages = [
    Message(role=Role.SYSTEM, content="You are a helpful assistant."),
    Message(role=Role.USER, content="Explain decorators."),
    Message(role=Role.ASSISTANT, content="A decorator is a function that wraps another function..."),
    Message(role=Role.USER, content="Give me an example."),
]
# Sent to OpenAI as-is:
# [
#   {"role": "system", "content": "You are a helpful assistant."},
#   {"role": "user", "content": "Explain decorators."},
#   {"role": "assistant", "content": "A decorator is a function that wraps..."},
#   {"role": "user", "content": "Give me an example."},
# ]
```

This makes it straightforward to include multi-turn conversation history in your requests.

---

## Extended Thinking (o-series models)

OpenAI's o-series models (o1, o3, o4-mini, o4, etc.) support extended reasoning via the `reasoning_effort` parameter. KitKat maps the normalized `ThinkingConfig` to this parameter.

```python
from kitkat import LLMRequest, Message, Role, ThinkingConfig

# Standard effort levels
request = LLMRequest(
    messages=[Message(role=Role.USER, content="Solve: ∫x² dx from 0 to 3")],
    model="o4-mini",
    thinking=ThinkingConfig(
        enabled=True,
        effort="medium",   # "low", "medium", or "high" — maps to reasoning_effort
    ),
    max_tokens=2048,
)

# Provider-level override
request_high = LLMRequest(
    messages=[Message(role=Role.USER, content="Prove the Riemann hypothesis (attempt).")],
    model="o3",
    thinking=ThinkingConfig(
        enabled=True,
        provider_options={"effort": "high"},   # takes precedence over effort field
    ),
    max_tokens=8192,
)
```

**Effort → `reasoning_effort` mapping:**

| `ThinkingConfig.effort` | OpenAI `reasoning_effort` |
|---|---|
| `"low"` | `"low"` |
| `"medium"` | `"medium"` |
| `"high"` | `"high"` |

> **📝 Note:** For standard GPT models (gpt-4o, gpt-4o-mini, etc.), `ThinkingConfig` is ignored — these models do not support `reasoning_effort`. Only o-series models expose separate reasoning tokens.

### Accessing reasoning tokens

For o-series models, `TokenUsage.thinking_tokens` reflects `completion_tokens_details.reasoning_tokens` from the OpenAI response. `completion_tokens` is the answer-only token count (total completion minus reasoning).

```python
response = await provider.complete(o_series_request)
print(f"Reasoning tokens: {response.usage.thinking_tokens}")
print(f"Answer tokens: {response.usage.completion_tokens}")
print(f"Total: {response.usage.total_tokens}")
```

---

## Token Counting

`count_tokens(text)` uses tiktoken's `cl100k_base` BPE encoding, which is the shared encoding base for GPT-4, GPT-3.5-turbo, and most NVIDIA NIM models. The count is a local approximation with no network call.

```python
provider = OpenAIProvider(config)
await provider.initialize()

# Estimate a single string
estimate = provider.count_tokens("What is the capital of France?")
print(estimate)  # 7

# Estimate a full conversation
from kitkat import Message, Role
messages = [
    Message(role=Role.SYSTEM, content="You are a helpful assistant."),
    Message(role=Role.USER, content="What is the capital of France?"),
]
estimate = provider.count_prompt_tokens(messages)
print(estimate)  # ~15
```

**Fallback:** If tiktoken cannot load its BPE data (air-gapped environments), the estimate falls back to `max(1, len(text) // 4)`.

> **💡 Tip:** Unlike the Anthropic provider, the OpenAI provider does not expose an `async_count_tokens` method for exact server-side token counts. Use `count_tokens` for pre-flight estimates and rely on `response.usage` for exact counts after the call.

---

## Health Check

`health_check()` calls `models.list()` with an 8-second timeout to verify the provider is reachable and the key is valid.

```python
is_healthy = await provider.health_check()
print(is_healthy)  # True or False
```

Returns `False` on any error (network failure, auth failure, timeout) rather than raising an exception.

---

## Retry Policy

| Parameter | Value | Rationale |
|---|---|---|
| `max_attempts` | `3` | 1 original + 2 retries |
| `base_delay_s` | `1.0` | Short initial delay |
| `max_delay_s` | `60.0` | Cap for degraded scenarios |
| `exponential_base` | `2.0` | Doubles per attempt: 1s → 2s → 4s |
| `jitter` | `True` | ±50% random variation |
| Retryable codes | `{408, 429, 500, 502, 503, 504}` | Standard transient codes |

The following errors are **not** retried:
- `LLMAuthenticationError` (401, 403)
- `LLMTokenLimitError` (context window exceeded)
- `LLMContentFilterError` (content policy block)

---

## Error Mapping

Every OpenAI SDK exception is mapped to a specific KitKat `LLMError` subclass:

| OpenAI SDK exception | KitKat exception | Notes |
|---|---|---|
| `AuthenticationError` | `LLMAuthenticationError` | Invalid or revoked API key |
| `PermissionDeniedError` | `LLMAuthenticationError` | Key lacks permission for the operation |
| `RateLimitError` | `LLMRateLimitError` | Parses `Retry-After` header into `retry_after_s` |
| `APITimeoutError` | `LLMTimeoutError` | SDK-level timeout |
| `asyncio.TimeoutError` | `LLMTimeoutError` | `asyncio.wait_for` timeout |
| `APIConnectionError` | `LLMProviderError` | Network-level connection failure |
| `NotFoundError` | `LLMProviderError` | Model or resource not found (404) |
| `ConflictError` | `LLMProviderError` | Resource conflict (409) |
| `BadRequestError` | `LLMProviderError` | Malformed request (400) |
| `UnprocessableEntityError` | `LLMProviderError` | Input validation failed (422) |
| `InternalServerError` | `LLMProviderError` | OpenAI server-side error (5xx) |
| `APIResponseValidationError` | `LLMProviderError` | SDK failed to parse the response |
| Any other `APIError` | `LLMProviderError` | Catch-all for unexpected API errors |

---

## `finish_reason` → `FinishReason` Mapping

| OpenAI `finish_reason` | `FinishReason` |
|---|---|
| `"stop"` | `STOP` |
| `"length"` | `LENGTH` |
| `"tool_calls"` | `TOOL_CALL` |
| `"function_call"` | `TOOL_CALL` |
| `"content_filter"` | `CONTENT_FILTER` |
| `None` | `UNKNOWN` |

---

## Further Reading

- [Providers Overview](../providers.md) — `LLMService` API and provider comparison table
- [Concepts](../concepts.md) — `LLMRequest`, `LLMResponse`, `ThinkingConfig` reference
- [Routing & Cache](../routing-cache.md) — Route across multiple providers
- [BYOK](../byok.md) — Per-request user API keys with `BYOKLLMService`
- [API Reference — Providers](../api-reference/providers.md) — Complete API surface
