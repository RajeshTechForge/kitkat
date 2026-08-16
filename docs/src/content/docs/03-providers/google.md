---
title: Google
description: Complete reference for Kitkat's Google provider, including installation, configuration, model selection, system prompt handling, streaming, extended thinking, token counting, error mapping, and retry policy.
order: 3
---

This page is the complete reference for Kitkat's Google provider. It covers installation, every configuration field, both API-key and Vertex AI modes, model selection, system prompt handling, streaming, extended thinking, exact token counting, the full error mapping, and the retry policy.

> **📝 Note:** This page assumes you have read [Concepts](../concepts.md). If not, start there first.

## Installation

```bash
pip install kitkat[google]
```

This installs the `google-genai` Python SDK (≥ 1.57.0) alongside Kitkat's core package.

## Quick Start

```python
import asyncio
import os

from kitkat.service import create_llm_service
from kitkat import ProviderType, LLMRequest, Message, Role
from kitkat.providers.google import GoogleProvider, GoogleConfig

async def main() -> None:
    config = GoogleConfig(api_key=os.environ["GOOGLE_API_KEY"])
    service = create_llm_service({ProviderType.GOOGLE: GoogleProvider(config)})
    await service.initialize()

    response = await service.complete(
        LLMRequest(
            messages=[Message(role=Role.USER, content="What is a Python context manager?")],
            model="gemini-3-flash-preview",
            max_tokens=256,
        ),
        ProviderType.GOOGLE,
    )
    print(response.content)

asyncio.run(main())
```

## `GoogleConfig`

`GoogleConfig` is a dataclass that holds all configuration for the Google provider. It supports two distinct authentication modes: **API key** (standard) and **Vertex AI**. All fields are validated in `__post_init__`.

### API key mode (standard)

```python
from kitkat.providers.google import GoogleConfig
import os

config = GoogleConfig(
    api_key=os.environ["GOOGLE_API_KEY"],  # Required when vertexai=False
    model="gemini-3-flash-preview",         # Default: "gemini-3-flash-preview"
    vertexai=False,                         # Default: False
    timeout_s=60.0,                         # Default: 60.0
    extra_headers={},                       # Default: {}
)
```

### Vertex AI mode

```python
vertex_config = GoogleConfig(
    vertexai=True,
    project=os.environ["GOOGLE_CLOUD_PROJECT"],  # Required when vertexai=True
    location="us-central1",                       # Required when vertexai=True
    model="gemini-3-flash-preview",
    timeout_s=60.0,
)
# Note: api_key is not used in Vertex AI mode.
# Authentication uses Application Default Credentials (ADC).
```

### Fields

| Field           | Type             | Default                    | Description                                                                     |
| --------------- | ---------------- | -------------------------- | ------------------------------------------------------------------------------- |
| `api_key`       | `str`            | `""`                       | Your Google API key. Required when `vertexai=False`. Ignored in Vertex AI mode. |
| `model`         | `str`            | `"gemini-3-flash-preview"` | The default model identifier. Used when `LLMRequest.model` is empty.            |
| `vertexai`      | `bool`           | `False`                    | When `True`, uses Vertex AI with ADC instead of the standard API.               |
| `project`       | `str`            | `""`                       | GCP project ID. Required when `vertexai=True`.                                  |
| `location`      | `str`            | `""`                       | GCP region (e.g., `"us-central1"`). Required when `vertexai=True`.              |
| `timeout_s`     | `float`          | `60.0`                     | Per-request wall-clock timeout in seconds. Overridden by `LLMRequest.timeout`.  |
| `extra_headers` | `dict[str, str]` | `{}`                       | Arbitrary HTTP headers injected into every request.                             |

### Validation rules

- When `vertexai=False`: `api_key` must be a non-empty, non-whitespace string.
- When `vertexai=True`: both `project` and `location` must be non-empty strings.
- `timeout_s` must be positive.

### Building from a dictionary

```python
config = GoogleConfig.from_dict({
    "api_key": os.environ["GOOGLE_API_KEY"],
    "model": "gemini-2.5-pro",
    "timeout_s": 90.0,
})
```

## `GoogleProvider`

`GoogleProvider` wraps `GoogleConfig` and implements the `LLMProvider` ABC using the official `google.genai.Client`.

```python
from kitkat.providers.google import GoogleProvider, GoogleConfig
import os

provider = GoogleProvider(GoogleConfig(api_key=os.environ["GOOGLE_API_KEY"]))
# Or pass a dict directly:
provider = GoogleProvider({"api_key": os.environ["GOOGLE_API_KEY"]})
```

### Class-level attributes

| Attribute                             | Value                      |
| ------------------------------------- | -------------------------- |
| `PROVIDER_TYPE`                       | `ProviderType.GOOGLE`      |
| `DEFAULT_MODEL`                       | `"gemini-3-flash-preview"` |
| `CAPABILITIES.supports_streaming`     | `True`                     |
| `CAPABILITIES.supports_system_prompt` | `True`                     |
| `CAPABILITIES.supports_tool_calling`  | `True`                     |
| `CAPABILITIES.supports_vision`        | `True`                     |
| `CAPABILITIES.supports_thinking`      | `True`                     |
| `CAPABILITIES.max_context_tokens`     | `1_048_576` (1M+)          |

## Vertex AI Support

Kitkat's Google provider supports Vertex AI deployments transparently. When `vertexai=True`, the `google-genai` SDK uses Application Default Credentials (ADC) rather than an API key.

**Setting up ADC:**

```bash
# Authenticate with your Google account (for local development)
gcloud auth application-default login

# Or set the service account key path (for production)
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

**Using Vertex AI with Kitkat:**

```python
import os
import asyncio
from kitkat.providers.google import GoogleProvider, GoogleConfig
from kitkat import ProviderType, LLMRequest, Message, Role
from kitkat.service import create_llm_service

async def main() -> None:
    config = GoogleConfig(
        vertexai=True,
        project=os.environ["GOOGLE_CLOUD_PROJECT"],
        location="us-central1",
        model="gemini-3-flash-preview",
    )
    service = create_llm_service({ProviderType.GOOGLE: GoogleProvider(config)})
    await service.initialize()

    response = await service.complete(
        LLMRequest(messages=[Message(role=Role.USER, content="Hello from Vertex AI!")]),
        ProviderType.GOOGLE,
    )
    print(response.content)

asyncio.run(main())
```

> **📝 Note:** The BYOK service path (`BYOKLLMService`) does not support Vertex AI mode, because Vertex AI uses ADC rather than a user-supplied API key. Use the managed service path for Vertex AI deployments.

## Lifecycle

### `async initialize()`

Constructs the `google.genai.Client` and runs a credential probe via `aio.models.count_tokens(model=..., contents="ping")`. The probe consumes no inference tokens and times out after 5 seconds.

**Authentication error handling:** If the probe returns HTTP 401 or 403, `LLMProviderInitError` is raised immediately. Other probe errors (network issues, etc.) are logged as warnings but do not block initialization — Google's API can be temporarily inconsistent during startup.

- Raises `LLMProviderInitError` if the client cannot be created or credentials fail.
- Is idempotent: calling it twice on an already-initialized provider is a no-op.

### Using as an async context manager

```python
async with GoogleProvider(config) as provider:
    response = await provider.complete(request)
```

### `async shutdown()`

Closes both the async and sync Google client connections, releasing the underlying HTTP connection pool.

## Completions

### Non-streaming

```python
import asyncio
import os

from kitkat.providers.google import GoogleProvider, GoogleConfig
from kitkat import LLMRequest, Message, Role

async def main() -> None:
    async with GoogleProvider(GoogleConfig(api_key=os.environ["GOOGLE_API_KEY"])) as provider:
        request = LLMRequest(
            messages=[
                Message(role=Role.SYSTEM, content="Answer concisely in one paragraph."),
                Message(role=Role.USER, content="Explain the transformer architecture."),
            ],
            model="gemini-3-flash-preview",
            max_tokens=512,
            temperature=0.4,
            top_p=0.95,
            stop_sequences=["\n\n"],
            timeout=30.0,
        )
        response = await provider.complete(request)

        print(response.content)
        print(f"Model: {response.model}")
        print(f"Finish reason: {response.finish_reason}")
        print(f"Prompt tokens: {response.usage.prompt_tokens}")
        print(f"Completion tokens: {response.usage.completion_tokens}")
        print(f"Thinking tokens: {response.usage.thinking_tokens}")
        print(f"Latency: {response.latency_ms:.0f} ms")

asyncio.run(main())
```

> **📝 Note:** `top_p` is only sent to the Google API when it differs from `1.0`. When `top_p=1.0` (the default), Kitkat omits it from the API call to avoid overriding Google's own default nucleus sampling settings.

### Streaming

```python
import asyncio
import os

from kitkat.providers.google import GoogleProvider, GoogleConfig
from kitkat import LLMRequest, Message, Role

async def main() -> None:
    async with GoogleProvider(GoogleConfig(api_key=os.environ["GOOGLE_API_KEY"])) as provider:
        request = LLMRequest(
            messages=[Message(role=Role.USER, content="Tell me a short story about a robot.")],
            model="gemini-3-flash-preview",
            max_tokens=256,
            stream=True,
        )

        async for chunk in provider.stream(request):
            if chunk.is_thinking:
                continue  # Skip thinking tokens
            if not chunk.is_final:
                print(chunk.delta, end="", flush=True)
            else:
                print()
                print(f"Finish: {chunk.finish_reason}")
                print(f"Tokens: {chunk.usage.total_tokens}")
                print(f"Latency: {chunk.latency_ms:.0f} ms")

asyncio.run(main())
```

## System Prompt Handling

Google uses a dedicated `system_instruction` top-level parameter separate from the conversation turns. Kitkat handles the extraction automatically.

When you include `Message(role=Role.SYSTEM, ...)` objects in your message list, Kitkat:

1. Extracts all system messages from the list.
2. Concatenates their content with `\n\n---\n\n` as a separator.
3. Passes the result as `system_instruction` in `GenerateContentConfig`.
4. Maps remaining messages to `genai_types.Content` objects with Google's role vocabulary.

**Role mapping:**

| Kitkat `Role`    | Google role                       |
| ---------------- | --------------------------------- |
| `Role.USER`      | `"user"`                          |
| `Role.ASSISTANT` | `"model"`                         |
| `Role.SYSTEM`    | Extracted to `system_instruction` |

```python
from kitkat import Message, Role

messages = [
    Message(role=Role.SYSTEM, content="You are a concise technical assistant."),
    Message(role=Role.USER, content="What is a tensor?"),
    Message(role=Role.ASSISTANT, content="A tensor is a multi-dimensional array..."),
    Message(role=Role.USER, content="Give me an example in PyTorch."),
]
# What Kitkat sends to Google:
# system_instruction="You are a concise technical assistant."
# contents=[
#   Content(role="user", parts=[Part(text="What is a tensor?")]),
#   Content(role="model", parts=[Part(text="A tensor is a multi-dimensional array...")]),
#   Content(role="user", parts=[Part(text="Give me an example in PyTorch.")]),
# ]
```

## Extended Thinking

Google supports extended thinking via the `thinking_level` parameter (`"LOW"`, `"MEDIUM"`, `"HIGH"`). Kitkat maps the normalized `ThinkingConfig.effort` field to Google's vocabulary.

### Effort → thinking level mapping

| `ThinkingConfig.effort` | Google `thinking_level` |
| ----------------------- |-------------------------|
| `"low"`                 | `"LOW"`                 |
| `"medium"`              | `"MEDIUM"`              |
| `"high"`                | `"HIGH"`                |

```python
from kitkat import LLMRequest, Message, Role, ThinkingConfig

request = LLMRequest(
    messages=[Message(role=Role.USER, content="Solve: d/dx [x³ sin(x)]")],
    model="gemini-3-flash-preview",
    thinking=ThinkingConfig(
        enabled=True,
        effort="medium",   # Maps to: thinking_level="MEDIUM"
    ),
    max_tokens=1024,
)
```

### Provider-level override

Use `provider_options` to pass Google-specific parameters directly. The `level` key maps directly to `thinking_level`:

```python
request = LLMRequest(
    messages=[Message(role=Role.USER, content="What are the prime factors of 1729?")],
    model="gemini-3-flash-preview",
    thinking=ThinkingConfig(
        enabled=True,
        provider_options={"level": "HIGH"},   # Maps directly to thinking_level="HIGH"
    ),
    max_tokens=512,
)
```

### Thinking in streaming

Google streaming distinguishes thinking and answer parts via the `thought` attribute on `Part` objects. Kitkat maps this to `StreamChunk.is_thinking`:

```python
thinking_buf: list[str] = []
answer_buf: list[str] = []

async for chunk in provider.stream(thinking_request):
    if chunk.is_final:
        break
    if chunk.is_thinking:
        thinking_buf.append(chunk.delta)
    else:
        answer_buf.append(chunk.delta)

print("Thinking:", "".join(thinking_buf))
print("Answer:", "".join(answer_buf))
```

### Thinking token reporting

Unlike Anthropic, Google reports thinking tokens separately in `usage_metadata.thoughts_token_count`. Kitkat maps this to `TokenUsage.thinking_tokens`:

```python
response = await provider.complete(thinking_request)
print(f"Thinking tokens: {response.usage.thinking_tokens}")
print(f"Answer tokens: {response.usage.completion_tokens}")
print(f"Total: {response.usage.total_tokens}")
```

> **📝 Note:** When `include_thoughts=True` (set automatically by Kitkat when thinking is enabled), Google includes the reasoning trace in both `thinking_content` (non-streaming) and as `is_thinking=True` stream chunks (streaming).

## Safety Filters

Google applies safety filters across multiple categories. When a response is blocked, Kitkat raises `LLMContentFilterError`.

### Filter categories that trigger `LLMContentFilterError`

| Google `finish_reason` | Category                                      |
| ---------------------- | --------------------------------------------- |
| `SAFETY`               | General safety policy violation               |
| `RECITATION`           | Copyright or recitation block                 |
| `BLOCKLIST`            | Blocked term list                             |
| `PROHIBITED_CONTENT`   | Prohibited content category                   |
| `SPII`                 | Sensitive personally identifiable information |
| `IMAGE_SAFETY`         | Image content safety (multimodal)             |

### Behaviour

**Non-streaming:** `LLMContentFilterError` is raised from `complete()` when `finish_reason` maps to `CONTENT_FILTER`.

**Streaming:** `LLMContentFilterError` is raised after the stream completes when the overall `finish_reason` is `CONTENT_FILTER`. Partial tokens emitted before the filter triggered are not returned.

```python
from kitkat import LLMContentFilterError

try:
    response = await provider.complete(request)
except LLMContentFilterError as exc:
    print(f"Blocked: {exc.message}")
    # Handle blocked content — e.g., return a safe default response
```

> **⚠️ Warning:** Content filter errors are **not retried** — the same content would be blocked on any subsequent attempt. Catch them explicitly and provide a user-facing error message rather than retrying.

## Token Counting

### Fast local estimate

`count_tokens(text)` uses tiktoken's `cl100k_base` BPE encoding as a fast approximation with no network call.

```python
provider = GoogleProvider(config)
await provider.initialize()

estimate = provider.count_tokens("What is deep learning?")
print(estimate)  # ~5

from kitkat import Message, Role
messages = [
    Message(role=Role.SYSTEM, content="You are a helpful assistant."),
    Message(role=Role.USER, content="Explain transformers."),
]
estimate = provider.count_prompt_tokens(messages)
print(estimate)  # ~12
```

**Fallback:** If tiktoken cannot load BPE data, the estimate falls back to `max(1, len(text) // 4)`.

### Exact count via Google API

`async_count_tokens(request)` calls `aio.models.count_tokens` for the exact prompt token count. No inference tokens are consumed.

```python
from kitkat import LLMRequest, Message, Role

exact = await provider.async_count_tokens(
    LLMRequest(
        messages=[
            Message(role=Role.SYSTEM, content="You are a helpful assistant."),
            Message(role=Role.USER, content="Explain transformers."),
        ]
    )
)
print(exact)  # Exact value from Google
```

> **💡 Tip:** With Gemini's 1M+ context window, token budget management is less critical than with Anthropic or OpenAI. However, `async_count_tokens` is still useful when processing very large documents to avoid unexpected truncation.

## Health Check

`health_check()` calls `aio.models.count_tokens(model=..., contents="ping")` with a 5-second timeout to verify the provider is reachable.

```python
is_healthy = await provider.health_check()
print(is_healthy)  # True or False
```

Returns `False` on any error rather than raising.

## Retry Policy

| Parameter          | Value                            | Rationale                                          |
| ------------------ | -------------------------------- | -------------------------------------------------- |
| `max_attempts`     | `3`                              | 1 original + 2 retries                             |
| `base_delay_s`     | `2.0`                            | Extended base delay for quota-limited environments |
| `max_delay_s`      | `60.0`                           | Cap for degraded scenarios                         |
| `exponential_base` | `2.0`                            | Doubles per attempt: 2s → 4s → 8s (before jitter)  |
| `jitter`           | `True`                           | ±50% random variation                              |
| Retryable codes    | `{408, 429, 500, 502, 503, 504}` | Standard transient codes                           |

> **📝 Note:** The Google provider's `base_delay_s` is `2.0` (vs `1.0` for Anthropic and OpenAI). Google quota limits are enforced on a per-minute basis, and a 2-second base delay gives the quota window more time to reset before the first retry.

The following errors are **not** retried:

- `LLMAuthenticationError` (401, 403)
- `LLMTokenLimitError` (context exceeded)
- `LLMContentFilterError` (safety policy block)

## Error Mapping

Google uses a three-tier SDK exception hierarchy: `ClientError`, `ServerError`, and `APIError`.

| Google error                                                    | Condition                      | Kitkat exception         |
| --------------------------------------------------------------- | ------------------------------ | ------------------------ |
| `ClientError` with code 401 or 403                              | Authentication failure         | `LLMAuthenticationError` |
| `ClientError` with code 429                                     | Rate limit exceeded            | `LLMRateLimitError`      |
| `ClientError` with code 400 and "token" or "context" in message | Prompt too long                | `LLMTokenLimitError`     |
| Any other `ClientError`                                         | Client-side API error          | `LLMProviderError`       |
| `ServerError`                                                   | Google server-side error (5xx) | `LLMProviderError`       |
| `APIError`                                                      | Generic Google API error       | `LLMProviderError`       |
| `asyncio.TimeoutError`                                          | `asyncio.timeout()` exceeded   | `LLMTimeoutError`        |

## `finish_reason` → `FinishReason` Mapping

| Google `finish_reason`        | `FinishReason`   |
| ----------------------------- | ---------------- |
| `"STOP"`                      | `STOP`           |
| `"MAX_TOKENS"`                | `LENGTH`         |
| `"SAFETY"`                    | `CONTENT_FILTER` |
| `"RECITATION"`                | `CONTENT_FILTER` |
| `"BLOCKLIST"`                 | `CONTENT_FILTER` |
| `"PROHIBITED_CONTENT"`        | `CONTENT_FILTER` |
| `"SPII"`                      | `CONTENT_FILTER` |
| `"IMAGE_SAFETY"`              | `CONTENT_FILTER` |
| `"MALFORMED_FUNCTION_CALL"`   | `TOOL_CALL`      |
| `"UNEXPECTED_TOOL_CALL"`      | `TOOL_CALL`      |
| `"LANGUAGE"`                  | `UNKNOWN`        |
| `"OTHER"`                     | `UNKNOWN`        |
| `"IMAGE_OTHER"`               | `UNKNOWN`        |
| `"FINISH_REASON_UNSPECIFIED"` | `UNKNOWN`        |

## Further Reading

- [Providers Overview](../providers.md) — `LLMService` API and provider comparison table
- [Concepts](../concepts.md) — `LLMRequest`, `LLMResponse`, `ThinkingConfig` reference
- [BYOK](../byok.md) — Per-request user API keys (note: Vertex AI not supported in BYOK mode)
- [Error Handling](../error-handling.md) — Full exception handling guide
- [API Reference — Providers](../api-reference/providers.md) — Complete API surface
