---
title: Google (Gemini & Vertex AI)
description: Complete reference for Kitkat's Google providers, covering both the Gemini API (AI Studio) and Vertex AI (GCP), including configuration, authentication, thinking levels, and error mapping.
order: 3
---

This page is the complete reference for Kitkat's Google providers. Google exposes two distinct APIs: the **Gemini API** (Google AI Studio) and **Vertex AI** (Google Cloud Platform). Because their capabilities, authentication, and "thinking" configurations diverge, Kitkat implements them as two separate, swappable providers: `GeminiProvider` and `VertexAIProvider`.

> **📝 Note:** This page assumes you have read [Concepts](../concepts.md). If not, start there first.

## Installation

```bash
pip install kitkat[google]
```

This installs the `google-genai` Python SDK (≥ 1.57.0) and `google-cloud-aiplatform` alongside Kitkat's core package.

## Quick Start

### Gemini API (AI Studio)

```python
import asyncio
import os

from kitkat.service import create_llm_service
from kitkat import ProviderType, LLMRequest, Message, Role
from kitkat.providers.google.gemini import GeminiProvider, GeminiConfig

async def main() -> None:
    config = GeminiConfig(api_key=os.environ["GOOGLE_API_KEY"])
    service = create_llm_service({ProviderType.GEMINI: GeminiProvider(config)})
    await service.initialize()

    response = await service.complete(
        LLMRequest(
            messages=[Message(role=Role.USER, content="What is a Python context manager?")],
            model="gemini-3-flash-preview",
            max_tokens=256,
        ),
        ProviderType.GEMINI,
    )
    print(response.content)

asyncio.run(main())
```

### Vertex AI (GCP)

```python
import asyncio
import os

from kitkat.service import create_llm_service
from kitkat import ProviderType, LLMRequest, Message, Role
from kitkat.providers.google.vertex_ai import VertexAIProvider, VertexAIConfig

async def main() -> None:
    config = VertexAIConfig(
        project=os.environ["GOOGLE_CLOUD_PROJECT"],
        location="us-central1",
        # credentials_path="/path/to/sa.json"  # Optional: defaults to ADC
    )
    service = create_llm_service({ProviderType.VERTEX_AI: VertexAIProvider(config)})
    await service.initialize()

    response = await service.complete(
        LLMRequest(
            messages=[Message(role=Role.USER, content="Hello from Vertex AI!")],
            model="gemini-1.5-pro-002",
        ),
        ProviderType.VERTEX_AI,
    )
    print(response.content)

asyncio.run(main())
```

## Configuration

Kitkat provides distinct configuration dataclasses for each Google API. Both validate fields in `__post_init__`.

### `GeminiConfig`

Targets the Google AI Studio endpoint. Authentication is strictly via API key.

```python
from kitkat.providers.google.gemini import GeminiConfig
import os

config = GeminiConfig(
    api_key=os.environ["GOOGLE_API_KEY"],  # Required
    model="gemini-3-flash-preview",         # Default: "gemini-3-flash-preview"
    timeout_s=60.0,                         # Default: 60.0
    extra_headers={},                       # Default: {}
)
```

| Field           | Type             | Default                    | Description                                                                    |
| --------------- | ---------------- | -------------------------- | ------------------------------------------------------------------------------ |
| `api_key`       | `str`            | `""`                       | Your Google AI Studio API key. **Required.**                                   |
| `model`         | `str`            | `"gemini-3-flash-preview"` | The default model identifier. Used when `LLMRequest.model` is empty.           |
| `timeout_s`     | `float`          | `60.0`                     | Per-request wall-clock timeout in seconds. Overridden by `LLMRequest.timeout`. |
| `extra_headers` | `dict[str, str]` | `{}`                       | Arbitrary HTTP headers injected into every request.                            |

### `VertexAIConfig`

Targets the GCP Vertex AI endpoint. Authentication uses Service Account JSON or Application Default Credentials (ADC).

```python
from kitkat.providers.google.vertex_ai import VertexAIConfig
import os

config = VertexAIConfig(
    project=os.environ["GOOGLE_CLOUD_PROJECT"],  # Required
    location="us-central1",                       # Required
    credentials_path="",                          # Optional: path to SA JSON. Defaults to ADC.
    model="gemini-1.5-pro-002",                   # Default: "gemini-1.5-pro-002"
    timeout_s=60.0,
)
```

| Field              | Type             | Default                | Description                                                                    |
| ------------------ | ---------------- | ---------------------- | ------------------------------------------------------------------------------ |
| `project`          | `str`            | `""`                   | GCP Project ID. **Required.**                                                  |
| `location`         | `str`            | `""`                   | GCP region (e.g., `"us-central1"`). **Required.**                              |
| `credentials_path` | `str`            | `""`                   | Path to service account JSON. If empty, uses ADC.                              |
| `model`            | `str`            | `"gemini-1.5-pro-002"` | The default model identifier. Used when `LLMRequest.model` is empty.           |
| `timeout_s`        | `float`          | `60.0`                 | Per-request wall-clock timeout in seconds. Overridden by `LLMRequest.timeout`. |
| `extra_headers`    | `dict[str, str]` | `{}`                   | Arbitrary HTTP headers injected into every request.                            |

### Building from a dictionary

Both classes expose a `from_dict` class method:

```python
config = GeminiConfig.from_dict({"api_key": "...", "model": "gemini-2.5-pro"})
vertex_config = VertexAIConfig.from_dict({"project": "my-proj", "location": "us-east1"})
```

## Provider Classes

### `GeminiProvider`

```python
from kitkat.providers.google.gemini import GeminiProvider, GeminiConfig
```

| Attribute                             | Value                      |
| ------------------------------------- | -------------------------- |
| `PROVIDER_TYPE`                       | `ProviderType.GEMINI`      |
| `DEFAULT_MODEL`                       | `"gemini-3-flash-preview"` |
| `CAPABILITIES.supports_streaming`     | `True`                     |
| `CAPABILITIES.supports_system_prompt` | `True`                     |
| `CAPABILITIES.supports_tool_calling`  | `True`                     |
| `CAPABILITIES.supports_vision`        | `True`                     |
| `CAPABILITIES.supports_thinking`      | `True`                     |
| `CAPABILITIES.max_context_tokens`     | `1_048_576` (1M+)          |

### `VertexAIProvider`

```python
from kitkat.providers.google.vertex_ai import VertexAIProvider, VertexAIConfig
```

| Attribute                             | Value                    |
| ------------------------------------- | ------------------------ |
| `PROVIDER_TYPE`                       | `ProviderType.VERTEX_AI` |
| `DEFAULT_MODEL`                       | `"gemini-1.5-pro-002"`   |
| `CAPABILITIES.supports_streaming`     | `True`                   |
| `CAPABILITIES.supports_system_prompt` | `True`                   |
| `CAPABILITIES.supports_tool_calling`  | `True`                   |
| `CAPABILITIES.supports_vision`        | `True`                   |
| `CAPABILITIES.supports_thinking`      | `True`                   |
| `CAPABILITIES.max_context_tokens`     | `2_097_152` (2M+)        |

## Authentication & Lifecycle

### `async initialize()`

Both providers construct the `google.genai.Client` and run a credential probe via `aio.models.count_tokens(model=..., contents="ping")`. The probe consumes no inference tokens and times out after 5 seconds.

- **Gemini**: Probes the API key validity. Raises `LLMProviderInitError` on HTTP 401/403.
- **Vertex AI**: Resolves ADC or loads the Service Account JSON. Raises `LLMProviderInitError` if IAM permissions are insufficient (`aiplatform.endpoints.predict`) or if the model is not available in the configured `location` (HTTP 404).

### Using as an async context manager

```python
async with GeminiProvider(config) as provider:
    response = await provider.complete(request)
```

### BYOK (Bring Your Own Key)

> **📝 Note:** The BYOK service path (`BYOKLLMService`) does not support `VertexAIProvider`, because Vertex AI uses GCP ADC/Service Accounts rather than a user-supplied API key. BYOK is fully supported for `GeminiProvider`.

## System Prompt Handling

Both Google APIs expect a dedicated `system_instruction` top-level parameter separate from the conversation turns. Kitkat handles the extraction automatically in a shared internal utility.

When you include `Message(role=Role.SYSTEM, ...)` objects in your message list, Kitkat:

1. Extracts all system messages from the list.
2. Concatenates their content with `\n\n---\n\n` as a separator.
3. Passes the result as `system_instruction` in `GenerateContentConfig`.
4. Maps remaining messages to `genai_types.Content` objects with Google's role vocabulary (`"user"` / `"model"`).

## Extended Thinking

Google supports extended thinking, but the API contracts diverge between Gemini and Vertex AI. Kitkat handles this natively for each provider.

### Gemini API (Thinking Levels)

The Gemini API uses discrete `thinking_level` values (`"LOW"`, `"MEDIUM"`, `"HIGH"`). Kitkat maps the normalized `ThinkingConfig.effort` field to Google's vocabulary.

| `ThinkingConfig.effort` | Google `thinking_level` |
| ----------------------- | ----------------------- |
| `"low"`                 | `"LOW"`                 |
| `"medium"`              | `"MEDIUM"`              |
| `"high"`                | `"HIGH"`                |

You can also bypass the effort mapping and set the level directly via `provider_options`:

```python
request = LLMRequest(
    messages=[Message(role=Role.USER, content="What are the prime factors of 1729?")],
    model="gemini-3-flash-preview",
    thinking=ThinkingConfig(
        enabled=True,
        provider_options={"level": "HIGH"},   # Maps directly to thinking_level="HIGH"
    ),
)
```

### Vertex AI (Thinking Budgets)

Enterprise Vertex AI deployments often require deterministic token budgets rather than discrete levels. The `VertexAIProvider` resolves thinking configuration using a priority sequence:

1. **Explicit Budget:** `thinking.provider_options.get("thinking_budget")` (int)
2. **Effort Mapping:** `thinking.effort` mapped to predefined enterprise token budgets (`low` → 1024, `medium` → 8192, `high` → 24576).
3. **Default:** If thinking is enabled but no budget/effort is provided, the budget parameter is omitted, letting Vertex AI decide dynamically.

```python
request = LLMRequest(
    messages=[Message(role=Role.USER, content="Analyze this large financial dataset...")],
    model="gemini-1.5-pro-002",
    thinking=ThinkingConfig(
        enabled=True,
        provider_options={"thinking_budget": 5000},  # Exact token limit for reasoning
    ),
)
```

### Thinking in streaming & token reporting

Both providers distinguish thinking and answer parts via the `thought` attribute on `Part` objects. Kitkat maps this to `StreamChunk.is_thinking`.

Thinking tokens are reported separately in `usage_metadata.thoughts_token_count`, which Kitkat maps to `TokenUsage.thinking_tokens`.

## Safety Filters

Google applies safety filters across multiple categories. When a response is blocked, both providers raise `LLMContentFilterError`.

**Filter categories:**
`SAFETY`, `RECITATION`, `BLOCKLIST`, `PROHIBITED_CONTENT`, `SPII`, `IMAGE_SAFETY`

> **⚠️ Warning:** Content filter errors are **not retried** — the same content would be blocked on any subsequent attempt. Catch them explicitly and provide a user-facing error message.

## Token Counting

### Fast local estimate

`count_tokens(text)` uses tiktoken's `cl100k_base` BPE encoding as a fast approximation with no network call. If tiktoken cannot load BPE data (e.g., air-gapped environments), it falls back to `max(1, len(text) // 4)`.

### Exact count via Google API

`async_count_tokens(request)` calls `aio.models.count_tokens` for the exact prompt token count. No inference tokens are consumed.

## Health Check

`health_check()` calls `aio.models.count_tokens(model=..., contents="ping")` with a 5-second timeout to verify the provider is reachable. Returns `False` on any error rather than raising.

## Retry Policy

Both providers share the same retry policy parameters:

| Parameter          | Value                            | Rationale                                          |
| ------------------ | -------------------------------- | -------------------------------------------------- |
| `max_attempts`     | `3`                              | 1 original + 2 retries                             |
| `base_delay_s`     | `2.0`                            | Extended base delay for quota-limited environments |
| `max_delay_s`      | `60.0`                           | Cap for degraded scenarios                         |
| `exponential_base` | `2.0`                            | Doubles per attempt: 2s → 4s → 8s (before jitter)  |
| `jitter`           | `True`                           | ±50% random variation                              |
| Retryable codes    | `{408, 429, 500, 502, 503, 504}` | Standard transient codes                           |

The following errors are **never** retried:

- `LLMAuthenticationError` (401, 403, IAM denied)
- `LLMTokenLimitError` (context exceeded)
- `LLMContentFilterError` (safety policy block)

## Error Mapping

While both providers map Google SDK errors to Kitkat exceptions, `VertexAIProvider` includes GCP-specific error semantics.

| Google error                                          | Condition                             | Kitkat exception         |
| ----------------------------------------------------- | ------------------------------------- | ------------------------ |
| `ClientError` 401/403                                 | Authentication failure                | `LLMAuthenticationError` |
| `ClientError` 403 ("Permission denied") _(Vertex AI)_ | Service account lacks IAM permissions | `LLMAuthenticationError` |
| `ClientError` 404 _(Vertex AI)_                       | Model not found in specified region   | `LLMProviderError`       |
| `ClientError` 429                                     | Rate limit / Quota exceeded           | `LLMRateLimitError`      |
| `ClientError` 400 ("token" or "context")              | Prompt too long                       | `LLMTokenLimitError`     |
| Any other `ClientError`                               | Client-side API error                 | `LLMProviderError`       |
| `ServerError`                                         | Google server-side error (5xx)        | `LLMProviderError`       |
| `APIError`                                            | Generic Google API error              | `LLMProviderError`       |
| `asyncio.TimeoutError`                                | `asyncio.timeout()` exceeded          | `LLMTimeoutError`        |

## `finish_reason` → `FinishReason` Mapping

Shared across both providers:

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
- [BYOK](../byok.md) — Per-request user API keys (Gemini only)
- [Error Handling](../error-handling.md) — Full exception handling guide

```

```
