---
title: Providers
description: This page explains every built-in provider, their configuration classes, default models, capabilities, and provider-specific behaviours.
order: 1
---

This page covers every built-in provider — Anthropic, OpenAI, and Gemini — their configuration classes, default models, capabilities, and provider-specific behaviours. It also explains the `LLMService` managed service that wraps them.

> **📝 Note:** Kitkat uses an opt-in extras model. Each provider requires its own extra to be installed (`kitkat[anthropic]`, `kitkat[openai]`, `kitkat[gemini]`). See [Installation](./installation.md) for details.

## LLMService — the Managed Service

`LLMService` is the central façade for the managed service path. It owns provider lifecycle, routes requests by `ProviderType`, and exposes a consistent API regardless of which providers are registered.

### Creating a service

The recommended way to create a service is the `create_llm_service` factory:

```python
import os
import asyncio

from kitkat.service import create_llm_service
from kitkat import ProviderType
from kitkat.providers.anthropic import AnthropicProvider, AnthropicConfig
from kitkat.providers.openai import OpenAIProvider, OpenAIConfig

async def main() -> None:
    service = create_llm_service({
        ProviderType.ANTHROPIC: AnthropicProvider(
            AnthropicConfig(api_key=os.environ["ANTHROPIC_API_KEY"])
        ),
        ProviderType.OPENAI: OpenAIProvider(
            OpenAIConfig(api_key=os.environ["OPENAI_API_KEY"])
        ),
    })
    # initialize() opens HTTP connection pools and validates credentials.
    await service.initialize()
    # ... use the service ...
    await service.shutdown()

asyncio.run(main())
```

You can also build `LLMService` manually for more control:

```python
from kitkat.service import LLMService
from kitkat import ProviderType
from kitkat.providers.anthropic import AnthropicProvider, AnthropicConfig
import os

service = LLMService()
service.register_provider(
    ProviderType.ANTHROPIC,
    AnthropicProvider(AnthropicConfig(api_key=os.environ["ANTHROPIC_API_KEY"]))
)
# register_provider raises ValueError if the same ProviderType is registered twice.
```

### LLMService API

#### `register_provider(provider_type, provider)`

Registers a provider instance under its canonical type key. The provider need not be initialized yet; `initialize()` calls each provider's `initialize()` method.

- Raises `ValueError` if `provider_type` is already registered.

#### `async initialize()`

Initializes all registered providers **sequentially** in insertion order. If any provider fails, the exception surfaces immediately without leaving later providers partially started.

- Raises `LLMProviderInitError` if any provider fails to initialize.

#### `async shutdown()`

Gracefully shuts down all providers. Errors during individual shutdown are logged as warnings and do not prevent remaining providers from being shut down. The internal provider registry is cleared after shutdown.

#### `async complete(request, provider_type) -> LLMResponse`

Executes a non-streaming completion. The retry policy configured on the provider is applied automatically.

```python
from kitkat import LLMRequest, Message, Role, ProviderType

request = LLMRequest(
    messages=[Message(role=Role.USER, content="What is 1 + 1?")],
    model="claude-opus-4-5",
    max_tokens=64,
)
response = await service.complete(request, ProviderType.ANTHROPIC)
print(response.content)  # 2
```

- Raises `LLMProviderError` if the provider type is not registered.
- Raises `LLMTimeoutError`, `LLMRateLimitError`, `LLMTokenLimitError` after retries are exhausted.

#### `async stream(request, provider_type) -> AsyncIterator[StreamChunk]`

Yields token deltas from a streaming completion.

```python
request = LLMRequest(
    messages=[Message(role=Role.USER, content="Count to five.")],
    stream=True,
)
async for chunk in service.stream(request, ProviderType.OPENAI):
    if not chunk.is_final:
        print(chunk.delta, end="", flush=True)
```

#### `async health_check(provider_type) -> bool`

Probes a single provider's liveness. Returns `True` if the provider is reachable and operational.

#### `async health_check_all() -> dict[ProviderType, bool]`

Probes every registered provider sequentially. A failing probe for one provider does not block the others.

```python
health = await service.health_check_all()
# {ProviderType.ANTHROPIC: True, ProviderType.OPENAI: False}
```

#### `count_tokens(provider_type, text) -> int`

Estimates the token count for a text string using the provider's local tokenizer (tiktoken `cl100k_base` for all three built-in providers). Falls back to a character-based estimate (4 chars ≈ 1 token) in air-gapped environments where tiktoken cannot download its BPE data.

#### `count_prompt_tokens(provider_type, messages) -> int`

Estimates the total token count for a list of `Message` objects. Concatenates all message contents with a single space before delegating to `count_tokens`.

#### `get_capabilities(provider_type) -> ProviderCapabilities`

Returns the static capabilities descriptor for a registered provider. Use this to check whether a provider supports streaming, thinking, tool calling, or vision before building a request.

```python
caps = service.get_capabilities(ProviderType.GEMINI)
if caps.supports_thinking:
    request = LLMRequest(..., thinking=ThinkingConfig(enabled=True))
```

#### Properties

| Property         | Type                              | Description                                       |
| ---------------- | --------------------------------- | ------------------------------------------------- |
| `providers`      | `dict[ProviderType, LLMProvider]` | Read-only copy of the registered provider mapping |
| `provider_count` | `int`                             | Number of registered providers                    |

## Anthropic Provider

### Installation

```bash
pip install kitkat[anthropic]
```

### Configuration: `AnthropicConfig`

```python
from kitkat.providers.anthropic import AnthropicConfig
import os

config = AnthropicConfig(
    api_key=os.environ["ANTHROPIC_API_KEY"],  # Required. Must be non-empty.
    model="claude-opus-4-5",                  # Default: "claude-sonnet-4-6"
    base_url=None,                            # Override for proxies. Default: None (uses api.anthropic.com)
    max_retries=0,                            # SDK-level retries. Default: 0 (Kitkat handles retries itself)
    timeout_s=30.0,                           # Per-request timeout in seconds. Default: 30.0
    extra_headers={},                         # Extra HTTP headers injected into every request. Default: {}
)
```

**Validation rules:**

- `api_key` must be a non-empty string. An empty string raises `LLMProviderInitError` immediately.
- `timeout_s` must be positive. Zero or negative values raise `LLMProviderInitError`.

**Why `max_retries=0`?** Kitkat's own `RetryPolicy` wraps every call with exponential back-off. Setting SDK-level retries to 0 prevents double-retrying and gives Kitkat full control over the retry schedule.

### Provider capabilities

| Capability         | Value               |
| ------------------ | ------------------- |
| Default model      | `claude-sonnet-4-6` |
| Max context tokens | 200,000             |
| Streaming          | ✅                  |
| System prompt      | ✅                  |
| Tool calling       | ✅                  |
| Vision             | ✅                  |
| Extended thinking  | ✅                  |

### Initialization probe

`initialize()` sends a lightweight `messages.count_tokens` call to validate the API key before serving requests. This probe costs no inference tokens and takes ≤ 5 seconds. If it fails with `AuthenticationError`, an `LLMAuthenticationError` is raised immediately.

### System prompt handling

Anthropic's API separates the system prompt from conversation turns via a dedicated `system` parameter. Kitkat handles this automatically: any `Message(role=Role.SYSTEM, ...)` objects are extracted from the message list and concatenated with `\n\n---\n\n` as the separator.

```python
from kitkat import Message, Role

messages = [
    Message(role=Role.SYSTEM, content="You are a helpful assistant."),
    Message(role=Role.USER, content="Hello!"),
]
# Kitkat sends: system="You are a helpful assistant.", messages=[{"role": "user", ...}]
```

### Extended thinking (Anthropic)

Anthropic supports two thinking modes: **adaptive** (let the model decide when to think) and **enabled** (always think, with a fixed token budget).

```python
from kitkat import LLMRequest, Message, Role, ThinkingConfig

# Adaptive mode — effort controls the thinking intensity.
request = LLMRequest(
    messages=[Message(role=Role.USER, content="Solve this: 17 × 24 + 38 / 2")],
    thinking=ThinkingConfig(enabled=True, effort="high"),
    max_tokens=1024,
)

# Fixed budget mode — always think with a specific token budget.
request = LLMRequest(
    messages=[Message(role=Role.USER, content="Solve this: 17 × 24 + 38 / 2")],
    thinking=ThinkingConfig(
        enabled=True,
        provider_options={"thinking_type": "enabled", "budget_tokens": 5000},
    ),
    max_tokens=1024,
)
```

> **📝 Note:** Anthropic does not report thinking tokens separately; `output_tokens` in the usage response covers both thinking and answer tokens. Kitkat reflects this: `TokenUsage.thinking_tokens` is always `0` for Anthropic responses, and `completion_tokens` equals the total output token count.

> **⚠️ Warning:** When `thinking` is enabled on an Anthropic request, `temperature` and `top_p` are automatically omitted from the API call (Anthropic requires this). Kitkat handles this silently — you do not need to adjust your `LLMRequest`.

### Token counting

`count_tokens(text)` uses tiktoken's `cl100k_base` BPE encoding as a fast local approximation. For exact counts (useful for pre-flight budget checks), use `async_count_tokens(request)`, which calls the Anthropic API's `messages.count_tokens` endpoint.

```python
provider = AnthropicProvider(config)
await provider.initialize()

# Fast local estimate (no network)
approx = provider.count_tokens("Hello, world!")
print(approx)  # 4

# Exact count via Anthropic API
from kitkat import LLMRequest, Message, Role
exact = await provider.async_count_tokens(
    LLMRequest(messages=[Message(role=Role.USER, content="Hello, world!")])
)
print(exact)  # 4
```

### Retry policy

| Parameter          | Value                        |
| ------------------ | ---------------------------- |
| `max_attempts`     | 3                            |
| `base_delay_s`     | 1.0                          |
| `max_delay_s`      | 60.0                         |
| `exponential_base` | 2.0                          |
| `jitter`           | `True`                       |
| Retryable codes    | 408, 429, 500, 502, 503, 504 |

## OpenAI Provider

### Installation

```bash
pip install kitkat[openai]
```

### Configuration: `OpenAIConfig`

```python
from kitkat.providers.openai import OpenAIConfig
import os

config = OpenAIConfig(
    api_key=os.environ["OPENAI_API_KEY"],  # Required. Must be non-empty.
    model="gpt-4o-mini",                   # Default: "gpt-4o-mini"
    base_url=None,                         # Override for NVIDIA NIM, Azure, proxies. Default: None
    max_retries=0,                         # SDK-level retries. Default: 0
    timeout_s=60.0,                        # Per-request timeout in seconds. Default: 60.0
    extra_headers={},                      # Extra HTTP headers. Default: {}
    organization=None,                     # OpenAI organization ID. Ignored by non-OpenAI endpoints. Default: None
)
```

**Validation rules:**

- `api_key` must be a non-empty string.
- `timeout_s` must be positive.
- `max_retries` must be ≥ 0.

**OpenAI-compatible endpoints:** Set `base_url` to target any OpenAI-spec endpoint, including NVIDIA NIM and self-hosted vLLM:

```python
# NVIDIA NIM example
nvidia_config = OpenAIConfig(
    api_key=os.environ["NVIDIA_API_KEY"],
    base_url="https://integrate.api.nvidia.com/v1",
    model="nvidia/llama-3.1-8b-instruct",
)
```

### Provider capabilities

| Capability         | Value                |
| ------------------ | -------------------- |
| Default model      | `gpt-4o-mini`        |
| Max context tokens | 128,000              |
| Streaming          | ✅                   |
| System prompt      | ✅                   |
| Tool calling       | ✅                   |
| Vision             | ✅                   |
| Extended thinking  | ✅ (o-series models) |

### Initialization probe

`initialize()` calls `models.list()` to validate credentials before serving requests. This call lists available models and costs no inference tokens.

### System prompt handling

OpenAI supports system prompts inline as standard `{"role": "system", "content": "..."}` messages in the conversation list. Kitkat serializes `Message` objects verbatim — no extraction or restructuring is needed.

### Extended thinking (OpenAI)

OpenAI's o-series models (o1, o3, o4-mini, etc.) use `reasoning_effort` to control thinking intensity. Kitkat maps the normalized `ThinkingConfig.effort` field to `reasoning_effort`:

```python
from kitkat import LLMRequest, Message, Role, ThinkingConfig

request = LLMRequest(
    messages=[Message(role=Role.USER, content="Explain quantum entanglement.")],
    model="o4-mini",
    thinking=ThinkingConfig(enabled=True, effort="medium"),
    # Maps to: reasoning_effort="medium"
    max_tokens=2048,
)
```

> **📝 Note:** Non-reasoning OpenAI models ignore `ThinkingConfig`. Only o-series models with `reasoning_effort` support produce separate thinking tokens. For those models, `TokenUsage.thinking_tokens` reflects `completion_tokens_details.reasoning_tokens`.

### Token counting

`count_tokens(text)` uses tiktoken's `cl100k_base` BPE encoding — the shared base for GPT-4, GPT-3.5-turbo, and most NVIDIA NIM models.

### Retry policy

| Parameter          | Value                        |
| ------------------ | ---------------------------- |
| `max_attempts`     | 3                            |
| `base_delay_s`     | 1.0                          |
| `max_delay_s`      | 60.0                         |
| `exponential_base` | 2.0                          |
| `jitter`           | `True`                       |
| Retryable codes    | 408, 429, 500, 502, 503, 504 |

## Gemini Provider

### Installation

```bash
pip install kitkat[gemini]
```

### Configuration: `GeminiConfig`

```python
from kitkat.providers.gemini import GeminiConfig
import os

# API key mode (standard)
config = GeminiConfig(
    api_key=os.environ["GOOGLE_API_KEY"],  # Required when vertexai=False
    model="gemini-3-flash-preview",         # Default: "gemini-3-flash-preview"
    vertexai=False,                         # Set True to use Vertex AI instead
    timeout_s=60.0,                         # Default: 60.0
    extra_headers={},                       # Default: {}
)

# Vertex AI mode
vertex_config = GeminiConfig(
    vertexai=True,
    project="my-gcp-project",   # Required when vertexai=True
    location="us-central1",     # Required when vertexai=True
    model="gemini-3-flash-preview",
)
```

**Validation rules:**

- When `vertexai=False`: `api_key` must be a non-empty string.
- When `vertexai=True`: both `project` and `location` must be non-empty strings.
- `timeout_s` must be positive.

### Provider capabilities

| Capability         | Value                    |
| ------------------ | ------------------------ |
| Default model      | `gemini-3-flash-preview` |
| Max context tokens | 1,048,576 (1M+)          |
| Streaming          | ✅                       |
| System prompt      | ✅                       |
| Tool calling       | ✅                       |
| Vision             | ✅                       |
| Extended thinking  | ✅                       |

### Vertex AI support

Kitkat's Gemini provider supports Vertex AI deployments transparently. Set `vertexai=True` and provide your GCP `project` and `location`. The google-genai SDK uses Application Default Credentials (ADC) in Vertex AI mode — no `api_key` is required.

```python
import os
from kitkat.providers.gemini import GeminiProvider, GeminiConfig
from kitkat import ProviderType
from kitkat.service import create_llm_service

vertex_config = GeminiConfig(
    vertexai=True,
    project=os.environ["GOOGLE_CLOUD_PROJECT"],
    location="us-central1",
)
service = create_llm_service({
    ProviderType.GEMINI: GeminiProvider(vertex_config)
})
await service.initialize()
```

### System prompt handling

Gemini uses a top-level `system_instruction` parameter separate from the conversation turns. Kitkat extracts all `Role.SYSTEM` messages and concatenates them with `\n\n---\n\n` as the separator, then passes the result as `system_instruction`.

Gemini's role vocabulary differs from the standard: Kitkat maps `Role.ASSISTANT` → `"model"` and `Role.USER` → `"user"` automatically.

### Extended thinking (Gemini)

Gemini uses `thinking_level` (`"LOW"`, `"MEDIUM"`, `"HIGH"`) to control reasoning intensity:

```python
from kitkat import LLMRequest, Message, Role, ThinkingConfig

request = LLMRequest(
    messages=[Message(role=Role.USER, content="Prove the Pythagorean theorem.")],
    model="gemini-3-flash-preview",
    thinking=ThinkingConfig(enabled=True, effort="high"),
    # Kitkat maps: effort="high" → thinking_level="HIGH"
    max_tokens=4096,
)
```

> **📝 Note:** `TokenUsage.thinking_tokens` reflects `thoughts_token_count` from Gemini's `usage_metadata` when thinking is enabled.

### Safety filter behaviour

Gemini raises `LLMContentFilterError` when its safety policies block a response. This covers all Gemini safety categories including `SAFETY`, `RECITATION`, `BLOCKLIST`, `PROHIBITED_CONTENT`, `SPII`, and `IMAGE_SAFETY`. Unlike `LLMRateLimitError`, content filter errors are **not retried** — a different provider would produce the same outcome.

### Retry policy

| Parameter          | Value                           |
| ------------------ | ------------------------------- |
| `max_attempts`     | 3                               |
| `base_delay_s`     | 2.0 (extended for quota limits) |
| `max_delay_s`      | 60.0                            |
| `exponential_base` | 2.0                             |
| `jitter`           | `True`                          |
| Retryable codes    | 408, 429, 500, 502, 503, 504    |

## Provider Comparison

| Feature                  | Anthropic               | OpenAI               | Gemini                       |
| ------------------------ | ----------------------- | -------------------- | ---------------------------- |
| Default model            | `claude-sonnet-4-6`     | `gpt-4o-mini`        | `gemini-3-flash-preview`     |
| Max context              | 200k tokens             | 128k tokens          | 1M+ tokens                   |
| System prompt            | Separate `system` param | Inline `role=system` | `system_instruction` param   |
| Thinking tokens reported | No (merged with output) | Yes (o-series only)  | Yes (`thoughts_token_count`) |
| Health probe             | `count_tokens("ping")`  | `models.list()`      | `count_tokens("ping")`       |
| Base retry delay         | 1.0 s                   | 1.0 s                | 2.0 s                        |
| Vertex AI support        | ❌                      | ❌                   | ✅                           |

## Further Reading

- [Routing & Cache](./routing-cache.md) — Route across multiple providers with automatic failover
- [BYOK](./byok.md) — Use per-request user API keys
- [Custom Providers](./custom-provider.md) — Implement your own `LLMProvider`
- [API Reference — Providers](./api-reference/providers.md) — Full API surface for all provider classes
