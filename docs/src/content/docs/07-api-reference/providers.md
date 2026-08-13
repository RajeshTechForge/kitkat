---
title: Providers
description: Documentation for the built-in provider classes, their configuration dataclasses, and the abstract `LLMProvider` base class.
order: 3
---

This page documents the three built-in provider classes, their configuration dataclasses, and the abstract `LLMProvider` base class.

**Import paths:**

```python
from kitkat.providers.anthropic import AnthropicProvider, AnthropicConfig
from kitkat.providers.openai import OpenAIProvider, OpenAIConfig
from kitkat.providers.gemini import GeminiProvider, GeminiConfig
from kitkat.abc import LLMProvider
```

## `LLMProvider` (ABC)

```python
from kitkat.abc import LLMProvider
```

Abstract base class for all providers. Subclass this to implement a custom provider. See [Custom Providers](../custom-provider.md) for the complete implementation guide.

### Required Class Attributes

| Attribute       | Type                   | Description                                 |
| --------------- | ---------------------- | ------------------------------------------- |
| `PROVIDER_TYPE` | `ProviderType`         | Canonical provider enum value               |
| `DEFAULT_MODEL` | `str`                  | Model used when `LLMRequest.model` is empty |
| `CAPABILITIES`  | `ProviderCapabilities` | Feature flags and context window size       |
| `RETRY_POLICY`  | `RetryPolicy`          | Exponential back-off configuration          |

### Abstract Methods (must implement)

| Method              | Signature                                                   | Description                                                                            |
| ------------------- | ----------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| `initialize`        | `async () -> None`                                          | Open HTTP client, probe credentials, set `_initialized = True`. Idempotent.            |
| `_init_client_only` | `async () -> None`                                          | Open HTTP client only — no credential probe. Called by `BYOKLLMService`. Idempotent.   |
| `shutdown`          | `async () -> None`                                          | Close HTTP client, set `_initialized = False`. Safe to call on uninitialized provider. |
| `complete`          | `async (request: LLMRequest) -> LLMResponse`                | Single non-streaming attempt. No retry.                                                |
| `stream`            | `async (request: LLMRequest) -> AsyncIterator[StreamChunk]` | Streaming via async generator. Final sentinel has `is_final=True`.                     |
| `health_check`      | `async () -> bool`                                          | Liveness probe. Must return `False`, never raise, on failure.                          |
| `count_tokens`      | `(text: str) -> int`                                        | Local token estimate. No network call.                                                 |

### Provided by Base Class (do not re-implement)

| Helper                                      | Description                                                        |
| ------------------------------------------- | ------------------------------------------------------------------ |
| `complete_with_retry(request, policy=None)` | Wraps `complete()` with `execute_with_retry` using `RETRY_POLICY`. |
| `count_prompt_tokens(messages)`             | Concatenates message content and delegates to `count_tokens()`.    |
| `run_sync(request)`                         | Synchronous blocking wrapper. Safe outside an event loop.          |
| `_assert_initialized()`                     | Raises `RuntimeError` if `_initialized` is `False`.                |
| `__aenter__` / `__aexit__`                  | Calls `initialize()` on enter, `shutdown()` on exit.               |

## `AnthropicProvider`

```python
from kitkat.providers.anthropic import AnthropicProvider, AnthropicConfig
```

**Extras required:** `pip install kitkat[anthropic]`

### `AnthropicConfig`

| Field         | Type    | Default                       | Description                                                                             |
| ------------- | ------- | ----------------------------- | --------------------------------------------------------------------------------------- |
| `api_key`     | `str`   | —                             | **Required.** Anthropic API key (`sk-ant-...`). Raises `LLMProviderInitError` if empty. |
| `model`       | `str`   | `"claude-opus-4-5"`           | Default model                                                                           |
| `base_url`    | `str`   | `"https://api.anthropic.com"` | API base URL. Override for proxies.                                                     |
| `max_retries` | `int`   | `3`                           | SDK-level retry count                                                                   |
| `timeout`     | `float` | `30.0`                        | Per-request timeout in seconds                                                          |

### Capabilities

| Capability               | Value     |
| ------------------------ | --------- |
| `supports_streaming`     | `True`    |
| `supports_system_prompt` | `True`    |
| `supports_tool_calling`  | `True`    |
| `supports_vision`        | `True`    |
| `supports_thinking`      | `True`    |
| `max_context_tokens`     | `200_000` |

### Notable Behaviour

- **Thinking mode**: Set `LLMRequest.thinking = ThinkingConfig(enabled=True, effort="high")`. Thinking tokens are billed separately and appear in `LLMResponse.thinking_content`.
- **System prompt**: The provider extracts `Role.SYSTEM` messages and passes them as the `system` parameter in the Anthropic API request.
- `LLMTokenLimitError` is raised when the provider returns HTTP 400 with a token-count error message.

## `OpenAIProvider`

```python
from kitkat.providers.openai import OpenAIProvider, OpenAIConfig
```

**Extras required:** `pip install kitkat[openai]`

### `OpenAIConfig`

| Field          | Type          | Default    | Description                                                                      |
| -------------- | ------------- | ---------- | -------------------------------------------------------------------------------- |
| `api_key`      | `str`         | —          | **Required.** OpenAI API key (`sk-...`). Raises `LLMProviderInitError` if empty. |
| `model`        | `str`         | `"gpt-4o"` | Default model                                                                    |
| `base_url`     | `str \| None` | `None`     | Override for Azure OpenAI or OpenAI-compatible endpoints (e.g., Ollama, vLLM).   |
| `organization` | `str \| None` | `None`     | OpenAI organization ID                                                           |
| `timeout`      | `float`       | `30.0`     | Per-request timeout in seconds                                                   |

### Capabilities

| Capability               | Value     |
| ------------------------ | --------- |
| `supports_streaming`     | `True`    |
| `supports_system_prompt` | `True`    |
| `supports_tool_calling`  | `True`    |
| `supports_vision`        | `True`    |
| `supports_thinking`      | `True`    |
| `max_context_tokens`     | `128_000` |

### Notable Behaviour

- **Reasoning models** (`o1`, `o3`): Thinking tokens are exposed via `usage.completion_tokens_details.reasoning_tokens` and mapped to `TokenUsage.thinking_tokens`.
- **OpenAI-compatible endpoints**: Set `base_url` to point at any Chat Completions-compatible API (Azure, Ollama, LM Studio, vLLM).
- `LLMContentFilterError` is raised when `finish_reason == "content_filter"`.

## `GeminiProvider`

```python
from kitkat.providers.gemini import GeminiProvider, GeminiConfig
```

**Extras required:** `pip install kitkat[gemini]`

### `GeminiConfig`

| Field     | Type    | Default              | Description                                                                     |
| --------- | ------- | -------------------- | ------------------------------------------------------------------------------- |
| `api_key` | `str`   | —                    | **Required.** Google AI Studio API key. Raises `LLMProviderInitError` if empty. |
| `model`   | `str`   | `"gemini-2.5-flash"` | Default model                                                                   |
| `timeout` | `float` | `30.0`               | Per-request timeout in seconds                                                  |

### Capabilities

| Capability               | Value       |
| ------------------------ | ----------- |
| `supports_streaming`     | `True`      |
| `supports_system_prompt` | `True`      |
| `supports_tool_calling`  | `True`      |
| `supports_vision`        | `True`      |
| `supports_thinking`      | `True`      |
| `max_context_tokens`     | `1_000_000` |

### Notable Behaviour

- **Safety categories** mapped to `LLMContentFilterError`: `SAFETY`, `RECITATION`, `BLOCKLIST`, `PROHIBITED_CONTENT`, `SPII`, `IMAGE_SAFETY`.
- **Thinking mode**: Use `GeminiConfig.model = "gemini-2.5-flash"` or `"gemini-2.5-pro"` and set `LLMRequest.thinking = ThinkingConfig(enabled=True)`.
- **System prompt**: Passed as `system_instruction` in the Gemini API. Extracted automatically from `Role.SYSTEM` messages.
- Token counting uses the Gemini `count_tokens` API for accurate counts (not a local estimate).

## Further Reading

- [Providers Overview](../providers.md) — Configuration guide and model tables for all providers
- [Anthropic](../anthropic.md) · [OpenAI](../openai.md) · [Gemini](../gemini.md) — Per-provider deep-dives
- [Custom Providers](../custom-provider.md) — Implementing `LLMProvider` for a new API
- [API Reference — Service](./service.md) — `LLMService`, `LLMRouter`, `BYOKLLMService`
