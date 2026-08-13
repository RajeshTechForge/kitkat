---
title: Custom Providers
description: Learn how to implement a custom LLM provider for KitKat by subclassing the `LLMProvider` abstract base class. This guide covers the complete contract, error mapping, streaming, token counting, health checks, and testing.
order: 1
---

KitKat's provider system is open: any class that implements the `LLMProvider` abstract base class (ABC) is a first-class provider. You can wrap any LLM API — a proprietary internal model, an on-premises vLLM deployment, a custom inference server, or a provider that KitKat doesn't ship with — and have it work identically to the built-in Anthropic, OpenAI, and Gemini providers.

This page walks through the complete `LLMProvider` contract, a full working implementation using `httpx`, BYOK compatibility, error mapping, streaming, token counting, health checks, retry configuration, testing, and shipping your provider as an installable plugin.

---

## When to Write a Custom Provider

Write a custom provider when:

- You are using an LLM API that KitKat does not ship with (e.g., Cohere, Mistral, a private internal model).
- You need to wrap a self-hosted vLLM server that uses a non-standard (non-OpenAI-compatible) API format.
- You want to add pre/post-processing hooks (prompt caching, PII scrubbing, cost metering) at the transport layer without modifying application code.
- You are building a mock provider for integration testing that simulates specific error conditions.

If the API you want to use is OpenAI-compatible (follows the Chat Completions spec), use `OpenAIProvider` with a custom `base_url` instead — it is simpler and maintained for you. See [OpenAI — OpenAI-Compatible Endpoints](./openai.md#openai-compatible-endpoints).

---

## Installation

```bash
pip install kitkat httpx
```

This guide uses `httpx` as the HTTP client. You can substitute any async HTTP library.

---

## The `LLMProvider` ABC

Every provider must subclass `kitkat.abc.LLMProvider`. Import it from the public ABC package:

```python
from kitkat.abc import LLMProvider
```

### Class-level attributes (mandatory)

Declare these as class variables on every provider. KitKat reads them at runtime to power routing decisions, capability queries, and default model resolution.

| Attribute | Type | Description |
|---|---|---|
| `PROVIDER_TYPE` | `ProviderType` | Canonical provider enum value. Use an existing `ProviderType` member or extend the enum for a completely new provider. |
| `DEFAULT_MODEL` | `str` | Fallback model identifier used when `LLMRequest.model` is empty. |
| `CAPABILITIES` | `ProviderCapabilities` | Feature flags the router queries when selecting a provider (streaming support, context window size, etc.). |
| `RETRY_POLICY` | `RetryPolicy` | Optional — a class-level `RetryPolicy` overrides the base class default (`max_attempts=3, base_delay_s=1.0`). |

### Abstract methods (mandatory)

Every subclass must implement these eight methods. The base class provides helpful shared utilities (see [Helpers provided by the base class](#helpers-provided-by-the-base-class)) that you should use rather than re-implement.

| Method | Signature | Purpose |
|---|---|---|
| `initialize` | `async () -> None` | Open HTTP client, validate credentials. |
| `_init_client_only` | `async () -> None` | Open HTTP client only — no credential probe. Used by `BYOKLLMService`. |
| `shutdown` | `async () -> None` | Close HTTP client, release resources. |
| `complete` | `async (request: LLMRequest) -> LLMResponse` | One non-streaming inference attempt. No retry. |
| `stream` | `async (request: LLMRequest) -> AsyncIterator[StreamChunk]` | Async generator of token chunks. |
| `health_check` | `async () -> bool` | Liveness probe. Must return `False`, never raise, on failure. |
| `count_tokens` | `(text: str) -> int` | Local token estimate. No network call. |

### Helpers provided by the base class

These methods are implemented by `LLMProvider` and available for free in every subclass. Do not re-implement them.

| Helper | Description |
|---|---|
| `complete_with_retry(request, policy=None)` | Calls `execute_with_retry` with the class-level `RETRY_POLICY`. Use this in your service-layer calls instead of calling `complete()` directly. |
| `count_prompt_tokens(messages)` | Concatenates message content strings and delegates to your `count_tokens()`. |
| `_assert_initialized()` | Raises `RuntimeError` if `self._initialized` is `False`. Call this at the top of `complete()`, `stream()`, and `health_check()`. |
| `run_sync(request)` | Blocking synchronous wrapper around `complete()`. Safe to use outside an asyncio event loop. |
| `__aenter__` / `__aexit__` | Async context manager support: `__aenter__` calls `initialize()`, `__aexit__` calls `shutdown()`. |

---

## Complete Implementation: `MyProvider`

This is a full, runnable custom provider that calls a hypothetical REST API at `https://api.myllm.example.com`. Study each section — the docstrings explain the contract requirements that go beyond "make it compile".

```python
# my_provider/provider.py
from __future__ import annotations

import time
from typing import Any, AsyncIterator

import httpx

from kitkat.abc import LLMProvider
from kitkat.core.enums import FinishReason, ProviderType
from kitkat.core.models import (
    LLMRequest,
    LLMResponse,
    ProviderCapabilities,
    RetryPolicy,
    StreamChunk,
    TokenUsage,
)
from kitkat import (
    LLMAuthenticationError,
    LLMContentFilterError,
    LLMProviderError,
    LLMProviderInitError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMTokenLimitError,
)


class MyProviderConfig:
    """Configuration dataclass for MyProvider."""

    def __init__(self, api_key: str, base_url: str = "https://api.myllm.example.com", timeout_s: float = 30.0) -> None:
        if not api_key or not api_key.strip():
            raise LLMProviderInitError("MyProviderConfig.api_key must be a non-empty string.", provider="my-llm")
        if timeout_s <= 0:
            raise LLMProviderInitError("MyProviderConfig.timeout_s must be positive.", provider="my-llm")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s


class MyProvider(LLMProvider):
    """Custom provider implementation for https://api.myllm.example.com."""

    # ── Class-level attributes ──────────────────────────────────────────────

    # Re-use an existing ProviderType or extend the StrEnum for a new provider.
    PROVIDER_TYPE = ProviderType.OPENAI

    DEFAULT_MODEL = "my-model-v2"

    CAPABILITIES = ProviderCapabilities(
        supports_streaming=True,
        supports_system_prompt=True,
        supports_tool_calling=False,
        supports_vision=False,
        supports_thinking=False,
        max_context_tokens=32_768,
        provider_type=ProviderType.OPENAI,
    )

    # Override the default RetryPolicy — this API recovers quickly from 429s.
    RETRY_POLICY = RetryPolicy(
        max_attempts=3,
        base_delay_s=0.5,       # Shorter base: 0.5s → 1s → 2s (before jitter)
        max_delay_s=30.0,
        exponential_base=2.0,
        jitter=True,
        retryable_status_codes=frozenset({408, 429, 500, 502, 503, 504}),
    )

    # ── Constructor ─────────────────────────────────────────────────────────

    def __init__(self, config: dict[str, Any] | MyProviderConfig) -> None:
        # The base class __init__ stores config in self._config and sets
        # self._initialized = False.
        super().__init__(config if isinstance(config, dict) else vars(config))

        # Extract and validate config. Raise LLMProviderInitError on bad values
        # — validation at construction time means callers get a clear error
        # before any network calls are made.
        if isinstance(config, MyProviderConfig):
            self._cfg = config
        else:
            self._cfg = MyProviderConfig(
                api_key=config.get("api_key", ""),
                base_url=config.get("base_url", "https://api.myllm.example.com"),
                timeout_s=float(config.get("timeout_s", 30.0)),
            )

        self._client: httpx.AsyncClient | None = None

    # ── Lifecycle ───────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Open the HTTP client and validate credentials via a lightweight probe.

        Preconditions:
            Config fields are validated in __init__.
        Postconditions:
            self._initialized = True.
            self._client is a live httpx.AsyncClient.
        Raises:
            LLMProviderInitError: if the credential probe fails.
            LLMAuthenticationError: if the API key is invalid.
        """
        if self._initialized:
            return  # Idempotent: second call is a no-op.

        self._client = httpx.AsyncClient(
            base_url=self._cfg.base_url,
            headers={
                "Authorization": f"Bearer {self._cfg.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=httpx.Timeout(self._cfg.timeout_s),
        )

        # Credential probe: a lightweight endpoint that consumes no credits.
        try:
            resp = await self._client.get("/v1/models", timeout=8.0)
            if resp.status_code == 401:
                raise LLMAuthenticationError(
                    "Invalid or revoked API key.",
                    status_code=401,
                    provider="my-llm",
                )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise LLMProviderInitError(
                f"Credential probe failed: HTTP {exc.response.status_code}",
                provider="my-llm",
            ) from exc
        except httpx.RequestError as exc:
            raise LLMProviderInitError(
                f"Cannot connect to MyLLM API: {exc}",
                provider="my-llm",
            ) from exc

        self._initialized = True

    async def _init_client_only(self) -> None:
        """Open the HTTP client WITHOUT a credential probe.

        This is the BYOK path: BYOKLLMService calls this instead of initialize()
        to avoid a preflight API call. Authentication failures surface on the
        first complete() or stream() call as LLMAuthenticationError.

        Preconditions: none (may be called on an uninitialized provider).
        Postconditions: self._initialized = True. self._client is live.
        """
        if self._initialized:
            return  # Idempotent.

        self._client = httpx.AsyncClient(
            base_url=self._cfg.base_url,
            headers={
                "Authorization": f"Bearer {self._cfg.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=httpx.Timeout(self._cfg.timeout_s),
        )
        self._initialized = True

    async def shutdown(self) -> None:
        """Close the HTTP connection pool and mark the provider as uninitialized.

        Safe to call if the provider was never initialized — it is a no-op.
        """
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        self._initialized = False

    # ── Inference ───────────────────────────────────────────────────────────

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Execute a single non-streaming completion attempt (no retry).

        Use complete_with_retry() when you want the built-in RetryPolicy applied.

        Preconditions:
            initialize() or _init_client_only() has been called.
        Postconditions:
            Returns a fully populated LLMResponse.
        Raises:
            LLMAuthenticationError, LLMRateLimitError, LLMTokenLimitError,
            LLMContentFilterError, LLMTimeoutError, LLMProviderError.
        """
        self._assert_initialized()
        assert self._client is not None  # Satisfied by _assert_initialized.

        model = request.model or self.DEFAULT_MODEL
        body = self._build_request_body(request, model, stream=False)
        start_time = time.monotonic()

        try:
            resp = await self._client.post(
                "/v1/chat/completions",
                json=body,
                timeout=request.timeout or self._cfg.timeout_s,
            )
        except httpx.TimeoutException as exc:
            elapsed = time.monotonic() - start_time
            raise LLMTimeoutError(
                f"Request timed out after {elapsed:.1f}s",
                provider="my-llm",
                elapsed_s=elapsed,
            ) from exc
        except httpx.RequestError as exc:
            raise LLMProviderError(
                f"Connection error: {exc}",
                provider="my-llm",
            ) from exc

        self._raise_for_status(resp)

        data = resp.json()
        choice = data["choices"][0]
        usage_data = data.get("usage", {})
        finish_reason = self._map_finish_reason(choice.get("finish_reason"))

        return LLMResponse(
            content=choice["message"]["content"],
            finish_reason=finish_reason,
            usage=TokenUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            ),
            model=data.get("model", model),
            provider=self.PROVIDER_TYPE,
            latency_ms=(time.monotonic() - start_time) * 1000,
            raw_response=data,
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        """Yield token chunks as an async generator.

        Ordering contract (enforced by this implementation):
            - All is_thinking=True chunks are emitted before any is_thinking=False chunks.
            - The final sentinel chunk has is_final=True, delta='', and carries
              aggregated usage and total latency.

        Raises:
            LLMAuthenticationError, LLMRateLimitError, LLMProviderError, LLMTimeoutError.
        """
        self._assert_initialized()
        assert self._client is not None

        model = request.model or self.DEFAULT_MODEL
        body = self._build_request_body(request, model, stream=True)
        start_time = time.monotonic()

        prompt_tokens = 0
        completion_tokens = 0

        try:
            async with self._client.stream(
                "POST",
                "/v1/chat/completions",
                json=body,
                timeout=request.timeout or self._cfg.timeout_s,
            ) as resp:
                self._raise_for_status(resp)

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[len("data: "):]
                    if payload.strip() == "[DONE]":
                        break

                    import json
                    data = json.loads(payload)
                    choice = data["choices"][0]
                    delta_text = (choice.get("delta") or {}).get("content") or ""

                    # Accumulate usage when provided inline (some APIs stream it).
                    usage_data = data.get("usage") or {}
                    if usage_data:
                        prompt_tokens = usage_data.get("prompt_tokens", prompt_tokens)
                        completion_tokens = usage_data.get("completion_tokens", completion_tokens)

                    finish_reason_raw = choice.get("finish_reason")
                    is_final = finish_reason_raw is not None

                    if delta_text:
                        yield StreamChunk(
                            delta=delta_text,
                            is_thinking=False,
                            is_final=False,
                        )

                    if is_final:
                        break

        except httpx.TimeoutException as exc:
            elapsed = time.monotonic() - start_time
            raise LLMTimeoutError(
                f"Stream timed out after {elapsed:.1f}s",
                provider="my-llm",
                elapsed_s=elapsed,
            ) from exc
        except httpx.RequestError as exc:
            raise LLMProviderError(
                f"Stream connection error: {exc}",
                provider="my-llm",
            ) from exc

        # Final sentinel chunk — always emitted, even if the stream was empty.
        yield StreamChunk(
            delta="",
            is_thinking=False,
            is_final=True,
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            model=model,
            provider=self.PROVIDER_TYPE,
            latency_ms=(time.monotonic() - start_time) * 1000,
        )

    # ── Health check ────────────────────────────────────────────────────────

    async def health_check(self) -> bool:
        """Return True if the provider is reachable, False on any failure.

        Never raises — callers use this in monitoring loops and must not crash
        on a transient failure.
        """
        if not self._initialized or self._client is None:
            return False
        try:
            resp = await self._client.get("/v1/models", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    # ── Token counting ──────────────────────────────────────────────────────

    def count_tokens(self, text: str) -> int:
        """Return a fast local token count estimate. No network call.

        Uses tiktoken's cl100k_base BPE encoding (shared with GPT-4 and most
        modern models). Falls back to a character-based estimate in air-gapped
        environments where tiktoken cannot download its BPE data.
        """
        try:
            from kitkat._internal.tokenizers import count_tokens_tiktoken
            return count_tokens_tiktoken(text)
        except Exception:
            # Fallback: ~4 characters per token (conservative approximation).
            return max(1, len(text) // 4)

    # ── Private helpers ─────────────────────────────────────────────────────

    def _build_request_body(
        self, request: LLMRequest, model: str, stream: bool
    ) -> dict[str, Any]:
        """Translate an LLMRequest into the API's JSON request format."""
        from kitkat.core.enums import Role

        system_parts = [m.content for m in request.messages if m.role == Role.SYSTEM]
        non_system = [m for m in request.messages if m.role != Role.SYSTEM]

        # Prepend system prompt as a system message (OpenAI-style format).
        messages = []
        if system_parts:
            messages.append({"role": "system", "content": "\n\n".join(system_parts)})
        messages.extend({"role": m.role.value, "content": m.content} for m in non_system)

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": stream,
        }
        if request.top_p != 1.0:
            body["top_p"] = request.top_p
        if request.stop_sequences:
            body["stop"] = request.stop_sequences
        return body

    def _raise_for_status(self, resp: httpx.Response) -> None:
        """Map HTTP error responses to KitKat exception types.

        Always map to the most specific exception available. The retry engine
        relies on the exception type — not the status code — to decide
        whether to retry.
        """
        if resp.status_code < 400:
            return  # Success — nothing to raise.

        try:
            body = resp.json()
            error_msg = (body.get("error") or {}).get("message", resp.text)
        except Exception:
            error_msg = resp.text

        if resp.status_code in (401, 403):
            raise LLMAuthenticationError(
                error_msg, status_code=resp.status_code, provider="my-llm"
            )

        if resp.status_code == 429:
            retry_after = None
            if "retry-after" in resp.headers:
                try:
                    retry_after = float(resp.headers["retry-after"])
                except ValueError:
                    pass
            raise LLMRateLimitError(
                error_msg,
                status_code=429,
                provider="my-llm",
                retry_after_s=retry_after,
            )

        if resp.status_code == 400 and any(
            kw in error_msg.lower() for kw in ("token", "context", "too long", "length")
        ):
            raise LLMTokenLimitError(
                error_msg, status_code=400, provider="my-llm"
            )

        if resp.status_code == 400 and any(
            kw in error_msg.lower() for kw in ("safety", "content policy", "blocked", "filtered")
        ):
            raise LLMContentFilterError(
                error_msg, status_code=400, provider="my-llm"
            )

        # All other HTTP errors: generic provider error.
        raise LLMProviderError(
            error_msg, status_code=resp.status_code, provider="my-llm"
        )

    @staticmethod
    def _map_finish_reason(raw: str | None) -> FinishReason:
        """Map API finish reason strings to the unified FinishReason enum."""
        mapping = {
            "stop": FinishReason.STOP,
            "length": FinishReason.LENGTH,
            "max_tokens": FinishReason.LENGTH,
            "tool_calls": FinishReason.TOOL_CALL,
            "content_filter": FinishReason.CONTENT_FILTER,
        }
        return mapping.get(raw or "", FinishReason.UNKNOWN)
```

---

## The `_init_client_only` Pattern

`_init_client_only` exists specifically for the BYOK service path. `BYOKLLMService.__aenter__` calls it instead of `initialize()` to avoid making a preflight API call for every user session.

The key difference:

| | `initialize()` | `_init_client_only()` |
|---|---|---|
| Creates HTTP client | ✅ Yes | ✅ Yes |
| Credential probe | ✅ Yes | ❌ No |
| Auth failures surface | At `initialize()` call | At first `complete()` / `stream()` call |
| Used by | Managed service startup | `BYOKLLMService.__aenter__` |

**Contract requirements for `_init_client_only`:**

- Must be **idempotent**: calling it on an already-initialized provider is a no-op.
- Must set `self._initialized = True` before returning.
- Must NOT make any network calls.

---

## Error Mapping Best Practices

The mapping in `_raise_for_status` above follows these rules:

1. **Always raise the most specific exception.** `LLMRateLimitError` over `LLMProviderError` for 429; `LLMAuthenticationError` for 401/403; `LLMTokenLimitError` for token-related 400s.

2. **Parse `retry_after_s` from the `Retry-After` header when present.** The retry engine uses this value directly instead of the exponential back-off delay, respecting the server's hint.

3. **Use `__cause__` chaining (`raise KitkatExc(...) from sdk_exc`).** This preserves the original SDK exception for debugging without exposing it to users.

4. **Non-retryable exceptions must never be wrapped in `LLMProviderError`.** If `LLMAuthenticationError` were wrapped in `LLMProviderError`, the retry engine would incorrectly retry it. Raise them directly.

5. **`health_check()` must NEVER raise.** Catch all exceptions inside `health_check()` and return `False`.

---

## Testing Your Provider

Use `pytest-asyncio` for async tests. Mock `httpx` responses with `respx` or `unittest.mock` to avoid real network calls.

```python
# tests/test_my_provider.py
import pytest
import httpx
import respx

from my_provider.provider import MyProvider, MyProviderConfig
from kitkat import LLMRequest, Message, Role, LLMAuthenticationError, LLMRateLimitError
from kitkat.core.enums import FinishReason


FAKE_COMPLETION_RESPONSE = {
    "id": "cmpl-test-001",
    "object": "chat.completion",
    "model": "my-model-v2",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from MyLLM!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


@pytest.fixture
def provider() -> MyProvider:
    return MyProvider(MyProviderConfig(api_key="test-key-123"))


@pytest.mark.asyncio
async def test_initialize_success(provider: MyProvider) -> None:
    with respx.mock(base_url="https://api.myllm.example.com") as mock:
        mock.get("/v1/models").respond(200, json={"data": []})
        await provider.initialize()
    assert provider._initialized is True
    await provider.shutdown()


@pytest.mark.asyncio
async def test_initialize_invalid_key(provider: MyProvider) -> None:
    with respx.mock(base_url="https://api.myllm.example.com") as mock:
        mock.get("/v1/models").respond(401)
        with pytest.raises(LLMAuthenticationError):
            await provider.initialize()


@pytest.mark.asyncio
async def test_complete_success(provider: MyProvider) -> None:
    with respx.mock(base_url="https://api.myllm.example.com") as mock:
        mock.get("/v1/models").respond(200, json={"data": []})
        mock.post("/v1/chat/completions").respond(200, json=FAKE_COMPLETION_RESPONSE)

        await provider.initialize()
        request = LLMRequest(
            messages=[Message(role=Role.USER, content="Hello!")],
            max_tokens=128,
        )
        response = await provider.complete(request)

    assert response.content == "Hello from MyLLM!"
    assert response.finish_reason == FinishReason.STOP
    assert response.usage.total_tokens == 15
    assert response.usage.prompt_tokens == 10
    assert response.latency_ms > 0
    await provider.shutdown()


@pytest.mark.asyncio
async def test_complete_rate_limited(provider: MyProvider) -> None:
    with respx.mock(base_url="https://api.myllm.example.com") as mock:
        mock.get("/v1/models").respond(200, json={"data": []})
        mock.post("/v1/chat/completions").respond(
            429,
            json={"error": {"message": "Rate limit exceeded"}},
            headers={"retry-after": "30"},
        )

        await provider.initialize()
        request = LLMRequest(messages=[Message(role=Role.USER, content="Hello!")])

        with pytest.raises(LLMRateLimitError) as exc_info:
            await provider.complete(request)

    assert exc_info.value.retry_after_s == 30.0
    await provider.shutdown()


@pytest.mark.asyncio
async def test_health_check_returns_false_when_uninitialized(provider: MyProvider) -> None:
    result = await provider.health_check()
    assert result is False


@pytest.mark.asyncio
async def test_context_manager(provider: MyProvider) -> None:
    with respx.mock(base_url="https://api.myllm.example.com") as mock:
        mock.get("/v1/models").respond(200, json={"data": []})
        mock.post("/v1/chat/completions").respond(200, json=FAKE_COMPLETION_RESPONSE)

        async with provider:
            assert provider._initialized is True
            request = LLMRequest(messages=[Message(role=Role.USER, content="Test")])
            response = await provider.complete(request)
            assert response.content == "Hello from MyLLM!"

    # After exiting the context manager, shutdown() has been called.
    assert provider._initialized is False
```

---

## Using Your Provider with `LLMService`

Once implemented, your provider is a drop-in replacement for any built-in provider:

```python
import asyncio
import os

from kitkat.service import create_llm_service
from kitkat import ProviderType, LLMRequest, Message, Role
from my_provider.provider import MyProvider, MyProviderConfig


async def main() -> None:
    provider = MyProvider(MyProviderConfig(
        api_key=os.environ["MY_LLM_API_KEY"],
        base_url="https://api.myllm.example.com",
    ))

    service = create_llm_service({ProviderType.OPENAI: provider})
    await service.initialize()

    response = await service.complete(
        LLMRequest(
            messages=[Message(role=Role.USER, content="What is Python?")],
            max_tokens=256,
        ),
        ProviderType.OPENAI,
    )
    print(response.content)
    print(f"Tokens: {response.usage.total_tokens}")
    await service.shutdown()


asyncio.run(main())
```

---

## Shipping as a Plugin Package

See [Plugin System](./plugins.md) for the complete guide. In brief:

**`pyproject.toml`:**

```toml
[project.entry-points."kitkat.providers"]
my-llm = "my_provider.provider:MyProvider"
```

**After `pip install my-provider-package`:**

```python
from kitkat.plugins import get_provider_class

MyProvider = get_provider_class("my-llm")
provider = MyProvider({"api_key": "sk-..."})
```

---

## Provider Checklist

Before shipping your provider, verify every item:

- [ ] `PROVIDER_TYPE`, `DEFAULT_MODEL`, `CAPABILITIES`, `RETRY_POLICY` are declared as class attributes.
- [ ] `initialize()` is idempotent (second call is a no-op).
- [ ] `_init_client_only()` is idempotent and makes no network calls.
- [ ] `shutdown()` is safe to call on an uninitialized provider.
- [ ] `complete()` calls `self._assert_initialized()` at the top.
- [ ] `stream()` emits a final sentinel chunk with `is_final=True`, `delta=""`, and aggregated `usage` + `latency_ms`.
- [ ] `health_check()` returns `False` (never raises) on any failure.
- [ ] `_raise_for_status()` maps 401/403 → `LLMAuthenticationError`, 429 → `LLMRateLimitError` with `retry_after_s`, token errors → `LLMTokenLimitError`, safety blocks → `LLMContentFilterError`.
- [ ] `count_tokens()` has a fallback for air-gapped environments.
- [ ] Tests cover: success, 401, 429, timeout, and the context manager lifecycle.

---

## Further Reading

- [Plugin System](./plugins.md) — Entry-point based discovery and `register_provider()`
- [Error Handling](./error-handling.md) — The full exception hierarchy
- [BYOK](./byok.md) — How `BYOKLLMService` uses `_init_client_only()`
- [Routing & Cache](./routing-cache.md) — Using custom providers in `LLMRouter`
- [API Reference — Core](./api-reference/core.md) — `LLMProvider`, `LLMRequest`, `LLMResponse`, `StreamChunk` API surface
