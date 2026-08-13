---
title: Error Handling
description: KitKat uses a single, coherent exception hierarchy rooted at `KitkatError`. Every error the library raises is a subclass of this root, so you can choose between catching the base class for a broad handler or specific subclasses for precise, actionable recovery.
order: 3
---

KitKat uses a single, coherent exception hierarchy rooted at `KitkatError`. Every error the library raises is a subclass of this root, so you can choose between catching the base class for a broad handler or specific subclasses for precise, actionable recovery.

This page documents the complete exception tree, every attribute on every class, how non-retryable vs. retryable errors are distinguished, and practical patterns for catching errors in web frameworks, CLIs, and background workers.

---

## The Exception Hierarchy

```
KitkatError
└── LLMError
    ├── LLMProviderInitError   ← provider failed to start
    ├── LLMProviderError       ← generic provider-side failure
    ├── LLMAuthenticationError ← invalid / revoked API credentials
    ├── LLMRateLimitError      ← HTTP 429 / quota exceeded
    ├── LLMTimeoutError        ← request exceeded wall-clock limit
    ├── LLMTokenLimitError     ← prompt exceeds context window
    └── LLMContentFilterError  ← response blocked by safety policy
```

All exceptions are importable directly from `kitkat`:

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

---

## `KitkatError` — root

The root class for all library errors. Catching this class guarantees you catch every exception KitKat can raise.

```python
class KitkatError(Exception):
    message: str       # Human-readable description of the error
    code: str          # Machine-readable error code. Default: "KITKAT_ERROR"
    details: dict | None  # Optional structured context (provider name, etc.)
    status_code: int   # HTTP-style status code. Default: 500
```

---

## `LLMError` — base for all provider errors

`LLMError` extends `KitkatError` and is the parent of every provider-related exception. Catch this to handle all LLM failures without worrying about which specific subclass was raised.

```python
class LLMError(KitkatError):
    message: str
    provider: str | None   # Which provider raised the error ("anthropic", "openai", "gemini")
    status_code: int        # HTTP status code from the provider, or 500 if unknown
```

**Practical usage:**

```python
from kitkat import LLMError
from kitkat.service import LLMService
from kitkat import ProviderType, LLMRequest, Message, Role

async def safe_complete(service: LLMService, prompt: str) -> str:
    request = LLMRequest(
        messages=[Message(role=Role.USER, content=prompt)],
        max_tokens=512,
    )
    try:
        response = await service.complete(request, ProviderType.ANTHROPIC)
        return response.content
    except LLMError as exc:
        # Catches every provider error in one place.
        print(f"LLM call failed: [{exc.status_code}] {exc.message} (provider={exc.provider})")
        return ""
```

---

## Individual Exception Classes

### `LLMProviderInitError`

Raised during `provider.initialize()` when the provider cannot start. This happens when:
- The API key is empty or invalid at construction time (caught in `__post_init__` validation).
- The HTTP client cannot be created (network misconfiguration).
- The credential probe fails with an authentication error.

```python
class LLMProviderInitError(LLMError):
    message: str
    provider: str | None   # e.g. "anthropic"
    status_code: int        # Usually 401 or 500
```

**When you see it:** During `await service.initialize()` or `await provider.initialize()`.

**Recovery:** Fix your API key, check your `base_url`, or verify network access to the provider's API endpoint.

```python
from kitkat import LLMProviderInitError
from kitkat.providers.anthropic import AnthropicProvider, AnthropicConfig
import os

try:
    provider = AnthropicProvider(AnthropicConfig(api_key=os.environ["ANTHROPIC_API_KEY"]))
    await provider.initialize()
except LLMProviderInitError as exc:
    # Application cannot start without a working provider.
    print(f"Provider init failed: {exc.message}")
    raise SystemExit(1) from exc
```

---

### `LLMProviderError`

The generic catch-all for provider-side failures that don't fall into a more specific category. Examples: connection refused, HTTP 400 bad request, HTTP 404 model not found, HTTP 5xx server errors after retries are exhausted.

```python
class LLMProviderError(LLMError):
    message: str
    provider: str | None
    status_code: int | None   # The HTTP status code from the provider, or None
```

**Retry behaviour:** `LLMProviderError` is retried by the built-in `RetryPolicy` when the HTTP status code is in the retryable set (`{408, 429, 500, 502, 503, 504}`). After all retries are exhausted, it is re-raised.

---

### `LLMAuthenticationError`

Raised when the API key is invalid, revoked, or lacks the permissions required for the requested operation.

```python
class LLMAuthenticationError(LLMError):
    message: str
    provider: str | None
    status_code: int   # Always 401 or 403
```

> **⚠️ Warning:** `LLMAuthenticationError` is **never retried**. A different attempt with the same key will always fail. Catch this error explicitly and surface it to the operator (not the end user) to fix the credential configuration.

```python
from kitkat import LLMAuthenticationError

try:
    response = await service.complete(request, ProviderType.OPENAI)
except LLMAuthenticationError as exc:
    # In a web API: return HTTP 500 (don't expose credential failures to end users)
    print(f"CRITICAL: OpenAI API key is invalid. Rotate immediately. ({exc.message})")
    raise
```

---

### `LLMRateLimitError`

Raised when the provider returns HTTP 429 (Too Many Requests) or signals that a quota has been exceeded.

```python
class LLMRateLimitError(LLMError):
    message: str
    provider: str | None
    status_code: int | None
    retry_after_s: float | None  # Seconds to wait before retrying, from Retry-After header.
                                  # None if the header was absent.
```

**Retry behaviour:** The built-in `RetryPolicy` **does** retry `LLMRateLimitError`. When `retry_after_s` is set, it sleeps for exactly that duration instead of the computed exponential back-off delay, respecting the provider's hint. After all retries are exhausted, the error is re-raised.

```python
from kitkat import LLMRateLimitError

try:
    response = await service.complete(request, ProviderType.GEMINI)
except LLMRateLimitError as exc:
    wait = exc.retry_after_s or 60.0
    print(f"Rate limited by {exc.provider}. Retry after {wait:.0f}s.")
    # In a web API: return HTTP 429 with Retry-After header
    raise
```

---

### `LLMTimeoutError`

Raised when the request exceeds its configured timeout. This can come from `LLMRequest.timeout` (via `asyncio.wait_for`) or from the provider's SDK-level timeout.

```python
class LLMTimeoutError(LLMError):
    message: str
    provider: str | None
    status_code: int | None
    elapsed_s: float | None  # Wall-clock seconds elapsed before the timeout fired. May be
                              # None when the timeout was raised by the SDK rather than
                              # asyncio.wait_for.
```

**Retry behaviour:** `LLMTimeoutError` **is** retried by the built-in `RetryPolicy`. Each retry uses the same timeout value from `LLMRequest.timeout`.

```python
from kitkat import LLMTimeoutError

try:
    response = await service.complete(request, ProviderType.ANTHROPIC)
except LLMTimeoutError as exc:
    elapsed = f"{exc.elapsed_s:.1f}s" if exc.elapsed_s else "unknown"
    print(f"Request timed out after {elapsed}. Provider: {exc.provider}.")
    # Consider increasing LLMRequest.timeout or switching providers.
```

---

### `LLMTokenLimitError`

Raised when the prompt is too long for the model's context window.

```python
class LLMTokenLimitError(LLMError):
    message: str
    provider: str | None
    status_code: int | None
    token_count: int | None    # Estimated token count of the prompt (may be None).
    context_limit: int | None  # The model's maximum context in tokens (may be None).
```

> **⚠️ Warning:** `LLMTokenLimitError` is **never retried**. The same prompt will always exceed the context window. Truncate your prompt, split the request into chunks, or use a model with a larger context window.

```python
from kitkat import LLMTokenLimitError

try:
    response = await service.complete(request, ProviderType.GEMINI)
except LLMTokenLimitError as exc:
    limit = exc.context_limit or "unknown"
    count = exc.token_count or "unknown"
    print(f"Prompt too long: {count} tokens, model limit is {limit}.")
    # Truncate the prompt and retry manually.
```

**Prevention:** Use `service.count_tokens(provider_type, text)` or `provider.count_prompt_tokens(messages)` before submitting a request to check whether it will fit.

```python
from kitkat import ProviderType

estimated = service.count_prompt_tokens(ProviderType.ANTHROPIC, messages)
caps = service.get_capabilities(ProviderType.ANTHROPIC)

if estimated > caps.max_context_tokens:
    print(f"Prompt too long ({estimated} tokens, limit {caps.max_context_tokens}). Truncating.")
    # Truncate messages before sending.
```

---

### `LLMContentFilterError`

Raised when the provider's safety policy blocks a response. This applies to all Gemini safety categories (`SAFETY`, `RECITATION`, `BLOCKLIST`, `PROHIBITED_CONTENT`, `SPII`, `IMAGE_SAFETY`) and to OpenAI's `content_filter` finish reason.

```python
class LLMContentFilterError(LLMError):
    message: str
    provider: str | None
    status_code: int | None
```

> **⚠️ Warning:** `LLMContentFilterError` is **never retried**. The same content would be blocked on every attempt. Catch this error and return a safe default response to the user.

```python
from kitkat import LLMContentFilterError

try:
    response = await service.complete(request, ProviderType.GEMINI)
except LLMContentFilterError as exc:
    print(f"Content blocked by {exc.provider} safety policy.")
    # Return a user-facing message explaining that the request cannot be fulfilled.
    return "I'm sorry, I can't help with that request."
```

---

## Non-Retryable vs. Retryable — Quick Reference

The retry engine (`execute_with_retry`) uses this rule: three exception types are always raised immediately without sleeping; everything else is retried according to `RetryPolicy`.

| Exception | Retried? | Rationale |
|---|---|---|
| `LLMAuthenticationError` | ❌ Never | Credentials are wrong — no retry can fix this |
| `LLMTokenLimitError` | ❌ Never | Prompt is deterministically too long |
| `LLMContentFilterError` | ❌ Never | Same content would be blocked every time |
| `LLMRateLimitError` | ✅ Yes | Transient — provider asks you to wait and retry |
| `LLMTimeoutError` | ✅ Yes | Transient — network blip or slow provider |
| `LLMProviderError` | ✅ Yes (on retryable codes) | Transient server errors |
| `LLMProviderInitError` | ❌ Not applicable | Raised at startup, outside the retry loop |

---

## Complete Handler Pattern

The recommended layered handler structure for a web API:

```python
from kitkat import (
    LLMAuthenticationError,
    LLMContentFilterError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMTokenLimitError,
    LLMProviderError,
    ProviderType,
    LLMRequest,
    Message,
    Role,
)
from kitkat.service import LLMService


async def call_llm(service: LLMService, user_message: str) -> dict:
    request = LLMRequest(
        messages=[Message(role=Role.USER, content=user_message)],
        max_tokens=512,
        timeout=20.0,
    )

    try:
        response = await service.complete(request, ProviderType.ANTHROPIC)
        return {
            "status": "ok",
            "content": response.content,
            "tokens": response.usage.total_tokens,
        }

    except LLMAuthenticationError as exc:
        # Operator-facing: the API key needs to be rotated.
        # Never expose this as a user-facing message.
        return {"status": "error", "code": 500, "message": "Internal configuration error."}

    except LLMContentFilterError:
        # User-facing: request was inappropriate.
        return {"status": "error", "code": 400, "message": "That request cannot be fulfilled."}

    except LLMTokenLimitError as exc:
        limit = exc.context_limit or "the model"
        return {
            "status": "error",
            "code": 413,
            "message": f"Your message is too long (exceeds {limit} token limit). Please shorten it.",
        }

    except LLMRateLimitError as exc:
        retry_in = exc.retry_after_s or 60.0
        return {
            "status": "error",
            "code": 429,
            "message": f"Service is temporarily overloaded. Please retry in {retry_in:.0f} seconds.",
        }

    except LLMTimeoutError as exc:
        return {
            "status": "error",
            "code": 504,
            "message": "The AI service took too long to respond. Please try again.",
        }

    except LLMProviderError as exc:
        return {
            "status": "error",
            "code": exc.status_code or 502,
            "message": "The AI service returned an unexpected error. Please try again.",
        }

    except LLMError as exc:
        # Catch-all for any new LLMError subclasses added in future versions.
        return {"status": "error", "code": exc.status_code, "message": "An unexpected AI error occurred."}
```

---

## Accessing Raw Provider Errors

Every `LLMError` is raised **from** the underlying SDK exception via `raise KitkatError(...) from sdk_exc`. You can access the original SDK exception via `__cause__`:

```python
try:
    response = await provider.complete(request)
except LLMRateLimitError as exc:
    original = exc.__cause__
    print(f"Original SDK error type: {type(original).__name__}")
    # e.g. anthropic.RateLimitError or openai.RateLimitError
```

This is useful for logging detailed provider-level context without exposing it to end users.

---

## Startup Error Handling

Separate startup errors from request errors. `LLMProviderInitError` only occurs during `initialize()` — it is never raised by `complete()` or `stream()`.

```python
import logging
import sys
from kitkat import LLMProviderInitError
from kitkat.service import create_llm_service
from kitkat import ProviderType
from kitkat.providers.anthropic import AnthropicProvider, AnthropicConfig
import os

logger = logging.getLogger(__name__)

async def create_service():
    service = create_llm_service({
        ProviderType.ANTHROPIC: AnthropicProvider(
            AnthropicConfig(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
        )
    })
    try:
        await service.initialize()
    except LLMProviderInitError as exc:
        logger.critical(
            "Failed to initialize LLM service: %s (provider=%s)",
            exc.message,
            exc.provider,
        )
        sys.exit(1)
    return service
```

---

## Further Reading

- [Providers Overview](./providers.md) — Per-provider error mapping tables
- [Routing & Cache](./routing-cache.md) — How the router handles errors and circuit breaking
- [BYOK](./byok.md) — Error handling patterns specific to the BYOK path
- [Custom Providers](./custom-provider.md) — How to map SDK errors to KitKat exceptions
- [API Reference — Core](./api-reference/core.md) — Complete exception API surface
