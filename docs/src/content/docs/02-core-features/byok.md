---
title: BYOK (Bring Your Own Key)
description: What it is, how it works, when to use it, how to configure it, and the security model that governs it.
order: 3
---

This page explains the BYOK (Bring Your Own Key) service path: what it is, how it works, when to use it, how to configure it, and the security model that governs it.


## What is BYOK?

In the managed service path, API keys live on your server and are shared across all requests. In the BYOK path, each request carries the **end user's own provider API key**. Your server never holds a shared key; users connect their own LLM accounts.

BYOK is the right architecture when:

- You are building a multi-tenant SaaS product where each user has their own Anthropic, OpenAI, or Gemini account.
- You want to avoid accumulating provider spend on behalf of users.
- Your product's value is the experience around LLM calls, not the LLM access itself.
- You need to guarantee that one user's key can never be used to serve another user's request.

---

## `BYOKLLMService`

`BYOKLLMService` is an async context manager that builds a **single-use, short-lived** provider client for one `(provider_type, api_key, model)` triple. The client is created on `__aenter__` and destroyed on `__aexit__`, regardless of whether inference raised an exception.

```python
import asyncio
from kitkat.service import BYOKLLMService
from kitkat import ProviderType, LLMRequest, Message, Role

async def handle_user_request(
    provider_type: ProviderType,
    user_api_key: str,
    model: str,
    user_message: str,
) -> str:
    async with BYOKLLMService(
        provider_type=provider_type,
        api_key=user_api_key,
        model=model,
    ) as svc:
        request = LLMRequest(
            messages=[Message(role=Role.USER, content=user_message)],
            max_tokens=512,
            temperature=0.3,
        )
        response = await svc.complete(request)
    # The provider client is destroyed here — the key is no longer in memory.
    return response.content

asyncio.run(handle_user_request(
    ProviderType.OPENAI,
    "sk-...",
    "gpt-4o-mini",
    "Summarize the Python GIL in one paragraph.",
))
```

### Constructor

```python
BYOKLLMService(
    provider_type: ProviderType,  # Which provider to use (ANTHROPIC, OPENAI, or GEMINI)
    api_key: str,                  # The caller-supplied API key
    model: str,                    # Model identifier. Empty string falls back to each provider's default.
)
```

All three arguments are required. Config validation (empty `api_key`, unsupported `provider_type`) raises `LLMProviderInitError` at construction time, before any network calls are made.

---

## Initialization: No Credential Probe

A key design decision: `BYOKLLMService.__aenter__` calls `_init_client_only()` rather than the full `initialize()`. This means:

- **No preflight token-count or model-list call.** The HTTP client is opened but no probe request is sent.
- **Authentication failures surface on the first `complete()` or `stream()` call** as `LLMAuthenticationError`, not at context entry.

**Why?** In a BYOK application, thousands of users may open sessions every minute. A preflight probe per session would double the API calls and double the latency for session setup. By skipping the probe, `__aenter__` completes in microseconds and the auth check happens exactly when the user's first request is dispatched.

---

## Complete (non-streaming)

```python
async with BYOKLLMService(ProviderType.ANTHROPIC, user_key, "claude-opus-4-5") as svc:
    request = LLMRequest(
        messages=[
            Message(role=Role.SYSTEM, content="You are a concise assistant."),
            Message(role=Role.USER, content="What is asyncio?"),
        ],
        max_tokens=256,
        temperature=0.2,
    )
    response = await svc.complete(request)
    print(response.content)
    print(f"Tokens: {response.usage.total_tokens}")
    print(f"Latency: {response.latency_ms:.0f} ms")
```

`complete()` applies the same exponential back-off retry policy as the managed path (`RetryPolicy` with `max_attempts=3` by default).

---

## Stream (token-by-token)

```python
async with BYOKLLMService(ProviderType.GEMINI, user_key, "gemini-3-flash-preview") as svc:
    request = LLMRequest(
        messages=[Message(role=Role.USER, content="Write a haiku about async programming.")],
        stream=True,
    )
    async for chunk in svc.stream(request):
        if chunk.is_thinking:
            continue  # Skip extended-thinking tokens
        print(chunk.delta, end="", flush=True)
        if chunk.is_final:
            print(f"\n\nTokens: {chunk.usage.total_tokens}")
```

> **⚠️ Warning:** You must consume the stream **inside** the `async with` block. `__aexit__` destroys the provider client, so iterating after exit will encounter a closed HTTP connection. The same applies to `asyncio.create_task` — the task must complete before the context manager exits.

---

## Error Handling in BYOK

The same exception hierarchy applies. The most common BYOK-specific error is `LLMAuthenticationError` when a user provides an invalid or revoked API key.

```python
from kitkat import (
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMTokenLimitError,
    LLMError,
    ProviderType,
    LLMRequest,
    Message,
    Role,
)
from kitkat.service import BYOKLLMService

async def safe_byok_complete(
    provider_type: ProviderType,
    user_api_key: str,
    model: str,
    prompt: str,
) -> dict:
    try:
        async with BYOKLLMService(provider_type, user_api_key, model) as svc:
            request = LLMRequest(
                messages=[Message(role=Role.USER, content=prompt)],
                max_tokens=512,
            )
            response = await svc.complete(request)
            return {"status": "ok", "content": response.content}

    except LLMAuthenticationError:
        # Map to HTTP 401 in a web framework handler.
        return {"status": "error", "code": 401, "message": "Invalid or revoked API key."}

    except LLMRateLimitError as exc:
        retry_in = exc.retry_after_s or "unknown"
        return {"status": "error", "code": 429, "message": f"Rate limited. Retry after {retry_in}s."}

    except LLMTokenLimitError:
        return {"status": "error", "code": 413, "message": "Prompt too long for this model."}

    except LLMError as exc:
        return {"status": "error", "code": exc.status_code, "message": exc.message}
```

---

## Using BYOK in a FastAPI Route

Here is a complete, production-ready FastAPI route handler using `BYOKLLMService`:

```python
import os
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from kitkat.service import BYOKLLMService
from kitkat import (
    ProviderType,
    LLMRequest,
    Message,
    Role,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMTokenLimitError,
    LLMError,
)

app = FastAPI()

class CompletionRequest(BaseModel):
    provider: str   # "anthropic", "openai", or "gemini"
    model: str
    message: str
    max_tokens: int = 512

@app.post("/v1/complete")
async def complete(
    body: CompletionRequest,
    x_api_key: str = Header(..., description="Your provider API key"),
) -> dict:
    try:
        provider_type = ProviderType(body.provider)
    except ValueError:
        raise HTTPException(400, detail=f"Unknown provider: {body.provider!r}")

    try:
        async with BYOKLLMService(provider_type, x_api_key, body.model) as svc:
            request = LLMRequest(
                messages=[Message(role=Role.USER, content=body.message)],
                max_tokens=body.max_tokens,
            )
            response = await svc.complete(request)
    except LLMAuthenticationError:
        raise HTTPException(401, detail="Invalid or revoked API key.")
    except LLMRateLimitError as exc:
        raise HTTPException(429, detail=f"Rate limited. Retry after {exc.retry_after_s}s.")
    except LLMTokenLimitError:
        raise HTTPException(413, detail="Prompt too long for the selected model.")
    except LLMError as exc:
        raise HTTPException(exc.status_code, detail=exc.message)

    return {
        "content": response.content,
        "provider": response.provider,
        "model": response.model,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
        "latency_ms": response.latency_ms,
    }
```

---

## Supported Providers

`BYOKLLMService` supports all three built-in providers:

| `provider_type` | Required extra | Key environment variable |
|---|---|---|
| `ProviderType.ANTHROPIC` | `kitkat[anthropic]` | `ANTHROPIC_API_KEY` |
| `ProviderType.OPENAI` | `kitkat[openai]` | `OPENAI_API_KEY` |
| `ProviderType.GEMINI` | `kitkat[gemini]` | `GOOGLE_API_KEY` |

Passing an unrecognized `provider_type` raises `LLMProviderError` at construction time.

> **📝 Note:** `BYOKLLMService` does not support Vertex AI mode for the Gemini provider. Vertex AI uses Application Default Credentials (ADC) rather than a user-supplied API key, which is incompatible with the BYOK model. Use the managed service path with `GeminiConfig(vertexai=True)` for Vertex AI.

---

## BYOK vs Managed: Decision Guide

| Concern | Managed (`LLMService`) | BYOK (`BYOKLLMService`) |
|---|---|---|
| API key owner | Your server | End user |
| Shared credentials | Yes | No |
| Provider billing | Your account | User's account |
| Persistent connection pool | Yes (long-lived) | No (per-request) |
| Credential probe at startup | Yes | No (deferred) |
| Routing / caching | Yes (via `LLMRouter`) | No |
| Token counting | Via `service.count_tokens()` | Not available |
| Health checks | Via `service.health_check_all()` | Not available |
| Best for | Internal services, batch jobs | Multi-tenant SaaS, user-owned keys |

---

## Security Model

- **No key persistence.** `BYOKLLMService` does not log, store, or cache the user-supplied API key. The key is only present in memory within the `async with` block.
- **Key isolation.** Each `BYOKLLMService` instance creates a separate provider client. Concurrent requests from different users are handled by separate, independent instances.
- **Fail-safe teardown.** `__aexit__` calls `provider.shutdown()` unconditionally — even if inference raised an exception — ensuring the HTTP connection pool is released and the key removed from memory.
- **No shared state.** `BYOKLLMService` does not attach to any global router or cache. Responses are returned directly to the caller without being stored anywhere.

> **🔒 Security:** Always transmit user API keys over HTTPS and never log the `X-API-Key` header or the `api_key` parameter. Consider using a secrets vault or encrypted field to store user keys at rest if your application needs to persist them.

---

## Further Reading

- [Providers](./providers.md) — Provider config classes and capabilities
- [Quick Start](./quickstart.md) — BYOK quick-start example
- [Error Handling](./error-handling.md) — Full exception hierarchy
- [FastAPI Integration](./fastapi.md) — Complete production FastAPI integration guide
- [API Reference — Service](./api-reference/service.md) — `BYOKLLMService` full API surface
