---
title: Quick Start
description: A five-minute guide to sending your first LLM request with Kitkat.
order: 1
---

This guide walks you from a fresh install to a working LLM completion in under five minutes. By the end you will have sent a message to a real provider, read the response, streamed tokens, and handled a basic error.

> **📝 Note:** If you have not installed Kitkat yet, read [Installation](./installation.md) first. This guide assumes you have `kitkat[anthropic]` installed and an `ANTHROPIC_API_KEY` environment variable set.

## Your First Completion

The managed service path is the recommended starting point. You configure a provider once, call `initialize()`, and then call `complete()` for every request.

```python
import asyncio
import os

from kitkat.service import create_llm_service
from kitkat import ProviderType, LLMRequest, Message, Role
from kitkat.providers.anthropic import AnthropicProvider, AnthropicConfig

async def main() -> None:
    # 1. Configure the provider.
    config = AnthropicConfig(api_key=os.environ["ANTHROPIC_API_KEY"])
    provider = AnthropicProvider(config)

    # 2. Create the service and register the provider.
    service = create_llm_service({ProviderType.ANTHROPIC: provider})

    # 3. Initialize opens the connection pool and validates credentials.
    await service.initialize()

    # 4. Build a request with at least one message.
    request = LLMRequest(
        messages=[Message(role=Role.USER, content="Explain asyncio in one sentence.")],
        model="claude-opus-4-5",   # provider-specific model string
        max_tokens=128,
        temperature=0.3,
    )

    # 5. Send the request and await the full response.
    response = await service.complete(request, ProviderType.ANTHROPIC)

    print(response.content)
    print(f"Tokens used: {response.usage.total_tokens}")
    print(f"Latency: {response.latency_ms:.0f} ms")

asyncio.run(main())
```

**Expected output:**

```
asyncio is Python's built-in library for writing concurrent code using the
async/await syntax, allowing you to run multiple I/O-bound tasks cooperatively
within a single thread.
Tokens used: 48
Latency: 832 ms
```

### What each step does

| Step                           | What happens                                                                                       |
| ------------------------------ | -------------------------------------------------------------------------------------------------- |
| `AnthropicConfig(api_key=...)` | Validates the key format with Pydantic and stores it in a settings model                           |
| `AnthropicProvider(config)`    | Wraps the config — no network calls yet                                                            |
| `create_llm_service({...})`    | Constructs an `LLMService` and registers the provider mapping                                      |
| `await service.initialize()`   | Calls each provider's `initialize()`, opening connection pools                                     |
| `LLMRequest(messages=[...])`   | Validates the request fields (min 1 message, temperature in [0.0, 2.0], max_tokens ≥ 1)            |
| `await service.complete(...)`  | Dispatches the request to the correct provider, retries on transient errors, returns `LLMResponse` |

## Using a System Prompt

Add a system message to give the model a persona or set behavioural constraints.

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

    request = LLMRequest(
        messages=[
            # System messages constrain the model's behaviour.
            Message(role=Role.SYSTEM, content="You are a concise technical assistant. Reply in bullet points only."),
            # User messages carry the actual query.
            Message(role=Role.USER, content="What are the benefits of async Python?"),
        ],
        model="claude-opus-4-5",
        max_tokens=256,
    )

    response = await service.complete(request, ProviderType.ANTHROPIC)
    print(response.content)

asyncio.run(main())
```

## Streaming Responses

For interactive applications you want tokens as they arrive, not the whole response at once. Pass `stream=True` in your request and iterate the returned `AsyncIterator[StreamChunk]`.

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

    request = LLMRequest(
        messages=[Message(role=Role.USER, content="Write a short poem about async programming.")],
        model="claude-opus-4-5",
        max_tokens=200,
        stream=True,   # Enable streaming
    )

    async for chunk in service.stream(request, ProviderType.ANTHROPIC):
        if chunk.is_thinking:
            # Extended-thinking tokens arrive before answer tokens.
            # Skip them in this example.
            continue

        print(chunk.delta, end="", flush=True)

        if chunk.is_final:
            # The final sentinel chunk carries usage and latency data.
            print()
            print(f"\nTokens: {chunk.usage.total_tokens}")

asyncio.run(main())
```

> **💡 Tip:** Kitkat guarantees that all thinking chunks (`is_thinking=True`) are emitted before any answer chunks (`is_thinking=False`). The transition is one-way and never interleaved. You can safely buffer thinking output and answer output into separate strings.

## BYOK: User-Supplied API Keys

If you are building a multi-tenant application where each user brings their own API key, use `BYOKLLMService`. It accepts the key per-request and never stores it beyond the request lifetime.

```python
import asyncio

from kitkat.service import BYOKLLMService
from kitkat import ProviderType, LLMRequest, Message, Role

async def handle_user_request(user_api_key: str, user_message: str) -> str:
    # The async context manager opens and closes the connection pool automatically.
    async with BYOKLLMService(
        provider_type=ProviderType.OPENAI,
        api_key=user_api_key,
        model="gpt-4o-mini",
    ) as svc:
        request = LLMRequest(
            messages=[Message(role=Role.USER, content=user_message)]
        )
        response = await svc.complete(request)
    return response.content

asyncio.run(handle_user_request("sk-...", "Summarize the Python GIL in one paragraph."))
```

> **🔒 Security:** `BYOKLLMService` does not cache or log the user-supplied API key. Each `async with` block creates a short-lived client that is destroyed on exit. See [BYOK](./byok.md) for the full security model.

## Switching Providers

Changing providers requires only two changes: the provider instance and the `ProviderType` passed to `complete()`. Everything else — request shape, response shape, error handling — stays identical.

```python
import asyncio
import os

from kitkat.service import create_llm_service
from kitkat import ProviderType, LLMRequest, Message, Role

# --- Anthropic setup ---
from kitkat.providers.anthropic import AnthropicProvider, AnthropicConfig

# --- OpenAI setup ---
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
    await service.initialize()

    request = LLMRequest(
        messages=[Message(role=Role.USER, content="What is 2 + 2?")],
        max_tokens=32,
    )

    # Send to Anthropic
    anthropic_response = await service.complete(request, ProviderType.ANTHROPIC)
    print(f"Anthropic: {anthropic_response.content}")

    # Send the exact same request to OpenAI — no other code changes.
    openai_response = await service.complete(request, ProviderType.OPENAI)
    print(f"OpenAI: {openai_response.content}")

asyncio.run(main())
```

## Basic Error Handling

Kitkat maps every provider failure to a specific typed exception. Catch exceptions as narrowly or as broadly as your use case requires.

```python
import asyncio
import os

from kitkat.service import create_llm_service
from kitkat import (
    ProviderType,
    LLMRequest,
    Message,
    Role,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMError,
)
from kitkat.providers.openai import OpenAIProvider, OpenAIConfig

async def main() -> None:
    service = create_llm_service({
        ProviderType.OPENAI: OpenAIProvider(
            OpenAIConfig(api_key=os.environ["OPENAI_API_KEY"])
        )
    })
    await service.initialize()

    request = LLMRequest(
        messages=[Message(role=Role.USER, content="Hello!")],
        max_tokens=64,
        timeout=5.0,   # fail fast if the provider is slow
    )

    try:
        response = await service.complete(request, ProviderType.OPENAI)
        print(response.content)

    except LLMAuthenticationError as exc:
        # Invalid or revoked API key.
        print(f"Auth failed (HTTP {exc.status_code}): {exc.message}")

    except LLMRateLimitError as exc:
        # Provider returned 429 or quota exhausted.
        retry_in = exc.retry_after_s or "unknown"
        print(f"Rate limited. Retry after {retry_in}s.")

    except LLMTimeoutError as exc:
        # Request exceeded the timeout= value on LLMRequest.
        print(f"Timed out after {exc.elapsed_s:.1f}s.")

    except LLMError as exc:
        # Catch-all for any other provider error.
        print(f"Provider error [{exc.status_code}]: {exc.message}")

asyncio.run(main())
```

> **📝 Note:** Kitkat automatically retries transient errors (HTTP 408, 429, 500, 502, 503, 504) according to the default `RetryPolicy` before raising an exception. What you catch in the `except` block is only raised after all retry attempts are exhausted. See [Error Handling](./error-handling.md) for customizing retry behaviour.

## Next Steps

You now know how to send completions, stream tokens, use BYOK, and handle errors. Here is where to go next depending on your use case:

| Goal                                                            | Guide                                 |
| --------------------------------------------------------------- | ------------------------------------- |
| Understand what `LLMRequest`, `Message`, and `ProviderType` are | [Concepts](./concepts.md)             |
| Configure Anthropic, OpenAI, or Google in detail                | [Providers](./providers.md)           |
| Set up automatic failover between providers                     | [Routing & Cache](./routing-cache.md) |
| Use per-user API keys in a SaaS product                         | [BYOK](./byok.md)                     |
| Build a PydanticAI agent on top of Kitkat                       | [Agent Overview](./agents/index.md)   |

## Further Reading

- [Concepts](./concepts.md) — Deep dive into the core model
- [Installation](./installation.md) — All extras and dependency details
- [Error Handling](./error-handling.md) — Full exception hierarchy and retry configuration
- [API Reference — Core](./api-reference/core.md) — Complete API surface for `LLMRequest`, `LLMResponse`, and all core types
