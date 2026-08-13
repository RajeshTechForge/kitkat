---
title: Overview
description: A modern & minimal Python library for talking to LLMs.
---

**Welcome to Kitkat !**

A modern & minimal Python library for talking to LLMs.


## What is Kitkat?

Kitkat gives you a **single, consistent interface** to every major LLM provider — Anthropic Claude, OpenAI GPT, and Google Gemini — with streaming, BYOK (Bring Your Own Key), extended thinking, and typed responses that work identically across every provider.

The core philosophy is simple: your request code, your stream-parsing logic, and your error handling should never change when you swap providers. Kitkat makes that real.

```python
import os
import asyncio
from kitkat.service import create_llm_service
from kitkat import ProviderType, LLMRequest, Message, Role
from kitkat.providers.anthropic import AnthropicProvider, AnthropicConfig

async def main() -> None:
    provider = AnthropicProvider(AnthropicConfig(api_key=os.environ["ANTHROPIC_API_KEY"]))
    service = create_llm_service({ProviderType.ANTHROPIC: provider})
    await service.initialize()

    response = await service.complete(
        LLMRequest(messages=[Message(role=Role.USER, content="Hello!")]),
        ProviderType.ANTHROPIC,
    )
    print(response.content)
    # Hello! How can I help you today?

asyncio.run(main())
```

## Why Kitkat?

Every major LLM SDK ships its own API shape, its own streaming protocol, its own error types, and its own retry semantics. Switching from OpenAI to Anthropic today means rewriting request construction, stream parsers, and every error handler throughout your codebase.

Kitkat solves this with a **thin, typed abstraction layer** that:

- **Lets you swap providers by changing two lines.** Your `LLMRequest`, `LLMResponse`, and exceptions stay identical across every provider.
- **Ships a real async-first design.** Not a sync wrapper hiding `asyncio.run` underneath — every call is a native coroutine.
- **Stays minimal by default.** Install only the provider SDKs you actually use; the core package pulls in only `httpx`, `pydantic`, `tiktoken`, and `redis`.
- **Is built to be extended.** A clear abstract base class makes writing a custom provider or plugin straightforward, with no framework magic involved.
- **Fails loudly and precisely.** Every failure maps to a specific, typed exception — `LLMRateLimitError`, `LLMAuthenticationError`, and so on — so your error handlers never have to parse raw strings.


## Key Features

| Feature | Description |
|---|---|
| **Unified API** | One `LLMRequest` / `LLMResponse` shape for all providers |
| **Async-first** | Built on `asyncio` and `httpx`; no hidden blocking calls |
| **Streaming** | Token-by-token `AsyncIterator[StreamChunk]` with a strict ordering contract |
| **Extended Thinking** | Provider-agnostic `ThinkingConfig` for Claude and o-series models |
| **BYOK** | `BYOKLLMService` accepts per-request user API keys — ideal for multi-tenant SaaS |
| **Smart Routing** | Failover, round-robin, least-latency, and random strategies with per-provider circuit breakers |
| **Response Caching** | In-process LRU cache or Redis-backed caching for identical requests |
| **Agent Layer** | PydanticAI adapters with managed and BYOK context objects |
| **LangGraph Workflows** | Pre-built research workflow with an extensible base class |
| **Plugin System** | Discover and load third-party extensions via Python entry points |
| **Typed Exceptions** | Named exception classes for auth, rate limits, timeouts, token limits, and content filters |


## Architecture Overview

Kitkat is organized into clearly separated layers. Each layer depends only on the layers below it, so you can use as much or as little of the stack as you need.

```text
┌─────────────────────────────────────────────────────────┐
│  Agents & Workflows                                     │
│  kitkat.agents · kitkat.workflows                       │
│  (requires: pydantic-ai extra, langgraph extra)         │
├─────────────────────────────────────────────────────────┤
│  Service Layer                                          │
│  LLMService · BYOKLLMService · LLMRouter · LLMCache     │
├─────────────────────────────────────────────────────────┤
│  Provider Implementations                               │
│  kitkat.providers.anthropic / openai / gemini           │
│  (requires provider extras)                             │
├─────────────────────────────────────────────────────────┤
│  Abstract Provider (ABC)                                │
│  kitkat.abc.provider.LLMProvider                        │
├─────────────────────────────────────────────────────────┤
│  Core Models, Enums & Exceptions                        │
│  kitkat.core  — no provider SDK dependencies            │
└─────────────────────────────────────────────────────────┘
```

**Core** (`kitkat.core`) holds models, enums, and exceptions with no provider SDK dependencies — it is importable with only the mandatory dependencies installed. The **ABC** (`kitkat.abc`) defines the interface every provider must implement. **Provider implementations** (`kitkat.providers`) contain the actual SDK calls, completely isolated behind the ABC. The **service layer** (`kitkat.service`) routes requests, manages caching, and handles retry loops. The **agent** and **workflow** layers are fully optional extras layered on top.


## Quick Navigation

| I want to… | Go to |
|---|---|
| Install Kitkat | [Installation](./installation.md) |
| Run my first completion | [Quick Start](./quickstart.md) |
| Understand the core model | [Concepts](./concepts.md) |
| Configure a specific provider | [Providers](./providers.md) |
| Set up routing and caching | [Routing & Cache](./routing-cache.md) |
| Let users bring their own API key | [BYOK](./byok.md) |
| Build agents with PydanticAI | [Agent Overview](./agents/index.md) |
| Handle errors robustly | [Error Handling](./error-handling.md) |
| Write a custom provider | [Custom Providers](./custom-provider.md) |
| Browse the full API surface | [API Reference](./api-reference/core.md) |


## Project Links

- **Source code:** [github.com/RajeshTechForge/kitkat](https://github.com/RajeshTechForge/kitkat)
- **Bug tracker:** [GitHub Issues](https://github.com/RajeshTechForge/kitkat/issues)
- **Changelog:** [CHANGELOG.md](https://github.com/RajeshTechForge/kitkat/blob/main/CHANGELOG.md)
- **PyPI:** [pypi.org/project/kitkat](https://pypi.org/project/kitkat/)


## License

MIT © 2026 [Rajesh Mondal](https://github.com/RajeshTechForge). See the [LICENSE](https://github.com/RajeshTechForge/kitkat/blob/main/LICENSE) file for the full text.


## Further Reading

- [Quick Start](./quickstart.md) — Get your first LLM response in under five minutes
- [Concepts](./concepts.md) — Understand the core model before diving deeper
- [Installation](./installation.md) — Extras, version requirements, and environment setup
