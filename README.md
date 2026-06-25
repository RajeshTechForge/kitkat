<div align="center">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://github.com/RajeshTechForge/kitkat/blob/main/.github/images/logo-dark.png">
      <source media="(prefers-color-scheme: light)" srcset="https://github.com/RajeshTechForge/kitkat/blob/main/.github/images/logo-light.png">
      <img alt="Kitkat Logo" src="https://github.com/RajeshTechForge/kitkat/blob/main/.github/images/logo-dark.png" width="100%">
    </picture>
</div>

<div align="center">
  <h3>A modern & minimal Python library for talking to LLMs.</h3>
</div>

<div align="center">

[![PyPI version](https://img.shields.io/pypi/v/kitkat?color=3b82f6&logo=pypi&logoColor=white)](https://pypi.org/project/kitkat/)
[![Python](https://img.shields.io/pypi/pyversions/kitkat?color=3b82f6&logo=python&logoColor=white)](https://pypi.org/project/kitkat/)
[![License: MIT](https://img.shields.io/badge/License-MIT-3b82f6.svg)](./LICENSE)
[![Ruff](https://img.shields.io/badge/linting-ruff-3b82f6)](https://github.com/astral-sh/ruff)

</div>
<br>

Kitkat gives you a single, consistent interface to **Anthropic Claude**, **OpenAI GPT**, and **Google Gemini** — with streaming, BYOK (Bring Your Own Key), extended thinking, and typed responses that work identically across every provider. You can switch provider by changing two lines. Your request, response, and error handling stay exactly the same.


## Why Kitkat?

Every major LLM SDK has a different API, different streaming protocol, different error shapes, and different retry semantics. Switching providers means rewriting request code, stream parsers, and error handlers.

Kitkat solves this with a **thin, typed abstraction layer** that:

- Lets you swap providers without touching business logic
- Ships a real async-first design — not a sync wrapper with `asyncio.run`
- Stays minimal — install only the providers you actually use
- Is built to be extended — a clear ABC makes writing custom providers trivial
- Fails loudly and precisely — every error maps to a specific, typed exception


## Installation

Kitkat uses an opt-in extras model. The core package is small and dependency-free; provider SDKs are installed only when you ask for them.

```bash
# Anthropic Claude only
pip install kitkat[anthropic]

# OpenAI (and OpenAI-compatible endpoints)
pip install kitkat[openai]

# Google Gemini (including Vertex AI)
pip install kitkat[gemini]

# All three providers at once
pip install kitkat[all-providers]

# Everything
pip install kitkat[all]
```

**Requires Python 3.11+**

> **Using `uv`?**
> ```bash
> uv add kitkat[all]
> ```


## Quick Start

### Blocking completion

```python
import asyncio
from kitkat import LLMRequest, Message, Role
from kitkat.providers.anthropic import AnthropicProvider, AnthropicConfig

async def main() -> None:
    config = AnthropicConfig(api_key="sk-ant-...")
    request = LLMRequest(
        messages=[Message(role=Role.USER, content="Explain async/await in one paragraph.")],
        max_tokens=512,
    )

    async with AnthropicProvider(config) as provider:
        response = await provider.complete(request)

    print(f"Model   : {response.model}")
    print(f"Tokens  : {response.usage.total_tokens}")
    print(f"Latency : {response.latency_ms:.0f}ms")
    print(response.content)

asyncio.run(main())
```

### Switching to Gemini

```python
from kitkat import LLMRequest, Message, Role
from kitkat.providers.gemini import GeminiProvider, GeminiConfig

config = GeminiConfig(api_key="AIza...")
# Same request object, same response shape — nothing else changes.
```


## Providers

| Provider | Extra | Streaming | Thinking | Vertex AI |
|---|---|---|---|---|
| Anthropic Claude | `kitkat[anthropic]` | ✅ | ✅ | — |
| OpenAI | `kitkat[openai]` | ✅ | ✅ | — |
| Google Gemini | `kitkat[gemini]` | ✅ | ✅ | ✅ |

Each provider ships its own typed config dataclass:

```python
from kitkat.providers.openai import OpenAIConfig

config = OpenAIConfig(
    api_key="sk-...",
    model="gpt-4o",
    base_url="https://integrate.api.nvidia.com/v1",  # NVIDIA NIM
    timeout_s=120.0,
)
```


## Features

### Streaming

All providers implement true async streaming. Every chunk is a typed `StreamChunk`; the final chunk carries aggregated usage and latency.

```python
from kitkat import LLMRequest, Message, Role
from kitkat.providers.anthropic import AnthropicProvider, AnthropicConfig

async with AnthropicProvider(AnthropicConfig(api_key="...")) as provider:
    async for chunk in provider.stream(request):
        if chunk.is_final:
            print(f"\n\nDone — {chunk.usage.total_tokens} tokens")
        else:
            print(chunk.delta, end="", flush=True)
```

### Extended Thinking

Enable chain-of-thought reasoning for providers that support it (Claude, Gemini, OpenAI o-series):

```python
from kitkat import LLMRequest, Message, Role, ThinkingConfig

request = LLMRequest(
    messages=[Message(role=Role.USER, content="Solve this step by step: ...")],
    thinking=ThinkingConfig(enabled=True, effort="high"),
)

response = await provider.complete(request)
print(response.thinking_content)  # the reasoning trace
print(response.content)           # the final answer
```

### BYOK — Bring Your Own Key

The `BYOKLLMService` accepts a user-supplied API key per-request, creating a lightweight client without a pre-flight credential probe. This is designed for multi-tenant applications where each user provides their own key.

```python
from kitkat.service import BYOKLLMService
from kitkat import LLMRequest, Message, Role, ProviderType

service = BYOKLLMService()
response = await service.complete(
    request=LLMRequest(messages=[Message(role=Role.USER, content="Hello")]),
    provider_type=ProviderType.OPENAI,
    api_key="sk-user-supplied-key",
)
```

### Token Counting

Every provider exposes a synchronous `count_tokens()` method. Providers delegate to their native tokenizer where available, with a tiktoken-based fallback for models not yet supported by tiktoken:

```python
n = provider.count_tokens("How many tokens is this sentence?")
print(n)  # e.g. 8
```

### Retry Policy

Retry behaviour is configurable per-provider or per-call. Transient errors (429, 500–504) are retried with exponential back-off and optional jitter. Auth failures and token limit errors are never retried.

```python
from kitkat import RetryPolicy

response = await provider.complete_with_retry(
    request,
    policy=RetryPolicy(
        max_attempts=4,
        base_delay_s=1.0,
        max_delay_s=30.0,
        jitter=True,
    ),
)
```


## Error Handling

All exceptions are typed subclasses of `LLMError`. Catch the base class for a broad handler or specific subclasses for fine-grained recovery:

```python
from kitkat import (
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMTokenLimitError,
    LLMError,
)

try:
    response = await provider.complete(request)
except LLMAuthenticationError:
    # Bad API key — do not retry
    raise
except LLMRateLimitError as exc:
    # Honour the Retry-After header if present
    await asyncio.sleep(exc.retry_after_s or 5.0)
except LLMTokenLimitError as exc:
    # Prompt is too long for the model's context window
    print(f"Prompt has ~{exc.token_count} tokens")
except LLMTimeoutError:
    # Request exceeded timeout_s
    pass
except LLMError as exc:
    # Everything else (provider errors, connection errors, etc.)
    print(f"[{exc.provider}] {exc}")
```

**Exception hierarchy:**

```
LLMError
└── LLMProviderError
    ├── LLMProviderInitError
    ├── LLMAuthenticationError
    ├── LLMRateLimitError
    ├── LLMTokenLimitError
    ├── LLMTimeoutError
    └── LLMContentFilterError
```


## Custom Providers

Implement `LLMProvider` to add any custom or private endpoint. The library discovers providers via Python entry-points — third-party packages can ship providers without modifying Kitkat itself.

```python
from collections.abc import AsyncIterator
from kitkat import (
    LLMProvider, FinishReason, LLMRequest, LLMResponse, Message,
    ProviderCapabilities, ProviderType, StreamChunk, TokenUsage,
)

class MyProvider(LLMProvider):
    PROVIDER_TYPE = ProviderType.OPENAI  # reuse an existing slot
    DEFAULT_MODEL = "my-model-v1"
    CAPABILITIES = ProviderCapabilities(
        supports_streaming=True,
        max_context_tokens=32_768,
        provider_type=ProviderType.OPENAI,
    )

    async def initialize(self) -> None:
        self._client = MySDKClient(api_key=self._config["api_key"])
        self._initialized = True

    async def shutdown(self) -> None:
        await self._client.aclose()
        self._initialized = False

    async def _init_client_only(self) -> None:
        if not self._initialized:
            self._client = MySDKClient(api_key=self._config["api_key"])
            self._initialized = True

    async def complete(self, request: LLMRequest) -> LLMResponse:
        ...

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(delta="hello")
        yield StreamChunk(delta="", is_final=True, finish_reason=FinishReason.STOP)

    async def health_check(self) -> bool:
        return self._initialized

    def count_tokens(self, text: str) -> int:
        from kitkat._internal.tokenizers import count_tokens_tiktoken
        return count_tokens_tiktoken(text)
```

Register it via `pyproject.toml` so it's auto-discovered:

```toml
[project.entry-points."kitkat.providers"]
my-llm = "mypkg.provider:MyProvider"
```


## Contributing

Contributions are welcome — bug reports, documentation improvements, new features, and tests are all appreciated. Please read [CONTRIBUTING.md](./CONTRIBUTING.md) before opening a pull request.

**Development setup:**

```bash
git clone https://github.com/RajeshTechForge/kitkat.git
cd kitkat

# Create the virtual environment and install all dev dependencies
uv sync --extra dev

# Run the unit test suite
uv run pytest tests/unit/ -v

# Lint and format
uv run ruff check .
uv run ruff format .
```


## License

MIT © 2026 [Rajesh Mondal](https://github.com/RajeshTechForge)

See [LICENSE](./LICENSE) for the full text.
