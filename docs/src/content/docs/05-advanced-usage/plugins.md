---
title: Plugin System
description: Learn how to ship a custom LLM provider as a standalone Python package and have it automatically discovered by any application that installs your package.
order: 5
---

Kitkat's plugin system lets you ship a custom LLM provider as a standalone Python package and have it automatically discovered by any application that installs your package. No configuration file is required — discovery is driven by Python's standard `importlib.metadata` entry-point mechanism, the same system used by pytest plugins and setuptools extras.

This page covers how discovery works, the full plugin registry API, and how to install and use third-party provider plugins.

## How Discovery Works

When `kitkat.providers` is first imported, it calls `_discover()` automatically. This function iterates over every installed Python package that declares an entry point in the `kitkat.providers` group. For each entry point, it:

1. Loads the class pointed to by the entry-point value.
2. Checks that the class is a valid `LLMProvider` subclass.
3. Registers it in the global `_REGISTRY` dictionary under the entry-point's name.
4. Logs a warning and **skips** any entry point that fails to load or is a duplicate — a broken third-party plugin never prevents the rest of the library from working.

Built-in providers are declared in Kitkat's own `pyproject.toml` using the same mechanism:

```toml
[project.entry-points."kitkat.providers"]
anthropic = "kitkat.providers.anthropic:AnthropicProvider"
openai    = "kitkat.providers.openai:OpenAIProvider"
gemini    = "kitkat.providers.gemini:GeminiProvider"
```

## Plugin Registry API

All registry functions are available from `kitkat.plugins`:

```python
from kitkat.plugins import (
    discover_plugins,
    get_provider_class,
    list_providers,
    register_provider,
)
```

### `list_providers() -> list[str]`

Returns a sorted list of all currently registered provider names.

```python
from kitkat.plugins import list_providers

names = list_providers()
print(names)
# ['anthropic', 'gemini', 'my-llm', 'openai']
```

### `get_provider_class(name: str) -> type[LLMProvider]`

Returns the provider class registered under `name`. Raises `KeyError` with a helpful message listing available providers when the name is not found.

```python
from kitkat.plugins import get_provider_class
from kitkat.service import create_llm_service
from kitkat import ProviderType

cls = get_provider_class("anthropic")
# <class 'kitkat.providers.anthropic.provider.AnthropicProvider'>

# Dynamic provider selection from user input
provider_name = "my-llm"
try:
    cls = get_provider_class(provider_name)
except KeyError as exc:
    print(exc)
    # No provider registered for 'my-llm'.
    # Available: ['anthropic', 'gemini', 'openai'].
    # Install the provider extra (e.g. 'pip install kitkat[my-llm]')
    # or call register_provider() before using this function.
```

### `register_provider(name: str, cls: type[LLMProvider]) -> None`

Programmatically registers a provider class without using entry points. Useful for testing, dynamic provider loading, or one-off scripts.

```python
from kitkat.plugins import register_provider
from my_custom_provider import MyProvider

register_provider("my-llm", MyProvider)

# Now usable anywhere that reads from the registry.
cls = get_provider_class("my-llm")
```

Raises `ValueError` if the name is already taken:

```python
from kitkat.plugins import register_provider
from kitkat.providers.anthropic import AnthropicProvider

try:
    register_provider("anthropic", AnthropicProvider)
except ValueError as exc:
    print(exc)
    # Provider 'anthropic' is already registered
    # (existing: 'AnthropicProvider'). Each provider name must be
    # unique across all installed packages.
```

### `discover_plugins() -> None`

Re-scans all installed packages for `kitkat.providers` entry points and registers any new providers found. This is called automatically at import time, but you can call it manually after dynamically installing a package at runtime (e.g., in a plugin marketplace scenario).

```python
import subprocess
from kitkat.plugins import discover_plugins

# Dynamically install and discover a plugin at runtime.
subprocess.run(["pip", "install", "kitkat-my-llm"], check=True)
discover_plugins()  # Re-scan now that the package is installed

from kitkat.plugins import get_provider_class
cls = get_provider_class("my-llm")
```

> **⚠️ Warning:** `register_provider` raises `ValueError` on duplicate names. If you call `discover_plugins()` after a plugin is already registered (e.g., at startup), duplicate entry points are silently skipped (logged as warnings). This means it is safe to call `discover_plugins()` multiple times.

## Shipping a Plugin Package

This section shows how to structure, package, and publish a third-party Kitkat provider plugin.

### Step 1 — Create the package structure

```
kitkat-my-llm/
├── pyproject.toml
└── src/
    └── kitkat_my_llm/
        ├── __init__.py
        └── provider.py
```

### Step 2 — Implement the provider

See [Custom Providers](./custom-provider.md) for the complete `LLMProvider` implementation guide. A minimal skeleton:

```python
# src/kitkat_my_llm/provider.py
from __future__ import annotations

from typing import Any, AsyncIterator

from kitkat.abc import LLMProvider
from kitkat.core.enums import ProviderType, FinishReason
from kitkat.core.models import (
    LLMRequest,
    LLMResponse,
    StreamChunk,
    ProviderCapabilities,
    RetryPolicy,
    TokenUsage,
)
from kitkat import LLMProviderInitError, LLMProviderError, LLMAuthenticationError


class MyLLMProvider(LLMProvider):
    PROVIDER_TYPE = ProviderType.OPENAI   # Re-use an existing type or add a new StrEnum value
    DEFAULT_MODEL = "my-model-v1"
    CAPABILITIES = ProviderCapabilities(
        supports_streaming=True,
        supports_system_prompt=True,
        supports_tool_calling=False,
        supports_vision=False,
        supports_thinking=False,
        max_context_tokens=32_768,
        provider_type=ProviderType.OPENAI,
    )
    RETRY_POLICY = RetryPolicy(max_attempts=3, base_delay_s=1.0)

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._api_key: str = config.get("api_key", "")
        self._client = None

    async def initialize(self) -> None:
        if self._initialized:
            return
        if not self._api_key:
            raise LLMProviderInitError("api_key is required", provider="my-llm")
        # Create HTTP client here (e.g., httpx.AsyncClient)
        self._client = object()   # Replace with real client
        self._initialized = True

    async def _init_client_only(self) -> None:
        if self._initialized:
            return
        self._client = object()   # Create client without probing credentials
        self._initialized = True

    async def shutdown(self) -> None:
        self._client = None
        self._initialized = False

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self._assert_initialized()
        import time
        start = time.monotonic()
        # Make your API call here using self._client
        content = "Hello from MyLLM!"
        return LLMResponse(
            content=content,
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            model=request.model or self.DEFAULT_MODEL,
            provider=self.PROVIDER_TYPE,
            latency_ms=(time.monotonic() - start) * 1000,
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        self._assert_initialized()
        words = ["Hello", " from", " MyLLM", "!"]
        for word in words:
            yield StreamChunk(delta=word, is_thinking=False, is_final=False)
        yield StreamChunk(
            delta="",
            is_final=True,
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=4, total_tokens=14),
        )

    async def health_check(self) -> bool:
        if not self._initialized or self._client is None:
            return False
        try:
            # Make a lightweight API call to verify connectivity
            return True
        except Exception:
            return False

    def count_tokens(self, text: str) -> int:
        try:
            from kitkat._internal.tokenizers import count_tokens_tiktoken
            return count_tokens_tiktoken(text)
        except Exception:
            return max(1, len(text) // 4)
```

### Step 3 — Configure the entry point in `pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "kitkat-my-llm"
version = "0.1.0"
description = "My custom LLM provider for Kitkat"
requires-python = ">=3.10"
dependencies = [
    "kitkat>=0.1.0",
    "httpx>=0.27",   # Your HTTP client
]

[project.entry-points."kitkat.providers"]
my-llm = "kitkat_my_llm.provider:MyLLMProvider"
#  ^^^          ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#  Name         Module path : Class name
```

The name (`my-llm`) is what users pass to `get_provider_class("my-llm")`. It must be unique across all installed Kitkat provider packages.

### Step 4 — Install the plugin

```bash
# Install from source (development)
pip install -e ./kitkat-my-llm

# Install from PyPI (production)
pip install kitkat-my-llm
```

After installation, the provider is automatically discovered the next time `kitkat.providers` is imported:

```python
from kitkat.plugins import list_providers, get_provider_class

print(list_providers())
# ['anthropic', 'gemini', 'my-llm', 'openai']

cls = get_provider_class("my-llm")
provider = cls({"api_key": "my-secret-key"})
```

## Using a Plugin with `LLMService`

Once discovered, a plugin provider is used exactly like a built-in provider:

```python
import asyncio
import os

from kitkat.plugins import get_provider_class
from kitkat.service import create_llm_service
from kitkat import ProviderType, LLMRequest, Message, Role


async def main() -> None:
    # Discover and instantiate the plugin provider.
    MyLLMProvider = get_provider_class("my-llm")
    provider = MyLLMProvider({"api_key": os.environ["MY_LLM_API_KEY"]})

    # Use it in the managed service.
    service = create_llm_service({ProviderType.OPENAI: provider})
    await service.initialize()

    response = await service.complete(
        LLMRequest(messages=[Message(role=Role.USER, content="Hello!")]),
        ProviderType.OPENAI,
    )
    print(response.content)
    await service.shutdown()


asyncio.run(main())
```

## Using a Plugin in `LLMRouter`

Plugin providers work seamlessly with `LLMRouter`:

```python
import asyncio
import os

from kitkat.plugins import get_provider_class
from kitkat.service import LLMRouter, RouterConfig
from kitkat.core.enums import RoutingStrategy, ProviderType
from kitkat import LLMRequest, Message, Role
from kitkat.providers.anthropic import AnthropicProvider, AnthropicConfig


async def main() -> None:
    MyLLMProvider = get_provider_class("my-llm")

    router = await LLMRouter.build(
        providers=[
            AnthropicProvider(AnthropicConfig(api_key=os.environ["ANTHROPIC_API_KEY"])),
            MyLLMProvider({"api_key": os.environ["MY_LLM_API_KEY"]}),
        ],
        config=RouterConfig(strategy=RoutingStrategy.FAILOVER),
    )

    response = await router.complete(
        LLMRequest(messages=[Message(role=Role.USER, content="Hello!")])
    )
    print(response.content)
    await router.shutdown()


asyncio.run(main())
```

## Further Reading

- [Custom Providers](./custom-provider.md) — The full `LLMProvider` implementation guide
- [Routing & Cache](./routing-cache.md) — Using plugins with `LLMRouter`
- [API Reference — Core](./api-reference/core.md) — `LLMProvider` ABC surface
- [Python entry-points specification](https://packaging.python.org/en/latest/specifications/entry-points/) — The standard this system builds on
