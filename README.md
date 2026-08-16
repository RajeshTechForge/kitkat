<div align="center">
  <h1><a href="https://kitkat.rajeshmondal.com">KitKat</a></h1>
  <h3>A modern & minimal Python library for talking to LLMs.</h3>
</div>

<div align="center">

[![PyPI version](https://img.shields.io/pypi/v/kitkat?color=3b82f6&logo=pypi&logoColor=white)](https://pypi.org/project/kitkat/)
[![Python](https://img.shields.io/pypi/pyversions/kitkat?color=3b82f6&logo=python&logoColor=white)](https://pypi.org/project/kitkat/)
[![License: MIT](https://img.shields.io/badge/License-MIT-3b82f6.svg)](https://github.com/RajeshTechForge/kitkat/blob/main/LICENSE)
[![Ruff](https://img.shields.io/badge/linting-ruff-3b82f6)](https://github.com/astral-sh/ruff)

[Documentation](https://kitkat.rajeshmondal.com/docs/) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Development Setup](#-development-setup)

</div>

---

<br>

**Kitkat** is an async-first Python 3.11+ unification framework and infrastructure layer built for enterprise LLM applications, AI agents, and multi-tenant SaaS backends. 

It provides a single, strongly typed interface to major LLM providers with zero code modifications when switching models. Beyond raw provider wrappers, Kitkat delivers production-essential infrastructure out of the box—including per-request Bring Your Own Key (**BYOK**) isolation, managed failover routing, circuit breaking, caching, extended thinking support, and normalized error handling.

> [!TIP]
> **Looking for full API documentation?**  
> Visit our official documentation site: **[Kitkat Documentation Site](https://kitkat.rajeshmondal.com)**


## Key Features

- **Unified Async Multi-Provider API**: Interact with Anthropic Claude, OpenAI, and Google Gemini/Vertex AI through a single, consistent interface. Switch provider or model by changing a single parameter while keeping your request, streaming, and error handling identical.

- **Dual-Mode Architecture (Managed & BYOK)**  
  - **Managed Mode:** Application-wide pooled provider registry with automatic connection pooling and pre-flight health checks.
  - **BYOK (Bring Your Own Key) Mode:** Lightweight, single-use per-request context managers designed for SaaS backends where end-users supply their own credentials.

- **Production Resilience & Smart Routing**: Built-in failover routing, latency/cost tiering, circuit breaking (`CircuitState`), response caching (In-Memory & Redis), and exponential backoff retry policies.

- **Extended Thinking & Vendor-Specific Capabilities**: First-class `ThinkingConfig` support for reasoning models, structured Pydantic outputs, streaming chunks, and runtime capability introspection.

- **Normalized Error Hierarchy**: Translates vendor-specific HTTP and API errors into predictable, catchable exceptions (`LLMRateLimitError`, `LLMAuthenticationError`, `LLMTokenLimitError`, `LLMTimeoutError`, `LLMContentFilterError`).

- **Zero-Dependency Core & Modular Extras**: The core `kitkat` package is lightweight and dependency-free. Provider SDKs and feature layers (PydanticAI bridges, LangGraph workflows, OpenTelemetry observability) are installed on demand.

- **PEP 561 Compliant & 100% Strictly Typed**: Targeting Python 3.11+, fully typed with `mypy` strict compliance and validated using Pydantic v2 schemas.


## Why Kitkat?

| Feature / Aspect | **Kitkat** | **LiteLLM** | **LangChain / LlamaIndex** | **Raw SDKs / PydanticAI** |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Focus** | Production LLM Framework & BYOK Infra | Model Proxy & Format Translation | High-Level Agent Chains & Workflows | Provider SDKs & Agent Frameworks |
| **BYOK Multi-Tenancy** | **Native** per-request context managers with zero client leaks | Basic API key forwarding | Complex custom wrapper required | Manual HTTP client setup per request |
| **Abstraction Level** | **Minimal & Explicit** (`LLMRequest`/`LLMResponse`) | String-based proxy functions | Heavy nested objects & hidden magic | Vendor-specific SDK methods |
| **Resilience & Routing**| **Built-in** failover, circuit breakers & caching | Proxy-level failover | Third-party callback handlers | Requires custom middleware |
| **Type Safety** | **100% Mypy Strict** & Pydantic v2 schemas | Dynamic kwargs & loose typing | Dynamic dicts & heavy typing overrides | Typed per vendor, incompatible types |
| **Agent / Workflow Integration** | **Opt-in Bridges** (PydanticAI & LangGraph) | None (Proxy focus) | Native (Opinionated lock-in) | Native to specific framework |

### How Kitkat Stands Apart

1. **Explicit Over Implicit (Zero Magic):** Unlike heavy frameworks that wrap calls in complex chain objects, Kitkat uses clean, transparent dataclasses (`LLMRequest`, `LLMResponse`, `StreamChunk`). You retain complete control over your control flow.

2. **Native BYOK (Bring Your Own Key) for SaaS:** Multi-tenant applications often need to pass user-supplied API keys per request. Kitkat's `BYOKLLMService` isolates HTTP client lifecycles without credential probing overhead or memory leaks.

3. **Enterprise Resilience out of the Box:** Kitkat handles production failure modes natively through circuit breakers, multi-tier fallback routers, and customizable retry policies before errors hit your users.

4. **Provider Neutrality Without Losing Vendor Features:** Easily switch between Claude, GPT and Gemini while retaining access to provider-specific features like extended thinking budgets and structured JSON modes.


## Installation

Kitkat uses an opt-in extras model. The core library is lightweight, allowing you to install only the provider SDKs and integrations your application requires.

```bash
# Core package (zero provider dependencies)
pip install kitkat

# Provider extras
pip install kitkat[anthropic]   # Anthropic Claude
pip install kitkat[openai]      # OpenAI & OpenAI-compatible APIs
pip install kitkat[google]      # Google Gemini / Vertex AI
pip install kitkat[all-providers]

# Feature extras
pip install kitkat[agents]        # PydanticAI agent bridges
pip install kitkat[workflows]     # LangGraph workflow layer
pip install kitkat[observability] # OpenTelemetry, Logfire & Langfuse

# Install everything
pip install kitkat[all]
```

> [!TIP]
> **Using `uv`?** (Recommended)
> ```bash
> uv add "kitkat[all]"
> ```

*Requires Python 3.11+*


## Quick Start

This guide walks you from a fresh install to a working LLM completion in under five minutes. By the end you will have sent a message to a real provider, read the response, streamed tokens, and handled a basic error.

> [!NOTE]
> This guide assumes you have `kitkat[anthropic]` installed and an `ANTHROPIC_API_KEY` environment variable set.

### Your First Completion

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


## Development Setup

We welcome contributions from the community! Whether you want to add support for a new LLM provider, improve error normalization, enhance documentation, or submit bug fixes, your help is warmly appreciated.

We maintain high engineering standards: explicit Python code, 100% strict `mypy` compliance, and automated testing across Python `3.11`, `3.12`, `3.13`, and `3.14`.

### Developer Experience Stack

- **Package & Workspace Management:** [`uv`](https://github.com/astral-sh/uv) for lightning-fast environment setup.
- **Linting & Formatting:** [`ruff`](https://github.com/astral-sh/ruff) for instant code checks.
- **Type Checking:** [`mypy`](https://mypy-lang.org/) with strict type enforcement.
- **Task Runner:** [`just`](https://github.com/casey/just) command runner.

### Step-by-Step Local Setup

```bash
# 1. Fork & clone the repository
git clone https://github.com/RajeshTechForge/kitkat.git
cd kitkat

# 2. Sync all dev dependencies with uv
uv sync --extra dev

# 3. Run the unit test suite across Python versions
just test-all
# Or run tests directly with pytest:
uv run pytest

# 4. Run linter and code formatter
uv run ruff check .
uv run ruff format .

# 5. Run static type checker
uv run ruff check . --select I  # import ordering check
uv run mypy src/kitkat
```

### Looking for a Place to Start?

Check out our open issues tagged with *[`good first issue`](https://github.com/RajeshTechForge/kitkat/labels/good%20first%20issue)* or *[`help wanted`](https://github.com/RajeshTechForge/kitkat/labels/help%20wanted)* !

Please read our [CONTRIBUTING.md](./CONTRIBUTING.md) guide and [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md) before opening a pull request.


## License

Distributed under the MIT License. See [LICENSE](./LICENSE) for full details.

---

<div align="center">

Developed with ❤️ by [Rajesh Mondal](https://github.com/RajeshTechForge)

</div>
