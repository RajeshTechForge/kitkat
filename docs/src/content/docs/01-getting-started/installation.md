---
title: Installation
description: How to install Kitkat, optional extras, and verify your installation
order: 2
---

This page explains every way to install Kitkat, which optional extras to choose, how to verify your installation, and what each dependency is responsible for.

## Requirements

| Requirement      | Minimum version                                         |
| ---------------- | ------------------------------------------------------- |
| Python           | 3.11                                                    |
| pip              | 23.0+ (or `uv`)                                         |
| Operating system | Linux, macOS, Windows (CPython and PyPy both supported) |

> **📝 Note:** Kitkat requires Python 3.11 or newer because it relies on `StrEnum` (added in 3.11) and several `typing` features that were stabilized in that release. Python 3.12, 3.13, and 3.14 are all fully tested in CI.

## Installing Kitkat

### Core package only

The core package contains the domain models, enums, exceptions, the abstract provider base class, the service layer, and the HTTP internals. It does **not** include any provider SDK — those are opt-in extras.

```bash
pip install kitkat
```

Or with `uv`:

```bash
uv add kitkat
```

### Provider extras

Each supported provider is an opt-in extra. Install only the ones your application uses.

```bash
# Anthropic Claude
pip install kitkat[anthropic]

# OpenAI GPT (and any OpenAI-compatible endpoint)
pip install kitkat[openai]

# Google Gemini (including Vertex AI)
pip install kitkat[google]

# All three providers at once
pip install kitkat[all-providers]
```

> **💡 Tip:** If you are unsure which provider you will use, start with `kitkat[anthropic]`. The Anthropic provider is the most feature-complete and supports streaming and extended thinking out of the box.

### Feature extras

These extras enable additional capabilities on top of the core library.

```bash
# PydanticAI agent adapters
pip install kitkat[agents]

# LangGraph workflow layer
pip install kitkat[workflows]

# Observability integrations (Logfire, Langfuse, OpenTelemetry)
pip install kitkat[observability]
```

### Install everything

```bash
pip install kitkat[all]
```

The `all` bundle includes `all-providers`, `agents`, `workflows`, and `observability`.

With `uv`:

```bash
uv add "kitkat[all]"
```

## Development installation

If you are contributing to Kitkat or hacking on the source, use the `dev` extra to install the full test and lint toolchain.

```bash
git clone https://github.com/RajeshTechForge/kitkat.git
cd kitkat

# Install all dev dependencies into a managed virtual environment
uv sync --extra dev
```

### Running tests

```bash
# Unit tests (no network)
uv run pytest tests/unit

# Lint
uv run ruff check .

# Type check
uv run mypy src/kitkat
```

> **📝 Note:** Integration tests hit real provider APIs and require environment variables (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`). Run them with `INTEGRATION_TESTS=1 uv run pytest tests/` only when those keys are available.

## Verifying the installation

After installing, confirm that the package is importable and check its version:

```python
import kitkat

print(kitkat.__version__)
# 0.6.0
```

You can also inspect what is available in the top-level namespace:

```python
import kitkat

# Core enums
print(kitkat.ProviderType.ANTHROPIC)   # anthropic
print(kitkat.Role.USER)                # user

# Core models
print(kitkat.LLMRequest)               # <class 'kitkat.core.models.LLMRequest'>
print(kitkat.LLMResponse)              # <class 'kitkat.core.models.LLMResponse'>

# Exception hierarchy root
print(kitkat.LLMError.__mro__)
```

## Environment variables

Kitkat never hard-codes API keys. The recommended approach is to pass credentials through environment variables and read them at initialization time.

```bash
# For the Anthropic provider
export ANTHROPIC_API_KEY="sk-ant-..."

# For the OpenAI provider
export OPENAI_API_KEY="sk-..."

# For the Google provider
export GOOGLE_API_KEY="AIza..."
```

> **🔒 Security:** Never commit API keys to source control. Use a secrets manager or a `.env` file excluded from version control. Kitkat's provider config classes use `pydantic-settings`, which automatically reads from environment variables — you do not need to pass keys explicitly if they are set in the environment.

## Further Reading

- [Quick Start](./quickstart.md) — Your first completion in five minutes
- [Concepts](./concepts.md) — Core models, enums, and the request/response lifecycle
- [Providers](./providers.md) — Detailed configuration for each provider
