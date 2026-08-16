---
title: Observability
description: Kitkat's observability layer provides automatic distributed tracing for every agent run. Every call to `agent.run()` or `agent.run_stream()` is captured as a trace span — including input messages, LLM output, tool calls, token usage, latency, and model metadata.
order: 4
---

Kitkat's observability layer provides automatic distributed tracing for every agent run. Every call to `agent.run()` or `agent.run_stream()` is captured as a trace span — including input messages, LLM output, tool calls, token usage, latency, and model metadata — with zero changes to your application code.

This page covers installation, `configure_observability`, Logfire setup, Langfuse integration, what gets traced, and practical patterns for filtering, annotating, and querying traces in production.

## Installation

```bash
pip install kitkat[agents,observability]
```

This installs:

- `pydantic-ai` — the agent runtime that PydanticAI's Logfire integration instruments.
- `logfire` — Pydantic's first-party OpenTelemetry tracing backend.
- `langfuse` — LLM observability platform with an OpenTelemetry OTLP exporter.
- `opentelemetry-exporter-otlp-proto-http` — the OTLP HTTP exporter used to forward spans to Langfuse.

> **📝 Note:** The `observability` extra requires the `agents` extra as a prerequisite. If you install `kitkat[observability]` without `kitkat[agents]`, importing `configure_observability` raises `ImportError` because `pydantic-ai` is missing.

## `configure_observability`

Call this function **once at application startup**, before any agents run. After calling it, all subsequent `agent.run()` calls are automatically instrumented.

```python
from kitkat.agents import configure_observability

configure_observability(
    logfire_token=None,               # Logfire project token. Falls back to LOGFIRE_TOKEN env var.
    langfuse_public_key=None,         # Langfuse public key. Falls back to LANGFUSE_PUBLIC_KEY env var.
    langfuse_secret_key=None,         # Langfuse secret key. Falls back to LANGFUSE_SECRET_KEY env var.
    langfuse_host="https://cloud.langfuse.com",  # Langfuse host. Falls back to LANGFUSE_HOST env var.
    service_name="kitkat",            # Service identifier tag in all traces.
    environment=None,                 # e.g. "production", "staging". Falls back to ENVIRONMENT env var or "production".
)
```

### Parameters

| Parameter             | Type          | Default                        | Description                                                                                                                                                                         |
| --------------------- | ------------- | ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `logfire_token`       | `str \| None` | `None`                         | Logfire project write token. When `None`, Logfire reads `LOGFIRE_TOKEN` from the environment. When no token is available at all, Logfire runs in local-only mode (no cloud upload). |
| `langfuse_public_key` | `str \| None` | `None`                         | Langfuse project public key. Required for Langfuse integration. Falls back to `LANGFUSE_PUBLIC_KEY` env var.                                                                        |
| `langfuse_secret_key` | `str \| None` | `None`                         | Langfuse project secret key. Required for Langfuse integration. Falls back to `LANGFUSE_SECRET_KEY` env var.                                                                        |
| `langfuse_host`       | `str`         | `"https://cloud.langfuse.com"` | Langfuse API host. Override for self-hosted Langfuse deployments. Falls back to `LANGFUSE_HOST` env var.                                                                            |
| `service_name`        | `str`         | `"kitkat"`                     | Service name tag applied to all spans. Use your application name so traces are identifiable across a multi-service deployment.                                                      |
| `environment`         | `str \| None` | `None`                         | Deployment environment label (`"production"`, `"staging"`, `"development"`). Falls back to `ENVIRONMENT` env var, then defaults to `"production"`.                                  |

### Return value

`None`. The function configures the global OpenTelemetry `TracerProvider` and calls `Agent.instrument_all()` to hook PydanticAI's tracing into it.

## Architecture

Kitkat's observability is built on OpenTelemetry and is designed to send traces to multiple backends simultaneously without conflicts.

```
agent.run() → PydanticAI instrumentation
                  │
                  ▼
           OpenTelemetry TracerProvider
           ┌──────────┬───────────────────┐
           │          │                   │
     Logfire SDK  BatchSpanProcessor  BatchSpanProcessor
      (Logfire      (→ Logfire cloud)  (→ Langfuse OTLP)
     dashboard)
```

Concretely, `configure_observability` does four things:

1. Calls `logfire.configure(service_name=..., environment=..., token=...)` to initialize the global Logfire TracerProvider.
2. Calls `Agent.instrument_all()` so every PydanticAI agent in the process is traced automatically.
3. If Langfuse credentials are present, adds a `BatchSpanProcessor` with an OTLP HTTP exporter pointing at `{langfuse_host}/api/public/otel/v1/traces`, using HTTP Basic auth (`public_key:secret_key` base64-encoded).
4. Initializes a `Langfuse` client (for SDK-level features like manual scoring and dataset management).

Errors during Langfuse configuration are logged as warnings and do not prevent the rest of the observability setup from working.

## Minimal Setup — Logfire Only

```python
import asyncio
import os

from kitkat.agents import configure_observability, build_chat_agent, BaseAgentContext, ManagedModelAdapter
from kitkat.service import create_llm_service
from kitkat import ProviderType
from kitkat.providers.anthropic import AnthropicProvider, AnthropicConfig


async def main() -> None:
    # Configure observability before creating agents.
    configure_observability(
        logfire_token=os.environ["LOGFIRE_TOKEN"],
        service_name="my-app",
        environment="development",
    )

    service = create_llm_service({
        ProviderType.ANTHROPIC: AnthropicProvider(
            AnthropicConfig(api_key=os.environ["ANTHROPIC_API_KEY"])
        )
    })
    await service.initialize()

    adapter = ManagedModelAdapter(
        service=service,
        provider_type=ProviderType.ANTHROPIC,
        default_model="claude-opus-4-5",
    )
    agent = build_chat_agent(model=adapter, context_type=BaseAgentContext)
    ctx = BaseAgentContext(user_id="user-001")

    # This call is automatically traced.
    result = await agent.run("Explain asyncio in one sentence.", deps=ctx)
    print(result.data)

    await service.shutdown()


asyncio.run(main())
```

After running this, open [Logfire](https://logfire.pydantic.dev) and you will see a trace with:

- The user prompt
- The system prompt
- The model response text
- Token counts (input + output)
- Latency in milliseconds
- Model name and provider

## Logfire + Langfuse (Dual Export)

```python
import asyncio
import os

from kitkat.agents import configure_observability, build_chat_agent, BaseAgentContext, ManagedModelAdapter
from kitkat.service import create_llm_service
from kitkat import ProviderType
from kitkat.providers.openai import OpenAIProvider, OpenAIConfig


async def main() -> None:
    configure_observability(
        logfire_token=os.environ["LOGFIRE_TOKEN"],
        langfuse_public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        langfuse_secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        langfuse_host="https://cloud.langfuse.com",
        service_name="my-app",
        environment="production",
    )

    service = create_llm_service({
        ProviderType.OPENAI: OpenAIProvider(
            OpenAIConfig(api_key=os.environ["OPENAI_API_KEY"])
        )
    })
    await service.initialize()

    adapter = ManagedModelAdapter(
        service=service,
        provider_type=ProviderType.OPENAI,
        default_model="gpt-4o",
    )
    agent = build_chat_agent(model=adapter, context_type=BaseAgentContext)
    ctx = BaseAgentContext(user_id="user-001")

    # Traces are sent to BOTH Logfire and Langfuse automatically.
    result = await agent.run("What is a Python decorator?", deps=ctx)
    print(result.data)

    await service.shutdown()


asyncio.run(main())
```

## Environment Variable Reference

All configuration can be provided via environment variables instead of constructor arguments:

```bash
export LOGFIRE_TOKEN="logfire_XXXXXXXXXXXXXX"
export LANGFUSE_PUBLIC_KEY="pk-lf-XXXXXX"
export LANGFUSE_SECRET_KEY="sk-lf-XXXXXX"
export LANGFUSE_HOST="https://cloud.langfuse.com"
export ENVIRONMENT="production"
```

With these variables set, `configure_observability()` with no arguments is sufficient:

```python
configure_observability()  # Reads all credentials from the environment
```

> **🔒 Security:** Never commit API tokens to version control. Use environment variables, a secrets manager (e.g., AWS Secrets Manager, GCP Secret Manager, HashiCorp Vault), or a `.env` file excluded from your repository.

## Self-Hosted Langfuse

For on-premises or self-hosted Langfuse deployments, set `langfuse_host` to your instance's base URL:

```python
configure_observability(
    langfuse_public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    langfuse_secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    langfuse_host="https://langfuse.internal.mycompany.com",
    service_name="my-app",
)
```

The OTLP endpoint is derived automatically as `{langfuse_host}/api/public/otel/v1/traces`.

## FastAPI Integration

In a FastAPI application, call `configure_observability` in a lifespan event to ensure it runs before the first request:

```python
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from kitkat.agents import configure_observability
from kitkat.service import create_llm_service, LLMService
from kitkat import ProviderType
from kitkat.providers.anthropic import AnthropicProvider, AnthropicConfig


_service: LLMService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _service

    # Configure observability before creating any agents.
    configure_observability(
        logfire_token=os.environ.get("LOGFIRE_TOKEN"),
        langfuse_public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
        langfuse_secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
        service_name="my-fastapi-app",
        environment=os.environ.get("ENVIRONMENT", "production"),
    )

    _service = create_llm_service({
        ProviderType.ANTHROPIC: AnthropicProvider(
            AnthropicConfig(api_key=os.environ["ANTHROPIC_API_KEY"])
        )
    })
    await _service.initialize()

    yield  # Application runs here

    await _service.shutdown()


app = FastAPI(lifespan=lifespan)
```

## What Gets Traced

PydanticAI's `Agent.instrument_all()` automatically captures the following attributes on every agent run span:

| Attribute                        | Description                                                 |
| -------------------------------- | ----------------------------------------------------------- |
| `gen_ai.system`                  | Provider identifier (`"anthropic"`, `"openai"`, `"google"`) |
| `gen_ai.request.model`           | Model identifier used for the request                       |
| `gen_ai.usage.input_tokens`      | Number of prompt tokens                                     |
| `gen_ai.usage.output_tokens`     | Number of completion tokens                                 |
| `gen_ai.response.finish_reasons` | List of finish reasons from all candidates                  |
| `logfire.span_type`              | `"llm"` for model calls, `"tool"` for tool invocations      |
| Input messages                   | Full conversation history at run start                      |
| Output content                   | The final model response text                               |
| Tool calls                       | Name, arguments, and return value for each tool invocation  |
| Latency                          | Wall-clock duration of the entire `agent.run()` call        |

> **📝 Note:** PydanticAI traces the agent run loop, including all LLM calls within a single `agent.run()` (there may be multiple if tools trigger follow-up calls). The individual Kitkat provider calls (`service.complete()`, `service.stream()`) are not separately traced by the observability module — they appear as part of the PydanticAI span.

## Checking Whether Observability Is Active

If you need to check programmatically whether the observability layer is configured before making decisions (e.g., in tests), check the Logfire configuration state:

```python
import logfire

# After configure_observability(), logfire is configured.
# Before it, logfire is in its default unconfigured state.
configured = logfire.DEFAULT_LOGFIRE_INSTANCE is not None
```

> **💡 Tip:** In tests, omit `configure_observability()` entirely. PydanticAI still works without it — traces are simply not emitted. This keeps unit tests fast and free of external dependencies.

## Further Reading

- [Agent Layer Overview](./agents/index.md) — Agents, adapters, and `Agent.instrument_all`
- [Logfire documentation](https://logfire.pydantic.dev) — Logfire project setup and dashboard
- [Langfuse documentation](https://langfuse.com/docs) — Langfuse tracing and scoring features
- [API Reference — Agents](./api-reference/agents.md) — `configure_observability` API surface
