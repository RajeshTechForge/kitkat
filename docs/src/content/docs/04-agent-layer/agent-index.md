---
title: Overview
description: Kitkat's agent layer bridges the library's provider and service infrastructure into PydanticAI, giving you structured, type-safe agents with full access to multi-provider routing, BYOK, streaming, and observability.
order: 1
---

Kitkat's agent layer bridges the library's provider and service infrastructure into [PydanticAI](https://ai.pydantic.dev/), giving you structured, type-safe agents with full access to multi-provider routing, BYOK, streaming, and observability, without any boilerplate.

This section covers the four pillars of the agent layer:

| Page                                          | What you will learn                                                                   |
| --------------------------------------------- | ------------------------------------------------------------------------------------- |
| [Agent Context](./context.md)                 | `BaseAgentContext`, `RoutingTier`, and how to extend the context for your application |
| [Structured Outputs](./structured-outputs.md) | `build_structured_agent` with Pydantic output validation and custom validators        |
| [Tool Calling](./tools.md)                    | `ToolRegistry`, `@agent.tool`, and writing context-aware tools                        |

## Installation

The agent layer requires the `agents` extra:

```bash
pip install kitkat[agents]
```

This installs `pydantic-ai` alongside Kitkat's core package. Importing any symbol from `kitkat.agents` without this extra installed raises `ImportError` immediately with an actionable message.

For observability (Logfire + Langfuse), install the `observability` extra:

```bash
pip install kitkat[agents,observability]
```

## Architecture Overview

The agent layer is organized into three groups:

```
kitkat.agents
├── adapters/
│   ├── managed.py      ManagedModelAdapter  ← bridges LLMService → PydanticAI Model
│   └── byok.py         BYOKModelAdapter     ← bridges BYOKLLMService → PydanticAI Model
├── builders.py         build_chat_agent, build_structured_agent
├── context.py          BaseAgentContext, RoutingTier
├── observability.py    configure_observability
└── tools/
    └── registry.py     ToolRegistry
```

The central design principle is **separation of concerns**: Kitkat owns the provider transport layer; PydanticAI owns the agent run loop, output validation, and tool orchestration. The two adapters (`ManagedModelAdapter`, `BYOKModelAdapter`) are the seam where the two worlds meet.

## Quick Start

This example builds a chat agent backed by Anthropic via the managed service, registers a tool, and runs it.

```python
import asyncio
import os
from datetime import UTC, datetime

from pydantic_ai import RunContext

from kitkat.service import create_llm_service
from kitkat import ProviderType
from kitkat.providers.anthropic import AnthropicProvider, AnthropicConfig
from kitkat.agents import (
    ManagedModelAdapter,
    BaseAgentContext,
    ToolRegistry,
    build_chat_agent,
)


async def main() -> None:
    # 1. Build and initialize the managed service.
    service = create_llm_service({
        ProviderType.ANTHROPIC: AnthropicProvider(
            AnthropicConfig(api_key=os.environ["ANTHROPIC_API_KEY"])
        )
    })
    await service.initialize()

    # 2. Create the model adapter.
    adapter = ManagedModelAdapter(
        service=service,
        provider_type=ProviderType.ANTHROPIC,
        default_model="claude-opus-4-5",
    )

    # 3. Register tools.
    registry = ToolRegistry()

    @registry.tool
    async def get_current_time(ctx: RunContext[BaseAgentContext]) -> str:
        # Return the current UTC time as an ISO-8601 string.
        return datetime.now(UTC).isoformat()

    # 4. Build the agent and attach tools.
    agent = build_chat_agent(model=adapter, context_type=BaseAgentContext)
    registry.register_on(agent)

    # 5. Run the agent.
    ctx = BaseAgentContext(user_id="user-001")
    result = await agent.run("What time is it right now?", deps=ctx)
    print(result.data)

    await service.shutdown()


asyncio.run(main())
```

## The Two Adapters

### `ManagedModelAdapter`

Wraps a long-lived `LLMService` instance. The service is initialized once at application startup and shared across all agent runs.

```python
from kitkat.agents import ManagedModelAdapter
from kitkat import ProviderType

adapter = ManagedModelAdapter(
    service=service,                         # Initialized LLMService
    provider_type=ProviderType.ANTHROPIC,
    default_model="claude-opus-4-5",
)
```

**Best for:** Web APIs, background workers, and any application where the service lifecycle outlives a single request.

### `BYOKModelAdapter`

Wraps a per-request `BYOKLLMService`. The BYOK service must be entered via its async context manager before constructing the adapter. The adapter borrows the service for the duration of the agent run.

```python
import os
from kitkat.agents import BYOKModelAdapter, build_chat_agent, BaseAgentContext
from kitkat.service import BYOKLLMService
from kitkat import ProviderType

async def handle_user_request(user_key: str, user_message: str) -> str:
    ctx = BaseAgentContext(user_id="user-42")
    async with BYOKLLMService(ProviderType.OPENAI, user_key, "gpt-4o-mini") as byok:
        adapter = BYOKModelAdapter(byok_service=byok)
        agent = build_chat_agent(model=adapter, context_type=BaseAgentContext)
        result = await agent.run(user_message, deps=ctx)
    return result.data
```

**Best for:** Multi-tenant SaaS products where each user provides their own API key.

## Model Settings

Both adapters translate PydanticAI's `ModelSettings` into `LLMRequest` parameters. Pass settings at the agent level or override them per-run:

```python
from pydantic_ai.settings import ModelSettings

agent = build_chat_agent(model=adapter, context_type=BaseAgentContext)

# Per-run override
result = await agent.run(
    "Summarise this text in one sentence.",
    deps=ctx,
    model_settings=ModelSettings(max_tokens=128, temperature=0.0),
)
```

| `ModelSettings` key | Mapped `LLMRequest` field | Default when absent     |
| ------------------- | ------------------------- | ----------------------- |
| `max_tokens`        | `max_tokens`              | `2048`                  |
| `temperature`       | `temperature`             | `0.1`                   |
| `model`             | `model`                   | `""` (provider default) |

## Further Reading

- [Agent Context](./context.md) — `BaseAgentContext` fields and subclassing patterns
- [Structured Outputs](./structured-outputs.md) — Type-safe Pydantic output validation
- [Tool Calling](./tools.md) — `ToolRegistry` and `@agent.tool`
- [Observability](../observability.md) — Logfire and Langfuse tracing
- [BYOK](../byok.md) — Per-request user API keys
- [API Reference — Agents](../api-reference/agents.md) — Complete API surface
