---
title: Agent Context
description: Documentation for the context object used in PydanticAI agents built with Kitkat. The context carries user identity, routing preferences, locale, and application-specific data for tools.
order: 2
---

Every PydanticAI agent built with Kitkat uses a **context object** as its dependency injection container — the `deps` argument you pass to `agent.run()`. The context carries the user's identity, routing preferences, locale, and any application-specific data your tools need to function.

This page documents `BaseAgentContext`, the `RoutingTier` enum, the subclassing pattern for application-level fields, and how Kitkat uses the context to power dynamic system prompts.

## Installation

```bash
pip install kitkat[agents]
```

## `BaseAgentContext`

`BaseAgentContext` is a plain Python dataclass. It defines the minimum set of fields that the Kitkat agent layer reads directly. All other fields you add via subclassing are opaque to library code — they exist solely for your application tools.

```python
from dataclasses import dataclass, field
from typing import Any
from kitkat.agents import BaseAgentContext, RoutingTier
```

### Fields

| Field                    | Type             | Default               | Description                                                                                                                                                                                                                                    |
| ------------------------ | ---------------- | --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `user_id`                | `str`            | —                     | **Required. No default.** Opaque identifier for the calling user. Used for logging, tracing, and per-user routing decisions. Never defaulted — omitting it raises `TypeError` at instantiation so that silent routing mistakes are impossible. |
| `routing_tier`           | `RoutingTier`    | `RoutingTier.MANAGED` | Determines which service path handles this request: managed server-side keys, BYOK user-supplied keys, or enterprise priority queue.                                                                                                           |
| `locale`                 | `str`            | `"en"`                | IETF BCP-47 locale tag injected into the default system prompt. For example `"fr"`, `"de"`, `"ja"`. The default prompt becomes: `"You are a helpful AI assistant. User locale: en."`                                                           |
| `system_prompt_override` | `str \| None`    | `None`                | When non-`None`, replaces the library's default system prompt entirely. Useful for per-user or per-tenant prompt customization.                                                                                                                |
| `metadata`               | `dict[str, Any]` | `{}`                  | Free-form application data. Library code never reads this field. It exists solely for application-registered tools that receive the context via `RunContext[YourContext]`.                                                                     |

### Creating a context

```python
from kitkat.agents import BaseAgentContext, RoutingTier

# Minimal — just user_id is required
ctx = BaseAgentContext(user_id="user-001")

# With locale and routing tier
ctx = BaseAgentContext(
    user_id="user-001",
    routing_tier=RoutingTier.MANAGED,
    locale="fr",
)

# With a full system prompt override
ctx = BaseAgentContext(
    user_id="user-001",
    system_prompt_override=(
        "You are a specialized legal assistant. "
        "Only answer questions within the jurisdiction of French civil law. "
        "Always recommend consulting a licensed attorney."
    ),
)

# With metadata for tools
ctx = BaseAgentContext(
    user_id="user-001",
    metadata={
        "organization_id": "org-123",
        "plan_tier": "enterprise",
        "allowed_topics": ["billing", "account"],
    },
)
```

## `RoutingTier`

`RoutingTier` is a `StrEnum` that signals which service path should handle the request. It is defined in `kitkat.core.enums` and re-exported from `kitkat.agents` for convenience.

```python
from kitkat.agents import RoutingTier
# or equivalently:
from kitkat.core.enums import RoutingTier
```

| Value                    | String value   | Description                                                                                                       |
| ------------------------ | -------------- | ----------------------------------------------------------------------------------------------------------------- |
| `RoutingTier.MANAGED`    | `"managed"`    | Use `LLMService` with server-side API keys. The service is initialized at startup and shared across all requests. |
| `RoutingTier.BYOK`       | `"byok"`       | Use `BYOKLLMService` with a per-request user-supplied API key. Each request gets an ephemeral provider client.    |
| `RoutingTier.ENTERPRISE` | `"enterprise"` | Managed path with a priority queue. Reserved for future use.                                                      |

> **📝 Note:** `RoutingTier` is a signal to your application code — Kitkat itself does not automatically switch between `LLMService` and `BYOKLLMService` based on this field. Your handler or factory is responsible for checking `ctx.routing_tier` and constructing the right adapter. See the [subclassing pattern](#subclassing-baseagentcontext) below for a complete example.

## Dynamic System Prompt

When you call `build_chat_agent()` without a `system_prompt` argument, Kitkat registers a dynamic system prompt function on the agent. This function runs before each `agent.run()` call and reads the context to determine what prompt to use.

The logic is straightforward:

1. If `ctx.system_prompt_override` is set, return it verbatim.
2. Otherwise, return the default prompt with the locale injected: `"You are a helpful AI assistant. User locale: {locale}."`.

```python
from kitkat.agents import build_chat_agent, ManagedModelAdapter, BaseAgentContext
from kitkat import ProviderType

adapter = ManagedModelAdapter(service=service, provider_type=ProviderType.ANTHROPIC)

# No system_prompt → dynamic prompt is registered
agent = build_chat_agent(model=adapter, context_type=BaseAgentContext)

# The agent picks up system_prompt_override per-run
ctx_default = BaseAgentContext(user_id="user-001", locale="en")
ctx_custom = BaseAgentContext(
    user_id="user-002",
    system_prompt_override="You are a terse assistant. Answer in fewer than 10 words.",
)

result_default = await agent.run("How are you?", deps=ctx_default)
# System prompt used: "You are a helpful AI assistant. User locale: en."

result_custom = await agent.run("How are you?", deps=ctx_custom)
# System prompt used: "You are a terse assistant. Answer in fewer than 10 words."
```

When you pass a static `system_prompt` to `build_chat_agent`, the dynamic prompt registration is skipped entirely and the static string is used for every run, regardless of context fields.

```python
agent = build_chat_agent(
    model=adapter,
    context_type=BaseAgentContext,
    system_prompt="You are a Python code review assistant.",
    # Dynamic prompt NOT registered — system_prompt_override in context is ignored.
)
```

> **⚠️ Warning:** If you pass a non-empty `system_prompt` to `build_chat_agent`, `ctx.system_prompt_override` has no effect. Choose one approach: static prompt via the builder, or dynamic prompt via the context.

## Subclassing `BaseAgentContext`

For most applications, you will want to add your own fields — database sessions, feature flags, user permissions, BYOK keys, and so on. Subclass `BaseAgentContext` with `@dataclass`:

```python
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from kitkat.agents import BaseAgentContext, RoutingTier


@dataclass
class AppContext(BaseAgentContext):
    # The user's BYOK provider API key. Used when routing_tier=BYOK.
    byok_api_key: str | None = None
    # The user's organization ID for scoping database queries.
    organization_id: str = ""
    # Feature flags resolved at session start.
    feature_flags: dict[str, bool] = field(default_factory=dict)
```

Pass your subclass as the `context_type` argument to `build_chat_agent` or `build_structured_agent`. PydanticAI uses it as the `deps_type`, so `RunContext[AppContext]` in your tool functions is fully typed.

```python
from kitkat.agents import build_chat_agent, ManagedModelAdapter, ToolRegistry
from pydantic_ai import RunContext

adapter = ManagedModelAdapter(service=service, provider_type=ProviderType.ANTHROPIC)
registry = ToolRegistry()

@registry.tool
async def get_user_plan(ctx: RunContext[AppContext]) -> str:
    # ctx.deps is AppContext — all your custom fields are accessible.
    org_id = ctx.deps.organization_id
    flags = ctx.deps.feature_flags
    plan = "enterprise" if flags.get("enterprise_plan") else "standard"
    return f"Organization {org_id!r} is on the {plan} plan."

agent = build_chat_agent(model=adapter, context_type=AppContext)
registry.register_on(agent)

ctx = AppContext(
    user_id="user-007",
    organization_id="org-42",
    feature_flags={"enterprise_plan": True},
)
result = await agent.run("What plan is my organization on?", deps=ctx)
print(result.data)
# Organization 'org-42' is on the enterprise plan.
```

## Routing Tier Dispatch Pattern

A common pattern in multi-tenant applications is to inspect `ctx.routing_tier` in a factory or handler and build the right adapter on the fly:

```python
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from kitkat.agents import (
    ManagedModelAdapter,
    BYOKModelAdapter,
    build_chat_agent,
)
from kitkat.service import LLMService, BYOKLLMService
from kitkat import ProviderType
from kitkat.agents import RoutingTier


@asynccontextmanager
async def agent_for_context(
    ctx: AppContext,
    service: LLMService,
) -> AsyncIterator:
    if ctx.routing_tier == RoutingTier.BYOK:
        if not ctx.byok_api_key:
            raise ValueError("routing_tier=BYOK requires byok_api_key to be set on the context.")
        async with BYOKLLMService(
            ProviderType.OPENAI, ctx.byok_api_key, "gpt-4o-mini"
        ) as byok:
            adapter = BYOKModelAdapter(byok_service=byok)
            agent = build_chat_agent(model=adapter, context_type=AppContext)
            yield agent
    else:
        # Default: managed path
        adapter = ManagedModelAdapter(
            service=service,
            provider_type=ProviderType.ANTHROPIC,
        )
        agent = build_chat_agent(model=adapter, context_type=AppContext)
        yield agent


async def handle_request(
    ctx: AppContext,
    service: LLMService,
    message: str,
) -> str:
    async with agent_for_context(ctx, service) as agent:
        result = await agent.run(message, deps=ctx)
    return result.data
```

## Context in Streaming Runs

The context is available in tools during streaming runs exactly as it is during blocking runs. The `agent.run_stream()` API mirrors `agent.run()`:

```python
async with agent.run_stream("Tell me about Python.", deps=ctx) as streamed:
    async for text in streamed.stream_text(delta=True):
        print(text, end="", flush=True)
    print()
    final = await streamed.get_data()
    print(f"Full response: {final}")
```

## Further Reading

- [Agent Layer Overview](./index.md) — Architecture, quick start, and the two adapters
- [Tool Calling](./tools.md) — How tools receive and use `RunContext[YourContext]`
- [Structured Outputs](./structured-outputs.md) — `build_structured_agent` and Pydantic validation
- [BYOK](../byok.md) — Per-request user API keys
- [API Reference — Agents](../api-reference/agents.md) — Complete API surface
