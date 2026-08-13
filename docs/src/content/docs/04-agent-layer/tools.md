---
title: Tool Calling
description: Documentation for calling tools within PydanticAI agents built with KitKat.
order: 4
---

KitKat's agent layer integrates PydanticAI's tool system natively. Tools are async functions that the LLM can invoke during an agent run. They receive the typed context object via `RunContext`, so they have full access to your application state — database sessions, user permissions, API tokens — without global variables.

This page covers the `ToolRegistry` for programmatic bulk registration, the `@agent.tool` decorator for inline registration, context-aware tool patterns, the `prep` hook for dynamic tool definitions, and multi-agent tool sharing.

---

## Installation

```bash
pip install kitkat[agents]
```

---

## Tool Basics

A tool is an async function whose first argument is `RunContext[ContextT]`. PydanticAI inspects its signature and docstring to build the tool definition it sends to the LLM. The remaining arguments are the parameters the LLM must supply when calling the tool.

```python
import asyncio
import os

from pydantic_ai import RunContext
from kitkat.agents import (
    ManagedModelAdapter,
    BaseAgentContext,
    build_chat_agent,
)
from kitkat.service import create_llm_service
from kitkat import ProviderType
from kitkat.providers.anthropic import AnthropicProvider, AnthropicConfig


async def main() -> None:
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

    # Register a tool directly on the agent using the decorator.
    @agent.tool
    async def calculate_square_root(ctx: RunContext[BaseAgentContext], number: float) -> float:
        import math
        return math.sqrt(number)

    ctx = BaseAgentContext(user_id="user-001")
    result = await agent.run("What is the square root of 144?", deps=ctx)
    print(result.data)
    # The square root of 144 is 12.

    await service.shutdown()


asyncio.run(main())
```

**Rules for tool functions:**

- Must be `async def`.
- First argument must be `RunContext[ContextT]` — typed to your context class.
- Remaining arguments are the tool parameters. Type annotations are required; they are used to generate the JSON schema the LLM sees.
- Return type annotation is required. Return `str` for simple text, or any JSON-serializable type (`dict`, `list`, `int`, `float`, `bool`).
- The docstring becomes the tool description. Write it as if you are describing the tool to the LLM — clear, specific, and action-oriented.

---

## `ToolRegistry`

`ToolRegistry` collects tool callables before an agent is built and then bulk-registers them onto one or more agents. This is the recommended approach when:

- Multiple agents should share the same tool set.
- Tools are assembled dynamically (e.g., loaded from plugins or feature flags).
- You want to keep tool definitions in a separate module from agent construction.

### Creating a registry

```python
from kitkat.agents import ToolRegistry

registry = ToolRegistry()
```

### Registering tools

`ToolRegistry` supports two decorator forms — **bare** and **metadata-rich**:

```python
from pydantic_ai import RunContext
from kitkat.agents import ToolRegistry, BaseAgentContext

registry = ToolRegistry()


# Bare form — tool name and description come from function name and docstring.
@registry.tool
async def get_weather(ctx: RunContext[BaseAgentContext], city: str) -> str:
    # In production, call a real weather API here.
    return f"The weather in {city} is sunny, 22°C."


# Metadata form — override name and description explicitly.
@registry.tool(
    name="translate_text",
    description="Translate a piece of text from a source language to a target language.",
)
async def translate(
    ctx: RunContext[BaseAgentContext],
    text: str,
    source_lang: str,
    target_lang: str,
) -> str:
    # In production, call a translation API here.
    return f"[{target_lang}] {text}"
```

### Registering on agents

Call `registry.register_on(agent)` after building the agent. Tools are registered in the order they were added to the registry.

```python
from kitkat.agents import build_chat_agent, ManagedModelAdapter, BaseAgentContext

agent = build_chat_agent(model=adapter, context_type=BaseAgentContext)
registry.register_on(agent)
# Both get_weather and translate_text are now available to the LLM.
```

### Sharing a registry across multiple agents

```python
chat_agent = build_chat_agent(model=anthropic_adapter, context_type=BaseAgentContext)
support_agent = build_chat_agent(model=openai_adapter, context_type=BaseAgentContext)

# Both agents get the same tools.
registry.register_on(chat_agent)
registry.register_on(support_agent)
```

### Inspecting a registry

```python
print(len(registry))          # Number of registered tools
print(registry.tools)         # List of the raw callables (without metadata)
print(get_weather in registry) # True
```

### `@registry.tool` parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str \| None` | `None` | Override the tool name exposed to the LLM. When `None`, the function name is used. |
| `description` | `str \| None` | `None` | Override the tool description. When `None`, the function docstring is used. |
| `prep` | `bool` | `False` | When `True`, registers the tool with `prepare=True` in PydanticAI v2.x, enabling the tool preparation hook for dynamic tool definitions. See [Prep Tools](#prep-tools-dynamic-tool-definitions). |

---

## Context-Aware Tools

The `RunContext.deps` attribute gives tools direct access to your context object. This is how tools perform user-scoped operations without global state.

```python
from __future__ import annotations

import os
from dataclasses import dataclass, field

from pydantic_ai import RunContext
from kitkat.agents import BaseAgentContext, ToolRegistry, build_chat_agent, ManagedModelAdapter
from kitkat import ProviderType


@dataclass
class UserContext(BaseAgentContext):
    organization_id: str = ""
    allowed_topics: list[str] = field(default_factory=list)
    # In production, this would be a database session or repository.
    user_documents: dict[str, str] = field(default_factory=dict)


registry = ToolRegistry()


@registry.tool
async def search_documents(
    ctx: RunContext[UserContext],
    query: str,
) -> str:
    # ctx.deps is UserContext — fully typed, no casting needed.
    org_id = ctx.deps.organization_id
    documents = ctx.deps.user_documents

    # Naively search for the query in document values (replace with real search).
    matches = [
        f"[{doc_id}] {content[:200]}"
        for doc_id, content in documents.items()
        if query.lower() in content.lower()
    ]
    if not matches:
        return f"No documents found in organization {org_id!r} matching {query!r}."
    return "\n".join(matches)


@registry.tool
async def check_topic_permission(
    ctx: RunContext[UserContext],
    topic: str,
) -> str:
    if topic.lower() in [t.lower() for t in ctx.deps.allowed_topics]:
        return f"Access granted: topic {topic!r} is allowed for this user."
    return f"Access denied: topic {topic!r} is not in the allowed topics list."
```

---

## Tools with Structured Return Types

Tools can return any JSON-serializable type. For complex tool output, return a `dict` or a Pydantic model. PydanticAI serializes the return value and passes it back to the LLM as a tool result.

```python
from pydantic import BaseModel
from pydantic_ai import RunContext
from kitkat.agents import BaseAgentContext, ToolRegistry

registry = ToolRegistry()


class ProductInfo(BaseModel):
    product_id: str
    name: str
    price_usd: float
    in_stock: bool
    category: str


@registry.tool
async def lookup_product(
    ctx: RunContext[BaseAgentContext],
    product_id: str,
) -> dict:
    # In production, query a database or external API.
    catalog = {
        "prod-001": ProductInfo(
            product_id="prod-001",
            name="Wireless Keyboard",
            price_usd=79.99,
            in_stock=True,
            category="Electronics",
        ),
        "prod-002": ProductInfo(
            product_id="prod-002",
            name="USB-C Hub",
            price_usd=39.99,
            in_stock=False,
            category="Electronics",
        ),
    }
    product = catalog.get(product_id)
    if product is None:
        return {"error": f"Product {product_id!r} not found."}
    return product.model_dump()
```

---

## Error Handling in Tools

When a tool raises an exception, PydanticAI surfaces it as a tool error to the LLM. Raise `ModelRetry` to give the model a chance to correct the parameters and try again. For unrecoverable errors, raise any other exception to abort the agent run.

```python
from pydantic_ai import RunContext, ModelRetry
from kitkat.agents import BaseAgentContext, ToolRegistry

registry = ToolRegistry()


@registry.tool
async def divide(
    ctx: RunContext[BaseAgentContext],
    numerator: float,
    denominator: float,
) -> float:
    if denominator == 0.0:
        # ModelRetry sends the message to the LLM and gives it another attempt.
        raise ModelRetry("Division by zero is not allowed. Please provide a non-zero denominator.")
    return numerator / denominator
```

---

## Prep Tools: Dynamic Tool Definitions

When `prep=True`, PydanticAI calls the function before the agent run to determine whether the tool should be available for that particular run. Return `None` to hide the tool, or return a `ToolDefinition` to expose it (optionally with modified parameters).

This is useful for capability gating — showing tools only to users who have the right permissions.

```python
from pydantic_ai import RunContext
from pydantic_ai.tools import ToolDefinition
from kitkat.agents import ToolRegistry

registry = ToolRegistry()


@registry.tool(prep=True)
async def admin_reset_cache(
    ctx: RunContext[UserContext],
    tool_def: ToolDefinition,
) -> ToolDefinition | None:
    # Only expose this tool to admins.
    if not ctx.deps.metadata.get("is_admin", False):
        return None  # Tool hidden from non-admins
    return tool_def  # Tool exposed as-is to admins


# The actual tool implementation (registered separately without prep).
@registry.tool
async def _admin_reset_cache_impl(ctx: RunContext[UserContext]) -> str:
    # Perform the cache reset operation.
    return "Cache cleared successfully."
```

> **📝 Note:** The `prep` flag maps to PydanticAI v2.x's `prepare` parameter in `agent.tool()`. The exact signature of the prep function may differ between pydantic-ai versions — check your installed version's documentation if you encounter signature errors.

---

## Complete Example: Multi-Tool Agent

This example builds a customer support agent with three tools: document search, ticket creation, and an escalation check gated by a permission.

```python
import asyncio
import os
from dataclasses import dataclass, field

from pydantic_ai import RunContext, ModelRetry

from kitkat.service import create_llm_service
from kitkat import ProviderType
from kitkat.providers.openai import OpenAIProvider, OpenAIConfig
from kitkat.agents import (
    ManagedModelAdapter,
    BaseAgentContext,
    ToolRegistry,
    build_chat_agent,
)


@dataclass
class SupportContext(BaseAgentContext):
    organization_id: str = ""
    can_escalate: bool = False
    open_tickets: list[str] = field(default_factory=list)


registry = ToolRegistry()


@registry.tool(description="Search the knowledge base for articles matching the query.")
async def search_knowledge_base(
    ctx: RunContext[SupportContext],
    query: str,
) -> str:
    # Replace with a real search backend call.
    return f"Found 3 articles about {query!r} in the knowledge base."


@registry.tool(description="Create a new support ticket for the user's issue.")
async def create_ticket(
    ctx: RunContext[SupportContext],
    subject: str,
    priority: str,
) -> dict:
    if priority not in ("low", "medium", "high", "critical"):
        raise ModelRetry(
            f"Invalid priority {priority!r}. Must be one of: low, medium, high, critical."
        )
    ticket_id = f"TKT-{len(ctx.deps.open_tickets) + 1001}"
    ctx.deps.open_tickets.append(ticket_id)
    return {
        "ticket_id": ticket_id,
        "subject": subject,
        "priority": priority,
        "status": "open",
    }


@registry.tool(description="Escalate the issue to a senior support engineer.")
async def escalate_issue(
    ctx: RunContext[SupportContext],
    ticket_id: str,
    reason: str,
) -> str:
    if not ctx.deps.can_escalate:
        return "Escalation is not available for your account tier."
    return (
        f"Ticket {ticket_id} has been escalated. "
        f"Reason: {reason}. A senior engineer will contact you within 2 hours."
    )


async def main() -> None:
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

    agent = build_chat_agent(
        model=adapter,
        context_type=SupportContext,
        system_prompt=(
            "You are a helpful customer support agent. "
            "Use the available tools to search the knowledge base, "
            "create tickets, and escalate issues when needed."
        ),
    )
    registry.register_on(agent)

    ctx = SupportContext(
        user_id="user-007",
        organization_id="org-enterprise",
        can_escalate=True,  # This user can escalate
    )
    result = await agent.run(
        "I cannot log in to my account. I've tried resetting my password three times.",
        deps=ctx,
    )
    print(result.data)
    print(f"Open tickets: {ctx.open_tickets}")

    await service.shutdown()


asyncio.run(main())
```

---

## Further Reading

- [Agent Layer Overview](./index.md) — Architecture and the two adapters
- [Agent Context](./context.md) — `BaseAgentContext` and subclassing patterns
- [Structured Outputs](./structured-outputs.md) — Type-safe Pydantic output validation
- [Observability](../observability.md) — Tracing tool calls with Logfire and Langfuse
- [API Reference — Agents](../api-reference/agents.md) — Complete API surface
