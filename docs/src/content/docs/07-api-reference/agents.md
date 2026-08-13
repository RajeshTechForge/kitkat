---
title: Agent Layer
description: This page documents the Agent layer (adapters, context, builder functions, tool registry, and observability configuration).
order: 4
---

This page documents the Agent layer: adapters, context, builder functions, tool registry, and observability configuration.

**Extras required:** `pip install kitkat[agents]`  
**Import path:** `from kitkat.agents import ...`

## Adapters

### `ManagedModelAdapter`

```python
from kitkat.agents import ManagedModelAdapter
```

Wraps a long-lived `LLMService` as a PydanticAI `Model`. Used for managed (server-side key) inference.

#### Constructor

```python
ManagedModelAdapter(
    service: LLMService,
    provider_type: ProviderType,
    default_model: str = "",
    model_settings: ModelSettings | None = None,
)
```

| Parameter        | Type                    | Default | Description                                                                  |
| ---------------- | ----------------------- | ------- | ---------------------------------------------------------------------------- |
| `service`        | `LLMService`            | —       | **Required.** Application-wide service instance.                             |
| `provider_type`  | `ProviderType`          | —       | **Required.** Which provider to route to.                                    |
| `default_model`  | `str`                   | `""`    | Model identifier. Empty string falls back to the provider's `DEFAULT_MODEL`. |
| `model_settings` | `ModelSettings \| None` | `None`  | PydanticAI `ModelSettings` for `max_tokens`, `temperature`, etc.             |

### `BYOKModelAdapter`

```python
from kitkat.agents import BYOKModelAdapter
```

Wraps a `BYOKLLMService` as a PydanticAI `Model`. Used inside an `async with BYOKLLMService(...)` block for ephemeral, per-user-key inference.

#### Constructor

```python
BYOKModelAdapter(
    byok_service: BYOKLLMService,
    model_settings: ModelSettings | None = None,
)
```

| Parameter        | Type                    | Default | Description                                               |
| ---------------- | ----------------------- | ------- | --------------------------------------------------------- |
| `byok_service`   | `BYOKLLMService`        | —       | **Required.** Active (entered) `BYOKLLMService` instance. |
| `model_settings` | `ModelSettings \| None` | `None`  | PydanticAI `ModelSettings`.                               |

> **📝 Note:** Both adapters implement PydanticAI's `Model` protocol (`request()` and `request_stream()`). They are direct drop-ins wherever PydanticAI expects a `Model`.

## `BaseAgentContext`

```python
from kitkat.agents import BaseAgentContext
```

The `deps_type` for every Kitkat agent. Subclass with `@dataclass` to add application-specific fields.

```python
from dataclasses import dataclass
from kitkat.agents import BaseAgentContext

@dataclass
class UserContext(BaseAgentContext):
    organization_id: str = ""
```

### Fields

| Field                    | Type             | Default               | Description                                                                                                 |
| ------------------------ | ---------------- | --------------------- | ----------------------------------------------------------------------------------------------------------- |
| `user_id`                | `str`            | —                     | **Required.** No default — must be set at construction. Identifies the user for logging, tracing, and auth. |
| `locale`                 | `str`            | `"en"`                | BCP-47 locale tag injected into the default dynamic system prompt.                                          |
| `routing_tier`           | `RoutingTier`    | `RoutingTier.MANAGED` | Selects the service path (`MANAGED`, `BYOK`, or `ENTERPRISE`).                                              |
| `system_prompt_override` | `str \| None`    | `None`                | When non-`None`, replaces the entire dynamic system prompt.                                                 |
| `metadata`               | `dict[str, Any]` | `{}`                  | Freeform bag for request-scoped data (feature flags, A/B test groups, auth claims).                         |

## Builder Functions

### `build_chat_agent`

```python
from kitkat.agents import build_chat_agent
```

Creates a PydanticAI `Agent[ContextT, str]` that returns a plain string.

When `system_prompt` is empty, a dynamic prompt is registered via `@agent.system_prompt` that injects `ctx.deps.locale` and respects `ctx.deps.system_prompt_override`.

#### Signature

```python
def build_chat_agent(
    model: Model,
    context_type: type[ContextT] = BaseAgentContext,
    system_prompt: str = "",
    output_type: type[str] = str,
    output_retries: int = 1,
) -> Agent[ContextT, str]
```

| Parameter        | Type             | Default            | Description                                                            |
| ---------------- | ---------------- | ------------------ | ---------------------------------------------------------------------- |
| `model`          | `Model`          | —                  | **Required.** `ManagedModelAdapter` or `BYOKModelAdapter`.             |
| `context_type`   | `type[ContextT]` | `BaseAgentContext` | The agent's `deps_type`. Pass your `UserContext` subclass.             |
| `system_prompt`  | `str`            | `""`               | Static system prompt. Empty → dynamic prompt registered via decorator. |
| `output_type`    | `type[str]`      | `str`              | Output type. Override only when a custom non-model output is needed.   |
| `output_retries` | `int`            | `1`                | PydanticAI retry count on output validation failure.                   |

**Returns:** `Agent[ContextT, str]`

### `build_structured_agent`

```python
from kitkat.agents import build_structured_agent
```

Creates a `Agent[ContextT, BaseModel]` that validates LLM output against a Pydantic model, with automatic retries on schema validation failure.

Default system prompt when `system_prompt=""`: `"You are a helpful AI assistant. Always respond in valid JSON matching the requested schema."`

#### Signature

```python
def build_structured_agent(
    model: Model,
    output_type: type[BaseModel],
    context_type: type[ContextT] = BaseAgentContext,
    system_prompt: str = "",
    output_retries: int = 1,
    validator: Callable | None = None,
) -> Agent[ContextT, BaseModel]
```

| Parameter        | Type               | Default            | Description                                                                                                                         |
| ---------------- | ------------------ | ------------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| `model`          | `Model`            | —                  | **Required.** `ManagedModelAdapter` or `BYOKModelAdapter`.                                                                          |
| `output_type`    | `type[BaseModel]`  | —                  | **Required.** Pydantic `BaseModel` subclass.                                                                                        |
| `context_type`   | `type[ContextT]`   | `BaseAgentContext` | The agent's `deps_type`.                                                                                                            |
| `system_prompt`  | `str`              | `""`               | Static system prompt.                                                                                                               |
| `output_retries` | `int`              | `1`                | Retry count on Pydantic validation failure. Each retry appends the error to the conversation.                                       |
| `validator`      | `Callable \| None` | `None`             | Optional post-validation hook following the pydantic-ai `output_validator` protocol. Raise `ModelRetry` to trigger another attempt. |

**Returns:** `Agent[ContextT, BaseModel]`

## `ToolRegistry`

```python
from kitkat.agents import ToolRegistry
```

Collects tool callables and bulk-registers them on one or more agents.

### Constructor

```python
registry = ToolRegistry()
```

### `@registry.tool` Decorator

Two forms:

```python
# Bare — uses function name and docstring
@registry.tool
async def my_tool(ctx: RunContext[BaseAgentContext], query: str) -> str: ...

# With metadata override
@registry.tool(name="search", description="Search the knowledge base.")
async def _search(ctx: RunContext[BaseAgentContext], query: str) -> str: ...
```

| Parameter     | Type          | Default | Description                                                                                                  |
| ------------- | ------------- | ------- | ------------------------------------------------------------------------------------------------------------ |
| `name`        | `str \| None` | `None`  | Override tool name. `None` → function `__name__`.                                                            |
| `description` | `str \| None` | `None`  | Override tool description. `None` → function docstring.                                                      |
| `prep`        | `bool`        | `False` | Register as a prep tool. The function receives `ToolDefinition` and returns it (exposed) or `None` (hidden). |

### Methods & Operators

| Method               | Description                                                                               |
| -------------------- | ----------------------------------------------------------------------------------------- |
| `register_on(agent)` | Register all tools in the registry onto `agent`. Tools are registered in insertion order. |
| `__len__()`          | Number of registered tools.                                                               |
| `__contains__(fn)`   | `True` if the callable is in the registry.                                                |

### Properties

| Property | Type             | Description                                       |
| -------- | ---------------- | ------------------------------------------------- |
| `tools`  | `list[Callable]` | Registered callables (without metadata wrappers). |

## `configure_observability`

```python
from kitkat.agents import configure_observability
```

**Extras required:** `pip install kitkat[agents,observability]`

Configures Logfire and (optionally) Langfuse tracing for all PydanticAI agents. Calls `Agent.instrument_all()` automatically.

#### Signature

```python
def configure_observability(
    *,
    logfire_token: str | None = None,
    langfuse_public_key: str | None = None,
    langfuse_secret_key: str | None = None,
    langfuse_host: str = "https://cloud.langfuse.com",
    service_name: str = "kitkat",
    environment: str | None = None,
) -> None
```

| Parameter             | Type          | Default                        | Env var fallback      | Description                                                     |
| --------------------- | ------------- | ------------------------------ | --------------------- | --------------------------------------------------------------- |
| `logfire_token`       | `str \| None` | `None`                         | `LOGFIRE_TOKEN`       | Logfire project write token. When absent, Logfire runs locally. |
| `langfuse_public_key` | `str \| None` | `None`                         | `LANGFUSE_PUBLIC_KEY` | Required for Langfuse OTel export.                              |
| `langfuse_secret_key` | `str \| None` | `None`                         | `LANGFUSE_SECRET_KEY` | Required for Langfuse OTel export.                              |
| `langfuse_host`       | `str`         | `"https://cloud.langfuse.com"` | `LANGFUSE_HOST`       | Langfuse API host. Override for self-hosted.                    |
| `service_name`        | `str`         | `"kitkat"`                     | —                     | Service name tag on all spans.                                  |
| `environment`         | `str \| None` | `None`                         | `ENVIRONMENT`         | Deployment environment label. Defaults to `"production"`.       |

**Returns:** `None`. Errors during Langfuse setup are logged as warnings and do not prevent Logfire from working.

## Further Reading

- [Agent Layer Overview](../agents/index.md) — Architecture and adapters
- [Agent Context](../agents/context.md) — `BaseAgentContext` subclassing patterns
- [Structured Outputs](../agents/structured-outputs.md) — `build_structured_agent` guide
- [Tool Calling](../agents/tools.md) — `ToolRegistry` and `@agent.tool` patterns
- [Observability](../observability.md) — Logfire + Langfuse setup guide
