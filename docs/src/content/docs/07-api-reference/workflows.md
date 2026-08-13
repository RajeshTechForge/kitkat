---
title: Workflows
description: This page documents `BaseWorkflow`, `ResearchWorkflow`, `ResearchState`, the plugin registry functions, and the `discover_plugins` API.
order: 5
---

This page documents `BaseWorkflow`, `ResearchWorkflow`, `ResearchState`, the plugin registry functions, and the `discover_plugins` API.

**Extras required:** `pip install kitkat langgraph`  
**Import paths:** `from kitkat.workflows import ...` · `from kitkat.plugins import ...`

## `BaseWorkflow`

```python
from kitkat.workflows.base import BaseWorkflow
```

Generic abstract base class for all LangGraph-based stateful workflows. `S` is the state type — typically a Pydantic `BaseModel`.

### Class Definition

```python
class BaseWorkflow(ABC, Generic[S]):
    @abstractmethod
    def build_graph(self) -> CompiledStateGraph: ...

    @abstractmethod
    async def run(self, initial_state: S | dict[str, Any]) -> S: ...
```

### Abstract Methods

| Method        | Signature                                         | Description                                                                                                                                                                        |
| ------------- | ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `build_graph` | `() -> CompiledStateGraph`                        | Construct and return the compiled LangGraph `StateGraph`. Call once at `__init__` and store the result — compiled graphs are safe to reuse across concurrent requests.             |
| `run`         | `async (initial_state: S \| dict[str, Any]) -> S` | Execute the workflow asynchronously from the initial state. Accepts either the typed state model or a plain dictionary (LangGraph validates it on entry). Returns the final state. |

### Implementation Pattern

```python
from langgraph.graph import START, StateGraph
from kitkat.workflows.base import BaseWorkflow


class MyWorkflow(BaseWorkflow[MyState]):
    def __init__(self) -> None:
        self._graph = self.build_graph()   # compile once

    def build_graph(self) -> CompiledStateGraph:
        g = StateGraph(MyState)
        g.add_node("step_a", step_a_fn)
        g.add_edge(START, "step_a")
        return g.compile()

    async def run(self, initial_state: MyState | dict) -> MyState:
        result = await self._graph.ainvoke(initial_state)
        return MyState.model_validate(result)
```

## `ResearchWorkflow`

```python
from kitkat.workflows.research import ResearchWorkflow
```

Concrete implementation of a multi-step research pipeline. Compile once at module level and reuse across requests.

### Graph Topology

```
START → plan → search ──┐
              └ retrieve ─┴─→ synthesise → END
```

`plan → search` and `plan → retrieve` run **concurrently** (fan-out). Both must complete before `synthesise` (fan-in).

### Constructor

```python
ResearchWorkflow()
```

Compiles the graph in `__init__`. No parameters.

### Methods

| Method        | Signature                                                       | Description                                                      |
| ------------- | --------------------------------------------------------------- | ---------------------------------------------------------------- |
| `build_graph` | `() -> CompiledStateGraph`                                      | Returns the compiled `StateGraph(ResearchState)`.                |
| `run`         | `async (initial_state: ResearchState \| dict) -> ResearchState` | Execute the workflow. Returns a fully validated `ResearchState`. |

## `ResearchState`

```python
from kitkat.workflows.research import ResearchState
```

Pydantic `BaseModel` state object for `ResearchWorkflow`.

| Field              | Type                       | Default | Set by            | Description                                                                |
| ------------------ | -------------------------- | ------- | ----------------- | -------------------------------------------------------------------------- |
| `user_query`       | `str`                      | `""`    | **Caller**        | The original user question. Set before calling `run()`.                    |
| `agent_context`    | `BaseAgentContext \| None` | `None`  | **Caller**        | Carries user identity through the workflow.                                |
| `research_plan`    | `list[str]`                | `[]`    | `plan_node`       | Research sub-tasks derived from `user_query`.                              |
| `search_results`   | `list[str]`                | `[]`    | `search_node`     | Search result strings.                                                     |
| `retrieved_docs`   | `list[str]`                | `[]`    | `retrieve_node`   | Retrieved document strings.                                                |
| `final_answer`     | `str`                      | `""`    | `synthesise_node` | Assembled final response.                                                  |
| `pending_approval` | `bool`                     | `False` | **Caller**        | Auth0 CIBA hook. Set `True` to signal approval is needed (future feature). |
| `approval_action`  | `str \| None`              | `None`  | **Caller**        | Describes the action awaiting approval.                                    |

## Plugin Registry

```python
from kitkat.plugins import (
    discover_plugins,
    get_provider_class,
    list_providers,
    register_provider,
)
```

### `list_providers`

```python
def list_providers() -> list[str]
```

Returns a sorted list of all currently registered provider names.

```python
list_providers()  # ['anthropic', 'gemini', 'openai']
```

### `get_provider_class`

```python
def get_provider_class(name: str) -> type[LLMProvider]
```

Returns the provider class registered under `name`.

**Raises `KeyError`** with a message listing all available providers and suggesting the relevant `pip install` command when `name` is not found.

```python
cls = get_provider_class("anthropic")   # AnthropicProvider
```

### `register_provider`

```python
def register_provider(name: str, cls: type[LLMProvider]) -> None
```

Programmatically register a provider class without using entry points.

**Raises `ValueError`** if `name` is already taken.

```python
from my_provider import MyProvider
register_provider("my-llm", MyProvider)
```

### `discover_plugins`

```python
def discover_plugins() -> None
```

Re-scan all installed packages for `kitkat.providers` entry points and register any new providers found. Called automatically at `kitkat.providers` import time. Safe to call multiple times — duplicates are logged and skipped.

Useful after dynamically installing a package at runtime.

```python
import subprocess
subprocess.run(["pip", "install", "kitkat-my-llm"], check=True)
discover_plugins()
cls = get_provider_class("my-llm")
```

## Further Reading

- [LangGraph Workflows Guide](../workflows.md) — Complete workflow building tutorial
- [Plugin System Guide](../plugins.md) — Shipping provider plugins with entry points
- [Custom Providers](../custom-provider.md) — Implementing `LLMProvider`
- [API Reference — Service](./service.md) — Using plugins with `LLMService` and `LLMRouter`
