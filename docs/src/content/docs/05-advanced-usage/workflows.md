---
title: LangGraph Workflows
description: Learn how to build multi-step, stateful agentic pipelines with Kitkat's workflow layer on top of LangGraph.
order: 2
---

Kitkat's workflow layer provides a thin, opinionated scaffold on top of [LangGraph](https://github.com/langchain-ai/langgraph) for building multi-step, stateful agentic pipelines. The `BaseWorkflow` abstract class defines a consistent interface — `build_graph()` and `async run()` — so all your workflows are testable, composable, and interchangeable.

This page covers installation, the `BaseWorkflow` interface, the built-in `ResearchWorkflow` as a reference implementation, how to build your own workflow from scratch, state management with Pydantic models, conditional edges and fan-out/fan-in patterns, and how to wire Kitkat providers into graph nodes.

> **📝 Note:** The workflow layer requires LangGraph to be installed separately. It is not included in any Kitkat extra. Install it with `pip install langgraph`.

## Installation

```bash
pip install kitkat langgraph
```

If you also need Kitkat's agent layer inside your workflow nodes:

```bash
pip install kitkat[agents] langgraph
```

## `BaseWorkflow`

`BaseWorkflow[S]` is a generic abstract base class where `S` is your state type (typically a Pydantic `BaseModel`). It declares two abstract methods that every concrete workflow must implement.

```python
from kitkat.workflows.base import BaseWorkflow
```

### `build_graph() -> CompiledStateGraph`

Constructs and returns a compiled LangGraph `StateGraph`. Call this once — in `__init__` or at module level — and store the result. Compiled graphs are safe to reuse across concurrent requests.

### `async run(initial_state: S | dict[str, Any]) -> S`

Executes the workflow asynchronously from the given initial state. Accepts either the typed state model or a plain dictionary (LangGraph validates it against the state schema on entry). Returns the final state after all nodes have completed.

```python
from abc import abstractmethod
from typing import Any, Generic, TypeVar
from langgraph.graph.state import CompiledStateGraph

S = TypeVar("S")

class BaseWorkflow(ABC, Generic[S]):
    @abstractmethod
    def build_graph(self) -> CompiledStateGraph: ...

    @abstractmethod
    async def run(self, initial_state: S | dict[str, Any]) -> S: ...
```

## Built-in: `ResearchWorkflow`

Kitkat ships a `ResearchWorkflow` as a complete, runnable reference implementation. It demonstrates:

- Pydantic state models
- Fan-out parallelism (plan → search + retrieve simultaneously)
- Fan-in convergence (search + retrieve → synthesise)
- Conditional edges for human-in-the-loop approval hooks

### Graph topology

```
START → plan → search ──┐
              └ retrieve ─┴─→ synthesise → (conditional) → END
```

The `plan` node runs first. Its output fans out to `search` and `retrieve` concurrently. Both results fan in to `synthesise`, which produces the final answer. A conditional edge from `synthesise` is pre-wired for a future human-approval hook (currently always routes to `END`).

### `ResearchState`

```python
from kitkat.workflows.research import ResearchState
from kitkat.agents import BaseAgentContext

state = ResearchState(
    user_query="How does Python's asyncio event loop work?",
    agent_context=BaseAgentContext(user_id="user-001"),
)
```

| Field              | Type                       | Description                                                                                                  |
| ------------------ | -------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `user_query`       | `str`                      | The user's original question. Set this before calling `run()`.                                               |
| `research_plan`    | `list[str]`                | Populated by `plan_node`. Contains the research sub-tasks.                                                   |
| `search_results`   | `list[str]`                | Populated by `search_node`. Contains search result strings.                                                  |
| `retrieved_docs`   | `list[str]`                | Populated by `retrieve_node`. Contains retrieved document strings.                                           |
| `final_answer`     | `str`                      | Populated by `synthesise_node`. The assembled final response.                                                |
| `agent_context`    | `BaseAgentContext \| None` | Carries user identity and routing tier through the workflow.                                                 |
| `pending_approval` | `bool`                     | Auth0 CIBA hook. When `True`, the conditional edge will route to a `request_approval` node (future feature). |
| `approval_action`  | `str \| None`              | Describes the action awaiting approval.                                                                      |

### Running the workflow

```python
import asyncio
from kitkat.workflows.research import ResearchWorkflow, ResearchState
from kitkat.agents import BaseAgentContext


async def main() -> None:
    # Compile the graph once.
    workflow = ResearchWorkflow()

    initial_state = ResearchState(
        user_query="Explain the Python GIL and its impact on concurrency.",
        agent_context=BaseAgentContext(user_id="user-001"),
    )

    final_state = await workflow.run(initial_state)

    print("Research plan:")
    for step in final_state.research_plan:
        print(f"  - {step}")

    print("\nSearch results:")
    for result in final_state.search_results:
        print(f"  - {result}")

    print("\nFinal answer:")
    print(final_state.final_answer)


asyncio.run(main())
```

## Building a Custom Workflow

This section walks through building a complete custom workflow step by step: a three-node pipeline that validates a user query, calls an LLM, and formats the output.

### Step 1 — Define the state model

Use a Pydantic `BaseModel` as your state. LangGraph validates state transitions against the schema.

```python
from pydantic import BaseModel, Field


class LLMPipelineState(BaseModel):
    # Input fields — set before calling run()
    raw_query: str = ""
    user_id: str = ""

    # Intermediate fields — set by nodes
    cleaned_query: str = ""
    validation_error: str = ""

    # Output fields — set by the final node
    llm_response: str = ""
    token_count: int = 0
    completed: bool = False
```

### Step 2 — Write the node functions

Each node is an `async def` that receives the current state and returns a dictionary of state updates. Return only the keys you want to change — LangGraph merges your updates into the existing state.

```python
import os
from typing import Any

from kitkat.service import create_llm_service, LLMService
from kitkat import ProviderType, LLMRequest, Message, Role
from kitkat.providers.anthropic import AnthropicProvider, AnthropicConfig


# Nodes receive the full current state.
async def validate_node(state: LLMPipelineState) -> dict[str, Any]:
    query = state.raw_query.strip()
    if not query:
        return {"validation_error": "Query cannot be empty."}
    if len(query) > 2000:
        return {"validation_error": f"Query too long ({len(query)} chars, max 2000)."}
    return {"cleaned_query": query}


# Nodes can close over external dependencies (service, config, etc.)
def make_llm_node(service: LLMService) -> Any:
    async def llm_node(state: LLMPipelineState) -> dict[str, Any]:
        if state.validation_error:
            # Skip LLM call if validation failed.
            return {}
        request = LLMRequest(
            messages=[Message(role=Role.USER, content=state.cleaned_query)],
            model="claude-opus-4-5",
            max_tokens=512,
        )
        response = await service.complete(request, ProviderType.ANTHROPIC)
        return {
            "llm_response": response.content,
            "token_count": response.usage.total_tokens,
        }
    return llm_node


async def format_node(state: LLMPipelineState) -> dict[str, Any]:
    if state.validation_error:
        return {"completed": True}  # Mark complete even on validation failure
    formatted = f"**Response for user {state.user_id!r}:**\n\n{state.llm_response}"
    return {"llm_response": formatted, "completed": True}
```

### Step 3 — Define a routing function

Routing functions receive the current state and return a string — the name of the next node, or `END` to terminate. They are used with `add_conditional_edges`.

```python
from langgraph.graph import END


def route_after_validate(state: LLMPipelineState) -> str:
    if state.validation_error:
        # Skip LLM, go straight to format (which will handle the error).
        return "format"
    return "llm"
```

### Step 4 — Implement `BaseWorkflow`

```python
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from kitkat.workflows.base import BaseWorkflow


class LLMPipelineWorkflow(BaseWorkflow[LLMPipelineState]):

    def __init__(self, service: LLMService) -> None:
        self._service = service
        # Compile once at construction time.
        self._graph: CompiledStateGraph = self.build_graph()

    def build_graph(self) -> CompiledStateGraph:
        llm_node = make_llm_node(self._service)

        g = StateGraph(LLMPipelineState)

        g.add_node("validate", validate_node)
        g.add_node("llm", llm_node)
        g.add_node("format", format_node)

        g.add_edge(START, "validate")

        # Conditional edge: after validate, route to "llm" or "format".
        g.add_conditional_edges(
            "validate",
            route_after_validate,
            {"llm": "llm", "format": "format"},
        )

        g.add_edge("llm", "format")
        g.add_edge("format", END)

        return g.compile()

    async def run(
        self,
        initial_state: LLMPipelineState | dict[str, Any],
    ) -> LLMPipelineState:
        result = await self._graph.ainvoke(initial_state)
        return LLMPipelineState.model_validate(result)
```

### Step 5 — Run it

```python
import asyncio
import os
from kitkat.providers.anthropic import AnthropicProvider, AnthropicConfig


async def main() -> None:
    service = create_llm_service({
        ProviderType.ANTHROPIC: AnthropicProvider(
            AnthropicConfig(api_key=os.environ["ANTHROPIC_API_KEY"])
        )
    })
    await service.initialize()

    workflow = LLMPipelineWorkflow(service=service)

    # Happy path
    final_state = await workflow.run(
        LLMPipelineState(raw_query="What is the GIL?", user_id="user-001")
    )
    print(f"Completed: {final_state.completed}")
    print(f"Tokens: {final_state.token_count}")
    print(final_state.llm_response)

    # Validation failure path
    error_state = await workflow.run(
        LLMPipelineState(raw_query="", user_id="user-002")
    )
    print(f"Validation error: {error_state.validation_error}")
    print(f"Completed: {error_state.completed}")

    await service.shutdown()


asyncio.run(main())
```

## Fan-Out / Fan-In Parallelism

LangGraph executes multiple edges from a single source node concurrently. To fan out:

```python
# Both "search" and "retrieve" run after "plan" completes — concurrently.
g.add_edge("plan", "search")
g.add_edge("plan", "retrieve")

# Both "search" and "retrieve" must complete before "synthesise" runs.
g.add_edge("search", "synthesise")
g.add_edge("retrieve", "synthesise")
```

> **🚀 Performance:** Fan-out parallelism is one of LangGraph's most powerful features. Use it whenever two nodes are independent — for example, calling two different APIs or running two independent LLM calls for a cross-check.

> **⚠️ Warning:** When multiple nodes write to the same state field concurrently, their updates are merged in an arbitrary order. Avoid having parallel nodes write to the same field. Design your state so each parallel branch writes to distinct fields that the convergence node reads from.

## Human-in-the-Loop Approval

`ResearchWorkflow` pre-wires an Auth0 CIBA (Client-Initiated Backchannel Authentication) approval hook. The `should_request_approval` conditional edge currently always returns `END`, but the state model carries two fields for the future approval flow:

```python
state = ResearchState(
    user_query="Delete all records from the database",
    pending_approval=True,           # Signals that human approval is required
    approval_action="database_delete",  # Describes what needs approval
)
```

When `should_request_approval` is updated to check `state.pending_approval`, it will return `"request_approval"` instead of `END`, routing to a node that pauses the workflow and triggers an Auth0 CIBA push notification to the user's device.

## Error Handling in Nodes

Nodes do not have built-in retry logic. Handle errors inside node functions and record them in the state for downstream nodes to inspect:

```python
from kitkat import LLMError


async def resilient_llm_node(state: LLMPipelineState) -> dict[str, Any]:
    try:
        request = LLMRequest(
            messages=[Message(role=Role.USER, content=state.cleaned_query)],
            max_tokens=512,
        )
        response = await service.complete(request, ProviderType.ANTHROPIC)
        return {"llm_response": response.content}
    except LLMError as exc:
        # Record the error in state so format_node can surface it.
        return {"validation_error": f"LLM call failed: {exc.message}"}
```

## Testing Workflows

Compile the graph in your test and invoke it directly with a known initial state:

```python
import asyncio
import pytest
from kitkat.workflows.research import ResearchWorkflow, ResearchState
from kitkat.agents import BaseAgentContext


@pytest.mark.asyncio
async def test_research_workflow_happy_path() -> None:
    workflow = ResearchWorkflow()

    initial = ResearchState(
        user_query="What is Python?",
        agent_context=BaseAgentContext(user_id="test-user"),
    )
    result = await workflow.run(initial)

    assert result.completed is True or result.final_answer  # final_answer is set
    assert len(result.research_plan) > 0
    assert len(result.search_results) > 0
```

## Further Reading

- [Agent Layer Overview](./agents/index.md) — Using Kitkat agents inside workflow nodes
- [Providers Overview](./providers.md) — Injecting `LLMService` into workflows
- [Error Handling](./error-handling.md) — Handling `LLMError` inside nodes
- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) — Full LangGraph API reference
- [API Reference — Workflows](./api-reference/workflows.md) — `BaseWorkflow` and `ResearchWorkflow` API
