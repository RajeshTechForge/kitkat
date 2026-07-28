"""Multi-step research workflow.

Topology: START → plan → [parallel: search + retrieve] → synthesise → END

Auth0 CIBA hooks are pre-wired as stubs:
  - ResearchState.pending_approval: set True to trigger approval flow
  - should_request_approval(): today returns END; after Auth0 returns
    "request_approval" node name when pending_approval is True
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from kitkat.workflows.base import BaseWorkflow

try:
    from langgraph.graph import END, START, StateGraph
except ImportError as exc:
    raise ImportError(
        "ResearchWorkflow requires the 'workflows' extra. "
        "Install with: pip install kitkat[workflows]"
    ) from exc

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from kitkat.agents.context import BaseAgentContext


class ResearchState(BaseModel):
    """Pydantic state model for the research workflow.

    Using Pydantic ensures type safety and automatic validation as state
    transitions between graph nodes.
    """

    user_query: str = ""
    research_plan: list[str] = Field(default_factory=list)
    search_results: list[str] = Field(default_factory=list)
    retrieved_docs: list[str] = Field(default_factory=list)
    final_answer: str = ""
    agent_context: BaseAgentContext | None = None

    # Auth0 CIBA hook: set True when action requires human approval
    pending_approval: bool = False
    approval_action: str | None = None


async def plan_node(state: ResearchState) -> dict[str, Any]:
    """Generates a research plan based on the user query."""
    plan = [f"Search: {state.user_query}", "Retrieve docs", "Synthesise"]
    return {"research_plan": plan}


async def search_node(state: ResearchState) -> dict[str, Any]:
    """Executes search based on the plan."""
    # Future: use agent_context to filter results by user_id
    results = [f"Result: {s}" for s in state.research_plan]
    return {"search_results": results}


async def retrieve_node(state: ResearchState) -> dict[str, Any]:
    """Retrieves relevant documents."""
    # Future: filter by agent_context.user_id via Auth0 FGA
    return {"retrieved_docs": ["[Doc 1]", "[Doc 2]"]}


async def synthesise_node(state: ResearchState) -> dict[str, Any]:
    """Synthesises the final answer from search results and retrieved docs."""
    combined = "\n".join(state.search_results + state.retrieved_docs)
    return {"final_answer": f"Based on research: {combined[:300]}..."}


def should_request_approval(state: ResearchState) -> str:
    """Conditional edge router for human-in-the-loop approval."""
    # Today: always END
    # After Auth0: return "request_approval" if state.pending_approval
    return END


class ResearchWorkflow(BaseWorkflow[ResearchState]):
    """Concrete implementation of a multi-step research workflow."""

    def __init__(self) -> None:
        self._graph: CompiledStateGraph = self.build_graph()

    def build_graph(self) -> CompiledStateGraph:
        """Constructs the LangGraph state graph."""
        g = StateGraph(ResearchState)

        # Add nodes
        g.add_node("plan", plan_node)
        g.add_node("search", search_node)
        g.add_node("retrieve", retrieve_node)
        g.add_node("synthesise", synthesise_node)

        # Add edges
        g.add_edge(START, "plan")
        # Fan-out: plan routes to both search and retrieve concurrently
        g.add_edge("plan", "search")
        g.add_edge("plan", "retrieve")
        # Fan-in: both search and retrieve route to synthesise
        g.add_edge("search", "synthesise")
        g.add_edge("retrieve", "synthesise")

        # Conditional edge from synthesise
        g.add_conditional_edges(
            "synthesise",
            should_request_approval,
            {END: END},  # Explicit path map
        )

        return g.compile()

    async def run(self, initial_state: ResearchState | dict[str, Any]) -> ResearchState:
        """Executes the workflow asynchronously."""
        result = await self._graph.ainvoke(initial_state)
        return ResearchState.model_validate(result)
