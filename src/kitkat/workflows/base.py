"""BaseWorkflow: common interface for all graph-based workflows."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

S = TypeVar("S")


class BaseWorkflow(ABC, Generic[S]):
    """ABC for LangGraph-based stateful workflows.

    Subclass this to implement multi-step agentic pipelines.
    Compile once at module level; reuse across requests.

    Generic Parameters:
        S: The state object type (typically a Pydantic BaseModel).
    """

    @abstractmethod
    def build_graph(self) -> CompiledStateGraph:
        """Construct and return the compiled state graph.

        Returns:
            A CompiledStateGraph ready for invocation via ``.invoke()`` or ``.ainvoke()``.
        """

    @abstractmethod
    async def run(self, initial_state: S | dict[str, Any]) -> S:
        """Execute the workflow asynchronously from the given initial state.

        Args:
            initial_state: The starting state for the workflow.

        Returns:
            The final state after the workflow completes.
        """
