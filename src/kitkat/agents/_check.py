"""
Internal dependency check for kitkat.agents.
Ensure pydantic-ai is installed before executing agent code.
"""

import importlib.util


def require_agents_extra() -> None:
    if importlib.util.find_spec("pydantic_ai") is None:
        raise ImportError(
            "Agent features require the 'agents' extra. Install with: pip install kitkat[agents]"
        )


def require_observability_extra() -> None:
    if (
        importlib.util.find_spec("logfire") is None
        or importlib.util.find_spec("langfuse") is None
        or importlib.util.find_spec("opentelemetry") is None
    ):
        raise ImportError(
            "Observability features require the 'observability' extra. Install with: "
            "pip install kitkat[observability]"
        )
