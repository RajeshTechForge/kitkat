"""kitkat — Production-grade LLM provider library.

Quick start (managed path)::

    from kitkat.service import LLMService, create_llm_service
    from kitkat import ProviderType, LLMRequest, Message, Role
    from kitkat.providers.anthropic import AnthropicProvider, AnthropicConfig
    import os

    provider = AnthropicProvider(AnthropicConfig(api_key=os.environ["ANTHROPIC_API_KEY"]))
    service = create_llm_service({ProviderType.ANTHROPIC: provider})
    await service.initialize()
    response = await service.complete(
        LLMRequest(messages=[Message(role=Role.USER, content="Hello!")]),
        ProviderType.ANTHROPIC
    )

Quick start (BYOK path)::

    from kitkat.service import BYOKLLMService
    from kitkat import ProviderType, LLMRequest, Message, Role

    async with BYOKLLMService(ProviderType.OPENAI, user_api_key, model) as svc:
        response = await svc.complete(
            LLMRequest(messages=[Message(role=Role.USER, content="Hello!")])
        )

Provider and feature extras must be installed separately::

    pip install kitkat[anthropic]   # Anthropic Claude
    pip install kitkat[openai]      # OpenAI + compatible endpoints
    pip install kitkat[google]      # Google Gemini / Vertex AI
    pip install kitkat[agents]      # PydanticAI agent adapters
    pip install kitkat[workflows]   # LangGraph workflow layer
    pip install kitkat[all]
"""

from __future__ import annotations

from importlib.metadata import version
from typing import TYPE_CHECKING, Any

__version__ = version("kitkat")

# ── ABC ───────────────────────────────────────────────────────────────────
from .abc.provider import LLMProvider

# ── Agent context (no pydantic-ai dep; always available) ─────────────────
from .agents.context import BaseAgentContext

# ── Core ──────────────────────────────────────────────────────────────────
from .core.enums import (
    CacheBackendType,
    CircuitState,
    FinishReason,
    ProviderType,
    Role,
    RoutingStrategy,
    RoutingTier,
)
from .core.exceptions import (
    LLMAuthenticationError,
    LLMContentFilterError,
    LLMError,
    LLMProviderError,
    LLMProviderInitError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMTokenLimitError,
)
from .core.models import (
    LLMRequest,
    LLMResponse,
    Message,
    ProviderCapabilities,
    RetryPolicy,
    StreamChunk,
    ThinkingConfig,
    TokenUsage,
)

if TYPE_CHECKING:
    from .agents.adapters.byok import BYOKKitkatStreamedResponse, BYOKModelAdapter
    from .agents.adapters.managed import KitkatStreamedResponse, ManagedModelAdapter
    from .agents.builders import build_chat_agent, build_structured_agent
    from .agents.observability import configure_observability
    from .agents.tools.registry import ToolRegistry
    from .service.byok import BYOKLLMService
    from .service.cache import CacheConfig, LLMCache
    from .service.factory import create_llm_router, create_llm_service
    from .service.managed import LLMService
    from .service.router import LLMRouter, RouterConfig
    from .workflows.base import BaseWorkflow
    from .workflows.research import ResearchState, ResearchWorkflow

__all__ = [
    "__version__",
    # Enums
    "CacheBackendType",
    "CircuitState",
    "FinishReason",
    "ProviderType",
    "Role",
    "RoutingStrategy",
    "RoutingTier",
    # Models
    "LLMRequest",
    "LLMResponse",
    "Message",
    "ProviderCapabilities",
    "RetryPolicy",
    "StreamChunk",
    "ThinkingConfig",
    "TokenUsage",
    # Exceptions
    "LLMError",
    "LLMProviderError",
    "LLMProviderInitError",
    "LLMAuthenticationError",
    "LLMRateLimitError",
    "LLMTimeoutError",
    "LLMTokenLimitError",
    "LLMContentFilterError",
    # ABC
    "LLMProvider",
    # Agent context (always available)
    "BaseAgentContext",
    # Agent adapters & builders (lazy)
    "BYOKKitkatStreamedResponse",
    "BYOKModelAdapter",
    "KitkatStreamedResponse",
    "ManagedModelAdapter",
    "ToolRegistry",
    "build_chat_agent",
    "build_structured_agent",
    "configure_observability",
    # Workflows (lazy)
    "BaseWorkflow",
    "ResearchState",
    "ResearchWorkflow",
    # Service (lazy)
    "BYOKLLMService",
    "CacheConfig",
    "LLMCache",
    "LLMRouter",
    "LLMService",
    "RouterConfig",
    "create_llm_router",
    "create_llm_service",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # Agents
    "BYOKKitkatStreamedResponse": (".agents.adapters.byok", "BYOKKitkatStreamedResponse"),
    "BYOKModelAdapter": (".agents.adapters.byok", "BYOKModelAdapter"),
    "KitkatStreamedResponse": (".agents.adapters.managed", "KitkatStreamedResponse"),
    "ManagedModelAdapter": (".agents.adapters.managed", "ManagedModelAdapter"),
    "build_chat_agent": (".agents.builders", "build_chat_agent"),
    "build_structured_agent": (".agents.builders", "build_structured_agent"),
    "configure_observability": (".agents.observability", "configure_observability"),
    "ToolRegistry": (".agents.tools.registry", "ToolRegistry"),
    # Workflows
    "BaseWorkflow": (".workflows.base", "BaseWorkflow"),
    "ResearchState": (".workflows.research", "ResearchState"),
    "ResearchWorkflow": (".workflows.research", "ResearchWorkflow"),
    # Service
    "BYOKLLMService": (".service.byok", "BYOKLLMService"),
    "CacheConfig": (".service.cache", "CacheConfig"),
    "LLMCache": (".service.cache", "LLMCache"),
    "create_llm_router": (".service.factory", "create_llm_router"),
    "create_llm_service": (".service.factory", "create_llm_service"),
    "LLMService": (".service.managed", "LLMService"),
    "LLMRouter": (".service.router", "LLMRouter"),
    "RouterConfig": (".service.router", "RouterConfig"),
}


def __getattr__(name: str) -> Any:
    """Lazily import optional and service attributes on first access.

    Args:
        name: Name of the attribute to look up.

    Returns:
        The requested attribute or class.

    Raises:
        AttributeError: If *name* is not an exported kitkat attribute.
    """
    if name in _LAZY_EXPORTS:
        module_path, attr_name = _LAZY_EXPORTS[name]
        import importlib

        mod = importlib.import_module(module_path, __package__)
        val = getattr(mod, attr_name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return all public module attributes including lazy exports."""
    return sorted(list(globals().keys()) + list(_LAZY_EXPORTS.keys()))
