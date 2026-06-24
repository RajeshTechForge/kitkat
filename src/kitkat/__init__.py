"""kitkat — Production-grade LLM provider library.

Quick start (managed path)::

    from kitkat import LLMService, ProviderType, LLMRequest, Message, Role
    from kitkat import create_llm_service
    from kitkat.providers.anthropic import AnthropicProvider, AnthropicConfig
    import os

    provider = AnthropicProvider(AnthropicConfig(api_key=os.environ["ANTHROPIC_API_KEY"]))
    service = create_llm_service({ProviderType.ANTHROPIC: provider})
    await service.initialize()
    response = await service.complete(
        LLMRequest(messages=[Message(role=Role.USER, content="Hello!")]),
        ProviderType.ANTHROPIC,
    )

Quick start (BYOK path)::

    from kitkat import BYOKLLMService, ProviderType, LLMRequest, Message, Role

    async with BYOKLLMService(ProviderType.OPENAI, user_api_key, model) as svc:
        response = await svc.complete(
            LLMRequest(messages=[Message(role=Role.USER, content="Hello!")])
        )

Provider extras must be installed separately::

    pip install kitkat[anthropic]   # Anthropic Claude
    pip install kitkat[openai]      # OpenAI + compatible endpoints
    pip install kitkat[gemini]      # Google Gemini / Vertex AI
    pip install kitkat[all-providers]
"""

from __future__ import annotations

from importlib.metadata import version

__version__ = version("kitkat")

# ── ABC ───────────────────────────────────────────────────────────────────
from .abc.provider import LLMProvider

# ── Core ──────────────────────────────────────────────────────────────────
from .core.enums import FinishReason, ProviderType, Role
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

# ── Service ───────────────────────────────────────────────────────────────
from .service.byok import BYOKLLMService
from .service.factory import create_llm_service
from .service.managed import LLMService

# ── Providers (lazy — importing eagerly would force all SDK installs) ─────

_LAZY_PROVIDER_MAP: dict[str, tuple[str, str]] = {
    "AnthropicProvider": ("kitkat.providers.anthropic", "anthropic"),
    "AnthropicConfig": ("kitkat.providers.anthropic", "anthropic"),
    "OpenAIProvider": ("kitkat.providers.openai", "openai"),
    "OpenAIConfig": ("kitkat.providers.openai", "openai"),
    "GeminiProvider": ("kitkat.providers.gemini", "gemini"),
    "GeminiConfig": ("kitkat.providers.gemini", "gemini"),
}


def __getattr__(name: str) -> object:
    """Lazy-load provider classes on first access.

    Args:
        name: The attribute name being looked up on the ``kitkat`` module.

    Returns:
        The requested class from the appropriate provider subpackage.

    Raises:
        ImportError: If the provider's SDK extra is not installed.
        AttributeError: If *name* is not a known lazy export.
    """
    if name in _LAZY_PROVIDER_MAP:
        import importlib

        module_path, extra = _LAZY_PROVIDER_MAP[name]
        try:
            return getattr(importlib.import_module(module_path), name)
        except ImportError:
            raise ImportError(
                f"'{name}' requires the '{extra}' extra. Install with: pip install kitkat[{extra}]"
            ) from None
    raise AttributeError(f"module 'kitkat' has no attribute {name!r}")


__all__ = [
    "__version__",
    # Enums
    "FinishReason",
    "ProviderType",
    "Role",
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
    # Service
    "LLMService",
    "BYOKLLMService",
    "create_llm_service",
    # Lazy provider re-exports
    "AnthropicProvider",
    "AnthropicConfig",
    "OpenAIProvider",
    "OpenAIConfig",
    "GeminiProvider",
    "GeminiConfig",
]
