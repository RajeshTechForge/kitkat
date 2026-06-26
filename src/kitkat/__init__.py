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
        ProviderType.ANTHROPIC,
    )

Quick start (BYOK path)::

    from kitkat.service import BYOKLLMService
    from kitkat import ProviderType, LLMRequest, Message, Role

    async with BYOKLLMService(ProviderType.OPENAI, user_api_key, model) as svc:
        response = await svc.complete(
            LLMRequest(messages=[Message(role=Role.USER, content="Hello!")])
        )

Provider extras must be installed separately::

    pip install kitkat[anthropic]   # Anthropic Claude
    pip install kitkat[openai]      # OpenAI + compatible endpoints
    pip install kitkat[gemini]      # Google Gemini / Vertex AI
    pip install kitkat[redis]       # Redis cache backend
    pip install kitkat[all]
"""

from __future__ import annotations

from importlib.metadata import version

__version__ = version("kitkat")

# ── ABC ───────────────────────────────────────────────────────────────────
from .abc.provider import LLMProvider

# ── Core ──────────────────────────────────────────────────────────────────
from .core.enums import CacheBackendType, FinishReason, ProviderType, Role, RoutingStrategy
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

__all__ = [
    "__version__",
    # Enums
    "CacheBackendType",
    "FinishReason",
    "ProviderType",
    "Role",
    "RoutingStrategy",
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
]
