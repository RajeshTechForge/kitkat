"""Core layer public API.

Zero-dependency foundation — no provider SDK imports, no optional extras.
Every other module in the library imports from here.

Usage::

    from kitkat.core import LLMRequest, LLMResponse, Role
    from kitkat.core import LLMAuthenticationError, LLMRateLimitError
"""

from .enums import FinishReason, ProviderType, Role
from .exceptions import (
    KitkatError,
    LLMAuthenticationError,
    LLMContentFilterError,
    LLMError,
    LLMProviderError,
    LLMProviderInitError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMTokenLimitError,
)
from .models import (
    LLMRequest,
    LLMResponse,
    Message,
    ProviderCapabilities,
    ProviderCapabilitiesModel,
    RetryPolicy,
    StreamChunk,
    ThinkingConfig,
    TokenUsage,
)

__all__ = [
    # Enums
    "Role",
    "FinishReason",
    "ProviderType",
    # Models
    "Message",
    "ThinkingConfig",
    "LLMRequest",
    "LLMResponse",
    "StreamChunk",
    "TokenUsage",
    "RetryPolicy",
    "ProviderCapabilities",
    "ProviderCapabilitiesModel",
    # Exceptions
    "KitkatError",
    "LLMError",
    "LLMProviderError",
    "LLMProviderInitError",
    "LLMAuthenticationError",
    "LLMRateLimitError",
    "LLMTokenLimitError",
    "LLMTimeoutError",
    "LLMContentFilterError",
]
