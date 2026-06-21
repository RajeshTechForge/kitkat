"""Backward-compatibility shim for kitkat.core.base.

.. deprecated::
    Import directly from the canonical locations instead:

    * Enums / models  → :mod:`kitkat.core`
    * LLMProvider ABC → :mod:`kitkat.abc`

    This module is kept so that existing code using
    ``from kitkat.core.base import LLMProvider`` continues to work
    without changes.  It will be removed in a future major version.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Re-export the LLMProvider ABC from its new canonical home.
# ---------------------------------------------------------------------------
from kitkat.abc.provider import LLMProvider  # noqa: F401

# ---------------------------------------------------------------------------
# Re-export enums and models for backward compatibility.
# ---------------------------------------------------------------------------
from .enums import FinishReason, ProviderType, Role
from .models import (
    LLMRequest,
    LLMResponse,
    Message,
    ProviderCapabilities,
    RetryPolicy,
    StreamChunk,
    ThinkingConfig,
    TokenUsage,
)

# ---------------------------------------------------------------------------
# Keep __all__ explicit so star-imports still work for existing consumers
# ---------------------------------------------------------------------------
__all__ = [
    "FinishReason",
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "Message",
    "ProviderCapabilities",
    "ProviderType",
    "RetryPolicy",
    "Role",
    "StreamChunk",
    "ThinkingConfig",
    "TokenUsage",
]
