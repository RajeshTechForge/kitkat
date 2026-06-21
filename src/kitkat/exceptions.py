"""Custom exception hierarchy for KitKat.

.. deprecated::
    Import exceptions from :mod:`kitkat.core.exceptions` or
    :mod:`kitkat.core` directly.  This module is kept for backward
    compatibility and re-exports everything from ``kitkat.core.exceptions``.
"""

from __future__ import annotations

from .core.exceptions import (
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

__all__ = [
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
