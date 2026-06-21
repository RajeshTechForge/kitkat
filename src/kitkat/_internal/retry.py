"""Shared retry logic for all provider implementations.

Provider subclasses call :func:`execute_with_retry` from within
``complete_with_retry()``.  This ensures consistent exponential back-off
behaviour regardless of which provider is in use.

Non-retriable errors (auth, token-limit, content-filter) are re-raised
immediately without sleeping, so callers never wait on deterministic
failures.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from ..core.models import RetryPolicy

from ..core.exceptions import (
    LLMAuthenticationError,
    LLMContentFilterError,
    LLMError,
    LLMRateLimitError,
    LLMTokenLimitError,
)

T = TypeVar("T")
logger = logging.getLogger(__name__)

# Errors that must never be retried
_NON_RETRIABLE = (LLMAuthenticationError, LLMTokenLimitError, LLMContentFilterError)


async def execute_with_retry(
    func: Callable[[], Coroutine[None, None, T]],
    policy: RetryPolicy,
    provider_name: str,
) -> T:
    """Execute an async callable with exponential back-off retry.

    Retries on :exc:`~kitkat.core.exceptions.LLMRateLimitError` and
    generic :exc:`~kitkat.core.exceptions.LLMError`.  Raises immediately
    on non-retriable errors (authentication, token limit, content filter).

    Args:
        func: Zero-argument async callable that performs one inference attempt.
        policy: Retry configuration (attempts, delays, jitter).
        provider_name: Used in log messages to identify the provider.

    Returns:
        The return value of *func* on a successful attempt.

    Raises:
        LLMAuthenticationError: Immediately — credentials are invalid.
        LLMTokenLimitError: Immediately — prompt is deterministically too long.
        LLMContentFilterError: Immediately — content policy violation.
        LLMRateLimitError: After all retry attempts are exhausted.
        LLMError: After all retry attempts are exhausted for other errors.
    """
    last_exc: Exception | None = None

    for attempt in range(policy.max_attempts):
        try:
            return await func()

        except _NON_RETRIABLE:
            raise  # Deterministic failure — skip retries entirely

        except LLMRateLimitError as exc:
            wait = exc.retry_after_s or policy.delay_for_attempt(attempt)
            logger.warning(
                "[%s] Rate limited. Waiting %.1fs (attempt %d/%d).",
                provider_name,
                wait,
                attempt + 1,
                policy.max_attempts,
            )
            last_exc = exc
            if attempt < policy.max_attempts - 1:
                await asyncio.sleep(wait)

        except LLMError as exc:
            wait = policy.delay_for_attempt(attempt)
            logger.warning(
                "[%s] Provider error: %s. Waiting %.1fs (attempt %d/%d).",
                provider_name,
                exc,
                wait,
                attempt + 1,
                policy.max_attempts,
            )
            last_exc = exc
            if attempt < policy.max_attempts - 1:
                await asyncio.sleep(wait)

    # All attempts exhausted.
    assert last_exc is not None, "execute_with_retry exited without an exception set"
    raise last_exc
