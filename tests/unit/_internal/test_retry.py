"""Unit tests for kitkat._internal.retry.execute_with_retry.

Tests cover:
  - Successful first attempt
  - Retry on rate-limit with retry_after_s
  - Retry on generic LLMError
  - Non-retriable errors raised immediately (auth, token limit, content filter)
  - Exhaustion of all attempts
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from kitkat._internal.retry import execute_with_retry
from kitkat.core.exceptions import (
    LLMAuthenticationError,
    LLMContentFilterError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTokenLimitError,
)
from kitkat.core.models import RetryPolicy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _policy(max_attempts: int = 3, jitter: bool = False) -> RetryPolicy:
    return RetryPolicy(
        max_attempts=max_attempts,
        base_delay_s=0.0,  # Zero delay so tests run instantly
        max_delay_s=0.0,
        exponential_base=2.0,
        jitter=jitter,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_success_on_first_attempt() -> None:
    func = AsyncMock(return_value="ok")
    result = await execute_with_retry(func, _policy(), provider_name="test")
    assert result == "ok"
    assert func.call_count == 1


@pytest.mark.asyncio
async def test_retries_on_llm_provider_error() -> None:
    """Generic LLMError should trigger retry up to max_attempts."""
    call_count = 0

    async def flaky() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise LLMProviderError("transient")
        return "recovered"

    result = await execute_with_retry(flaky, _policy(max_attempts=3), provider_name="test")
    assert result == "recovered"
    assert call_count == 3


@pytest.mark.asyncio
async def test_retries_on_rate_limit_error() -> None:
    call_count = 0

    async def rate_limited() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise LLMRateLimitError("429", retry_after_s=0.0)
        return "ok"

    result = await execute_with_retry(rate_limited, _policy(max_attempts=3), provider_name="test")
    assert result == "ok"
    assert call_count == 2


@pytest.mark.asyncio
async def test_authentication_error_raised_immediately() -> None:
    """Auth errors must NOT be retried."""
    call_count = 0

    async def always_auth_error() -> str:
        nonlocal call_count
        call_count += 1
        raise LLMAuthenticationError("bad key")

    with pytest.raises(LLMAuthenticationError):
        await execute_with_retry(always_auth_error, _policy(max_attempts=3), provider_name="test")

    assert call_count == 1, "LLMAuthenticationError must not be retried"


@pytest.mark.asyncio
async def test_token_limit_error_raised_immediately() -> None:
    call_count = 0

    async def too_long() -> str:
        nonlocal call_count
        call_count += 1
        raise LLMTokenLimitError("too long", token_count=100_000)

    with pytest.raises(LLMTokenLimitError):
        await execute_with_retry(too_long, _policy(max_attempts=3), provider_name="test")

    assert call_count == 1, "LLMTokenLimitError must not be retried"


@pytest.mark.asyncio
async def test_content_filter_error_raised_immediately() -> None:
    call_count = 0

    async def blocked() -> str:
        nonlocal call_count
        call_count += 1
        raise LLMContentFilterError("blocked")

    with pytest.raises(LLMContentFilterError):
        await execute_with_retry(blocked, _policy(max_attempts=3), provider_name="test")

    assert call_count == 1, "LLMContentFilterError must not be retried"


@pytest.mark.asyncio
async def test_raises_last_exception_after_all_attempts() -> None:
    """After exhausting max_attempts the last exception is re-raised."""

    async def always_fails() -> str:
        raise LLMProviderError("permanent failure")

    with pytest.raises(LLMProviderError, match="permanent failure"):
        await execute_with_retry(always_fails, _policy(max_attempts=2), provider_name="test")


@pytest.mark.asyncio
async def test_rate_limit_uses_retry_after_s_for_wait() -> None:
    """When retry_after_s is set on LLMRateLimitError, that value is used for sleep."""
    call_count = 0
    sleep_args: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleep_args.append(seconds)

    async def rate_limited() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise LLMRateLimitError("429", retry_after_s=5.0)
        return "ok"

    with patch("kitkat._internal.retry.asyncio.sleep", side_effect=fake_sleep):
        await execute_with_retry(rate_limited, _policy(max_attempts=3), provider_name="test")

    assert sleep_args == [5.0]
