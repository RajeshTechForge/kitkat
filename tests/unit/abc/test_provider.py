"""Unit tests for kitkat.abc.provider.LLMProvider.

Tests the concrete shared behaviours (helpers, context manager, retry
delegation, run_sync guard) using a minimal stub provider.  Abstract
methods are not tested here — they are covered in provider-specific tests.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from kitkat.abc import LLMProvider
from kitkat.core import (
    FinishReason,
    LLMRequest,
    LLMResponse,
    Message,
    ProviderCapabilities,
    ProviderType,
    RetryPolicy,
    Role,
    StreamChunk,
    TokenUsage,
)

# ---------------------------------------------------------------------------
# Minimal concrete stub for testing shared ABC behaviours
# ---------------------------------------------------------------------------


class _StubProvider(LLMProvider):
    """Minimal LLMProvider implementation for testing shared helper methods."""

    PROVIDER_TYPE = ProviderType.ANTHROPIC
    DEFAULT_MODEL = "stub-model"
    CAPABILITIES = ProviderCapabilities(
        supports_streaming=True,
        max_context_tokens=8_192,
        provider_type=ProviderType.ANTHROPIC,
    )
    RETRY_POLICY = RetryPolicy(max_attempts=1, base_delay_s=0.0, max_delay_s=0.0, jitter=False)

    def __init__(self) -> None:
        super().__init__(config={"api_key": "stub"})
        self._complete_mock: AsyncMock = AsyncMock()
        self._stream_mock: AsyncMock = AsyncMock()

    async def initialize(self) -> None:
        self._initialized = True

    async def shutdown(self) -> None:
        self._initialized = False

    async def _init_client_only(self) -> None:
        self._initialized = True

    async def complete(self, request: LLMRequest) -> LLMResponse:
        return await self._complete_mock(request)

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        async for chunk in self._stream_mock(request):
            yield chunk

    async def health_check(self) -> bool:
        return self._initialized

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


def _make_request() -> LLMRequest:
    return LLMRequest(messages=[Message(role=Role.USER, content="hello")])


def _make_response() -> LLMResponse:
    return LLMResponse(
        content="hi",
        finish_reason=FinishReason.STOP,
        usage=TokenUsage(prompt_tokens=5, completion_tokens=2, total_tokens=7),
        model="stub-model",
        provider=ProviderType.ANTHROPIC,
    )


# ---------------------------------------------------------------------------
# Canonical import test (Phase 2 done-criterion)
# ---------------------------------------------------------------------------


def test_canonical_abc_import() -> None:
    """from kitkat.abc import LLMProvider must work."""
    from kitkat.abc import LLMProvider as LP  # noqa: F401

    assert LP is LLMProvider


def test_backward_compat_base_import() -> None:
    """from kitkat.core.base import LLMProvider must still work."""
    from kitkat.core.base import LLMProvider as LP  # noqa: F401

    assert LP is LLMProvider


# ---------------------------------------------------------------------------
# Lifecycle helpers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_manager_initializes_and_shuts_down() -> None:
    provider = _StubProvider()
    assert not provider._initialized
    async with provider:
        assert provider._initialized
    assert not provider._initialized


@pytest.mark.asyncio
async def test_assert_initialized_raises_before_init() -> None:
    provider = _StubProvider()
    with pytest.raises(RuntimeError, match="initialize()"):
        provider._assert_initialized()


@pytest.mark.asyncio
async def test_assert_initialized_passes_after_init() -> None:
    provider = _StubProvider()
    await provider.initialize()
    provider._assert_initialized()  # Must not raise


# ---------------------------------------------------------------------------
# complete_with_retry delegation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_with_retry_delegates_to_complete() -> None:
    provider = _StubProvider()
    await provider.initialize()
    expected = _make_response()
    provider._complete_mock.return_value = expected

    result = await provider.complete_with_retry(_make_request())
    assert result is expected
    provider._complete_mock.assert_called_once()


@pytest.mark.asyncio
async def test_complete_with_retry_uses_custom_policy() -> None:
    provider = _StubProvider()
    await provider.initialize()
    provider._complete_mock.return_value = _make_response()

    custom_policy = RetryPolicy(max_attempts=2, base_delay_s=0.0, max_delay_s=0.0, jitter=False)
    await provider.complete_with_retry(_make_request(), policy=custom_policy)
    provider._complete_mock.assert_called_once()


# ---------------------------------------------------------------------------
# Token counting helpers
# ---------------------------------------------------------------------------


def test_count_tokens_non_empty() -> None:
    provider = _StubProvider()
    assert provider.count_tokens("hello") >= 1


def test_count_prompt_tokens_empty_list() -> None:
    provider = _StubProvider()
    assert provider.count_prompt_tokens([]) == 0


def test_count_prompt_tokens_sums_content() -> None:
    provider = _StubProvider()
    msgs = [
        Message(role=Role.USER, content="hello world"),
        Message(role=Role.ASSISTANT, content="hi there"),
    ]
    # Stub: max(1, len(text) // 4)
    combined = "hello world hi there"
    expected = max(1, len(combined) // 4)
    assert provider.count_prompt_tokens(msgs) == expected


# ---------------------------------------------------------------------------
# build_base_response_kwargs
# ---------------------------------------------------------------------------


def test_build_base_response_kwargs() -> None:
    import time

    provider = _StubProvider()
    start = time.monotonic()
    kwargs = provider._build_base_response_kwargs(_make_request(), start)
    assert kwargs["provider"] == ProviderType.ANTHROPIC
    assert kwargs["latency_ms"] >= 0.0


# ---------------------------------------------------------------------------
# run_sync guard
# ---------------------------------------------------------------------------


def test_run_sync_raises_inside_event_loop() -> None:
    provider = _StubProvider()
    provider._initialized = True
    provider._complete_mock.return_value = _make_response()

    async def _inner() -> None:
        with pytest.raises(RuntimeError, match="run_sync()"):
            provider.run_sync(_make_request())

    asyncio.run(_inner())


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


def test_repr_uninitialised() -> None:
    provider = _StubProvider()
    r = repr(provider)
    assert "uninitialised" in r
    assert "anthropic" in r


def test_repr_ready() -> None:
    provider = _StubProvider()
    provider._initialized = True
    r = repr(provider)
    assert "ready" in r


# ---------------------------------------------------------------------------
# health_check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_check_false_before_init() -> None:
    provider = _StubProvider()
    assert await provider.health_check() is False


@pytest.mark.asyncio
async def test_health_check_true_after_init() -> None:
    provider = _StubProvider()
    await provider.initialize()
    assert await provider.health_check() is True
