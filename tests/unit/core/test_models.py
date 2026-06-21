"""Unit tests for kitkat.core models, enums, and exceptions.

Phase 1 done-criteria: all tests pass, `from kitkat.core import LLMRequest,
LLMResponse, Role` works.
"""

from __future__ import annotations

import pytest

from kitkat.core import (
    FinishReason,
    LLMAuthenticationError,
    LLMContentFilterError,
    LLMError,
    LLMProviderError,
    LLMProviderInitError,
    LLMRateLimitError,
    LLMRequest,
    LLMResponse,
    LLMTimeoutError,
    LLMTokenLimitError,
    Message,
    ProviderCapabilities,
    ProviderType,
    RetryPolicy,
    Role,
    StreamChunk,
    ThinkingConfig,
    TokenUsage,
)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestEnums:
    def test_role_values(self) -> None:
        assert Role.SYSTEM.value == "system"
        assert Role.USER.value == "user"
        assert Role.ASSISTANT.value == "assistant"

    def test_finish_reason_values(self) -> None:
        assert FinishReason.STOP.value == "stop"
        assert FinishReason.LENGTH.value == "length"
        assert FinishReason.TOOL_CALL.value == "tool_call"
        assert FinishReason.CONTENT_FILTER.value == "content_filter"
        assert FinishReason.ERROR.value == "error"
        assert FinishReason.UNKNOWN.value == "unknown"

    def test_provider_type_values(self) -> None:
        assert ProviderType.ANTHROPIC.value == "anthropic"
        assert ProviderType.OPENAI.value == "openai"
        assert ProviderType.GEMINI.value == "gemini"

    def test_enums_are_str_subclass(self) -> None:
        """str-enum values compare equal to their string equivalents."""
        assert Role.USER == "user"
        assert FinishReason.STOP == "stop"
        assert ProviderType.ANTHROPIC == "anthropic"


# ---------------------------------------------------------------------------
# Message tests
# ---------------------------------------------------------------------------


class TestMessage:
    def test_to_dict(self) -> None:
        msg = Message(role=Role.USER, content="hello")
        assert msg.to_dict() == {"role": "user", "content": "hello"}

    def test_frozen(self) -> None:
        msg = Message(role=Role.USER, content="hello")
        with pytest.raises((AttributeError, TypeError)):
            msg.content = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TokenUsage tests
# ---------------------------------------------------------------------------


class TestTokenUsage:
    def test_total_tokens_field(self) -> None:
        u = TokenUsage(prompt_tokens=100, completion_tokens=50, thinking_tokens=20, total_tokens=170)
        assert u.total_tokens == 170

    def test_empty_factory(self) -> None:
        u = TokenUsage.empty()
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0
        assert u.thinking_tokens == 0
        assert u.total_tokens == 0

    def test_add(self) -> None:
        a = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        b = TokenUsage(prompt_tokens=80, completion_tokens=30, total_tokens=110)
        c = a + b
        assert c.prompt_tokens == 180
        assert c.completion_tokens == 80
        assert c.total_tokens == 260

    def test_add_thinking_tokens(self) -> None:
        a = TokenUsage(thinking_tokens=10, total_tokens=10)
        b = TokenUsage(thinking_tokens=5, total_tokens=5)
        c = a + b
        assert c.thinking_tokens == 15


# ---------------------------------------------------------------------------
# RetryPolicy tests
# ---------------------------------------------------------------------------


class TestRetryPolicy:
    def test_delay_capped_at_max(self) -> None:
        policy = RetryPolicy(base_delay_s=1.0, max_delay_s=10.0, exponential_base=2.0, jitter=False)
        # 2^10 = 1024 >> 10, so delay should be capped
        assert policy.delay_for_attempt(10) == 10.0

    def test_delay_exponential_growth(self) -> None:
        policy = RetryPolicy(base_delay_s=1.0, max_delay_s=100.0, exponential_base=2.0, jitter=False)
        assert policy.delay_for_attempt(0) == 1.0
        assert policy.delay_for_attempt(1) == 2.0
        assert policy.delay_for_attempt(2) == 4.0

    def test_jitter_reduces_delay(self) -> None:
        policy = RetryPolicy(base_delay_s=2.0, max_delay_s=100.0, exponential_base=2.0, jitter=True)
        # With jitter, delay should be between 50% and 100% of the base delay
        for _ in range(20):
            d = policy.delay_for_attempt(0)
            assert 1.0 <= d <= 2.0

    def test_default_retryable_status_codes(self) -> None:
        policy = RetryPolicy()
        assert 429 in policy.retryable_status_codes
        assert 500 in policy.retryable_status_codes


# ---------------------------------------------------------------------------
# LLMRequest tests
# ---------------------------------------------------------------------------


class TestLLMRequest:
    def test_defaults(self) -> None:
        req = LLMRequest(messages=[Message(role=Role.USER, content="hi")])
        assert req.max_tokens == 2048
        assert req.temperature == 0.1
        assert req.stream is False
        assert req.model == ""
        assert req.thinking is None

    def test_empty_messages_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one message"):
            LLMRequest(messages=[])

    def test_invalid_temperature_raises(self) -> None:
        with pytest.raises(ValueError, match="temperature"):
            LLMRequest(
                messages=[Message(role=Role.USER, content="hi")],
                temperature=3.0,
            )

    def test_invalid_max_tokens_raises(self) -> None:
        with pytest.raises(ValueError, match="max_tokens"):
            LLMRequest(
                messages=[Message(role=Role.USER, content="hi")],
                max_tokens=0,
            )

    def test_with_thinking_config(self) -> None:
        req = LLMRequest(
            messages=[Message(role=Role.USER, content="hi")],
            thinking=ThinkingConfig(enabled=True, effort="high"),
        )
        assert req.thinking is not None
        assert req.thinking.enabled is True
        assert req.thinking.effort == "high"


# ---------------------------------------------------------------------------
# LLMResponse tests
# ---------------------------------------------------------------------------


class TestLLMResponse:
    def _make_response(self, finish_reason: FinishReason = FinishReason.STOP) -> LLMResponse:
        return LLMResponse(
            content="hello",
            finish_reason=finish_reason,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            model="claude-3",
            provider=ProviderType.ANTHROPIC,
        )

    def test_was_truncated_false_on_stop(self) -> None:
        assert self._make_response(FinishReason.STOP).was_truncated is False

    def test_was_truncated_true_on_length(self) -> None:
        assert self._make_response(FinishReason.LENGTH).was_truncated is True

    def test_thinking_content_defaults_empty(self) -> None:
        resp = self._make_response()
        assert resp.thinking_content == ""


# ---------------------------------------------------------------------------
# StreamChunk tests
# ---------------------------------------------------------------------------


class TestStreamChunk:
    def test_default_not_final(self) -> None:
        chunk = StreamChunk(delta="hello")
        assert chunk.is_final is False
        assert chunk.is_thinking is False

    def test_final_chunk_fields(self) -> None:
        chunk = StreamChunk(
            delta="",
            is_final=True,
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            model="gpt-4",
            provider=ProviderType.OPENAI,
            latency_ms=120.5,
        )
        assert chunk.is_final is True
        assert chunk.finish_reason == FinishReason.STOP
        assert chunk.latency_ms == 120.5


# ---------------------------------------------------------------------------
# ThinkingConfig tests
# ---------------------------------------------------------------------------


class TestThinkingConfig:
    def test_disabled_by_default(self) -> None:
        tc = ThinkingConfig()
        assert tc.enabled is False
        assert tc.effort is None
        assert tc.provider_options is None

    def test_frozen(self) -> None:
        tc = ThinkingConfig(enabled=True)
        with pytest.raises((AttributeError, TypeError)):
            tc.enabled = False  # type: ignore[misc]

    def test_provider_options(self) -> None:
        tc = ThinkingConfig(
            enabled=True,
            provider_options={"thinking_type": "enabled", "budget_tokens": 10000},
        )
        assert tc.provider_options is not None
        assert tc.provider_options["budget_tokens"] == 10000


# ---------------------------------------------------------------------------
# ProviderCapabilities tests
# ---------------------------------------------------------------------------


class TestProviderCapabilities:
    def test_defaults(self) -> None:
        caps = ProviderCapabilities()
        assert caps.supports_streaming is True
        assert caps.supports_thinking is False
        assert caps.supports_tool_calling is False

    def test_frozen(self) -> None:
        caps = ProviderCapabilities()
        with pytest.raises((AttributeError, TypeError)):
            caps.supports_streaming = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Exception hierarchy tests
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    def test_llm_error_is_base(self) -> None:
        exc = LLMError("test error", status_code=500, provider="anthropic")
        assert isinstance(exc, Exception)
        assert str(exc) == "test error"
        assert exc.provider == "anthropic"

    def test_authentication_error_is_llm_error(self) -> None:
        exc = LLMAuthenticationError("bad key", provider="openai")
        assert isinstance(exc, LLMError)
        assert exc.status_code == 401

    def test_rate_limit_error_carries_retry_after(self) -> None:
        exc = LLMRateLimitError("rate limited", retry_after_s=30.0, provider="gemini")
        assert isinstance(exc, LLMError)
        assert exc.retry_after_s == 30.0

    def test_timeout_error_carries_elapsed(self) -> None:
        exc = LLMTimeoutError("timed out", elapsed_s=45.2, provider="anthropic")
        assert isinstance(exc, LLMError)
        assert exc.elapsed_s == 45.2

    def test_token_limit_error_carries_counts(self) -> None:
        exc = LLMTokenLimitError("too long", token_count=10000, context_limit=8192)
        assert isinstance(exc, LLMError)
        assert exc.token_count == 10000
        assert exc.context_limit == 8192

    def test_provider_init_error_is_provider_error(self) -> None:
        exc = LLMProviderInitError("init failed", provider="anthropic")
        assert isinstance(exc, LLMError)

    def test_content_filter_error(self) -> None:
        exc = LLMContentFilterError("blocked", provider="openai")
        assert isinstance(exc, LLMError)
        assert exc.provider == "openai"

    def test_provider_error_default_status_code(self) -> None:
        exc = LLMProviderError("oops")
        assert exc.status_code == 500


# ---------------------------------------------------------------------------
# Backward-compatibility: imports from kitkat.core still work
# ---------------------------------------------------------------------------


def test_canonical_import() -> None:
    """Verify the Phase 1 done criterion: from kitkat.core import works."""
    from kitkat.core import LLMRequest, LLMResponse, Role  # noqa: F401

    assert Role.USER == "user"


def test_backward_compat_exceptions_module() -> None:
    """Verify kitkat.exceptions still works as a re-export shim."""
    from kitkat.exceptions import (  # noqa: F401
        LLMAuthenticationError,
        LLMError,
        LLMProviderError,
    )


def test_backward_compat_base_module() -> None:
    """Verify kitkat.core.base still exports enums and models."""
    from kitkat.core.base import (  # noqa: F401
        FinishReason,
        LLMProvider,
        LLMRequest,
        Message,
        ProviderType,
        Role,
    )
