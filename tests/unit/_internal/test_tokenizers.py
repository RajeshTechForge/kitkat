"""Unit tests for kitkat._internal.tokenizers.

Tests both the tiktoken path and the fallback path.
"""

from __future__ import annotations

from unittest.mock import patch

from kitkat._internal.tokenizers import count_tokens_fallback, count_tokens_tiktoken


class TestCountTokensFallback:
    def test_empty_string_returns_zero(self) -> None:
        assert count_tokens_fallback("") == 0

    def test_short_text_returns_at_least_one(self) -> None:
        assert count_tokens_fallback("hi") >= 1

    def test_exact_ratio(self) -> None:
        # 40 chars / 4 = 10 tokens
        text = "a" * 40
        assert count_tokens_fallback(text) == 10

    def test_rounding(self) -> None:
        # 10 chars → round(10 / 4) = round(2.5) = 2
        text = "a" * 10
        assert count_tokens_fallback(text) == max(1, round(10 / 4.0))


class TestCountTokensTiktoken:
    def test_empty_string_returns_zero(self) -> None:
        assert count_tokens_tiktoken("") == 0

    def test_non_empty_returns_positive(self) -> None:
        result = count_tokens_tiktoken("hello world")
        assert result >= 1

    def test_falls_back_on_tiktoken_failure(self) -> None:
        """Simulate tiktoken being unavailable — must use char fallback."""
        from kitkat._internal import tokenizers as tok_module

        # Clear encoder cache to force re-load attempt.
        original_cache = tok_module._ENCODER_CACHE.copy()
        tok_module._ENCODER_CACHE.clear()

        try:
            with patch.dict("sys.modules", {"tiktoken": None}):
                result = count_tokens_tiktoken("hello world", encoding_name="_test_enc")
            # Fallback: 11 chars → max(1, round(11/4)) = max(1, 3) = 3
            assert result == max(1, round(len("hello world") / 4.0))
        finally:
            tok_module._ENCODER_CACHE.clear()
            tok_module._ENCODER_CACHE.update(original_cache)

    def test_encoder_cached_after_first_call(self) -> None:
        """Second call for the same encoding should reuse the cached encoder."""
        from kitkat._internal import tokenizers as tok_module

        encoding = "cl100k_base"
        # Warm the cache.
        count_tokens_tiktoken("warmup", encoding_name=encoding)
        cached = tok_module._ENCODER_CACHE.get(encoding)

        assert cached is not None
        assert cached is not tok_module._TIKTOKEN_UNAVAILABLE
