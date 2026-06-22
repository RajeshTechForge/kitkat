"""Unit tests for GeminiProvider._build_generation_config thinking-path.

Validates that ThinkingConfig domain objects are correctly translated to
``genai_types.ThinkingConfig`` at the ``GenerateContentConfig.thinking_config``
attribute.
"""

from __future__ import annotations

from google.genai import types as genai_types

from kitkat.core.enums import Role
from kitkat.core.models import LLMRequest, Message, ThinkingConfig
from kitkat.providers.gemini.provider import GeminiProvider


def _make_request() -> LLMRequest:
    """Build a minimal LLMRequest for testing generation config."""
    return LLMRequest(messages=[Message(role=Role.USER, content="test")])


class TestBuildGenerationConfigThinking:
    def test_none_config(self) -> None:
        tc = GeminiProvider._build_generation_config(_make_request(), "", None)
        assert tc.thinking_config is None

    def test_disabled_config(self) -> None:
        thinking = ThinkingConfig(enabled=False)
        tc = GeminiProvider._build_generation_config(_make_request(), "", thinking)
        assert tc.thinking_config is None

    def test_enabled_no_effort(self) -> None:
        """enabled=True with no effort → ThinkingConfig with include_thoughts only."""
        thinking = ThinkingConfig(enabled=True)
        tc = GeminiProvider._build_generation_config(_make_request(), "", thinking)

        assert tc.thinking_config is not None
        assert tc.thinking_config.include_thoughts is True
        assert tc.thinking_config.thinking_level is None

    def test_enabled_low_effort(self) -> None:
        thinking = ThinkingConfig(enabled=True, effort="low")
        tc = GeminiProvider._build_generation_config(_make_request(), "", thinking)

        assert tc.thinking_config is not None
        assert tc.thinking_config.thinking_level == genai_types.ThinkingLevel.LOW

    def test_enabled_medium_effort(self) -> None:
        thinking = ThinkingConfig(enabled=True, effort="medium")
        tc = GeminiProvider._build_generation_config(_make_request(), "", thinking)

        assert tc.thinking_config is not None
        assert tc.thinking_config.thinking_level == genai_types.ThinkingLevel.MEDIUM

    def test_enabled_high_effort(self) -> None:
        thinking = ThinkingConfig(enabled=True, effort="high")
        tc = GeminiProvider._build_generation_config(_make_request(), "", thinking)

        assert tc.thinking_config is not None
        assert tc.thinking_config.thinking_level == genai_types.ThinkingLevel.HIGH

    def test_provider_options_level_overrides_effort(self) -> None:
        """provider_options.level takes precedence over ThinkingConfig.effort."""
        thinking = ThinkingConfig(
            enabled=True,
            effort="low",
            provider_options={"level": "HIGH"},
        )
        tc = GeminiProvider._build_generation_config(_make_request(), "", thinking)

        assert tc.thinking_config is not None
        assert tc.thinking_config.thinking_level == genai_types.ThinkingLevel.HIGH

    def test_provider_options_level_without_effort(self) -> None:
        thinking = ThinkingConfig(
            enabled=True,
            provider_options={"level": "MEDIUM"},
        )
        tc = GeminiProvider._build_generation_config(_make_request(), "", thinking)

        assert tc.thinking_config is not None
        assert tc.thinking_config.thinking_level == genai_types.ThinkingLevel.MEDIUM

    def test_return_type(self) -> None:
        thinking = ThinkingConfig(enabled=True)
        tc = GeminiProvider._build_generation_config(_make_request(), "", thinking)
        assert isinstance(tc, genai_types.GenerateContentConfig)
