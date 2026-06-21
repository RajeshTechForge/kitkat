"""Unit tests for OpenAIProvider._build_thinking_params.

Validates that ThinkingConfig domain objects are correctly translated to
the ``reasoning_effort`` OpenAI SDK keyword argument.
"""

from __future__ import annotations

import pytest

from kitkat.providers.openai.provider import OpenAIProvider
from kitkat.core.models import ThinkingConfig


class TestBuildThinkingParams:
    def test_none_config_returns_empty(self) -> None:
        assert OpenAIProvider._build_thinking_params(None) == {}

    def test_disabled_config_returns_empty(self) -> None:
        tc = ThinkingConfig(enabled=False)
        assert OpenAIProvider._build_thinking_params(tc) == {}

    def test_enabled_no_effort_returns_empty(self) -> None:
        """enabled=True with no effort → empty dict (model decides)."""
        tc = ThinkingConfig(enabled=True)
        assert OpenAIProvider._build_thinking_params(tc) == {}

    def test_enabled_with_thinking_effort(self) -> None:
        tc = ThinkingConfig(enabled=True, effort="high")
        result = OpenAIProvider._build_thinking_params(tc)
        assert result == {"reasoning_effort": "high"}

    def test_enabled_medium_effort(self) -> None:
        tc = ThinkingConfig(enabled=True, effort="medium")
        result = OpenAIProvider._build_thinking_params(tc)
        assert result == {"reasoning_effort": "medium"}

    def test_provider_options_effort_overrides_thinking_effort(self) -> None:
        """provider_options.effort takes precedence over ThinkingConfig.effort."""
        tc = ThinkingConfig(
            enabled=True,
            effort="low",
            provider_options={"effort": "high"},
        )
        result = OpenAIProvider._build_thinking_params(tc)
        assert result == {"reasoning_effort": "high"}

    def test_provider_options_effort_used_when_thinking_effort_none(self) -> None:
        tc = ThinkingConfig(
            enabled=True,
            provider_options={"effort": "medium"},
        )
        result = OpenAIProvider._build_thinking_params(tc)
        assert result == {"reasoning_effort": "medium"}

    def test_reasoning_effort_is_string(self) -> None:
        tc = ThinkingConfig(enabled=True, effort="high")
        result = OpenAIProvider._build_thinking_params(tc)
        assert isinstance(result.get("reasoning_effort"), str)

    def test_return_type_is_dict(self) -> None:
        tc = ThinkingConfig(enabled=True)
        result = OpenAIProvider._build_thinking_params(tc)
        assert isinstance(result, dict)
