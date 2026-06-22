"""Unit tests for AnthropicProvider._build_thinking_params.

Validates that ThinkingConfig domain objects are correctly mapped to the
Anthropic SDK keyword argument dictionaries for both ``enabled`` and
``adaptive`` thinking modes.
"""

from __future__ import annotations

from kitkat.core.models import ThinkingConfig
from kitkat.providers.anthropic.provider import AnthropicProvider


class TestBuildThinkingParams:
    def test_none_config_returns_empty(self) -> None:
        assert AnthropicProvider._build_thinking_params(None) == {}

    def test_disabled_config_returns_empty(self) -> None:
        tc = ThinkingConfig(enabled=False)
        assert AnthropicProvider._build_thinking_params(tc) == {}

    def test_enabled_adaptive_default(self) -> None:
        """enabled=True with no options → adaptive mode, effort='high'."""
        tc = ThinkingConfig(enabled=True)
        result = AnthropicProvider._build_thinking_params(tc)

        assert result["thinking"] == {"type": "adaptive"}
        assert result["output_config"] == {"effort": "high"}

    def test_enabled_adaptive_with_effort(self) -> None:
        """effort field on ThinkingConfig populates output_config."""
        tc = ThinkingConfig(enabled=True, effort="medium")
        result = AnthropicProvider._build_thinking_params(tc)

        assert result["thinking"] == {"type": "adaptive"}
        assert result["output_config"] == {"effort": "medium"}

    def test_provider_options_effort_overrides_thinking_effort(self) -> None:
        """provider_options.effort takes precedence over ThinkingConfig.effort."""
        tc = ThinkingConfig(
            enabled=True,
            effort="low",
            provider_options={"effort": "high"},
        )
        result = AnthropicProvider._build_thinking_params(tc)

        assert result["output_config"] == {"effort": "high"}

    def test_enabled_explicit_mode(self) -> None:
        """thinking_type='enabled' uses budget_tokens."""
        tc = ThinkingConfig(
            enabled=True,
            provider_options={"thinking_type": "enabled", "budget_tokens": 5000},
        )
        result = AnthropicProvider._build_thinking_params(tc)

        assert result["thinking"] == {"type": "enabled", "budget_tokens": 5000}
        assert "output_config" not in result

    def test_enabled_explicit_mode_default_budget(self) -> None:
        """thinking_type='enabled' without budget_tokens defaults to 10_000."""
        tc = ThinkingConfig(
            enabled=True,
            provider_options={"thinking_type": "enabled"},
        )
        result = AnthropicProvider._build_thinking_params(tc)

        assert result["thinking"] == {"type": "enabled", "budget_tokens": 10_000}

    def test_budget_tokens_coerced_to_int(self) -> None:
        """budget_tokens passed as a string (from TOML parsing) must be cast to int."""
        tc = ThinkingConfig(
            enabled=True,
            provider_options={"thinking_type": "enabled", "budget_tokens": "8000"},
        )
        result = AnthropicProvider._build_thinking_params(tc)

        assert isinstance(result["thinking"]["budget_tokens"], int)
        assert result["thinking"]["budget_tokens"] == 8000

    def test_return_type_is_dict(self) -> None:
        tc = ThinkingConfig(enabled=True)
        result = AnthropicProvider._build_thinking_params(tc)
        assert isinstance(result, dict)
