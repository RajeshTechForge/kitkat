"""Unit tests for AnthropicProvider._build_thinking_params.

Validates that ThinkingConfig domain objects are correctly mapped to the
Anthropic SDK typed parameters for both ``enabled`` and ``adaptive``
thinking modes.

The method now returns a typed 2-tuple
``(thinking_param, output_config_param)`` instead of a flat dict. Both
elements are ``anthropic.omit`` when thinking is disabled, signalling
the SDK to omit those fields from the request body.
"""

from __future__ import annotations

import anthropic

from kitkat.core.models import ThinkingConfig
from kitkat.providers.anthropic.provider import AnthropicProvider


class TestBuildThinkingParams:
    def test_none_config_returns_omit_tuple(self) -> None:
        thinking_param, output_config_param = AnthropicProvider._build_thinking_params(None)
        assert thinking_param is anthropic.omit
        assert output_config_param is anthropic.omit

    def test_disabled_config_returns_omit_tuple(self) -> None:
        tc = ThinkingConfig(enabled=False)
        thinking_param, output_config_param = AnthropicProvider._build_thinking_params(tc)
        assert thinking_param is anthropic.omit
        assert output_config_param is anthropic.omit

    def test_enabled_adaptive_default(self) -> None:
        """enabled=True with no options → adaptive mode, effort='high'."""
        tc = ThinkingConfig(enabled=True)
        thinking_param, output_config_param = AnthropicProvider._build_thinking_params(tc)

        assert thinking_param == {"type": "adaptive"}
        assert output_config_param == {"effort": "high"}

    def test_enabled_adaptive_with_effort(self) -> None:
        """effort field on ThinkingConfig populates output_config."""
        tc = ThinkingConfig(enabled=True, effort="medium")
        thinking_param, output_config_param = AnthropicProvider._build_thinking_params(tc)

        assert thinking_param == {"type": "adaptive"}
        assert output_config_param == {"effort": "medium"}

    def test_provider_options_effort_overrides_thinking_effort(self) -> None:
        """provider_options.effort takes precedence over ThinkingConfig.effort."""
        tc = ThinkingConfig(
            enabled=True,
            effort="low",
            provider_options={"effort": "high"},
        )
        _, output_config_param = AnthropicProvider._build_thinking_params(tc)

        assert output_config_param == {"effort": "high"}

    def test_enabled_explicit_mode(self) -> None:
        """thinking_type='enabled' uses budget_tokens; output_config is omitted."""
        tc = ThinkingConfig(
            enabled=True,
            provider_options={"thinking_type": "enabled", "budget_tokens": 5000},
        )
        thinking_param, output_config_param = AnthropicProvider._build_thinking_params(tc)

        assert thinking_param == {"type": "enabled", "budget_tokens": 5000}
        assert output_config_param is anthropic.omit

    def test_enabled_explicit_mode_default_budget(self) -> None:
        """thinking_type='enabled' without budget_tokens defaults to 10_000."""
        tc = ThinkingConfig(
            enabled=True,
            provider_options={"thinking_type": "enabled"},
        )
        thinking_param, _ = AnthropicProvider._build_thinking_params(tc)

        assert thinking_param == {"type": "enabled", "budget_tokens": 10_000}

    def test_budget_tokens_coerced_to_int(self) -> None:
        """budget_tokens passed as a string (from TOML parsing) must be cast to int."""
        tc = ThinkingConfig(
            enabled=True,
            provider_options={"thinking_type": "enabled", "budget_tokens": "8000"},
        )
        thinking_param, _ = AnthropicProvider._build_thinking_params(tc)

        assert isinstance(thinking_param["budget_tokens"], int)  # type: ignore[index]
        assert thinking_param["budget_tokens"] == 8000  # type: ignore[index]

    def test_return_type_is_tuple(self) -> None:
        tc = ThinkingConfig(enabled=True)
        result = AnthropicProvider._build_thinking_params(tc)
        assert isinstance(result, tuple)
        assert len(result) == 2
