"""Anthropic Claude provider for kitkat.

Install the required extra before importing::

    pip install kitkat[anthropic]
    # or
    uv add kitkat[anthropic]

Usage::

    from kitkat.providers.anthropic import AnthropicProvider, AnthropicConfig

    config = AnthropicConfig(api_key="sk-ant-...")
    async with AnthropicProvider(config) as provider:
        response = await provider.complete(request)
"""

from __future__ import annotations

try:
    import anthropic as _anthropic_sdk  # noqa: F401
except ImportError as _exc:
    raise ImportError(
        "AnthropicProvider requires the 'anthropic' extra. "
        "Install it with:\n\n"
        "    pip install kitkat[anthropic]\n"
        "    # or\n"
        "    uv add kitkat[anthropic]"
    ) from _exc

from .provider import AnthropicConfig, AnthropicProvider

__all__ = ["AnthropicConfig", "AnthropicProvider"]
