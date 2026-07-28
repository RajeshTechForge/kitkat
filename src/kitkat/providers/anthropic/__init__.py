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

import importlib.util

if importlib.util.find_spec("anthropic") is None:
    raise ImportError(
        "AnthropicProvider requires the 'anthropic' extra. Install with: pip install "
        "kitkat[anthropic]"
    )

from .provider import AnthropicConfig, AnthropicProvider

__all__ = ["AnthropicConfig", "AnthropicProvider"]
