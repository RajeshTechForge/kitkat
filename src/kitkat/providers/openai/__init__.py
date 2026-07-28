"""OpenAI (and OpenAI-compatible) provider for kitkat.

Install the required extra before importing::

    pip install kitkat[openai]
    # or
    uv add kitkat[openai]

Usage::

    from kitkat.providers.openai import OpenAIProvider, OpenAIConfig

    config = OpenAIConfig(api_key="sk-...")
    async with OpenAIProvider(config) as provider:
        response = await provider.complete(request)

This provider is also compatible with NVIDIA NIM and any endpoint that
implements the OpenAI Chat Completions API.  Pass ``base_url`` in
:class:`OpenAIConfig` to point at an alternative endpoint.
"""

from __future__ import annotations

import importlib.util

if importlib.util.find_spec("openai") is None:
    raise ImportError(
        "OpenAIProvider requires the 'openai' extra. Install with: pip install kitkat[openai]"
    )

from .provider import OpenAIConfig, OpenAIProvider

__all__ = ["OpenAIConfig", "OpenAIProvider"]
