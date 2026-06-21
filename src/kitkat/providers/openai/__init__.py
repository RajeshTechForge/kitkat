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

try:
    import openai as _openai_sdk  # noqa: F401
except ImportError as _exc:
    raise ImportError(
        "OpenAIProvider requires the 'openai' extra. "
        "Install it with:\n\n"
        "    pip install kitkat[openai]\n"
        "    # or\n"
        "    uv add kitkat[openai]"
    ) from _exc

from .provider import OpenAIConfig, OpenAIProvider

__all__ = ["OpenAIConfig", "OpenAIProvider"]
