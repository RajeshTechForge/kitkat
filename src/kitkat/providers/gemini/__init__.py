"""Google Gemini provider for kitkat.

Install the required extra before importing::

    pip install kitkat[gemini]
    # or
    uv add kitkat[gemini]

Usage::

    from kitkat.providers.gemini import GeminiProvider, GeminiConfig

    config = GeminiConfig(api_key="AIza...")
    async with GeminiProvider(config) as provider:
        response = await provider.complete(request)

Vertex AI is also supported — set ``vertexai=True`` and provide
``project`` and ``location`` in :class:`GeminiConfig`.
"""

from __future__ import annotations

import importlib.util

if importlib.util.find_spec("google.genai") is None:
    raise ImportError(
        "GeminiProvider requires the 'gemini' extra. Install with: pip install kitkat[gemini]"
    )

from .provider import GeminiConfig, GeminiProvider

__all__ = ["GeminiConfig", "GeminiProvider"]
