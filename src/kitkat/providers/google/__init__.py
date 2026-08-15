"""Google provider for kitkat.

Install the required extra before importing::

    pip install kitkat[google]
    # or
    uv add kitkat[google]

Usage::

    from kitkat.providers.google import GoogleProvider, GoogleConfig

    config = GoogleConfig(api_key="AIza...")
    async with GoogleProvider(config) as provider:
        response = await provider.complete(request)

Vertex AI is also supported — set ``vertexai=True`` and provide
``project`` and ``location`` in :class:`GoogleConfig`.
"""

from __future__ import annotations

import importlib.util

if importlib.util.find_spec("google.genai") is None:
    raise ImportError(
        "GoogleProvider requires the 'google' extra. Install with: pip install kitkat[google]"
    )

from .provider import GoogleConfig, GoogleProvider

__all__ = ["GoogleConfig", "GoogleProvider"]
