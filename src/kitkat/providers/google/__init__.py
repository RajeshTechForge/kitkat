"""Google provider package — split into Gemini API and Vertex AI sub-providers.

Install the required extra before importing::

    pip install kitkat[google]
    # or
    uv add kitkat[google]

Usage::

    from kitkat.providers.google import GeminiProvider, GeminiConfig
    from kitkat.providers.google import VertexAIProvider, VertexAIConfig

"""

from __future__ import annotations

import importlib.util

if importlib.util.find_spec("google.genai") is None:
    raise ImportError(
        "Google Provider requires the 'google' extra. Install with: pip install kitkat[google]"
    )

from .gemini import GeminiConfig, GeminiProvider
from .vertex_ai import VertexAIConfig, VertexAIProvider

__all__ = ["GeminiConfig", "GeminiProvider", "VertexAIConfig", "VertexAIProvider"]
