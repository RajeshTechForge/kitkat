"""Provider package.

Import providers directly from their subpackage::

    from kitkat.providers.anthropic import AnthropicProvider, AnthropicConfig
    from kitkat.providers.openai import OpenAIProvider, OpenAIConfig
    from kitkat.providers.gemini import GeminiProvider, GeminiConfig

Do **not** import concrete providers from this top-level package — doing so
would force all provider SDK dependencies to be installed regardless of which
extras the caller has requested.  Use lazy per-provider imports or
:mod:`kitkat.providers._registry` for dynamic dispatch instead.
"""
