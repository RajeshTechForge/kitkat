"""Shared token-counting utilities.

Provides a tiktoken-based counter with a conservative character-ratio
fallback for models not yet supported by tiktoken (e.g. Gemini variants).

All provider ``count_tokens()`` implementations should delegate here so the
behaviour is consistent across the library.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Conservative approximation: 4 chars ≈ 1 token (valid for most Latin-script
# text with GPT-style BPE tokenisers).
_CHARS_PER_TOKEN: float = 4.0

# Sentinel that is stored on first failed tiktoken load so we never try again
# in the same process (avoids repeated BPE-download attempts in air-gapped envs).
_TIKTOKEN_UNAVAILABLE = object()

# Cache per encoding name so we only call get_encoding() once.
_ENCODER_CACHE: dict[str, object] = {}


def count_tokens_tiktoken(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens using tiktoken, with a char-ratio fallback.

    Args:
        text: The text to tokenise.
        encoding_name: The tiktoken BPE encoding to use.
            ``cl100k_base`` covers GPT-4 / GPT-3.5 and approximates
            Anthropic Claude tokenisation well enough for budgeting.

    Returns:
        Token count (always ≥ 0; 0 for empty input; ≥ 1 for non-empty).
    """
    if not text:
        return 0

    enc = _ENCODER_CACHE.get(encoding_name)
    if enc is None:
        try:
            import tiktoken

            enc = tiktoken.get_encoding(encoding_name)
            _ENCODER_CACHE[encoding_name] = enc
        except Exception as exc:
            logger.warning(
                "tiktoken BPE load failed (%s); falling back to "
                "character-based token estimate (4 chars ≈ 1 token).",
                exc,
            )
            _ENCODER_CACHE[encoding_name] = _TIKTOKEN_UNAVAILABLE
            enc = _TIKTOKEN_UNAVAILABLE

    if enc is _TIKTOKEN_UNAVAILABLE:
        return count_tokens_fallback(text)

    return max(1, len(enc.encode(text)))  # type: ignore[union-attr]


def count_tokens_fallback(text: str) -> int:
    """Character-ratio token estimate for models without tiktoken support.

    Args:
        text: The text to estimate.

    Returns:
        Estimated token count (0 for empty input; ≥ 1 for non-empty).
    """
    if not text:
        return 0
    return max(1, round(len(text) / _CHARS_PER_TOKEN))
