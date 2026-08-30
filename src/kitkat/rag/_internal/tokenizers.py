"""Token counting utilities for RAG chunking."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_TIKTOKEN_UNAVAILABLE = object()
_encoder = None


def _get_encoder() -> object | None:
    """Lazily load a tiktoken encoder."""
    global _encoder
    if _encoder is _TIKTOKEN_UNAVAILABLE:
        return None
    if _encoder is not None:
        return _encoder
    try:
        import tiktoken

        _encoder = tiktoken.get_encoding("cl100k_base")
        return _encoder
    except Exception as exc:
        logger.warning("Failed to load tiktoken, falling back to char-ratio: %s", exc)
        _encoder = _TIKTOKEN_UNAVAILABLE
        return None


def count_tokens(text: str, encoding: str = "cl100k_base") -> int:
    """Count the number of tokens in a text string.

    Args:
        text: The text to count tokens for.
        encoding: The tiktoken encoding name (ignored in fallback).

    Returns:
        The estimated token count.
    """
    enc = _get_encoder()
    if enc is not None:
        return len(enc.encode(text))
    # Fallback: ~4 chars per token
    return max(1, len(text) // 4)
