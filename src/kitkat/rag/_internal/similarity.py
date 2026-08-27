"""Pure-Python similarity functions for vector math."""

from __future__ import annotations

import math


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Calculate cosine similarity between two vectors.

    Args:
        vec_a: First vector.
        vec_b: Second vector.

    Returns:
        Cosine similarity score between -1.0 and 1.0.
    """
    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def dot_product(vec_a: list[float], vec_b: list[float]) -> float:
    """Calculate dot product between two vectors.

    Args:
        vec_a: First vector.
        vec_b: Second vector.

    Returns:
        Dot product score.
    """
    return sum(a * b for a, b in zip(vec_a, vec_b, strict=True))


def euclidean_distance(vec_a: list[float], vec_b: list[float]) -> float:
    """Calculate Euclidean distance between two vectors.

    Args:
        vec_a: First vector.
        vec_b: Second vector.

    Returns:
        Euclidean distance (lower is closer).
    """
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec_a, vec_b, strict=True)))
