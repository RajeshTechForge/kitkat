"""Enumeration types used across the library.

These are the only types permitted to be imported by any module in the
library without restriction — they have no dependencies themselves.
"""

from __future__ import annotations

from enum import StrEnum


class Role(StrEnum):
    """Conversation participant roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class FinishReason(StrEnum):
    """Provide specific reasons for the model stopping its generation."""

    STOP = "stop"  # Natural completion
    LENGTH = "length"  # Truncated by max_tokens
    TOOL_CALL = "tool_call"  # Requests tool execution
    CONTENT_FILTER = "content_filter"  # Blocked by safety filters
    ERROR = "error"  # Provider generation failure
    UNKNOWN = "unknown"  # Fallback for unmapped values


class ProviderType(StrEnum):
    """Canonical provider identifiers used throughout the routing layer."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"


class RoutingStrategy(StrEnum):
    """Provider-selection strategy used by :class:`~kitkat.service.router.LLMRouter`.

    Attributes:
        FAILOVER: Always try providers in priority order; only advance on error.
        ROUND_ROBIN: Cycle through healthy providers in insertion order.
        LEAST_LATENCY: Pick the provider with the lowest average response latency.
        RANDOM: Uniformly random selection from the healthy provider pool.
    """

    FAILOVER = "failover"
    ROUND_ROBIN = "round_robin"
    LEAST_LATENCY = "least_latency"
    RANDOM = "random"


class CircuitState(StrEnum):
    """Current state of a per-provider circuit breaker."""

    CLOSED = "CLOSED"  # Normal operation
    OPEN = "OPEN"  # Block requests pending recovery
    HALF_OPEN = "HALF_OPEN"  # Allow single recovery test probe


class CacheBackendType(StrEnum):
    """Selects the storage backend used by :class:`~kitkat.service.cache.LLMCache`.

    Attributes:
        MEMORY: In-process LRU cache backed by :class:`collections.OrderedDict`.
            Zero external dependencies; suitable for single-process deployments.
        REDIS: Async Redis backend via ``redis.asyncio``.
            Requires the ``redis`` extra (``pip install kitkat[redis]``).
            Suitable for multi-process or multi-instance deployments.
    """

    MEMORY = "memory"
    REDIS = "redis"
