"""Enumeration types used across the library.

These are the only types permitted to be imported by any module in the
library without restriction — they have no dependencies themselves.
"""

from __future__ import annotations

from enum import Enum


class Role(str, Enum):
    """Conversation participant roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class FinishReason(str, Enum):
    """Provide specific reasons for the model stopping its generation."""

    STOP = "stop"  # Natural completion
    LENGTH = "length"  # Truncated by max_tokens
    TOOL_CALL = "tool_call"  # Requests tool execution
    CONTENT_FILTER = "content_filter"  # Blocked by safety filters
    ERROR = "error"  # Provider generation failure
    UNKNOWN = "unknown"  # Fallback for unmapped values


class ProviderType(str, Enum):
    """Canonical provider identifiers used throughout the routing layer."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"
