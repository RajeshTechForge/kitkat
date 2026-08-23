"""Shared utilities for Google providers (Gemini API and Vertex AI).

Contains logic that is structurally identical across both Google endpoints,
such as message translation and finish-reason mapping.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from google.genai import types as genai_types

from ...core.enums import FinishReason, Role

if TYPE_CHECKING:
    from ...core.models import Message

logger = logging.getLogger(__name__)

# Guards against repeated tiktoken BPE download attempts in air-gapped environments.
TIKTOKEN_UNAVAILABLE = object()

# Translates Google finish reasons to internal FinishReason enum.
# This mapping is standardized across Google AI Studio and Vertex AI for Gemini models.
FINISH_REASON_MAP: dict[str, FinishReason] = {
    "STOP": FinishReason.STOP,
    "MAX_TOKENS": FinishReason.LENGTH,
    "SAFETY": FinishReason.CONTENT_FILTER,
    "RECITATION": FinishReason.CONTENT_FILTER,  # Copyright / recitation block
    "BLOCKLIST": FinishReason.CONTENT_FILTER,
    "PROHIBITED_CONTENT": FinishReason.CONTENT_FILTER,
    "SPII": FinishReason.CONTENT_FILTER,  # Sensitive PII detection
    "IMAGE_SAFETY": FinishReason.CONTENT_FILTER,
    "MALFORMED_FUNCTION_CALL": FinishReason.TOOL_CALL,
    "UNEXPECTED_TOOL_CALL": FinishReason.TOOL_CALL,
    "LANGUAGE": FinishReason.UNKNOWN,
    "OTHER": FinishReason.UNKNOWN,
    "IMAGE_OTHER": FinishReason.UNKNOWN,
    "FINISH_REASON_UNSPECIFIED": FinishReason.UNKNOWN,
}


def split_messages(
    messages: list[Message],
) -> tuple[str, list[genai_types.Content]]:
    """Separate the system instruction from conversation turns.

    Google's API expects the system prompt as a top-level ``system_instruction``
    parameter rather than as a message in the ``contents`` array.

    Args:
        messages: The list of combined messages.

    Returns:
        A tuple of ``(system_instruction, contents)``.
    """
    system_parts: list[str] = []
    contents: list[genai_types.Content] = []

    for msg in messages:
        if msg.role == Role.SYSTEM:
            system_parts.append(msg.content)
        else:
            google_role = "model" if msg.role == Role.ASSISTANT else "user"
            contents.append(
                genai_types.Content(
                    role=google_role,
                    parts=[genai_types.Part(text=msg.content)],
                )
            )

    return "\n\n---\n\n".join(system_parts), contents
