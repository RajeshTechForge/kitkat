"""Core data models: the currency exchanged between all library layers.

These are plain dataclasses with no provider SDK dependencies.
They must remain importable with only pydantic and stdlib installed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from .enums import FinishReason, ProviderType, Role


@dataclass(frozen=True)
class Message:
    """A single turn in a conversation."""

    role: Role
    content: str

    def to_dict(self) -> dict[str, str]:
        """Serialize the message to a provider-agnostic dictionary format.

        Returns:
            A dictionary containing the message role and content.
        """
        return {"role": self.role.value, "content": self.content}


@dataclass(frozen=True)
class ThinkingConfig:
    """Provider-agnostic thinking/reasoning configuration.

    Carried through the domain layer from the validated schema boundary
    to provider implementations. Each provider maps these fields to its
    native SDK parameters.

    Attributes:
        enabled: Whether thinking/reasoning is active for this request.
        effort: Normalized effort level ("low", "medium", "high") that each
            provider maps to its native vocabulary. None defers to the
            provider's default.
        provider_options: Provider-specific overrides, validated as a typed
            Pydantic model at the schema boundary and converted to a plain
            dict via "model_dump()". Takes precedence over "effort"
            when both are set. Keys and value types are constrained by
            the schema-layer union; "str | int | None" covers all
            current provider option fields.
    """

    enabled: bool = False
    effort: str | None = None
    provider_options: dict[str, str | int | None] | None = None


@dataclass
class LLMRequest:
    """Everything a provider needs to fulfil a completion or streaming request."""

    messages: list[Message]

    model: str = ""
    max_tokens: int = 2048
    temperature: float = 0.1
    top_p: float = 1.0
    stop_sequences: list[str] = field(default_factory=list)

    stream: bool = False
    timeout: float | None = 30.0  # in seconds
    thinking: ThinkingConfig | None = None

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("LLMRequest must contain at least one message.")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be in [0.0, 2.0], got {self.temperature}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be ≥ 1, got {self.max_tokens}")


@dataclass
class TokenUsage:
    """Token consumption for a single provider call.

    Semantic contract:
        - "completion_tokens" counts answer tokens **only** (excludes thinking).
        - "thinking_tokens" counts reasoning/thinking tokens. Reported as 0
          when the provider does not expose a separate count (e.g. Anthropic).
        - "total_tokens" equals "prompt_tokens + completion_tokens + thinking_tokens".
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    thinking_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def empty(cls) -> TokenUsage:
        """Return a zero-valued usage object.

        Returns:
            An empty TokenUsage instance.
        """
        return cls()

    def __add__(self, other: TokenUsage) -> TokenUsage:
        """Aggregate usage from two calls.

        Args:
            other: Another TokenUsage object.

        Returns:
            A new TokenUsage representing the combined totals.
        """
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            thinking_tokens=self.thinking_tokens + other.thinking_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


@dataclass
class LLMResponse:
    """A completed (non-streaming) response from any provider."""

    content: str
    finish_reason: FinishReason
    usage: TokenUsage
    model: str
    provider: ProviderType
    thinking_content: str = ""
    latency_ms: float = 0.0

    raw_response: Any = field(default=None, repr=False)

    @property
    def was_truncated(self) -> bool:
        """Return True when finish_reason indicates the max_tokens limit was hit."""
        return self.finish_reason == FinishReason.LENGTH


@dataclass
class StreamChunk:
    """One token or delta fragment from a streaming response.

    Ordering contract: all thinking chunks ("is_thinking=True") are emitted
    before any answer chunks ("is_thinking=False"). The transition from
    "True" to "False" is one-way and never interleaved. The final sentinel
    chunk always has "is_thinking=False".
    """

    delta: str
    is_thinking: bool = False
    is_final: bool = False

    finish_reason: FinishReason = FinishReason.UNKNOWN
    usage: TokenUsage = field(default_factory=TokenUsage.empty)
    model: str = ""
    provider: ProviderType = ProviderType.ANTHROPIC
    latency_ms: float = 0.0


@dataclass
class RetryPolicy:
    """Exponential back-off configuration for provider retry loops."""

    max_attempts: int = 3
    base_delay_s: float = 1.0
    max_delay_s: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_status_codes: frozenset[int] = field(
        default_factory=lambda: frozenset({408, 429, 500, 502, 503, 504})
    )

    def delay_for_attempt(self, attempt: int) -> float:
        """Calculate how many seconds to wait before the specified attempt."""
        delay = min(
            self.base_delay_s * (self.exponential_base**attempt),
            self.max_delay_s,
        )
        if self.jitter:
            delay *= 0.5 + random.random() * 0.5
        return delay


@dataclass(frozen=True)
class ProviderCapabilities:
    """Feature flags that the router queries when selecting a provider."""

    supports_streaming: bool = True
    supports_system_prompt: bool = True
    supports_tool_calling: bool = False
    supports_vision: bool = False
    supports_thinking: bool = False
    max_context_tokens: int = 8_192
    provider_type: ProviderType = ProviderType.ANTHROPIC


class ProviderCapabilitiesModel(BaseModel):
    """Pydantic model variant of ProviderCapabilities for serialization use cases.

    Use :class:`ProviderCapabilities` (the dataclass) inside the library.
    This model is provided for callers that need JSON serialization.
    """

    supports_streaming: bool = True
    supports_system_prompt: bool = True
    supports_tool_calling: bool = False
    supports_vision: bool = False
    supports_thinking: bool = False
    max_context_tokens: int = 8_192
