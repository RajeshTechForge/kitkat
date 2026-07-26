"""Unit tests for ManagedModelAdapter, BYOKModelAdapter, and streaming helpers.

Tests run against stub providers — no real API calls are made.  The pydantic-ai
Model protocol is exercised via direct method calls (not through Agent.run) to
keep tests fast and dependency-free beyond pydantic-ai itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.messages import (
    ModelResponse as PydanticModelResponse,
)
from pydantic_ai.models import ModelRequestParameters, ModelSettings

from kitkat.agents.adapters.byok import BYOKModelAdapter
from kitkat.agents.adapters.managed import (
    KitkatStreamedResponse,
    ManagedModelAdapter,
    _to_llm_request,
    _to_request_usage,
)
from kitkat.core.enums import FinishReason, ProviderType, Role
from kitkat.core.models import (
    LLMResponse,
    StreamChunk,
    TokenUsage,
)

# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------


def _make_token_usage(
    prompt: int = 10,
    completion: int = 5,
    thinking: int = 0,
) -> TokenUsage:
    return TokenUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        thinking_tokens=thinking,
    )


def _make_llm_response(content: str = "hello") -> LLMResponse:
    return LLMResponse(
        content=content,
        finish_reason=FinishReason.STOP,
        usage=_make_token_usage(),
        model="claude-3-5-sonnet",
        provider=ProviderType.ANTHROPIC,
        latency_ms=42.0,
    )


def _make_model_request_parameters() -> ModelRequestParameters:
    """Minimal ModelRequestParameters for tests."""
    return ModelRequestParameters(
        function_tools=[],
        native_tools=[],
        output_mode="text",
        output_object=None,
        output_tools=[],
        prompted_output_template=None,
        allow_text_output=True,
        allow_image_output=False,
        instruction_parts=[],
        thinking=None,
    )


def _make_messages(
    system: str = "You are helpful.",
    user: str = "Hello!",
) -> list:
    return [
        ModelRequest(
            parts=[
                SystemPromptPart(content=system),
                UserPromptPart(content=user),
            ]
        )
    ]


async def _make_chunk_iter(
    deltas: list[str],
    usage: TokenUsage | None = None,
) -> AsyncIterator[StreamChunk]:
    for i, delta in enumerate(deltas):
        is_final = i == len(deltas) - 1
        yield StreamChunk(
            delta=delta,
            is_final=is_final,
            finish_reason=FinishReason.STOP if is_final else None,
            usage=usage if is_final else None,
        )


# ---------------------------------------------------------------------------
# _to_llm_request
# ---------------------------------------------------------------------------


class TestToLlmRequest:
    def test_system_prompt_extracted(self) -> None:
        messages = [ModelRequest(parts=[SystemPromptPart(content="Be brief.")])]
        req = _to_llm_request(messages, None)
        assert any(m.role == Role.SYSTEM and "brief" in m.content for m in req.messages)

    def test_user_prompt_extracted(self) -> None:
        messages = [ModelRequest(parts=[UserPromptPart(content="Hi there!")])]
        req = _to_llm_request(messages, None)
        assert any(m.role == Role.USER and "Hi" in m.content for m in req.messages)

    def test_assistant_text_part_extracted(self) -> None:
        messages = [
            PydanticModelResponse(
                parts=[TextPart(content="I'm an assistant response.")],
                model_name="test",
            )
        ]
        req = _to_llm_request(messages, None)
        assert any(m.role == Role.ASSISTANT for m in req.messages)

    def test_multi_turn_conversation(self) -> None:
        messages = [
            ModelRequest(
                parts=[
                    SystemPromptPart(content="sys"),
                    UserPromptPart(content="user turn 1"),
                ]
            ),
            PydanticModelResponse(parts=[TextPart(content="assistant turn 1")], model_name="m"),
            ModelRequest(parts=[UserPromptPart(content="user turn 2")]),
        ]
        req = _to_llm_request(messages, None)
        roles = [m.role for m in req.messages]
        assert roles == [Role.SYSTEM, Role.USER, Role.ASSISTANT, Role.USER]

    def test_settings_applied(self) -> None:
        settings: ModelSettings = {"max_tokens": 512, "temperature": 0.7}  # type: ignore[assignment]
        msgs = [ModelRequest(parts=[UserPromptPart(content="hi")])]
        req = _to_llm_request(msgs, settings)
        assert req.max_tokens == 512
        assert req.temperature == pytest.approx(0.7)

    def test_stream_flag_passed(self) -> None:
        msgs = [ModelRequest(parts=[UserPromptPart(content="hi")])]
        req = _to_llm_request(msgs, None, stream=True)
        assert req.stream is True

    def test_no_settings_uses_defaults(self) -> None:
        msgs = [ModelRequest(parts=[UserPromptPart(content="hi")])]
        req = _to_llm_request(msgs, None)
        assert req.max_tokens == 2048
        assert req.temperature == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# _to_request_usage
# ---------------------------------------------------------------------------


class TestToRequestUsage:
    def test_input_tokens_mapped(self) -> None:
        usage = _to_request_usage(_make_token_usage(prompt=20))
        assert usage.input_tokens == 20

    def test_output_tokens_mapped(self) -> None:
        usage = _to_request_usage(_make_token_usage(completion=8))
        assert usage.output_tokens == 8

    def test_thinking_tokens_in_details(self) -> None:
        usage = _to_request_usage(_make_token_usage(thinking=100))
        assert usage.details.get("thinking_tokens") == 100

    def test_no_thinking_tokens_no_details(self) -> None:
        usage = _to_request_usage(_make_token_usage(thinking=0))
        assert "thinking_tokens" not in (usage.details or {})


# ---------------------------------------------------------------------------
# ManagedModelAdapter.request()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_managed_adapter_request_returns_model_response() -> None:
    service = MagicMock()
    service.complete = AsyncMock(return_value=_make_llm_response("world"))

    adapter = ManagedModelAdapter(
        service=service,
        provider_type=ProviderType.ANTHROPIC,
        default_model="claude-3-5-sonnet",
    )
    result = await adapter.request(
        _make_messages(),
        None,
        _make_model_request_parameters(),
    )

    from pydantic_ai.messages import TextPart as TP

    assert any(isinstance(p, TP) and p.content == "world" for p in result.parts)


@pytest.mark.asyncio
async def test_managed_adapter_request_embeds_usage() -> None:
    service = MagicMock()
    service.complete = AsyncMock(return_value=_make_llm_response())

    adapter = ManagedModelAdapter(service=service, provider_type=ProviderType.ANTHROPIC)
    result = await adapter.request(_make_messages(), None, _make_model_request_parameters())

    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 5


@pytest.mark.asyncio
async def test_managed_adapter_model_name_uses_default_model() -> None:
    service = MagicMock()
    adapter = ManagedModelAdapter(
        service=service,
        provider_type=ProviderType.ANTHROPIC,
        default_model="claude-3-haiku",
    )
    assert adapter.model_name == "claude-3-haiku"


@pytest.mark.asyncio
async def test_managed_adapter_model_name_falls_back_to_provider_type() -> None:
    service = MagicMock()
    adapter = ManagedModelAdapter(service=service, provider_type=ProviderType.OPENAI)
    assert adapter.model_name == "openai"


@pytest.mark.asyncio
async def test_managed_adapter_injects_default_model_into_request() -> None:
    service = MagicMock()
    service.complete = AsyncMock(return_value=_make_llm_response())

    adapter = ManagedModelAdapter(
        service=service,
        provider_type=ProviderType.ANTHROPIC,
        default_model="claude-3-opus",
    )
    await adapter.request(_make_messages(), None, _make_model_request_parameters())

    call_args = service.complete.call_args
    llm_req = call_args[0][0]
    assert llm_req.model == "claude-3-opus"


# ---------------------------------------------------------------------------
# ManagedModelAdapter.request_stream()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_managed_adapter_stream_yields_streamed_response() -> None:
    from pydantic_ai.models import StreamedResponse

    service = MagicMock()
    usage = _make_token_usage(prompt=5, completion=3)

    async def _fake_stream(*args: Any, **kwargs: Any) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(delta="hel", is_final=False)
        yield StreamChunk(delta="lo", is_final=True, finish_reason=FinishReason.STOP, usage=usage)

    service.stream = _fake_stream

    adapter = ManagedModelAdapter(service=service, provider_type=ProviderType.ANTHROPIC)
    mrp = _make_model_request_parameters()

    async with adapter.request_stream(_make_messages(), None, mrp) as stream:
        assert isinstance(stream, StreamedResponse)
        chunks = [ev async for ev in stream]

    assert len(chunks) >= 1  # at least one PartStartEvent or PartDeltaEvent


@pytest.mark.asyncio
async def test_managed_stream_usage_populated_after_iteration() -> None:
    service = MagicMock()
    usage = _make_token_usage(prompt=7, completion=4)

    async def _fake_stream(*args: Any, **kwargs: Any) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(delta="hi", is_final=False)
        yield StreamChunk(delta="", is_final=True, finish_reason=FinishReason.STOP, usage=usage)

    service.stream = _fake_stream

    adapter = ManagedModelAdapter(service=service, provider_type=ProviderType.ANTHROPIC)
    mrp = _make_model_request_parameters()

    async with adapter.request_stream(_make_messages(), None, mrp) as stream:
        async for _ in stream:
            pass
        response = stream.get()

    assert response.usage.input_tokens == 7
    assert response.usage.output_tokens == 4


# ---------------------------------------------------------------------------
# BYOKModelAdapter.request()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_byok_adapter_request_delegates_to_byok_service() -> None:
    byok_service = MagicMock()
    byok_service.complete = AsyncMock(return_value=_make_llm_response("byok response"))
    byok_service.__repr__ = MagicMock(return_value="BYOKLLMService(openai)")

    adapter = BYOKModelAdapter(byok_service=byok_service)
    result = await adapter.request(_make_messages(), None, _make_model_request_parameters())

    from pydantic_ai.messages import TextPart as TP

    assert any(isinstance(p, TP) and p.content == "byok response" for p in result.parts)


@pytest.mark.asyncio
async def test_byok_adapter_model_name_uses_service_repr() -> None:
    byok_service = MagicMock()
    byok_service.__repr__ = MagicMock(return_value="BYOKLLMService(anthropic)")

    adapter = BYOKModelAdapter(byok_service=byok_service)
    assert adapter.model_name == "BYOKLLMService(anthropic)"


# ---------------------------------------------------------------------------
# KitkatStreamedResponse metadata properties
# ---------------------------------------------------------------------------


def test_kitkat_streamed_response_properties() -> None:
    async def _empty_iter() -> AsyncIterator[StreamChunk]:
        return
        yield  # make it an async generator

    mrp = _make_model_request_parameters()
    stream = KitkatStreamedResponse(
        model_request_parameters=mrp,
        _kitkat_chunks=_empty_iter(),
        _kitkat_model_name="claude-3",
        _kitkat_provider_name="anthropic",
        _kitkat_provider_url="https://api.anthropic.com",
    )
    assert stream.model_name == "claude-3"
    assert stream.provider_name == "anthropic"
    assert stream.provider_url == "https://api.anthropic.com"
    assert stream.timestamp is not None
