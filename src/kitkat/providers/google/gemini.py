"""Gemini API provider for the Kitkat service layer.

This module integrates the official google-genai SDK targeting Google's
Generative Language API (Google AI Studio), supporting synchronized and
asynchronous (streaming) LLM calls, robust error mapping, and Gemini-specific
thinking-level configuration.

Supported features
------------------
 - Blocking completions via aio.models.generate_content
 - True async streaming via aio.models.generate_content_stream
 - Pre-flight token counting via aio.models.count_tokens (async)
 - Sync approximation via tiktoken cl100k_base (count_tokens)
 - Automatic system_instruction extraction (Google top-level param)
 - Full Google FinishReason → our FinishReason mapping (SAFETY, RECITATION…)
 - Health-check via zero-cost count_tokens probe
 - Gemini thinking-level configuration (LOW / MEDIUM / HIGH)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import tiktoken
from google.genai import Client
from google.genai import errors as genai_errors
from google.genai import types as genai_types

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from ...abc.provider import LLMProvider
from ...core.enums import FinishReason, ProviderType, Role
from ...core.exceptions import (
    LLMAuthenticationError,
    LLMContentFilterError,
    LLMProviderError,
    LLMProviderInitError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMTokenLimitError,
)
from ...core.models import (
    LLMRequest,
    LLMResponse,
    Message,
    ProviderCapabilities,
    RetryPolicy,
    StreamChunk,
    ThinkingConfig,
    TokenUsage,
)

logger = logging.getLogger(__name__)


_DEFAULT_MODEL = "gemini-3.5-flash-lite"
_MAX_CONTEXT_TOKENS = 1_048_576
_HEALTH_CHECK_TIMEOUT_S = 5.0

# Translates Google finish reasons to internal FinishReason enum.
_FINISH_REASON_MAP: dict[str, FinishReason] = {
    "STOP": FinishReason.STOP,
    "MAX_TOKENS": FinishReason.LENGTH,
    "SAFETY": FinishReason.CONTENT_FILTER,
    "RECITATION": FinishReason.CONTENT_FILTER,
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

# Guards against repeated tiktoken BPE download attempts in air-gapped environments.
_TIKTOKEN_UNAVAILABLE = object()

_EFFORT_TO_LEVEL: dict[str, str] = {
    "low": "LOW",
    "medium": "MEDIUM",
    "high": "HIGH",
}


# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------


@dataclass
class GeminiConfig:
    """Typed configuration for the Gemini API provider.

    This provider targets Google's Generative Language API (Google AI Studio)
    and authenticates exclusively via an API key. For GCP-hosted Vertex AI
    deployments, use :class:`~providers.google.vertex_ai.VertexAIProvider`
    instead.

    Attributes:
        api_key: Google AI Studio API key (GOOGLE_API_KEY).
        model: Default model identifier when ``LLMRequest.model`` is empty.
        timeout_s: Per-request timeout in seconds.
        extra_headers: Additional HTTP headers injected into every request.
    """

    api_key: str = ""
    model: str = _DEFAULT_MODEL
    timeout_s: float = 60.0
    extra_headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise LLMProviderInitError(
                "GeminiConfig.api_key must be a non-empty string. "
                "Set GOOGLE_API_KEY in your env, or use VertexAIProvider "
                "for GCP-hosted deployments.",
                provider="gemini",
            )
        if self.timeout_s <= 0:
            raise LLMProviderInitError(
                f"GeminiConfig.timeout_s must be positive, got {self.timeout_s}",
                provider="gemini",
            )

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> GeminiConfig:
        """Build from the raw config slice.

        Args:
            cfg: The configuration dictionary. Recognised keys:
                ``api_key``, ``model``, ``timeout_s``, ``extra_headers``.
                Vertex-specific keys (``vertexai``, ``project``, ``location``)
                are silently ignored — use :class:`VertexAIProvider` for those.

        Returns:
            A :class:`GeminiConfig` instance.
        """
        return cls(
            api_key=cfg.get("api_key", ""),
            model=cfg.get("model", _DEFAULT_MODEL),
            timeout_s=float(cfg.get("timeout_s", 60.0)),
            extra_headers=dict(cfg.get("extra_headers", {})),
        )


# ===========================================================================
# Provider implementation
# ===========================================================================


class GeminiProvider(LLMProvider):
    """Google Gemini API provider implementation.

    Wraps the ``google-genai`` SDK in API-key mode, targeting the
    Generative Language endpoint (Google AI Studio). Supports blocking
    completions, true async streaming, token counting, health checks,
    and Gemini-specific thinking-level configuration.

    For GCP-hosted Vertex AI deployments, use
    :class:`~providers.google.vertex_ai.VertexAIProvider`.
    """

    PROVIDER_TYPE = ProviderType.GEMINI
    DEFAULT_MODEL = _DEFAULT_MODEL
    CAPABILITIES = ProviderCapabilities(
        supports_streaming=True,
        supports_system_prompt=True,
        supports_tool_calling=True,
        supports_vision=True,
        supports_thinking=True,
        max_context_tokens=_MAX_CONTEXT_TOKENS,
        provider_type=ProviderType.GEMINI,
    )
    RETRY_POLICY = RetryPolicy(
        max_attempts=3,
        base_delay_s=2.0,  # Uses extended back-off for quota limits
        max_delay_s=60.0,
        exponential_base=2.0,
        jitter=True,
        retryable_status_codes=frozenset({408, 429, 500, 502, 503, 504}),
    )

    def __init__(self, config: GeminiConfig | dict[str, Any]) -> None:
        """Initialize the GeminiProvider.

        Args:
            config: A :class:`GeminiConfig` instance or a raw dict that
                will be coerced via :meth:`GeminiConfig.from_dict`.
        """
        if isinstance(config, dict):
            config = GeminiConfig.from_dict(config)

        super().__init__(config.__dict__)
        self._cfg: GeminiConfig = config
        self._client: Client | None = None
        self._encoder: Any = None

    # ------------------------------------------------------------------
    # Lifecycle methods
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Instantiate the Client and run a credential probe.

        Raises:
            LLMProviderInitError: If credentials or network communication fail.
        """
        if self._initialized:
            logger.debug("GeminiProvider already initialised — skipping.")
            return

        logger.info(
            "Initialising GeminiProvider (model=%r).",
            self._cfg.model,
        )

        try:
            self._client = Client(
                api_key=self._cfg.api_key,
                http_options=self._build_http_options(),
            )
        except Exception as exc:
            raise LLMProviderInitError(
                "Failed to create google-genai Client.",
                provider="gemini",
            ) from exc

        # Probes credentials via zero-inference token check.
        try:
            await asyncio.wait_for(
                self._client.aio.models.count_tokens(
                    model=self._cfg.model,
                    contents="ping",
                ),
                timeout=_HEALTH_CHECK_TIMEOUT_S,
            )
        except genai_errors.ClientError as exc:
            if exc.code in {401, 403}:
                raise LLMProviderInitError(
                    f"Gemini API key is invalid or lacks permission: {exc.message}",
                    provider="gemini",
                ) from exc
            logger.warning(
                "GeminiProvider credential probe returned %s (non-fatal): %s",
                exc.code,
                exc.message,
            )
        except Exception as exc:
            logger.warning("GeminiProvider credential probe failed (non-fatal): %s", exc)

        self._initialized = True
        logger.info("GeminiProvider initialised successfully.")

    async def shutdown(self) -> None:
        """Close the Google SDK client."""
        if self._client is not None:
            try:
                await self._client.aio.aclose()
                self._client.close()
            except Exception as exc:
                logger.warning("Error closing Gemini client: %s", exc)
            finally:
                self._client = None
                self._initialized = False
                logger.debug("GeminiProvider shut down.")

    async def _init_client_only(self) -> None:
        """Create the google-genai Client without running a credential probe.

        Intended for use by :class:`~kitkat.services.llm.byok.BYOKLLMService`
        so that authentication errors surface from the first inference call
        rather than from a pre-flight ``count_tokens`` probe, avoiding extra
        latency and billable probe requests for each BYOK user key.

        Raises:
            LLMProviderInitError: If the google-genai Client cannot be created.
        """
        if self._initialized:
            return

        try:
            self._client = Client(
                api_key=self._cfg.api_key,
                http_options=self._build_http_options(),
            )
        except Exception as exc:
            raise LLMProviderInitError(
                "Failed to create google-genai Client.",
                provider="gemini",
            ) from exc

        self._initialized = True
        logger.debug("GeminiProvider client created (credential probe skipped).")

    # ------------------------------------------------------------------
    # Core inference methods
    # ------------------------------------------------------------------

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Execute a blocking completion request.

        Args:
            request: The generation request.

        Returns:
            A populated :class:`LLMResponse`.

        Raises:
            LLMTimeoutError: If the execution time limit is reached.
            LLMProviderError: On error executing the completion request.
            LLMContentFilterError: If the response is blocked by safety filters.
        """
        self._assert_initialized()
        assert self._client is not None

        model = request.model or self._cfg.model
        system_instruction, contents = self._split_messages(request.messages)
        gen_cfg = self._build_generation_config(request, system_instruction, request.thinking)
        timeout = request.timeout if request.timeout is not None else self._cfg.timeout_s
        start = time.monotonic()

        logger.debug(
            "Gemini complete | model=%s turns=%d thinking=%s",
            model,
            len(contents),
            request.thinking.enabled if request.thinking else False,
        )

        try:
            raw: genai_types.GenerateContentResponse = await asyncio.wait_for(
                self._client.aio.models.generate_content(
                    model=model,
                    contents=contents,
                    config=gen_cfg,
                ),
                timeout=timeout,
            )
        except TimeoutError as exc:
            elapsed = time.monotonic() - start
            raise LLMTimeoutError(
                f"Gemini request timed out after {elapsed:.1f}s (limit={timeout}s)",
                elapsed_s=elapsed,
                provider="gemini",
            ) from exc
        except genai_errors.ClientError as exc:
            raise self._map_client_error(exc) from exc
        except genai_errors.ServerError as exc:
            raise LLMProviderError(
                f"Gemini server error: {exc.message}",
                status_code=exc.code,
                provider="gemini",
            ) from exc
        except genai_errors.APIError as exc:
            raise LLMProviderError(
                f"Gemini API error: {exc.message}",
                status_code=exc.code,
                provider="gemini",
            ) from exc

        return self._build_response(raw, request, start)

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        """Yield token deltas as an async stream from the Gemini API.

        Args:
            request: The streaming generation request.

        Yields:
            :class:`StreamChunk` objects — one per token delta. The final
            chunk has ``is_final=True`` and carries aggregated ``usage``,
            ``model``, ``provider``, ``finish_reason``, and ``latency_ms``.

        Raises:
            LLMTimeoutError: If stream connection operations time out.
            LLMProviderError: On error streaming from the API.
            LLMContentFilterError: If the response is blocked by safety filters.
        """
        self._assert_initialized()
        assert self._client is not None

        model = request.model or self._cfg.model
        system_instruction, contents = self._split_messages(request.messages)
        gen_cfg = self._build_generation_config(request, system_instruction, request.thinking)
        timeout = request.timeout if request.timeout is not None else self._cfg.timeout_s
        start = time.monotonic()

        logger.debug(
            "Gemini stream | model=%s thinking=%s",
            model,
            request.thinking.enabled if request.thinking else False,
        )

        finish_reason = FinishReason.UNKNOWN
        usage = TokenUsage.empty()
        model_version = model
        any_chunk_yielded = False

        try:
            async with asyncio.timeout(timeout):
                async for chunk in await self._client.aio.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=gen_cfg,
                ):
                    # Updates usage and finish_reason properties incrementally.
                    # Retains the last non-None value as authoritative final state.
                    if chunk.usage_metadata is not None:
                        thinking_toks = (
                            getattr(chunk.usage_metadata, "thoughts_token_count", None) or 0
                        )
                        usage = TokenUsage(
                            prompt_tokens=chunk.usage_metadata.prompt_token_count or 0,
                            completion_tokens=(chunk.usage_metadata.candidates_token_count or 0),
                            thinking_tokens=thinking_toks,
                            total_tokens=chunk.usage_metadata.total_token_count or 0,
                        )
                    if chunk.candidates:
                        candidate = chunk.candidates[0]
                        if candidate.finish_reason is not None:
                            finish_reason = _FINISH_REASON_MAP.get(
                                candidate.finish_reason.value, FinishReason.UNKNOWN
                            )
                    if chunk.model_version:
                        model_version = chunk.model_version

                    # Distinguish thinking parts from answer parts.
                    if chunk.candidates:
                        content = chunk.candidates[0].content
                        if content is not None and content.parts is not None:
                            for part in content.parts:
                                if hasattr(part, "thought") and part.thought:
                                    if part.text:
                                        any_chunk_yielded = True
                                        yield StreamChunk(delta=part.text, is_thinking=True)
                                elif part.text:
                                    any_chunk_yielded = True
                                    yield StreamChunk(delta=part.text, is_thinking=False)

        except TimeoutError as exc:
            elapsed = time.monotonic() - start
            raise LLMTimeoutError(
                f"Gemini stream timed out after {elapsed:.1f}s",
                elapsed_s=elapsed,
                provider="gemini",
            ) from exc
        except genai_errors.ClientError as exc:
            raise self._map_client_error(exc) from exc
        except genai_errors.ServerError as exc:
            raise LLMProviderError(
                f"Gemini server error (stream): {exc.message}",
                status_code=exc.code,
                provider="gemini",
            ) from exc
        except genai_errors.APIError as exc:
            raise LLMProviderError(
                f"Gemini API error (stream): {exc.message}",
                status_code=exc.code,
                provider="gemini",
            ) from exc

        # Raises content-filter error after stream if safety policies block all output.
        if finish_reason == FinishReason.CONTENT_FILTER:
            raise LLMContentFilterError(
                "Gemini stream blocked by content/safety filter.",
                provider="gemini",
            )

        # Guarantees minimum one chunk yield for empty responses.
        if not any_chunk_yielded:
            yield StreamChunk(delta="")

        # Yields final sentinel chunk with authoritative metadata.
        yield StreamChunk(
            delta="",
            is_final=True,
            finish_reason=finish_reason,
            usage=usage,
            model=model_version,
            provider=ProviderType.GEMINI,
            latency_ms=(time.monotonic() - start) * 1_000,
        )

    # ------------------------------------------------------------------
    # Health and introspection
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        """Probe reachability via a lightweight token count call.

        Returns:
            ``True`` if the provider is fully operational, ``False`` otherwise.
        """
        if self._client is None:
            return False
        try:
            await asyncio.wait_for(
                self._client.aio.models.count_tokens(
                    model=self._cfg.model,
                    contents="ping",
                ),
                timeout=_HEALTH_CHECK_TIMEOUT_S,
            )
            return True
        except Exception as exc:
            logger.warning("GeminiProvider health_check failed: %s", exc)
            return False

    def count_tokens(self, text: str) -> int:
        """Approximate the token count for a text sequence.

        Uses tiktoken's ``cl100k_base`` encoding as a fast local estimator.
        Falls back to a character-based heuristic (``len(text) // 4``) when
        tiktoken is unavailable (e.g., air-gapped environments).

        Args:
            text: The targeted text string.

        Returns:
            The estimated number of tokens (≥ 1 for non-empty input).
        """
        if self._encoder is None:
            try:
                self._encoder = tiktoken.get_encoding("cl100k_base")
            except Exception as exc:
                logger.warning(
                    "tiktoken BPE load failed (%s); falling back to character-based estimate.",
                    exc,
                )
                self._encoder = _TIKTOKEN_UNAVAILABLE

        if self._encoder is _TIKTOKEN_UNAVAILABLE:
            return max(1, len(text) // 4)

        return len(self._encoder.encode(text))

    async def async_count_tokens(self, request: LLMRequest) -> int:
        """Return the exact prompt token count via the Gemini API.

        Delegates to the SDK's native ``count_tokens`` endpoint, which
        accounts for model-specific tokenization, system instructions,
        and multi-modal content.

        Args:
            request: The generation request carrying target messages.

        Returns:
            The exact token count according to the Gemini model.
        """
        self._assert_initialized()
        assert self._client is not None

        _, contents = self._split_messages(request.messages)
        result = await self._client.aio.models.count_tokens(
            model=request.model or self._cfg.model,
            contents=contents,
        )
        return result.total_tokens or 0

    # ------------------------------------------------------------------
    # Private helper methods
    # ------------------------------------------------------------------

    def _build_http_options(self) -> genai_types.HttpOptions:
        """Build :class:`genai_types.HttpOptions` from the provider config.

        Injects any ``extra_headers`` from the config into every outbound
        HTTP request.

        Returns:
            The constructed ``HttpOptions`` instance.
        """
        kwargs: dict[str, Any] = {}
        if self._cfg.extra_headers:
            kwargs["headers"] = self._cfg.extra_headers
        return genai_types.HttpOptions(**kwargs)

    @staticmethod
    def _split_messages(
        messages: list[Message],
    ) -> tuple[str, list[genai_types.Content]]:
        """Separate the system instruction from conversation turns.

        Google's API expects the system prompt as a top-level
        ``system_instruction`` parameter rather than as a message in the
        ``contents`` array. This method extracts all ``SYSTEM`` role
        messages, concatenates them, and maps the remaining messages to
        :class:`genai_types.Content` objects with appropriate Google roles
        (``"user"`` for user/tool, ``"model"`` for assistant).

        Args:
            messages: The list of combined messages.

        Returns:
            A tuple of ``(system_instruction, contents)`` where
            ``system_instruction`` is a joined string (possibly empty) and
            ``contents`` is a list of ``Content`` objects.
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

    @staticmethod
    def _build_generation_config(
        request: LLMRequest,
        system_instruction: str,
        thinking: ThinkingConfig | None = None,
    ) -> genai_types.GenerateContentConfig:
        """Build :class:`genai_types.GenerateContentConfig` from an :class:`LLMRequest`.

        Gemini-specific thinking configuration
        ---------------------------------------
        The Gemini API supports discrete ``thinking_level`` values
        (``LOW``, ``MEDIUM``, ``HIGH``) rather than a numeric token budget.
        The level is resolved in priority order:

        1. ``thinking.provider_options["level"]`` — explicit Gemini-level
           override (must be one of ``"LOW"``, ``"MEDIUM"``, ``"HIGH"``).
        2. ``thinking.effort`` — generic effort string (``"low"``,
           ``"medium"``, ``"high"``) mapped to the corresponding level.
        3. If neither is provided but thinking is enabled, a default
           ``ThinkingConfig`` with ``include_thoughts=True`` is created
           (Gemini picks the level automatically).

        Args:
            request: The generation request.
            system_instruction: The separated system instruction string.
            thinking: Optional thinking configuration.

        Returns:
            The constructed ``GenerateContentConfig`` object.
        """
        thinking_config = None
        if thinking is not None and thinking.enabled:
            opts = thinking.provider_options or {}
            level = opts.get("level")

            if not level and thinking.effort:
                level = _EFFORT_TO_LEVEL.get(thinking.effort)

            if level:
                thinking_config = genai_types.ThinkingConfig(
                    thinking_level=genai_types.ThinkingLevel(level),
                    include_thoughts=True,
                )
            else:
                thinking_config = genai_types.ThinkingConfig(
                    include_thoughts=True,
                )

        return genai_types.GenerateContentConfig(
            system_instruction=system_instruction if system_instruction else None,
            temperature=request.temperature,
            top_p=request.top_p if request.top_p != 1.0 else None,
            max_output_tokens=request.max_tokens,
            stop_sequences=request.stop_sequences if request.stop_sequences else None,
            thinking_config=thinking_config,
        )

    def _build_response(
        self,
        raw: genai_types.GenerateContentResponse,
        request: LLMRequest,
        start: float,
    ) -> LLMResponse:
        """Map a :class:`genai_types.GenerateContentResponse` to an :class:`LLMResponse`.

        Extracts text content and thinking content from the first candidate,
        maps finish reasons, and builds token usage from the response's
        ``usage_metadata``.

        Args:
            raw: The native generate content response.
            request: The originating generation request.
            start: The performance start time (monotonic).

        Returns:
            The constructed :class:`LLMResponse` domain object.

        Raises:
            LLMContentFilterError: If the response was blocked by safety filters.
        """
        content_parts: list[str] = []
        thinking_parts: list[str] = []

        if raw.candidates:
            candidate_content = raw.candidates[0].content
            if candidate_content is not None and candidate_content.parts is not None:
                for part in candidate_content.parts:
                    if hasattr(part, "thought") and part.thought:
                        if part.text:
                            thinking_parts.append(part.text)
                    elif part.text:
                        content_parts.append(part.text)

        content = "".join(content_parts) if content_parts else ""

        finish_reason = FinishReason.UNKNOWN
        if raw.candidates:
            candidate = raw.candidates[0]
            if candidate.finish_reason is not None:
                finish_reason = _FINISH_REASON_MAP.get(
                    candidate.finish_reason.value, FinishReason.UNKNOWN
                )

        if finish_reason == FinishReason.CONTENT_FILTER:
            raise LLMContentFilterError(
                "Gemini response blocked by content/safety filter.",
                provider="gemini",
            )

        usage = TokenUsage.empty()
        if raw.usage_metadata is not None:
            thinking_toks = getattr(raw.usage_metadata, "thoughts_token_count", None) or 0
            usage = TokenUsage(
                prompt_tokens=raw.usage_metadata.prompt_token_count or 0,
                completion_tokens=raw.usage_metadata.candidates_token_count or 0,
                thinking_tokens=thinking_toks,
                total_tokens=raw.usage_metadata.total_token_count or 0,
            )

        model_version = getattr(raw, "model_version", None) or (request.model or self._cfg.model)

        return LLMResponse(
            content=content,
            thinking_content="".join(thinking_parts),
            finish_reason=finish_reason,
            usage=usage,
            model=model_version,
            provider=ProviderType.GEMINI,
            latency_ms=(time.monotonic() - start) * 1_000,
            raw_response=raw,
        )

    def _map_client_error(
        self,
        exc: genai_errors.ClientError,
    ) -> Exception:
        """Map a Google :class:`genai_errors.ClientError` to the most specific :class:`LLMError`.

        Resolution order:
        1. HTTP 401/403 or API-key-related messages → :class:`LLMAuthenticationError`
        2. HTTP 429 → :class:`LLMRateLimitError`
        3. HTTP 400 with token/context mentions → :class:`LLMTokenLimitError`
        4. Fallback → :class:`LLMProviderError`

        Args:
            exc: The native client error.

        Returns:
            The mapped domain exception ready to be raised.
        """
        code = exc.code or 0
        message = exc.message
        msg_lower = (message or "").lower()

        if code in {401, 403} or "api key" in msg_lower or "api_key" in msg_lower:
            return LLMAuthenticationError(
                "Gemini authentication failed.",
                status_code=code,
                provider="gemini",
            )
        if code == 429:
            return LLMRateLimitError(
                "Gemini rate limit exceeded.",
                provider="gemini",
            )
        if code == 400 and ("token" in msg_lower or "context" in msg_lower):
            return LLMTokenLimitError(
                "Prompt exceeds Gemini context window.",
                context_limit=_MAX_CONTEXT_TOKENS,
                provider="gemini",
            )
        return LLMProviderError(
            f"Gemini client error: {message}",
            status_code=code,
            provider="gemini",
        )
