"""Vertex AI provider for the Kitkat service layer.

This module integrates the official google-genai SDK in Vertex AI mode,
targeting Google Cloud's enterprise AI platform. It supports synchronized
and asynchronous (streaming) LLM calls, GCP authentication (Service Account
or ADC), and Vertex AI's specific thinking-budget configuration.

Supported features
------------------
 - Blocking completions via aio.models.generate_content
 - True async streaming via aio.models.generate_content_stream
 - Pre-flight token counting via aio.models.count_tokens (async)
 - Sync approximation via tiktoken cl100k_base (count_tokens)
 - Authentication via Service Account JSON or Application Default Credentials (ADC)
 - Vertex AI-specific thinking_budget configuration (token-based)
 - Robust error mapping for GCP IAM, Quotas, and regional availability
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
from google.oauth2 import service_account

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from ...abc.provider import LLMProvider
from ...core.enums import FinishReason, ProviderType
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
    ProviderCapabilities,
    RetryPolicy,
    StreamChunk,
    ThinkingConfig,
    TokenUsage,
)
from ._shared import FINISH_REASON_MAP, TIKTOKEN_UNAVAILABLE, split_messages

logger = logging.getLogger(__name__)


_DEFAULT_MODEL = "gemini-1.5-pro-002"
_MAX_CONTEXT_TOKENS = 2_097_152
_HEALTH_CHECK_TIMEOUT_S = 5.0

# Vertex AI uses token budgets for thinking, unlike AI Studio's LOW/MEDIUM/HIGH enum.
# This maps generic effort strings to safe enterprise token budgets.
_EFFORT_TO_THINKING_BUDGET: dict[str, int] = {
    "low": 1024,
    "medium": 8192,
    "high": 24576,
}


@dataclass
class VertexAIConfig:
    """Typed configuration for the Vertex AI provider.

    Authentication is resolved in the following order:
    1. If ``credentials_path`` is provided, loads the service account JSON.
    2. Otherwise, falls back to Application Default Credentials (ADC), which
       is standard for GCP workloads (Cloud Run, GKE, Compute Engine).

    Attributes:
        project: The GCP Project ID where the API is billed.
        location: The GCP region (e.g., "us-central1", "europe-west1").
        credentials_path: Optional path to a service account JSON file.
        model: Default model identifier (e.g., "gemini-1.5-pro-002").
        timeout_s: Per-request timeout in seconds.
        extra_headers: Additional HTTP headers injected into every request.
    """

    project: str = ""
    location: str = ""
    credentials_path: str = ""
    model: str = _DEFAULT_MODEL
    timeout_s: float = 60.0
    extra_headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.project.strip() or not self.location.strip():
            raise LLMProviderInitError(
                "VertexAIConfig.project and VertexAIConfig.location are required.",
                provider="vertex_ai",
            )
        if self.timeout_s <= 0:
            raise LLMProviderInitError(
                f"VertexAIConfig.timeout_s must be positive, got {self.timeout_s}",
                provider="vertex_ai",
            )

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> VertexAIConfig:
        """Build from the raw config slice.

        Args:
            cfg: The configuration dictionary.

        Returns:
            A VertexAIConfig instance.
        """
        return cls(
            project=cfg.get("project", ""),
            location=cfg.get("location", ""),
            credentials_path=cfg.get("credentials_path", ""),
            model=cfg.get("model", _DEFAULT_MODEL),
            timeout_s=float(cfg.get("timeout_s", 60.0)),
            extra_headers=dict(cfg.get("extra_headers", {})),
        )


class VertexAIProvider(LLMProvider):
    """Google Vertex AI provider implementation.

    Wraps the ``google-genai`` SDK in Vertex mode for enterprise deployments.
    Handles GCP authentication, regional routing, and token-based thinking
    budgets.
    """

    PROVIDER_TYPE = ProviderType.VERTEX_AI
    DEFAULT_MODEL = _DEFAULT_MODEL
    CAPABILITIES = ProviderCapabilities(
        supports_streaming=True,
        supports_system_prompt=True,
        supports_tool_calling=True,
        supports_vision=True,
        supports_thinking=True,
        max_context_tokens=_MAX_CONTEXT_TOKENS,
        provider_type=ProviderType.VERTEX_AI,
    )
    RETRY_POLICY = RetryPolicy(
        max_attempts=3,
        base_delay_s=2.0,
        max_delay_s=60.0,
        exponential_base=2.0,
        jitter=True,
        retryable_status_codes=frozenset({408, 429, 500, 502, 503, 504}),
    )

    def __init__(self, config: VertexAIConfig | dict[str, Any]) -> None:
        """Initialize the VertexAIProvider.

        Args:
            config: A VertexAIConfig instance or a raw dict.
        """
        if isinstance(config, dict):
            config = VertexAIConfig.from_dict(config)

        super().__init__(config.__dict__)
        self._cfg: VertexAIConfig = config
        self._client: Client | None = None
        self._encoder: Any = None

    # ------------------------------------------------------------------
    # Lifecycle methods
    # ------------------------------------------------------------------

    def _load_credentials(self) -> service_account.Credentials | None:
        """Load GCP credentials if a path is provided.

        Returns:
            Credentials object, or None to defer to ADC.

        Raises:
            LLMProviderInitError: If the credentials file is invalid.
        """
        if not self._cfg.credentials_path:
            return None

        try:
            return service_account.Credentials.from_service_account_file(self._cfg.credentials_path)
        except Exception as exc:
            raise LLMProviderInitError(
                f"Failed to load Vertex AI credentials from {self._cfg.credentials_path}: {exc}",
                provider="vertex_ai",
            ) from exc

    async def initialize(self) -> None:
        """Instantiate the Client and run a credential probe.

        Raises:
            LLMProviderInitError: If credentials or network communication fail.
        """
        if self._initialized:
            logger.debug("VertexAIProvider already initialised — skipping.")
            return

        logger.info(
            "Initialising VertexAIProvider (project=%s, location=%s, model=%r).",
            self._cfg.project,
            self._cfg.location,
            self._cfg.model,
        )

        creds = self._load_credentials()
        client_kwargs: dict[str, Any] = {
            "vertexai": True,
            "project": self._cfg.project,
            "location": self._cfg.location,
            "http_options": self._build_http_options(),
        }
        if creds:
            client_kwargs["credentials"] = creds

        try:
            self._client = Client(**client_kwargs)
        except Exception as exc:
            raise LLMProviderInitError(
                "Failed to create Vertex AI Client. Ensure GCP credentials are configured.",
                provider="vertex_ai",
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
                    f"Vertex AI authentication or IAM permission failed: {exc.message}",
                    provider="vertex_ai",
                ) from exc
            if exc.code == 404:
                raise LLMProviderInitError(
                    f"Vertex AI model '{self._cfg.model}' not found in {self._cfg.location}.",
                    provider="vertex_ai",
                ) from exc
            logger.warning(
                "VertexAIProvider credential probe returned %s (non-fatal): %s",
                exc.code,
                exc.message,
            )
        except Exception as exc:
            logger.warning("VertexAIProvider credential probe failed (non-fatal): %s", exc)

        self._initialized = True
        logger.info("VertexAIProvider initialised successfully.")

    async def shutdown(self) -> None:
        """Close the Vertex AI SDK client."""
        if self._client is not None:
            try:
                await self._client.aio.aclose()
                self._client.close()
            except Exception as exc:
                logger.warning("Error closing Vertex AI client: %s", exc)
            finally:
                self._client = None
                self._initialized = False
                logger.debug("VertexAIProvider shut down.")

    async def _init_client_only(self) -> None:
        """Create the Vertex AI Client without running a credential probe.

        Intended for use by :class:`~kitkat.services.llm.byok.BYOKLLMService`.

        Raises:
            LLMProviderInitError: If the Client cannot be created.
        """
        if self._initialized:
            return

        creds = self._load_credentials()
        client_kwargs: dict[str, Any] = {
            "vertexai": True,
            "project": self._cfg.project,
            "location": self._cfg.location,
            "http_options": self._build_http_options(),
        }
        if creds:
            client_kwargs["credentials"] = creds

        try:
            self._client = Client(**client_kwargs)
        except Exception as exc:
            raise LLMProviderInitError(
                "Failed to create Vertex AI Client.",
                provider="vertex_ai",
            ) from exc

        self._initialized = True
        logger.debug("VertexAIProvider client created (credential probe skipped).")

    # ------------------------------------------------------------------
    # Core inference methods
    # ------------------------------------------------------------------

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Execute a blocking completion request.

        Args:
            request: The generation request.

        Returns:
            A populated LLMResponse.

        Raises:
            LLMTimeoutError: If the execution time limit is reached.
            LLMProviderError: On error executing the completion request.
            LLMContentFilterError: If the response is blocked by safety filters.
        """
        self._assert_initialized()
        assert self._client is not None

        model = request.model or self._cfg.model
        system_instruction, contents = split_messages(request.messages)
        gen_cfg = self._build_generation_config(request, system_instruction, request.thinking)
        timeout = request.timeout if request.timeout is not None else self._cfg.timeout_s
        start = time.monotonic()

        logger.debug(
            "Vertex AI complete | model=%s turns=%d thinking=%s",
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
                f"Vertex AI request timed out after {elapsed:.1f}s (limit={timeout}s)",
                elapsed_s=elapsed,
                provider="vertex_ai",
            ) from exc
        except genai_errors.ClientError as exc:
            raise self._map_client_error(exc) from exc
        except genai_errors.ServerError as exc:
            raise LLMProviderError(
                f"Vertex AI server error: {exc.message}",
                status_code=exc.code,
                provider="vertex_ai",
            ) from exc
        except genai_errors.APIError as exc:
            raise LLMProviderError(
                f"Vertex AI API error: {exc.message}",
                status_code=exc.code,
                provider="vertex_ai",
            ) from exc

        return self._build_response(raw, request, start)

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        """Yield token deltas as an async stream from the Vertex AI API.

        Args:
            request: The streaming generation request.

        Yields:
            StreamChunk objects.

        Raises:
            LLMTimeoutError: If stream connection operations time out.
            LLMProviderError: On error streaming from the API.
            LLMContentFilterError: If the response is blocked by safety filters.
        """
        self._assert_initialized()
        assert self._client is not None

        model = request.model or self._cfg.model
        system_instruction, contents = split_messages(request.messages)
        gen_cfg = self._build_generation_config(request, system_instruction, request.thinking)
        timeout = request.timeout if request.timeout is not None else self._cfg.timeout_s
        start = time.monotonic()

        logger.debug(
            "Vertex AI stream | model=%s thinking=%s",
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
                            finish_reason = FINISH_REASON_MAP.get(
                                candidate.finish_reason.value, FinishReason.UNKNOWN
                            )
                    if chunk.model_version:
                        model_version = chunk.model_version

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
                f"Vertex AI stream timed out after {elapsed:.1f}s",
                elapsed_s=elapsed,
                provider="vertex_ai",
            ) from exc
        except genai_errors.ClientError as exc:
            raise self._map_client_error(exc) from exc
        except genai_errors.ServerError as exc:
            raise LLMProviderError(
                f"Vertex AI server error (stream): {exc.message}",
                status_code=exc.code,
                provider="vertex_ai",
            ) from exc
        except genai_errors.APIError as exc:
            raise LLMProviderError(
                f"Vertex AI API error (stream): {exc.message}",
                status_code=exc.code,
                provider="vertex_ai",
            ) from exc

        if finish_reason == FinishReason.CONTENT_FILTER:
            raise LLMContentFilterError(
                "Vertex AI stream blocked by content/safety filter.",
                provider="vertex_ai",
            )

        if not any_chunk_yielded:
            yield StreamChunk(delta="")

        yield StreamChunk(
            delta="",
            is_final=True,
            finish_reason=finish_reason,
            usage=usage,
            model=model_version,
            provider=ProviderType.VERTEX_AI,
            latency_ms=(time.monotonic() - start) * 1_000,
        )

    # ------------------------------------------------------------------
    # Health and introspection
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        """Probe reachability via a lightweight token count call.

        Returns:
            True if the provider is fully operational, False otherwise.
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
            logger.warning("VertexAIProvider health_check failed: %s", exc)
            return False

    def count_tokens(self, text: str) -> int:
        """Approximate the token count for a text sequence.

        Uses tiktoken's ``cl100k_base`` encoding as a fast local estimator.
        Falls back to a character-based heuristic when tiktoken is unavailable.

        Args:
            text: The targeted text string.

        Returns:
            The estimated number of tokens.
        """
        if self._encoder is None:
            try:
                self._encoder = tiktoken.get_encoding("cl100k_base")
            except Exception as exc:
                logger.warning(
                    "tiktoken BPE load failed (%s); falling back to character-based estimate.",
                    exc,
                )
                self._encoder = TIKTOKEN_UNAVAILABLE

        if self._encoder is TIKTOKEN_UNAVAILABLE:
            return max(1, len(text) // 4)

        return len(self._encoder.encode(text))

    async def async_count_tokens(self, request: LLMRequest) -> int:
        """Return the exact prompt token count via the Vertex AI API.

        Args:
            request: The generation request carrying target text.

        Returns:
            The specific token count according to Google's models.
        """
        self._assert_initialized()
        assert self._client is not None

        _, contents = split_messages(request.messages)
        result = await self._client.aio.models.count_tokens(
            model=request.model or self._cfg.model,
            contents=contents,
        )
        return result.total_tokens or 0

    # ------------------------------------------------------------------
    # Private helper methods
    # ------------------------------------------------------------------

    def _build_http_options(self) -> genai_types.HttpOptions:
        """Build HttpOptions, injecting any extra headers from the config.

        Returns:
            The constructed HttpOptions.
        """
        kwargs: dict[str, Any] = {}
        if self._cfg.extra_headers:
            kwargs["headers"] = self._cfg.extra_headers
        return genai_types.HttpOptions(**kwargs)

    @staticmethod
    def _build_generation_config(
        request: LLMRequest,
        system_instruction: str,
        thinking: ThinkingConfig | None = None,
    ) -> genai_types.GenerateContentConfig:
        """Build GenerateContentConfig from an LLMRequest.

        Vertex AI-specific thinking configuration
        -------------------------------------------
        Unlike the Gemini API which uses discrete levels (LOW, MEDIUM, HIGH),
        Vertex AI enterprise contracts often require deterministic token budgets.
        The resolution priority is:
        1. ``thinking.provider_options["thinking_budget"]`` (int)
        2. ``thinking.effort`` mapped to predefined enterprise budgets
        3. Default (omit budget, let Vertex decide dynamically)

        Args:
            request: The generation request.
            system_instruction: The separated system instruction.
            thinking: Optional thinking configuration.

        Returns:
            The constructed generation config object.
        """
        thinking_config = None
        if thinking is not None and thinking.enabled:
            opts = thinking.provider_options or {}
            budget = opts.get("thinking_budget")

            if budget is None and thinking.effort:
                budget = _EFFORT_TO_THINKING_BUDGET.get(thinking.effort)

            if budget is not None:
                thinking_config = genai_types.ThinkingConfig(
                    thinking_budget=int(budget),
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
        """Map a GenerateContentResponse to an LLMResponse.

        Args:
            raw: The native generate content response.
            request: The originating generation request.
            start: The performance start time.

        Returns:
            The constructed LLMResponse domain object.

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
                finish_reason = FINISH_REASON_MAP.get(
                    candidate.finish_reason.value, FinishReason.UNKNOWN
                )

        if finish_reason == FinishReason.CONTENT_FILTER:
            raise LLMContentFilterError(
                "Vertex AI response blocked by content/safety filter.",
                provider="vertex_ai",
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
            provider=ProviderType.VERTEX_AI,
            latency_ms=(time.monotonic() - start) * 1_000,
            raw_response=raw,
        )

    def _map_client_error(
        self,
        exc: genai_errors.ClientError,
    ) -> Exception:
        """Map a Vertex AI ClientError to the most specific LLMError.

        Tailored for GCP-specific semantics like IAM permissions, regional
        availability, and project-scoped quotas.

        Args:
            exc: The native client error.

        Returns:
            The mapped domain exception.
        """
        code = exc.code or 0
        message = exc.message
        msg_lower = (message or "").lower()

        # 403 on Vertex often means the service account lacks aiplatform.endpoints.predict
        if code in {401, 403} or "permission" in msg_lower or "denied" in msg_lower:
            return LLMAuthenticationError(
                "Vertex AI authentication failed or IAM permissions insufficient.",
                status_code=code,
                provider="vertex_ai",
            )
        # 404 usually indicates the model name is invalid or not available in the chosen region
        if code == 404 or "not found" in msg_lower:
            return LLMProviderError(
                f"Vertex AI model or resource not found: {message}",
                status_code=code,
                provider="vertex_ai",
            )
        if code == 429:
            return LLMRateLimitError(
                "Vertex AI quota exceeded for project/region.",
                provider="vertex_ai",
            )
        if code == 400 and ("token" in msg_lower or "context" in msg_lower):
            return LLMTokenLimitError(
                "Prompt exceeds Vertex AI context window.",
                context_limit=_MAX_CONTEXT_TOKENS,
                provider="vertex_ai",
            )
        return LLMProviderError(
            f"Vertex AI client error: {message}",
            status_code=code,
            provider="vertex_ai",
        )
