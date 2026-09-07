"""OpenAI embedding provider for the Sentinel RAG pipeline.

This module implements the embedding provider interface for the official openai
Python SDK, ensuring true async operation, native batching, and full retry
logics. All vendor coupling and configuration are contained here.

Supported features
------------------
  - True async via openai.AsyncOpenAI — no run_in_executor
  - Native batch embedding: one API call for any number of texts
  - asyncio.wait_for() for hard per-request timeout enforcement
  - Full retry loop with exponential back-off (reuses RetryPolicy from base.py)
  - Retry-After header honoured on 429 responses
  - Complete openai error → EmbeddingError mapping
  - output dimensions override via the `dimensions` config param
    (supported by text-embedding-3-* family; ignored for ada-002)
  - Token usage captured from response.usage
  - Health-check via a minimal single-string embed call
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import openai
from openai import AsyncOpenAI

from ..abc.embedder import EmbeddingProvider
from ..core.exceptions import (
    EmbeddingAuthError,
    EmbeddingProviderError,
    EmbeddingProviderInitError,
    EmbeddingRateLimitError,
    EmbeddingTimeoutError,
)
from ..core.models import EmbeddingRequest, EmbeddingResult, RetryPolicy

logger = logging.getLogger(__name__)


_DEFAULT_MODEL = "text-embedding-3-small"
_DEFAULT_DIMENSIONS = 1536
_HEALTH_CHECK_TIMEOUT_S = 5.0

_TIKTOKEN_UNAVAILABLE = object()


@dataclass
class OpenAIEmbeddingConfig:
    """
    Configuration definition for the OpenAI embedding provider.

    Attributes:
        api_key: The authentication key for the OpenAI API.
        model: The specified OpenAI model.
        dimensions: Dimensionality of the resulting embeddings.
        base_url: Optional base URL for proxy endpoints.
        timeout_s: Requested timeout per request, in seconds.
        extra_headers: Additional headers forwarded to every API call.
    """

    api_key: str
    model: str = _DEFAULT_MODEL
    dimensions: int | None = None
    base_url: str | None = None
    timeout_s: float = 30.0
    extra_headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Validate configuration after initialization.

        Raises:
            EmbeddingProviderInitError: If api_key is missing, timeout_s is non-positive,
            or dimensions falls below 1.
        """
        if not self.api_key.strip():
            raise EmbeddingProviderInitError(
                "OpenAIEmbeddingConfig.api_key must be a non-empty string. Set OPENAI_API_KEY in "
                "your environment.",
                provider="openai",
            )
        if self.timeout_s <= 0:
            raise EmbeddingProviderInitError(
                f"OpenAIEmbeddingConfig.timeout_s must be positive, got {self.timeout_s}",
                provider="openai",
            )
        if self.dimensions is not None and self.dimensions < 1:
            raise EmbeddingProviderInitError(
                f"OpenAIEmbeddingConfig.dimensions must be ≥ 1, got {self.dimensions}",
                provider="openai",
            )

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> OpenAIEmbeddingConfig:
        """Build the configuration from a raw dictionary."""

        return cls(
            api_key=cfg.get("api_key", ""),
            model=cfg.get("model_name") or cfg.get("model") or _DEFAULT_MODEL,
            dimensions=cfg.get("dimensions"),
            base_url=cfg.get("base_url"),
            timeout_s=float(cfg.get("timeout_s", 30.0)),
            extra_headers=dict(cfg.get("extra_headers", {})),
        )


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI embedding provider for the Sentinel RAG pipeline.

    Wraps the official async SDK to natively batch-embed multiple documents
    with retries, rate limits, and custom dimensionalities configured.
    """

    PROVIDER_NAME = "openai"
    DEFAULT_MODEL = _DEFAULT_MODEL
    RETRY_POLICY = RetryPolicy(
        max_attempts=3,
        base_delay_s=1.0,
        max_delay_s=60.0,
        exponential_base=2.0,
        jitter=True,
        retryable_status_codes=frozenset({408, 429, 500, 502, 503, 504}),
    )

    def __init__(self, config: OpenAIEmbeddingConfig | dict[str, Any]) -> None:
        if isinstance(config, dict):
            config = OpenAIEmbeddingConfig.from_dict(config)
        super().__init__(config.__dict__)
        self._cfg: OpenAIEmbeddingConfig = config
        self._client: AsyncOpenAI | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """
        Instantiate the AsyncOpenAI client and attempt a single credential probe.

        Raises:
            EmbeddingProviderInitError: If instantiation fails or the authentication key
            is immediately flagged as invalid.
        """
        if self._initialized:
            logger.debug("OpenAIEmbeddingProvider already initialised — skipping.")
            return

        logger.info("Initialising OpenAIEmbeddingProvider (model=%r).", self._cfg.model)

        try:
            self._client = AsyncOpenAI(
                api_key=self._cfg.api_key,
                base_url=self._cfg.base_url,
                max_retries=0,
                default_headers=self._cfg.extra_headers or None,
            )
        except Exception as exc:
            raise EmbeddingProviderInitError(
                "Failed to create AsyncOpenAI client.",
                provider="openai",
                details={"error": str(exc)},
            ) from exc

        try:
            await asyncio.wait_for(
                self._client.embeddings.create(
                    model=self._cfg.model,
                    input="ping",
                    encoding_format="float",
                ),
                timeout=_HEALTH_CHECK_TIMEOUT_S,
            )
        except openai.AuthenticationError as exc:
            raise EmbeddingProviderInitError(
                f"OpenAI API key is invalid or revoked: {exc.message}",
                provider="openai",
                details={"error": str(exc)},
            ) from exc
        except Exception as exc:  # noqa: BLE001
            logger.warning("OpenAIEmbeddingProvider credential probe failed (non-fatal): %s", exc)

        self._initialized = True
        logger.info("OpenAIEmbeddingProvider initialised successfully.")

    async def shutdown(self) -> None:
        """Close the underlying HTTPX connection pool."""
        if self._client is not None:
            try:
                await self._client.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error closing OpenAI client: %s", exc)
            finally:
                self._client = None
                self._initialized = False
                logger.debug("OpenAIEmbeddingProvider shut down.")

    # ------------------------------------------------------------------
    # Core embedding
    # ------------------------------------------------------------------

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        """
        Embed all texts in a single batched API call.

        Args:
            request: The payload containing texts to process.

        Returns:
            The embedding result including generated vectors, model configuration,
            and token usage statistics.

        Raises:
            EmbeddingTimeoutError: The overall request exceeded configured timeouts.
            EmbeddingAuthError: Encountered an API key error during processing.
            EmbeddingRateLimitError: Retries failed despite backoff logic.
            EmbeddingProviderError: Underlying OpenAI service error or failure.
        """
        self._assert_initialized()
        assert self._client is not None

        model = request.model or self._cfg.model
        timeout = request.timeout if request.timeout is not None else self._cfg.timeout_s

        dims_kwarg: dict[str, Any] = {}
        if self._cfg.dimensions is not None:
            dims_kwarg["dimensions"] = self._cfg.dimensions

        last_exc: Exception | None = None
        policy = self.RETRY_POLICY

        for attempt in range(policy.max_attempts):
            start = time.monotonic()
            try:
                raw = await asyncio.wait_for(
                    self._client.embeddings.create(
                        model=model,
                        input=request.texts,
                        encoding_format="float",
                        **dims_kwarg,
                    ),
                    timeout=timeout,
                )
            except asyncio.TimeoutError as exc:
                elapsed = self._elapsed_ms(start)
                raise EmbeddingTimeoutError(
                    f"OpenAI embedding timed out after {elapsed / 1000:.1f}s (limit={timeout}s)",
                    elapsed_s=elapsed / 1000,
                    provider="openai",
                    details={"error": str(exc)},
                ) from exc

            except openai.AuthenticationError as exc:
                raise EmbeddingAuthError(
                    f"OpenAI authentication failed: {exc.message}",
                    provider="openai",
                    details={"error": str(exc)},
                ) from exc

            except openai.RateLimitError as exc:
                retry_after = self._parse_retry_after(exc)
                last_exc = EmbeddingRateLimitError(
                    f"OpenAI rate limit exceeded: {exc.message}",
                    retry_after_s=retry_after,
                    provider="openai",
                    details={"error": str(exc)},
                )

            except openai.APIStatusError as exc:
                last_exc = EmbeddingProviderError(
                    f"OpenAI API error: {exc.message}",
                    status_code=exc.status_code,
                    provider="openai",
                    details={"error": str(exc)},
                )

            except openai.APIConnectionError as exc:
                last_exc = EmbeddingProviderError(
                    "OpenAI connection error.",
                    provider="openai",
                    details={"error": str(exc)},
                )

            else:
                latency = self._elapsed_ms(start)

                sorted_data = sorted(raw.data, key=lambda e: e.index)
                vectors = [e.embedding for e in sorted_data]

                return EmbeddingResult(
                    vectors=vectors,
                    model=raw.model,
                    provider=self.PROVIDER_NAME,
                    prompt_tokens=raw.usage.prompt_tokens,
                    latency_ms=latency,
                )

            if attempt < policy.max_attempts - 1:
                delay = policy.delay_for_attempt(attempt)
                logger.warning(
                    "OpenAI embedding attempt %d/%d failed: %s. Retrying in %.2fs.",
                    attempt + 1,
                    policy.max_attempts,
                    last_exc,
                    delay,
                )
                await asyncio.sleep(delay)

        assert last_exc is not None
        raise last_exc

    # ------------------------------------------------------------------
    # Health & introspection
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        """Probe liveness via a single-text mock call."""

        if self._client is None:
            return False
        try:
            await asyncio.wait_for(
                self._client.embeddings.create(
                    model=self._cfg.model,
                    input="ping",
                    encoding_format="float",
                ),
                timeout=_HEALTH_CHECK_TIMEOUT_S,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("OpenAIEmbeddingProvider health_check failed: %s", exc)
            return False

    @property
    def dimensions(self) -> int:
        """Return the configured output dimensionality."""

        if self._cfg.dimensions is not None:
            return self._cfg.dimensions
        return _DEFAULT_DIMENSIONS

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_retry_after(exc: openai.RateLimitError) -> float | None:
        """
        Extract retry-after context from an HTTP 429 response.

        Args:
            exc: The raised exception embedding rate-limits details.

        Returns:
            The parsed wait time in seconds, or None if unspecified.
        """
        try:
            val = exc.response.headers.get("retry-after")
            return float(val) if val else None
        except (AttributeError, ValueError):
            return None
