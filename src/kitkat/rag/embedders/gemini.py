"""Google Gemini embedding provider for the Sentinel RAG pipeline.

This module encapsulates all vendor coupling with the google-genai Python SDK
for embeddings, enabling concurrency-managed batching, detailed retry
logic, explicit error mapping, and asynchronous operations.

Supported features
------------------
  - True async via client.aio.models.embed_content — no run_in_executor
  - Task-type hints: RETRIEVAL_QUERY for queries, RETRIEVAL_DOCUMENT for docs
    (EmbeddingRequest.is_query drives this automatically)
  - Batch embedding via asyncio.gather() with a concurrency semaphore
    to respect API rate limits without blocking the event loop
  - asyncio.timeout() for hard per-request timeout enforcement
  - Full retry loop with exponential back-off (reuses RetryPolicy from base.py)
  - Output dimensionality override via EmbedContentConfig.output_dimensionality
    (supported by text-embedding-004, NOT by models/embedding-001)
  - Health-check via a zero-cost single-string embed call
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from google.genai import Client
from google.genai import errors as genai_errors
from google.genai import types as genai_types

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


_DEFAULT_MODEL = "gemini-embedding-2"
_DEFAULT_DIMENSIONS = 1536
_HEALTH_CHECK_TIMEOUT_S = 5.0

_AUTH_CODES: frozenset[int] = frozenset({401, 403})
_RATE_LIMIT_CODES: frozenset[int] = frozenset({429})


@dataclass
class GeminiEmbeddingConfig:
    """
    Configuration definition for the Gemini embedding provider.

    Attributes:
        api_key: The authentication key for the Gemini API.
        model: The specified Gemini model.
        task_type_query: Gemini task type used when EmbeddingRequest.is_query=True.
        task_type_document: Gemini task type used when EmbeddingRequest.is_query=False.
        output_dimensionality: Reduce output dimensions (supported by text-embedding-004).
        vertexai: Indicates if the Vertex AI endpoint is active.
        project: The GCP project for Vertex AI.
        location: The GCP region for Vertex AI.
        timeout_s: Requested timeout per request, in seconds.
        max_concurrent: Maximum number of concurrent embed calls for batching.
    """

    api_key: str = ""
    model: str = _DEFAULT_MODEL
    task_type_query: str = "RETRIEVAL_QUERY"
    task_type_document: str = "RETRIEVAL_DOCUMENT"
    output_dimensionality: int | None = None
    vertexai: bool = False
    project: str = ""
    location: str = ""
    timeout_s: float = 30.0
    max_concurrent: int = 10

    def __post_init__(self) -> None:
        """
        Validate configuration after initialization.

        Raises:
            EmbeddingProviderInitError: If vertexai configuration is invalid,
                api_key is missing, max_concurrent is less than 1, or
                timeout is non-positive.
        """
        if not self.vertexai and not self.api_key.strip():
            raise EmbeddingProviderInitError(
                "GeminiEmbeddingConfig.api_key must be a non-empty string when not using Vertex AI."
                " Set GEMINI_API_KEY in your environment.",
                provider="gemini",
            )
        if self.vertexai and (not self.project or not self.location):
            raise EmbeddingProviderInitError(
                "GeminiEmbeddingConfig.project and location are required when 'vertexai=True'.",
                provider="gemini",
            )
        if self.timeout_s <= 0:
            raise EmbeddingProviderInitError(
                f"GeminiEmbeddingConfig.timeout_s must be positive, got {self.timeout_s}",
                provider="gemini",
            )
        if self.max_concurrent < 1:
            raise EmbeddingProviderInitError(
                f"GeminiEmbeddingConfig.max_concurrent must be ≥ 1, got {self.max_concurrent}",
                provider="gemini",
            )

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> GeminiEmbeddingConfig:
        """Build the configuration from a raw dictionary."""

        return cls(
            api_key=cfg.get("api_key", ""),
            model=cfg.get("model_name") or cfg.get("model") or _DEFAULT_MODEL,
            task_type_query=cfg.get("task_type_query", "RETRIEVAL_QUERY"),
            task_type_document=cfg.get("task_type_document", "RETRIEVAL_DOCUMENT"),
            output_dimensionality=cfg.get("output_dimensionality"),
            vertexai=bool(cfg.get("vertexai", False)),
            project=cfg.get("project", ""),
            location=cfg.get("location", ""),
            timeout_s=float(cfg.get("timeout_s", 30.0)),
            max_concurrent=int(cfg.get("max_concurrent", 10)),
        )


class GeminiEmbeddingProvider(EmbeddingProvider):
    """
    Google Gemini embedding provider for the Sentinel RAG pipeline.

    Provides concurrent batch embedding to respect rate-limiting limits via
    task-type hints matching document type.
    """

    PROVIDER_NAME = "gemini"
    DEFAULT_MODEL = _DEFAULT_MODEL
    RETRY_POLICY = RetryPolicy(
        max_attempts=3,
        base_delay_s=2.0,  # Gemini quota errors need longer back-off
        max_delay_s=60.0,
        exponential_base=2.0,
        jitter=True,
        retryable_status_codes=frozenset({408, 429, 500, 502, 503, 504}),
    )

    def __init__(self, config: GeminiEmbeddingConfig | dict[str, Any]) -> None:
        if isinstance(config, dict):
            config = GeminiEmbeddingConfig.from_dict(config)
        super().__init__(config.__dict__)
        self._cfg: GeminiEmbeddingConfig = config
        self._client: Client | None = None
        self._semaphore: asyncio.Semaphore | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """
        Instantiate the google-genai client and attempt credential validation.

        Raises:
            EmbeddingProviderInitError: If instantiation fails or the proxy
                credentials lack adequate permissions.
        """
        if self._initialized:
            logger.debug("GeminiEmbeddingProvider already initialised — skipping.")
            return

        logger.info(
            "Initialising GeminiEmbeddingProvider (model=%r, vertexai=%s).",
            self._cfg.model,
            self._cfg.vertexai,
        )

        try:
            if self._cfg.vertexai:
                self._client = Client(
                    vertexai=True,
                    project=self._cfg.project,
                    location=self._cfg.location,
                )
            else:
                self._client = Client(api_key=self._cfg.api_key)
        except Exception as exc:
            raise EmbeddingProviderInitError(
                f"Failed to create google-genai Client: {exc}",
                provider="gemini",
                details={"error": str(exc)},
            ) from exc

        self._semaphore = asyncio.Semaphore(self._cfg.max_concurrent)

        try:
            await asyncio.wait_for(
                self._client.aio.models.embed_content(
                    model=self._cfg.model,
                    contents="ping",
                ),
                timeout=_HEALTH_CHECK_TIMEOUT_S,
            )
        except genai_errors.ClientError as exc:
            if exc.code in _AUTH_CODES:
                raise EmbeddingProviderInitError(
                    f"Gemini API key is invalid or lacks permission: {exc.message}",
                    provider="gemini",
                    details={"error": str(exc)},
                ) from exc
            logger.warning(
                "GeminiEmbeddingProvider credential probe returned %s (non-fatal): %s",
                exc.code,
                exc.message,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("GeminiEmbeddingProvider credential probe failed (non-fatal): %s", exc)

        self._initialized = True
        logger.info("GeminiEmbeddingProvider initialised successfully.")

    async def shutdown(self) -> None:
        """Close the underlying HTTPX async client pool."""
        if self._client is not None:
            try:
                await self._client.aio.aclose()
                self._client.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error closing Gemini client: %s", exc)
            finally:
                self._client = None
                self._semaphore = None
                self._initialized = False
                logger.debug("GeminiEmbeddingProvider shut down.")

    # ------------------------------------------------------------------
    # Core embedding
    # ------------------------------------------------------------------

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        """
        Embed all texts concurrently using bounded concurrent execution.

        Args:
            request: The payload containing texts to process.

        Returns:
            The embedding result including generated vectors, model configuration,
            and total operation latency.

        Raises:
            EmbeddingTimeoutError: Execution exceeded provided timeout.
            EmbeddingRateLimitError: Quota constraints triggered failure after retries.
            EmbeddingAuthError: Encountered invalid authorization headers or token.
            EmbeddingProviderError: Encountered a native client or API error.
        """
        self._assert_initialized()
        assert self._client is not None
        assert self._semaphore is not None

        model = request.model or self._cfg.model
        task_type = self._cfg.task_type_query if request.is_query else self._cfg.task_type_document
        embed_config = self._build_embed_config(task_type)
        timeout = request.timeout if request.timeout is not None else self._cfg.timeout_s
        per_text_timeout = min(timeout, self._cfg.timeout_s)

        start = time.monotonic()

        try:
            vectors: list[list[float]] = await asyncio.wait_for(
                self._embed_batch(
                    texts=request.texts,
                    model=model,
                    embed_config=embed_config,
                    per_text_timeout=per_text_timeout,
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError as exc:
            elapsed = self._elapsed_ms(start)
            raise EmbeddingTimeoutError(
                f"Gemini embedding timed out after {elapsed / 1000:.1f}s (limit={timeout}s)",
                elapsed_s=elapsed / 1000,
                provider="gemini",
                details={"error": str(exc)},
            ) from exc

        return EmbeddingResult(
            vectors=vectors,
            model=model,
            provider=self.PROVIDER_NAME,
            prompt_tokens=0,
            latency_ms=self._elapsed_ms(start),
        )

    async def _embed_batch(
        self,
        texts: list[str],
        model: str,
        embed_config: genai_types.EmbedContentConfig,
        per_text_timeout: float,
    ) -> list[list[float]]:
        """
        Embed distinct records efficiently bounded by concurrency tokens.

        Args:
            texts: List of text components acting as queries or documents.
            model: Designated model specification token for the engine.
            embed_config: Generated model task limits and type contexts.
            per_text_timeout: Fallback task resolution maximum allowed duration.

        Returns:
            Vectors aligned linearly according to the input configuration payload.
        """

        async def embed_one(text: str, idx: int) -> list[float]:
            policy = self.RETRY_POLICY
            last_exc: Exception | None = None

            for attempt in range(policy.max_attempts):
                async with self._semaphore:  # type: ignore[union-attr]
                    try:
                        response = await asyncio.wait_for(
                            self._client.aio.models.embed_content(  # type: ignore[union-attr]
                                model=model,
                                contents=text,
                                config=embed_config,
                            ),
                            timeout=per_text_timeout,
                        )
                        embeddings = response.embeddings
                        if not embeddings or not embeddings[0].values:
                            raise EmbeddingProviderError(
                                f"Gemini returned empty embedding for text at index {idx}.",
                                provider="gemini",
                            )
                        return list(embeddings[0].values)

                    except asyncio.TimeoutError as exc:
                        raise EmbeddingTimeoutError(
                            f"Gemini embed_content timed out for text index {idx}.",
                            provider="gemini",
                            details={"error": str(exc)},
                        )

                    except genai_errors.ClientError as exc:
                        mapped = self._map_client_error(exc)
                        if isinstance(mapped, (EmbeddingAuthError,)):
                            raise mapped from exc  # non-retryable
                        last_exc = mapped

                    except genai_errors.ServerError as exc:
                        last_exc = EmbeddingProviderError(
                            f"Gemini server error: {exc.message}",
                            status_code=exc.code,
                            provider="gemini",
                            details={"error": str(exc)},
                        )

                    except genai_errors.APIError as exc:
                        last_exc = EmbeddingProviderError(
                            f"Gemini API error: {exc.message}",
                            status_code=exc.code,
                            provider="gemini",
                            details={"error": str(exc)},
                        )

                    except (EmbeddingAuthError, EmbeddingTimeoutError):
                        raise

                if attempt < policy.max_attempts - 1:
                    delay = policy.delay_for_attempt(attempt)
                    logger.warning(
                        "Gemini embed attempt %d/%d failed for text index %d: %s. "
                        "Retrying in %.2fs.",
                        attempt + 1,
                        policy.max_attempts,
                        idx,
                        last_exc,
                        delay,
                    )
                    await asyncio.sleep(delay)

            assert last_exc is not None
            raise last_exc

        # gather preserves input order and propagates first exception
        return list(await asyncio.gather(*[embed_one(text, i) for i, text in enumerate(texts)]))

    # ------------------------------------------------------------------
    # Health & introspection
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        """
        Probe liveness via a single-text mock call.

        Returns:
            True if the call succeeds, otherwise False.
        """
        if self._client is None:
            return False
        try:
            await asyncio.wait_for(
                self._client.aio.models.embed_content(
                    model=self._cfg.model,
                    contents="ping",
                ),
                timeout=_HEALTH_CHECK_TIMEOUT_S,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("GeminiEmbeddingProvider health_check failed: %s", exc)
            return False

    @property
    def dimensions(self) -> int:
        """Return the configured output dimensionality.

        When "output_dimensionality" is explicitly set it is the authoritative
        value — it matches what the Gemini API will produce and what Qdrant
        expects.  When omitted the provider falls back to "_DEFAULT_DIMENSIONS"
        so the "_validate_embedding_dimensions" check in the engine still has
        a concrete value to compare against.
        """
        if self._cfg.output_dimensionality is not None:
            return self._cfg.output_dimensionality
        return _DEFAULT_DIMENSIONS

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_embed_config(self, task_type: str) -> genai_types.EmbedContentConfig:
        """Assemble the Gemini embed request config for a single batch.

        "output_dimensionality" is forwarded unconditionally when set;
        the Gemini API itself enforces which models accept it, making a
        client-side allowlist unnecessary and fragile.

        Args:
            task_type: Gemini task-type hint (e.g. "RETRIEVAL_QUERY").
            model: Model identifier used for this request.

        Returns:
            "EmbedContentConfig" passed to every "embed_content" call.
        """
        return genai_types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=self._cfg.output_dimensionality,
        )

    def _map_client_error(
        self,
        exc: genai_errors.ClientError,
    ) -> EmbeddingProviderError | EmbeddingAuthError | EmbeddingRateLimitError:
        """Generate strict domain typed errors corresponding from proxy rejections."""

        code = exc.code or 0

        if code in _AUTH_CODES:
            return EmbeddingAuthError(
                "Gemini authentication failed",
                provider="gemini",
                details={"error": str(exc)},
            )
        if code in _RATE_LIMIT_CODES:
            return EmbeddingRateLimitError(
                "Gemini rate limit exceeded",
                provider="gemini",
                details={"error": str(exc)},
            )
        return EmbeddingProviderError(
            f"Gemini client error: {exc.message}",
            provider="gemini",
            details={"error": str(exc)},
        )
