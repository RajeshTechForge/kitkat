"""
Deterministic fake embedding provider for testing the Sentinel RAG pipeline.

This module provides a deterministic, L2-normalised unit vector derived from the
SHA-256 hash of each input text. This means:
    - The same text always produces the same vector (determinism)
    - Different texts almost always produce different vectors (uniqueness)
    - Vectors have unit norm (cosine similarity behaves predictably in tests)
    - health_check() always returns True
    - initialize() and shutdown() are instant no-ops

Designed for
------------
  - Unit tests that need real vector shapes without API calls
  - Integration tests that need Qdrant to accept and return vectors
  - CI/CD pipelines where no API keys are available
  - Local development with docker-compose

Configuration keys
------------------
  dimensions   int    optional — vector length (default: 1536, matches OpenAI small)
  latency_ms   float  optional — artificial delay per embed() call (default: 0)
                                 Use in tests to simulate slow providers
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import time
from dataclasses import dataclass
from typing import Any

from ..abc.embedder import EmbeddingProvider, EmbeddingRequest, EmbeddingResult
from ..core.exceptions import EmbeddingProviderInitError
from ..core.models import RetryPolicy

logger = logging.getLogger(__name__)


_DEFAULT_DIMENSIONS = 1536


@dataclass
class FakeEmbeddingConfig:
    """
    Configuration for the fake embedding provider.

    Attributes:
        dimensions: Output vector length.
        latency_ms: Artificial delay in milliseconds per embed() call.
    """

    dimensions: int = _DEFAULT_DIMENSIONS
    latency_ms: float = 0.0

    def __post_init__(self) -> None:
        """Validate the configuration values after object instantiation."""

        if self.dimensions < 1:
            raise EmbeddingProviderInitError(
                f"FakeEmbeddingConfig.dimensions must be ≥ 1, got {self.dimensions}",
                provider="fake",
            )
        if self.latency_ms < 0:
            raise EmbeddingProviderInitError(
                f"FakeEmbeddingConfig.latency_ms must be ≥ 0, got {self.latency_ms}",
                provider="fake",
            )

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> FakeEmbeddingConfig:
        """Build the configuration from a raw dictionary."""

        return cls(
            dimensions=int(cfg.get("dimensions", _DEFAULT_DIMENSIONS)),
            latency_ms=float(cfg.get("latency_ms", 0.0)),
        )


class FakeEmbeddingProvider(EmbeddingProvider):
    """
    Deterministic fake embedding provider for tests and local development.

    Produces a unique, normalised vector for each distinct input text using
    SHA-256 as a pseudo-random number generator seeded by the text content.
    The output is a unit vector so cosine similarity arithmetic behaves
    sensibly in integration tests.
    """

    PROVIDER_NAME = "fake"
    DEFAULT_MODEL = "fake-embedding-model"
    RETRY_POLICY = RetryPolicy(
        max_attempts=1,  # no retries needed — no network
        base_delay_s=0.0,
        max_delay_s=0.0,
        exponential_base=1.0,
        jitter=False,
    )

    def __init__(self, config: FakeEmbeddingConfig | dict[str, Any] | None = None) -> None:
        if config is None:
            config = FakeEmbeddingConfig()
        elif isinstance(config, dict):
            config = FakeEmbeddingConfig.from_dict(config)
        super().__init__(config.__dict__)
        self._cfg: FakeEmbeddingConfig = config

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Initialize the provider."""
        self._initialized = True
        logger.debug(
            "FakeEmbeddingProvider initialised (dimensions=%d, latency_ms=%.1f).",
            self._cfg.dimensions,
            self._cfg.latency_ms,
        )

    async def shutdown(self) -> None:
        """Shut down the provider."""
        self._initialized = False
        logger.debug("FakeEmbeddingProvider shut down.")

    # ------------------------------------------------------------------
    # Core embedding
    # ------------------------------------------------------------------

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        """
        Return a deterministic unit vector for each input text.

        Args:
            request: The embedding request containing a list of texts to embed.

        Returns:
            The embedding result containing the vectors and elapsed time.
        """
        self._assert_initialized()

        start = time.monotonic()

        if self._cfg.latency_ms > 0:
            await asyncio.sleep(self._cfg.latency_ms / 1_000)

        vectors = [_text_to_unit_vector(text, self._cfg.dimensions) for text in request.texts]

        return EmbeddingResult(
            vectors=vectors,
            model=self.DEFAULT_MODEL,
            provider=self.PROVIDER_NAME,
            prompt_tokens=sum(len(t.split()) for t in request.texts),
            latency_ms=self._elapsed_ms(start),
        )

    # ------------------------------------------------------------------
    # Health & introspection
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        """Perform a health check on the provider."""
        return True

    @property
    def dimensions(self) -> int:
        """Return the configured vector length."""

        return self._cfg.dimensions


def _text_to_unit_vector(text: str, dimensions: int) -> list[float]:
    """
    Produce a deterministic, L2-normalised float vector from input text.

    Args:
        text: The text to be converted into a unit vector.
        dimensions: The desired length of the resulting vector.

    Returns:
        A list of floats representing the normalised vector.
    """
    seed = text.encode("utf-8")
    raw_bytes = bytearray()
    counter = 0

    needed = dimensions * 4
    while len(raw_bytes) < needed:
        digest = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
        raw_bytes.extend(digest)
        counter += 1

    floats: list[float] = []
    for i in range(dimensions):
        chunk = raw_bytes[i * 4 : i * 4 + 4]
        int_val = int.from_bytes(chunk, "big", signed=True)
        floats.append(int_val / (2**31))  # Normalise to [-1.0, 1.0]

    magnitude = math.sqrt(sum(x * x for x in floats))
    if magnitude > 0:
        floats = [x / magnitude for x in floats]

    return floats
