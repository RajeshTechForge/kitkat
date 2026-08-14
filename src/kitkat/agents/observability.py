"""Observability configuration for the agent layer.

Call :func:`configure_observability` once at application startup before any
agents run. After that, every ``agent.run()`` / ``agent.run_stream()`` is
automatically traced with input, output, model, token usage, and latency.

Architecture:
    PydanticAI uses Logfire natively for instrumentation. We configure Logfire
    to capture traces. If Langfuse credentials are provided, we attach an
    OpenTelemetry OTLP exporter to the same TracerProvider to forward spans
    to Langfuse. This allows both tools to receive traces simultaneously without
    conflicts.
"""

from __future__ import annotations

from ._check import require_agents_extra, require_observability_extra

require_agents_extra()
require_observability_extra()

import base64
import logging
import os
from typing import Any

import logfire
from langfuse import Langfuse
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pydantic_ai import Agent as _Agent

logger = logging.getLogger(__name__)


def configure_observability(
    *,
    logfire_token: str | None = None,
    langfuse_public_key: str | None = None,
    langfuse_secret_key: str | None = None,
    langfuse_host: str = "https://cloud.langfuse.com",
    service_name: str = "kitkat",
    environment: str | None = None,
) -> None:
    """Wire up observability integrations for Logfire and Langfuse.

    Call once at application startup prior to executing agents.

    Args:
        logfire_token: Logfire project token. If omitted, Logfire will use
            the ``LOGFIRE_TOKEN`` environment variable if available.
        langfuse_public_key: Langfuse public key. Required for Langfuse OTel exporter.
        langfuse_secret_key: Langfuse secret key. Required for Langfuse OTel exporter.
        langfuse_host: Langfuse host URL (defaults to ``"https://cloud.langfuse.com"``).
        service_name: Service identifier tag in traces (defaults to ``"kitkat"``).
        environment: Deployment environment (e.g. ``"production"``, ``"staging"``).
            Defaults to ``ENVIRONMENT`` environment variable or ``"production"``.
    """
    env = environment or os.getenv("ENVIRONMENT", "production")

    config_kwargs: dict[str, Any] = {
        "service_name": service_name,
        "environment": env,
    }
    token = logfire_token or os.getenv("LOGFIRE_TOKEN")
    if token:
        config_kwargs["token"] = token

    logfire.configure(**config_kwargs)

    if _Agent is not None:
        _Agent.instrument_all()

    pk = langfuse_public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
    sk = langfuse_secret_key or os.getenv("LANGFUSE_SECRET_KEY")
    host = langfuse_host or os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if pk and sk and trace is not None:
        try:
            auth_str = f"{pk}:{sk}"
            auth_b64 = base64.b64encode(auth_str.encode("ascii")).decode("ascii")
            headers = {"Authorization": f"Basic {auth_b64}"}
            endpoint = f"{host.rstrip('/')}/api/public/otel/v1/traces"

            langfuse_exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers)
            provider = trace.get_tracer_provider()

            if isinstance(provider, SDKTracerProvider):
                provider.add_span_processor(BatchSpanProcessor(langfuse_exporter))
        except Exception as exc:
            logger.warning("Failed to configure Langfuse OpenTelemetry exporter: %s", exc)

    if Langfuse is not None and pk and sk:
        try:
            Langfuse(
                public_key=pk,
                secret_key=sk,
                host=host,
            )
        except Exception as exc:
            logger.warning("Failed to initialize Langfuse client: %s", exc)
