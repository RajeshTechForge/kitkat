"""Unit tests for kitkat.agents.observability."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from kitkat.agents.observability import configure_observability


class TestConfigureObservability:
    """Test suite for configure_observability function."""

    def test_configure_observability_logfire_defaults(self) -> None:
        """Logfire should be configured with default service_name and environment."""
        with (
            patch("kitkat.agents.observability.logfire") as mock_logfire,
            patch("kitkat.agents.observability._Agent") as mock_agent,
        ):
            configure_observability(service_name="my_service", environment="test")

            mock_logfire.configure.assert_called_once_with(
                service_name="my_service",
                environment="test",
            )
            mock_agent.instrument_all.assert_called_once()

    def test_configure_observability_with_logfire_token(self) -> None:
        """Token parameter should be forwarded to logfire.configure when provided."""
        with (
            patch("kitkat.agents.observability.logfire") as mock_logfire,
            patch("kitkat.agents.observability._Agent"),
        ):
            configure_observability(
                logfire_token="lf_secret_token",
                service_name="kitkat",
            )

            mock_logfire.configure.assert_called_once_with(
                service_name="kitkat",
                environment="production",
                token="lf_secret_token",
            )

    def test_configure_observability_with_langfuse(self) -> None:
        """Langfuse OTel exporter and client should be configured when keys are supplied."""
        mock_provider = MagicMock()

        with (
            patch("kitkat.agents.observability.logfire"),
            patch("kitkat.agents.observability._Agent"),
            patch("kitkat.agents.observability.trace") as mock_trace,
            patch("kitkat.agents.observability.OTLPSpanExporter") as mock_exporter,
            patch("kitkat.agents.observability.BatchSpanProcessor") as mock_processor,
            patch("kitkat.agents.observability.Langfuse") as mock_langfuse_cls,
        ):
            mock_trace.get_tracer_provider.return_value = mock_provider

            configure_observability(
                langfuse_public_key="pk-lf-123",
                langfuse_secret_key="sk-lf-456",
                langfuse_host="https://langfuse.mycompany.com",
            )

            mock_exporter.assert_called_once_with(
                endpoint="https://langfuse.mycompany.com/api/public/otel/v1/traces",
                headers={"Authorization": "Basic cGstbGYtMTIzOnNrLWxmLTQ1Ng=="},
            )
            mock_processor.assert_called_once_with(mock_exporter.return_value)
            mock_provider.add_span_processor.assert_called_once_with(mock_processor.return_value)
            mock_langfuse_cls.assert_called_once_with(
                public_key="pk-lf-123",
                secret_key="sk-lf-456",
                host="https://langfuse.mycompany.com",
            )

    def test_configure_observability_otel_exception_handled_gracefully(self) -> None:
        """Exceptions during OTel exporter setup should be caught and logged without raising."""
        with (
            patch("kitkat.agents.observability.logfire"),
            patch("kitkat.agents.observability._Agent"),
            patch("kitkat.agents.observability.trace") as mock_trace,
            patch("kitkat.agents.observability.logger") as mock_logger,
        ):
            mock_trace.get_tracer_provider.side_effect = RuntimeError("OTel provider failed")

            configure_observability(
                langfuse_public_key="pk-lf-123",
                langfuse_secret_key="sk-lf-456",
            )

            mock_logger.warning.assert_called()
