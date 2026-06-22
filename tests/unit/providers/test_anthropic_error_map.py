"""Unit tests for AnthropicProvider._map_anthropic_error.

Tests every branch of the error mapping without making network calls.
The Anthropic SDK exceptions are constructed via their public constructors
where possible, or via MagicMock otherwise.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import anthropic

from kitkat.core.exceptions import (
    LLMAuthenticationError,
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from kitkat.providers.anthropic.provider import AnthropicProvider

# ---------------------------------------------------------------------------
# Helpers to build minimal SDK exceptions without real HTTP responses
# ---------------------------------------------------------------------------


def _api_status_exc(cls: type, status_code: int, message: str = "err") -> object:
    """Build an Anthropic APIStatusError subclass with a mocked response."""
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.headers = {}
    exc = cls.__new__(cls)
    exc.status_code = status_code
    exc.message = message
    exc.response = mock_response
    exc.body = {}
    return exc


def _connection_exc() -> anthropic.APIConnectionError:
    mock_req = MagicMock()
    exc = anthropic.APIConnectionError.__new__(anthropic.APIConnectionError)
    exc.request = mock_req
    exc.message = "connection failed"
    return exc


def _timeout_exc() -> anthropic.APITimeoutError:
    mock_req = MagicMock()
    exc = anthropic.APITimeoutError.__new__(anthropic.APITimeoutError)
    exc.request = mock_req
    exc.message = "timeout"
    return exc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMapAnthropicError:
    def test_authentication_error(self) -> None:
        exc = _api_status_exc(anthropic.AuthenticationError, 401)
        result = AnthropicProvider._map_anthropic_error(exc)
        assert isinstance(result, LLMAuthenticationError)
        assert result.provider == "anthropic"

    def test_permission_denied_maps_to_auth_error(self) -> None:
        exc = _api_status_exc(anthropic.PermissionDeniedError, 403)
        result = AnthropicProvider._map_anthropic_error(exc)
        assert isinstance(result, LLMAuthenticationError)
        assert result.provider == "anthropic"

    def test_timeout_error(self) -> None:
        exc = _timeout_exc()
        result = AnthropicProvider._map_anthropic_error(exc)
        assert isinstance(result, LLMTimeoutError)
        assert result.provider == "anthropic"

    def test_connection_error(self) -> None:
        exc = _connection_exc()
        result = AnthropicProvider._map_anthropic_error(exc)
        assert isinstance(result, LLMProviderError)
        assert result.provider == "anthropic"

    def test_not_found_error(self) -> None:
        exc = _api_status_exc(anthropic.NotFoundError, 404)
        result = AnthropicProvider._map_anthropic_error(exc)
        assert isinstance(result, LLMProviderError)
        assert result.status_code == 404

    def test_bad_request_error(self) -> None:
        exc = _api_status_exc(anthropic.BadRequestError, 400)
        result = AnthropicProvider._map_anthropic_error(exc)
        assert isinstance(result, LLMProviderError)
        assert result.status_code == 400

    def test_unprocessable_entity_error(self) -> None:
        exc = _api_status_exc(anthropic.UnprocessableEntityError, 422)
        result = AnthropicProvider._map_anthropic_error(exc)
        assert isinstance(result, LLMProviderError)
        assert result.status_code == 422

    def test_rate_limit_error_without_retry_after(self) -> None:
        exc = _api_status_exc(anthropic.RateLimitError, 429)
        exc.response = MagicMock()
        exc.response.headers = {}
        result = AnthropicProvider._map_anthropic_error(exc)
        assert isinstance(result, LLMRateLimitError)
        assert result.retry_after_s is None
        assert result.provider == "anthropic"

    def test_rate_limit_error_with_retry_after_header(self) -> None:
        exc = _api_status_exc(anthropic.RateLimitError, 429)
        exc.response = MagicMock()
        exc.response.headers = {"retry-after": "30"}
        result = AnthropicProvider._map_anthropic_error(exc)
        assert isinstance(result, LLMRateLimitError)
        assert result.retry_after_s == 30.0

    def test_internal_server_error_generic(self) -> None:
        exc = _api_status_exc(anthropic.InternalServerError, 500)
        result = AnthropicProvider._map_anthropic_error(exc)
        assert isinstance(result, LLMProviderError)
        assert result.status_code == 500
        assert result.provider == "anthropic"

    def test_api_status_error_billing(self) -> None:
        exc = _api_status_exc(anthropic.APIStatusError, 402)
        result = AnthropicProvider._map_anthropic_error(exc)
        assert isinstance(result, LLMProviderError)
        assert "billing" in result.message.lower()

    def test_unknown_exception_falls_through(self) -> None:
        exc = RuntimeError("totally unexpected")
        result = AnthropicProvider._map_anthropic_error(exc)
        assert isinstance(result, LLMProviderError)
        assert result.provider == "anthropic"

    def test_all_results_are_llm_errors(self) -> None:
        """Every mapped exception must be an LLMError subclass."""
        test_cases = [
            _api_status_exc(anthropic.AuthenticationError, 401),
            _api_status_exc(anthropic.PermissionDeniedError, 403),
            _timeout_exc(),
            _connection_exc(),
            _api_status_exc(anthropic.NotFoundError, 404),
            _api_status_exc(anthropic.BadRequestError, 400),
            _api_status_exc(anthropic.RateLimitError, 429),
            _api_status_exc(anthropic.InternalServerError, 500),
            RuntimeError("unexpected"),
        ]
        for exc in test_cases:
            result = AnthropicProvider._map_anthropic_error(exc)
            assert isinstance(result, LLMError), f"Expected LLMError for {type(exc).__name__}"
