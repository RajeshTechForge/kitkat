"""Unit tests for OpenAIProvider._map_openai_error.

Tests every branch of the error mapping without making network calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)

from kitkat.providers.openai.provider import OpenAIProvider
from kitkat.core.exceptions import (
    LLMAuthenticationError,
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _status_exc(cls: type, status_code: int, message: str = "error") -> object:
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.headers = MagicMock()
    mock_response.headers.get = MagicMock(return_value=None)
    exc = cls.__new__(cls)
    exc.status_code = status_code
    exc.message = message
    exc.response = mock_response
    exc.body = {}
    exc.args = (message,)
    return exc


def _connection_exc() -> APIConnectionError:
    mock_req = MagicMock()
    exc = APIConnectionError.__new__(APIConnectionError)
    exc.request = mock_req
    exc.message = "connection refused"
    exc.args = ("connection refused",)
    return exc


def _timeout_exc() -> APITimeoutError:
    mock_req = MagicMock()
    exc = APITimeoutError.__new__(APITimeoutError)
    exc.request = mock_req
    exc.message = "request timed out"
    exc.args = ("request timed out",)
    return exc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMapOpenAIError:
    def test_authentication_error(self) -> None:
        exc = _status_exc(AuthenticationError, 401)
        result = OpenAIProvider._map_openai_error(exc)
        assert isinstance(result, LLMAuthenticationError)
        assert result.provider == "openai"

    def test_permission_denied_maps_to_auth_error(self) -> None:
        exc = _status_exc(PermissionDeniedError, 403)
        result = OpenAIProvider._map_openai_error(exc)
        assert isinstance(result, LLMAuthenticationError)
        assert result.provider == "openai"

    def test_timeout_error(self) -> None:
        exc = _timeout_exc()
        result = OpenAIProvider._map_openai_error(exc)
        assert isinstance(result, LLMTimeoutError)
        assert result.provider == "openai"

    def test_connection_error(self) -> None:
        exc = _connection_exc()
        result = OpenAIProvider._map_openai_error(exc)
        assert isinstance(result, LLMProviderError)
        assert result.provider == "openai"

    def test_not_found_error(self) -> None:
        exc = _status_exc(NotFoundError, 404)
        result = OpenAIProvider._map_openai_error(exc)
        assert isinstance(result, LLMProviderError)
        assert result.status_code == 404

    def test_bad_request_error(self) -> None:
        exc = _status_exc(BadRequestError, 400)
        result = OpenAIProvider._map_openai_error(exc)
        assert isinstance(result, LLMProviderError)
        assert result.status_code == 400

    def test_conflict_error(self) -> None:
        exc = _status_exc(ConflictError, 409)
        result = OpenAIProvider._map_openai_error(exc)
        assert isinstance(result, LLMProviderError)
        assert result.status_code == 409

    def test_unprocessable_entity_error(self) -> None:
        exc = _status_exc(UnprocessableEntityError, 422)
        result = OpenAIProvider._map_openai_error(exc)
        assert isinstance(result, LLMProviderError)
        assert result.status_code == 422

    def test_rate_limit_error(self) -> None:
        exc = _status_exc(RateLimitError, 429)
        exc.response.headers.get = MagicMock(return_value=None)
        result = OpenAIProvider._map_openai_error(exc)
        assert isinstance(result, LLMRateLimitError)
        assert result.provider == "openai"

    def test_internal_server_error(self) -> None:
        exc = _status_exc(InternalServerError, 500)
        result = OpenAIProvider._map_openai_error(exc)
        assert isinstance(result, LLMProviderError)
        assert result.status_code == 500

    def test_unknown_exception_falls_through(self) -> None:
        exc = RuntimeError("totally unexpected")
        result = OpenAIProvider._map_openai_error(exc)
        assert isinstance(result, LLMProviderError)
        assert result.provider == "openai"

    def test_all_results_are_llm_errors(self) -> None:
        test_cases = [
            _status_exc(AuthenticationError, 401),
            _status_exc(PermissionDeniedError, 403),
            _timeout_exc(),
            _connection_exc(),
            _status_exc(NotFoundError, 404),
            _status_exc(BadRequestError, 400),
            _status_exc(RateLimitError, 429),
            _status_exc(InternalServerError, 500),
            RuntimeError("unexpected"),
        ]
        for exc in test_cases:
            result = OpenAIProvider._map_openai_error(exc)
            assert isinstance(result, LLMError), f"Expected LLMError for {type(exc).__name__}"
