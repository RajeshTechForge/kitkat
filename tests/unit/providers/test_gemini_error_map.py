"""Unit tests for GeminiProvider._map_client_error.

Tests every branch of the error mapping without making network calls.
The ``google.genai.errors.ClientError`` exceptions are constructed via
their public constructor (``code`` + ``response_json`` dict).
"""

from __future__ import annotations

from google.genai import errors as genai_errors

from kitkat.core.exceptions import (
    LLMAuthenticationError,
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTokenLimitError,
)
from kitkat.providers.gemini.provider import GeminiConfig, GeminiProvider

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _provider() -> GeminiProvider:
    """Return a GeminiProvider with a dummy config (no real API calls)."""
    return GeminiProvider(GeminiConfig(api_key="test"))


def _client_error(
    code: int,
    message: str = "error",
    status: str = "INVALID_ARGUMENT",
) -> genai_errors.ClientError:
    """Build a minimal ClientError with the given code and message."""
    return genai_errors.ClientError(code, {"message": message, "status": status})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMapClientError:
    def test_authentication_error_401(self) -> None:
        exc = _client_error(401, "API key invalid")
        result = _provider()._map_client_error(exc)
        assert isinstance(result, LLMAuthenticationError)
        assert result.provider == "gemini"

    def test_authentication_error_403(self) -> None:
        exc = _client_error(403, "permission denied")
        result = _provider()._map_client_error(exc)
        assert isinstance(result, LLMAuthenticationError)
        assert result.provider == "gemini"

    def test_rate_limit_error(self) -> None:
        exc = _client_error(429, "rate limit exceeded")
        result = _provider()._map_client_error(exc)
        assert isinstance(result, LLMRateLimitError)
        assert result.provider == "gemini"

    def test_token_limit_error_token_in_message(self) -> None:
        exc = _client_error(400, "token limit exceeded")
        result = _provider()._map_client_error(exc)
        assert isinstance(result, LLMTokenLimitError)
        assert result.provider == "gemini"

    def test_token_limit_error_context_in_message(self) -> None:
        exc = _client_error(400, "context length exceeded")
        result = _provider()._map_client_error(exc)
        assert isinstance(result, LLMTokenLimitError)
        assert result.provider == "gemini"

    def test_bad_request_generic(self) -> None:
        exc = _client_error(400, "bad request")
        result = _provider()._map_client_error(exc)
        assert isinstance(result, LLMProviderError)
        assert result.status_code == 400
        assert result.provider == "gemini"

    def test_not_found_error(self) -> None:
        exc = _client_error(404, "model not found")
        result = _provider()._map_client_error(exc)
        assert isinstance(result, LLMProviderError)
        assert result.status_code == 404
        assert result.provider == "gemini"

    def test_code_zero_falls_through(self) -> None:
        exc = _client_error(0, "unexpected error")
        result = _provider()._map_client_error(exc)
        assert isinstance(result, LLMProviderError)
        assert result.status_code == 0
        assert result.provider == "gemini"

    def test_all_results_are_llm_errors(self) -> None:
        """Every mapped exception must be an LLMError subclass."""
        test_cases = [
            _client_error(401, "bad key"),
            _client_error(403, "no access"),
            _client_error(429, "too many"),
            _client_error(400, "token limit"),
            _client_error(400, "context window"),
            _client_error(400, "bad request"),
            _client_error(404, "not found"),
            _client_error(0, "weird"),
        ]
        for exc in test_cases:
            result = _provider()._map_client_error(exc)
            assert isinstance(result, LLMError), f"Expected LLMError for code {exc.code}"
