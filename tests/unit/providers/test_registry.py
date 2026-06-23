"""Unit tests for kitkat.providers._registry.

Tests manual registration, duplicate prevention, missing-provider errors,
and the list_providers helper.  The auto-discovery (_discover) is exercised
indirectly through the registered built-in providers.
"""

from __future__ import annotations

import pytest

from kitkat.abc import LLMProvider
from kitkat.providers._registry import (
    _REGISTRY,
    get_provider_class,
    list_providers,
    register_provider,
)


class TestManualRegistration:
    def test_get_registered_provider(self) -> None:
        """Built-in providers registered via entry-points should be discoverable."""
        # At least one built-in provider must be registered if the package is
        # installed in editable mode with all extras.
        # This test is skipped gracefully if no providers are installed.
        if not _REGISTRY:
            pytest.skip("No providers installed — install kitkat[all-providers]")
        name = next(iter(_REGISTRY))
        cls = get_provider_class(name)
        assert issubclass(cls, LLMProvider)

    def test_get_unknown_provider_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="no_such_provider"):
            get_provider_class("no_such_provider")

    def test_error_message_lists_available_providers(self) -> None:
        try:
            get_provider_class("no_such_provider")
        except KeyError as exc:
            assert "Available:" in str(exc)

    def test_duplicate_registration_raises_value_error(self) -> None:
        """Registering the same name twice must raise ValueError."""
        from kitkat.abc import LLMProvider
        from kitkat.core.models import LLMRequest, LLMResponse, ProviderCapabilities, ProviderType

        class _DummyA(LLMProvider):
            PROVIDER_TYPE = ProviderType.ANTHROPIC
            DEFAULT_MODEL = "dummy"
            CAPABILITIES = ProviderCapabilities(provider_type=ProviderType.ANTHROPIC)

            async def initialize(self) -> None: ...
            async def shutdown(self) -> None: ...
            async def _init_client_only(self) -> None: ...
            async def complete(self, request: LLMRequest) -> LLMResponse: ...  # type: ignore[override,empty-body]
            async def stream(self, request: LLMRequest):  # type: ignore[override
                return
                yield  # make it an async generator

            async def health_check(self) -> bool:
                return True

            def count_tokens(self, text: str) -> int:
                return 1

        class _DummyB(_DummyA):
            pass

        unique_name = "_test_dup_provider_xyz"
        register_provider(unique_name, _DummyA)
        try:
            with pytest.raises(ValueError, match=unique_name):
                register_provider(unique_name, _DummyB)
        finally:
            # Clean up the test entry so it doesn't affect other tests.
            _REGISTRY.pop(unique_name, None)


class TestListProviders:
    def test_returns_sorted_list(self) -> None:
        names = list_providers()
        assert names == sorted(names)

    def test_returns_list_type(self) -> None:
        assert isinstance(list_providers(), list)
