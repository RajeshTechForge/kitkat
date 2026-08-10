"""Unit tests for kitkat.plugins.loader and the plugin system interface."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from kitkat.abc import LLMProvider
from kitkat.core import (
    LLMRequest,
    LLMResponse,
    ProviderCapabilities,
    ProviderType,
)
from kitkat.plugins import (
    discover_plugins,
    get_provider_class,
    list_providers,
    register_provider,
)
from kitkat.providers._registry import _REGISTRY


class DummyTestProvider(LLMProvider):
    """Dummy provider for testing plugin loader functionality."""

    PROVIDER_TYPE = ProviderType.OPENAI
    DEFAULT_MODEL = "test-model"
    CAPABILITIES = ProviderCapabilities(provider_type=ProviderType.OPENAI)

    async def initialize(self) -> None:
        self._initialized = True

    async def shutdown(self) -> None:
        self._initialized = False

    async def _init_client_only(self) -> None:
        self._initialized = True

    async def complete(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError

    async def stream(self, request: LLMRequest):  # type: ignore[override]
        raise NotImplementedError
        yield  # type: ignore[unreachable]

    async def health_check(self) -> bool:
        return True

    def count_tokens(self, text: str) -> int:
        return len(text)


class TestPluginLoaderAPI:
    """Tests for public plugin loader functions."""

    def test_register_and_get_provider(self) -> None:
        """Registering a new provider name should make it retrievable."""
        provider_name = "_test_plugin_provider_123"
        try:
            register_provider(provider_name, DummyTestProvider)
            retrieved_cls = get_provider_class(provider_name)
            assert retrieved_cls is DummyTestProvider
            assert provider_name in list_providers()
        finally:
            _REGISTRY.pop(provider_name, None)

    def test_duplicate_registration_raises_value_error(self) -> None:
        """Registering the same provider name twice must raise ValueError."""
        provider_name = "_test_plugin_duplicate_xyz"
        register_provider(provider_name, DummyTestProvider)
        try:
            with pytest.raises(ValueError, match=provider_name):
                register_provider(provider_name, DummyTestProvider)
        finally:
            _REGISTRY.pop(provider_name, None)

    def test_get_unregistered_provider_raises_key_error(self) -> None:
        """Requesting an unregistered provider name should raise KeyError with suggestions."""
        unregistered = "_non_existent_provider_456"
        with pytest.raises(KeyError) as exc_info:
            get_provider_class(unregistered)
        assert unregistered in str(exc_info.value)
        assert "Available:" in str(exc_info.value)

    def test_list_providers_returns_sorted(self) -> None:
        """list_providers should return a sorted list of registered provider names."""
        providers = list_providers()
        assert isinstance(providers, list)
        assert providers == sorted(providers)

    def test_discover_plugins_success(self) -> None:
        """discover_plugins should locate entry points in the kitkat.providers group."""
        mock_ep = MagicMock()
        mock_ep.name = "_discovered_mock_provider"
        mock_ep.load.return_value = DummyTestProvider

        try:
            with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
                discover_plugins()
                assert "_discovered_mock_provider" in _REGISTRY
                assert get_provider_class("_discovered_mock_provider") is DummyTestProvider
        finally:
            _REGISTRY.pop("_discovered_mock_provider", None)

    def test_discover_plugins_handles_failed_loads_gracefully(self) -> None:
        """Broken plugin entry points raising load exceptions should log warnings and not fail."""
        broken_ep = MagicMock()
        broken_ep.name = "_broken_plugin"
        broken_ep.value = "invalid.module:Class"
        broken_ep.load.side_effect = RuntimeError("Plugin import failed")

        with (
            patch("importlib.metadata.entry_points", return_value=[broken_ep]),
            patch("kitkat.providers._registry.logger.warning") as mock_warning,
        ):
            discover_plugins()
            mock_warning.assert_called_once()
            assert "_broken_plugin" not in _REGISTRY

    def test_discover_plugins_handles_duplicate_entry_points_gracefully(self) -> None:
        """Duplicate entry points encountered during discovery should log a warning and skip."""
        mock_ep = MagicMock()
        mock_ep.name = "_dup_discovery_provider"
        mock_ep.load.return_value = DummyTestProvider

        try:
            register_provider("_dup_discovery_provider", DummyTestProvider)
            with (
                patch("importlib.metadata.entry_points", return_value=[mock_ep]),
                patch("kitkat.providers._registry.logger.warning") as mock_warning,
            ):
                discover_plugins()
                mock_warning.assert_called_once()
        finally:
            _REGISTRY.pop("_dup_discovery_provider", None)
