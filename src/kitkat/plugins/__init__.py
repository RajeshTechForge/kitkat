"""Plugin discovery and registration surface for kitkat providers."""

from __future__ import annotations

from kitkat.plugins.loader import (
    discover_plugins,
    get_provider_class,
    list_providers,
    register_provider,
)

__all__ = [
    "discover_plugins",
    "get_provider_class",
    "list_providers",
    "register_provider",
]
