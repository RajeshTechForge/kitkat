"""Entry-point-based auto-discovery for RAG embedding providers."""

from __future__ import annotations

import logging
from importlib.metadata import entry_points

from kitkat.rag.abc.embedder import EmbeddingProvider

logger = logging.getLogger(__name__)


_EMBEDDER_REGISTRY: dict[str, type[EmbeddingProvider]] = {}


def register_embedding_provider(name: str, provider_cls: type[EmbeddingProvider]) -> None:
    """Register an embedding provider programmatically.

    Args:
        name: The name to register the provider under.
        provider_cls: The provider class (must inherit from EmbeddingProvider).
    """
    if not issubclass(provider_cls, EmbeddingProvider):
        raise TypeError(f"{provider_cls.__name__} must inherit from EmbeddingProvider")
    _EMBEDDER_REGISTRY[name] = provider_cls
    logger.debug("Registered embedding provider: %s -> %s", name, provider_cls.__name__)


def get_embedding_provider_class(name: str) -> type[EmbeddingProvider]:
    """Retrieve a registered embedding provider class by name.

    Args:
        name: The registered name of the provider.

    Returns:
        The provider class.

    Raises:
        KeyError: If the provider is not registered.
    """
    if name not in _EMBEDDER_REGISTRY:
        # Trigger lazy load if not found
        _load_entry_points()

    if name not in _EMBEDDER_REGISTRY:
        raise KeyError(f"Embedding provider '{name}' not found. Is the package installed?")

    return _EMBEDDER_REGISTRY[name]


def _load_entry_points() -> None:
    """Load all providers registered via the 'kitkat.rag.embedders' entry_points group."""
    try:
        eps = entry_points(group="kitkat.rag.embedders")
    except TypeError:
        # Python 3.9 compatibility (though we require 3.11+)
        eps = entry_points().get("kitkat.rag.embedders", [])

    for ep in eps:
        try:
            provider_cls = ep.load()
            register_embedding_provider(ep.name, provider_cls)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load embedding provider '%s': %s", ep.name, exc)


# Load built-in and installed providers on module import
_load_entry_points()
