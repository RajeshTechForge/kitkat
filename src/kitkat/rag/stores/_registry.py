"""Registry for custom vector stores."""

from __future__ import annotations

import logging
from importlib.metadata import entry_points

from kitkat.rag.abc.vector_store import VectorStore

logger = logging.getLogger(__name__)

_STORE_REGISTRY: dict[str, type[VectorStore]] = {}


def register_vector_store(name: str, store_cls: type[VectorStore]) -> None:
    """Register a vector store programmatically."""
    if not issubclass(store_cls, VectorStore):
        raise TypeError(f"{store_cls.__name__} must inherit from VectorStore")
    _STORE_REGISTRY[name] = store_cls


def get_vector_store_class(name: str) -> type[VectorStore]:
    """Retrieve a registered vector store class by name."""
    if name not in _STORE_REGISTRY:
        _load_entry_points()
    if name not in _STORE_REGISTRY:
        raise KeyError(f"Vector store '{name}' not found.")
    return _STORE_REGISTRY[name]


def _load_entry_points() -> None:
    """Load all stores registered via the 'kitkat.rag.stores' entry_points group."""
    try:
        eps = entry_points(group="kitkat.rag.stores")
    except TypeError:
        eps = entry_points().get("kitkat.rag.stores", [])

    for ep in eps:
        try:
            store_cls = ep.load()
            register_vector_store(ep.name, store_cls)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load vector store '%s': %s", ep.name, exc)


_load_entry_points()
