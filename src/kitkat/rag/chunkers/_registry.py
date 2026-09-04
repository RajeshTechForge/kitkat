"""Registry for custom chunkers."""

from __future__ import annotations

from kitkat.rag.abc.chunker import Chunker

_CHUNKER_REGISTRY: dict[str, type[Chunker]] = {}


def register_chunker(name: str, chunker_cls: type[Chunker]) -> None:
    """Register a custom chunker programmatically."""
    if not issubclass(chunker_cls, Chunker):
        raise TypeError(f"{chunker_cls.__name__} must inherit from Chunker")
    _CHUNKER_REGISTRY[name] = chunker_cls


def get_chunker_class(name: str) -> type[Chunker]:
    """Retrieve a registered chunker class by name."""
    if name not in _CHUNKER_REGISTRY:
        raise KeyError(f"Chunker '{name}' not found.")
    return _CHUNKER_REGISTRY[name]
