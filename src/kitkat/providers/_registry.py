"""Entry-point–based provider registry.

Built-in providers are declared in ``pyproject.toml``::

    [project.entry-points."kitkat.providers"]
    anthropic = "kitkat.providers.anthropic:AnthropicProvider"
    openai    = "kitkat.providers.openai:OpenAIProvider"
    gemini    = "kitkat.providers.gemini:GeminiProvider"

Third-party packages ship custom providers using the same mechanism::

    # In their pyproject.toml:
    [project.entry-points."kitkat.providers"]
    my-llm = "mypkg.provider:MyLLMProvider"

The registry is populated automatically by :func:`_discover` at import time,
using the same plugin mechanism as pytest.  Providers are loaded lazily —
only when their entry point is first requested — so installing a provider
extra does not unconditionally import its SDK.

Thread-safety: ``_REGISTRY`` is written once at import time by ``_discover``
and is read-only thereafter.  Calling :func:`register_provider` after import
is supported but callers are responsible for avoiding concurrent writes.
"""

from __future__ import annotations

import importlib.metadata
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kitkat.abc.provider import LLMProvider

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, type[LLMProvider]] = {}


def register_provider(name: str, cls: type[LLMProvider]) -> None:
    """Register a provider class under a canonical name.

    Args:
        name: The identifier used to look up this provider (e.g. ``"anthropic"``).
        cls: The concrete :class:`~kitkat.abc.provider.LLMProvider` subclass.

    Raises:
        ValueError: If *name* is already present in the registry.  Duplicate
            registration is treated as an error because silent overwriting
            would mask misconfigured entry-points in dependent packages.
    """
    if name in _REGISTRY:
        raise ValueError(
            f"Provider {name!r} is already registered "
            f"(existing: {_REGISTRY[name].__qualname__!r}). "
            "Each provider name must be unique across all installed packages."
        )
    _REGISTRY[name] = cls
    logger.debug("Registered provider %r → %s.", name, cls.__qualname__)


def get_provider_class(name: str) -> type[LLMProvider]:
    """Return the provider class registered under *name*.

    Args:
        name: The canonical provider identifier (e.g. ``"anthropic"``).

    Returns:
        The registered :class:`~kitkat.abc.provider.LLMProvider` subclass.

    Raises:
        KeyError: If no provider is registered under *name*.  The error
            message lists all currently registered providers and suggests
            installing the missing extra.
    """
    if name not in _REGISTRY:
        available = sorted(_REGISTRY.keys())
        raise KeyError(
            f"No provider registered for {name!r}. "
            f"Available: {available or ['(none)']}.  "
            f"Install the provider extra (e.g. 'pip install kitkat[{name}]') "
            "or call register_provider() before using this function."
        )
    return _REGISTRY[name]


def list_providers() -> list[str]:
    """Return a sorted list of all registered provider names.

    Returns:
        Sorted list of registered provider name strings.
    """
    return sorted(_REGISTRY.keys())


def _discover() -> None:
    """Auto-discover and register providers from installed entry-points.

    Iterates the ``kitkat.providers`` entry-point group.  Each entry-point
    that can be loaded and passes the :class:`~kitkat.abc.provider.LLMProvider`
    subclass check is registered.  Failed loads are logged at WARNING level
    and skipped — a broken third-party plugin must not prevent the rest of
    the library from functioning.
    """
    for ep in importlib.metadata.entry_points(group="kitkat.providers"):
        try:
            cls = ep.load()
        except Exception as exc:
            logger.warning(
                "Failed to load provider plugin %r from %r: %s",
                ep.name,
                ep.value,
                exc,
            )
            continue

        try:
            register_provider(ep.name, cls)
        except ValueError as exc:
            logger.warning("Skipping duplicate provider entry-point %r: %s", ep.name, exc)


_discover()
