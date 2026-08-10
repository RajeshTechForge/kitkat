"""Public plugin API for library consumers and third-party provider authors.

To write and ship a custom provider package:

1. Subclass :class:`~kitkat.abc.provider.LLMProvider`:

   .. code-block:: python

       from kitkat.abc import LLMProvider

       class MyProvider(LLMProvider):
           ...

2. Register your provider in ``pyproject.toml``:

   .. code-block:: toml

       [project.entry-points."kitkat.providers"]
       my-provider = "mypkg.provider:MyProvider"

3. After installing your package (``pip install mypkg``), the provider is
   automatically discovered by kitkat:

   .. code-block:: python

       from kitkat.plugins import get_provider_class

       cls = get_provider_class("my-provider")
       provider = cls(...)
"""

from __future__ import annotations

from kitkat.providers._registry import (
    _discover,
    get_provider_class,
    list_providers,
    register_provider,
)


def discover_plugins() -> None:
    """Discover and register installed third-party provider plugins.

    Scans the ``kitkat.providers`` entry-point group in installed packages
    and registers valid provider classes into the global registry. Broken
    or duplicate plugins are logged as warnings without crashing discovery.

    This function is automatically called when :mod:`kitkat.providers` is first
    imported, but can be called manually to re-scan for dynamically installed
    plugins during runtime.
    """
    _discover()


__all__ = [
    "discover_plugins",
    "get_provider_class",
    "list_providers",
    "register_provider",
]
