"""Shared httpx async-client configuration.

All providers that use httpx directly should build their clients through
:func:`build_async_client` so connection-pool settings, timeout defaults,
and the ``User-Agent`` header are consistent across the library.

Provider SDKs (anthropic, openai, google-genai) manage their own HTTP
transport internally, so this factory is primarily useful for custom
providers or future internal HTTP callers.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import httpx

try:
    _LIB_VERSION = version("kitkat")
except PackageNotFoundError:
    _LIB_VERSION = "dev"

_USER_AGENT = f"kitkat/{_LIB_VERSION} httpx/{httpx.__version__}"


def build_async_client(
    base_url: str = "",
    timeout: float = 120.0,
    **kwargs: object,
) -> httpx.AsyncClient:
    """Create a pre-configured :class:`httpx.AsyncClient`.

    Args:
        base_url: Optional base URL prefix applied to all requests.
        timeout: Default request timeout in seconds.
        **kwargs: Additional keyword arguments forwarded to
            :class:`httpx.AsyncClient`.

    Returns:
        A ready-to-use async HTTP client with library defaults applied.
    """
    return httpx.AsyncClient(
        base_url=base_url,
        timeout=httpx.Timeout(timeout),
        headers={"User-Agent": _USER_AGENT},
        **kwargs,  # type: ignore[arg-type]
    )
