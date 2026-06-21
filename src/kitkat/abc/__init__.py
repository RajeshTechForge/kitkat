"""kitkat.abc — abstract base classes for the library.

The only stable public export is :class:`~kitkat.abc.provider.LLMProvider`.
Third-party providers should import from here::

    from kitkat.abc import LLMProvider
"""

from .provider import LLMProvider

__all__ = ["LLMProvider"]
