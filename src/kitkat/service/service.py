"""Backward-compatibility shim for kitkat.service.service.

.. deprecated::
    Import from the canonical locations instead::

        from kitkat.service import LLMService, create_llm_service
        # or
        from kitkat.service.managed import LLMService
        from kitkat.service.factory import create_llm_service

    This module is preserved for v0.1.x consumers and will be removed in v1.0.
"""

from .factory import create_llm_service  # noqa: F401
from .managed import LLMService  # noqa: F401

__all__ = ["LLMService", "create_llm_service"]
