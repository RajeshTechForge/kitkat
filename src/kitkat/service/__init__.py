"""kitkat.service — service layer for managed, BYOK, and routed inference.

Canonical imports::

    from kitkat.service import LLMService, BYOKLLMService, create_llm_service
    from kitkat.service import LLMRouter, RouterConfig, RoutingStrategy
    from kitkat.service import LLMCache, CacheConfig, CacheBackendType
    from kitkat.service import create_llm_router
"""

from ..core.enums import CacheBackendType
from .byok import BYOKLLMService
from .cache import CacheConfig, LLMCache
from .factory import create_llm_router, create_llm_service
from .managed import LLMService
from .router import LLMRouter, RouterConfig

__all__ = [
    "BYOKLLMService",
    "CacheBackendType",
    "CacheConfig",
    "LLMCache",
    "LLMRouter",
    "LLMService",
    "RouterConfig",
    "create_llm_router",
    "create_llm_service",
]
