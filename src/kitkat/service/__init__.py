"""kitkat.service — service layer for managed and BYOK inference.

Canonical imports::

    from kitkat.service import LLMService, BYOKLLMService, create_llm_service
"""

from .byok import BYOKLLMService
from .factory import create_llm_service
from .managed import LLMService

__all__ = ["BYOKLLMService", "LLMService", "create_llm_service"]
