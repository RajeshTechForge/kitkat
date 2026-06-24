"""Convenience factories for common LLMService configurations.

Provides :func:`create_llm_service` as the recommended entry point for
constructing a fully configured service from a dict of provider instances,
avoiding the boilerplate of repeated :meth:`~.managed.LLMService.register_provider`
calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..abc.provider import LLMProvider
    from ..core.enums import ProviderType

from .managed import LLMService


def create_llm_service(
    providers: dict[ProviderType, LLMProvider],
) -> LLMService:
    """Create a fully configured :class:`LLMService` from a provider mapping.

    The returned service still requires :meth:`~.managed.LLMService.initialize`
    to be called before serving requests, giving the caller control over when
    SDK credentials are probed and connection pools are opened.

    Args:
        providers: Mapping of :class:`~kitkat.core.enums.ProviderType` to
            concrete :class:`~kitkat.abc.provider.LLMProvider` instances.
            Providers need not be pre-initialized; ``initialize()`` will
            call each provider's ``initialize()`` method.

    Returns:
        An :class:`LLMService` with all providers registered.

    Example::

        import os
        from kitkat import create_llm_service, ProviderType
        from kitkat.providers.anthropic import AnthropicProvider, AnthropicConfig

        service = create_llm_service({
            ProviderType.ANTHROPIC: AnthropicProvider(
                AnthropicConfig(api_key=os.environ["ANTHROPIC_API_KEY"])
            )
        })
        await service.initialize()
    """
    service = LLMService()
    for provider_type, provider in providers.items():
        service.register_provider(provider_type, provider)
    return service
