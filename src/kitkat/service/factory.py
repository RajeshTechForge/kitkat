"""Convenience factories for common service configurations.

Provides :func:`create_llm_service` and :func:`create_llm_router` as the
recommended entry points for constructing a service or router from provider
instances, avoiding boilerplate registration calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..abc.provider import LLMProvider
    from ..core.enums import CacheBackendType, ProviderType, RoutingStrategy
    from .router import LLMRouter

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


def create_llm_router(
    providers: list[LLMProvider],
    strategy: RoutingStrategy | None = None,
    enable_cache: bool = True,
    cache_backend: CacheBackendType | None = None,
    redis_url: str = "redis://localhost:6379/0",
) -> LLMRouter:
    """Create a configured :class:`~kitkat.service.router.LLMRouter` from a provider list.

    This factory covers the two most common configurations — failover with
    in-memory caching, and failover with Redis caching.  For full lifecycle
    control (e.g. parallel initialization, custom circuit-breaker thresholds)
    call :meth:`~kitkat.service.router.LLMRouter.build` directly.

    Args:
        providers: Un-initialized :class:`~kitkat.abc.provider.LLMProvider`
            instances.  The router does **not** call ``initialize()`` — the
            caller must either call ``await router.build(providers, config)``
            or manage lifecycle explicitly.
        strategy: Provider-selection strategy.  Defaults to
            :attr:`~kitkat.core.enums.RoutingStrategy.FAILOVER`.
        enable_cache: Whether to attach an :class:`~kitkat.service.cache.LLMCache`
            to the router.  Defaults to ``True``.
        cache_backend: Cache storage backend.  Defaults to
            :attr:`~kitkat.core.enums.CacheBackendType.MEMORY`.
        redis_url: Redis connection URL used when *cache_backend* is
            :attr:`~kitkat.core.enums.CacheBackendType.REDIS`.
            Ignored for other backends.

    Returns:
        An :class:`~kitkat.service.router.LLMRouter` ready for use.
        Call ``await router.build(providers, config)`` for concurrent
        provider initialization.

    Example::

        import os
        from kitkat import create_llm_router, RoutingStrategy
        from kitkat.providers.anthropic import AnthropicProvider, AnthropicConfig
        from kitkat.providers.openai import OpenAIProvider, OpenAIConfig

        router = await LLMRouter.build([
            AnthropicProvider(AnthropicConfig(api_key=os.environ["ANTHROPIC_API_KEY"])),
            OpenAIProvider(OpenAIConfig(api_key=os.environ["OPENAI_API_KEY"])),
        ])
        # or use the factory for default config:
        router = create_llm_router(
            providers=[anthropic_provider, openai_provider],
            strategy=RoutingStrategy.FAILOVER,
        )
    """
    from ..core.enums import CacheBackendType as _CBT
    from ..core.enums import RoutingStrategy as _RS
    from .cache import CacheConfig
    from .router import LLMRouter, RouterConfig

    _strategy = strategy if strategy is not None else _RS.FAILOVER
    _backend = cache_backend if cache_backend is not None else _CBT.MEMORY

    config = RouterConfig(
        strategy=_strategy,
        enable_cache=enable_cache,
        cache=CacheConfig(backend=_backend, redis_url=redis_url),
    )
    return LLMRouter(providers, config)
