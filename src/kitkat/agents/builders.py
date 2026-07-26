"""Public agent factory functions.

These are what most library consumers call directly.  They create configured
PydanticAI agents bound to a model adapter and a context type.  The pattern
for both managed and BYOK paths is identical from the caller's perspective;
the adapter handles the routing difference.

Usage::

    from kitkat.agents.builders import build_chat_agent
    from kitkat.agents.adapters.managed import ManagedModelAdapter
    from kitkat.core.enums import ProviderType

    adapter = ManagedModelAdapter(service=llm_service, provider_type=ProviderType.ANTHROPIC)
    agent = build_chat_agent(model=adapter, context_type=UserContext)
    result = await agent.run("Hello!", deps=user_ctx)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

try:
    from pydantic_ai import Agent
except ImportError as exc:
    raise ImportError(
        "Agent builders require the 'agents' extra. Install with: pip install kitkat[agents]"
    ) from exc

if TYPE_CHECKING:
    from pydantic import BaseModel
    from pydantic_ai.models import Model

from .context import BaseAgentContext

ContextT = TypeVar("ContextT", bound=BaseAgentContext)

_DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant. User locale: {locale}."


def build_chat_agent(
    model: Model,
    context_type: type[ContextT] = BaseAgentContext,  # type: ignore[assignment]
    system_prompt: str = "",
) -> Agent[ContextT, str]:
    """Build a chat agent that returns plain strings.

    Creates a PydanticAI ``Agent`` with a dynamic system prompt derived from
    :class:`~kitkat.agents.context.BaseAgentContext` when no static override
    is provided.  The dynamic prompt injects the user's ``locale`` and respects
    ``system_prompt_override`` for per-user or per-tenant customisation.

    Args:
        model: A :class:`~kitkat.agents.adapters.managed.ManagedModelAdapter` or
            :class:`~kitkat.agents.adapters.byok.BYOKModelAdapter` instance.
        context_type: The ``deps_type`` for the agent.  Defaults to
            :class:`~kitkat.agents.context.BaseAgentContext`.  Pass your
            application's ``UserContext`` subclass here.
        system_prompt: Static system prompt.  When non-empty, used verbatim and
            the dynamic prompt is skipped.  When empty, a dynamic prompt is
            registered via ``@agent.system_prompt``.

    Returns:
        A configured ``Agent[context_type, str]`` ready for ``.run()``.

    Example (managed path)::

        agent = build_chat_agent(
            model=ManagedModelAdapter(service, ProviderType.ANTHROPIC),
            context_type=UserContext,
        )
        result = await agent.run("Hello!", deps=user_ctx)

    Example (BYOK path)::

        async with BYOKLLMService(ProviderType.OPENAI, key, model) as byok:
            agent = build_chat_agent(model=BYOKModelAdapter(byok))
            result = await agent.run("Hello!", deps=user_ctx)
    """
    from pydantic_ai import RunContext

    agent: Agent[ContextT, str] = Agent(
        model=model,
        deps_type=context_type,
        output_type=str,
        system_prompt=system_prompt or "You are a helpful AI assistant.",
    )

    if not system_prompt:

        @agent.system_prompt
        def _dynamic_prompt(ctx: RunContext[ContextT]) -> str:
            if ctx.deps.system_prompt_override:
                return ctx.deps.system_prompt_override
            return _DEFAULT_SYSTEM_PROMPT.format(locale=ctx.deps.locale)

    return agent


def build_structured_agent(
    model: Model,
    result_type: type[BaseModel],
    context_type: type[ContextT] = BaseAgentContext,  # type: ignore[assignment]
    system_prompt: str = "",
) -> Agent[ContextT, BaseModel]:
    """Build an agent that returns a validated Pydantic model.

    PydanticAI validates the LLM output against ``result_type`` automatically,
    retrying up to the agent's ``max_retries`` when the model produces malformed
    JSON.

    Args:
        model: A :class:`~kitkat.agents.adapters.managed.ManagedModelAdapter` or
            :class:`~kitkat.agents.adapters.byok.BYOKModelAdapter` instance.
        result_type: A :class:`~pydantic.BaseModel` subclass that the LLM output
            will be validated against.
        context_type: The ``deps_type`` for the agent.  Defaults to
            :class:`~kitkat.agents.context.BaseAgentContext`.
        system_prompt: Optional static system prompt.  When empty, a default
            prompt instructing JSON output is used.

    Returns:
        A configured ``Agent[context_type, result_type]`` ready for ``.run()``.

    Example::

        class ChatResponse(BaseModel):
            content: str
            confidence: float

        agent = build_structured_agent(
            model=adapter,
            result_type=ChatResponse,
            context_type=UserContext,
        )
        result = await agent.run("Summarise this.", deps=user_ctx)
        response: ChatResponse = result.data
    """
    return Agent(
        model=model,
        deps_type=context_type,
        output_type=result_type,
        system_prompt=system_prompt
        or (
            "You are a helpful AI assistant. "
            "Always respond in valid JSON matching the requested schema."
        ),
    )
