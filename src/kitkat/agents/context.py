"""BaseAgentContext — minimal deps dataclass for all kitkat agents.

Consumers subclass this to add application-specific fields.  The library
only references fields declared here; application-level fields are opaque
to library code and accessible only to application-registered tools.

Subclass pattern::

    from dataclasses import dataclass, field
    from kitkat.agents.context import BaseAgentContext

    @dataclass
    class UserContext(BaseAgentContext):
        conversation_id: str = ""
        byok_api_key: str | None = None
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.enums import RoutingTier

__all__ = ["BaseAgentContext", "RoutingTier"]


@dataclass
class BaseAgentContext:
    """Minimal context required by the kitkat agent layer.

    Every PydanticAI agent built with :func:`~kitkat.agents.builders.build_chat_agent`
    or :func:`~kitkat.agents.builders.build_structured_agent` uses this class (or
    a subclass) as its ``deps_type``.

    Attributes:
        user_id: Opaque identifier for the calling user.  Required — must be
            set by the caller; never defaulted to an empty string to avoid
            silent routing mistakes.
        routing_tier: Determines which service path handles this request.
            Defaults to :attr:`~kitkat.core.enums.RoutingTier.MANAGED`.
        locale: IETF BCP-47 locale tag injected into the default system prompt.
            Defaults to ``"en"``.
        system_prompt_override: When non-``None``, replaces the library's default
            system prompt entirely.  Useful for per-user or per-tenant prompts.
        metadata: Free-form application data.  Library code never reads this;
            it exists solely for application tools that receive the context.
    """

    user_id: str
    routing_tier: RoutingTier = RoutingTier.MANAGED
    locale: str = "en"
    system_prompt_override: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
