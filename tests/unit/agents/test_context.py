"""Unit tests for BaseAgentContext and RoutingTier."""

from __future__ import annotations

from dataclasses import dataclass

from kitkat.agents.context import BaseAgentContext, RoutingTier
from kitkat.core.enums import RoutingTier as CoreRoutingTier


class TestRoutingTier:
    def test_values(self) -> None:
        assert RoutingTier.MANAGED == "managed"
        assert RoutingTier.BYOK == "byok"
        assert RoutingTier.ENTERPRISE == "enterprise"

    def test_is_str_subclass(self) -> None:
        assert isinstance(RoutingTier.MANAGED, str)

    def test_core_enum_is_same(self) -> None:
        """RoutingTier re-exported from context must be identical to core enum."""
        assert RoutingTier is CoreRoutingTier


class TestBaseAgentContext:
    def test_required_user_id(self) -> None:
        ctx = BaseAgentContext(user_id="u123")
        assert ctx.user_id == "u123"

    def test_default_routing_tier(self) -> None:
        ctx = BaseAgentContext(user_id="u123")
        assert ctx.routing_tier == RoutingTier.MANAGED

    def test_default_locale(self) -> None:
        ctx = BaseAgentContext(user_id="u123")
        assert ctx.locale == "en"

    def test_default_system_prompt_override_is_none(self) -> None:
        ctx = BaseAgentContext(user_id="u123")
        assert ctx.system_prompt_override is None

    def test_default_metadata_is_empty_dict(self) -> None:
        ctx = BaseAgentContext(user_id="u123")
        assert ctx.metadata == {}

    def test_metadata_instances_are_independent(self) -> None:
        """Each instance gets a fresh dict — no mutable default sharing."""
        ctx1 = BaseAgentContext(user_id="u1")
        ctx2 = BaseAgentContext(user_id="u2")
        ctx1.metadata["key"] = "val"
        assert "key" not in ctx2.metadata

    def test_custom_values(self) -> None:
        ctx = BaseAgentContext(
            user_id="u999",
            routing_tier=RoutingTier.BYOK,
            locale="fr",
            system_prompt_override="Tu es un assistant.",
            metadata={"plan": "pro"},
        )
        assert ctx.routing_tier == RoutingTier.BYOK
        assert ctx.locale == "fr"
        assert ctx.system_prompt_override == "Tu es un assistant."
        assert ctx.metadata == {"plan": "pro"}

    def test_subclass_pattern(self) -> None:
        """Consumers can add fields; library fields remain accessible."""

        @dataclass
        class UserContext(BaseAgentContext):
            conversation_id: str = ""
            byok_api_key: str | None = None

        ctx = UserContext(user_id="u1", conversation_id="conv-42")
        assert ctx.user_id == "u1"
        assert ctx.conversation_id == "conv-42"
        assert ctx.routing_tier == RoutingTier.MANAGED
