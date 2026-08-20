"""Unit tests for PEP 562 lazy exports in kitkat and kitkat.agents."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

import pytest


class TestLazyExports:
    """Verify PEP 562 lazy attribute loading across kitkat packages."""

    def test_kitkat_dir_includes_lazy_exports(self) -> None:
        """Top-level kitkat dir() should contain both eager and lazy exports."""
        import kitkat

        all_dir = dir(kitkat)
        assert "LLMRequest" in all_dir
        assert "BaseAgentContext" in all_dir
        assert "BYOKModelAdapter" in all_dir
        assert "ManagedModelAdapter" in all_dir
        assert "BaseWorkflow" in all_dir
        assert "ResearchWorkflow" in all_dir
        assert "LLMService" in all_dir

    def test_kitkat_agents_dir_includes_lazy_exports(self) -> None:
        """kitkat.agents dir() should contain both context types and lazy adapters."""
        import kitkat.agents

        agents_dir = dir(kitkat.agents)
        assert "BaseAgentContext" in agents_dir
        assert "RoutingTier" in agents_dir
        assert "BYOKModelAdapter" in agents_dir
        assert "ManagedModelAdapter" in agents_dir
        assert "build_chat_agent" in agents_dir
        assert "build_structured_agent" in agents_dir
        assert "ToolRegistry" in agents_dir
        assert "configure_observability" in agents_dir

    def test_kitkat_getattr_unknown_attribute_raises_attribute_error(self) -> None:
        """Accessing a non-existent attribute should raise standard AttributeError."""
        import kitkat

        with pytest.raises(AttributeError, match="has no attribute 'NonExistentSymbol'"):
            _ = kitkat.NonExistentSymbol  # type: ignore[attr-defined]

    def test_kitkat_agents_getattr_unknown_attribute_raises_attribute_error(self) -> None:
        """Accessing non-existent attribute on kitkat.agents raises AttributeError."""
        import kitkat.agents

        with pytest.raises(AttributeError, match="has no attribute 'NonExistentAgentSymbol'"):
            _ = kitkat.agents.NonExistentAgentSymbol  # type: ignore[attr-defined]

    def test_top_level_lazy_access_returns_attribute(self) -> None:
        """Accessing lazy attributes on kitkat returns the correct imported classes."""
        import kitkat
        from kitkat.agents.adapters.byok import BYOKModelAdapter
        from kitkat.workflows.base import BaseWorkflow

        assert kitkat.BYOKModelAdapter is BYOKModelAdapter
        assert kitkat.BaseWorkflow is BaseWorkflow

    def test_agents_lazy_access_returns_attribute(self) -> None:
        """Accessing lazy attributes on kitkat.agents returns the correct imported classes."""
        import kitkat.agents
        from kitkat.agents.adapters.byok import BYOKModelAdapter
        from kitkat.agents.builders import build_chat_agent

        assert kitkat.agents.BYOKModelAdapter is BYOKModelAdapter
        assert kitkat.agents.build_chat_agent is build_chat_agent


class TestZeroDependencyAgentContext:
    """Verify BaseAgentContext and root imports succeed when optional extras are missing."""

    def test_root_and_context_import_without_pydantic_ai(self) -> None:
        """Importing kitkat and BaseAgentContext must never fail if pydantic-ai is missing."""
        # Unload kitkat modules from sys.modules for clean reload test
        modules_to_unload = [
            mod for mod in list(sys.modules.keys()) if mod == "kitkat" or mod.startswith("kitkat.")
        ]
        saved_modules = {m: sys.modules.pop(m) for m in modules_to_unload}

        try:
            with patch.dict(sys.modules, {"pydantic_ai": None}):
                # Core package import
                kitkat_mod = importlib.import_module("kitkat")
                assert hasattr(kitkat_mod, "LLMRequest")
                assert hasattr(kitkat_mod, "ProviderType")
                assert hasattr(kitkat_mod, "BaseAgentContext")

                # Context module directly
                context_mod = importlib.import_module("kitkat.agents.context")
                ctx_cls = context_mod.BaseAgentContext
                ctx = ctx_cls(user_id="user_test_123")
                assert ctx.user_id == "user_test_123"

                # Accessing agent adapter lazily when pydantic_ai is missing must raise ImportError
                agents_mod = importlib.import_module("kitkat.agents")
                with pytest.raises(ImportError, match="Agent features require the 'agents' extra"):
                    _ = agents_mod.BYOKModelAdapter
        finally:
            # Restore saved modules to prevent side-effects on other tests
            for mod in list(sys.modules.keys()):
                if mod == "kitkat" or mod.startswith("kitkat."):
                    sys.modules.pop(mod, None)
            sys.modules.update(saved_modules)
