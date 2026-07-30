---
title: The Public API Map
description: Defines the canonical public surface of the Kitkat library.
category: Reference
order: 3
---

This document defines the canonical public surface of the Kitkat library. Every symbol
that users can import is listed here by namespace. Anything not listed here is **internal**
and should not be relied upon outside of the Kitkat source tree.

---

## Root Namespace: `from kitkat import ...`

**Guideline.** Only the most common entry points and data types live at the root.
These must have **zero optional dependencies** — importing them must never raise an
`ImportError`.

```python
# Service entry points
from kitkat import LLMService, BYOKLLMService, create_llm_service

# Data models
from kitkat import LLMRequest, LLMResponse, StreamChunk, TokenUsage, Message

# Enums for request configuration
from kitkat import Role, FinishReason, ProviderType

# Base exception for generic catch blocks
from kitkat import LLMError
```

---

## Providers Namespace: `from kitkat.providers import ...`

**Guideline.** Each concrete provider lives in its own sub-module so that a missing
SDK dependency (e.g. `anthropic`) fails at the provider import site, not at the
library root.

```python
# Concrete providers (each requires an optional extra)
from kitkat.providers.anthropic import AnthropicProvider, AnthropicConfig    # pip install kitkat[anthropic]
from kitkat.providers.openai import OpenAIProvider, OpenAIConfig             # pip install kitkat[openai]
from kitkat.providers.gemini import GeminiProvider, GeminiConfig              # pip install kitkat[gemini]

# Plugin / registry API (for advanced users adding custom providers)
from kitkat.providers import register_provider, get_provider_class
```

---

## Service Configuration Namespace: `from kitkat.service import ...`

**Guideline.** Advanced orchestration features (routers, caches) are used during
service setup, not in the hot path of every request. They live here.

```python
# Routing
from kitkat.service import LLMRouter, RoutingStrategy

# Caching
from kitkat.service import InMemoryCache, RedisCache
```

---

## Agents Namespace: `from kitkat.agents import ...`

**Guideline.** Requires `pip install kitkat[agents]`. Everything needed to build
and run a PydanticAI agent backed by a Kitkat model.

```python
# Agent builders (primary DX)
from kitkat.agents import build_chat_agent, build_structured_agent

# Adapters that bridge a Kitkat service to PydanticAI
from kitkat.agents import ManagedModelAdapter, BYOKModelAdapter

# Context and routing
from kitkat.agents import BaseAgentContext, RoutingTier

# Tool infrastructure
from kitkat.agents import ToolRegistry
```

---

## Workflows Namespace: `from kitkat.workflows import ...`

**Guideline.** Requires `pip install kitkat[workflows]`. Separated from agents
because LangGraph is a heavy, distinct dependency.

```python
from kitkat.workflows import ResearchWorkflow, ResearchState
```

---

## Advanced / Extension Namespaces

**Guideline.** Only used by developers writing custom providers or performing
fine-grained error handling.

### `from kitkat.abc import ...`

```python
# Base class for custom providers
from kitkat.abc import LLMProvider
```

### `from kitkat.core.exceptions import ...`

```python
# Specific error types
from kitkat.core.exceptions import LLMRateLimitError, LLMAuthenticationError, LLMTimeoutError
```

---

## Usage by Scenario

### Scenario 1: Basic Managed Completion (the 90 % use-case)

```python
from kitkat import LLMService, LLMRequest, Message, Role, ProviderType, create_llm_service
from kitkat.providers.anthropic import AnthropicProvider, AnthropicConfig

service = create_llm_service({
    ProviderType.ANTHROPIC: AnthropicProvider(AnthropicConfig(api_key="sk-...")),
})
response = await service.complete(
    LLMRequest(messages=[Message(role=Role.USER, content="Hi")]),
)
```

### Scenario 2: BYOK Chat Agent (FastAPI server use-case)

```python
from kitkat import BYOKLLMService, ProviderType
from kitkat.agents import BYOKModelAdapter, build_chat_agent, BaseAgentContext

async with BYOKLLMService(ProviderType.OPENAI, user_key, "gpt-4o") as byok:
    adapter = BYOKModelAdapter(byok_service=byok)
    agent = build_chat_agent(model=adapter)
    result = await agent.run("Hello", deps=BaseAgentContext(user_id="u1"))
```

### Scenario 3: Custom Provider Author (the 1 % use-case)

```python
from kitkat import LLMRequest, LLMResponse, TokenUsage, FinishReason, ProviderType
from kitkat.abc import LLMProvider
from kitkat.providers import register_provider

class MyProvider(LLMProvider):
    ...

register_provider("my-prov", MyProvider)
```

---

## Implementation Note: Lazy Root Imports

The root `src/kitkat/__init__.py` relies on **lazy imports** so that
`from kitkat import LLMService` never crashes when optional dependencies
(`anthropic`, `pydantic-ai`, etc.) are absent. Only the core and service
modules — which have zero optional dependencies — are imported eagerly.

```python
# src/kitkat/__init__.py
"""Kitkat: Production-grade LLM provider library with BYOK and Agent support."""

__version__ = "0.4.0"

# --- Core & Service (zero optional deps, safe to import directly) ---
from kitkat.core.enums import Role, FinishReason, ProviderType
from kitkat.core.models import LLMRequest, LLMResponse, StreamChunk, TokenUsage, Message
from kitkat.core.exceptions import LLMError

from kitkat.service.managed import LLMService
from kitkat.service.byok import BYOKLLMService
from kitkat.service.factory import create_llm_service

# --- Public API contract ---
__all__ = [
    "__version__",
    # Core
    "Role", "FinishReason", "ProviderType",
    "LLMRequest", "LLMResponse", "StreamChunk", "TokenUsage", "Message",
    "LLMError",
    # Service
    "LLMService", "BYOKLLMService", "create_llm_service",
]

# Optional features (agents, workflows, providers) are NOT imported here.
# They are accessed via sub-module imports such as:
#   from kitkat.providers.anthropic import AnthropicProvider
# This prevents ImportError when an SDK is missing.
```

---

## Design Rationale

The namespace structure mirrors the library's vertical architecture but presents
it **horizontally to the user based on frequency of use**:

| User type              | Primary namespace(s)             | Optional extras          |
| ---------------------- | -------------------------------- | ------------------------ |
| Managed-service user   | `kitkat` (root)                  | None                     |
| BYOK-service user      | `kitkat`, `kitkat.agents`        | `kitkat[agents]`         |
| Orchestrator           | `kitkat`, `kitkat.service`       | None                     |
| Agent builder          | `kitkat.agents`                  | `kitkat[agents]`         |
| Workflow builder       | `kitkat.workflows`               | `kitkat[workflows]`      |
| Custom-provider author | `kitkat.abc`, `kitkat.providers` | varies by provider extra |

This protects users from dependency hell, keeps the main API flat and discoverable,
and tucks heavy, domain-specific features (agents, LangGraph workflows) into their
own isolated worlds.
