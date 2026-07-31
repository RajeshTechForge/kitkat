---
title: The Public API Map
description: Defines the canonical public surface of the Kitkat library.
category: Reference
order: 3
---

This document defines the canonical public surface of the Kitkat library. Every symbol
that users can import is listed here by namespace. Anything not listed here is **internal**
and should not be relied upon outside of the Kitkat source tree.


## Root Namespace

`from kitkat import ...`

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


## Providers Namespace

`from kitkat.providers import ...`

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


## Service Configuration Namespace

`from kitkat.service import ...`

**Guideline.** Advanced orchestration features (routers, caches) are used during
service setup, not in the hot path of every request. They live here.

```python
# Routing
from kitkat.service import LLMRouter, RoutingStrategy

# Caching
from kitkat.service import InMemoryCache, RedisCache
```


## Agents Namespace

`from kitkat.agents import ...`

**Guideline.** Requires `pip install kitkat[agents]`. Everything needed to build
and run a PydanticAI agent backed by a Kitkat model.

```python
# Agent builders (primary DX)
from kitkat.agents import build_chat_agent, build_structured_agent

# Adapters that bridge a Kitkat service to PydanticAI
from kitkat.agents import ManagedModelAdapter, BYOKModelAdapter

# Context, routing and Tool infrastructure
from kitkat.agents import BaseAgentContext, RoutingTier, ToolRegistry
```


## Workflows Namespace

`from kitkat.workflows import ...`

**Guideline.** Requires `pip install kitkat[workflows]`. Separated from agents
because LangGraph is a heavy, distinct dependency.

```python
from kitkat.workflows import ResearchWorkflow, ResearchState
```


## Design Rationale

The namespace structure mirrors the library's vertical architecture but presents
it **horizontally to the user based on frequency of use**:

| User type              | Primary namespace(s)             | Optional extras          |
| ---------------------- | -------------------------------- | ------------------------ |
| Managed-service user   | `kitkat` (root)                  | None                     |
| Orchestrator           | `kitkat.service`                 | None                     |
| BYOK-service user      | `kitkat`, `kitkat.agents`        | `kitkat[agents]`         |
| Agent builder          | `kitkat.agents`                  | `kitkat[agents]`         |
| Workflow builder       | `kitkat.workflows`               | `kitkat[workflows]`      |
| Custom-provider author | `kitkat.abc`, `kitkat.providers` | varies by provider extra |

This protects users from dependency hell, keeps the main API flat and discoverable,
and tucks heavy, domain-specific features (agents, LangGraph workflows) into their
own isolated worlds.
