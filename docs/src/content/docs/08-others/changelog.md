---
title: Changelog
description: All notable changes to this project will be documented in this file.
order: 1
---

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.1] - 2026-08-20

### Fixed

- **Resolved Eager Import of Optional Agent Dependencies**: Fixed an issue where importing the top-level `kitkat` package or `kitkat.agents.context` without `kitkat[agents]` installed raised an `ImportError` due to eager evaluation of `pydantic-ai` adapters in `kitkat.agents.__init__.py`.
- **Implemented PEP 562 Lazy Attribute Resolution**: Refactored `kitkat.agents` to eagerly export only zero-dependency context symbols (`BaseAgentContext`, `RoutingTier`), deferring all `pydantic-ai` adapter and builder imports to module-level `__getattr__`.
- **Workflows Missing Extra Message**: Corrected the installation extra name in the `kitkat.workflows` missing dependency error message to `kitkat[workflows]`.

## [0.7.0] - 2026-08-16

### Changed

- **Renamed Gemini Provider to Google (Breaking Change)**: Refactored the Gemini provider to `Google` for alignment with the official `google-genai` SDK and Vertex AI nomenclature.
  - Package path updated from `kitkat.providers.gemini` to `kitkat.providers.google`.
  - Class `GeminiProvider` renamed to `GoogleProvider`, and `ProviderType.GEMINI` changed to `ProviderType.GOOGLE`.
  - Optional dependency extra changed from `kitkat[gemini]` to `kitkat[google]`.
- **Made Redis a Core Dependency**: Promoted `redis>=7.1.1` from optional dependency extras (`kitkat[redis]`) to a mandatory core requirement to support async response caching out of the box.
- **Typed Provider Thinking Parameters**: Updated Anthropic, Google, and OpenAI provider interfaces to replace generic keyword arguments with explicit, typed parameters for thinking and reasoning configurations.
- **Router Strategy & Cache Type Safety**: Refactored `LLMRouter` internals and strategy contracts to use explicit return types and strict annotations.

### Added

- **PEP 561 Package Typing Marker (`py.typed`)**: Added `py.typed` marker file to package root to expose inline type annotations to downstream consumers.
- **Strict Static Type Checking**: Integrated `mypy` and `pyright` / `basedpyright` (via `pyrightconfig.json`) into the repository and CI pipeline.
- **Integration & End-to-End Test Suites**: Added E2E tests for agent execution and live provider integration test suites (`tests/integration/` and `tests/e2e/`).
- **Documentation Site Overhaul**: Added comprehensive Astro-based documentation website under `docs/` covering architecture, core concepts, agent layers, BYOK, provider configuration and API reference.

### Fixed

- **OpenTelemetry Observability Type Check**: Resolved typing inconsistencies with `opentelemetry-sdk` in `kitkat.agents.observability` to ensure valid span creation and context propagation.

## [0.6.0] - 2026-08-11

### Added

- **Plugin System Surface (`kitkat.plugins`)**: Public discovery and registration API for custom third-party LLM providers.
  - **Public Functions**: Exposes `discover_plugins()`, `register_provider()`, `get_provider_class()`, and `list_providers()`.
  - **Entry-Point Auto-Discovery**: Automatically loads custom providers registered under the `kitkat.providers` entry-point group in `pyproject.toml`, with fault-tolerant warning logging for invalid or duplicate entry points.
- **Observability Integration (`kitkat.agents.configure_observability`)**: One-line configuration for agent tracing, latency tracking, token metrics, and execution spans.
  - **Logfire Integration**: Native PydanticAI agent instrumentation (`Agent.instrument_all()`) and Logfire tracer configuration (`logfire.configure()`).
  - **Langfuse Integration**: Connects Langfuse tracing by attaching an OpenTelemetry OTLP HTTP span exporter (`OTLPSpanExporter` + `BatchSpanProcessor`) to the TracerProvider, enabling simultaneous trace collection in Logfire and Langfuse without conflicts.
  - **Package Extra (`kitkat[observability]`)**: Added `observability` optional dependency bundle (`logfire`, `langfuse`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp-proto-http`) and updated `kitkat[all]` bundle.

## [0.5.0] - 2026-08-01

### Added

- **LangGraph Workflows Layer (`kitkat.workflows`)**: Added stateful, multi-step agentic workflows using LangGraph.
  - **Base Workflow Abstract Class (`BaseWorkflow`)**: A generic abstract base class defining a unified contract (`build_graph()` and `run()`) for LangGraph-based workflows.
  - **Research Workflow (`ResearchWorkflow`)**: A concrete multi-step research workflow utilizing parallel execution for search and document retrieval, state verification via Pydantic model validation (`ResearchState`), and support for Auth0-ready human-in-the-loop approval conditional hooks.
  - **State Schema (`ResearchState`)**: Type-safe Pydantic state model for the research pipeline tracking queries, plans, retrieved documents, final synthesis, and approval states.
- **Package Extra & Lazy Exports**:
  - Added `workflows = ["langgraph>=1.2.0"]` optional dependency extra (`pip install kitkat[workflows]`).
  - Added lazy `__getattr__` exports on top-level `kitkat` package for `BaseWorkflow`, `ResearchWorkflow`, and `ResearchState` to keep core package imports lightweight.

## [0.4.0] - 2026-07-28

### Added

- **PydanticAI Model Adapters (`kitkat.agents`)**: Seamless integration between kitkat LLM services and PydanticAI 2.x agents.
  - **Managed Model Adapter (`ManagedModelAdapter`)**: Implements PydanticAI's `Model` protocol backed by `LLMService` for managed server-side API key routing.
  - **BYOK Model Adapter (`BYOKModelAdapter`)**: Implements PydanticAI's `Model` protocol backed by `BYOKLLMService` for per-request user-supplied API keys.
  - **Async Stream Adapter (`KitkatStreamedResponse`, `BYOKKitkatStreamedResponse`)**: Subclasses PydanticAI `StreamedResponse` to bridge kitkat async stream chunks into PydanticAI event streams, accurately tracking input, output, and reasoning/thinking tokens.
  - **Multi-Turn & Tool Message Translation**: Internal `_to_llm_request()` handles `SystemPromptPart`, `InstructionPart`, `UserPromptPart`, `TextPart`, `ToolCallPart`, and `ToolReturnPart` for multi-turn agent reasoning loops.
- **Agent Context & Routing Tier (`kitkat.agents.BaseAgentContext`)**:
  - **`BaseAgentContext`**: Minimal, stdlib-only context container (`user_id`, `routing_tier`, `locale`, `system_prompt_override`, `metadata`) used as the `deps_type` for agent runs with zero third-party dependencies.
  - **`RoutingTier` Enum (`kitkat.core.enums.RoutingTier`)**: Added `MANAGED`, `BYOK`, and `ENTERPRISE` routing tiers to `kitkat.core.enums`.
- **Agent Builders (`kitkat.agents.builders`)**:
  - **`build_chat_agent()`**: Factory function creating `Agent[ContextT, str]` instances with automatic locale-aware system prompts (`User locale: {locale}`) and support for static prompt overrides, custom `output_type`, and `output_retries`.
  - **`build_structured_agent()`**: Factory function creating `Agent[ContextT, BaseModel]` for schema-validated Pydantic outputs, JSON formatting instructions, `output_retries` handling, and custom post-validation hooks via `validator`.
- **Tool Registry (`kitkat.agents.ToolRegistry`)**: Programmatic bulk tool registration system for PydanticAI agents.
  - **Flexible Decorator Overloads**: Supports both bare decorators (`@registry.tool`) and metadata-rich decorators (`@registry.tool(name=..., description=..., prep=True)`).
  - **Bulk Registration (`register_on`)**: Registers all collected tools onto a PydanticAI `Agent` instance with custom tool names, descriptions, and preparation flags.
  - **Container Ergonomics**: Added `__len__`, `__contains__`, and copy-safe `tools` property for membership checks and inspection.
- **Package Extra & Lazy Exports**:
  - Added `agents = ["pydantic-ai>=2.0"]` optional dependency extra (`pip install kitkat[agents]`).
  - Eagerly exports `BaseAgentContext` and `RoutingTier` at `kitkat` top level with zero third-party dependencies.
  - Module-level lazy `__getattr__` exports for `ManagedModelAdapter`, `BYOKModelAdapter`, `KitkatStreamedResponse`, `build_chat_agent`, `build_structured_agent`, and `ToolRegistry` so importing `kitkat` does not pull in `pydantic_ai` unless requested.

## [0.3.0] - 2026-07-25

### Added

- **Multi-Provider LLM Router (`kitkat.service.LLMRouter`)**: In-process multi-provider resilience and routing facade.
  - **Routing Strategies (`RoutingStrategy`)**: Configurable provider selection strategies including `FAILOVER`, `ROUND_ROBIN`, `LEAST_LATENCY`, and `RANDOM`.
  - **Circuit Breaker (`CircuitBreaker`)**: Asyncio-safe state machine (`CLOSED`, `OPEN`, `HALF_OPEN`) per provider slot to isolate failing endpoints and prevent thundering herds.
  - **Resilience & Rate-Limit Tracking**: Tracks `Retry-After` windows to skip 429'd endpoints, while immediately bubbling non-retryable errors (`LLMTokenLimitError`, `LLMContentFilterError`, `LLMAuthenticationError`).
  - **Mid-Stream Protection**: Ensures fallback occurs only before first token emission to avoid streaming payload corruption.
  - **Management**: Provides async `reset_circuit_breaker()` and detailed pool `status()` reporting.
- **Async Response Cache (`kitkat.service.LLMCache`)**: Deterministic caching system for non-streaming LLM completions.
  - **Deterministic Hashing**: SHA-256 key generation based on semantic request attributes (`messages`, `model`, `max_tokens`, `temperature`, `top_p`, `stop_sequences`).
  - **In-Memory Backend (`CacheBackendType.MEMORY`)**: Asyncio-safe LRU cache backed by `OrderedDict` with automatic TTL eviction.
  - **Redis Backend (`CacheBackendType.REDIS`)**: Distributed caching via `redis.asyncio` with non-blocking `SCAN` key iteration, batched purging, and configurable `key_prefix`.
  - **Fail-Safe Orchestrator**: Backend operational errors are safely caught to guarantee cache issues never interrupt LLM inference.
  - **Selective Caching**: Skips storing truncated responses by default and ignores non-cacheable finish reasons (`CONTENT_FILTER`, `ERROR`).
- **Factories & API Surface**:
  - Added `create_llm_router()` convenience factory in `kitkat.service.factory`.
  - Re-exported all router and cache entities (`LLMRouter`, `RouterConfig`, `RoutingStrategy`, `LLMCache`, `CacheConfig`, `CacheBackendType`, `create_llm_router`) at `kitkat` and `kitkat.service`.
- **Package Extras**:
  - Added `redis = ["redis>=5.0"]` optional dependency extra (`kitkat[redis]`).

## [0.2.0] - 2026-06-25

### Added

- **Service Layer (`kitkat.service`)**: The core entry points for API integrations have been cleanly organized.
  - `LLMService` (Managed Service) handles provider registry, routing, health checks, and lifecycle.
  - `BYOKLLMService` provides a safe, short-lived async context manager for per-request user-supplied API keys (BYOK).
  - `create_llm_service` factory function for simplified setup of multiple providers.
- **Stable Public API Surface (`kitkat.__init__.py`)**: Exposes the entire framework via a single top-level import.

### Changed

- The legacy `kitkat.service.service` and `kitkat.exceptions` modules have been deprecated.

## [0.1.0] - 2026-06-23

### Added

- **Core Models Layer (`kitkat.core`)**: Zero-dependency domain models (`LLMRequest`, `LLMResponse`, `StreamChunk`, `TokenUsage`) and enums (`Role`, `FinishReason`, `ProviderType`).
- **Exception Hierarchy (`kitkat.core.exceptions`)**: Unified, typed error handling across all providers (e.g., `LLMRateLimitError`, `LLMAuthenticationError`, `LLMTimeoutError`).
- **Provider ABC (`kitkat.abc.LLMProvider`)**: The abstract base class defining the contract for all provider implementations.
- **Anthropic Provider (`kitkat[anthropic]`)**: Full support for Claude models, including extended thinking via `ThinkingConfig`.
- **OpenAI Provider (`kitkat[openai]`)**: Compatible with OpenAI's Chat Completions API and alternative endpoints like NVIDIA NIM or vLLM via `base_url`. Includes o-series reasoning support.
- **Gemini Provider (`kitkat[gemini]`)**: Uses the new official `google-genai` SDK, with standard API key auth and Vertex AI enterprise deployment support (`vertexai=True`).
- **Async Streaming**: First-class async streaming for all providers via the `stream()` method, yielding typed `StreamChunk` objects.
- **Token Estimation**: Synchronous `count_tokens()` across all providers using a shared `tiktoken` implementation with an air-gapped character-ratio fallback.
- **Retry Logic**: Built-in exponential back-off wrapper with jitter that automatically handles transient errors (429, 5xx) and respects `Retry-After` HTTP headers.
- **Plugin Registry (`kitkat.providers._registry`)**: Provider auto-discovery via Python `entry-points`, allowing third-party packages to inject custom providers seamlessly.
