# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
