"""Example demonstrating how to implement and register a custom third-party provider.

This example implements a custom provider for a self-hosted vLLM instance using an
OpenAI-compatible HTTP endpoint, and registers it with kitkat's plugin system.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

import httpx

from kitkat._internal.tokenizers import count_tokens_tiktoken
from kitkat.abc import LLMProvider
from kitkat.core import (
    FinishReason,
    LLMProviderError,
    LLMProviderInitError,
    LLMRequest,
    LLMResponse,
    ProviderCapabilities,
    ProviderType,
    StreamChunk,
    TokenUsage,
)
from kitkat.plugins import get_provider_class, list_providers, register_provider
from kitkat.service import create_llm_service


class VLLMProvider(LLMProvider):
    """Provider implementation for self-hosted vLLM server with OpenAI-compatible API."""

    PROVIDER_TYPE = ProviderType.OPENAI
    DEFAULT_MODEL = "meta-llama/Llama-3-8b-instruct"
    CAPABILITIES = ProviderCapabilities(
        supports_streaming=True,
        supports_thinking=False,
        max_context_tokens=8_192,
        provider_type=ProviderType.OPENAI,
    )

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize vLLM provider configuration.

        Args:
            config: Provider configuration dictionary. Must contain ``base_url``.
                Optional keys: ``model`` (str), ``timeout`` (float).
        """
        super().__init__(config)
        self._base_url = config.get("base_url", "http://localhost:8000")
        self._model = config.get("model", self.DEFAULT_MODEL)
        self._timeout = float(config.get("timeout", 30.0))
        self._client: httpx.AsyncClient | None = None

    async def initialize(self) -> None:
        """Initialize the HTTP client and verify connection liveness.

        Raises:
            LLMProviderInitError: If the server endpoint is unreachable.
        """
        await self._init_client_only()
        assert self._client is not None
        try:
            resp = await self._client.get("/v1/models")
            resp.raise_for_status()
        except Exception as exc:
            await self.shutdown()
            raise LLMProviderInitError(
                f"Failed to connect to vLLM server at {self._base_url}: {exc}"
            ) from exc

    async def _init_client_only(self) -> None:
        """Instantiate the httpx client without liveness probing."""
        if self._initialized:
            return
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
        )
        self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown the underlying HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        self._initialized = False

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Execute a single non-streaming completion call.

        Args:
            request: The generation request details.

        Returns:
            The provider LLMResponse.

        Raises:
            LLMProviderError: If the server returns an HTTP error.
        """
        self._assert_initialized()
        assert self._client is not None

        start_time = time.monotonic()
        messages = [{"role": m.role.value, "content": m.content} for m in request.messages]
        payload = {
            "model": request.model or self._model,
            "messages": messages,
            "max_tokens": request.max_tokens or 512,
            "temperature": request.temperature,
        }

        try:
            resp = await self._client.post("/v1/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            raise LLMProviderError(f"vLLM completion failed: {exc}") from exc

        choice = data["choices"][0]
        usage_data = data.get("usage", {})
        usage = TokenUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
        )

        return LLMResponse(
            content=choice["message"]["content"],
            model=data.get("model", self._model),
            provider=self.PROVIDER_TYPE,
            usage=usage,
            finish_reason=FinishReason(choice.get("finish_reason", "stop")),
            latency_ms=(time.monotonic() - start_time) * 1_000,
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        """Yield token stream deltas from the provider endpoint.

        Args:
            request: The streaming generation request details.

        Yields:
            StreamChunk objects for each delta.
        """
        self._assert_initialized()
        assert self._client is not None

        start_time = time.monotonic()
        messages = [{"role": m.role.value, "content": m.content} for m in request.messages]
        payload = {
            "model": request.model or self._model,
            "messages": messages,
            "max_tokens": request.max_tokens or 512,
            "temperature": request.temperature,
            "stream": True,
        }

        try:
            async with self._client.stream(
                "POST", "/v1/chat/completions", json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line.removeprefix("data: ").strip()
                    if data_str == "[DONE]":
                        break
                    # For demonstration purposes, yield streaming chunks:
                    yield StreamChunk(
                        delta=data_str,
                        is_final=False,
                    )
        except httpx.HTTPError as exc:
            raise LLMProviderError(f"vLLM stream failed: {exc}") from exc

        yield StreamChunk(
            delta="",
            is_final=True,
            provider=self.PROVIDER_TYPE,
            model=self._model,
            finish_reason=FinishReason.STOP,
            latency_ms=(time.monotonic() - start_time) * 1_000,
        )

    async def health_check(self) -> bool:
        """Perform a liveness check on the vLLM server."""
        if not self._initialized or self._client is None:
            return False
        try:
            resp = await self._client.get("/v1/models")
            return resp.status_code == 200
        except Exception:
            return False

    def count_tokens(self, text: str) -> int:
        """Estimate token count using tiktoken."""
        return count_tokens_tiktoken(text)


async def main() -> None:
    """Demonstrate custom provider registration and usage."""
    # 1. Register the custom provider programmatically
    provider_name = "vllm_custom"
    register_provider(provider_name, VLLMProvider)
    print(f"Registered custom provider: {provider_name!r}")
    print(f"Available providers in registry: {list_providers()}")

    # 2. Retrieve provider class via plugin loader
    provider_cls = get_provider_class(provider_name)
    print(f"Retrieved class: {provider_cls.__name__}")

    # 3. Instantiate and use with LLMService
    provider_instance = provider_cls({"base_url": "http://localhost:8000"})
    service = create_llm_service({ProviderType.OPENAI: provider_instance})

    # Introspect health (will be False if no local server is running)
    is_healthy = await provider_instance.health_check()
    print(f"Provider health status: {is_healthy}")
    await service.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
