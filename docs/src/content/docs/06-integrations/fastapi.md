---
title: FastAPI Integration Guide
description: Learn how to integrate KitKat into FastAPI web applications, including lifecycle management, dependency injection, streaming, error handling, and BYOK endpoints.
order: 1
---

This page is the complete guide to integrating KitKat into [FastAPI](https://fastapi.tiangolo.com/) web applications. It covers application lifecycle management with lifespan context managers, dependency injection, managed and BYOK endpoints, Server-Sent Events (SSE) streaming, exception handling middleware, health check routes, and structured output integration.

---

## Overview

When building web applications with KitKat and FastAPI:

- **Lifecycle Management**: Initialize long-lived `LLMService` or `LLMRouter` instances in the FastAPI lifespan handler so connection pools are opened at application startup and closed gracefully on shutdown.
- **Dependency Injection**: Use FastAPI's `Depends` system to inject configured services into route functions.
- **Error Mapping**: Register a custom exception handler for `LLMError` to map domain exceptions (`LLMAuthenticationError`, `LLMRateLimitError`, etc.) directly to HTTP status codes (401, 429, 413, 504).
- **Streaming**: Use FastAPI's `StreamingResponse` with `service.stream()` to push Server-Sent Events (SSE) to frontend clients token-by-token.
- **BYOK Isolation**: Isolate per-request user API keys in dedicated route handlers using `BYOKLLMService`.

---

## Installation

```bash
pip install kitkat[all-providers,agents] fastapi uvicorn
```

---

## Application Lifespan & Service Setup

FastAPI uses lifespan context managers to handle startup and shutdown logic. Initialize your `LLMService` or `LLMRouter` inside the lifespan function and store it in `app.state`.

```python
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from kitkat.service import create_llm_service, LLMService
from kitkat import ProviderType
from kitkat.providers.anthropic import AnthropicProvider, AnthropicConfig
from kitkat.providers.openai import OpenAIProvider, OpenAIConfig
from kitkat.agents import configure_observability


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # 1. Configure observability (Logfire / Langfuse) before service initialization
    if os.environ.get("LOGFIRE_TOKEN"):
        configure_observability(
            service_name="fastapi-kitkat-app",
            environment=os.environ.get("ENVIRONMENT", "production"),
        )

    # 2. Instantiate and initialize long-lived provider services
    service = create_llm_service({
        ProviderType.ANTHROPIC: AnthropicProvider(
            AnthropicConfig(api_key=os.environ["ANTHROPIC_API_KEY"])
        ),
        ProviderType.OPENAI: OpenAIProvider(
            OpenAIConfig(api_key=os.environ["OPENAI_API_KEY"])
        ),
    })
    await service.initialize()

    # 3. Store the service instance in app.state for dependency injection
    app.state.llm_service = service

    yield  # Application runs and handles incoming requests

    # 4. Clean shutdown: release HTTP connection pools
    await service.shutdown()


app = FastAPI(
    title="KitKat Powered AI API",
    version="1.0.0",
    lifespan=lifespan,
)
```

---

## Dependency Injection

Define a FastAPI dependency to retrieve the `LLMService` from `request.app.state`.

```python
from fastapi import Request, Depends
from kitkat.service import LLMService


def get_llm_service(request: Request) -> LLMService:
    """FastAPI dependency providing the application-wide LLMService."""
    service: LLMService | None = getattr(request.app.state, "llm_service", None)
    if service is None:
        raise RuntimeError("LLMService is not initialized in app.state.")
    return service
```

---

## Global Error Handling Middleware

Register a custom exception handler for `LLMError`. This converts all KitKat exceptions into standardized JSON responses with proper HTTP status codes and headers.

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from kitkat import (
    LLMError,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMTokenLimitError,
    LLMContentFilterError,
    LLMTimeoutError,
    LLMProviderError,
)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(LLMError)
    async def llm_error_handler(request: Request, exc: LLMError) -> JSONResponse:
        headers: dict[str, str] = {}
        status_code = exc.status_code or 500

        if isinstance(exc, LLMAuthenticationError):
            status_code = 401
            message = "Authentication with the underlying AI provider failed."

        elif isinstance(exc, LLMRateLimitError):
            status_code = 429
            if exc.retry_after_s:
                headers["Retry-After"] = str(int(exc.retry_after_s))
            message = f"AI provider rate limit exceeded. Retry after {exc.retry_after_s or 'a few'} seconds."

        elif isinstance(exc, LLMTokenLimitError):
            status_code = 413
            message = f"Prompt exceeds maximum context length ({exc.context_limit or 'unknown'} tokens)."

        elif isinstance(exc, LLMContentFilterError):
            status_code = 400
            message = "Response was flagged or blocked by safety content filters."

        elif isinstance(exc, LLMTimeoutError):
            status_code = 504
            message = f"AI provider timed out after {exc.elapsed_s or 'unknown'} seconds."

        else:
            message = exc.message or "An unexpected AI provider error occurred."

        return JSONResponse(
            status_code=status_code,
            headers=headers,
            content={
                "error": {
                    "code": exc.code if hasattr(exc, "code") else "LLM_ERROR",
                    "message": message,
                    "provider": exc.provider,
                }
            },
        )
```

> **🔒 Security:** Never expose internal API keys or detailed provider tracebacks to web callers in `LLMAuthenticationError` responses. Map 401 errors to generic message strings as shown above.

---

## Endpoint 1: Managed Completions (Non-Streaming)

Use Pydantic request and response schemas to build a non-streaming chat completions route.

```python
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from kitkat.service import LLMService
from kitkat import ProviderType, LLMRequest, Message, Role

router = APIRouter(prefix="/v1", tags=["Completions"])


class CompletionRequest(BaseModel):
    provider: str = Field(default="anthropic", description="Provider name: 'anthropic', 'openai', or 'gemini'")
    model: str = Field(default="", description="Model ID or empty string for provider default")
    prompt: str = Field(min_length=1, max_length=10000, description="User prompt text")
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)


class TokenUsageSchema(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    content: str
    provider: str
    model: str
    finish_reason: str
    usage: TokenUsageSchema
    latency_ms: float


@router.post("/chat/completions", response_model=CompletionResponse)
async def create_chat_completion(
    body: CompletionRequest,
    service: LLMService = Depends(get_llm_service),
) -> CompletionResponse:
    try:
        provider_type = ProviderType(body.provider.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unsupported provider: '{body.provider}'")

    request = LLMRequest(
        messages=[Message(role=Role.USER, content=body.prompt)],
        model=body.model,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
    )

    response = await service.complete(request, provider_type)

    return CompletionResponse(
        content=response.content,
        provider=response.provider.value,
        model=response.model,
        finish_reason=response.finish_reason.value,
        usage=TokenUsageSchema(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        ),
        latency_ms=response.latency_ms,
    )
```

---

## Endpoint 2: Server-Sent Events (SSE) Streaming

Stream token deltas to frontend clients in real time using `StreamingResponse` and Server-Sent Events (SSE).

```python
import json
from typing import AsyncGenerator
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from kitkat.service import LLMService
from kitkat import ProviderType, LLMRequest, Message, Role

stream_router = APIRouter(prefix="/v1", tags=["Streaming"])


class StreamRequest(BaseModel):
    provider: str = Field(default="anthropic")
    prompt: str = Field(min_length=1)
    model: str = Field(default="")
    max_tokens: int = Field(default=1024)


async def sse_generator(
    service: LLMService,
    provider_type: ProviderType,
    request: LLMRequest,
) -> AsyncGenerator[str, None]:
    """Yield Server-Sent Event formatted chunks."""
    async for chunk in service.stream(request, provider_type):
        if chunk.is_thinking:
            # Optionally send reasoning tokens under a distinct event type
            payload = json.dumps({"type": "thinking", "delta": chunk.delta})
            yield f"event: thinking\ndata: {payload}\n\n"
        elif not chunk.is_final:
            payload = json.dumps({"type": "content", "delta": chunk.delta})
            yield f"event: message\ndata: {payload}\n\n"
        else:
            # Final sentinel chunk carrying complete token usage and finish reason
            payload = json.dumps({
                "type": "done",
                "finish_reason": chunk.finish_reason.value,
                "usage": {
                    "prompt_tokens": chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                    "total_tokens": chunk.usage.total_tokens,
                },
                "latency_ms": chunk.latency_ms,
            })
            yield f"event: done\ndata: {payload}\n\n"


@stream_router.post("/chat/stream")
async def stream_chat_completion(
    body: StreamRequest,
    service: LLMService = Depends(get_llm_service),
) -> StreamingResponse:
    try:
        provider_type = ProviderType(body.provider.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unsupported provider: '{body.provider}'")

    request = LLMRequest(
        messages=[Message(role=Role.USER, content=body.prompt)],
        model=body.model,
        max_tokens=body.max_tokens,
        stream=True,
    )

    return StreamingResponse(
        sse_generator(service, provider_type, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable NGINX buffering
        },
    )
```

---

## Endpoint 3: BYOK (Bring Your Own Key) Route

Multi-tenant SaaS platforms allow end users to provide their own provider API keys. Extract the user's API key from the `X-API-Key` HTTP header and scope the request inside an `async with BYOKLLMService(...)` block.

```python
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field
from kitkat.service import BYOKLLMService
from kitkat import ProviderType, LLMRequest, Message, Role

byok_router = APIRouter(prefix="/v1/byok", tags=["BYOK"])


class BYOKRequest(BaseModel):
    provider: str = Field(description="'anthropic', 'openai', or 'gemini'")
    model: str = Field(default="gpt-4o-mini")
    message: str = Field(min_length=1)


@byok_router.post("/completions")
async def byok_completion(
    body: BYOKRequest,
    x_api_key: str = Header(..., alias="X-API-Key", description="User's private LLM provider API key"),
) -> dict:
    try:
        provider_type = ProviderType(body.provider.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid provider: '{body.provider}'")

    request = LLMRequest(
        messages=[Message(role=Role.USER, content=body.message)],
        max_tokens=512,
    )

    # Short-lived provider client created on __aenter__ and destroyed on __aexit__
    async with BYOKLLMService(
        provider_type=provider_type,
        api_key=x_api_key,
        model=body.model,
    ) as svc:
        response = await svc.complete(request)

    # API key is wiped from memory once the context block exits
    return {
        "content": response.content,
        "provider": response.provider.value,
        "model": response.model,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    }
```

> **🚀 Performance:** `BYOKLLMService` skips credential probing during initialization, allowing context setup to finish in microseconds. Authentication validation occurs on the completion request itself.

---

## Endpoint 4: Agent & Structured Output Integration

Integrate KitKat agent builders (`build_structured_agent`) into FastAPI routes to return validated Pydantic models.

```python
from dataclasses import dataclass
from fastapi import APIRouter, Depends, Header
from pydantic import BaseModel, Field

from kitkat.agents import (
    BaseAgentContext,
    ManagedModelAdapter,
    build_structured_agent,
)
from kitkat.service import LLMService
from kitkat import ProviderType

agent_router = APIRouter(prefix="/v1/agents", tags=["Agents"])


class CodeReviewRequest(BaseModel):
    code_snippet: str = Field(min_length=10, description="Python source code to review")


class CodeIssue(BaseModel):
    line: int = Field(ge=1)
    severity: str = Field(description="'error', 'warning', or 'info'")
    message: str = Field(description="Description of the issue")


class CodeReviewResult(BaseModel):
    summary: str
    score: int = Field(ge=0, le=100)
    issues: list[CodeIssue]


@dataclass
class UserContext(BaseAgentContext):
    auth_token: str = ""


@agent_router.post("/review", response_model=CodeReviewResult)
async def review_code(
    body: CodeReviewRequest,
    authorization: str = Header(default="Bearer anonymous"),
    service: LLMService = Depends(get_llm_service),
) -> CodeReviewResult:
    adapter = ManagedModelAdapter(
        service=service,
        provider_type=ProviderType.ANTHROPIC,
        default_model="claude-opus-4-5",
    )

    agent = build_structured_agent(
        model=adapter,
        output_type=CodeReviewResult,
        context_type=UserContext,
        output_retries=2,
    )

    ctx = UserContext(
        user_id="fastapi-user-123",
        auth_token=authorization,
    )

    result = await agent.run(
        f"Review this code snippet:\n```python\n{body.code_snippet}\n```",
        deps=ctx,
    )

    output: CodeReviewResult = result.data
    return output
```

---

## Health Check & Monitoring Route

Expose a `/healthz` endpoint to probe the health of all registered providers.

```python
from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from kitkat.service import LLMService

health_router = APIRouter(tags=["Health"])


@health_router.get("/healthz")
async def health_check(
    service: LLMService = Depends(get_llm_service),
) -> JSONResponse:
    # Probes all initialized providers in parallel
    results: dict[str, bool] = await service.health_check_all()
    all_healthy = all(results.values()) and len(results) > 0

    return JSONResponse(
        status_code=status.HTTP_200_OK if all_healthy else status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "status": "ok" if all_healthy else "degraded",
            "providers": results,
        },
    )
```

---

## Complete Working FastAPI Application Example

Here is an import-complete, single-file FastAPI server containing all components wired together:

```python
import os
import json
from contextlib import asynccontextmanager
from typing import AsyncIterator, AsyncGenerator

from fastapi import FastAPI, Request, Depends, Header, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from kitkat.service import create_llm_service, LLMService, BYOKLLMService
from kitkat import (
    ProviderType,
    LLMRequest,
    Message,
    Role,
    LLMError,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMTokenLimitError,
    LLMTimeoutError,
)
from kitkat.providers.anthropic import AnthropicProvider, AnthropicConfig
from kitkat.providers.openai import OpenAIProvider, OpenAIConfig


# ── 1. Lifespan Handler ──────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    service = create_llm_service({
        ProviderType.ANTHROPIC: AnthropicProvider(
            AnthropicConfig(api_key=os.environ.get("ANTHROPIC_API_KEY", "dummy-key"))
        ),
        ProviderType.OPENAI: OpenAIProvider(
            OpenAIConfig(api_key=os.environ.get("OPENAI_API_KEY", "dummy-key"))
        ),
    })
    # In production, initialize connects and probes credentials
    # await service.initialize()
    app.state.llm_service = service

    yield

    await service.shutdown()


app = FastAPI(title="KitKat Production API", lifespan=lifespan)


# ── 2. Dependency ─────────────────────────────────────────────────────────

def get_llm_service(request: Request) -> LLMService:
    return request.app.state.llm_service


# ── 3. Exception Handler ──────────────────────────────────────────────────

@app.exception_handler(LLMError)
async def handle_llm_errors(request: Request, exc: LLMError) -> JSONResponse:
    if isinstance(exc, LLMAuthenticationError):
        return JSONResponse(status_code=401, content={"error": "Authentication failed"})
    if isinstance(exc, LLMRateLimitError):
        headers = {"Retry-After": str(int(exc.retry_after_s))} if exc.retry_after_s else {}
        return JSONResponse(status_code=429, headers=headers, content={"error": "Rate limit exceeded"})
    if isinstance(exc, LLMTokenLimitError):
        return JSONResponse(status_code=413, content={"error": "Prompt too long"})
    if isinstance(exc, LLMTimeoutError):
        return JSONResponse(status_code=504, content={"error": "Provider timed out"})
    return JSONResponse(status_code=500, content={"error": exc.message})


# ── 4. Routes ─────────────────────────────────────────────────────────────

class ChatPrompt(BaseModel):
    prompt: str = Field(min_length=1)
    provider: str = Field(default="anthropic")


@app.post("/v1/chat")
async def chat(
    body: ChatPrompt,
    service: LLMService = Depends(get_llm_service),
) -> dict:
    provider_enum = ProviderType(body.provider.lower())
    request = LLMRequest(messages=[Message(role=Role.USER, content=body.prompt)])
    response = await service.complete(request, provider_enum)
    return {"content": response.content, "tokens": response.usage.total_tokens}


@app.get("/healthz")
async def health(service: LLMService = Depends(get_llm_service)) -> dict:
    return {"status": "healthy"}
```

---

## Production Deployment & Best Practices

1. **Uvicorn / Gunicorn Command**:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4 --loop uvloop
   ```
2. **Connection Pooling**: Re-use the `LLMService` stored in `app.state`. Creating a new `LLMService` per HTTP request causes severe socket exhaustion.
3. **Graceful Shutdown**: Always allow active requests to finish during deployments by configuring `--timeout-graceful-shutdown 30` in Uvicorn.
4. **SSE Headers**: When serving streaming responses through proxies like NGINX or Cloudflare, ensure response headers include `X-Accel-Buffering: no` to prevent proxy buffering.

---

## Further Reading

- [Providers Overview](./providers.md) — Managed `LLMService` API reference
- [BYOK Guide](./byok.md) — Security model and key lifecycle for `BYOKLLMService`
- [Error Handling](./error-handling.md) — Comprehensive `LLMError` hierarchy reference
- [Observability](./observability.md) — Tracing FastAPI endpoints with Logfire and Langfuse
- [API Reference — Service](./api-reference/service.md) — Complete `LLMService` and `LLMRouter` signatures
