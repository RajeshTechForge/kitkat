## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    YOUR FASTAPI SERVER (consumer)                       │
│                                                                         │
│  JWT Auth → UserContext → PydanticAI Agent → SSE Stream Response        │
│                                                                         │
│  from kitkat import LLMService, AnthropicProvider, LLMRequest           │
│  from kitkat.agents.builders import build_chat_agent                    │
│  from kitkat.agents.adapters.managed import ManagedModelAdapter         │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │  pip install kitkat[anthropic,agents]
┌──────────────────────────────▼──────────────────────────────────────────┐
│                    KITKAT (the library)                                 │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  AGENT LAYER (pydantic-ai optional extra)                        │   │
│  │  ManagedModelAdapter  BYOKModelAdapter  BaseAgentContext         │   │
│  │  build_chat_agent()   build_structured_agent()  ToolRegistry     │   │
│  └──────────────────────────────────┬───────────────────────────────┘   │
│                                     │                                   │
│  ┌──────────────────────────────────▼───────────────────────────────┐   │
│  │  WORKFLOW LAYER (langgraph optional extra)                       │   │
│  │  BaseWorkflow  ResearchWorkflow  (stateful graph-based tasks)    │   │
│  └──────────────────────────────────┬───────────────────────────────┘   │
│                                     │                                   │
│  ┌──────────────────────────────────▼───────────────────────────────┐   │
│  │  SERVICE LAYER (always installed)                                │   │
│  │  LLMService (managed registry)  BYOKLLMService (per-request)     │   │
│  └──────────────────────────────────┬───────────────────────────────┘   │
│                                     │                                   │
│  ┌──────────────────────────────────▼───────────────────────────────┐   │
│  │  PROVIDER LAYER (one extra per provider)                         │   │
│  │  AnthropicProvider  OpenAIProvider  GeminiProvider  + plugins    │   │
│  └──────────────────────────────────┬───────────────────────────────┘   │
│                                     │                                   │
│  ┌──────────────────────────────────▼───────────────────────────────┐   │
│  │  CORE LAYER (always installed — zero optional deps)              │   │
│  │  LLMRequest  LLMResponse  StreamChunk  TokenUsage  Message       │   │
│  │  LLMProvider ABC  RetryPolicy  ProviderCapabilities              │   │
│  │  Role  FinishReason  ProviderType  Full exception hierarchy      │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## What Goes in the Library and Where
```
core/        All data models, enums, exceptions — your base.py split by concern
abc/         LLMProvider abstract base class
providers/   AnthropicProvider, OpenAIProvider, GeminiProvider + Configs
service/     LLMService, BYOKLLMService, factory functions
agents/      ManagedModelAdapter, BYOKModelAdapter, BaseAgentContext,
             build_chat_agent(), build_structured_agent(), ToolRegistry
workflows/   BaseWorkflow ABC, ResearchWorkflow
plugins/     Entry-point registry and discovery
```