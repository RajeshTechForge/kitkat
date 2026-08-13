---
title: Structured Outputs
description: Kitkat's `build_structured_agent` function creates a PydanticAI agent that returns a validated Pydantic model instead of a raw string. Learn how to design output schemas, handle validation retries, implement custom validators, and stream structured output.
order: 3
---

Kitkat's `build_structured_agent` function creates a PydanticAI agent that returns a **validated Pydantic model** instead of a raw string. The LLM is instructed to produce JSON, PydanticAI parses and validates it against your schema, and the result is a fully typed Python object — with automatic retries when the model produces malformed output.

This page covers `build_structured_agent`, designing output schemas, validation retries, custom validators, and streaming structured output.

## Installation

```bash
pip install kitkat[agents]
```

## Why Structured Output?

Raw LLM responses are unstructured strings. Structured output guarantees:

- **Type safety** — your IDE and type checker know exactly what fields exist.
- **Schema enforcement** — the LLM is guided by the JSON schema derived from your Pydantic model.
- **Automatic retries** — when the model produces invalid JSON or fails Pydantic validation, the agent automatically retries (up to `output_retries` times) with the validation error appended to the context.
- **Custom validation** — plug in domain-specific rules (field ranges, cross-field constraints, business logic) via a validator function.

## Quick Start

```python
import asyncio
import os

from pydantic import BaseModel, Field
from kitkat.service import create_llm_service
from kitkat import ProviderType
from kitkat.providers.openai import OpenAIProvider, OpenAIConfig
from kitkat.agents import ManagedModelAdapter, BaseAgentContext, build_structured_agent


class SentimentResult(BaseModel):
    sentiment: str = Field(description="One of: positive, negative, neutral")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    reasoning: str = Field(description="One-sentence explanation of the classification")


async def main() -> None:
    service = create_llm_service({
        ProviderType.OPENAI: OpenAIProvider(
            OpenAIConfig(api_key=os.environ["OPENAI_API_KEY"])
        )
    })
    await service.initialize()

    adapter = ManagedModelAdapter(
        service=service,
        provider_type=ProviderType.OPENAI,
        default_model="gpt-4o-mini",
    )

    agent = build_structured_agent(
        model=adapter,
        output_type=SentimentResult,
        context_type=BaseAgentContext,
    )

    ctx = BaseAgentContext(user_id="user-001")
    result = await agent.run(
        "Classify the sentiment: 'This library is an absolute joy to work with!'",
        deps=ctx,
    )

    output: SentimentResult = result.data
    print(f"Sentiment:   {output.sentiment}")
    print(f"Confidence:  {output.confidence:.0%}")
    print(f"Reasoning:   {output.reasoning}")
    # Sentiment:   positive
    # Confidence:  97%
    # Reasoning:   The phrase "absolute joy" is strongly positive.

    await service.shutdown()


asyncio.run(main())
```

## `build_structured_agent`

```python
from kitkat.agents import build_structured_agent

agent = build_structured_agent(
    model=adapter,
    output_type=SentimentResult,
    context_type=BaseAgentContext,
    system_prompt="",
    output_retries=1,
    validator=None,
)
```

### Parameters

| Parameter        | Type              | Default            | Description                                                                                                                                                                                                                                         |
| ---------------- | ----------------- | ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model`          | `Model`           | —                  | **Required.** A `ManagedModelAdapter` or `BYOKModelAdapter` instance.                                                                                                                                                                               |
| `output_type`    | `type[BaseModel]` | —                  | **Required.** A Pydantic `BaseModel` subclass. The LLM output is validated against this schema.                                                                                                                                                     |
| `context_type`   | `type[ContextT]`  | `BaseAgentContext` | The `deps_type` for the agent. Pass your application's context subclass here so tools are fully typed.                                                                                                                                              |
| `system_prompt`  | `str`             | `""`               | Static system prompt. When empty, the default prompt is used: `"You are a helpful AI assistant. Always respond in valid JSON matching the requested schema."`                                                                                       |
| `output_retries` | `int`             | `1`                | Number of times PydanticAI retries when the model output fails Pydantic validation. Each retry appends the validation error to the conversation, giving the model a chance to correct itself. Default is `1` (one retry after the initial failure). |
| `validator`      | `Callable         | None`              | `None`                                                                                                                                                                                                                                              | Optional callable following the pydantic-ai v2.x `output_validator` protocol. Called after successful Pydantic parsing for additional domain validation. |

### Returns

`Agent[ContextT, BaseModel]` — a configured PydanticAI agent ready for `.run()`.

## Designing Output Schemas

Use Pydantic v2 field annotations to guide the LLM with precise descriptions and constraints. The JSON schema derived from your model is included in the prompt sent to the provider.

```python
from pydantic import BaseModel, Field
from typing import Literal


class ArticleSummary(BaseModel):
    title: str = Field(
        description="The main topic of the article in 5–10 words",
        min_length=5,
        max_length=80,
    )
    key_points: list[str] = Field(
        description="Exactly three key takeaways as concise bullet points",
        min_length=3,
        max_length=3,
    )
    difficulty: Literal["beginner", "intermediate", "advanced"] = Field(
        description="Estimated reading difficulty level"
    )
    estimated_read_minutes: int = Field(
        ge=1, le=120,
        description="Estimated reading time in minutes"
    )
```

> **💡 Tip:** Always add `description` to every field. The description is included in the JSON schema the model sees and significantly improves output quality. Generic field names without descriptions lead to inconsistent or hallucinated values.

## Validation Retries

When the LLM produces output that fails Pydantic validation, PydanticAI automatically sends a follow-up message with the validation error and asks the model to correct its output. The number of correction attempts is controlled by `output_retries`.

```python
agent = build_structured_agent(
    model=adapter,
    output_type=ArticleSummary,
    output_retries=2,   # Up to 2 correction attempts after initial failure
)
```

**Example retry flow:**

1. Agent sends the prompt.
2. Model returns `{"title": "Python", "key_points": ["Fast"], "difficulty": "easy", "estimated_read_minutes": 5}`.
3. Pydantic validation fails: `key_points` has only 1 item (min 3), `difficulty` is `"easy"` (not in Literal).
4. Agent sends the error back to the model: _"Output validation failed: key_points must have at least 3 items; difficulty must be one of 'beginner', 'intermediate', 'advanced'."_
5. Model corrects its output.
6. Pydantic validation passes. `result.data` is a valid `ArticleSummary`.

> **📝 Note:** Each retry is a separate LLM call and counts against your token usage. For production use cases with strict cost budgets, keep `output_retries=1` (the default) and invest in clear schema descriptions and system prompts to reduce first-pass failures.

## Custom Validators

For validation logic that goes beyond Pydantic schema constraints — cross-field rules, external lookups, business logic — pass a `validator` callable to `build_structured_agent`.

The validator follows pydantic-ai's `output_validator` protocol: it receives a `RunContext` and the parsed model instance, and either returns the (optionally modified) instance or raises `ModelRetry` to trigger another attempt with a custom error message.

```python
import asyncio
import os

from pydantic import BaseModel, Field
from pydantic_ai import RunContext, ModelRetry

from kitkat.agents import build_structured_agent, ManagedModelAdapter, BaseAgentContext
from kitkat.service import create_llm_service
from kitkat import ProviderType
from kitkat.providers.anthropic import AnthropicProvider, AnthropicConfig


class PriceEstimate(BaseModel):
    item: str = Field(description="The item being priced")
    low_usd: float = Field(ge=0, description="Lower bound of price estimate in USD")
    high_usd: float = Field(ge=0, description="Upper bound of price estimate in USD")
    reasoning: str = Field(description="Brief explanation of the estimate")


async def validate_price_range(
    ctx: RunContext[BaseAgentContext],
    estimate: PriceEstimate,
) -> PriceEstimate:
    # Cross-field constraint: low must be less than high.
    if estimate.low_usd >= estimate.high_usd:
        raise ModelRetry(
            f"low_usd ({estimate.low_usd}) must be strictly less than "
            f"high_usd ({estimate.high_usd}). Please correct the price range."
        )
    # Domain constraint: spread must be at least 10% of the midpoint.
    midpoint = (estimate.low_usd + estimate.high_usd) / 2
    spread = estimate.high_usd - estimate.low_usd
    if spread < 0.1 * midpoint:
        raise ModelRetry(
            f"The price range is suspiciously narrow ({spread:.2f} USD spread on a "
            f"{midpoint:.2f} USD midpoint). Please widen the estimate to reflect uncertainty."
        )
    return estimate


async def main() -> None:
    service = create_llm_service({
        ProviderType.ANTHROPIC: AnthropicProvider(
            AnthropicConfig(api_key=os.environ["ANTHROPIC_API_KEY"])
        )
    })
    await service.initialize()

    adapter = ManagedModelAdapter(
        service=service,
        provider_type=ProviderType.ANTHROPIC,
        default_model="claude-opus-4-5",
    )

    agent = build_structured_agent(
        model=adapter,
        output_type=PriceEstimate,
        context_type=BaseAgentContext,
        output_retries=2,
        validator=validate_price_range,
    )

    ctx = BaseAgentContext(user_id="user-001")
    result = await agent.run(
        "Estimate the price of a used 2019 MacBook Pro 16-inch.",
        deps=ctx,
    )

    estimate: PriceEstimate = result.data
    print(f"Item:      {estimate.item}")
    print(f"Range:     ${estimate.low_usd:,.0f} – ${estimate.high_usd:,.0f}")
    print(f"Reasoning: {estimate.reasoning}")

    await service.shutdown()


asyncio.run(main())
```

## Nested and Complex Schemas

Pydantic's full model composition features are available. Nested models, `list` fields, `Optional` fields, and `Literal` unions all work as expected.

```python
from pydantic import BaseModel, Field
from typing import Optional


class CodeReviewComment(BaseModel):
    line_number: int = Field(ge=1, description="Line number in the file")
    severity: str = Field(description="One of: error, warning, suggestion")
    message: str = Field(description="Concise description of the issue")
    suggested_fix: Optional[str] = Field(
        default=None,
        description="A corrected version of the line, or null if no fix is suggested",
    )


class CodeReview(BaseModel):
    overall_quality: str = Field(
        description="One of: excellent, good, needs_improvement, poor"
    )
    summary: str = Field(
        description="Two-sentence summary of the code quality"
    )
    comments: list[CodeReviewComment] = Field(
        description="List of specific line-level issues found in the code"
    )
    approved: bool = Field(
        description="True if the code is ready to merge, False if changes are required"
    )


agent = build_structured_agent(
    model=adapter,
    output_type=CodeReview,
    context_type=BaseAgentContext,
    system_prompt=(
        "You are an expert Python code reviewer. "
        "Analyse the code thoroughly and return a structured review."
    ),
    output_retries=2,
)

ctx = BaseAgentContext(user_id="reviewer-01")
result = await agent.run(
    "Review this Python function:\n\ndef add(a, b):\n    return a+b\n",
    deps=ctx,
)

review: CodeReview = result.data
print(f"Approved: {review.approved}")
print(f"Quality:  {review.overall_quality}")
for comment in review.comments:
    print(f"  Line {comment.line_number} [{comment.severity}]: {comment.message}")
```

## BYOK with Structured Output

`build_structured_agent` works identically with `BYOKModelAdapter`:

```python
from kitkat.agents import BYOKModelAdapter, build_structured_agent, BaseAgentContext
from kitkat.service import BYOKLLMService
from kitkat import ProviderType

async def classify_with_byok(user_key: str, text: str) -> SentimentResult:
    ctx = BaseAgentContext(user_id="user-42")
    async with BYOKLLMService(ProviderType.OPENAI, user_key, "gpt-4o-mini") as byok:
        adapter = BYOKModelAdapter(byok_service=byok)
        agent = build_structured_agent(
            model=adapter,
            output_type=SentimentResult,
            context_type=BaseAgentContext,
        )
        result = await agent.run(f"Classify: {text!r}", deps=ctx)
    return result.data
```

## Accessing Usage and Metadata

The `RunResult` returned by `agent.run()` exposes token usage and model metadata alongside `result.data`:

```python
result = await agent.run("Summarise this document...", deps=ctx)

output: ArticleSummary = result.data
print(f"Title: {output.title}")

# Usage is aggregated across all attempts (including retries).
usage = result.usage()
print(f"Input tokens:  {usage.input_tokens}")
print(f"Output tokens: {usage.output_tokens}")
```

## Further Reading

- [Agent Layer Overview](./index.md) — Architecture and the two adapters
- [Agent Context](./context.md) — `BaseAgentContext`, `RoutingTier`, subclassing
- [Tool Calling](./tools.md) — Adding tools to structured agents
- [API Reference — Agents](../api-reference/agents.md) — Complete API surface
