---
title: CLI Scripts & Batch Processing
description: Learn how to use KitKat in command-line interface (CLI) applications, interactive terminal REPLs, high-throughput batch processing scripts, and automated background jobs.
order: 2
---

This page is the complete reference for using KitKat in command-line interface (CLI) applications, interactive terminal REPLs, high-throughput batch processing scripts, and automated background jobs.


## Overview

KitKat's architecture supports both asynchronous (`asyncio`) and synchronous script execution models:

- **Async Script Pattern**: Ideal for high-throughput batch processing, interactive streaming REPLs, and concurrent multi-provider tasks.
- **Synchronous Execution (`run_sync`)**: Use `provider.run_sync(request)` or `service.run_sync(request, provider_type)` for quick one-off CLI scripts, legacy synchronous toolchains, or `click`/`argparse` commands.
- **Concurrency Throttling**: Use `asyncio.Semaphore` alongside `LLMService` retry logic to process thousands of prompts without exceeding provider rate limits or memory constraints.
- **Graceful Termination**: Intercept `SIGINT` (Ctrl+C) and `SIGTERM` signals to cleanly shutdown HTTP connection pools before exiting scripts.

---

## Synchronous Execution with `run_sync()`

If you are writing a simple CLI utility or working within a synchronous framework where managing an `asyncio` event loop creates unnecessary boilerplate, call `.run_sync()` on any provider or `LLMService` instance.

```python
import os
from kitkat import LLMRequest, Message, Role, ProviderType
from kitkat.providers.anthropic import AnthropicProvider, AnthropicConfig
from kitkat.service import create_llm_service


def main() -> None:
    # Instantiate the provider
    provider = AnthropicProvider(
        AnthropicConfig(api_key=os.environ["ANTHROPIC_API_KEY"])
    )

    request = LLMRequest(
        messages=[
            Message(role=Role.SYSTEM, content="You are a Unix CLI expert. Provide concise shell commands."),
            Message(role=Role.USER, content="How do I find all files larger than 100MB in /var/log?"),
        ],
        model="claude-opus-4-5",
        max_tokens=128,
    )

    # run_sync handles event loop creation, initialization, completion, and shutdown internally
    response = provider.run_sync(request)

    print("Suggested Command:")
    print(response.content)
    print(f"\nTokens used: {response.usage.total_tokens}")


if __name__ == "__main__":
    main()
```

> **📝 Note:** `run_sync()` automatically initializes the provider if it is not already initialized and cleans up connection resources after execution completes.

---

## Building a CLI Tool with `argparse`

This import-complete script demonstrates a versatile command-line utility accepting arguments for provider, model, prompt text, system prompt, temperature, and max tokens.

```python
import argparse
import asyncio
import os
import sys

from kitkat.service import create_llm_service
from kitkat import ProviderType, LLMRequest, Message, Role
from kitkat.providers.anthropic import AnthropicProvider, AnthropicConfig
from kitkat.providers.openai import OpenAIProvider, OpenAIConfig
from kitkat.providers.gemini import GeminiProvider, GeminiConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KitKat CLI: Unified LLM Command-Line Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("prompt", type=str, help="Prompt string or query for the model")
    parser.add_argument("-p", "--provider", choices=["anthropic", "openai", "gemini"], default="anthropic", help="LLM Provider")
    parser.add_argument("-m", "--model", type=str, default="", help="Model name (empty for provider default)")
    parser.add_argument("-s", "--system", type=str, default="You are a helpful CLI assistant.", help="System prompt")
    parser.add_argument("-t", "--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum completion tokens")
    parser.add_argument("--stream", action="store_true", help="Stream response tokens to stdout")
    return parser.parse_args()


async def run_cli() -> None:
    args = parse_args()
    provider_type = ProviderType(args.provider.lower())

    # Build provider map dynamically based on available API keys
    providers = {}
    if "ANTHROPIC_API_KEY" in os.environ:
        providers[ProviderType.ANTHROPIC] = AnthropicProvider(
            AnthropicConfig(api_key=os.environ["ANTHROPIC_API_KEY"])
        )
    if "OPENAI_API_KEY" in os.environ:
        providers[ProviderType.OPENAI] = OpenAIProvider(
            OpenAIConfig(api_key=os.environ["OPENAI_API_KEY"])
        )
    if "GOOGLE_API_KEY" in os.environ:
        providers[ProviderType.GEMINI] = GeminiProvider(
            GeminiConfig(api_key=os.environ["GOOGLE_API_KEY"])
        )

    if provider_type not in providers:
        print(f"Error: API key for provider '{args.provider}' is missing from environment.", file=sys.stderr)
        sys.exit(1)

    service = create_llm_service(providers)
    await service.initialize()

    request = LLMRequest(
        messages=[
            Message(role=Role.SYSTEM, content=args.system),
            Message(role=Role.USER, content=args.prompt),
        ],
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        stream=args.stream,
    )

    try:
        if args.stream:
            async for chunk in service.stream(request, provider_type):
                if not chunk.is_final:
                    print(chunk.delta, end="", flush=True)
                else:
                    print()  # Newline after stream finish
                    print(f"\n[Tokens: {chunk.usage.total_tokens} | Latency: {chunk.latency_ms:.0f}ms]", file=sys.stderr)
        else:
            response = await service.complete(request, provider_type)
            print(response.content)
            print(f"\n[Tokens: {response.usage.total_tokens} | Latency: {response.latency_ms:.0f}ms]", file=sys.stderr)
    finally:
        await service.shutdown()


def main() -> None:
    try:
        asyncio.run(run_cli())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
```

---

## Interactive Terminal REPL Chat

Build an interactive multi-turn terminal chat session with token tracking, slash commands (`/clear`, `/provider`, `/help`, `/quit`), and real-time streaming output.

```python
import asyncio
import os
import sys

from kitkat.service import create_llm_service, LLMService
from kitkat import ProviderType, LLMRequest, Message, Role
from kitkat.providers.anthropic import AnthropicProvider, AnthropicConfig
from kitkat.providers.openai import OpenAIProvider, OpenAIConfig


# ANSI Color Codes for terminal formatting
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


class ChatREPL:
    def __init__(self, service: LLMService) -> None:
        self.service = service
        self.active_provider = ProviderType.ANTHROPIC
        self.messages: list[Message] = [
            Message(role=Role.SYSTEM, content="You are a helpful, concise AI assistant.")
        ]
        self.total_tokens_used = 0

    def print_help(self) -> None:
        print(f"\n{BOLD}Available Commands:{RESET}")
        print("  /help             Display this help message")
        print("  /clear            Reset conversation history")
        print("  /provider <name>  Switch active provider (anthropic, openai)")
        print("  /tokens           Show session total token usage")
        print("  /quit or /exit    Exit the chat session\n")

    async def run(self) -> None:
        print(f"{BOLD}{GREEN}=== KitKat Interactive Terminal Chat ==={RESET}")
        print(f"Active Provider: {BOLD}{self.active_provider.value}{RESET}. Type '/help' for commands.")

        while True:
            try:
                user_input = input(f"\n{BOLD}{BLUE}You > {RESET}").strip()
            except (KeyboardInterrupt, EOFError):
                print(f"\n{YELLOW}Exiting chat session...{RESET}")
                break

            if not user_input:
                continue

            # Command handling
            if user_input.startswith("/"):
                parts = user_input.split()
                cmd = parts[0].lower()

                if cmd in ("/quit", "/exit"):
                    print(f"{YELLOW}Goodbye! Total session tokens: {self.total_tokens_used}{RESET}")
                    break
                elif cmd == "/help":
                    self.print_help()
                elif cmd == "/clear":
                    self.messages = [self.messages[0]]  # Preserve system message
                    print(f"{GREEN}Conversation history cleared.{RESET}")
                elif cmd == "/tokens":
                    print(f"{YELLOW}Session total tokens: {self.total_tokens_used}{RESET}")
                elif cmd == "/provider":
                    if len(parts) > 1 and parts[1].lower() in ("anthropic", "openai"):
                        self.active_provider = ProviderType(parts[1].lower())
                        print(f"{GREEN}Switched provider to: {self.active_provider.value}{RESET}")
                    else:
                        print(f"{YELLOW}Usage: /provider <anthropic|openai>{RESET}")
                else:
                    print(f"{YELLOW}Unknown command '{cmd}'. Type '/help' for command list.{RESET}")
                continue

            # Append user message to history
            self.messages.append(Message(role=Role.USER, content=user_input))

            # Prepare request
            request = LLMRequest(
                messages=self.messages,
                max_tokens=1024,
                temperature=0.4,
                stream=True,
            )

            print(f"{BOLD}{GREEN}AI ({self.active_provider.value}) > {RESET}", end="", flush=True)

            assistant_response_parts: list[str] = []
            try:
                async for chunk in self.service.stream(request, self.active_provider):
                    if not chunk.is_final:
                        assistant_response_parts.append(chunk.delta)
                        print(chunk.delta, end="", flush=True)
                    else:
                        print()
                        self.total_tokens_used += chunk.usage.total_tokens

                full_assistant_text = "".join(assistant_response_parts)
                self.messages.append(Message(role=Role.ASSISTANT, content=full_assistant_text))

            except Exception as exc:
                print(f"\n{YELLOW}Error during completion: {exc}{RESET}")


async def main() -> None:
    service = create_llm_service({
        ProviderType.ANTHROPIC: AnthropicProvider(
            AnthropicConfig(api_key=os.environ["ANTHROPIC_API_KEY"])
        ),
        ProviderType.OPENAI: OpenAIProvider(
            OpenAIConfig(api_key=os.environ["OPENAI_API_KEY"])
        ),
    })
    await service.initialize()

    repl = ChatREPL(service)
    try:
        await repl.run()
    finally:
        await service.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
```

---

## High-Throughput Batch Processing Script

When processing thousands of prompts from a JSONL file or database, use `asyncio.Semaphore` to throttle concurrency and prevent exceeding provider rate limits while keeping throughput high.

```python
import asyncio
import json
import os
import sys
import time
from typing import Any

from kitkat.service import create_llm_service, LLMService
from kitkat import ProviderType, LLMRequest, Message, Role, LLMError
from kitkat.providers.anthropic import AnthropicProvider, AnthropicConfig


# Concurrency Control Settings
MAX_CONCURRENT_REQUESTS = 10
INPUT_FILE = "prompts.jsonl"
OUTPUT_FILE = "results.jsonl"


async def process_single_item(
    semaphore: asyncio.Semaphore,
    service: LLMService,
    item: dict[str, Any],
) -> dict[str, Any]:
    """Process a single batch record inside a semaphore lock."""
    item_id = item.get("id", "unknown")
    prompt_text = item.get("prompt", "")

    async with semaphore:
        request = LLMRequest(
            messages=[Message(role=Role.USER, content=prompt_text)],
            max_tokens=256,
            temperature=0.1,
        )
        try:
            start_time = time.monotonic()
            response = await service.complete(request, ProviderType.ANTHROPIC)
            elapsed = time.monotonic() - start_time

            return {
                "id": item_id,
                "status": "success",
                "content": response.content,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "latency_ms": round(elapsed * 1000, 2),
            }
        except LLMError as exc:
            return {
                "id": item_id,
                "status": "error",
                "error_code": getattr(exc, "code", "LLM_ERROR"),
                "error_message": str(exc),
            }


async def run_batch() -> None:
    # Create sample input file if it doesn't exist
    if not os.path.exists(INPUT_FILE):
        print(f"Creating sample '{INPUT_FILE}'...", file=sys.stderr)
        sample_records = [
            {"id": f"task-{i}", "prompt": f"Write a one-sentence tip for Python topic #{i}."}
            for i in range(1, 21)
        ]
        with open(INPUT_FILE, "w", encoding="utf-8") as f:
            for rec in sample_records:
                f.write(json.dumps(rec) + "\n")

    # Load input records
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"Starting batch processing for {len(records)} records (max concurrency={MAX_CONCURRENT_REQUESTS})...")

    service = create_llm_service({
        ProviderType.ANTHROPIC: AnthropicProvider(
            AnthropicConfig(api_key=os.environ["ANTHROPIC_API_KEY"])
        )
    })
    await service.initialize()

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = [process_single_item(semaphore, service, item) for item in records]

    # Gather results concurrently
    start_all = time.monotonic()
    results = await asyncio.gather(*tasks)
    total_time = time.monotonic() - start_all

    # Write output JSONL
    success_count = 0
    total_tokens = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for res in results:
            if res["status"] == "success":
                success_count += 1
                total_tokens += res.get("prompt_tokens", 0) + res.get("completion_tokens", 0)
            f.write(json.dumps(res) + "\n")

    await service.shutdown()

    print(f"\nBatch Completed in {total_time:.2f}s!")
    print(f"Success: {success_count}/{len(records)} | Total Tokens: {total_tokens}")
    print(f"Output saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(run_batch())
```

---

## Scheduled Automation & Signal Handling

For automated cron jobs, systemd timers, or daemonized workers, handle `SIGINT` and `SIGTERM` signals cleanly to avoid leaving dangling HTTP sockets or corrupted state files.

```python
import asyncio
import logging
import os
import signal
import sys
from kitkat.service import create_llm_service, LLMService
from kitkat import ProviderType, LLMRequest, Message, Role
from kitkat.providers.anthropic import AnthropicProvider, AnthropicConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cron-job")


class ScheduledWorker:
    def __init__(self) -> None:
        self.service: LLMService | None = None
        self.running = True

    async def setup(self) -> None:
        self.service = create_llm_service({
            ProviderType.ANTHROPIC: AnthropicProvider(
                AnthropicConfig(api_key=os.environ["ANTHROPIC_API_KEY"])
            )
        })
        await self.service.initialize()
        logger.info("LLMService initialized successfully.")

    async def shutdown(self) -> None:
        logger.info("Shutting down worker resources...")
        self.running = False
        if self.service:
            await self.service.shutdown()
        logger.info("Shutdown complete.")

    async def run_task(self) -> None:
        if not self.service:
            raise RuntimeError("Worker not set up.")

        logger.info("Executing scheduled LLM maintenance task...")
        request = LLMRequest(
            messages=[
                Message(role=Role.SYSTEM, content="You are a system monitoring assistant."),
                Message(role=Role.USER, content="Generate a 3-bullet point system status report summary."),
            ],
            max_tokens=256,
        )

        response = await self.service.complete(request, ProviderType.ANTHROPIC)
        logger.info("Report Summary:\n%s", response.content)
        logger.info("Task completed. Tokens used: %d", response.usage.total_tokens)


async def main() -> None:
    worker = ScheduledWorker()
    await worker.setup()

    loop = asyncio.get_running_loop()

    # Register OS signal handlers for graceful shutdown
    def handle_signal(sig: int) -> None:
        logger.warning("Received signal %s. Initiating graceful shutdown...", signal.Signals(sig).name)
        asyncio.create_task(worker.shutdown())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: handle_signal(s))

    try:
        await worker.run_task()
    except Exception as exc:
        logger.error("Job failed with exception: %s", exc, exc_info=True)
        await worker.shutdown()
        sys.exit(1)
    else:
        await worker.shutdown()
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Further Reading

- [FastAPI Integration](./fastapi.md) — Integrating KitKat into web APIs
- [Error Handling](./error-handling.md) — Handling `LLMRateLimitError` and `LLMTimeoutError`
- [Providers Overview](./providers.md) — Service-layer configuration and models
- [BYOK Guide](./byok.md) — Command-line tools with user-supplied API keys
- [API Reference — Service](./api-reference/service.md) — Complete `LLMService` API documentation
