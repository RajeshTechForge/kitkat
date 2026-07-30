---
title: Introduction
description: What KitKat is and a 60-second example.
category: Getting Started
order: 1
---

Kitkat gives you a single, consistent interface to all providers with streaming, BYOK (Bring Your Own Key), extended thinking, and typed responses that work identically across every provider. You can switch provider by changing two lines. Your request, response, and error handling stay exactly the same.

## Why Kitkat?

Every major LLM SDK has a different API, different streaming protocol, different error shapes, and different retry semantics. Switching providers means rewriting request code, stream parsers, and error handlers.

Kitkat solves this with a **thin, typed abstraction layer** that:

- Lets you swap providers without touching business logic
- Ships a real async-first design — not a sync wrapper with `asyncio.run`
- Stays minimal — install only the providers you actually use
- Is built to be extended — a clear ABC makes writing custom providers trivial
- Fails loudly and precisely — every error maps to a specific, typed exception
