---
title: Installation
description: Requirements and install steps for Quarry.
---

Kitkat uses an opt-in extras model. The core package is small and dependency-free; provider SDKs are installed only when you ask for them.

**Requirements**

| Requirement      | Version               |
| ---------------- | --------------------- |
| Python           | 3.11 or newer         |
| Operating system | Linux, macOS, Windows |

```bash
# Anthropic Claude only
pip install kitkat[anthropic]

# OpenAI (and OpenAI-compatible endpoints)
pip install kitkat[openai]

# Google Gemini (including Vertex AI)
pip install kitkat[gemini]

# Redis cache backend (for multi-process / multi-instance deployments)
pip install kitkat[redis]

# All three providers at once
pip install kitkat[all-providers]

# Everything (all providers + Redis)
pip install kitkat[all]
```

**Using `uv`?**

```bash
uv add kitkat[all]
```
