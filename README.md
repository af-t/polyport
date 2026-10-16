# Polyport

Polyport is a multi-account LLM reverse proxy written in Rust.

It aggregates accounts across different AI providers and exposes standard OpenAI (`/v1/chat/completions`) and Anthropic (`/v1/messages`) API endpoints, allowing AI coding tools, CLI agents, and IDE extensions to access these models transparently.

## What It Does

- **Unified Upstream Routing:** Routes requests to multiple AI backends, including Google Cloud Code (Gemini models and Claude tiers), Anthropic direct API, and OpenAI/ChatGPT Codex endpoints.
- **Multi-Account Pooling & Failover:** Manages account pools with automatic token refresh, LRU load distribution, rate-limit cooldown isolation, and multi-turn session affinity.
- **Protocol Translation:** Bridges client and upstream schemas in real time, translating SSE streaming chunks and tool calls between Gemini, Codex, Anthropic, and OpenAI formats.
- **Transparent Response Caching:** Automatically caches turn completions and thought signatures to reduce redundant upstream queries.
- **Optional Workspace Addons:** Provides an optional sandboxed agent execution loop and embedded web interface, while remaining compilable as a lean, headless proxy.
