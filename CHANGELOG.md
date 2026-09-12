0.1.0 — Phase 0: the path works
OpenAI-compatible /v1/chat/completions served end-to-end via MockProvider(ADR-0006) — zero credentials required.
OpenAI-style error envelope on every failure path, including theFastAPI-validation 422 → 400 fix.
One structured log line per request from request #1 (request_id, provider,tokens, latency, cost placeholder).
/v1/health with per-provider status; the four contract headers(x-cache-hit, x-provider-used, x-routing-tier, x-cost-usd) on everycompletion.
Demo moment: point the official OpenAI SDK at localhost:8000/v1 with noAPI keys and get a completion back.

