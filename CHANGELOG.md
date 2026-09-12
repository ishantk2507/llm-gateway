0.1.0 — Phase 0: the path works
OpenAI-compatible /v1/chat/completions served end-to-end via MockProvider(ADR-0006) — zero credentials required.
OpenAI-style error envelope on every failure path, including theFastAPI-validation 422 → 400 fix.
One structured log line per request from request #1 (request_id, provider,tokens, latency, cost placeholder).
/v1/health with per-provider status; the four contract headers(x-cache-hit, x-provider-used, x-routing-tier, x-cost-usd) on everycompletion.
Demo moment: point the official OpenAI SDK at localhost:8000/v1 with noAPI keys and get a completion back.

0.2.0 — Phase 1: multi-provider reality
OpenAI + Anthropic adapters behind one ProviderAdapter seam; all dialecttranslation (system extraction, max_tokens defaulting, stop_sequences,response remapping) concentrated in providers/normalizer.py — theparameter whitelist is enforced there and contract-tested from the outside.
Reliability layer: retries with full jitter (3 total attempts), per-attemptand total timeout budgets, ordered fallback chains. Adapters make exactlyone attempt — retry and fallback live above them.
Failure semantics made mechanical: pinned-provider failure → 502;whole-chain exhaustion → 503 fail-loud (ADR-0004); budget death → 504.
Contract tests with respx + committed fixtures — the suite stays offline,in CI and locally, at zero cost.
Demo moment: kill the primary provider (dead port) under the officialOpenAI SDK — the request still succeeds via fallback; x-provider-usedshows who served it.

