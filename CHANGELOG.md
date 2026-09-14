0.1.0 — Phase 0: the path works
-OpenAI-compatible /v1/chat/completions served end-to-end via MockProvider(ADR-0006) — zero credentials required.
-OpenAI-style error envelope on every failure path, including theFastAPI-validation 422 → 400 fix.
-One structured log line per request from request #1 (request_id, provider,tokens, latency, cost placeholder).
/v1/health with per-provider status; the four contract headers(x-cache-hit, x-provider-used, x-routing-tier, x-cost-usd) on everycompletion.
-Demo moment: point the official OpenAI SDK at localhost:8000/v1 with noAPI keys and get a completion back.

0.2.0 — Phase 1: multi-provider reality
-OpenAI + Anthropic adapters behind one ProviderAdapter seam; all dialecttranslation (system extraction, max_tokens defaulting, stop_sequences,response remapping) concentrated in providers/normalizer.py — theparameter whitelist is enforced there and contract-tested from the outside.
-Reliability layer: retries with full jitter (3 total attempts), per-attemptand total timeout budgets, ordered fallback chains. Adapters make exactlyone attempt — retry and fallback live above them.
-Failure semantics made mechanical: pinned-provider failure → 502;whole-chain exhaustion → 503 fail-loud (ADR-0004); budget death → 504.
-Contract tests with respx + committed fixtures — the suite stays offline,in CI and locally, at zero cost.
-Demo moment: kill the primary provider (dead port) under the officialOpenAI SDK — the request still succeeds via fallback; x-provider-usedshows who served it.

0.3.0 — Phase 2: the two-layer cache
-Exact layer: SHA-256(normalized prompt + family + temp bucket) → Redis, TTL-native.
-Semantic layer: FAISS IndexIDMap2(IndexFlatIP), cosine ≥ 0.95, SQLite payloadstore, lazy expiry, capacity eviction, persist-on-write -(ADR-0005).
-Policy gates: single-turn, temperature ≤ 0.3, no tools, non-streaming (ADR-0003).Near-miss band [0.85, 0.95) logged as threshold calibration data.
-X-Cache: bypass header; POST /v1/cache/invalidate (by prompt_hash, bymodel_family, or flush); cache feature-flagged end to end.
-Adversarial cache-correctness suite against the REAL embedding model —similar prompts with different answers must never cross-hit.
-Demo moment: same question twice → second response in milliseconds at$0.0000; paraphrase hits semantically; "circuit breaker pattern" vs "homeelectrical wiring" correctly misses.

