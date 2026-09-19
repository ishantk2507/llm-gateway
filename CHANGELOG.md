### 0.5.0 — Phase 4: observability
- data/pricing.yaml + pricing.py: published per-Mtok rates (source-dated,provider-tagged, dormant openai/anthropic rows) × provider-reported usage.Unmatched models are never a silent zero — priced=false + a warning.
- Prometheus layer at /v1/metrics: requests by provider/tier/status/cache,cache hits by layer, spend, spend saved by cache, latency histogram.Instruments registered at module import (single-worker design — no multiproc).
- Request/cost persistence (SQLAlchemy, SQLite local / Postgres in compose):one row per request INCLUDING error rows, plus daily aggregates keyed(date, provider, tier). request_id correlates log line ↔ DB row ↔ metric.Observability failures never break the request path — enforced and tested.
- Middleware is the single observation point: one shared field dict →log line + metric + DB row, three projections of one record.
- Grafana dashboard (8 panels, auto-provisioned at :3001) — the breaker panelis reserved and empty until Phase 5, by design.
- Demo moment: one real Groq request logs a real cost_usd; the same requesttwice shows cost_saved_usd equal to what the first one spent; :9090 showsthe gateway target UP for the first time.

### 0.4.1 — Gemini adapter (third dialect)
- Native Gemini adapter behind the same ProviderAdapter seam — no route,router, or tier-mapping changes; the registry and data/tiers.yaml (whichalready listed gemini rows) activate it the moment GEMINI_API_KEY is set.
- Dialect translation in normalizer.py: system → systemInstruction,assistant → model role, parts structure, generationConfig mapping,finishReason/usageMetadata remapping. Model lives in the URL path.
- Safety-blocked responses (200 with no candidates) surface asProviderRejected — not retried, but the fallback chain can still serve.
- Zero routing-layer changes — the adapter pattern's payoff, third time.
- Demo moment: a real cloud provider behind the gateway at zero cost; killthe 
key and watch requests fall through to local/mock transparently.
- Agent compatibility: LangChain smoke test proving third-party harnesses
  route through the gateway unmodified; ADR-0008 (dynamic model catalog)
  and the Phase 6 agent-compatibility roadmap (streaming + tool passthrough)
  documented.


### 0.4.0 — Phase 3: cost-aware routing
- Rule-based tier classifier (ADR-0001): features → first-match rules indata/keywords.yaml (word-boundary keyword matching, token bounds, codeand structured-data signals). Every decision logs rule_fired.
- Tier→provider mapping in data/tiers.yaml: cost-ordered chains, emptytiers escalate up silently, total absence falls back loudly; dormantentries activate when keys/models register.
- Resolution order: X-Routing-Strategy header > explicit model pin >classifier. Model pins (gpt-/claude-/llama*/...) preserve 502 semantics.
- Local open-source models via Ollama/vLLM/LM Studio (LocalAdapter — anOpenAIAdapter named honestly), zero credentials, zero cost.
- Golden-set eval: labeling rubric + hand-labeled prompts + make evalwriting docs/ROUTER_EVAL.md with confusion matrix, per-tier P/R, macro F1,and misclassification analysis.
- Demo moment: "Hello, gateway." → cheap/greeting · "Explain the circuitbreaker pattern..." → standard/explain_concept · "Write a Pythonfunction..." → premium/code_task — each visible in x-routing-tier andthe log line's rule_fired.

### 0.3.0 — Phase 2: the two-layer cache
- Exact layer: SHA-256(normalized prompt + family + temp bucket) → Redis, TTL-native.
- Semantic layer: FAISS IndexIDMap2(IndexFlatIP), cosine ≥ 0.95, SQLite payloadstore, lazy expiry, capacity eviction, persist-on-write -(ADR-0005).
- Policy gates: single-turn, temperature ≤ 0.3, no tools, non-streaming (ADR-0003).Near-miss band [0.85, 0.95) logged as threshold calibration data.
- X-Cache: bypass header; POST /v1/cache/invalidate (by prompt_hash, bymodel_family, or flush); cache feature-flagged end to end.
- Adversarial cache-correctness suite against the REAL embedding model —similar prompts with different answers must never cross-hit.
- Demo moment: same question twice → second response in milliseconds at$0.0000; paraphrase hits semantically; "circuit breaker pattern" vs "homeelectrical wiring" correctly misses.

### 0.2.0 — Phase 1: multi-provider reality
- OpenAI + Anthropic adapters behind one ProviderAdapter seam; all dialecttranslation (system extraction, max_tokens defaulting, stop_sequences,response remapping) concentrated in providers/normalizer.py — theparameter whitelist is enforced there and contract-tested from the outside.
- Reliability layer: retries with full jitter (3 total attempts), per-attemptand total timeout budgets, ordered fallback chains. Adapters make exactlyone attempt — retry and fallback live above them.
- Failure semantics made mechanical: pinned-provider failure → 502;whole-chain exhaustion → 503 fail-loud (ADR-0004); budget death → 504.
- Contract tests with respx + committed fixtures — the suite stays offline,in CI and locally, at zero cost.
- Demo moment: kill the primary provider (dead port) under the officialOpenAI SDK — the request still succeeds via fallback; x-provider-usedshows who served it.

### 0.1.0 — Phase 0: the path works
- OpenAI-compatible /v1/chat/completions served end-to-end via MockProvider(ADR-0006) — zero credentials required.
- OpenAI-style error envelope on every failure path, including theFastAPI-validation 422 → 400 fix.
- One structured log line per request from request #1 (request_id, provider,tokens, latency, cost placeholder).
/v1/health with per-provider status; the four contract headers(x-cache-hit, x-provider-used, x-routing-tier, x-cost-usd) on everycompletion.
- Demo moment: point the official OpenAI SDK at localhost:8000/v1 with noAPI keys and get a completion back.


