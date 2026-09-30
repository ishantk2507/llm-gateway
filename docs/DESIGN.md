
---

## 3. `docs/DESIGN.md` — the complete, corrected design doc

````markdown
# LLM Gateway — System Design

**Status:** v1 (implemented)
**Version:** 1.0

---

## 1. Problem statement

Teams calling LLM APIs directly end up with three recurring problems: no unified
interface across providers (OpenAI/Anthropic/local all differ), no cost control
(every call hits a paid API even for near-duplicate prompts), and no visibility
into latency, cost, or failure behavior in production.

LLM Gateway is a single proxy between clients and LLM providers. One
OpenAI-compatible API surface, a two-layer semantic cache that avoids redundant
paid calls, a deterministic router that sends each request to the cheapest model
capable of handling it, and an observability layer that makes cost, latency, and
reliability visible.

## 2. Goals

- Single OpenAI-compatible API surface across providers (OpenAI, Anthropic, local vLLM/Ollama, mock)
- Cut redundant spend via a two-layer cache (exact-match + FAISS semantic)
- Route each request to the cheapest sufficient model via a deterministic classifier
- Full observability: latency, cost, cache hit rate, error rate, per-provider health
- Production reliability: retries, circuit breakers, provider fallback chains
- CPU-only operation; embedding model footprint stays well inside a 6GB VRAM budget if a GPU is present

## 3. Non-goals (v1)

- No fine-tuning or model training
- No RL or online learning in the router — routing is deterministic by design
- No multi-tenant billing
- No agentic orchestration — request/response proxy, not a planner
- No streaming — `stream: true` returns a documented 400 in v1 (see §7.3)

## 4. High-level architecture

```
Client ──POST /v1/chat/completions──▶ API Gateway (FastAPI)
                                        auth · rate limit · validation
                                        │
                     ┌── exact-match cache (SHA-256, Redis) ── hit ──▶ respond
                     │
                     ▼
              embed prompt (MiniLM, thread executor)
                     │
              FAISS semantic search (cosine ≥ threshold)
                     │ miss
                     ▼
              Complexity Router (rule-based → tier)
                     │
              Provider Adapter Layer
              circuit breaker · retry · fallback chain
                     │
              Response Normalizer → OpenAI schema
              write-through cache update · metrics emit
                     │
                     ▼
              Client + response headers (x-cache-hit, x-provider-used,
                                        x-routing-tier, x-cost-usd)
```

## 5. Component design

### 5.1 API Gateway layer

- FastAPI exposing `/v1/chat/completions` and `/v1/embeddings`, schema-compatible
  with the official OpenAI client so existing SDKs work unmodified.
- Auth via `Authorization: Bearer <key>`; keys are stored hashed (SHA-256) and
  checked against a Redis-backed token bucket for per-key rate limiting.
- Request validation via Pydantic **before** the request reaches cache or router.

**Model field semantics.** The OpenAI SDK requires a `model` field, but the
router decides the actual model. Resolution order:

| `model` value | Behavior |
|---|---|
| `auto` | Router assigns tier (default) |
| `cheap` / `standard` / `premium` | Tier pinned, router skipped |
| Real model name (e.g. `gpt-4o`) | Pinned directly to that provider, no routing |
| `mock` | MockProvider — deterministic, zero cost |

**Parameter whitelist.** v1 supports `temperature`, `max_tokens`, `top_p`,
`stop`, and the system message. These are mapped per provider (e.g.
`max_tokens` → Anthropic `max_tokens`, Gemini `maxOutputTokens`). Unknown
parameters are dropped with a logged warning rather than blind-forwarded —
silent parameter drift across providers is a correctness bug, not a convenience.

**Streaming stance.** `stream: true` returns `400` with a machine-readable error
body explaining v1 is non-streaming. It never crashes, never silently
degrades to non-streaming. SSE passthrough is Phase 6.

### 5.2 Semantic cache (two layers)

**Layer 1 — exact match.** `sha256(normalized_prompt + model_family +
temperature_bucket)` → response, stored in Redis. Checked *before* embedding.
100% safe, ~1ms, and carries most of the measured hit rate.

**Layer 2 — semantic.** Prompt embedding → FAISS `IndexIDMap2` wrapping
`IndexFlatIP` over L2-normalized vectors (cosine similarity). FAISS stores
vectors only; response payloads live in a SQLite/Postgres table keyed by
`faiss_id`, so entries can be expired and deleted. Index persisted via
`faiss.write_index` on write.

- Embedding model: `all-MiniLM-L6-v2` (384-dim), run in a thread executor so the
  event loop never blocks.
- Similarity threshold: default cosine ≥ 0.95, tunable via env.
- **Near-miss band (0.85–0.95) is logged separately** — accumulated calibration
  data for the threshold instead of guessing once. A benchmark sweep
  (`benchmarks/threshold_sweep.py`) plots false-positive rate vs. threshold over
  an adversarial prompt set.

**Cacheability policy (all must hold):**
- Single-turn only: exactly one user message, no conversation history.
  Multi-turn follow-ups near-match tails of unrelated conversations — the top
  correctness hazard. v1 always routes multi-turn live.
- `temperature ≤ 0.3` — creative/sampling requests are never cached; a
  "similar" prompt shouldn't return someone else's random sample.
- No tool/function calls — semantics depend on live state, not prompt text.
- Non-streaming requests only.

- TTL-based expiry plus manual invalidation by exact `prompt_hash`, by
  `model_family`, or flush-all.

### 5.3 Complexity router

- v1: deterministic rule-based classifier. Features: token count, code-block
  presence, multi-step reasoning keywords ("explain step by step", "prove",
  "derive"), structured-data presence. Rules live in `data/keywords.yaml` —
  data, not code.
- Tiers: **cheap** → **standard** → **premium**.
- Every decision logs `rule_fired` — any routing outcome is explainable in one
  line during an incident or an interview.
- Evaluated against a labeled golden set (~150 prompts) with a written labeling
  rubric: "the minimum tier that answers this prompt well." Confusion matrix and
  per-tier precision/recall are tracked in `docs/ROUTER_EVAL.md`.
- Always overridable via `X-Routing-Strategy` header — the router is a default,
  not a gate.
- v2 (Phase 6): supervised classifier trained offline on labeled examples,
  swapped in behind the same `TierClassifier` interface. Still deterministic at
  inference.

**Provider selection within a tier.** Each tier maps to an ordered provider
list (ordered by cost within tier); the first provider with a closed circuit
breaker serves the request; the rest of the list is the fallback chain.

### 5.4 Provider adapter layer

- Common `ProviderAdapter` protocol: `complete()`, `embed()`, `health_check()`.
- Concrete adapters: `OpenAIAdapter`, `AnthropicAdapter`, `MockAdapter` (and a
  stretch `GeminiAdapter` / `LocalAdapter` for vLLM/Ollama).
- **MockProvider** is a first-class adapter with configurable latency and error
  rate. It powers contract-free integration tests, chaos tests, load tests, and
  the zero-credential demo mode.
- Per-provider circuit breaker (closed → open → half-open), exponential backoff
  with full jitter on retries, configurable fallback chains.
- **Idempotency (gateway-side).** Provider completion APIs don't honor
  idempotency keys, so dedup happens at the gateway: `X-Idempotency-Key` →
  response cached in Redis (short TTL); a retried request with the same key
  replays the stored response instead of re-billing.

### 5.5 Observability pipeline

- Structured JSON log per request: latency, tokens in/out, `cost_usd`,
  `cache_hit`, `similarity_score`, `provider_used`, `routing_tier`, `rule_fired`,
  error (if any).
- Prometheus counters/histograms exposed at `/v1/metrics`; request logs and cost
  records persisted to Postgres (SQLite for local dev).
- Grafana dashboard: p50/p95/p99 latency by provider and tier, cost per day,
  **cost saved via cache hits**, cache hit rate over time, routing tier
  distribution, error rate, circuit breaker state per provider.
- Cost estimation from a versioned pricing table (`data/pricing.yaml`), stamped
  on every response as `x-cost-usd` and aggregated daily.

## 6. Request lifecycle

1. Client sends an OpenAI-compatible request.
2. Gateway authenticates, rate-limits, validates.
3. **Exact-match cache** check → hit returns immediately (`x-cache-hit: true`).
4. Miss → prompt embedded (thread executor) → FAISS search → hit returns with
   similarity score logged; near-miss (0.85–0.95) logged for calibration.
5. Miss → router assigns tier (logs `rule_fired`).
6. Adapter layer picks the first healthy (closed-breaker) provider for the tier,
   executes with retry/timeout/breaker protection; on failure walks the
   fallback chain transparently.
7. Response normalized to OpenAI schema, cache updated (write-through),
   metrics + logs emitted.
8. Client receives response plus `x-cache-hit`, `x-provider-used`,
   `x-routing-tier`, `x-cost-usd` headers.

## 7. API specification

| Endpoint | Method | Purpose |
|---|---|---|
| `/v1/chat/completions` | POST | Main proxy endpoint, OpenAI-compatible |
| `/v1/embeddings` | POST | Passthrough embeddings (also used internally) |
| `/v1/cache/invalidate` | POST | Evict by `prompt_hash`, by `model_family`, or flush-all |
| `/v1/health` | GET | Gateway + per-provider health + breaker states |
| `/v1/metrics` | GET | Prometheus scrape endpoint |

### 7.1 Request headers (custom, optional)

| Header | Purpose |
|---|---|
| `X-Routing-Strategy` | `auto` (default) \| `cheap` \| `standard` \| `premium` — overrides the router |
| `X-Cache` | `auto` (default) \| `bypass` — force a live call |
| `X-Idempotency-Key` | Gateway-side response dedup for retries |

### 7.2 Response headers (always present)

| Header | Meaning |
|---|---|
| `x-cache-hit` | `true` / `false` |
| `x-provider-used` | e.g. `anthropic`, `openai`, `mock` |
| `x-routing-tier` | `cheap` \| `standard` \| `premium` |
| `x-cost-usd` | Estimated cost of this request ($0.0000 on cache hits) |

### 7.3 Error codes

| Code | Meaning |
|---|---|
| 400 | Validation failure · `stream: true` in v1 · non-cacheable + forced cache |
| 401 | Missing/invalid API key |
| 429 | Rate limit exceeded (Redis token bucket) |
| 502 | Provider errored after all retries and fallbacks exhausted |
| 503 | All providers down — explicit failure (see ADR-0004) |
| 504 | Total request timeout budget exceeded |

## 8. Data models

```
RequestLog
  id: uuid
  timestamp: datetime
  prompt_hash: str
  provider_used: str
  routing_tier: str
  rule_fired: str | null
  cache_hit: bool
  similarity_score: float | null      # logged on semantic hits and near-misses
  near_miss: bool
  latency_ms: int
  tokens_in: int
  tokens_out: int
  cost_usd: float
  error: str | null

CacheEntry (payload store row)
  faiss_id: int64                     # maps to vector in IndexIDMap2
  prompt_hash: str
  model_family: str
  temperature_bucket: str             # "low" (≤0.3) — the only cached bucket
  response: json
  created_at: datetime
  expires_at: datetime

IdempotencyRecord (Redis, TTL)
  key: str
  response: json

ProviderConfig
  name: str
  base_url: str
  api_key_ref: str                    # secret reference, never the raw key
  tier_mapping: str[]                 # tiers this provider serves, cost-ordered
  circuit_state: enum(closed, open, half_open)

RoutingDecision
  request_id: uuid
  features: json                      # token_count, has_code, reasoning_keywords, ...
  tier_selected: str
  rule_fired: str                     # which rule made the call

CostRecord
  date: date
  provider: str
  tier: str
  total_cost_usd: float
  total_saved_by_cache_usd: float
```

## 9. Reliability & failure handling

- **Retries:** max 3 attempts per provider, exponential backoff with full jitter.
- **Timeout budgets:** total request budget (default 30s) with per-attempt
  sub-budget (default 10s) — one slow provider can't consume the whole request.
- **Circuit breaker:** opens at ≥ 50% failures over the last 20 requests, stays
  open for a 30s cooldown, then half-opens to probe recovery.
- **Fallback chain:** each tier has a cost-ordered provider list; on breaker
  trip or exhaustion, the next provider is tried transparently.
- **Degradation policy — fail loud.** If every provider is down, the gateway
  returns an explicit 503 with a machine-readable error. It does *not* silently
  serve near-match cache entries that look healthy but may be semantically
  wrong. An opt-in config (`CACHE_ONLY_DEGRADED_MODE=true`) serves
  exact-match entries only, clearly flagged via response headers (ADR-0004).

## 10. Security

- Gateway API keys hashed at rest; provider credentials from env vars/secrets
  manager, never committed or logged.
- Per-key rate limits enforced before any provider call.
- PII-redaction hook interface exists but is a no-op in v1 (documented future work).

## 11. Tech stack

| Layer | Choice | Why |
|---|---|---|
| API framework | FastAPI | Async-native, Pydantic validation, OpenAPI docs for free |
| Vector index | FAISS (`IndexIDMap2` + `IndexFlatIP`) | Fast, correct, no external service; documented IVF+PQ upgrade path |
| Embedding model | `all-MiniLM-L6-v2` | 384-dim, CPU-viable, no GPU requirement |
| Rate limiting / ephemeral state | Redis | Token bucket, TTL storage, exact-cache + idempotency records |
| Logs / cost records | Postgres (SQLite local) | Queryable history for the dashboard |
| Metrics | Prometheus + Grafana | Standard, interview-legible observability stack |
| Test HTTP boundary | respx | Contract tests with recorded fixtures — no live calls, no quota burn |
| Load testing | Locust | Characterize latency under concurrency |
| Deployment | Docker Compose | Reproducible local + demo environment |

## 12. Testing strategy

- **Unit tests** per component — router, cache policy, breaker, backoff, normalizer.
- **Contract tests** against provider adapters using recorded HTTP fixtures
  (respx) — the suite runs offline in CI in seconds.
- **Router accuracy eval** against the labeled golden set; precision/recall per
  tier tracked over time (`docs/ROUTER_EVAL.md`).
- **Cache-correctness tests** — adversarial pairs of similar prompts with
  different correct answers must never cross-hit. This is the failure mode that
  matters most for a cache layer.
- **Chaos tests** — simulate provider outage; assert fallback chain and breaker
  transitions behave as designed.
- **Load testing** (Locust) against MockProvider for deterministic throughput
  characterization.

## 13. Deployment

- Docker Compose: gateway + Redis + Postgres + Prometheus + Grafana, dashboard
  auto-provisioned.
- Single stateless gateway container; all config via environment variables.
- Embedding model pre-baked into the image — no runtime downloads, fast cold start.
- **Single uvicorn worker by design**: in-memory FAISS index and breaker state
  stay coherent. Multi-worker deployment requires shared state (Redis-backed
  breaker state, sharded or service-backed index) — documented as the Phase 6
  scaling path, not silently assumed.

## 14. Roadmap

| Phase | Scope | Proves |
|---|---|---|
| 0 | Passthrough proxy to MockProvider + OpenAI adapter, OpenAI-compatible in/out, request logging from request #1 | The real request path works end-to-end |
| 1 | Anthropic adapter + normalizer + fallback chain + retries + timeout budgets | Multi-provider abstraction holds up |
| 2 | Two-layer semantic cache behind a feature flag | Cost savings with zero semantic drift |
| 3 | Rule-based router + golden-set eval | Cost-aware routing without ML risk |
| 4 | Observability pipeline + Grafana dashboard | System behavior is visible, not assumed |
| 5 | Circuit breakers, rate limiting, auth, chaos tests | Production-shaped, not demo-shaped |
| 6 (stretch) | Streaming passthrough, learned router v2 (same interface), Gemini adapter, K8s | Depth beyond core scope |


| Phase 6 additions | Scope | Proves |
|---|---|---|
| Agent compatibility | SSE streaming passthrough; tool-call passthrough (request side) + `tool_calls` response normalization; documented agent-harness contract | The gateway serves real agent loops, not just simple chains |
| Dynamic model catalog (ADR-0008) | `list_models()` per adapter, vendored metadata, snapshot-for-reproducibility, generated tiers | Routing scales to the provider ecosystem without manual maintenance |
## 15. Design rationale

- **Systems thinking:** cache, router, adapters, observability are independent,
  individually testable, individually swappable components.
- **Reliability engineering:** circuit breakers, fallback chains, and timeout
  budgets are the same primitives used in trading infrastructure.
- **Cost discipline:** caching + tiered routing turn "we called an LLM" into a
  measurable cost-optimization problem.
- **Determinism over cleverness:** the router is rule-based so its behavior is
  debuggable in an incident, not a black box.
- **Observability-first:** every request is logged with enough structure to
  answer "why did this cost X" or "why was this slow" without guessing.
- **Fail loud over silent degradation:** a gateway that returns
  plausible-but-wrong answers while looking healthy is a liability, not a feature.

## 16. Decision log

See [`docs/adr/`](adr/) — every significant decision is recorded as an ADR with
context, decision, and consequences.
````

---

