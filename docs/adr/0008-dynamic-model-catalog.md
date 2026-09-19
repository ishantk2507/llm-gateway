# ADR-0008: Dynamic model catalog — discovered, priced, snapshotted

**Status:** Accepted (implementation scheduled Phase 6) · **Date:** 2025-XX-XX

## Context

v1's tier→model knowledge is hand-maintained: `data/tiers.yaml` lists
providers, `model_family()` prefix-matches model strings, and pricing
(Phase 5) will be a hand-written table. New models — and there are dozens
per quarter — require manual edits, and the free-tier ecosystem the gateway
targets (Groq, Gemini, Ollama) rotates fastest.

The goal: the gateway automatically discovers every model every registered
provider offers, paid and unpaid, classifies each into a tier with cost data,
and routes on it — without manual catalog maintenance.

## Decision

**1. Discovery at deploy time.** Every adapter implements `list_models()`:
OpenAI-compatible hosts via `GET /models` (covers Groq, Ollama, OpenAI),
Anthropic/Gemini via their model-list endpoints. A `catalog.py` builder runs
at startup, unions the results, and tags each model with its provider —
which makes **cache family assignment by construction** rather than prefix
guessing: family = provider, because discovery said so.

**2. Classification from vendored metadata, with heuristics fallback.**
A model ID alone carries no tier or price. Three sources, in order:
   a. `data/model_catalog.yaml` — vendored, seeded once from OpenRouter's
      public `/api/v1/models` endpoint (hundreds of models with pricing and
      context windows; used as a metadata source, not a routing hop), plus
      provider docs. Committed, versioned, reviewable.
   b. Name heuristics for unmatched models: flash/mini/8b → cheap;
      pro/o1/70b+ → premium; unmatched → standard.
   c. Unmatched models boot-log loudly and default to standard — never
      silently.
   Never an unpriced model: catalog entries without pricing data get cost
   estimated as `null` and are flagged in metrics rather than billed at 0.

**3. Snapshot for reproducibility.** The discovered catalog is written to a
committed snapshot at build time and routing runs against the snapshot —
the same discipline as docs/ROUTER_EVAL.md. "Which model served this
request" must never depend on what a provider launched yesterday. Discovery
is refreshed deliberately (a `make refresh-catalog` target), not continuously.

**4. Routing compatibility.** The tier→model mapping remains data
(`tiers.yaml`), now generated from the catalog rather than hand-written;
per-tier model selection within a single provider (gemini-flash vs
gemini-pro) becomes expressible, closing the documented Phase 6 gap.

## Consequences

+ New provider models appear without code changes; free-tier rotation is
  absorbed by a catalog refresh.
+ Cache family isolation stops being string-guessing — one class of
  cross-provider cache bugs eliminated by construction.
+ Pricing (Phase 5's pricing.yaml) shares the catalog's model registry,
  so cost attribution and routing read one source of truth.
− One more moving part at startup; mitigated by snapshotting and by
  hard failure when a registered provider's model list is unreachable
  (fail loud, same as the Redis startup ping).
− Heuristic misclassification routes a capable model to a lower tier or
  vice versa; bounded by the escape hatch (`X-Routing-Strategy`) and
  visible in routing metrics.
− The catalog is eventually stale between refreshes — accepted; staleness
  is visible (snapshot date) and deliberate (reproducibility).