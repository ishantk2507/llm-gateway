
# ADR-0007: Equivalence-aware semantic cache verification

**Status:** Accepted (L1 implementation scheduled Day 6) · **Date:** 2025-XX-XX

## Context

A live session on Day 4 produced the cache's first observed false positive:

> Stored: "Write a Python function that reverses a linked list."
> Query:  "Write a Python function that reverses a **doubly** linked list."
> Served from the semantic cache at cosine **0.96** — above the 0.95 threshold.
> One adjective changed the correct answer. A typo'd variant ("doubely")
> scored **0.99** against the same stored prompt — higher than the correct
> spelling. See [BENCHMARKS.md](../BENCHMARKS.md), preliminary observations.

The structural cause: a single embedding-similarity score cannot encode all
dimensions of answer-equivalence. The observed bands overlap — adversarial
pairs (different answers) ~0.85–0.92, paraphrases (same answer) ~0.93–0.97 —
so threshold tuning trades precision against recall rather than solving
correctness. ADR-0002's adversarial suite and the near-miss calibration band
exist precisely because of this overlap.

Options considered:
- (a) raise the threshold — trades false positives for misses, solves nothing;
- (b) cheap deterministic verification before serving a hit;
- (c) learned pairwise verification (cross-encoder) on ambiguous candidates;
- (d) LLM-as-judge in the hot path — circular (an LLM call to avoid an LLM
  call) and latency-hostile. Rejected for the hot path; retained offline.

## Decision

Cosine similarity remains the **candidate-retrieval** mechanism; it is no
longer the final correctness decision. The semantic layer becomes:

    exact → embedding retrieval → deterministic verification → hit / miss

**L1 — deterministic verifier (Day 6).** Two checks, pure string ops (~1ms):

1. **Constraint tokens.** Negation markers (`not`, `never`, `without`,
   `avoid`) and typed values (years, numbers) extracted from both prompts.
   Mismatch → MISS, logged explainably (`guard_block: year 2023≠2024`).
2. **Rare-word containment.** Rare content words in the query must appear in
   the stored prompt (`doubly` absent → MISS). "Rare" is corpus frequency
   against `evals/golden_set.jsonl` (words in >N% of prompts are common) —
   no new dependency. Deliberately softened from full content-word
   containment: "Which city is the capital of the US?" vs "What is the
   capital of the United States?" is a legitimate paraphrase that full
   containment would reject. `not` is never a stopword in this guard.

Guard refusals log as a near-miss subtype (`reason=lexical_guard`), so the
guard measures itself in production — every refusal is a counted
would-have-been false positive.

**Selection by measurement.** Variants (constraints-only, full containment,
rare-word + constraints) run as a bake-off against the adversarial and
paraphrase suites before one is wired in. Prediction to beat: constraints-only
scores zero on the founding incident — `doubly` is an adjective, not an
entity, number, or negation.

**L2 — Query Equivalence Verifier (future, gated).** A cross-encoder reading
both prompts jointly. Runs ONLY on the near-miss band [floor, threshold) —
where cosine is genuinely ambiguous — never on clear hits, preserving
millisecond hit latency. Named "Query Equivalence Verifier", not NLI
verifier: mutual entailment is not answer-equivalence for facts, numbers,
and time-sensitive queries.

**Offline.** LLM-as-judge is legitimate for dataset construction: the
near-miss log stream → judged equivalence pairs → golden dataset →
train/evaluate the verifier. The gateway's logs are thus a training-data
generator for both the verifier v2 and a learned router v2 (cf.
ulab-uiuc/LLMRouter) — the deterministic v1 produces the labeled data the
learned v2 needs.

## Consequences

+ The `doubly` class of false positives is blocked deterministically and
  explainably; hit latency preserved (string ops, no model).
+ The verifier is designed to be **subsumed, not grown**: constraint parsing
  beyond regex-able tokens (comparison targets, output types) belongs to L2.
+ Guard refusals are logged and counted — its value is measurable in
  production and in the Day 6 threshold sweep.
− The payload store gains a `prompt_text` column; entries written before it
  skip the guard (fail-open) until TTL ages them out.
− Legitimate paraphrases introducing rare words are rejected — a hit-rate
  cost in the correctness-safe direction: a miss costs a live call, a bad
  hit costs a wrong answer (ADR-0004's asymmetry, applied to the cache).
− A long stored prompt can lexically contain a short query's vocabulary by
  coincidence; the cosine gate mostly mitigates. Noted for v2.
− Two more code paths in the cache layer; both pure functions, both
  independently testable.



