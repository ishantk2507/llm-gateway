ADR-0002: Two-layer cache — exact first, then semantic
Status: Accepted · Date: 2025-09-14

Context
Near-duplicate prompts waste spend, but semantic matching risks returning aplausible-but-wrong answer. Options: semantic-only, exact-only, or both.

Decision
Layer 1: SHA-256(normalized prompt + family + temp bucket) → Redis — ~1ms,100% correct, carries most of the hit rate. Layer 2: FAISS cosine ≥ 0.95,exists ONLY for paraphrase. Exact is always checked first; the near-miss band([0.85, 0.95)) is logged as calibration data for the threshold.

Consequences
Exact hits are free and provably safe; the risky layer can be disabled(GW_CACHE__SEMANTIC_ENABLED=false) without losing most of the savings.
Each layer is independently testable; the adversarial suite exercises thecomposed service.− Two code paths; mitigated by a shared identity hash (policy.prompt_hash)so invalidation addresses both.