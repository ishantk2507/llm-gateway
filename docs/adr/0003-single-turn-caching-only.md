ADR-0003: Cache single-turn requests only
Status: Accepted · Date: 2025-09-14

Context
Multi-turn follow-ups ("and why?", "do the same for X") near-match the tailsof unrelated conversations stored in a semantic index. That is the #1correctness hazard for this cache — it turns an optimization into awrong-answer machine.

Decision
Only single-turn requests (exactly one user message, optionally one systemmessage) are ever cached — by either layer. Multi-turn always routes live.

Consequences
Conversational traffic can never cross-contaminate.− Lower hit rate on conversational workloads — documented and accepted.
The gate lives in one pure function (cache/policy.is_cacheable), testedexhaustively and cheaply.