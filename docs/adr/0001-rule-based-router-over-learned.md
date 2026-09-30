ADR-0001: Rule-based router, not a learned classifier
Status: Accepted · Date: 2025-09-15

Context
Every request needs a tier (cheap/standard/premium). Options: (a) learnedclassifier, (b) LLM-as-judge, (c) deterministic rules over extracted features.

Decision
Rule-based classifier: token count, code-block detection, reasoning keywords,structured-data detection. Rules live in data/keywords.yaml — data, notcode; the day's iteration loop is editing YAML, never code. Every decisionreturns and logs rule_fired, so any routing outcome is explainable in oneline during an incident. Tier→provider mapping lives in data/tiers.yaml,with unregistered entries dormant until keys/models exist.

Consequences
Every routing outcome is explainable; no training data, no drift, noreproducibility problems.
Evaluated against a hand-labeled golden set with a written rubric —confusion matrix and per-tier P/R in docs/ROUTER_EVAL.md.
Swappable behind TierClassifier; a learned v2 trains on the logs thissystem already produces.− Rules underfit: short-but-genuinely-hard prompts get the cheap tier("Explain gravity."), multi-step phrasings without trigger keywords getstandard ("Create a study plan..."). Mitigated by the X-Routing-Strategyoverride header and by tracking misclassification in the eval — the knownerrors are enumerated in ROUTER_EVAL.md, not hidden.