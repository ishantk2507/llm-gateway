Router golden-set labeling rubric:
The standard: the minimum tier that answers this prompt well — not the tierthat answers it best, the cheapest one that answers it well.

Tiers
->cheap — small local models (1–3B). One-shot factual lookups, greetings,spelling, single-sentence translation, anything answerable in a shortparagraph with no domain depth. "What is the capital of France?"
->standard — mid-tier cloud or larger local models. Concept explanations,summarization, moderate rewrites, single-step how-does-X-work. A competentgeneral answer with no multi-step reasoning chain. "Explain the circuitbreaker pattern."
->premium — frontier models. Code writing/debugging/refactoring, proofs,derivations, multi-step plans, trade-off analysis, system design, longstructured outputs. Anything where being wrong is expensive.

Edge rules
1. Code → premium regardless of length (syntax errors are failures).
2. One-line factual → cheap even if the topic is technical.
3. Ambiguous → standard (never cheapa — cheap-and-wrong beats standard-and-slowin cost, not in value).
4. Multi-step ANYTHING (plans, step-by-step, derivations) → premium.
5. Label what the prompt NEEDS, not what a great model might volunteer.