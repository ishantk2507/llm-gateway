# Benchmarks 

> Day 6 fills the full methodology and results — latency, cache economics,
> threshold sweep. Committed now: preliminary live observations, so that
> ADR-0007 cites a repo artifact rather than a chat memory.

## Preliminary live observations — cache similarity bands (Day 4, single session)

| Observation | Cosine |
|---|---|
| Adversarial pairs (different correct answers) — observed range | ~0.85–0.92 |
| Paraphrase pairs (same answer) — observed range | ~0.93–0.97 |
| **False positive — ADR-0007's founding incident:** "reverses a linked list" vs "reverses a doubly linked list" | **0.96** |
| Typo variant ("doubely") vs the same stored prompt | **0.99** — higher than the correct spelling |

Implication: the adversarial and paraphrase bands overlap, so no single
threshold separates them — threshold tuning trades precision for recall
rather than solving correctness. These values are single-session and
live-observed; they will be superseded by the Day 6 threshold sweep
(`benchmarks/threshold_sweep.py`).