"""Cost estimation from the versioned pricing table (DESIGN.md §5.5).

Hard rules, enforced here:
- ``estimate()`` reads the PROVIDER-REPORTED usage block, never an estimate.
  The Day-4 live finding — tokens_in=75 for "Hello, gateway." — makes token
  estimation a billing fiction.
- An unmatched model is NEVER a silent zero: cost 0.0 with ``priced=False``
  and a once-per-model warning (ADR-0008's rule, in code). Metrics and logs
  carry the flag so the gap is visible until a row is added.
- The table is committed data: a missing file is a hard boot failure, same
  rule as the router's YAML files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from llm_gateway.observability.logging import get_logger
from llm_gateway.schemas.openai_api import Usage

logger = get_logger(__name__)


@dataclass(frozen=True)
class CostEstimate:
    cost_usd: float
    priced: bool


@dataclass(frozen=True)
class _PriceRow:
    pattern: str
    provider: str
    input_per_mtok: float | None
    output_per_mtok: float | None


class PricingService:
    def __init__(self, pricing_path: Path) -> None:
        path = Path(pricing_path)
        if not path.exists():
            raise RuntimeError(
                f"pricing table missing: {path} — committed data; restore it from git"
            )
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        self._rows = []
        for entry in raw.get("models", []):
            in_rate = entry["input_per_mtok"]
            out_rate = entry["output_per_mtok"]
            if (in_rate is None) != (out_rate is None):
                raise RuntimeError(  # fail loud: half-priced rows are typos
                    f"pricing.yaml: {entry['pattern']!r} must have both rates or neither"
                )
            self._rows.append(
                _PriceRow(
                    pattern=str(entry["pattern"]),
                    provider=str(entry.get("provider", "unknown")),
                    input_per_mtok=None if in_rate is None else float(in_rate),
                    output_per_mtok=None if out_rate is None else float(out_rate),
                )
            )
        self._warned: set[str] = set()

    def estimate(self, model: str, usage: Usage) -> CostEstimate:
        """Published-rate cost of one completion. First pattern match wins."""
        for row in self._rows:
            if model.startswith(row.pattern):
                if row.input_per_mtok is None:
                    # Matched a deliberately-unpriced row (null rates): the
                    # model is known, its rate is unavailable or enterprise-
                    # only. A decision, not an oversight — no "add a row" nudge.
                    return CostEstimate(0.0, False)
                cost = (
                    usage.prompt_tokens / 1_000_000 * row.input_per_mtok
                    + usage.completion_tokens / 1_000_000 * row.output_per_mtok
                )
                return CostEstimate(round(cost, 6), True)
        if model not in self._warned:  # once per model, not per request
            self._warned.add(model)
            logger.warning(
                "unpriced_model",
                model=model,
                note="cost recorded as 0.0 with priced=false — add a row to data/pricing.yaml",
            )
        return CostEstimate(0.0, False)
