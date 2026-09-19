"""Pricing — pure table arithmetic. Tests build the service from a tiny
round-number table in tmp_path, so real-world rate changes never break them;
one separate test pins the COMMITTED table's patterns."""

from pathlib import Path

import pytest

from llm_gateway.observability.pricing import PricingService
from llm_gateway.schemas.openai_api import Usage

ROOT = Path(__file__).resolve().parents[2]


def write_table(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "pricing.yaml"
    lines = ["models:"]
    for row in rows:
        lines.append(
            f'  - {{pattern: "{row["pattern"]}", provider: test, '
            f"input_per_mtok: {row['in']}, output_per_mtok: {row['out']}}}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def usage(p: int = 100, c: int = 50) -> Usage:
    return Usage(prompt_tokens=p, completion_tokens=c, total_tokens=p + c)


def test_exact_arithmetic_with_round_numbers(tmp_path):
    table = write_table(tmp_path, [{"pattern": "m-", "in": 1.0, "out": 2.0}])
    estimate = PricingService(table).estimate("m-xl", usage(100, 50))
    assert estimate.priced
    assert estimate.cost_usd == pytest.approx(0.0002)  # 100/1M*1 + 50/1M*2


def test_first_match_wins_specific_before_generic(tmp_path):
    table = write_table(
        tmp_path,
        [
            {"pattern": "m-mini", "in": 0.5, "out": 1.0},
            {"pattern": "m-", "in": 5.0, "out": 10.0},
        ],
    )
    estimate = PricingService(table).estimate("m-mini-2024", usage(100, 0))
    assert estimate.cost_usd == pytest.approx(0.00005)  # the mini row, not the generic


def test_mock_is_priced_at_exactly_zero(tmp_path):
    table = write_table(tmp_path, [{"pattern": "mock", "in": 0.0, "out": 0.0}])
    estimate = PricingService(table).estimate("mock", usage(10, 10))
    assert estimate.priced and estimate.cost_usd == 0.0


def test_unmatched_model_is_unpriced_never_an_exception(tmp_path):
    table = write_table(tmp_path, [{"pattern": "m-", "in": 1.0, "out": 2.0}])
    estimate = PricingService(table).estimate("totally-unknown", usage(10, 10))
    assert estimate.priced is False and estimate.cost_usd == 0.0


def test_committed_table_prices_the_real_fleet():
    service = PricingService(ROOT / "data" / "pricing.yaml")
    # the workhorse
    e = service.estimate("openai/gpt-oss-120b", usage(10, 10))
    assert e.priced and e.cost_usd > 0
    # local + mock: priced at zero by definition
    assert service.estimate("llama3.2:1b", usage(10, 10)).priced
    assert service.estimate("mock", usage(10, 10)).priced
    # deliberately unpriced (null row) — the daily-driver Gemini until a rate lands
    assert service.estimate("gemini-3.7-flash", usage(10, 10)).priced is False
    # specific-before-generic ordering
    luna = service.estimate("gpt-5.6-luna-2026-05", usage(1_000_000, 0))
    assert luna.cost_usd == pytest.approx(0.20)
    # unknown → unpriced, never a silent zero
    assert service.estimate("no-such-model-anywhere", usage(10, 10)).priced is False


async def test_null_rate_row_is_matched_but_unpriced(tmp_path):
    path = tmp_path / "pricing.yaml"
    path.write_text(
        "models:\n"
        '  - {pattern: "m-pro", provider: test, input_per_mtok: null, output_per_mtok: null}\n',
        encoding="utf-8",
    )
    estimate = PricingService(path).estimate("m-pro-2026", usage(10, 10))
    assert estimate.priced is False and estimate.cost_usd == 0.0


def test_half_null_row_fails_loud_at_load(tmp_path):
    path = tmp_path / "pricing.yaml"
    path.write_text(
        "models:\n"
        '  - {pattern: "m-", provider: test, input_per_mtok: 1.0, output_per_mtok: null}\n',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="both rates or neither"):
        PricingService(path)
