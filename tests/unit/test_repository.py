"""Repository — rows, daily aggregates, and the never-break-the-path guard."""

from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from llm_gateway.observability.repository import RequestRepository


def row_fields(request_id: str, **overrides) -> dict:
    base = dict(
        request_id=request_id,
        timestamp=datetime.now(UTC),
        model="auto",
        provider="groq",
        tier="standard",
        rule_fired="explain_concept",
        cache_hit=False,
        cache_layer=None,
        similarity=None,
        tokens_in=100,
        tokens_out=50,
        cost_usd=0.0002,
        cost_saved_usd=0.0,
        latency_ms=123.4,
        status_code=200,
        error=None,
    )
    base.update(overrides)
    return base


async def test_request_rows_and_daily_aggregates(tmp_path):
    repo = RequestRepository(f"sqlite:///{tmp_path}/obs.db")
    await repo.log_request(**row_fields("r1"))
    await repo.log_request(
        **row_fields(
            "r2",
            provider="cache",
            tier="cache",
            cache_hit=True,
            cache_layer="exact",
            cost_usd=0.0,
            cost_saved_usd=0.0002,
        )
    )

    rows = repo.request_rows()
    assert len(rows) == 2
    assert {r.request_id for r in rows} == {"r1", "r2"}
    assert rows[0].cost_usd == 0.0002 and rows[0].latency_ms == 123.4

    day = datetime.now(UTC).date()
    groq_record = repo.cost_record(day, "groq", "standard")
    assert groq_record is not None
    assert groq_record.request_count == 1
    assert groq_record.total_cost_usd == pytest.approx(0.0002)

    cache_record = repo.cost_record(day, "cache", "cache")
    assert cache_record.total_saved_usd == pytest.approx(0.0002)  # the money aggregate

    await repo.aclose()


async def test_persist_failure_never_raises(tmp_path):
    repo = RequestRepository(f"sqlite:///{tmp_path}/obs.db")
    # Sabotage AFTER construction: engine pointed at an unopenable path
    # (missing parent dir — sqlite refuses to create directories).
    repo._engine = create_engine(f"sqlite:///{tmp_path}/no/such/dir/x.db")
    repo._sessionmaker = sessionmaker(repo._engine)
    await repo.log_request(**row_fields("r3"))  # must swallow and warn, not raise


# noqa: E402 — kept at bottom so the module reads top-down
