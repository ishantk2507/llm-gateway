"""Request/cost persistence — the queryable history (DESIGN.md §5.5, §8).

SQLAlchemy on a SYNC engine, every call wrapped in asyncio.to_thread — the
same pattern as the semantic cache's SQLite: the event loop never blocks on
DB I/O. SQLite locally, Postgres in compose (both configured since Day 1).

Governing principle, ENFORCED IN CODE: an observability failure must never
break the request path. log_request catches its own exceptions and warns —
the plane that watches cannot be the plane that kills. (The middleware
additionally guards: belt-and-suspenders.)

request_id arrives from the logging middleware's shared scope dict, so a log
line, a DB row, and a Grafana point correlate 1:1 — three projections of one
record.

The daily upsert (select-then-update inside one thread call) is race-free
BECAUSE the app is single-worker by design (DESIGN.md §13) — an old decision
paying off. Multi-worker requires a real upsert (ON CONFLICT) or an
aggregation job; that is the documented Phase 6 scaling path.

Error rows ARE persisted, with the `error` column — per DESIGN.md §8's
RequestLog model. A 502/503/504 lands in Postgres with provider='unknown'
when the request died before any provider served it.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime

from sqlalchemy import Date, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from llm_gateway.observability.logging import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    pass


class RequestLogRow(Base):
    __tablename__ = "request_logs"

    request_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    timestamp: Mapped[datetime]
    model: Mapped[str | None] = mapped_column(String(128))
    provider: Mapped[str | None] = mapped_column(String(32))
    tier: Mapped[str | None] = mapped_column(String(16))
    rule_fired: Mapped[str | None] = mapped_column(String(64))
    cache_hit: Mapped[bool] = mapped_column(default=False)
    cache_layer: Mapped[str | None] = mapped_column(String(16))
    similarity: Mapped[float | None]
    tokens_in: Mapped[int | None]
    tokens_out: Mapped[int | None]
    cost_usd: Mapped[float] = mapped_column(default=0.0)
    cost_saved_usd: Mapped[float] = mapped_column(default=0.0)
    latency_ms: Mapped[float]
    status_code: Mapped[int]
    error: Mapped[str | None] = mapped_column(String(512))


class CostRecordRow(Base):
    __tablename__ = "cost_records"

    day: Mapped[date] = mapped_column(Date, primary_key=True)
    provider: Mapped[str] = mapped_column(String(32), primary_key=True)
    tier: Mapped[str] = mapped_column(String(16), primary_key=True)
    total_cost_usd: Mapped[float] = mapped_column(default=0.0)
    total_saved_usd: Mapped[float] = mapped_column(default=0.0)
    request_count: Mapped[int] = mapped_column(default=0)


class RequestRepository:
    def __init__(self, database_url: str) -> None:
        self._engine = create_engine(database_url)
        self._sessionmaker = sessionmaker(self._engine, expire_on_commit=False)
        Base.metadata.create_all(self._engine)

    async def log_request(self, **fields: object) -> None:
        """Persist one request row + daily aggregate. Never raises."""
        try:
            await asyncio.to_thread(self._log_sync, fields)
        except Exception as exc:
            logger.warning("repository_persist_failed", error=repr(exc))

    def _log_sync(self, f: dict) -> None:
        with self._sessionmaker() as session:
            session.add(
                RequestLogRow(
                    request_id=str(f["request_id"]),
                    timestamp=f["timestamp"],
                    model=f.get("model"),
                    provider=f.get("provider"),
                    tier=f.get("tier"),
                    rule_fired=f.get("rule_fired"),
                    cache_hit=bool(f.get("cache_hit", False)),
                    cache_layer=f.get("cache_layer"),
                    similarity=f.get("similarity"),
                    tokens_in=f.get("tokens_in"),
                    tokens_out=f.get("tokens_out"),
                    cost_usd=float(f.get("cost_usd") or 0.0),
                    cost_saved_usd=float(f.get("cost_saved_usd") or 0.0),
                    latency_ms=float(f.get("latency_ms") or 0.0),
                    status_code=int(f.get("status_code") or 0),
                    error=f.get("error"),
                )
            )
            key_day = f["timestamp"].date()
            key_provider = str(f.get("provider") or "unknown")
            key_tier = str(f.get("tier") or "unknown")
            record = session.scalars(
                select(CostRecordRow).where(
                    CostRecordRow.day == key_day,
                    CostRecordRow.provider == key_provider,
                    CostRecordRow.tier == key_tier,
                )
            ).first()
            if record is None:
                record = CostRecordRow(
                    day=key_day,
                    provider=key_provider,
                    tier=key_tier,
                    total_cost_usd=0.0,
                    total_saved_usd=0.0,
                    request_count=0,
                )
                session.add(record)
            record.total_cost_usd += float(f.get("cost_usd") or 0.0)
            record.total_saved_usd += float(f.get("cost_saved_usd") or 0.0)
            record.request_count += 1
            session.commit()

    def request_rows(self) -> list[RequestLogRow]:
        """Test/diagnostic accessor (sync, by design — call from tests only)."""
        with self._sessionmaker() as session:
            return list(session.scalars(select(RequestLogRow)).all())

    def cost_record(self, day: date, provider: str, tier: str) -> CostRecordRow | None:
        with self._sessionmaker() as session:
            return session.get(CostRecordRow, (day, provider, tier))

    async def aclose(self) -> None:
        await asyncio.to_thread(self._engine.dispose)
