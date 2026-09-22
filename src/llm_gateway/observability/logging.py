"""Structured request logging — one line per request, from request #1 (DESIGN.md §5.5).

How fields travel from route to the final log line:

- ``RequestLoggingMiddleware`` creates a plain dict per request and stores it
  in the ASGI scope (reachable by exception handlers via
  ``request.state.log_fields``) and in a ContextVar.
- Route code calls ``bind_log_fields(provider_used=..., ...)`` which writes
  into that same dict (plus structlog contextvars, so mid-request logs also
  carry the fields).
- When the response leaves, the middleware emits exactly one line: the dict
  plus ``status_code`` and ``latency_ms``.

The dict-sharing is load-bearing: contextvars *set* inside the route's task do
not propagate back up to middleware, but mutating a shared dict does.

Day 5 — the middleware is also the single OBSERVATION point. After the
response, for every completion request (gated on ``model`` in the field
dict, so health checks and /v1/metrics never pollute the counters):

  1. metrics.observe_request(...) — every completion, including 5xx
     (provider reads as "unknown" when the request died before one served)
  2. repository.log_request(...) — persisted rows, error rows included

Both read the same shared dict the route populated: log line, metric, and DB
row are three projections of one record, correlated by request_id.
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = structlog.get_logger(__name__)

_request_fields: ContextVar[dict[str, Any] | None] = ContextVar(
    "llm_gateway_request_fields", default=None
)


def configure_logging(level: str = "INFO", *, pretty: bool | None = None) -> None:
    """Configure structlog once at startup.

    ``pretty=None`` means "console renderer when stderr is a TTY" — readable
    local development, JSON everywhere else (CI, Docker, production), which is
    what DESIGN.md §5.5 actually requires.
    """
    if pretty is None:
        pretty = sys.stderr.isatty()
    renderer = structlog.dev.ConsoleRenderer() if pretty else structlog.processors.JSONRenderer()
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> Any:
    """Thin re-export so the codebase imports logging from exactly one place."""
    return structlog.get_logger(name)


def bind_log_fields(**fields: Any) -> None:
    """Attach fields to the current request's final log line.

    Safe outside a request (e.g. startup code): it then only binds to
    structlog's contextvars, which never reaches a request line.
    """
    structlog.contextvars.bind_contextvars(**fields)
    bound = _request_fields.get()
    if bound is not None:
        bound.update(fields)


def _emit(fields: dict[str, Any], start: float, status_code: int | None = None) -> None:
    record = {**fields, "latency_ms": round((time.perf_counter() - start) * 1000, 1)}
    if status_code is not None:
        record["status_code"] = status_code
    logger.info("http_request", **record)


async def _observe_and_persist(
    app: Any, fields: dict[str, Any], latency_ms: float, status_code: int
) -> None:
    """Metrics + DB row for one completion request. Never breaks the response."""
    from llm_gateway.observability import (
        metrics,  # local import: keeps logging importable standalone
    )

    metrics.observe_request(
        provider=str(fields.get("provider_used", "unknown")),
        tier=str(fields.get("routing_tier", "unknown")),
        status=str(status_code),
        cache_hit=bool(fields.get("cache_hit", False)),
        layer=fields.get("cache_layer"),
        latency_ms=latency_ms,
        cost_usd=float(fields.get("cost_usd") or 0.0),
        saved_usd=float(fields.get("cost_saved_usd") or 0.0),
    )

    repository = getattr(app.state, "repository", None)
    if repository is None:
        return
    try:
        await repository.log_request(
            request_id=fields["request_id"],
            timestamp=datetime.now(UTC),
            model=fields.get("model"),
            provider=fields.get("provider_used"),
            tier=fields.get("routing_tier"),
            rule_fired=fields.get("rule_fired"),
            cache_hit=bool(fields.get("cache_hit", False)),
            cache_layer=fields.get("cache_layer"),
            similarity=fields.get("similarity_score"),
            tokens_in=fields.get("tokens_in"),
            tokens_out=fields.get("tokens_out"),
            cost_usd=float(fields.get("cost_usd") or 0.0),
            cost_saved_usd=float(fields.get("cost_saved_usd") or 0.0),
            latency_ms=round(latency_ms, 1),
            status_code=status_code,
            error=fields.get("error"),
        )
    except Exception as exc:
        # The repository guards itself; this is belt-and-suspenders for
        # injected test doubles that raise deliberately.
        logger.warning("repository_persist_failed", error=repr(exc))


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """One structured line per HTTP request — plus metrics + persistence."""

    async def dispatch(self, request: Request, call_next):  # noqa: ANN001
        fields: dict[str, Any] = {
            "request_id": uuid.uuid4().hex,
            "method": request.method,
            "path": request.url.path,
        }
        # The scope dict is shared by reference down the whole ASGI chain, so
        # exception handlers can also write into it via request.state.log_fields.
        request.scope.setdefault("state", {})["log_fields"] = fields
        token = _request_fields.set(fields)
        structlog.contextvars.bind_contextvars(**fields)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            fields["error"] = "unhandled_exception"
            _emit(fields, start)
            raise
        finally:
            structlog.contextvars.clear_contextvars()
            _request_fields.reset(token)

        latency_ms = (time.perf_counter() - start) * 1000
        _emit(fields, start, response.status_code)

        if "model" in fields:  # completion requests only
            await _observe_and_persist(request.app, fields, latency_ms, response.status_code)
        return response
