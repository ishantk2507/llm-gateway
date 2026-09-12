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
not propagate back up to middleware, but mutating a shared dict does. Every
later phase adds fields (``similarity_score`` Day 3, ``rule_fired`` Day 4)
instead of touching this pipeline.
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from contextvars import ContextVar
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


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """One structured line per HTTP request — the seed of DESIGN.md §5.5."""

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
        _emit(fields, start, response.status_code)
        return response
