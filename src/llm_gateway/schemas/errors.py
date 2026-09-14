"""OpenAI-style error envelope + the gateway's exception hierarchy (DESIGN.md §7.3).

Every failure the gateway produces — validation, streaming, provider
exhaustion, total outage, timeout — exits in exactly one shape:

    {"error": {"message": ..., "type": ..., "param": ..., "code": ...}}

That includes Pydantic validation failures, which FastAPI would otherwise
return as a 422 in its own format. OpenAI's real API returns 400 with this
envelope; being compatible "except when things go wrong" is not compatible.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from llm_gateway.observability.logging import bind_log_fields, get_logger

logger = get_logger(__name__)


# ── the wire shape ─────────────────────────────────────────────────────────


class ErrorBody(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: str | None = None


class ErrorEnvelope(BaseModel):
    error: ErrorBody


# ── exception hierarchy ───────────────────────────────────────────────────


class GatewayError(Exception):
    """Base for every deliberate gateway failure. Knows its own HTTP mapping."""

    status_code: int = 500
    error_type: str = "api_error"
    default_message: str = "gateway error"
    retryable: bool = True

    def __init__(
        self,
        message: str | None = None,
        *,
        param: str | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message or self.default_message)
        self.message = message or self.default_message
        self.param = param
        self.code = code


class StreamNotSupported(GatewayError):
    status_code = 400
    error_type = "invalid_request_error"
    default_message = (
        "stream=true is not supported in v1; the gateway is non-streaming. "
        "SSE passthrough is Phase 6 (docs/DESIGN.md §5.1)."
    )


class ProviderFailure(GatewayError):
    status_code = 502
    error_type = "api_error"
    default_message = "provider errored after all retries and fallbacks"
    retryable = True


class ProviderRejected(GatewayError):
    status_code = 502
    error_type = "api_error"
    default_message = "provider rejected the request"
    retryable = False

class CacheDisabled(GatewayError):
    status_code = 400
    error_type = "invalid_request_error"
    default_message = "cache is disabled (GW_CACHE__ENABLED=false)"



class AllProvidersDown(GatewayError):
    """ADR-0004's fail-loud path. Nothing raises this on Day 1; Day 2's
    fallback chain walker raises it when the chain (mock included) is exhausted."""

    status_code = 503
    error_type = "service_unavailable"
    default_message = (
        "all providers are down — failing loudly rather than serving stale guesses (ADR-0004)"
    )


class RequestTimedOut(GatewayError):
    status_code = 504
    error_type = "timeout"
    default_message = "request exceeded its timeout budget"


# ── handlers ──────────────────────────────────────────────────────────────


def _envelope_response(
    *,
    status_code: int,
    message: str,
    error_type: str,
    param: str | None,
    code: str | None,
) -> JSONResponse:
    body = ErrorEnvelope(error=ErrorBody(message=message, type=error_type, param=param, code=code))
    return JSONResponse(status_code=status_code, content=body.model_dump())


async def _gateway_error_handler(request: Request, exc: GatewayError) -> JSONResponse:
    bind_log_fields(error=f"{type(exc).__name__}: {exc.message}")
    return _envelope_response(
        status_code=exc.status_code,
        message=exc.message,
        error_type=exc.error_type,
        param=exc.param,
        code=exc.code,
    )


def _format_validation_error(err: dict[str, Any]) -> str:
    loc = ".".join(str(part) for part in err.get("loc", ()) if part != "body")
    return f"{loc or 'request'}: {err.get('msg', 'invalid')}"


async def _validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """FastAPI's 422 → OpenAI's 400 envelope. This is the compatibility fix."""
    errors = exc.errors() or [{"loc": (), "msg": "invalid request"}]
    details = "; ".join(_format_validation_error(e) for e in errors)
    first_param = next(
        (str(e["loc"][-1]) for e in errors if e.get("loc") and isinstance(e["loc"][-1], str)),
        None,
    )
    bind_log_fields(error=f"validation_error: {details}")
    return _envelope_response(
        status_code=400,
        message=f"Invalid request: {details}",
        error_type="invalid_request_error",
        param=first_param,
        code="invalid_request",
    )


async def _unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    # Full traceback goes to the log; the client gets a generic body (no leak).
    logger.error("unhandled_exception", exc_info=exc, error=repr(exc))
    return _envelope_response(
        status_code=500,
        message="internal server error",
        error_type="api_error",
        code="internal_error",
    )


def install_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(GatewayError, _gateway_error_handler)
    app.add_exception_handler(RequestValidationError, _validation_error_handler)
    app.add_exception_handler(Exception, _unhandled_error_handler)
