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
    status_code: int = 500
    error_type: str = "api_error"
    default_message: str = "gateway error"
    default_code: str | None = None
    retryable: bool = False

    def __init__(
        self, message: str | None = None, *, param: str | None = None, code: str | None = None
    ) -> None:
        super().__init__(message or self.default_message)
        self.message = message or self.default_message
        self.param = param
        self.code = code or self.default_code


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


class AttemptTimedOut(RequestTimedOut):
    """One provider's attempt exceeded its timeout slice. NOT retried on the
    same provider (restarting the same slow work just burns the budget — the
    live Day-4 finding); the fallback walker catches it and spends the
    remaining budget on the next provider. A chain that dies entirely of
    timeouts is a 504, not a 503."""

    default_message = "provider attempt exceeded its timeout slice — yielding to fallback"


class AuthError(GatewayError):
    status_code = 401
    error_type = "invalid_request_error"
    default_message = "missing or invalid API key"
    default_code = "invalid_api_key"


class RateLimited(GatewayError):
    status_code = 429
    error_type = "rate_limit_error"
    default_message = "rate limit exceeded for this API key"
    default_code = "rate_limit_exceeded"

    def __init__(self, message: str | None = None, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ChaosDisabled(GatewayError):
    status_code = 403
    error_type = "invalid_request_error"
    default_message = "chaos endpoints are available in development/test only"
    default_code = "chaos_disabled"


class ChaosTargetError(GatewayError):
    status_code = 400
    error_type = "invalid_request_error"
    default_message = "chaos targets must be mock providers"
    default_code = "chaos_target_invalid"


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
    headers: dict[str, str] = {}
    if getattr(exc, "retry_after", None):
        headers["Retry-After"] = str(max(1, int(exc.retry_after)))
    body = ErrorEnvelope(
        error=ErrorBody(message=exc.message, type=exc.error_type, param=exc.param, code=exc.code)
    )
    return JSONResponse(status_code=exc.status_code, content=body.model_dump(), headers=headers)


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
