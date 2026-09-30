"""Auth + rate limiting — the gate in front of /v1/chat/completions.

Fail-CLOSED where identity is concerned: pure SHA-256 compare against the
configured hashes, no I/O, no fallback. Fail-OPEN where availability is
concerned: the limiter degrades to allow-all when Redis is down.

Open mode: no keys configured → the gate is a no-op. Zero-config demo boot
(ADR-0006) is preserved; production sets GW_AUTH__API_KEYS.
"""

from __future__ import annotations

import hashlib

from fastapi import Request

from llm_gateway.config import Settings
from llm_gateway.reliability.rate_limiter import TokenBucketLimiter
from llm_gateway.schemas.errors import AuthError, RateLimited


async def gateway_gate(request: Request) -> None:
    settings: Settings = request.app.state.settings
    if not settings.auth.api_keys:
        return  # open mode — no keys configured

    token = _bearer(request)
    key_hash = hashlib.sha256(token.encode()).hexdigest() if token else ""
    if key_hash not in settings.auth.api_key_hashes:
        raise AuthError()

    limiter: TokenBucketLimiter | None = getattr(request.app.state, "rate_limiter", None)
    if limiter is None:
        return  # limiter not built (cache disabled + lazy client pending) — auth alone stands
    allowed, retry_after = await limiter.check(key_hash)
    if not allowed:
        raise RateLimited(retry_after=retry_after)


def _bearer(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    return token.strip() if scheme.lower() == "bearer" and token.strip() else None
