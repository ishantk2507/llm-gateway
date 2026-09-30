"""Redis token-bucket rate limiter — per API key (DESIGN.md §9, §10).

Bucket state in one Redis hash per key: ``ratelimit:<sha256(key)>`` with
``tokens`` and ``updated`` (monotonic clock — single-process consistent).
Refill = rpm/60 tokens per second; burst = capacity. TTL is 2x the full
refill time so idle keys vanish.

Atomicity argument, stated truthfully: single-worker async is INTERLEAVED,
not atomic — the read-compute-write spans awaits, so two concurrent requests
would both read the same token count and both admit. One asyncio.Lock around
the check-admit section PLUS the single-worker design (DESIGN.md §13) is what
makes this correct without Lua. Multi-worker needs Lua or Redis transactions
— the documented Phase 6 path.

Redis unavailable → FAIL OPEN with a warning: availability over strictness
for rate limiting. Authentication is a separate concern and stays
fail-closed (pure hash compare, no I/O).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping

from llm_gateway.observability.logging import get_logger

logger = get_logger(__name__)


def _to_float(raw: Mapping, name: str, default: float) -> float:
    """Redis hashes come back bytes-or-str depending on client config."""
    for key in (name, name.encode()):
        if key in raw:
            try:
                return float(raw[key])
            except (TypeError, ValueError):
                return default
    return default


class TokenBucketLimiter:
    def __init__(self, redis, *, rpm: int, burst: int) -> None:
        self._redis = redis
        self._rate = rpm / 60.0
        self._capacity = float(burst)
        self._lock = asyncio.Lock()
        self._ttl = int(self._capacity / self._rate * 2) + 1 if self._rate > 0 else 60
        self._warned = False

    async def check(self, key_hash: str) -> tuple[bool, float | None]:
        """→ (allowed, retry_after_s). Never raises — fails open."""
        redis_key = f"ratelimit:{key_hash}"
        async with self._lock:  # see module docstring: interleaving ≠ atomicity
            try:
                raw = await self._redis.hgetall(redis_key)
                now = time.monotonic()
                tokens = _to_float(raw, "tokens", self._capacity)
                last = _to_float(raw, "updated", now)
                elapsed = max(0.0, now - last)
                tokens = min(self._capacity, tokens + elapsed * self._rate)
                if tokens >= 1.0:
                    tokens -= 1.0
                    allowed, retry_after = True, None
                else:
                    allowed = False
                    retry_after = (1.0 - tokens) / self._rate if self._rate > 0 else 60.0
                await self._redis.hset(redis_key, mapping={"tokens": tokens, "updated": now})
                await self._redis.expire(redis_key, self._ttl)
                return allowed, retry_after
            except Exception as exc:
                if not self._warned:
                    self._warned = True
                    logger.warning(
                        "rate_limiter_unavailable",
                        error=repr(exc),
                        note="failing OPEN — availability over strictness; auth still stands",
                    )
                return True, None
