"""Token bucket — capacity, refill, fail-open. fakeredis-honest (no Lua)."""

import asyncio

from fakeredis.aioredis import FakeRedis

from llm_gateway.reliability.rate_limiter import TokenBucketLimiter


class BrokenRedis:
    async def hgetall(self, *a, **k):
        raise ConnectionError("down")

    async def hset(self, *a, **k):
        raise ConnectionError("down")

    async def expire(self, *a, **k):
        raise ConnectionError("down")


async def test_burst_capacity_then_denied():
    limiter = TokenBucketLimiter(FakeRedis(), rpm=60, burst=2)
    assert (await limiter.check("k"))[0] is True
    assert (await limiter.check("k"))[0] is True
    allowed, retry_after = await limiter.check("k")
    assert allowed is False
    assert retry_after is not None and retry_after > 0


async def test_tokens_refill_over_time():
    limiter = TokenBucketLimiter(FakeRedis(), rpm=3600, burst=1)  # 1 token/sec
    assert (await limiter.check("k"))[0] is True
    assert (await limiter.check("k"))[0] is False
    await asyncio.sleep(1.05)
    assert (await limiter.check("k"))[0] is True


async def test_redis_failure_fails_open():
    limiter = TokenBucketLimiter(BrokenRedis(), rpm=60, burst=5)
    allowed, retry_after = await limiter.check("k")
    assert allowed is True and retry_after is None


async def test_keys_are_isolated():
    limiter = TokenBucketLimiter(FakeRedis(), rpm=60, burst=1)
    assert (await limiter.check("k1"))[0] is True
    assert (await limiter.check("k2"))[0] is True  # separate buckets
    assert (await limiter.check("k1"))[0] is False
