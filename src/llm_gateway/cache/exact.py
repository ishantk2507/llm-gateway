"""Layer 1 — exact match: SHA-256 → Redis (ADR-0002).

Checked before embedding. ~1ms, 100% safe, carries most of the hit rate.
Redis TTL deletes for us — no expiry logic lives in Python.
"""

from __future__ import annotations

from redis import asyncio as aioredis

from llm_gateway.schemas.openai_api import ChatCompletionResponse

KEY_PREFIX = "cache:exact:"


def exact_key(prompt_hash: str) -> str:
    return f"{KEY_PREFIX}{prompt_hash}"


class ExactCache:
    def __init__(self, client: aioredis.Redis, ttl_seconds: int) -> None:
        self._redis = client
        self._ttl = ttl_seconds

    async def lookup(self, key: str) -> ChatCompletionResponse | None:
        raw = await self._redis.get(key)
        if raw is None:
            return None
        return ChatCompletionResponse.model_validate_json(raw)

    async def store(self, key: str, response: ChatCompletionResponse) -> None:
        await self._redis.set(key, response.model_dump_json(), ex=self._ttl)

    async def delete(self, key: str) -> None:
        await self._redis.delete(key)

    async def flush(self) -> int:
        """Delete only OUR keys — never FLUSHDB: this Redis also holds
        rate-limit buckets (Day 6) that are not ours to kill."""
        count = 0
        async for key in self._redis.scan_iter(match=f"{KEY_PREFIX}*"):
            await self._redis.delete(key)
            count += 1
        return count