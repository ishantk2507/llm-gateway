"""Cache orchestration — exact first, then semantic (ADR-0002).

The route talks only to this class. Near-misses (similarity ∈ [floor,
threshold)) are NOT errors — the service surfaces them so the route can bind
them onto the request log line as calibration data (§5.2), the input for
Day 6's threshold sweep.
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_gateway.cache import policy
from llm_gateway.cache.embedder import Embedder, MiniLMEmbedder
from llm_gateway.cache.exact import ExactCache, exact_key
from llm_gateway.cache.semantic import SemanticCache
from llm_gateway.config import CacheSettings, Settings
from llm_gateway.schemas.openai_api import ChatCompletionRequest, ChatCompletionResponse


@dataclass
class CacheResult:
    hit: bool
    layer: str | None = None  # "exact" | "semantic"
    similarity: float | None = None  # semantic hits and near-misses only (§5.5)
    response: ChatCompletionResponse | None = None
    near_miss: bool = False


class CacheService:
    def __init__(
        self, *, exact: ExactCache, semantic: SemanticCache | None, settings: CacheSettings
    ) -> None:
        self._exact = exact
        self._semantic = semantic
        self._settings = settings

    async def lookup(self, request: ChatCompletionRequest) -> CacheResult:
        if not self._settings.enabled or not policy.is_cacheable(request):
            return CacheResult(hit=False)
        normalized = policy.normalized_prompt(request)
        family = policy.model_family(request.model)
        key_hash = policy.prompt_hash(normalized, family, policy.temperature_bucket(request))

        exact_response = await self._exact.lookup(exact_key(key_hash))
        if exact_response is not None:
            return CacheResult(hit=True, layer="exact", response=exact_response)

        if self._semantic is None or not self._settings.semantic_enabled:
            return CacheResult(hit=False)
        result = await self._semantic.lookup(normalized, family)
        if result.status == "hit":
            return CacheResult(
                hit=True, layer="semantic", similarity=result.similarity, response=result.response
            )
        if result.status == "near_miss":
            return CacheResult(hit=False, near_miss=True, similarity=result.similarity)
        return CacheResult(hit=False)

    async def store(self, request: ChatCompletionRequest, response: ChatCompletionResponse) -> None:
        if not self._settings.enabled or not policy.is_cacheable(request):
            return
        normalized = policy.normalized_prompt(request)
        family = policy.model_family(request.model)
        key_hash = policy.prompt_hash(normalized, family, policy.temperature_bucket(request))
        await self._exact.store(exact_key(key_hash), response)
        if self._semantic is not None and self._settings.semantic_enabled:
            await self._semantic.store(normalized, family, key_hash, response)

    async def invalidate(
        self,
        *,
        prompt_hash: str | None = None,
        model_family: str | None = None,
        flush: bool = False,
    ) -> int:
        """Counts of evicted entries. Note the deliberate asymmetry: by-family
        invalidation covers the semantic layer only — the exact layer's hash
        embeds the family, and its 1h TTL ages wrong-family entries out on its
        own. The long-lived (24h) layer gets precise invalidation."""
        if flush:
            n = await self._exact.flush()
            if self._semantic is not None:
                n += await self._semantic.flush_all()
            return n
        if prompt_hash is not None:
            await self._exact.delete(exact_key(prompt_hash))
            return await self._semantic.remove_by_prompt_hash(prompt_hash) if self._semantic else 0
        if model_family is not None:
            return await self._semantic.remove_by_family(model_family) if self._semantic else 0
        return 0

    async def aclose(self) -> None:
        if self._semantic is not None:
            await self._semantic.aclose()


async def build_cache(settings: Settings) -> tuple[CacheService, object]:
    """Production wiring. Fails LOUDLY at startup if Redis is unreachable —
    a gateway that silently can't cache is a lie about its own behavior."""
    from redis import asyncio as aioredis

    client = aioredis.from_url(settings.redis.url)
    try:
        await client.ping()
    except Exception as exc:
        await client.aclose()
        raise RuntimeError(
            f"cache is enabled but Redis is unreachable at {settings.redis.url!r} — "
            "start it (`docker compose up -d redis`) or set GW_CACHE__ENABLED=false"
        ) from exc

    embedder: Embedder = MiniLMEmbedder(
        settings.cache.embedding_model, settings.cache.embedder_threads
    )
    semantic = SemanticCache(
        embedder,
        index_path=settings.cache.faiss_index_path,
        db_path=settings.cache.payload_store_path,
        settings=settings.cache,
    )
    service = CacheService(
        exact=ExactCache(client, settings.cache.exact_ttl_seconds),
        semantic=semantic,
        settings=settings.cache,
    )
    return service, client
