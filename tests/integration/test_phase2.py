"""Phase 2 acceptance — through the real app with FakeEmbedder + fakeredis.

No paraphrase test lives here BY DESIGN: the fake can't paraphrase (different
texts land near-orthogonal). Paraphrase belongs to the correctness file with
the real model. What lives here: exact hits, policy gates, bypass,
invalidation — and one ConstantEmbedder test proving the semantic path works
through the app without the model.
"""

import time

import numpy as np
from fakeredis.aioredis import FakeRedis
from fastapi.testclient import TestClient

from llm_gateway.cache import policy
from llm_gateway.cache.embedder import FakeEmbedder
from llm_gateway.cache.exact import ExactCache
from llm_gateway.cache.semantic import SemanticCache
from llm_gateway.cache.service import CacheService
from llm_gateway.config import AppSettings, CacheSettings, Environment, MockSettings, Settings
from llm_gateway.main import create_app

HELLO = {"model": "auto", "messages": [{"role": "user", "content": "Hello, gateway."}]}


class ConstantEmbedder:
    """Every text → the same unit vector: similarity 1.0 for anything."""

    async def embed(self, text: str) -> np.ndarray:
        v = np.zeros(384, dtype=np.float32)
        v[0] = 1.0
        return v

    async def aclose(self) -> None:
        pass


def make_settings() -> Settings:
    return Settings(
        _env_file=None,
        app=AppSettings(environment=Environment.TEST),
        mock=MockSettings(latency_ms=50),
    )


def make_cache(tmp_path, embedder=None) -> CacheService:
    settings = CacheSettings()
    semantic = SemanticCache(
        embedder or FakeEmbedder(),
        index_path=tmp_path / "v.index",
        db_path=tmp_path / "p.db",
        settings=settings,
    )
    return CacheService(
        exact=ExactCache(FakeRedis(), ttl_seconds=3600),
        semantic=semantic,
        settings=settings,
    )


def client_for(settings, cache) -> TestClient:
    return TestClient(create_app(settings, cache=cache))


def test_repeat_request_hits_cache_identically(tmp_path):
    with client_for(make_settings(), make_cache(tmp_path)) as client:
        t0 = time.perf_counter()
        first = client.post("/v1/chat/completions", json=HELLO)
        t1 = time.perf_counter()
        second = client.post("/v1/chat/completions", json=HELLO)
        t2 = time.perf_counter()

    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()  # byte-identical body
    assert second.headers["x-cache-hit"] == "true"
    assert second.headers["x-provider-used"] == "cache"
    assert second.headers["x-cost-usd"] == "0.0000"
    assert (t2 - t1) < (t1 - t0)  # hit skips the mock's 50ms sleep


def test_different_prompt_misses(tmp_path):
    with client_for(make_settings(), make_cache(tmp_path)) as client:
        client.post("/v1/chat/completions", json=HELLO)
        other = client.post(
            "/v1/chat/completions",
            json={"model": "auto", "messages": [{"role": "user", "content": "Different."}]},
        )
    assert other.headers["x-cache-hit"] == "false"
    assert other.headers["x-provider-used"] == "mock"


def test_high_temperature_never_caches(tmp_path):
    with client_for(make_settings(), make_cache(tmp_path)) as client:
        hot = {**HELLO, "temperature": 0.9}
        client.post("/v1/chat/completions", json=hot)
        second = client.post("/v1/chat/completions", json=hot)
    assert second.headers["x-cache-hit"] == "false"


def test_multi_turn_never_caches(tmp_path):
    convo = {"model": "auto", "messages": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "Continue"},
    ]}
    with client_for(make_settings(), make_cache(tmp_path)) as client:
        client.post("/v1/chat/completions", json=convo)
        second = client.post("/v1/chat/completions", json=convo)
    assert second.headers["x-cache-hit"] == "false"


def test_bypass_header_forces_live(tmp_path):
    with client_for(make_settings(), make_cache(tmp_path)) as client:
        client.post("/v1/chat/completions", json=HELLO)
        live = client.post("/v1/chat/completions", json=HELLO, headers={"x-cache": "bypass"})
    assert live.headers["x-cache-hit"] == "false"
    assert live.headers["x-provider-used"] == "mock"


def test_semantic_layer_serves_through_app_with_matching_vector(tmp_path):
    # ConstantEmbedder: ANY two prompts are similarity 1.0, so a different
    # prompt must be served by the SEMANTIC layer, not the exact layer.
    cache = make_cache(tmp_path, embedder=ConstantEmbedder())
    with client_for(make_settings(), cache) as client:
        client.post("/v1/chat/completions", json=HELLO)
        other = client.post(
            "/v1/chat/completions",
            json={"model": "auto", "messages": [{"role": "user", "content": "Different text"}]},
        )
    assert other.headers["x-cache-hit"] == "true"
    assert other.headers["x-provider-used"] == "cache"


def test_invalidate_by_prompt_hash_then_miss(tmp_path):
    with client_for(make_settings(), make_cache(tmp_path)) as client:
        client.post("/v1/chat/completions", json=HELLO)
        hit = client.post("/v1/chat/completions", json=HELLO)
        assert hit.headers["x-cache-hit"] == "true"

        # The documented identity: sha256("family|bucket|normalized_prompt")
        h = policy.prompt_hash(
            policy.normalized_prompt(
                __import__(
                    "llm_gateway.schemas.openai_api", fromlist=["ChatCompletionRequest"]
                ).ChatCompletionRequest(**HELLO)
            ),
            policy.model_family("auto"),
            "low",
        )
        invalidated = client.post("/v1/cache/invalidate", json={"prompt_hash": h})
        assert invalidated.status_code == 200
        assert invalidated.json()["evicted"] >= 1

        after = client.post("/v1/chat/completions", json=HELLO)
    assert after.headers["x-cache-hit"] == "false"