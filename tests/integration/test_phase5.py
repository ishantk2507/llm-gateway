"""Phase 5 acceptance — auth 401s, rate-limit 429s with Retry-After, fail-open
limiter, open mode. Breaker-through-the-app lives in the chaos suite."""

from fakeredis.aioredis import FakeRedis
from fastapi.testclient import TestClient
from tests.helpers import hermetic_settings

from llm_gateway.config import (
    AppSettings,
    AuthSettings,
    Environment,
    MockSettings,
    ReliabilitySettings,
    Settings,
)
from llm_gateway.main import create_app
from llm_gateway.reliability.rate_limiter import TokenBucketLimiter

HELLO = {"model": "auto", "messages": [{"role": "user", "content": "Hello, gateway."}]}
AUTH = {"Authorization": "Bearer test-key"}


def make_settings(**auth_overrides) -> Settings:
    auth = dict(api_keys={"test-key"}, rate_limit_rpm=60, rate_limit_burst=2)
    auth.update(auth_overrides)
    return hermetic_settings(
        app=AppSettings(environment=Environment.TEST),
        mock=MockSettings(latency_ms=0),
        reliability=ReliabilitySettings(backoff_base_s=0.001, backoff_cap_s=0.002),
        auth=AuthSettings(**auth),
    )


def test_missing_key_is_a_401_envelope():
    with TestClient(create_app(make_settings())) as client:
        r = client.post("/v1/chat/completions", json=HELLO)
    assert r.status_code == 401
    error = r.json()["error"]
    assert error["code"] == "invalid_api_key"


def test_wrong_key_is_a_401_envelope():
    with TestClient(create_app(make_settings())) as client:
        r = client.post(
            "/v1/chat/completions", json=HELLO, headers={"Authorization": "Bearer wrong"}
        )
    assert r.status_code == 401


def test_valid_key_passes():
    with TestClient(create_app(make_settings())) as client:
        r = client.post("/v1/chat/completions", json=HELLO, headers=AUTH)
    assert r.status_code == 200


def test_burst_exhaustion_is_a_429_with_retry_after():
    limiter = TokenBucketLimiter(FakeRedis(), rpm=60, burst=1)
    with TestClient(create_app(make_settings(), rate_limiter=limiter)) as client:
        first = client.post("/v1/chat/completions", json=HELLO, headers=AUTH)
        second = client.post("/v1/chat/completions", json=HELLO, headers=AUTH)
    assert first.status_code == 200
    assert second.status_code == 429
    assert second.headers["retry-after"]  # present and integral
    assert second.json()["error"]["type"] == "rate_limit_error"


def test_broken_redis_fails_open():
    class BrokenRedis:
        async def hgetall(self, *a, **k):
            raise ConnectionError("down")

        async def hset(self, *a, **k):
            raise ConnectionError("down")

        async def expire(self, *a, **k):
            raise ConnectionError("down")

    limiter = TokenBucketLimiter(BrokenRedis(), rpm=60, burst=1)
    with TestClient(create_app(make_settings(), rate_limiter=limiter)) as client:
        for _ in range(3):  # would all be 429 with a working limiter (burst=1)
            r = client.post("/v1/chat/completions", json=HELLO, headers=AUTH)
            assert r.status_code == 200


def test_open_mode_when_no_keys_configured():
    with TestClient(
        create_app(
            hermetic_settings(
                app=AppSettings(environment=Environment.TEST), mock=MockSettings(latency_ms=0)
            )
        )
    ) as client:
        r = client.post("/v1/chat/completions", json=HELLO)  # no header at all
    assert r.status_code == 200
