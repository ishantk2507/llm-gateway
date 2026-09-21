"""Chaos acceptance — kill under load: breaker trips, failover transparent,
cooldown + probe close the circuit, health tells the truth. All mock (ADR-0006)."""

from fastapi.testclient import TestClient
from tests.helpers import hermetic_settings

from llm_gateway.config import (
    AppSettings,
    CacheSettings,
    Environment,
    MockSettings,
    ReliabilitySettings,
    Settings,
)
from llm_gateway.main import create_app
from llm_gateway.observability.repository import RequestRepository
from llm_gateway.providers.base import ProviderAdapter, ProviderRegistry
from llm_gateway.providers.mock_adapter import MockProvider
from llm_gateway.reliability.circuit_breaker import BreakerBoard

HELLO = {"model": "auto", "messages": [{"role": "user", "content": "Hello, gateway."}]}


class FakeClock:
    def __init__(self) -> None:
        self._t = 1000.0

    def __call__(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:
        self._t += seconds


def make_settings() -> Settings:
    # window=5 keeps the trip math short; the COMMENT matters: a breaker only
    # trips on a FULL window of terminal outcomes — the kill test MUST fire
    # >= 5 requests post-kill before asserting anything about the circuit.
    return hermetic_settings(
        app=AppSettings(environment=Environment.TEST),
        mock=MockSettings(latency_ms=0),
        reliability=ReliabilitySettings(
            max_retries=1,
            backoff_base_s=0.001,
            backoff_cap_s=0.002,
            per_attempt_timeout_s=5.0,
            request_timeout_budget_s=30.0,
            breaker_window=5,
            breaker_failure_rate=0.5,
            breaker_cooldown_s=30.0,
            breaker_half_open_probes=1,
        ),
    )


def registry_with(*adapters: ProviderAdapter) -> ProviderRegistry:
    registry = ProviderRegistry()
    for a in adapters:
        registry.register(a)
    return registry


def breaker_states(client) -> dict[str, str]:
    return {p["name"]: p["breaker"] for p in client.get("/v1/health").json()["providers"]}


def test_kill_trip_skip_revive_probe_close():
    settings = make_settings()
    clock = FakeClock()
    board = BreakerBoard(settings.reliability, clock=clock)
    primary = MockProvider(MockSettings(latency_ms=0), name="primary")
    backup = MockProvider(MockSettings(latency_ms=0), name="backup")

    with TestClient(
        create_app(settings, providers=registry_with(primary, backup), breakers=board)
    ) as client:
        client.post("/v1/chaos/providers/primary/kill", json={})

        # Full window = 5 terminal outcomes: 5 requests, each one conclusion
        # (failure) on primary, then served TRANSPARENTLY by backup.
        for _ in range(5):
            r = client.post("/v1/chat/completions", json=HELLO)
            assert r.status_code == 200
            assert r.headers["x-provider-used"] == "backup"
        assert primary.calls == 5

        # Circuit open: primary skipped — calls frozen, backup serves
        r = client.post("/v1/chat/completions", json=HELLO)
        assert r.headers["x-provider-used"] == "backup"
        assert primary.calls == 5
        assert breaker_states(client)["primary"] == "open"

        # Cooldown elapses (fake clock) → half-open; revive → probe closes
        client.post("/v1/chaos/providers/primary/revive", json={})
        clock.advance(31.0)
        r = client.post("/v1/chat/completions", json=HELLO)
        assert r.headers["x-provider-used"] == "primary"  # probe admitted
        assert primary.calls == 6
        assert breaker_states(client)["primary"] == "closed"


def test_all_circuits_open_fails_loud_503():
    settings = make_settings()
    board = BreakerBoard(settings.reliability)
    a = MockProvider(MockSettings(latency_ms=0), name="a")
    b = MockProvider(MockSettings(latency_ms=0), name="b")

    with TestClient(create_app(settings, providers=registry_with(a, b), breakers=board)) as client:
        client.post("/v1/chaos/providers/a/kill", json={})
        client.post("/v1/chaos/providers/b/kill", json={})
        for _ in range(5):  # fill both windows: chain-exhausted 503s
            assert client.post("/v1/chat/completions", json=HELLO).status_code == 503
        # Both circuits open now — the 503 flavor CHANGES to circuits-open
        r = client.post("/v1/chat/completions", json=HELLO)
        assert r.status_code == 503
        assert "circuits open" in r.json()["error"]["message"]
        assert a.calls == 5 and b.calls == 5  # skipped entirely


def test_chaos_forbidden_outside_dev_and_test():
    # PRODUCTION env makes the lifespan auto-build pricing + repository and
    # (cache enabled) demand Redis — hermeticity requires pinning both off
    # without touching the guarded code path under test.
    settings = hermetic_settings(
        app=AppSettings(environment=Environment.PRODUCTION),
        mock=MockSettings(latency_ms=0),
        cache=CacheSettings(enabled=False),
    )
    with TestClient(
        create_app(settings, repository=RequestRepository("sqlite:///:memory:"))
    ) as client:
        r = client.post("/v1/chaos/providers/mock/kill", json={})
    assert r.status_code == 403


def test_chaos_rejects_non_mock_and_unknown_targets():
    class RealAdapter(ProviderAdapter):
        name = "openai"

        async def complete(self, request):
            raise NotImplementedError

        async def health_check(self):
            return True

    settings = make_settings()
    real = RealAdapter()
    with TestClient(create_app(settings, providers=registry_with(real))) as client:
        r = client.post("/v1/chaos/providers/openai/kill", json={})
        assert r.status_code == 400  # remote providers can't be killed — honest tooling
        r = client.post("/v1/chaos/providers/ghost/kill", json={})
        assert r.status_code == 400
