"""Phase 4 acceptance — real costs, real savings, metrics endpoint, and the
never-break-the-request-path principle, through the real app."""

import pytest
from fakeredis.aioredis import FakeRedis
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY
from tests.helpers import hermetic_settings

from llm_gateway.cache.embedder import FakeEmbedder
from llm_gateway.cache.exact import ExactCache
from llm_gateway.cache.semantic import SemanticCache
from llm_gateway.cache.service import CacheService
from llm_gateway.config import AppSettings, CacheSettings, Environment, MockSettings, Settings
from llm_gateway.main import create_app
from llm_gateway.observability.pricing import PricingService
from llm_gateway.observability.repository import RequestRepository
from llm_gateway.providers.base import ProviderAdapter, ProviderRegistry
from llm_gateway.schemas.openai_api import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    Message,
    Usage,
)

HELLO = {"model": "auto", "messages": [{"role": "user", "content": "Hello, gateway."}]}


class ScriptedPricedAdapter(ProviderAdapter):
    """Returns a PRICED model with fixed usage — 100 in / 50 out on
    $1/$2 per Mtok = exactly $0.0002."""

    name = "scripted"

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        return ChatCompletionResponse(
            id="chatcmpl-priced",
            created=1,
            model="openai/gpt-oss-120b",
            choices=[Choice(message=Message(role="assistant", content="priced"))],
            usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        )

    async def health_check(self) -> bool:
        return True


class ExplodingRepository:
    async def log_request(self, **kwargs):
        raise RuntimeError("boom")

    async def aclose(self) -> None:
        pass


def make_settings() -> Settings:
    return hermetic_settings(
        app=AppSettings(environment=Environment.TEST),
        mock=MockSettings(latency_ms=1),
    )


def make_pricing(tmp_path) -> PricingService:
    path = tmp_path / "pricing.yaml"
    path.write_text(
        "models:\n"
        '  - {pattern: "openai/gpt-oss-120b", provider: groq, '
        "input_per_mtok: 1.0, output_per_mtok: 2.0}\n"
        '  - {pattern: "mock", provider: mock, input_per_mtok: 0.0, output_per_mtok: 0.0}\n',
        encoding="utf-8",
    )
    return PricingService(path)


def make_repo(tmp_path) -> RequestRepository:
    return RequestRepository(f"sqlite:///{tmp_path}/obs.db")


def make_cache(tmp_path) -> CacheService:
    settings = CacheSettings()
    return CacheService(
        exact=ExactCache(FakeRedis(), ttl_seconds=3600),
        semantic=SemanticCache(
            FakeEmbedder(),
            index_path=tmp_path / "v.index",
            db_path=tmp_path / "p.db",
            settings=settings,
        ),
        settings=settings,
    )


def priced_registry() -> ProviderRegistry:
    registry = ProviderRegistry()
    registry.register(ScriptedPricedAdapter())
    return registry


def test_priced_request_costs_exactly_the_table_arithmetic(tmp_path):
    with TestClient(
        create_app(
            make_settings(),
            providers=priced_registry(),
            pricing=make_pricing(tmp_path),
            repository=make_repo(tmp_path),
        )
    ) as client:
        response = client.post("/v1/chat/completions", json=HELLO)

    assert response.status_code == 200
    assert response.headers["x-cost-usd"] == "0.0002"  # 100/1M*1 + 50/1M*2

    rows = [r for r in make_repo_row_check(tmp_path) if r.cost_usd > 0]
    assert rows and rows[0].cost_usd == pytest.approx(0.0002)


def make_repo_row_check(tmp_path):
    # reopen the same sqlite file to read rows (repository accessor)
    repo = RequestRepository(f"sqlite:///{tmp_path}/obs.db")
    return repo.request_rows()


def test_cache_hit_costs_zero_and_saves_the_priced_amount(tmp_path):
    with TestClient(
        create_app(
            make_settings(),
            providers=priced_registry(),
            pricing=make_pricing(tmp_path),
            repository=make_repo(tmp_path),
            cache=make_cache(tmp_path),
        )
    ) as client:
        first = client.post("/v1/chat/completions", json=HELLO)
        second = client.post("/v1/chat/completions", json=HELLO)

    assert first.headers["x-cost-usd"] == "0.0002"
    assert second.headers["x-cache-hit"] == "true"
    assert second.headers["x-cost-usd"] == "0.0000"

    rows = make_repo_row_check(tmp_path)
    hit_row = next(r for r in rows if r.cache_hit)
    assert hit_row.provider == "cache"
    assert hit_row.cost_usd == 0.0
    assert hit_row.cost_saved_usd == pytest.approx(0.0002)  # what you didn't spend

    saved = REGISTRY.get_sample_value("gateway_cost_saved_usd_total")
    assert saved is not None and saved >= 0.0002  # >= : counters accumulate across tests


def test_metrics_endpoint_exposes_the_counters(tmp_path):
    with TestClient(
        create_app(
            make_settings(),
            providers=priced_registry(),
            pricing=make_pricing(tmp_path),
            repository=make_repo(tmp_path),
        )
    ) as client:
        client.post("/v1/chat/completions", json=HELLO)
        metrics = client.get("/v1/metrics")

    assert metrics.status_code == 200
    assert "gateway_requests_total" in metrics.text
    assert "gateway_request_latency_ms" in metrics.text


def test_mock_stays_free_but_priced(tmp_path):
    with TestClient(
        create_app(make_settings(), pricing=make_pricing(tmp_path), repository=make_repo(tmp_path))
    ) as client:
        response = client.post("/v1/chat/completions", json=HELLO)
    assert response.headers["x-cost-usd"] == "0.0000"


def test_repository_failure_never_breaks_the_request(tmp_path):
    with TestClient(
        create_app(
            make_settings(),
            providers=priced_registry(),
            pricing=make_pricing(tmp_path),
            repository=ExplodingRepository(),
        )
    ) as client:
        response = client.post("/v1/chat/completions", json=HELLO)
    assert response.status_code == 200  # the principle, as a test
