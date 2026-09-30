from pathlib import Path

from llm_gateway.config import MockSettings, Tier
from llm_gateway.providers.base import ProviderRegistry
from llm_gateway.providers.mock_adapter import MockProvider
from llm_gateway.router.classifier import StaticClassifier
from llm_gateway.router.mapping import RouterService, TierMapping

DATA = Path(__file__).resolve().parents[2] / "data"


def registry_with(*names: str) -> ProviderRegistry:
    registry = ProviderRegistry()
    for n in names:
        registry.register(MockProvider(MockSettings(latency_ms=0), name=n))
    return registry


def service(registry: ProviderRegistry) -> RouterService:
    return RouterService(
        StaticClassifier(Tier.STANDARD), TierMapping.from_yaml(DATA / "tiers.yaml"), registry
    )


def test_pins_resolve_by_prefix():
    mapping = TierMapping.from_yaml(DATA / "tiers.yaml")
    assert mapping.pinned_provider("gpt-4o") == "openai"
    assert mapping.pinned_provider("claude-3-5-haiku-latest") == "anthropic"
    assert mapping.pinned_provider("qwen2.5:1.5b") == "local"
    assert mapping.pinned_provider("auto") is None
    assert mapping.pinned_provider("premium") is None


def test_full_registry_chains_follow_yaml_cost_order():
    chains = service(registry_with("openai", "local", "mock"))._chains
    assert chains[Tier.CHEAP] == ["local", "mock"]
    assert chains[Tier.STANDARD] == ["openai", "local", "mock"]
    assert chains[Tier.PREMIUM] == ["openai", "local", "mock"]


def test_empty_tier_escalates_up_then_appends_mock():
    # local not registered → cheap escalates to standard's chain, mock still terminates
    chains = service(registry_with("openai", "mock"))._chains
    assert chains[Tier.CHEAP] == ["openai", "mock"]


def test_mock_only_mode_every_tier_serves_mock():
    chains = service(registry_with("mock"))._chains
    assert all(chains[t] == ["mock"] for t in Tier)


def test_unknown_names_fall_back_to_registration_order():
    chains = service(registry_with("a", "b"))._chains
    assert all(chains[t] == ["a", "b"] for t in Tier)


def test_pin_returns_registered_adapter():
    svc = service(registry_with("openai", "mock"))
    assert svc.pinned("gpt-4o") is not None and svc.pinned("gpt-4o").name == "openai"


def test_unregistered_pin_returns_none():
    svc = service(registry_with("mock"))
    assert svc.pinned("gpt-4o") is None
