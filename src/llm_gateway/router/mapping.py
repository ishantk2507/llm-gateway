"""Tier → chain resolution (DESIGN.md §5.3) — where providers meet routing.

Policies, both surfaced as boot-time logs (the mapping is computed once, so
visibility is free):
- Route UP silently: an empty tier escalates to the next-higher tier — a more
  capable model answering a simple request is always safe.
- Fall back LOUDLY: if nothing maps at all, chains fall back to registration
  order with a warning — less capability than a prompt needs is exactly the
  plausible-but-wrong territory this project refuses to enter quietly.
- The mock terminates every chain it's registered for (ADR-0006). Strict
  fail-loud deployments disable it (GW_MOCK__ENABLED=false) — that's the
  documented ADR-0004 posture.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from llm_gateway.config import Tier
from llm_gateway.observability.logging import get_logger
from llm_gateway.providers.base import ProviderAdapter, ProviderRegistry
from llm_gateway.router.classifier import RuleBasedClassifier, TierClassifier
from llm_gateway.schemas.openai_api import VIRTUAL_MODELS

logger = get_logger(__name__)

_TIER_ORDER = (Tier.CHEAP, Tier.STANDARD, Tier.PREMIUM)


class TierMapping:
    def __init__(self, tiers: dict[Tier, list[str]], model_pins: dict[str, list[str]]) -> None:
        self._tiers = tiers
        self._model_pins = model_pins

    @classmethod
    def from_yaml(cls, path: Path) -> TierMapping:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        tiers = {Tier(k): list(v) for k, v in data.get("tiers", {}).items()}
        pins = {p: list(prefixes) for p, prefixes in data.get("model_pins", {}).items()}
        for tier in _TIER_ORDER:
            tiers.setdefault(tier, [])
        return cls(tiers, pins)

    def pinned_provider(self, model: str) -> str | None:
        for provider, prefixes in self._model_pins.items():
            if model.startswith(tuple(prefixes)):
                return provider
        return None


class RouterService:
    """Classifier + mapping, precomputed against the live registry at startup."""

    def __init__(
        self, classifier: TierClassifier, mapping: TierMapping, registry: ProviderRegistry
    ) -> None:
        self._classifier = classifier
        self._mapping = mapping
        self._registry = registry
        self._chains = self._resolve_chains(mapping, registry)
        logger.info(
            "router_ready",
            tier_chains={tier.value: chain for tier, chain in self._chains.items()},
        )

    def classify(self, request):
        return self._classifier.classify(request)

    def chain_for(self, tier: Tier) -> list[ProviderAdapter]:
        return [self._registry.get(name) for name in self._chains[tier]]

    def pinned(self, model: str) -> ProviderAdapter | None:
        """Explicit model pin — None for virtual names and unknown prefixes."""
        if model in VIRTUAL_MODELS:
            return None
        name = self._mapping.pinned_provider(model)
        if name is None or name not in self._registry.names():
            return None
        return self._registry.get(name)

    def pin_target(self, model: str) -> str | None:
        """Provider NAME a pin resolves to, even if unregistered (for warnings)."""
        if model in VIRTUAL_MODELS:
            return None
        return self._mapping.pinned_provider(model)

    def _resolve_chains(
        self, mapping: TierMapping, registry: ProviderRegistry
    ) -> dict[Tier, list[str]]:
        names = set(registry.names())
        resolved: dict[Tier, list[str]] = {
            tier: [n for n in mapping._tiers.get(tier, []) if n in names and n != "mock"]
            for tier in _TIER_ORDER
        }

        # Route up silently (empty tier → next non-empty higher tier).
        for i, tier in enumerate(_TIER_ORDER):
            if resolved[tier]:
                continue
            for higher in _TIER_ORDER[i + 1 :]:
                if resolved[higher]:
                    logger.warning(
                        "tier_escalated",
                        tier=tier.value,
                        source=higher.value,
                        reason="no registered providers for tier",
                    )
                    resolved[tier] = list(resolved[higher])
                    break

        # Mock terminator — zero-credential mode keeps working (ADR-0006).
        if "mock" in names:
            for tier in _TIER_ORDER:
                resolved[tier].append("mock")

        # Nothing mapped at all (e.g. injected test registries): registration
        # order, loudly.
        if not any(resolved.values()):
            fallback = [p.name for p in registry.default_chain()]
            logger.warning("no_tier_mapped_providers", action="falling_back_to_registration_order")
            for tier in _TIER_ORDER:
                resolved[tier] = list(fallback)
        return resolved


def build_router(
    registry: ProviderRegistry, keywords_path: Path, tiers_path: Path
) -> RouterService:
    for path in (keywords_path, tiers_path):
        if not Path(path).exists():
            raise RuntimeError(f"router data file missing: {path} — restore it from git")
    return RouterService(
        RuleBasedClassifier.from_yaml(Path(keywords_path)),
        TierMapping.from_yaml(Path(tiers_path)),
        registry,
    )
