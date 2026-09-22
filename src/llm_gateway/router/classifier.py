"""Rule-based tier classifier (ADR-0001).

Deterministic by design: every decision returns the rule that made it, so any
routing outcome is explainable in one line during an incident. The
TierClassifier protocol is the seam a learned v2 (Phase 6) swaps in behind —
same interface, still deterministic at inference.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import yaml

from llm_gateway.config import Tier
from llm_gateway.router.features import PromptFeatures, extract_features
from llm_gateway.schemas.openai_api import ChatCompletionRequest


@dataclass(frozen=True)
class RoutingDecision:
    tier: Tier
    rule_fired: str  # THE explainability field — logged on every request
    features: PromptFeatures


class TierClassifier(Protocol):
    def classify(self, request: ChatCompletionRequest) -> RoutingDecision: ...


@dataclass(frozen=True)
class RouterRule:
    name: str
    tier: Tier
    any_keywords: tuple[str, ...] = ()
    all_keywords: tuple[str, ...] = ()
    not_keywords: tuple[str, ...] = ()
    min_tokens: int | None = None
    max_tokens: int | None = None
    has_code: bool | None = None
    has_structured_data: bool | None = None

    def matches(self, features: PromptFeatures, lower_text: str) -> bool:  # type: ignore[name-defined]
        """All specified conditions must hold; unspecified = no constraint."""
        if self.any_keywords and not self._any(lower_text, self.any_keywords):
            return False
        if self.all_keywords and not all(self._hit(lower_text, k) for k in self.all_keywords):
            return False
        if self.not_keywords and any(self._hit(lower_text, k) for k in self.not_keywords):
            return False
        if self.min_tokens is not None and features.token_count < self.min_tokens:
            return False
        if self.max_tokens is not None and features.token_count > self.max_tokens:
            return False
        if self.has_code is not None and features.has_code != self.has_code:
            return False

        return (
            self.has_structured_data is None
            or features.has_structured_data == self.has_structured_data
        )

    @staticmethod
    def _any(text: str, keywords: tuple[str, ...]) -> bool:
        return any(RouterRule._hit(text, k) for k in keywords)

    @staticmethod
    def _hit(text: str, keyword: str) -> bool:
        return re.search(rf"\b{re.escape(keyword)}\b", text) is not None


class RuleBasedClassifier:
    def __init__(self, rules: list[RouterRule], default_tier: Tier) -> None:
        self._rules = rules
        self._default_tier = default_tier
        seen: set[str] = set()
        for rule in rules:
            if rule.name in seen:
                raise ValueError(f"duplicate rule name: {rule.name!r}")
            seen.add(rule.name)
        self._all_keywords = tuple(
            dict.fromkeys(  # dedupe, keep order
                k for r in rules for k in (r.any_keywords + r.all_keywords)
            )
        )

    @classmethod
    def from_yaml(cls, path: Path) -> RuleBasedClassifier:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        rules = [
            RouterRule(
                name=raw["name"],
                tier=Tier(raw["tier"]),
                any_keywords=tuple(raw.get("any_keywords", [])),
                all_keywords=tuple(raw.get("all_keywords", [])),
                not_keywords=tuple(raw.get("not_keywords", [])),
                min_tokens=raw.get("min_tokens"),
                max_tokens=raw.get("max_tokens"),
                has_code=raw.get("has_code"),
                has_structured_data=raw.get("has_structured_data"),
            )
            for raw in data.get("rules", [])
        ]
        return cls(rules, Tier(data.get("default_tier", "standard")))

    def classify(self, request: ChatCompletionRequest) -> RoutingDecision:
        text = (request.last_user_message() or "").lower()
        features = extract_features(request, self._all_keywords)
        for rule in self._rules:
            if rule.matches(features, text):
                return RoutingDecision(rule.tier, rule.name, features)
        return RoutingDecision(self._default_tier, "default", features)


class StaticClassifier:
    """Pins one tier — used by tests and (via the header) callers who know better."""

    def __init__(self, tier: Tier) -> None:
        self._tier = tier

    def classify(self, request: ChatCompletionRequest) -> RoutingDecision:
        return RoutingDecision(self._tier, "static", extract_features(request))
