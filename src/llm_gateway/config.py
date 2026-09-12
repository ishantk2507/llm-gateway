"""Central configuration.

Every knob is environment-driven, and every default works with an empty
``.env``: the MockProvider terminates every fallback chain (ADR-0006),
storage falls back to local SQLite, and no credentials are required.
Production is a matter of setting variables, not editing code.

Variable format — note the double underscore (pydantic-settings' nested
delimiter) between group and field:

    GW_<GROUP>__<FIELD>       e.g.  GW_CACHE__SIMILARITY_THRESHOLD=0.95

Provider credentials keep their standard names (``OPENAI_API_KEY``,
``ANTHROPIC_API_KEY``) so the shell that configures your other tools
configures this one too.

The annotated reference for every variable lives in ``.env.example``.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root — anchors data-file paths so behavior is independent of CWD.
REPO_ROOT = Path(__file__).resolve().parents[2]


# ───────────────────────────── shared enums ─────────────────────────────


class Environment(StrEnum):
    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


class Tier(StrEnum):
    """Routing tiers, cheapest → most capable. Day 4's router imports this."""

    CHEAP = "cheap"
    STANDARD = "standard"
    PREMIUM = "premium"


# ───────────────────────────── settings groups ───────────────────────────


class AppSettings(BaseModel):
    app_name: str = "llm-gateway"
    environment: Environment = Environment.DEVELOPMENT
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "INFO"
    workers: int = 1

    @field_validator("log_level")
    @classmethod
    def _uppercase(cls, v: str) -> str:
        return v.upper()

    @field_validator("workers")
    @classmethod
    def _single_worker_by_design(cls, v: int) -> int:
        if v != 1:
            raise ValueError(
                "GW_APP__WORKERS must stay 1 in v1: the FAISS index and "
                "circuit-breaker state are per-process and must stay coherent "
                "(DESIGN.md §13). Multi-worker requires shared state — that's "
                "the documented Phase 6 path, not a flag flip. Failing loud at "
                "boot beats silently incoherent caches later."
            )
        return v


class AuthSettings(BaseModel):
    """API-key auth + per-key token bucket (wired up Day 6)."""

    # Comma-separated in env: GW_AUTH__API_KEYS=gw-demo-key,second-key
    api_keys: set[str] = Field(default_factory=lambda: {"gw-demo-key"})
    rate_limit_rpm: int = Field(default=120, ge=1)
    rate_limit_burst: int = Field(default=30, ge=1)

    @field_validator("api_keys", mode="before")
    @classmethod
    def _split_csv(cls, v: object) -> object:
        if isinstance(v, str):
            return {key.strip() for key in v.split(",") if key.strip()}
        return v

    @property
    def api_key_hashes(self) -> set[str]:
        """SHA-256 digests — the only form ever stored or compared (§10)."""
        return {sha256(key.encode()).hexdigest() for key in self.api_keys}


class RedisSettings(BaseModel):
    """Exact cache · idempotency records · rate-limit buckets."""

    url: str = "redis://localhost:6379/0"


class CacheSettings(BaseModel):
    """Two-layer cache: exact (SHA-256 → Redis), then semantic (FAISS).

    Policy: single-turn, temperature ≤ 0.3, no tools, non-streaming
    (ADR-0003). Scores in the near-miss band are logged as calibration
    data, not discarded (DESIGN.md §5.2).
    """

    enabled: bool = True  # master flag — Phase 2 shipped behind it
    semantic_enabled: bool = True  # false → exact-match-only mode
    degraded_mode: bool = False  # opt-in: during total outage serve exact hits only (ADR-0004)

    similarity_threshold: float = Field(default=0.95, gt=0.0, lt=1.0)
    near_miss_floor: float = Field(default=0.85, gt=0.0, lt=1.0)

    exact_ttl_seconds: int = Field(default=3600, ge=1)
    semantic_ttl_seconds: int = Field(default=86_400, ge=1)
    max_semantic_entries: int = Field(default=50_000, ge=1)  # memory bound (ADR-0005)

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, CPU-viable
    embedder_threads: int = Field(default=4, ge=1)  # thread executor — never block the loop
    payload_store_url: str = f"sqlite:///{REPO_ROOT / 'cache_payloads.db'}"

    @model_validator(mode="after")
    def _near_miss_band_below_threshold(self) -> CacheSettings:
        if self.near_miss_floor >= self.similarity_threshold:
            raise ValueError(
                "near_miss_floor must be strictly below similarity_threshold — "
                "the band between them is what gets logged for calibration"
            )
        return self


class RouterSettings(BaseModel):
    """Rule-based complexity router (ADR-0001). Rules are data, not code."""

    enabled: bool = True
    keywords_path: Path = REPO_ROOT / "data" / "keywords.yaml"
    default_tier: Tier = Tier.STANDARD


class MockSettings(BaseModel):
    """MockProvider — first-class adapter, fallback-chain terminator (ADR-0006).

    With no provider credentials set, every tier's chain ends here, so the
    entire system — routing, caching, chaos tests, load tests, demo — runs
    at zero cost.
    """

    enabled: bool = True
    latency_ms: int = Field(default=800, ge=0)  # realistic enough that cache wins are visible
    error_rate: float = Field(default=0.0, ge=0.0, le=1.0)  # >0 simulates a flaky provider


class ReliabilitySettings(BaseModel):
    """Retries, timeout budgets, circuit breakers (DESIGN.md §9)."""

    max_retries: int = Field(default=3, ge=0)  # per provider
    backoff_base_s: float = Field(default=0.5, gt=0)  # exponential, full jitter
    backoff_cap_s: float = Field(default=5.0, gt=0)

    request_timeout_budget_s: float = Field(default=30.0, gt=0)  # whole request, all providers
    per_attempt_timeout_s: float = Field(default=10.0, gt=0)  # one provider can't eat the budget

    breaker_window: int = Field(default=20, ge=1)  # requests considered
    breaker_failure_rate: float = Field(default=0.5, gt=0.0, le=1.0)  # ≥ this → open
    breaker_cooldown_s: float = Field(default=30.0, gt=0)  # open → half-open after this
    breaker_half_open_probes: int = Field(default=3, ge=1)

    @model_validator(mode="after")
    def _budgets_are_coherent(self) -> ReliabilitySettings:
        if self.per_attempt_timeout_s >= self.request_timeout_budget_s:
            raise ValueError("per_attempt_timeout_s must be < request_timeout_budget_s")
        if self.backoff_base_s > self.backoff_cap_s:
            raise ValueError("backoff_base_s must be ≤ backoff_cap_s")
        return self


class IdempotencySettings(BaseModel):
    """Gateway-side dedup: X-Idempotency-Key → cached response in Redis (§5.4).
    Providers don't honor idempotency keys, so the gateway does."""

    enabled: bool = True
    ttl_seconds: int = Field(default=300, ge=1)


class ObservabilitySettings(BaseModel):
    """Structured logs, Prometheus metrics, request/cost persistence."""

    database_url: str = f"sqlite:///{REPO_ROOT / 'gateway.db'}"  # Postgres in compose
    pricing_path: Path = REPO_ROOT / "data" / "pricing.yaml"  # versioned cost table


# Provider groups are BaseSettings with their OWN standard prefixes — they
# read OPENAI_API_KEY / ANTHROPIC_API_KEY directly, not GW_-nested names.
class OpenAIProviderSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OPENAI_", env_file=".env", extra="ignore")

    api_key: SecretStr = SecretStr("")  # empty → adapter not registered
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"  # placeholder default; tier→model mapping lands Day 4
    timeout_s: float = Field(default=10.0, gt=0)


class AnthropicProviderSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ANTHROPIC_", env_file=".env", extra="ignore")

    api_key: SecretStr = SecretStr("")
    base_url: str = "https://api.anthropic.com"
    model: str = "claude-3-5-haiku-latest"
    timeout_s: float = Field(default=10.0, gt=0)


# ───────────────────────────── root settings ─────────────────────────────


class Settings(BaseSettings):
    """Root settings. Groups map 1:1 onto the .env.example sections."""

    model_config = SettingsConfigDict(
        env_prefix="GW_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app: AppSettings = Field(default_factory=AppSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    router: RouterSettings = Field(default_factory=RouterSettings)
    mock: MockSettings = Field(default_factory=MockSettings)
    reliability: ReliabilitySettings = Field(default_factory=ReliabilitySettings)
    idempotency: IdempotencySettings = Field(default_factory=IdempotencySettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    openai: OpenAIProviderSettings = Field(default_factory=OpenAIProviderSettings)
    anthropic: AnthropicProviderSettings = Field(default_factory=AnthropicProviderSettings)

    @model_validator(mode="after")
    def _fallback_chain_terminates(self) -> Settings:
        has_credentials = bool(
            self.openai.api_key.get_secret_value() or self.anthropic.api_key.get_secret_value()
        )
        if not has_credentials and not self.mock.enabled:
            raise ValueError(
                "No provider available: set OPENAI_API_KEY or ANTHROPIC_API_KEY, "
                "or keep GW_MOCK__ENABLED=true (zero-credential demo mode, "
                "ADR-0006). A gateway with nothing to route to should fail at "
                "boot, not per request."
            )
        return self


@lru_cache
def get_settings() -> Settings:
    """Cached accessor. Use via FastAPI ``Depends(get_settings)`` so tests can
    override it with a fresh ``Settings(_env_file=None)`` instead of
    monkeypatching a module-level global."""
    return Settings()
