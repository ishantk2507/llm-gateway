"""Prometheus instruments — registered at MODULE IMPORT, never per-app.

Why module-level: prometheus_client registers instruments in a process-global
REGISTRY at construction time. Constructing per-app means the second test app
explodes with "duplicate timeseries". Module-level = registered once, which
also matches the single-worker design (DESIGN.md §13): no multiproc mode is
needed, and that is an argument, not an oversight.

Label cardinality is deliberately bounded — NO 'model' label on any
instrument. Per-model breakdowns live in Postgres (repository.py), where
cardinality is free.

The single call site is RequestLoggingMiddleware (observability/logging.py),
after the response: every COMPLETION request is observed, including 5xx
(where provider_used was never bound and reads as "unknown").

Test discipline: counters accumulate across tests — assert via
REGISTRY.get_sample_value(...) with >=, never ==.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

gateway_requests_total = Counter(
    "gateway_requests_total",
    "Completion requests by provider, tier, status and cache outcome.",
    ["provider", "tier", "status", "cache"],
)
gateway_cache_hits_total = Counter(
    "gateway_cache_hits_total",
    "Cache hits by layer (exact | semantic).",
    ["layer"],
)
gateway_cost_usd_total = Counter(
    "gateway_cost_usd_total",
    "Estimated provider spend in USD (priced requests only).",
)
gateway_cost_saved_usd_total = Counter(
    "gateway_cost_saved_usd_total",
    "Estimated spend avoided by cache hits, in USD.",
)
gateway_request_latency_ms = Histogram(
    "gateway_request_latency_ms",
    "End-to-end completion latency in milliseconds.",
    buckets=(5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000),
)
gateway_breaker_state = Gauge(
    "gateway_breaker_state",
    "Circuit-breaker state per provider: 0=closed, 1=open, 2=half-open.",
    ["provider"],
)


def set_breaker_state(provider: str, state: int) -> None:
    """Single setter — main initializes it for every registered provider at
    startup (absence reads as 'no data', which is a lie on day one of the
    panel's life) and the BreakerBoard's on_state callback keeps it current."""
    gateway_breaker_state.labels(provider=provider).set(state)


def observe_request(
    *,
    provider: str,
    tier: str,
    status: str,
    cache_hit: bool,
    layer: str | None,
    latency_ms: float,
    cost_usd: float = 0.0,
    saved_usd: float = 0.0,
) -> None:
    """One call per completion request — the only function routes/middleware touch."""
    gateway_requests_total.labels(
        provider=provider, tier=tier, status=status, cache="hit" if cache_hit else "miss"
    ).inc()
    if cache_hit and layer:
        gateway_cache_hits_total.labels(layer=layer).inc()
    if cost_usd > 0:
        gateway_cost_usd_total.inc(cost_usd)
    if saved_usd > 0:
        gateway_cost_saved_usd_total.inc(saved_usd)
    gateway_request_latency_ms.observe(latency_ms)
