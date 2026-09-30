# Runbook

| Symptom | Likely cause | Check | Action |
|---|---|---|---|
| 503 "all provider circuits open" | breakers tripped | `/v1/health` breaker field, Grafana breaker panel | wait out cooldown (auto half-open); if probes keep failing the provider is down — its status page |
| Breaker stuck open | sustained real failures | `provider_failed_moving_on` rate in logs | cooldown auto half-opens; persistent probe failures = provider down |
| Coherent-rejection loop (~100ms, breaker never trips) | invalid API key at one provider | adapter error text in logs | fix the key — by design the breaker tracks health; rejections are coherent (alive) |
| 503 "all providers failed" (no circuit mention) | chain exhausted with circuits closed | `provider_failed_moving_on` lines | verify keys/quota — ADR-0004 fail-loud |
| 504 "budget exhausted" | slow provider(s) consuming the total budget | latency by provider panel | attempt-yield already moves on; raise `REQUEST_TIMEOUT_BUDGET_S` for slow reasoning models |
| 429s for a valid key | token bucket exhausted | `Retry-After` header | raise `GW_AUTH__RATE_LIMIT_RPM` / `RATE_LIMIT_BURST` |
| 401 invalid_api_key | wrong/missing Bearer | client code | use a key from `GW_AUTH__API_KEYS` |
| p95 spike, no provider errors | cache miss storm / cold cache | hit-rate panel | check threshold; run threshold sweep |
| Hit rate near 0% | TTL too short / threshold too high | near-miss ratio in logs | recalibrate via `benchmarks/threshold_sweep.py` |
| Cost higher than expected | reasoning tokens billing as output | cost by provider | see pricing.yaml notes; consider tier routing |
