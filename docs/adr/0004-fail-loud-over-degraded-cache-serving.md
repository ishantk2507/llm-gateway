ADR-0004: Fail loud when all providers are down
Status: Accepted · Date: 2025-09-12

Context
When every provider in a chain fails, a gateway has three options: (a) serveplausible-but-wrong near-match cache entries so clients keep seeing 200s,(b) return a vague 500, (c) fail explicitly with a machine-readable error.Option (a) is the tempting one — the system looks healthy while returninganswers that may be semantically wrong for the request actually asked.

Decision
Exhausting the fallback chain raises AllProvidersDown → an explicit 503in the OpenAI-style error envelope, naming the providers that failed. Thegateway never serves cache entries as a degradation strategy by default. Anopt-in degraded mode (GW_CACHE__DEGRADED_MODE, landing with the Phase-2cache) will serve exact-match entries only, flagged via response headers.

Related semantics (DESIGN.md §7.3): a request pinned to one provider thatfails surfaces as 502 — "this upstream errored after its retries"; a routedrequest that burned the whole chain surfaces as 503 — "no provider available".

Consequences
Every 200 is trustworthy — clients can treat success as success.
Full outages are visible in logs and metrics, not hidden behind healthy codes.− Clients see errors during total outages. That is the point: a gateway thatsilently serves wrong-but-plausible answers is a liability, not a feature.