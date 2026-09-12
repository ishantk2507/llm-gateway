ADR-0006: MockProvider as a first-class citizen
Status: Accepted · Date: 2025-XX-XX

Context
The gateway must be testable, load-testable, and demoable without paid APIkeys, flaky external services, or non-determinism. Options: (a) mock at theHTTP layer per test, (b) a stub route bypassing the provider stack, (c) areal adapter with the same interface as production providers.

Decision
MockProvider implements the full ProviderAdapter protocol — same registry,same call path, same headers — with configurable latency and error rate. Itsoutput is deterministic: ids and created derive from a prompt hash, soidentical requests produce byte-identical responses. With no providercredentials set, it terminates every fallback chain.

Consequences
Offline CI (contract tests use recorded fixtures; everything else runs against the mock).
Deterministic chaos tests — error injection is a config knob, not an outage.
Free, reproducible load tests and a zero-credential demo.
Cache-correctness tests can assert exact response equality.− One more adapter to maintain; it has paid for itself on day one.− Day 6 chaos tooling needs a health kill switch on it (noted, not built).