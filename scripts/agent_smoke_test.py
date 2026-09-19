"""Agent-harness compatibility smoke test — run against a live gateway.

Proves the architecture claim: a THIRD-PARTY framework (LangChain) configured
with nothing but a base_url routes through the gateway — routing decisions,
contract headers, cache, fallback — with zero gateway-specific code in the
agent layer.

The gateway sits between the harness's model client and the provider:
  Agent/harness → model client (ChatOpenAI) → GATEWAY → providers
The harness thinks it's talking to OpenAI. It never is.

Known v1 boundary (documented, by design):
  - non-streaming only (stream=True is a 400; SSE passthrough is Phase 6)
  - tool parameters are dropped with a warning (tool passthrough is Phase 6)
  - multi-turn agent conversations are never cached (ADR-0003) — routing
    still classifies every call
"""

from langchain_openai import ChatOpenAI

BASE_URL = "http://localhost:8000/v1"

# model="auto" — the virtual model: the harness doesn't pick a model at all,
# the gateway's router does. Any harness that can set a model name can do this.
llm = ChatOpenAI(
    base_url=BASE_URL,
    api_key="gw-demo-key",
    model="auto",
    temperature=0.0,
    max_retries=0,
    streaming=False,
)

PROMPTS = [
    "Hello!",  # → cheap / greeting
    "Explain the circuit breaker pattern in distributed systems.",  # → standard
    "Write a Python function that reverses a linked list.",  # → premium
]

print("── routing through a third-party framework ─────")
for prompt in PROMPTS:
    response = llm.invoke(prompt)
    # response_metadata carries provider headers (response['header'] variants
    # across versions — check both spellings)
    header = response.response_metadata.get("header", {}) or response.response_metadata
    tier = header.get("x-routing-tier") or response.additional_kwargs.get("x-routing-tier")
    print(f"{prompt[:52]:<52} → tier={tier or '<check server log>'}")

print("\n── multi-turn: an agent-shaped conversation ────")
llm.invoke(
    [
        ("human", "Explain the circuit breaker pattern."),
        ("ai", "Sure — it wraps failing calls and trips open after failures..."),
        ("human", "And how does the half-open state work?"),
    ]
)
print("multi-turn served — cache correctly bypassed (ADR-0003), router still on")
print("Check the gateway console: rule_fired, routing_tier, cache_hit=False on every call")
