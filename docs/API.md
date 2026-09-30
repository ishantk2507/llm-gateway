## Agent harness compatibility

The gateway is a drop-in proxy for any harness whose model client accepts a
custom base URL (LangChain, LlamaIndex, AutoGen, the raw OpenAI SDK):

```python
llm = ChatOpenAI(base_url="http://localhost:8000/v1", api_key="gw-demo-key", model="auto")
```

Contract with agent frameworks:

- **Messages pass through verbatim.** The gateway reads messages (features,
  embedding) and forwards them unchanged — envelope shape is normalized,
  content never mutated. Harness conversation state is untouched.
- **Every call is routed** by the complexity router — including calls inside
  agent loops. `model="auto"` means the harness never picks a model.
- **Multi-turn conversations are never cached** (ADR-0003): agent traffic
  routes live; only single-turn, low-temperature calls can hit the cache.
- **v1 boundary, documented:** `stream: true` returns a 400 (SSE passthrough
  is Phase 6) — set `streaming=False` on the client. Tool parameters
  (`tools`/`tool_choice`) are currently dropped with a warning (Phase 6).
- **Long conversations drift up tiers** (token count spans the whole
  conversation; 400+ tokens → premium). Escape hatches: pin
  `X-Routing-Strategy`, or per-request model pins.
- **Session-sticky routing** (planned): route an entire agent session to one
  tier, keyed on a client-supplied session header — noted for Phase 6.