"""Manual smoke test — run against a live gateway (`make run` in another terminal).

The Day-1 acceptance script: the official OpenAI SDK against the gateway,
including the four contract headers and the determinism check.
Grows into scripts/demo.sh on Day 7.
"""

from openai import BadRequestError, OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="gw-demo-key")

print("── 1. basic completion ─────────────────────────────")
r = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Hello, gateway."}],
)
print("content :", r.choices[0].message.content)
print("usage   :", r.usage.prompt_tokens, "in /", r.usage.completion_tokens, "out")

print("\n── 2. contract headers (DESIGN.md §7.2) ───────────")
raw = client.chat.completions.with_raw_response.create(
    model="auto",
    messages=[{"role": "user", "content": "Hello, gateway."}],
)
for header in ("x-cache-hit", "x-provider-used", "x-routing-tier", "x-cost-usd"):
    print(f"{header:18}: {raw.headers[header]}")

print("\n── 3. determinism (Day 3's cache builds on this) ──")
r2 = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Hello, gateway."}],
)
print("same response id :", r.id == r2.id)
print("same content    :", r.choices[0].message.content == r2.choices[0].message.content)

print("\n── 4. error paths through the SDK ─────────────────")
try:
    client.chat.completions.create(
        model="auto", messages=[{"role": "user", "content": "hi"}], stream=True
    )
except BadRequestError as e:
    print("stream=true → 400, message:", str(e)[:90])

try:
    client.chat.completions.create(model="auto", messages=[])
except BadRequestError as e:
    print("empty messages → 400, message:", str(e)[:90])
