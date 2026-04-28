# agent-quintet-demo

A small reference agent that uses the [`@mukundakatta`](https://www.npmjs.com/~mukundakatta) agent-stack libraries together.

| library | role in this demo |
|---|---|
| `agentfit` | truncates conversation history to fit a token budget |
| `agentcast` | gets a strictly-typed `ToolCall` from the LLM, retries on JSON / validation failure |
| `agentguard` | declares an egress policy and checks every outbound URL before the call |
| `agentvet` | wraps each tool function; bad args throw a recoverable `ToolArgError` |
| `agentsnap` | snapshots the tool-call trace for CI regression tests |

> **Sibling library**: [`agenttrace`](https://github.com/MukundaKatta/agenttrace)
> adds cost + latency tracking. It's Node-only at v0.1.0 (npm publish coming),
> so this Python demo doesn't include it; the JS reference would wrap each
> `measureLLM(...)` around your LLM call to get a per-step breakdown.

## Run it

```bash
git clone https://github.com/MukundaKatta/agent-quintet-demo.git
cd agent-quintet-demo
uv sync

# Mock mode — deterministic, no API key needed
uv run python -m quintet_demo.agent --mock "what is RAG?"

# Live mode — uses Gemini's free tier
cp .env.example .env
$EDITOR .env   # paste GEMINI_API_KEY (free at aistudio.google.com/app/apikey)
uv run python -m quintet_demo.agent "what is RAG?"
```

## What you'll see

```
Answer: RAG = retrieval-augmented generation.

Trace:
  {'step': 0, 'action': 'search', 'args': {'query': 'rag'},
   'result': 'RAG = retrieve-then-generate; pull docs, then answer ...'}
  {'step': 1, 'action': 'finish', 'answer': 'RAG = retrieval-augmented generation.'}

[snap] trace matches snapshot ✓
```

## How the libraries compose

```python
# agent.py — the inner loop, simplified
from agentfit import fit
from agentcast import cast
from agentguard import check, policy
from agentvet import vet, ToolArgError
from agentsnap import expect_snapshot

network = policy({"network": {"allow": ["*.googleapis.com"]}, "violations": "throw"})

@vet(name="search", schema=search_args_validator, fn=search_impl)
def search(args): ...

async def loop(question: str):
    history = [system_prompt, {"role": "user", "content": question}]
    trace = []
    for step in range(max_steps):
        history = fit(history, max_tokens=4000).messages       # 1. budget
        decision = await cast(                                  # 2. structured output
            llm=gemini, validate=toolcall_validator,
            prompt=history[-1]["content"], system=history[0]["content"],
        )
        check(network, {"kind": "network", "url": llm_url})    # 3. egress
        try:
            result = TOOLS[decision.action](decision.args)     # 4. agentvet via decorator
        except ToolArgError as e:
            history.append({"role": "user", "content": str(e)})
            continue
        trace.append({...})
    expect_snapshot({"steps": trace}, ".snapshots/loop.json")  # 5. snapshot CI
```

## Tests

```bash
uv pip install ".[test]"
uv run pytest -v
```

The first run records snapshots under `.snapshots/`; subsequent runs diff against them. Commit the snapshot files so CI fails on regressions.

## Why these five and not a framework

Each library does one thing, has zero or near-zero dependencies, and is composable. You can use any subset; you don't pay for what you don't use. Drop them into the agent loop you already have — no migration to a new framework, no DSL, no breaking abstractions over the LLM SDK you actually want to call.

## License

MIT
