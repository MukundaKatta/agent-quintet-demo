"""Reference agent loop composing all 5 @mukundakatta libraries.

```
        ┌─────────────────────────────────────┐
        │ 1. agentfit.fit() — keep history under budget
        │ 2. agentcast.cast() — get a structured tool call from LLM
        │ 3. agentguard.check() — verify the agent isn't egressing somewhere weird
        │ 4. agentvet — already wraps each tool; validates args before fn runs
        │ 5. agentsnap — record the trace so CI catches regressions
        └─────────────────────────────────────┘
```

Run live (needs GEMINI_API_KEY in env or .env):

    uv run python -m quintet_demo.agent "what is RAG?"

Run with the deterministic mock LLM (no key needed):

    uv run python -m quintet_demo.agent --mock "what is RAG?"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any

from agentcast import cast
from agentfit import fit
from agentguard import check, policy
from agentsnap import expect_snapshot
from agentvet import ToolArgError
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from .tools import TOOL_DESCRIPTIONS, TOOLS

load_dotenv()


# ----- Validators / schemas ----- #

class ToolCall(BaseModel):
    """LLM-emitted next step."""
    action: str = Field(description="One of: search, calc, finish")
    args: dict[str, Any] = Field(default_factory=dict)
    answer: str = Field(default="", description="When action=finish, the user-facing answer.")


def _toolcall_validate(payload: Any) -> dict:
    try:
        value = ToolCall.model_validate(payload)
        if value.action not in {"search", "calc", "finish"}:
            return {"valid": False, "error": f"action must be search|calc|finish, got {value.action!r}"}
        return {"valid": True, "value": value}
    except Exception as e:  # noqa: BLE001
        return {"valid": False, "error": str(e)}


# ----- Network egress policy (agentguard) ----- #

NETWORK_POLICY = policy({
    "network": {
        "allow": ["generativelanguage.googleapis.com", "*.googleapis.com"],
        "methods": ["POST", "GET"],
    },
    "violations": "throw",
})


def assert_egress_ok(url: str) -> None:
    """Call before any outbound request. Throws on policy violation."""
    decision = check(NETWORK_POLICY, {"kind": "network", "url": url, "method": "POST"})
    if decision.action == "deny":
        raise RuntimeError(f"Egress blocked: {decision.reason} ({decision.detail})")


# ----- LLM adapters (live + mock) ----- #

async def gemini_llm(messages: list[dict]) -> str:
    """Live Gemini call. Goes through the egress check first."""
    api_key = os.environ["GEMINI_API_KEY"]
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")
    assert_egress_ok("https://generativelanguage.googleapis.com/v1beta/models/")

    from google import genai
    client = genai.Client(api_key=api_key)
    contents = "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)
    resp = client.models.generate_content(model=model, contents=contents)
    return (resp.text or "").strip()


def make_mock_llm(scripted: list[dict]):
    """Deterministic mock that returns each scripted response in turn."""
    iter_resp = iter(scripted)

    async def mock(_messages: list[dict]) -> str:
        try:
            return json.dumps(next(iter_resp))
        except StopIteration:
            return json.dumps({"action": "finish", "answer": "(mock exhausted)"})

    return mock


# ----- The agent loop ----- #

SYSTEM_PROMPT = (
    "You are an agent that answers questions by calling tools.\n\n"
    f"Available tools:\n{TOOL_DESCRIPTIONS}\n"
    "Reply with strict JSON: {\"action\": \"search|calc|finish\", \"args\": {...}, \"answer\": \"...\"}.\n"
    "Use action=finish with the user-facing answer when you're done.\n"
    "Use args={} for finish."
)


@dataclass
class Trace:
    """Tool-call trace — fed to agentsnap.expect_snapshot for CI testing."""
    steps: list[dict] = field(default_factory=list)


async def run(question: str, *, llm, max_steps: int = 6) -> tuple[str, Trace]:
    history: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    trace = Trace()

    for step in range(max_steps):
        # 1. agentfit — truncate before the LLM call (mock budget for the demo)
        fitted = fit(history, max_tokens=4000)
        bounded = fitted.messages if hasattr(fitted, "messages") else fitted

        # 2. agentcast — get a validated ToolCall (retries on JSON / validation failure)
        decision = await cast(
            llm=llm,
            validate=_toolcall_validate,
            prompt=bounded[-1]["content"],
            system=bounded[0]["content"],
            max_retries=2,
        )

        if decision.action == "finish":
            trace.steps.append({"step": step, "action": "finish", "answer": decision.answer})
            return decision.answer, trace

        # 4. agentvet runs inside the tool wrapper; ToolArgError → recoverable
        tool = TOOLS.get(decision.action)
        if tool is None:
            history.append({"role": "user",
                            "content": f"unknown tool {decision.action!r}; pick search|calc|finish"})
            trace.steps.append({"step": step, "action": decision.action, "result": "(unknown tool)"})
            continue

        try:
            result = tool(decision.args)
        except ToolArgError as e:
            # Hand the error back to the LLM as the next user turn
            history.append({"role": "user", "content": f"tool {e.tool_name} rejected args: {e}"})
            trace.steps.append({"step": step, "action": decision.action, "args": decision.args, "error": str(e)})
            continue

        history.append({"role": "assistant",
                        "content": json.dumps({"action": decision.action, "args": decision.args})})
        history.append({"role": "user",
                        "content": f"tool {decision.action} returned: {result}"})
        trace.steps.append({"step": step, "action": decision.action, "args": decision.args, "result": result})

    return "(max_steps exceeded)", trace


# ----- CLI ----- #

def _main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?", default="what is RAG?")
    ap.add_argument("--mock", action="store_true",
                    help="Use a deterministic mock LLM instead of Gemini.")
    args = ap.parse_args()

    if args.mock:
        # Deterministic 2-step trace for the demo: search, then finish
        llm = make_mock_llm([
            {"action": "search", "args": {"query": "rag"}},
            {"action": "finish", "answer": "RAG = retrieval-augmented generation."},
        ])
    else:
        if not os.environ.get("GEMINI_API_KEY"):
            print("GEMINI_API_KEY not set. Either set it (see .env.example) or run with --mock.",
                  file=sys.stderr)
            return 1
        llm = gemini_llm

    answer, trace = asyncio.run(run(args.question, llm=llm))
    print(f"\nAnswer: {answer}\n")
    print("Trace:")
    for s in trace.steps:
        print(f"  {s}")

    # 5. agentsnap — record/diff the trace for CI
    if args.mock:
        try:
            expect_snapshot({"steps": trace.steps}, ".snapshots/quintet-demo-mock.json")
            print("\n[snap] trace matches snapshot ✓")
        except Exception as e:  # noqa: BLE001
            print(f"\n[snap] {e}")

    return 0


if __name__ == "__main__":
    sys.exit(_main())
