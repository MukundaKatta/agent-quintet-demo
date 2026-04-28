"""A toy tool registry — two tools the agent can call.

Each tool is wrapped with `agentvet.vet()` so bad LLM args throw a
recoverable `ToolArgError` instead of crashing the agent.
"""

from __future__ import annotations

from typing import Any, Callable

from agentvet import vet
from pydantic import BaseModel, Field, ValidationError


def _pydantic_validator(model: type[BaseModel]) -> Callable[[Any], dict]:
    """agentvet expects {valid: bool, value/error}. Adapt a pydantic model."""

    def validate(args: Any) -> dict:
        try:
            value = model.model_validate(args)
            return {"valid": True, "value": value}
        except ValidationError as e:
            return {"valid": False, "error": str(e)}

    return validate


class SearchArgs(BaseModel):
    query: str = Field(min_length=1, max_length=200)
    limit: int = Field(default=3, ge=1, le=10)


class CalcArgs(BaseModel):
    expression: str = Field(min_length=1, max_length=200)


def _search_impl(args: SearchArgs) -> str:
    """Pretend search — returns canned 'results' for predictable demos."""
    canned = {
        "agent": "Agents are LLMs in a loop with tool access.",
        "rag": "RAG = retrieve-then-generate; pull docs, then answer with them in context.",
        "embedding": "Embeddings are dense float vectors representing semantic meaning.",
        "llm": "Large language models predict next tokens; modern ones are decoder-only Transformers.",
    }
    q = args.query.lower()
    hits = [canned[k] for k in canned if k in q][: args.limit]
    return " // ".join(hits) if hits else f"No results for {args.query!r}"


def _calc_impl(args: CalcArgs) -> str:
    """Tiny safe-ish eval. Demo only — never use bare `eval` in real code."""
    allowed = "0123456789+-*/() ."
    if not all(c in allowed for c in args.expression):
        return f"Calc rejected: only digits + - * / ( ) and spaces are allowed."
    try:
        return f"= {eval(args.expression)}"  # noqa: S307 — demo only
    except Exception as e:  # noqa: BLE001
        return f"Calc error: {e}"


search = vet(name="search", schema=_pydantic_validator(SearchArgs), fn=_search_impl)
calc = vet(name="calc", schema=_pydantic_validator(CalcArgs), fn=_calc_impl)


TOOLS: dict[str, Callable[[Any], Any]] = {
    "search": search,
    "calc": calc,
}

TOOL_DESCRIPTIONS = """
- search(query: string, limit: int = 3) — search a small knowledge base; returns matching snippets.
- calc(expression: string) — evaluate a simple arithmetic expression.
"""
