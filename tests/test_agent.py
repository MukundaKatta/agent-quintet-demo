"""Snapshot test for the demo agent. Uses the deterministic mock LLM so the
trace is reproducible and `expect_snapshot` can compare against a baseline.
"""

from __future__ import annotations

import json

import pytest
from agentsnap import AgentSnapshotMismatch, expect_snapshot

from quintet_demo.agent import make_mock_llm, run


@pytest.mark.asyncio
async def test_demo_trace_matches_snapshot():
    llm = make_mock_llm([
        {"action": "search", "args": {"query": "rag"}},
        {"action": "finish", "answer": "RAG = retrieval-augmented generation."},
    ])
    answer, trace = await run("what is RAG?", llm=llm)
    assert "RAG" in answer

    # First run records the snapshot; subsequent runs assert against it.
    expect_snapshot({"steps": trace.steps}, ".snapshots/demo-rag-trace.json")


@pytest.mark.asyncio
async def test_bad_tool_args_recover():
    """vet() should surface a ToolArgError back to the LLM and let it retry."""
    llm = make_mock_llm([
        {"action": "search", "args": {"query": ""}},          # too short → ToolArgError
        {"action": "search", "args": {"query": "agent"}},      # valid retry
        {"action": "finish", "answer": "Agents are LLMs in a loop."},
    ])
    answer, trace = await run("what is an agent?", llm=llm)
    # Trace should show one error step then a successful one
    errors = [s for s in trace.steps if "error" in s]
    successes = [s for s in trace.steps if s.get("action") == "search" and "result" in s]
    assert len(errors) >= 1
    assert len(successes) >= 1
    assert "Agent" in answer or "agent" in answer
