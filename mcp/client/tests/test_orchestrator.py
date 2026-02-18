"""Tests for PlanExecuteOrchestrator and Executor._select_tool_call."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from client.executor import Executor, _parse_tool_call
from client.orchestrator import PlanExecuteOrchestrator

# ── shared plan strings ───────────────────────────────────────────────────────

_TWO_STEP_PLAN = """\
#Task1: Get IoT sites
#Agent1: IoTAgent
#Dependency1: None
#ExpectedOutput1: List of site names

#Task2: Get current datetime
#Agent2: Utilities
#Dependency2: None
#ExpectedOutput2: Current date and time"""

_TOOL_CALL_SITES = json.dumps({"tool": "sites", "args": {}})
_TOOL_CALL_TIME = json.dumps({"tool": "current_date_time", "args": {}})
_FINAL_ANSWER = "Sites: MAIN. Current time: 2026-02-18T13:00:00."

_MOCK_TOOLS = [
    {"name": "sites", "description": "List IoT sites"},
    {"name": "current_date_time", "description": "Get current datetime"},
]
_TOOL_RESPONSE = json.dumps({"sites": ["MAIN"]})


# ── helper to patch MCP helpers ───────────────────────────────────────────────


def _patch_mcp(tool_response: str = _TOOL_RESPONSE):
    return (
        patch("client.executor._list_tools", new=AsyncMock(return_value=_MOCK_TOOLS)),
        patch(
            "client.executor._call_tool", new=AsyncMock(return_value=tool_response)
        ),
    )


# ── orchestrator tests ────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_orchestrator_run_returns_result(sequential_llm):
    llm = sequential_llm(
        [
            _TWO_STEP_PLAN,  # planner call
            _TOOL_CALL_SITES,  # tool selection — step 1
            _TOOL_CALL_TIME,  # tool selection — step 2
            _FINAL_ANSWER,  # summarisation
        ]
    )
    with _patch_mcp()[0], _patch_mcp()[1]:
        result = await PlanExecuteOrchestrator(llm).run("What are the IoT sites?")

    assert result.question == "What are the IoT sites?"
    assert result.answer == _FINAL_ANSWER
    assert len(result.plan.steps) == 2
    assert len(result.history) == 2


@pytest.mark.anyio
async def test_orchestrator_all_steps_succeed(sequential_llm):
    llm = sequential_llm(
        [_TWO_STEP_PLAN, _TOOL_CALL_SITES, _TOOL_CALL_TIME, _FINAL_ANSWER]
    )
    with _patch_mcp()[0], _patch_mcp()[1]:
        result = await PlanExecuteOrchestrator(llm).run("Q")

    assert all(r.success for r in result.history)


@pytest.mark.anyio
async def test_orchestrator_unknown_agent_recorded_as_error(sequential_llm):
    bad_plan = (
        "#Task1: Do something\n"
        "#Agent1: GhostAgent\n"
        "#Dependency1: None\n"
        "#ExpectedOutput1: Result\n"
    )
    llm = sequential_llm([bad_plan, _FINAL_ANSWER])
    with _patch_mcp()[0], _patch_mcp()[1]:
        result = await PlanExecuteOrchestrator(llm).run("Q")

    assert len(result.history) == 1
    assert result.history[0].success is False
    assert "GhostAgent" in result.history[0].error


@pytest.mark.anyio
async def test_orchestrator_direct_answer_from_context(sequential_llm):
    """If the LLM returns tool=null the executor uses the inline answer."""
    direct = json.dumps({"tool": None, "answer": "42"})
    single_step_plan = (
        "#Task1: Answer from context\n"
        "#Agent1: IoTAgent\n"
        "#Dependency1: None\n"
        "#ExpectedOutput1: Answer\n"
    )
    llm = sequential_llm([single_step_plan, direct, "Final: 42"])
    with _patch_mcp()[0], _patch_mcp()[1]:
        result = await PlanExecuteOrchestrator(llm).run("Simple Q")

    assert result.history[0].response == "42"
    assert result.history[0].success is True


# ── executor unit tests ───────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_executor_unknown_agent(mock_llm):
    llm = mock_llm(_TOOL_CALL_SITES)
    executor = Executor(llm, server_paths={})  # no servers registered

    from client.models import Plan, PlanStep

    plan = Plan(
        steps=[
            PlanStep(
                step_number=1,
                task="get sites",
                agent="IoTAgent",
                dependencies=[],
                expected_output="sites",
            )
        ],
        raw="",
    )
    with _patch_mcp()[0], _patch_mcp()[1]:
        results = await executor.execute_plan(plan, "Q")

    assert results[0].success is False
    assert "IoTAgent" in results[0].error


@pytest.mark.anyio
async def test_executor_get_agent_descriptions(mock_llm):
    llm = mock_llm()
    executor = Executor(llm, server_paths={"TestServer": None})

    with patch(
        "client.executor._list_tools",
        new=AsyncMock(
            return_value=[{"name": "foo", "description": "does foo"}]
        ),
    ):
        descs = await executor.get_agent_descriptions()

    assert "TestServer" in descs
    assert "foo" in descs["TestServer"]


# ── _parse_tool_call tests ────────────────────────────────────────────────────


def test_parse_tool_call_plain_json():
    raw = '{"tool": "sites", "args": {}}'
    result = _parse_tool_call(raw)
    assert result["tool"] == "sites"
    assert result["args"] == {}


def test_parse_tool_call_with_markdown_fence():
    raw = '```json\n{"tool": "history", "args": {"site_name": "MAIN"}}\n```'
    result = _parse_tool_call(raw)
    assert result["tool"] == "history"
    assert result["args"]["site_name"] == "MAIN"


def test_parse_tool_call_null_tool():
    raw = '{"tool": null, "answer": "42"}'
    result = _parse_tool_call(raw)
    assert result["tool"] is None
    assert result["answer"] == "42"


def test_parse_tool_call_embedded_json():
    raw = "Here is my response: {\"tool\": \"sites\", \"args\": {}} done."
    result = _parse_tool_call(raw)
    assert result["tool"] == "sites"


def test_parse_tool_call_unrecoverable_returns_direct_answer():
    raw = "I cannot decide which tool to use."
    result = _parse_tool_call(raw)
    assert result["tool"] is None
    assert result["answer"] == raw
