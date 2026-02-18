"""MCP-based step executor for the plan-execute orchestrator.

Maps to AgentHive's SequentialWorkflow + ReactAgent: each plan step is routed
to the appropriate MCP server, where an LLM selects the tool and generates its
arguments.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from llm import LLMBackend
from .models import Plan, PlanStep, StepResult

_log = logging.getLogger(__name__)

_MCP_ROOT = Path(__file__).parent.parent

# Default server script paths — extend this dict as new MCP servers are added.
DEFAULT_SERVER_PATHS: dict[str, Path] = {
    "IoTAgent": _MCP_ROOT / "servers" / "iot" / "main.py",
    "Utilities": _MCP_ROOT / "servers" / "utilities" / "main.py",
}

_TOOL_SELECTION_PROMPT = """\
You are executing a specific task as part of a larger plan for industrial asset operations.

Task: {task}
Expected output: {expected_output}

You are connected to agent "{agent}" which exposes these tools:
{tools}

Context from completed steps:
{context}

Select the most appropriate tool and generate its arguments.
Respond with a single JSON object (no markdown fences):
{{"tool": "<tool_name>", "args": {{<arg_name>: <value>, ...}}}}

If the task can be answered directly from the context without calling a tool, respond with:
{{"tool": null, "answer": "<answer>"}}

Response:"""


class Executor:
    """Executes plan steps by routing tool calls to MCP servers.

    For each step the assigned MCP server is queried for its tools, the LLM
    selects the best tool and generates its arguments, then the tool is called
    via the MCP stdio protocol.
    """

    def __init__(
        self,
        llm: LLMBackend,
        server_paths: dict[str, Path] | None = None,
    ) -> None:
        self._llm = llm
        self._server_paths = DEFAULT_SERVER_PATHS if server_paths is None else server_paths

    async def get_agent_descriptions(self) -> dict[str, str]:
        """Query each registered MCP server and return a capability summary.

        Returns:
            Mapping of server_name -> human-readable description of its tools,
            suitable for passing to the Planner.
        """
        descriptions: dict[str, str] = {}
        for name, path in self._server_paths.items():
            try:
                tools = await _list_tools(path)
                tool_names = ", ".join(t["name"] for t in tools)
                descriptions[name] = f"Tools: {tool_names}"
            except Exception as exc:  # noqa: BLE001
                descriptions[name] = f"(unavailable: {exc})"
        return descriptions

    async def execute_plan(self, plan: Plan, question: str) -> list[StepResult]:
        """Execute all plan steps in dependency order.

        Args:
            plan: The execution plan.
            question: The original user question (passed as context).

        Returns:
            List of StepResult in execution order.
        """
        ordered = plan.resolved_order()
        total = len(ordered)
        context: dict[int, StepResult] = {}
        results: list[StepResult] = []
        for step in ordered:
            _log.info(
                "Step %d/%d [%s]: %s",
                step.step_number, total, step.agent, step.task,
            )
            result = await self.execute_step(step, context, question)
            if result.success:
                _log.info("Step %d OK.", step.step_number)
            else:
                _log.warning("Step %d FAILED: %s", step.step_number, result.error)
            context[step.step_number] = result
            results.append(result)
        return results

    async def execute_step(
        self,
        step: PlanStep,
        context: dict[int, StepResult],
        question: str,
    ) -> StepResult:
        """Execute a single plan step via MCP tool call.

        1. Resolve the MCP server assigned to this step.
        2. List the server's available tools.
        3. Ask the LLM to select a tool and generate its arguments.
        4. Call the tool and return the result.
        """
        server_path = self._server_paths.get(step.agent)
        if server_path is None:
            return StepResult(
                step_number=step.step_number,
                task=step.task,
                agent=step.agent,
                response="",
                error=(
                    f"Unknown agent '{step.agent}'. "
                    f"Registered agents: {list(self._server_paths)}"
                ),
            )

        try:
            tools = await _list_tools(server_path)
            dep_context = {
                f"Step {n}": r.response
                for n, r in context.items()
                if n in step.dependencies
            }
            tool_call = self._select_tool_call(step, tools, dep_context)

            if tool_call.get("tool") is None:
                response = tool_call.get("answer", "")
            else:
                response = await _call_tool(
                    server_path,
                    tool_call["tool"],
                    tool_call.get("args", {}),
                )

            return StepResult(
                step_number=step.step_number,
                task=step.task,
                agent=step.agent,
                response=response,
            )
        except Exception as exc:  # noqa: BLE001
            return StepResult(
                step_number=step.step_number,
                task=step.task,
                agent=step.agent,
                response="",
                error=str(exc),
            )

    def _select_tool_call(
        self,
        step: PlanStep,
        tools: list[dict],
        dep_context: dict[str, str],
    ) -> dict:
        """Ask the LLM which tool to call and what arguments to pass."""
        tools_text = "\n".join(
            f"- {t['name']}: {t.get('description', 'no description')}"
            for t in tools
        )
        context_text = (
            "\n".join(f"{k}: {v}" for k, v in dep_context.items())
            if dep_context
            else "None"
        )
        prompt = _TOOL_SELECTION_PROMPT.format(
            task=step.task,
            expected_output=step.expected_output,
            agent=step.agent,
            tools=tools_text,
            context=context_text,
        )
        raw = self._llm.generate(prompt)
        return _parse_tool_call(raw)


# ── MCP protocol helpers ──────────────────────────────────────────────────────


async def _list_tools(server_path: Path) -> list[dict]:
    """Connect to an MCP server via stdio and list its tools."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command="python", args=[str(server_path)])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            return [
                {"name": t.name, "description": t.description or ""}
                for t in result.tools
            ]


async def _call_tool(server_path: Path, tool_name: str, args: dict) -> str:
    """Connect to an MCP server via stdio and call a tool."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command="python", args=[str(server_path)])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, args)
            return _extract_content(result.content)


def _extract_content(content: list[Any]) -> str:
    """Extract text from MCP tool call result content."""
    return "\n".join(getattr(item, "text", str(item)) for item in content)


def _parse_tool_call(raw: str) -> dict:
    """Parse LLM output into a {tool, args} dict."""
    text = raw.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner)
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return {"tool": None, "answer": text}
