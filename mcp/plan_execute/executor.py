"""MCP-based step executor for the plan-execute orchestrator.

Each PlanStep already contains the tool name and arguments decided by the
planner, so the executor calls the tool directly with no additional LLM calls.
Step argument values may contain {{step_N}} placeholders that are resolved
from prior step results at execution time.
"""

from __future__ import annotations

import json
import logging
import re
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

_PLACEHOLDER_RE = re.compile(r"\{\{step_(\d+)\}\}")


class Executor:
    """Executes plan steps by routing tool calls to MCP servers.

    The tool name and arguments are taken directly from the PlanStep (decided
    by the Planner).  {{step_N}} placeholders in argument values are resolved
    from prior step results before the tool is called.
    """

    def __init__(
        self,
        llm: LLMBackend,
        server_paths: dict[str, Path] | None = None,
    ) -> None:
        self._llm = llm
        self._server_paths = DEFAULT_SERVER_PATHS if server_paths is None else server_paths

    async def get_agent_descriptions(self) -> dict[str, str]:
        """Query each registered MCP server and return a formatted tool signature.

        Returns:
            Mapping of server_name -> multi-line string listing tool signatures
            with parameter names and types, suitable for passing to the Planner.
        """
        descriptions: dict[str, str] = {}
        for name, path in self._server_paths.items():
            try:
                tools = await _list_tools(path)
                lines = []
                for t in tools:
                    params = ", ".join(
                        f"{p['name']}: {p['type']}{'?' if not p['required'] else ''}"
                        for p in t.get("parameters", [])
                    )
                    lines.append(f"  - {t['name']}({params}): {t['description']}")
                descriptions[name] = "\n".join(lines)
            except Exception as exc:  # noqa: BLE001
                descriptions[name] = f"  (unavailable: {exc})"
        return descriptions

    async def execute_plan(self, plan: Plan, question: str) -> list[StepResult]:
        """Execute all plan steps in dependency order."""
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
        """Execute a single plan step.

        1. Resolve the MCP server assigned to this step.
        2. If the step has no tool (tool is "none"/empty), return expected_output.
        3. Resolve {{step_N}} placeholders in tool_args from prior results.
        4. Call the tool via MCP stdio and return the result.
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

        # No tool — answer comes directly from the plan
        if not step.tool or step.tool.lower() in ("none", "null"):
            return StepResult(
                step_number=step.step_number,
                task=step.task,
                agent=step.agent,
                response=step.expected_output,
            )

        try:
            resolved_args = _resolve_args(step.tool_args, context)
            response = await _call_tool(server_path, step.tool, resolved_args)
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


# ── MCP protocol helpers ──────────────────────────────────────────────────────


async def _list_tools(server_path: Path) -> list[dict]:
    """Connect to an MCP server via stdio and list its tools with parameter info."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command="python", args=[str(server_path)])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            tools = []
            for t in result.tools:
                schema = t.inputSchema or {}
                props = schema.get("properties", {})
                required = set(schema.get("required", []))
                parameters = [
                    {
                        "name": k,
                        "type": v.get("type", "any"),
                        "required": k in required,
                    }
                    for k, v in props.items()
                ]
                tools.append({
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": parameters,
                })
            return tools


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


def _resolve_args(args: dict, context: dict[int, StepResult]) -> dict:
    """Replace {{step_N}} placeholders in arg values with prior step responses."""
    resolved = {}
    for key, val in args.items():
        if isinstance(val, str):
            def _sub(m: re.Match) -> str:
                n = int(m.group(1))
                return context[n].response if n in context else m.group(0)
            resolved[key] = _PLACEHOLDER_RE.sub(_sub, val)
        else:
            resolved[key] = val
    return resolved


def _parse_tool_call(raw: str) -> dict:
    """Parse LLM output into a {tool, args} dict.

    Kept as a utility; no longer used in the main execution path.
    """
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
