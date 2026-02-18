"""CLI entry point for the plan-execute runner.

Usage:
    plan-execute "What assets are available at site MAIN?"
    plan-execute --platform watsonx --model-id 19 --show-plan "List sensors for asset CH-1"
    plan-execute --server FMSRAgent=servers/fmsr/main.py "What are the failure modes?"
    plan-execute --json "What is the current time?"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

_PLATFORMS = ["watsonx", "litellm"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plan-execute",
        description="Run a question through the MCP plan-execute workflow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
environment variables (watsonx platform):
  WATSONX_APIKEY        IBM WatsonX API key (required)
  WATSONX_PROJECT_ID    IBM WatsonX project ID (required)
  WATSONX_URL           IBM WatsonX endpoint (optional, defaults to us-south)

examples:
  plan-execute "What assets are at site MAIN?"
  plan-execute --platform watsonx --model-id 19 --show-plan "List sensors for asset CH-1"
  plan-execute --server FMSRAgent=servers/fmsr/main.py "What are the failure modes?"
  plan-execute --show-history --json "How many IoT observations exist for CH-1?"
""",
    )
    parser.add_argument("question", help="The question to answer.")
    parser.add_argument(
        "--platform",
        choices=_PLATFORMS,
        default="watsonx",
        help="LLM platform to use (default: watsonx).",
    )
    parser.add_argument(
        "--model-id",
        default="meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
        metavar="MODEL_ID",
        help="Model ID string for the selected platform (default: meta-llama/llama-4-maverick-17b-128e-instruct-fp8).",
    )
    parser.add_argument(
        "--server",
        action="append",
        metavar="NAME=PATH",
        dest="servers",
        default=[],
        help=(
            "Register an MCP server as NAME=PATH. "
            "Overrides the default IoTAgent and Utilities servers. "
            "Repeatable."
        ),
    )
    parser.add_argument(
        "--show-plan",
        action="store_true",
        help="Print the generated plan before execution.",
    )
    parser.add_argument(
        "--show-history",
        action="store_true",
        help="Print each step result after execution.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output the full result (answer, plan, history) as JSON.",
    )
    return parser


def _build_llm(platform: str, model_id: int):
    """Instantiate the LLM backend for the given platform."""
    if platform == "watsonx":
        try:
            from llm.watsonx import WatsonXLLM
        except ImportError as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(1)
        try:
            return WatsonXLLM(model_id=model_id)
        except KeyError as exc:
            print(f"error: missing environment variable {exc}", file=sys.stderr)
            sys.exit(1)

    if platform == "litellm":
        print("error: litellm platform is not yet implemented", file=sys.stderr)
        sys.exit(1)

    print(f"error: unknown platform {platform!r}", file=sys.stderr)
    sys.exit(1)


def _parse_servers(entries: list[str]) -> dict[str, Path] | None:
    """Parse NAME=PATH pairs into a server_paths dict, or None if empty."""
    if not entries:
        return None
    result: dict[str, Path] = {}
    for entry in entries:
        if "=" not in entry:
            print(
                f"error: --server requires NAME=PATH format, got: {entry!r}",
                file=sys.stderr,
            )
            sys.exit(1)
        name, _, path = entry.partition("=")
        result[name.strip()] = Path(path.strip())
    return result


async def _run(args: argparse.Namespace) -> None:
    from plan_execute.runner import PlanExecuteRunner

    llm = _build_llm(args.platform, args.model_id)
    server_paths = _parse_servers(args.servers)
    runner = PlanExecuteRunner(llm=llm, server_paths=server_paths)
    result = await runner.run(args.question)

    if args.show_plan:
        print("=== Plan ===")
        for step in result.plan.steps:
            deps = ", ".join(f"#{d}" for d in step.dependencies) or "none"
            print(f"  Step {step.step_number} [{step.agent}]: {step.task}")
            print(f"    deps={deps}  expected: {step.expected_output}")
        print()

    if args.show_history:
        print("=== Execution History ===")
        for r in result.history:
            status = "OK" if r.success else "ERROR"
            print(f"  [{status}] Step {r.step_number} ({r.agent}): {r.task}")
            detail = r.response if r.success else f"Error: {r.error}"
            snippet = detail[:200] + ("..." if len(detail) > 200 else "")
            print(f"    {snippet}")
        print()

    if args.output_json:
        output = {
            "question": result.question,
            "answer": result.answer,
            "plan": [
                {
                    "step": s.step_number,
                    "task": s.task,
                    "agent": s.agent,
                    "dependencies": s.dependencies,
                    "expected_output": s.expected_output,
                }
                for s in result.plan.steps
            ],
            "history": [
                {
                    "step": r.step_number,
                    "task": r.task,
                    "agent": r.agent,
                    "response": r.response,
                    "error": r.error,
                    "success": r.success,
                }
                for r in result.history
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        print(result.answer)


def main() -> None:
    args = _build_parser().parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
