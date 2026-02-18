"""LLM-based plan generation for the plan-execute orchestrator.

Maps to AgentHive's PlanningWorkflow.generate_steps(), preserving the same
#Task / #Agent / #Dependency / #ExpectedOutput plan format so plans remain
readable and compatible with existing benchmark tooling.
"""

from __future__ import annotations

import re

from llm import LLMBackend
from .models import Plan, PlanStep

_PLAN_PROMPT = """\
You are a planning assistant for industrial asset operations and maintenance.

Decompose the question below into a sequence of subtasks. Assign each subtask
to exactly one of the available agents.

Available agents:
{agents}

Output format — follow this template exactly:

#Task1: <task description>
#Agent1: <exact agent name from the list above>
#Dependency1: None
#ExpectedOutput1: <what this step should produce>

#Task2: <task description>
#Agent2: <exact agent name from the list above>
#Dependency2: #S1
#ExpectedOutput2: <what this step should produce>

Rules:
- Agent names must exactly match those listed above.
- Dependencies use #S<N> notation (e.g., #S1, #S2). Use "None" if there are
  no dependencies.
- Keep tasks specific and actionable.

Question: {question}

Plan:
"""

_TASK_RE = re.compile(r"#Task(\d+):\s*(.+)")
_AGENT_RE = re.compile(r"#Agent(\d+):\s*(.+)")
_DEP_RE = re.compile(r"#Dependency(\d+):\s*(.+)")
_OUTPUT_RE = re.compile(r"#ExpectedOutput(\d+):\s*(.+)")
_DEP_NUM_RE = re.compile(r"#S(\d+)")


def parse_plan(raw: str) -> Plan:
    """Parse an LLM-generated plan string into a Plan object.

    Supports the same #Task / #Agent / #Dependency / #ExpectedOutput format
    used by AgentHive's PlanningWorkflow.
    """
    tasks = {int(m.group(1)): m.group(2).strip() for m in _TASK_RE.finditer(raw)}
    agents = {int(m.group(1)): m.group(2).strip() for m in _AGENT_RE.finditer(raw)}
    deps_raw = {int(m.group(1)): m.group(2).strip() for m in _DEP_RE.finditer(raw)}
    outputs = {int(m.group(1)): m.group(2).strip() for m in _OUTPUT_RE.finditer(raw)}

    steps = [
        PlanStep(
            step_number=n,
            task=tasks[n],
            agent=agents.get(n, ""),
            dependencies=(
                []
                if deps_raw.get(n, "None").strip().lower() == "none"
                else [int(x) for x in _DEP_NUM_RE.findall(deps_raw.get(n, ""))]
            ),
            expected_output=outputs.get(n, ""),
        )
        for n in sorted(tasks)
    ]
    return Plan(steps=steps, raw=raw)


class Planner:
    """Decomposes a question into a structured execution plan using an LLM."""

    def __init__(self, llm: LLMBackend) -> None:
        self._llm = llm

    def generate_plan(
        self,
        question: str,
        agent_descriptions: dict[str, str],
    ) -> Plan:
        """Generate a plan for a question given available agents.

        Args:
            question: The user question to answer.
            agent_descriptions: Mapping of agent_name -> capability description.

        Returns:
            A Plan with PlanStep objects ready for execution.
        """
        agents_text = "\n".join(
            f"- {name}: {desc}" for name, desc in agent_descriptions.items()
        )
        prompt = _PLAN_PROMPT.format(agents=agents_text, question=question)
        raw = self._llm.generate(prompt)
        return parse_plan(raw)
