from __future__ import annotations
from typing import List, Optional, Dict, Any
from uuid import UUID
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic import ConfigDict


class FlexibleModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class DataSpec(FlexibleModel):
    type: str
    origin: Optional[str] = None
    dest: Optional[str] = None


class ToolSpec(FlexibleModel):
    type: str
    name: Optional[str] = None
    url: Optional[str] = None
    port: Optional[int] = None
    spec_url: Optional[str] = None
    api_url: Optional[str] = None


class Dependency(FlexibleModel):
    type: str
    name: Optional[str] = None


class AgentSpec(FlexibleModel):
    name: str
    url: Optional[str] = None
    port: Optional[int] = None
    dependencies: Optional[List[Dependency]] = None


class UtterancePart(FlexibleModel):
    kind: str
    text: Optional[str] = None


class ExecutionUtterance(FlexibleModel):
    role: Optional[str] = None
    parts: Optional[List[UtterancePart]] = None


class ExecutionNode(FlexibleModel):
    name: str
    action: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None


class ExecutionLink(FlexibleModel):
    source: str
    target: str


class ScenarioEvaluationResult(FlexibleModel):
    type: Optional[str] = None
    value: Optional[Any] = None


class ScenarioEvaluation(FlexibleModel):
    deterministic: Optional[bool] = False
    result: Optional[ScenarioEvaluationResult] = None
    metric: Optional[List[str]] = None


class Scenario(FlexibleModel):
    uuid: UUID = Field(..., description="Unique identifier for scenario")
    name: Optional[str] = None
    category: Optional[str] = None
    type: Optional[str] = None
    expected_result: Optional[Any] = None
    data: Optional[List[DataSpec]] = None
    tools: Optional[List[ToolSpec]] = None
    agents: Optional[List[AgentSpec]] = None
    environment: Optional[Dict[str, Any]] = None
    execution: Optional[Dict[str, Any]] = None
    planning_steps: Optional[List[str]] = None
    execution_steps: Optional[List[ExecutionNode]] = None
    execution_links: Optional[List[ExecutionLink]] = None
    possible_alternatives: Optional[Dict[str, Any]] = None
    evaluation: Optional[ScenarioEvaluation] = None

    @field_validator("uuid", mode="before")
    @classmethod
    def _coerce_uuid(cls, v):
        if isinstance(v, UUID):
            return v
        return UUID(str(v))

    @model_validator(mode="after")
    def _semantic_checks(self):
        if self.execution is not None:
            if not isinstance(self.execution, dict):
                raise ValueError("execution must be a dict")
            if "entrypoint" not in self.execution:
                raise ValueError("execution must contain 'entrypoint'")
            if "utterance" in self.execution:
                ut = self.execution["utterance"]
                if not isinstance(ut, dict):
                    raise ValueError("execution.utterance must be a dict")
                parts = ut.get("parts")
                if parts is not None:
                    if not isinstance(parts, list):
                        raise ValueError("execution.utterance.parts must be a list")
                    for i, p in enumerate(parts):
                        if not isinstance(p, dict):
                            raise ValueError(f"execution.utterance.parts[{i}] must be a dict")
                        if "kind" not in p and "text" not in p:
                            raise ValueError(f"execution.utterance.parts[{i}] must contain 'kind' or 'text'")

        if self.tools:
            if not isinstance(self.tools, list):
                raise ValueError("tools must be a list")
            for i, t in enumerate(self.tools):
                t_type = t.get("type") if isinstance(t, dict) else getattr(t, "type", None)
                if not t_type:
                    raise ValueError(f"tools[{i}] missing 'type'")

        if self.agents:
            if not isinstance(self.agents, list):
                raise ValueError("agents must be a list")
            for i, a in enumerate(self.agents):
                name = a.get("name") if isinstance(a, dict) else getattr(a, "name", None)
                if not name:
                    raise ValueError(f"agents[{i}] missing 'name'")

        if self.evaluation and getattr(self.evaluation, "deterministic", False):
            res = getattr(self.evaluation, "result", None)
            if res is None or getattr(res, "value", None) is None:
                raise ValueError("evaluation.result.value required when deterministic")

        return self


class Utterance(FlexibleModel):
    id: Optional[int] = None
    text: str
    type: Optional[str] = None
    deterministic: Optional[bool] = False
    characteristic_form: Optional[str] = None
    note: Optional[str] = None
    category: Optional[str] = None


def generate_json_schema(model: type[BaseModel]) -> Dict[str, Any]:
    return model.model_json_schema()


def load_scenario(obj: Dict[str, Any]) -> Scenario:
    return Scenario.model_validate(obj)


def load_utterance(obj: Dict[str, Any]) -> Utterance:
    return Utterance.model_validate(obj)
