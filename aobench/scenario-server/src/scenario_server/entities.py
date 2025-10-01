from dataclasses import dataclass


@dataclass
class ScenarioType:
    id: str
    title: str
    description: str


@dataclass
class Scenario:
    id: str
    query: str


@dataclass
class ScenarioSet:
    scenarios: list[Scenario]

    def get_scenario(self, sid: str):
        return next((entry for entry in self.scenarios if entry.id == sid), None)


@dataclass
class SubmissionAnswer:
    scenario_id: str
    answer: str


@dataclass
class SubmissionScore:
    scenario_id: str
    correct: bool
    details: list


@dataclass
class Submission:
    experiment_id: str
    run_id: str
    scenario_set_id: str
    submission: list[SubmissionAnswer]
