"""Tests for the Planner and parse_plan()."""

from client.planner import Planner, parse_plan

_TWO_STEP = """\
#Task1: List all available IoT sites
#Agent1: IoTAgent
#Dependency1: None
#ExpectedOutput1: A list of site names

#Task2: Get assets at site MAIN
#Agent2: IoTAgent
#Dependency2: #S1
#ExpectedOutput2: A list of asset IDs"""

_MULTI_DEP = """\
#Task1: Get sites
#Agent1: IoTAgent
#Dependency1: None
#ExpectedOutput1: Sites

#Task2: Get current time
#Agent2: Utilities
#Dependency2: None
#ExpectedOutput2: Current time

#Task3: Combine results
#Agent3: Utilities
#Dependency3: #S1, #S2
#ExpectedOutput3: Combined output"""

_NO_TASKS = "No tasks here."


class TestParsePlan:
    def test_two_steps_parsed(self):
        plan = parse_plan(_TWO_STEP)
        assert len(plan.steps) == 2

    def test_step_numbers(self):
        plan = parse_plan(_TWO_STEP)
        assert plan.steps[0].step_number == 1
        assert plan.steps[1].step_number == 2

    def test_task_text(self):
        plan = parse_plan(_TWO_STEP)
        assert "IoT sites" in plan.steps[0].task
        assert "assets" in plan.steps[1].task

    def test_agent_names(self):
        plan = parse_plan(_TWO_STEP)
        assert plan.steps[0].agent == "IoTAgent"
        assert plan.steps[1].agent == "IoTAgent"

    def test_no_dependency(self):
        plan = parse_plan(_TWO_STEP)
        assert plan.steps[0].dependencies == []

    def test_single_dependency(self):
        plan = parse_plan(_TWO_STEP)
        assert plan.steps[1].dependencies == [1]

    def test_multiple_dependencies(self):
        plan = parse_plan(_MULTI_DEP)
        assert set(plan.steps[2].dependencies) == {1, 2}

    def test_raw_preserved(self):
        plan = parse_plan(_TWO_STEP)
        assert plan.raw == _TWO_STEP

    def test_expected_output_captured(self):
        plan = parse_plan(_TWO_STEP)
        assert "site names" in plan.steps[0].expected_output.lower()

    def test_empty_input_yields_empty_plan(self):
        plan = parse_plan("")
        assert plan.steps == []

    def test_no_matching_blocks_yields_empty_plan(self):
        plan = parse_plan(_NO_TASKS)
        assert plan.steps == []


class TestPlanner:
    def test_generate_plan_uses_llm_output(self, mock_llm):
        llm = mock_llm(_TWO_STEP)
        planner = Planner(llm)
        plan = planner.generate_plan(
            "List all assets",
            {"IoTAgent": "Tools: sites, assets, sensors, history"},
        )
        assert len(plan.steps) == 2
        assert plan.steps[0].agent == "IoTAgent"

    def test_generate_plan_prompt_contains_question(self, mock_llm, monkeypatch):
        captured = []
        llm = mock_llm(_TWO_STEP)
        original = llm.generate
        llm.generate = lambda p, **kw: (captured.append(p), original(p))[1]

        Planner(llm).generate_plan(
            "What sensors exist for CH-1?",
            {"IoTAgent": "IoT tools"},
        )
        assert "What sensors exist for CH-1?" in captured[0]

    def test_generate_plan_prompt_contains_agent_names(self, mock_llm, monkeypatch):
        captured = []
        llm = mock_llm(_TWO_STEP)
        original = llm.generate
        llm.generate = lambda p, **kw: (captured.append(p), original(p))[1]

        Planner(llm).generate_plan(
            "Q",
            {"IoTAgent": "IoT tools", "Utilities": "Utility tools"},
        )
        assert "IoTAgent" in captured[0]
        assert "Utilities" in captured[0]
