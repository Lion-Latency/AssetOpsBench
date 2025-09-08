import argparse
import json
import os

from dotenv import load_dotenv

load_dotenv()

from datasets import load_dataset
from huggingface_hub import login

login(os.getenv("HF_APIKEY", None))

from agent_hive.task import Task
from agent_hive.tools.fmsr import (
    fmsr_tools,
    fmsr_fewshots,
    fmsr_task_examples,
    fmsr_agent_name,
    fmsr_agent_description,
)
from agent_hive.tools.skyspark import (
    iot_bms_tools,
    iot_bms_fewshots,
    iot_agent_description,
    iot_agent_name,
)

from agent_hive.workflows.track2_execution import DynamicWorkflow
from agent_hive.agents.react_agent import ReactAgent

from agent_hive.logger import get_custom_logger

logger = get_custom_logger(__name__)

import warnings

warnings.filterwarnings("ignore")

RESULT_DIR = '/home/track1_result/'
PLAN_DIR = RESULT_DIR + 'plan/'
TRAJECTORY_DIR = RESULT_DIR + 'trajectory/'


def load_scenarios(utterance_ids):
    ds = load_dataset(
        "ibm-research/AssetOpsBench", "scenarios")
    train_ds = ds["train"]
    df = train_ds.to_pandas()

    filtered_df = df[df["id"].isin(utterance_ids)]

    return filtered_df.to_dict(orient="records")


def run_execution_workflow(
        question, qid, llm_model=16, generate_steps_only=False
):
    iot_r_agent = ReactAgent(
        name=iot_agent_name,
        description=iot_agent_description,
        tools=iot_bms_tools,
        llm=llm_model,
        few_shots=iot_bms_fewshots,
    )

    fmsr_r_agent = ReactAgent(
        name=fmsr_agent_name,
        description=fmsr_agent_description,
        tools=fmsr_tools,
        llm=llm_model,
        task_examples=fmsr_task_examples,
        few_shots=fmsr_fewshots,
    )

    task = Task(
        description=question,
        expected_output="",
        agents=[iot_r_agent, fmsr_r_agent],
    )

    wf = DynamicWorkflow(
        tasks=[task],
        llm=llm_model,
    )

    if generate_steps_only:
        os.makedirs(PLAN_DIR, exist_ok=True)

        return wf.generate_steps(
            save_plan=True,
            saved_plan_filename=RESULT_DIR + f"Model_{llm_model}_Q_{qid}_plan",
        )

    return wf.run()


def run(utterances, generate_steps_only=False):
    os.makedirs(TRAJECTORY_DIR, exist_ok=True)

    for utterance in utterances:
        logger.info('=' * 10)
        logger.info(
            f"ID: {utterance['id']}, Task: {utterance['text']}"
        )
        trajectory_file = f"{TRAJECTORY_DIR}Q_{utterance['id']}_trajectory.json"

        ans = run_execution_workflow(
            utterance["text"],
            utterance["id"],
            generate_steps_only=generate_steps_only,
        )

        if generate_steps_only:
            continue

        output = {"id": utterance["id"], "text": utterance["text"], "trajectory": ans}

        with open(trajectory_file, "w") as f:
            json.dump(output, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--utterance_ids", type=str, default='1,106')
    parser.add_argument("--generate_steps_only", type=bool, default=False)

    args = parser.parse_args()
    utterance_ids = [int(uid.strip()) for uid in args.utterance_ids.split(',')]
    utterances = load_scenarios(utterance_ids)

    run(
        utterances,
        generate_steps_only=args.generate_steps_only,
    )
