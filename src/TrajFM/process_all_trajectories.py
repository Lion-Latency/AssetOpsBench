import os
import json
import pandas as pd
import logging
from json_repair import repair_json

from utils import get_llm_answer_from_json, extract_json_from_response

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all messages

# Create formatter
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")


timestamps = [
    "2025-05-12T21_26_41",
    "2025-05-10T21_47_48",
    "2025-05-10T13_27_09",
    "2025-05-09T18_14_56",
    "2025-05-09T17_05_45",
    "2025-05-09T15_26_51",
    "2025-05-09T14_21_04",
    "2025-05-09T12_22_18",
]

failure_mode_keys = [
    "1.1 Disobey Task Specification",
    "1.2 Disobey Role Specification",
    "1.3 Step Repetition",
    "1.4 Loss of Conversation History",
    "1.5 Unaware of Termination Conditions",
    "2.1 Conversation Reset",
    "2.2 Fail to Ask for Clarification",
    "2.3 Task Derailment",
    "2.4 Information Withholding",
    "2.5 Ignored Other Agent's Input",
    "2.6 Action-Reasoning Mismatch",
    "3.1 Premature Termination",
    "3.2 No or Incorrect Verification",
    "3.3 Weak Verification",
]

model_id = 16
timestamp = timestamps[0]
logger_filename = "logs/" + timestamp + "_m" + str(model_id) + "_debug.log"

# File handler
file_handler = logging.FileHandler(logger_filename, mode="w")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)  # Log everything to file

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)  # Only show INFO+ in console

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


logger.debug("Script started")
logger.info("Logging initialized at debug level")
logger.info(f"Selected LLM model ID: {model_id}")


def load_all_json_files(root_path):
    json_data = {}

    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.isdigit():  # Only load files like '0001', '0002', etc.
                file_path = os.path.join(dirpath, filename)
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                        json_data[file_path] = data
                except Exception as e:
                    logger.debug(f"Failed to load {file_path}: {e}")

    return json_data


root_directory = (
    "/Users/jzhou/work/notebooks/agenticfram/trajectories" + "/" + timestamp
)

all_jsons = load_all_json_files(root_directory)

df_columns = [
    "model_id",
    "counter",
    "timestamp",
    "vendor",
    "model",
    "ut_id",
    "addi_fm_cnt",
    "addi_fm_list",
] + failure_mode_keys
df = pd.DataFrame(columns=df_columns)

counter = 1
max_test = 10

for path, content in all_jsons.items():

    logger.info(f"Trajectory Counter: {counter}")

    relative_path = os.path.relpath(path, root_directory)
    logger.debug(f"{relative_path}:\n")

    parts = relative_path.split(os.sep)
    if len(parts) < 2:
        logger.warning(f"Skipping {path}: cannot extract vendor and model")
        continue

    vendor = parts[0]
    model = parts[1]
    ut_id = parts[2]

    test = 0

    while test < max_test:
        try:

            logger.info(f"Test : {test}")
            raw_output = get_llm_answer_from_json(data=content, model_id=model_id)
            logger.debug(f"raw_output: \n{raw_output}")

            logger.debug(f"raw_output: \n{raw_output}")

            # response_text = raw_output["generated_text"]
            # logger.debug(f"response_text: \n{response_text}")
            # good_json_string = repair_json(response_text)
            # logger.debug(f"good_json_string: \n{good_json_string}")
            # response_json = json.loads(good_json_string)

            response_text = raw_output["generated_text"]
            logger.debug(f"response_text: \n{response_text}")
            response_json = extract_json_from_response(response_text)

            failure_modes = response_json.get("failure_modes", {})

            additional_failure_modes = response_json.get("additional_failure_modes", {})

            logger.info(
                f"Type of additional_failure_modes {type(additional_failure_modes)}"
            )

            num_additional_failure_modes = len(additional_failure_modes)
            logger.info(
                f"Number of additional_failure_modes {num_additional_failure_modes}"
            )

            if num_additional_failure_modes > 0:
                logger.info(
                    f"Discovered additional failure mode(s): {additional_failure_modes}"
                )

            # Build row with counter
            row = {
                "model_id": model_id,
                "counter": counter,
                "timestamp": timestamp,
                "vendor": vendor,
                "model": model,
                "ut_id": ut_id,
                "addi_fm_cnt": num_additional_failure_modes,
                "addi_fm_list": additional_failure_modes,
            }

            for key in failure_mode_keys:
                row[key] = failure_modes.get(key, False)

            df.loc[len(df)] = row
            logger.info(f"Row added successfully: {row}")

            break

        except Exception as e:
            logger.error(f"Failed to process {relative_path}: {e}", exc_info=True)

        test = test + 1

    counter += 1
    # if counter > 5:
    #     break

# Define the file path

df_file_path = "processed_trajectories/" + timestamp + "_m" + str(model_id) + "_db.pkl"
df.to_pickle(df_file_path)

df_new = pd.read_pickle(df_file_path)
print(df_new.iloc[:10, :10])

print(df_new.loc[:9, ["addi_fm_cnt", "addi_fm_list"]])
