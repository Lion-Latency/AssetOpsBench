from prompt import system_prompt
from reactxen.utils.model_inference import watsonx_llm

import json

# filename = "0008"
# with open("0008", "r") as file:
#     data = json.load(file)

filename = "trajectories/2025-05-10T13_27_09/meta-llama/llama-4-maverick-17b-128e-instruct-fp8/0112"
with open(filename, "r") as file:
    data = json.load(file)


trajectory = data.get("trajectory", [])

formatted_steps = []
question = data["task"]
formatted_steps.append(f"Question: {question}")
for idx, step in enumerate(trajectory, 1):
    thought = step.get("thought", "[No thought]")
    action = step.get("action", "[No action]")
    observation = step.get("observation", "[No observation]")

    step_text = (
        f"Thought {idx}: {thought}\n"
        f"Action {idx}: {action}\n"
        f"Observation {idx}: {observation}\n"
    )
    formatted_steps.append(step_text)

final_answer = data["final_answer"]
formatted_steps.append(f"Answer: {final_answer}")

# Combine all steps into one string
final_prompt_string = "\n" + "-" * 40 + "\n".join(formatted_steps)
print(final_prompt_string)

prompt = system_prompt.format(trace=final_prompt_string)
ans = watsonx_llm(prompt=prompt)
print(ans)
