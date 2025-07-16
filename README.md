<div align="center">

# AssetOpsBench: Benchmarking AI Agents for Task Automation in Industrial Asset Operations and Maintenance

![AssetOps](https://img.shields.io/badge/Domain-Asset_Operations-blue) 
![MultiAgentBench](https://img.shields.io/badge/Domain-Multi--agent_Bench-blue) 
![OpenAI](https://img.shields.io/badge/Model-OpenAI-21C2A4)
![Llama](https://img.shields.io/badge/Model-Llama-21C2A4)
![Mistral](https://img.shields.io/badge/Model-Mistral-21C2A4) 
![Granite](https://img.shields.io/badge/Model-Granite-21C2A4)

📰 [Paper](https://arxiv.org/pdf/2506.03828), 🤗 [Huggingface](https://huggingface.co/papers/2506.03828)

</div>

## Introduction
AssetOpsBench is a unified framework and environment designed to guide the development, orchestration, and evaluation of domain-specific agents for task automation in industrial asset operations and maintenance. The release of the benchmark focuses on scenarios commonly posed by domain experts—such as maintenance engineers, reliability specialists, and facility planners. We devloped 4 individual domain-specific agents and 2 multi-agent orchestration frameworks to create a simulated industrial environment enabling end-to-end benchmarking of multi-agent workflows in asset operations.
 
## Datasets: 140+ Scenarios
AssetOpsBench created a collection of tasks that we call scenarios, which covers domains of IoT data retrieval (IoT), failure mode and sensor relation discovery (FSMR), time series anomaly detection (TSFM) and work order generation (WO). Some of the tasks are focused on solving problems in single domain, e.g. "List all sensors of Chiller 6 in MAIN site". Others are focused on end-to-end multi-step tasks, e.g. "What is the forecast for 'Chiller 9 Condenser Water Flow' in the week of 2020-04-27 based on data from the MAIN site?" All scenarios can be found [here](https://github.com/IBM/AssetOpsBench/tree/main/scenarios).

## AI Agents and Multi-agent Frameworks
We developed 4 domain-specific AI agents while each agent has its own agent tools to be invoked.
- IoT Agent: `get_sites`, `get_history`, `get_assets`, `get_sensors`, ...
- FMSR Agent: `get_sensors`, `get_failure_modes`, `get_failure_sensor_mapping`.
- TSFM Agent: `forecasting`, `timeseries_anomaly_detection`, ...
- WO Agent: `generate_word_order`

To orchestrate multiple agents and run end-to-end workflow, we developed two frameworks:
- [MetaAgent](https://github.com/IBM/AssetOpsBench/tree/main/src/meta_agent): a reAct based single-agent-as-tool agent
- [AgentHive](https://github.com/IBM/AssetOpsBench/tree/main/src/agent_hive): a plan-and-execute sequential workflow

## Leaderboards
We run AssetOpsBench with 7 Large Language Models and evaluate the trajectories of each run using LLM judge (Llama-4-Maverick-17B) on 6-dimentional criteria. The following is the result of MetaAgent. Please find more results in the paper.
![meta_agent_leaderboard](https://github.com/user-attachments/assets/615059be-e296-40d3-90ec-97ee6cb00412)


## Run AssetOpsBench in Docker
We provide a comprehensive documentation on how to run AssetOpsBench in a pre-built dockerized environment. Please refer to the [guidance](https://github.com/IBM/AssetOpsBench/tree/main/benchmark/README.md).

## Contributors and Contact(*)

- Dhaval Patel (pateldha@us.ibm.com)
- Shuxin Lin
- James Rayfield
- Nianjun Zhou
