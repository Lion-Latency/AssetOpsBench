<div align="center">

# AssetOpsBench: Benchmarking AI Agents for Industrial Asset Operations & Maintenance

![AssetOps](https://img.shields.io/badge/Domain-Asset_Operations-blue) 
![MultiAgentBench](https://img.shields.io/badge/Domain-Multi--agent_Bench-blue) 
![OpenAI](https://img.shields.io/badge/Model-OpenAI-21C2A4)
![Llama](https://img.shields.io/badge/Model-Llama-21C2A4)    
![Mistral](https://img.shields.io/badge/Model-Mistral-21C2A4) 
![Granite](https://img.shields.io/badge/Model-Granite-21C2A4)

📄 [Paper](https://arxiv.org/pdf/2506.03828) | 🤗 [HF-Dataset](https://huggingface.co/datasets/ibm-research/AssetOpsBench) | 📢 [Blog](https://research.ibm.com/blog/asset-ops-benchmark) | [Contributors](#contributors)

</div>

---

## 📑 Table of Contents
1. [Announcements](#announcements)
2. [Introduction](#introduction)
3. [Datasets](#datasets-140-scenarios)
4. [AI Agents](#ai-agents)
5. [Multi-Agent Frameworks](#multi-agent-frameworks)
6. [System Diagram](#system-diagram)
7. [Leaderboards](#leaderboards)
8. [Docker Setup](#run-assetopsbench-in-docker)
9. [Talks & Events](#talks--events)
10. [External Resources](#external-resources)
11. [Contributors](#contributors)

---

## Announcements
- **2025-06-01**: AssetOpsBench v1.0 released with 140+ industrial scenarios.  
- **2025-09-01**: [CODS](https://ikdd.acm.org/cods-2025/) Competition launched. Access AI Agentic Challenge [AssetOpsBench-Live](https://www.codabench.org/competitions/10206/)    
- **Upcoming Events**: *Tutorial at AAAI 2026* – [Agents for Industry 4.0 Applications](https://ibm.github.io/AssetOpsBench/aaai_website/).  
- Stay tuned for new tracks, competitions, and community events.
---

## Introduction
AssetOpsBench is a **unified framework for developing, orchestrating, and evaluating domain-specific AI agents** in industrial asset operations and maintenance.  

It provides:
- 4 **domain-specific agents**  
- 2 **multi-agent orchestration frameworks**  

Designed for **maintenance engineers, reliability specialists, and facility planners**, it allows reproducible evaluation of multi-step workflows in simulated industrial environments.

---

## Datasets: 140+ Scenarios
AssetOpsBench scenarios span multiple domains:  

| Domain | Example Task |
|--------|--------------|
| IoT | "List all sensors of Chiller 6 in MAIN site" |
| FSMR | "Identify failure modes detected by Chiller 6 Supply Temperature" |
| TSFM | "Forecast 'Chiller 9 Condenser Water Flow' for the week of 2020-04-27" |
| WO | "Generate a work order for Chiller 6 anomaly detection" |

Some tasks focus on a **single domain**, others are **multi-step end-to-end workflows**.  
Explore all scenarios [here](https://github.com/IBM/AssetOpsBench/tree/main/scenarios).

---

## AI Agents
### Domain-Specific Agents
- **IoT Agent**: `get_sites`, `get_history`, `get_assets`, `get_sensors`  
- **FMSR Agent**: `get_sensors`, `get_failure_modes`, `get_failure_sensor_mapping`  
- **TSFM Agent**: `forecasting`, `timeseries_anomaly_detection`  
- **WO Agent**: `generate_work_order`  

### Multi-Agent Frameworks
- **[MetaAgent](https://github.com/IBM/AssetOpsBench/tree/main/src/meta_agent)**: reAct-based single-agent-as-tool orchestration  
- **[AgentHive](https://github.com/IBM/AssetOpsBench/tree/main/src/agent_hive)**: plan-and-execute sequential workflow  

---

## System Diagram
Visual overview of AssetOpsBench workflow:  

![System Diagram](path/to/system_diagram.png)  <!-- Replace with your image path -->

---

## Leaderboards
- Evaluated with **7 Large Language Models**  
- Trajectories scored using **LLM Judge (Llama-4-Maverick-17B)**  
- **6-dimensional criteria** measure reasoning, execution, and data handling  

Example: MetaAgent leaderboard  

![meta_agent_leaderboard](https://github.com/user-attachments/assets/615059be-e296-40d3-90ec-97ee6cb00412)

---

## 🐳 Run AssetOpsBench in Docker
- Pre-built Docker Images: `assetopsbench-basic` (minimal) & `assetopsbench-extra` (full)  
- Conda environment: `assetopsbench`  
- [Full setup guide](https://github.com/IBM/AssetOpsBench/tree/main/benchmark/README.md)  

```bash
cd /path/to/AssetOpsBench
chmod +x benchmark/entrypoint.sh
docker-compose -f benchmark/docker-compose.yml build
docker-compose -f benchmark/docker-compose.yml up
```

---

## Talks & Events
- **Workshops**: Participate in *GenAIBench-26* at AAAI 2025 focusing on multi-agent AI workflows.  
- **Webinars & Seminars**: Learn best practices for industrial task automation with AI agents.  
- **Competitions**: Benchmark your agents on real-world industrial scenarios using AssetOpsBench.

---

## External Resources
- 📄 **Paper**: [AssetOpsBench: Benchmarking AI Agents for Industrial Asset Operations](https://arxiv.org/pdf/2506.03828)  
- 🤗 **HuggingFace**: [Scenario & Model Hub](https://huggingface.co/papers/2506.03828)  
- 📢 **Blog**: [Insights, Tutorials, and Updates](https://research.ibm.com/blog/asset-ops-benchmark)  
- 🎥 **Recorded Talks**: Link coming soon.

---

## Contributors

Thanks goes to these wonderful people ✨

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/DhavalRepo18"><img src="https://github.com/DhavalRepo18.png?s=50" width="50px;" alt="DhavalRepo18"/><br /><sub><b>DhavalRepo18</b></sub></a><br /><a href="https://github.com/IBM/AssetOpsBench/commits?author=DhavalRepo18" title="Code">💻</a> <a href="https://github.com/IBM/AssetOpsBench/commits?author=DhavalRepo18" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ShuxinLin"><img src="https://github.com/ShuxinLin.png?s=50" width="50px;" alt="ShuxinLin"/><br /><sub><b>ShuxinLin</b></sub></a><br /><a href="https://github.com/IBM/AssetOpsBench/commits?author=ShuxinLin" title="Code">💻</a> <a href="https://github.com/IBM/AssetOpsBench/commits?author=ShuxinLin" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jtrayfield"><img src="https://github.com/jtrayfield.png?s=50" width="50px;" alt="jtrayfield"/><br /><sub><b>jtrayfield</b></sub></a><br /><a href="https://github.com/IBM/AssetOpsBench/commits?author=jtrayfield" title="Code">💻</a> <a href="https://github.com/IBM/AssetOpsBench/commits?author=jtrayfield" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/nianjunz"><img src="https://github.com/nianjunz.png?s=50" width="50px;" alt="nianjunz"/><br /><sub><b>nianjunz</b></sub></a><br /><a href="https://github.com/IBM/AssetOpsBench/commits?author=nianjunz" title="Code">💻</a> <a href="https://github.com/IBM/AssetOpsBench/commits?author=nianjunz" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ChathurangiShyalika"><img src="https://github.com/ChathurangiShyalika.png?s=50" width="50px;" alt="ChathurangiShyalika"/><br /><sub><b>ChathurangiShyalika</b></sub></a><br /><a href="https://github.com/IBM/AssetOpsBench/commits?author=ChathurangiShyalika" title="Code">💻</a> <a href="https://github.com/IBM/AssetOpsBench/commits?author=ChathurangiShyalika" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/PUSHPAK-JAISWAL"><img src="https://github.com/PUSHPAK-JAISWAL.png?s=50" width="50px;" alt="PUSHPAK-JAISWAL"/><br /><sub><b>PUSHPAK-JAISWAL</b></sub></a><br /><a href="https://github.com/IBM/AssetOpsBench/commits?author=PUSHPAK-JAISWAL" title="Code">💻</a> <a href="https://github.com/IBM/AssetOpsBench/commits?author=PUSHPAK-JAISWAL" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/bradleyjeck"><img src="https://github.com/bradleyjeck.png?s=50" width="50px;" alt="bradleyjeck"/><br /><sub><b>bradleyjeck</b></sub></a><br /><a href="https://github.com/IBM/AssetOpsBench/commits?author=bradleyjeck" title="Code">💻</a> <a href="https://github.com/IBM/AssetOpsBench/commits?author=bradleyjeck" title="Documentation">📖</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/florenzi002"><img src="https://github.com/florenzi002.png?s=50" width="50px;" alt="florenzi002"/><br /><sub><b>florenzi002</b></sub></a><br /><a href="https://github.com/IBM/AssetOpsBench/commits?author=florenzi002" title="Code">💻</a> <a href="https://github.com/IBM/AssetOpsBench/commits?author=florenzi002" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/kushwaha001"><img src="https://github.com/kushwaha001.png?s=50" width="50px;" alt="kushwaha001"/><br /><sub><b>kushwaha001</b></sub></a><br /><a href="https://github.com/IBM/AssetOpsBench/commits?author=kushwaha001" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://mohit-gupta.me/"><img src="https://avatars.githubusercontent.com/u/52665879?v=4?s=50" width="50px;" alt="Mohit Gupta"/><br /><sub><b>Mohit Gupta</b></sub></a><br /><a href="https://github.com/IBM/AssetOpsBench/commits?author=Mohit-15" title="Documentation">📖</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

---
