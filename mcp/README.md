# AssetOpsBench MCP Environment

This directory contains the MCP servers and infrastructure for the AssetOpsBench project.

## Quick Start

### 0. Install dependencies

We use `uv` for dependency and environment management.

```bash
uv sync
```

### 1. Start CouchDB

```bash
docker compose up -d
```

Validate CouchDB is running:

```bash
curl -X GET http://localhost:5984/
```

### 2. Run servers locally

Use `uv run` to run the mcp servers:

```bash
uv run python servers/utilities/main.py
uv run python servers/iot/main.py
```

## [Optional] Connect to MCP Client: Claude Desktop

Add the following to your Claude Desktop `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "utilities": {
      "command": "/path/to/uv",
      "args": [
        "run",
        "--project",
        "/path/to/AssetOpsBench/mcp",
        "python",
        "/path/to/AssetOpsBench/mcp/servers/utilities/main.py"
      ]
    },
    "IoTAgent": {
      "command": "/path/to/uv",
      "args": [
        "run",
        "--project",
        "/path/to/AssetOpsBench/mcp",
        "python",
        "/path/to/AssetOpsBench/mcp/servers/iot/main.py"
      ]
    }
  },
  "preferences": {
    "sidebarMode": "chat",
    "coworkScheduledTasksEnabled": false
  }
}
```

## Plan-Execute Runner (Custom Agentic Orchestration)

`plan_execute/` is a custom MCP client that implements a **plan-and-execute**
workflow over the MCP servers.  It replaces AgentHive's bespoke orchestration
with the standard MCP protocol.

### How it works

```
PlanExecuteRunner.run(question)
  │
  ├─ 1. Discover   query each MCP server for its available tools
  │
  ├─ 2. Plan       LLM decomposes the question into ordered steps,
  │                each assigned to an MCP server
  │
  ├─ 3. Execute    for each step (in dependency order):
  │                  • LLM selects the right tool + generates arguments
  │                  • tool is called via MCP stdio protocol
  │                  • result is stored and passed as context to later steps
  │
  └─ 4. Summarise  LLM synthesises step results into a final answer
```

### Usage

```python
import asyncio
from plan_execute import PlanExecuteRunner
from plan_execute.llm import WatsonXLLM

runner = PlanExecuteRunner(llm=WatsonXLLM(model_id=16))
result = asyncio.run(runner.run("What assets are available at site MAIN?"))
print(result.answer)
```

`WatsonXLLM` reads credentials from the environment:

```bash
export WATSONX_APIKEY=...
export WATSONX_PROJECT_ID=...
export WATSONX_URL=https://us-south.ml.cloud.ibm.com   # optional
```

Install the optional WatsonX dependency if you have not done so:

```bash
uv pip install ".[watsonx]"
```

### Bring your own LLM

Implement `LLMBackend` to use any other model:

```python
from plan_execute.llm import LLMBackend

class MyLLM(LLMBackend):
    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        ...  # call your model here

runner = PlanExecuteRunner(llm=MyLLM())
```

### Add more MCP servers

Pass `server_paths` to register additional servers.  Keys must match the
agent names the planner assigns steps to:

```python
from pathlib import Path
from plan_execute import PlanExecuteRunner

runner = PlanExecuteRunner(
    llm=my_llm,
    server_paths={
        "IoTAgent":  Path("servers/iot/main.py"),
        "Utilities": Path("servers/utilities/main.py"),
        "FMSRAgent": Path("servers/fmsr/main.py"),   # once implemented
    },
)
```

## Running Tests

### Unit Tests (No Services Required)

```bash
# MCP servers
uv run pytest servers/iot/tests/test_tools.py -k "not integration"
uv run pytest servers/utilities/tests

# plan_execute (all unit tests, no CouchDB or WatsonX needed)
uv run pytest plan_execute/tests/
```

Run the full non-integration suite in one command:

```bash
uv run pytest servers/iot/tests/test_tools.py servers/utilities/tests plan_execute/tests/ -k "not integration"
```

### Integration Tests (Requires CouchDB)

Integration tests are skipped unless `COUCHDB_URL` is set (loaded from `.env` via `dotenv`):

```bash
docker compose up -d
uv run pytest servers/iot/tests
uv run pytest servers/utilities/tests
```

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                   plan_execute/                      │
│                                                      │
│  PlanExecuteRunner.run(question)                     │
│  ┌────────────┐   ┌────────────┐   ┌──────────────┐ │
│  │  Planner   │ → │  Executor  │ → │  Summariser  │ │
│  │            │   │            │   │              │ │
│  │ LLM breaks │   │ Routes each│   │ LLM combines │ │
│  │ question   │   │ step to the│   │ step results │ │
│  │ into steps │   │ right MCP  │   │ into answer  │ │
│  └────────────┘   │ server via │   └──────────────┘ │
│                   │ stdio      │                     │
└───────────────────┼────────────┼─────────────────────┘
                    │ MCP protocol (stdio)
         ┌──────────┼──────────┐
         ▼          ▼          ▼
      IoTAgent   Utilities   FMSRAgent ...
      (tools)    (tools)     (planned)
```
