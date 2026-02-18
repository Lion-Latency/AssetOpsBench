# AssetOpsBench MCP Environment

This directory contains the MCP servers and infrastructure for the AssetOpsBench project.

## Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Environment Variables](#environment-variables)
- [MCP Servers](#mcp-servers)
  - [IoTAgent](#iotagent)
  - [Utilities](#utilities)
- [Plan-Execute Runner](#plan-execute-runner)
  - [How it works](#how-it-works)
  - [CLI](#cli)
  - [Python API](#python-api)
  - [Bring your own LLM](#bring-your-own-llm)
  - [Add more MCP servers](#add-more-mcp-servers)
- [Connect to Claude Desktop](#connect-to-claude-desktop)
- [Running Tests](#running-tests)
- [Architecture](#architecture)

---

## Prerequisites

- **Python 3.14+** — required by `pyproject.toml`
- **[uv](https://docs.astral.sh/uv/)** — dependency and environment manager
- **Docker** — for running CouchDB (IoT data store)

## Quick Start

### 1. Install dependencies

Run from the **repo root**:

```bash
uv sync
```

### 2. Configure environment

Copy `mcp/.env` and fill in the required values (see [Environment Variables](#environment-variables)):

```bash
cp mcp/.env mcp/.env.local  # optional; load_dotenv picks up mcp/.env automatically
```

### 3. Start CouchDB

```bash
docker compose -f mcp/couchdb/docker-compose.yaml up -d
```

Verify CouchDB is running:

```bash
curl -X GET http://localhost:5984/
```

### 4. Run servers locally

Use `uv run` to start the MCP servers (paths relative to repo root):

```bash
uv run python mcp/servers/utilities/main.py
uv run python mcp/servers/iot/main.py
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `COUCHDB_URL` | IoT server | CouchDB connection URL, e.g. `http://localhost:5984` |
| `COUCHDB_DBNAME` | IoT server | Database name (default fixture: `chiller`) |
| `COUCHDB_USERNAME` | IoT server | CouchDB admin username |
| `COUCHDB_PASSWORD` | IoT server | CouchDB admin password |
| `WATSONX_APIKEY` | plan-execute | IBM WatsonX API key |
| `WATSONX_PROJECT_ID` | plan-execute | IBM WatsonX project ID |
| `WATSONX_URL` | plan-execute | WatsonX endpoint (optional; defaults to `https://us-south.ml.cloud.ibm.com`) |

---

## MCP Servers

### IoTAgent

**Path:** `mcp/servers/iot/main.py`
**Requires:** CouchDB (`COUCHDB_URL`, `COUCHDB_DBNAME`, `COUCHDB_USERNAME`, `COUCHDB_PASSWORD`)

| Tool | Arguments | Description |
|---|---|---|
| `sites` | — | List all available sites |
| `assets` | `site_name` | List all asset IDs for a site |
| `sensors` | `site_name`, `asset_id` | List sensor names for an asset |
| `history` | `site_name`, `asset_id`, `start`, `final?` | Fetch historical sensor readings for a time range (ISO 8601 timestamps) |

### Utilities

**Path:** `mcp/servers/utilities/main.py`
**Requires:** nothing (no external services)

| Tool | Arguments | Description |
|---|---|---|
| `json_reader` | `file_name` | Read and parse a JSON file from disk |
| `current_date_time` | — | Return the current UTC date and time as JSON |
| `current_time_english` | — | Return the current UTC time as a human-readable string |

---

## Plan-Execute Runner

`mcp/plan_execute/` is a custom MCP client that implements a **plan-and-execute** workflow over the MCP servers. It replaces AgentHive's bespoke orchestration with the standard MCP protocol.

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

### CLI

After `uv sync`, the `plan-execute` command is available:

```bash
plan-execute "What assets are available at site MAIN?"
```

Flags:

| Flag | Description |
|---|---|
| `--model-id INT` | WatsonX model ID (default: `16` = llama-4-maverick) |
| `--server NAME=PATH` | Override MCP servers with `NAME=PATH` pairs (repeatable) |
| `--show-plan` | Print the generated plan before execution |
| `--show-history` | Print each step result after execution |
| `--json` | Output answer + plan + history as JSON |

Examples:

```bash
# Use a different model and inspect the plan
plan-execute --model-id 19 --show-plan "List sensors for asset CH-1"

# Register an additional MCP server
plan-execute --server FMSRAgent=mcp/servers/fmsr/main.py "What are the failure modes?"

# Machine-readable output
plan-execute --show-history --json "How many observations exist for CH-1?" | jq .answer
```

### Python API

```python
import asyncio
from plan_execute import PlanExecuteRunner
from plan_execute.llm import WatsonXLLM

runner = PlanExecuteRunner(llm=WatsonXLLM(model_id=16))
result = asyncio.run(runner.run("What assets are available at site MAIN?"))
print(result.answer)
```

`OrchestratorResult` fields:

| Field | Type | Description |
|---|---|---|
| `answer` | `str` | Final synthesised answer |
| `plan` | `Plan` | The generated plan with its steps |
| `history` | `list[StepResult]` | Per-step execution results |

### Bring your own LLM

Implement `LLMBackend` to use any model:

```python
from plan_execute.llm import LLMBackend

class MyLLM(LLMBackend):
    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        ...  # call your model here

runner = PlanExecuteRunner(llm=MyLLM())
```

### Add more MCP servers

Pass `server_paths` to register additional servers. Keys must match the agent names the planner assigns steps to:

```python
from pathlib import Path
from plan_execute import PlanExecuteRunner

runner = PlanExecuteRunner(
    llm=my_llm,
    server_paths={
        "IoTAgent":  Path("mcp/servers/iot/main.py"),
        "Utilities": Path("mcp/servers/utilities/main.py"),
        "FMSRAgent": Path("mcp/servers/fmsr/main.py"),   # once implemented
    },
)
```

> **Note:** passing `server_paths` replaces the defaults entirely. Include all servers you need.

---

## Connect to Claude Desktop

Add the following to your Claude Desktop `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "utilities": {
      "command": "/path/to/uv",
      "args": [
        "run",
        "--project",
        "/path/to/AssetOpsBench",
        "python",
        "/path/to/AssetOpsBench/mcp/servers/utilities/main.py"
      ]
    },
    "IoTAgent": {
      "command": "/path/to/uv",
      "args": [
        "run",
        "--project",
        "/path/to/AssetOpsBench",
        "python",
        "/path/to/AssetOpsBench/mcp/servers/iot/main.py"
      ]
    }
  }
}
```

---

## Running Tests

### Unit tests (no services required)

```bash
# MCP servers
uv run pytest mcp/servers/iot/tests/test_tools.py -k "not integration"
uv run pytest mcp/servers/utilities/tests/

# plan_execute (all unit tests, no CouchDB or WatsonX needed)
uv run pytest mcp/plan_execute/tests/
```

Run the full non-integration suite in one command:

```bash
uv run pytest mcp/servers/iot/tests/test_tools.py mcp/servers/utilities/tests/ mcp/plan_execute/tests/ -k "not integration"
```

### Integration tests (requires CouchDB)

Integration tests are skipped unless `COUCHDB_URL` is set (loaded from `.env` via `dotenv`):

```bash
docker compose -f mcp/couchdb/docker-compose.yaml up -d
uv run pytest mcp/servers/iot/tests/
uv run pytest mcp/servers/utilities/tests/
```

---

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
