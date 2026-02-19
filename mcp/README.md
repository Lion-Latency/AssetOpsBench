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

## [WIP] Connect to MCP Client: Local Client with Custom Agentic Orchestration

TBD.

## Running Tests

### Unit Tests (No Services Required)

```bash
uv run pytest servers/iot/tests/test_tools.py -k "not integration"
uv run pytest servers/utilities/tests
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
┌─────────────────────────────────────────────┐
│           SERVER-SIDE RUNTIME               │
│                                             │
│  ┌──────────────────────────────────────┐   │
│  │     ReAct Orchestrator (MCP Client)  │   │
│  │                                      │   │
│  │  Thought → Action → Observation loop │   │
│  │                                      │   │
│  │  1. Reason about the goal            │   │
│  │  2. Pick a tool (from MCP servers)   │   │
│  │  3. Call it                          │   │
│  │  4. Observe result                   │   │
│  │  5. Repeat until done                │   │
│  └──────────┬───────────────────────────┘   │
│             │ MCP protocol                  │
│    ┌────────┼────────┐                      │
│    ▼        ▼        ▼                      │
│  iot       fmsr    tsfm    ...              │
│  (tools)  (tools)  (tools)                  │
└─────────────────────────────────────────────┘
```
