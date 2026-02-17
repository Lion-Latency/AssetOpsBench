# AssetOpsBench MCP Environment

This directory contains the MCP servers and infrastructure for the AssetOpsBench project.

## Quick Start

### 1. Start CouchDB

```bash
docker compose up -d
```

### 2. Run servers locally

```bash
uv run fastmcp run servers/iot/main.py
uv run fastmcp run servers/utilities/main.py
```

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

### Test Structure

```
servers/iot/tests/
  conftest.py       # shared fixtures (mock_db, no_db) and requires_couchdb marker
  test_tools.py     # unit + integration tests for all 4 tools
  test_couchdb.py   # CouchDB infrastructure/connectivity tests

servers/utilities/tests/
  conftest.py        # shared call_tool helper
  test_utilities.py  # tests for json_reader and time tools
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
