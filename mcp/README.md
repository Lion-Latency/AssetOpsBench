# AssetOpsBench MCP Environment

This directory contains the MCP servers and infrastructure for the AssetOpsBench project.

## Components

- **IoT MCP Server**: A Python-based MCP server providing tools to interact with BMS (Building Management System) data.
- **CouchDB**: A NoSQL database populated with sample chiller data.

## Quick Start

### 1. Build and Start Services

This starts both the IoT server and CouchDB. The first time it runs, it will also populate the `chiller` database with sample records.

```bash
docker compose up -d --build
```

### 2. Verify everything is running

```bash
docker compose ps
```

The `couchdb` service should have a status of `(healthy)`.

## Running Tests

### Unit Tests (No Services Required)

Unit tests use mocked dependencies and can run without Docker:

```bash
docker compose exec iot-server python3 -m pytest tests/test_tools.py -k "not integration"
docker compose exec utilities-server python3 -m pytest tests
```

### Integration Tests (Requires Services)

Integration tests run against the live CouchDB instance and are skipped unless `COUCHDB_URL` is set (which is provided automatically inside Docker):

```bash
docker compose exec iot-server python3 -m pytest tests
docker compose exec utilities-server python3 -m pytest tests
```

### Test Structure

```
iot_server/tests/
  conftest.py       # shared fixtures (mock_db, no_db) and requires_couchdb marker
  test_tools.py     # unit + integration tests for all 4 tools
  test_couchdb.py   # CouchDB infrastructure/connectivity tests

utilities_server/tests/
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
