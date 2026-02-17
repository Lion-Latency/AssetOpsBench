# AssetOpsBench MCP Environment

This directory contains the MCP servers and infrastructure for the AssetOpsBench project.

## Components

- **IoT MCP Server**: A Python-based MCP server providing tools to interact with BMS (Building Management System) data.
- **CouchDB**: A NoSQL database populated with sample chiller data.

## Quick Start

### 1. Build and Start Services

This starts both the IoT server and CouchDB. The first time it runs, it will also populate the `chiller` database with sample records.

```bash
docker compose build
docker compose up -d
```

### 2. Verify everything is running

```bash
docker compose ps
```

The `couchdb` service should have a status of `(healthy)`.

## Running Tests

### Integration Tests (Requires Services)

These tests run against the live CouchDB instance. Run them inside the existing `iot-server` container:

```bash
docker compose exec iot-server python3 -m pytest tests
docker compose exec utilities-server python3 -m pytest tests
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
