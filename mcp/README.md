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

### 1. Unit Tests (Fast & Local)

These tests use **mocks** and do not require CouchDB or Docker. They are ideal for rapid development.

```bash
# Install dependencies first
pip install -r iot_server/requirements.txt

# Run unit tests
pytest iot_server/tests/test_tools_unit.py
```

### 2. Integration Tests (Requires Services)

These tests run against the live CouchDB instance. Run them inside the existing `iot-server` container:

```bash
docker compose exec iot-server python3 -m pytest tests
```

## MCP Tools

The **IoTAgent** server provides the following tools:

- `sites()`: List available sites (currently returns `["MAIN"]`).
- `assets(site_name)`: List assets (dynamically discovered from CouchDB).
- `sensors(site_name, assetnum)`: List sensors for an asset (dynamically discovered from CouchDB).
- `history(site_name, assetnum, start, final)`: Fetch historical sensor data from CouchDB.
