import pytest
import json
import os
from main import mcp

# Integration tests for tools via MCP interface
# These run against the live CouchDB instance in Docker.


@pytest.mark.anyio
async def test_sites():
    contents, _ = await mcp.call_tool("sites", {})
    data = json.loads(contents[0].text)
    assert "sites" in data
    assert data["sites"] == ["MAIN"]


@pytest.mark.anyio
async def test_assets_error_site():
    contents, _ = await mcp.call_tool("assets", {"site_name": "INVALID"})
    data = json.loads(contents[0].text)
    assert "error" in data
    assert "unknown site" in data["error"]


@pytest.mark.skipif(
    os.environ.get("COUCHDB_URL") is None, reason="CouchDB not available"
)
@pytest.mark.anyio
async def test_assets_discovery():
    contents, _ = await mcp.call_tool("assets", {"site_name": "MAIN"})
    data = json.loads(contents[0].text)
    assert "assets" in data
    # Assuming Chiller 6 exists in sample data
    assert "Chiller 6" in data["assets"]
    assert data["total_assets"] > 0


@pytest.mark.anyio
async def test_sensors_error_site():
    contents, _ = await mcp.call_tool(
        "sensors", {"site_name": "INVALID", "assetnum": "Chiller 6"}
    )
    data = json.loads(contents[0].text)
    assert "error" in data


@pytest.mark.anyio
async def test_sensors_error_asset():
    contents, _ = await mcp.call_tool(
        "sensors", {"site_name": "MAIN", "assetnum": "INVALID"}
    )
    data = json.loads(contents[0].text)
    assert "error" in data
    assert "no sensors found" in data["error"]


@pytest.mark.skipif(
    os.environ.get("COUCHDB_URL") is None, reason="CouchDB not available"
)
@pytest.mark.anyio
async def test_sensors_success():
    contents, _ = await mcp.call_tool(
        "sensors", {"site_name": "MAIN", "assetnum": "Chiller 6"}
    )
    data = json.loads(contents[0].text)
    assert "sensors" in data
    assert len(data["sensors"]) > 0
    # Common sensors in the benchmark data have prefixes
    assert any("Power" in s or "Efficiency" in s for s in data["sensors"])


@pytest.mark.anyio
async def test_history_error_invalid_range():
    # start >= final
    contents, _ = await mcp.call_tool(
        "history",
        {
            "site_name": "MAIN",
            "assetnum": "Chiller 6",
            "start": "2020-06-01T00:00:00",
            "final": "2020-05-01T00:00:00",
        },
    )
    data = json.loads(contents[0].text)
    assert "error" in data
    assert "start >= final" in data["error"]


@pytest.mark.anyio
async def test_history_error_malformed_date():
    contents, _ = await mcp.call_tool(
        "history", {"site_name": "MAIN", "assetnum": "Chiller 6", "start": "not-a-date"}
    )
    data = json.loads(contents[0].text)
    assert "error" in data


@pytest.mark.skipif(
    os.environ.get("COUCHDB_URL") is None, reason="CouchDB not available"
)
@pytest.mark.anyio
async def test_history_success_simple():
    contents, _ = await mcp.call_tool(
        "history",
        {"site_name": "MAIN", "assetnum": "Chiller 6", "start": "2020-06-01T00:00:00"},
    )
    data = json.loads(contents[0].text)
    assert "observations" in data
    assert "total_observations" in data


@pytest.mark.skipif(
    os.environ.get("COUCHDB_URL") is None, reason="CouchDB not available"
)
@pytest.mark.anyio
async def test_history_success_range():
    contents, _ = await mcp.call_tool(
        "history",
        {
            "site_name": "MAIN",
            "assetnum": "Chiller 6",
            "start": "2020-06-01T00:00:00",
            "final": "2020-06-01T01:00:00",
        },
    )
    data = json.loads(contents[0].text)
    assert "observations" in data
    # Ensure all timestamps are within range
    for obs in data["observations"]:
        assert obs["timestamp"] >= "2020-06-01T00:00:00"
        assert obs["timestamp"] < "2020-06-01T01:00:00"
