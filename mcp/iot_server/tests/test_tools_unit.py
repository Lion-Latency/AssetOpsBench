import pytest
import json
from unittest.mock import patch
from main import mcp


@pytest.fixture
def mock_db():
    with patch("main.db") as mock:
        yield mock


@pytest.mark.anyio
async def test_sites_unit():
    contents, _ = await mcp.call_tool("sites", {})
    data = json.loads(contents[0].text)
    assert data["sites"] == ["MAIN"]


@pytest.mark.anyio
async def test_assets_dynamic_unit(mock_db):
    # Mock db.find result for assets discovery
    mock_db.find.return_value = {
        "docs": [{"asset_id": "Chiller 1"}, {"asset_id": "Chiller 2"}]
    }

    contents, _ = await mcp.call_tool("assets", {"site_name": "MAIN"})
    data = json.loads(contents[0].text)

    assert data["total_assets"] == 2
    assert "Chiller 1" in data["assets"]
    assert "Chiller 2" in data["assets"]
    mock_db.find.assert_called_once()


@pytest.mark.anyio
async def test_sensors_dynamic_unit(mock_db):
    # Mock db.find result for sensor discovery (takes first doc keys)
    mock_db.find.return_value = {
        "docs": [
            {
                "asset_id": "Chiller 1",
                "timestamp": "2024-01-01T00:00:00",
                "Temp": 25.5,
                "Pressure": 10.2,
                "_id": "doc1",
                "_rev": "rev1",
            }
        ]
    }

    contents, _ = await mcp.call_tool(
        "sensors", {"site_name": "MAIN", "asset_id": "Chiller 1"}
    )
    data = json.loads(contents[0].text)

    assert data["total_sensors"] == 2
    assert "Temp" in data["sensors"]
    assert "Pressure" in data["sensors"]
    assert "_id" not in data["sensors"]


@pytest.mark.anyio
async def test_history_unit(mock_db):
    # Mock db.find result for history
    mock_db.find.return_value = {
        "docs": [
            {"timestamp": "2024-01-01T00:00:00", "Temp": 20.0},
            {"timestamp": "2024-01-01T00:15:00", "Temp": 21.0},
        ]
    }

    contents, _ = await mcp.call_tool(
        "history",
        {"site_name": "MAIN", "asset_id": "Chiller 1", "start": "2024-01-01T00:00:00"},
    )
    data = json.loads(contents[0].text)

    assert data["total_observations"] == 2
    assert len(data["observations"]) == 2
    assert data["observations"][0]["Temp"] == 20.0
