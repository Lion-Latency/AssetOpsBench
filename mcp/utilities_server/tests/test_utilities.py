import pytest
import json
import os
import tempfile
from pathlib import Path
from main import mcp


@pytest.mark.anyio
async def test_current_date_time():
    contents, _ = await mcp.call_tool("current_date_time", {})
    data = json.loads(contents[0].text)
    assert "currentDateTime" in data
    assert "currentDateTimeDescription" in data
    assert "Today's date is" in data["currentDateTimeDescription"]


@pytest.mark.anyio
async def test_current_time_english():
    contents, _ = await mcp.call_tool("current_time_english", {})
    data = json.loads(contents[0].text)
    assert "english" in data
    assert "iso" in data


@pytest.mark.anyio
async def test_json_reader_success():
    # Create a dummy json file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump({"test": "data"}, tmp)
        tmp_name = tmp.name

    try:
        contents, _ = await mcp.call_tool("json_reader", {"file_name": tmp_name})
        data = json.loads(contents[0].text)
        assert data == {"test": "data"}
    finally:
        os.remove(tmp_name)
