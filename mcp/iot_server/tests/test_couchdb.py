import pytest
import os
import couchdb3
import requests

# Use credentials in URL for better compatibility with couchdb3
COUCHDB_HOST = os.environ.get("COUCHDB_URL", "http://couchdb:5984").replace(
    "http://", ""
)
COUCHDB_USER = os.environ.get("COUCHDB_USERNAME", "admin")
COUCHDB_PASS = os.environ.get("COUCHDB_PASSWORD", "password")
COUCHDB_DBNAME = os.environ.get("COUCHDB_DBNAME", "chiller")

FULL_URL = f"http://{COUCHDB_USER}:{COUCHDB_PASS}@{COUCHDB_HOST}"


@pytest.mark.skipif(
    os.environ.get("COUCHDB_URL") is None, reason="CouchDB not available"
)
def test_couchdb_connection():
    try:
        # Use simple requests to verify endpoint first
        resp = requests.get(f"http://{COUCHDB_HOST}", auth=(COUCHDB_USER, COUCHDB_PASS))
        assert resp.status_code == 200

        client = couchdb3.Server(FULL_URL)
        assert client.info() is not None
    except Exception as e:
        pytest.fail(f"Could not connect to CouchDB: {e}")


@pytest.mark.skipif(
    os.environ.get("COUCHDB_URL") is None, reason="CouchDB not available"
)
def test_database_exists():
    client = couchdb3.Server(FULL_URL)
    assert COUCHDB_DBNAME in client.all_dbs()


@pytest.mark.skipif(
    os.environ.get("COUCHDB_URL") is None, reason="CouchDB not available"
)
def test_data_populated():
    client = couchdb3.Server(FULL_URL)
    db = client[COUCHDB_DBNAME]
    # Check for Chiller 6 data
    res = db.find({"asset_id": "Chiller 6"}, limit=1)
    assert len(res["docs"]) > 0
    assert res["docs"][0]["asset_id"] == "Chiller 6"
