from pathlib import Path
import json
import tempfile
import sys

from dotenv import load_dotenv
import pandas as pd

repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

load_dotenv(repo_root / ".env")

SOURCE_DATASET_PATH = repo_root / "tsfm_profiling" / "data" / "sample.csv"
if not SOURCE_DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at {SOURCE_DATASET_PATH}. "
        "Place sample.csv at tsfm_profiling/data/sample.csv to run this check."
    )
TARGET_COLUMN = "Chiller 4 Liquid Refrigerant Evaporator Temperature"
TIMESTAMP_COLUMN = "timestamp"
FORECAST_HORIZON = 24

from src.servers.tsfm.main import run_tsfm_forecasting_chronos

print("\n" + "="*60)
print("FORECASTING CHECK (Interchangeable Model with Chronos)")
print("="*60)

result = run_tsfm_forecasting_chronos(
    dataset_path=str(SOURCE_DATASET_PATH),
    timestamp_column=TIMESTAMP_COLUMN,
    target_columns=[TARGET_COLUMN],
    model_checkpoint="amazon/chronos-2",
    forecast_horizon=FORECAST_HORIZON,
)

status = getattr(result, 'status', None)
if status == 'success' and hasattr(result, 'results_file'):
    with open(result.results_file, "r") as fh:
        payload = json.load(fh)

    predicted_values = [float(step[0]) for step in payload["target_prediction"][0]]

    print("\n✓ STATUS: SUCCESS")
    print(f"Predicted Values: {[round(v, 4) for v in predicted_values]}")
else:
    print("\n✗ STATUS: FAILED")
    error = getattr(result, 'error', str(result))
    print(f"  Error: {error}")
print("="*60)