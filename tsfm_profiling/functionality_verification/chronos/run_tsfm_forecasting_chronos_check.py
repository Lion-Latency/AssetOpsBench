from pathlib import Path
import json
import os
import tempfile
import sys

from dotenv import load_dotenv
import pandas as pd

repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

load_dotenv(repo_root / ".env")

datasets_dir = os.getenv("PATH_TO_DATASETS_DIR")
if not datasets_dir:
    raise RuntimeError("PATH_TO_DATASETS_DIR is not set")

DATA_DIR = Path(datasets_dir) / "dhaval_data"
SOURCE_DATASET_PATH = DATA_DIR / "main_flat.csv"
TARGET_COLUMN = "Chiller 4 Liquid Refrigerant Evaporator Temperature"
TIMESTAMP_COLUMN = "timestamp"
SUBSET_START_ROW = 789000
FORECAST_HORIZON = 24
SUBSET_NROWS = 2000

from src.servers.tsfm.main import run_tsfm_forecasting_chronos

print("\n" + "="*60)
print("FORECASTING CHECK (Interchangeable Model with Chronos)")
print("="*60)

subset_df = pd.read_csv(
    SOURCE_DATASET_PATH,
    low_memory=False,
    skiprows=lambda idx: idx != 0 and not (
        SUBSET_START_ROW <= idx < SUBSET_START_ROW + SUBSET_NROWS
    ),
    usecols=[TIMESTAMP_COLUMN, TARGET_COLUMN],
).dropna()
subset_dataset_path = Path(tempfile.mkdtemp()) / "dhaval_main_flat_subset.csv"
subset_df.to_csv(subset_dataset_path, index=False)

# Compare the predicted values from Chronos against the actual values in the original TTM forcasting check to ensure this is working correctly.
result = run_tsfm_forecasting_chronos(
    dataset_path=str(subset_dataset_path),
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