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

from src.servers.tsfm.main import run_tsfm_forecasting

print("\n" + "="*60)
print("FORECASTING CHECK (Original Model with TTM)")
print("="*60)

subset_df = pd.read_csv(
    SOURCE_DATASET_PATH,
    usecols=[TIMESTAMP_COLUMN, TARGET_COLUMN],
    low_memory=False,
).dropna()

result = run_tsfm_forecasting(
    dataset_path=str(SOURCE_DATASET_PATH),
    timestamp_column=TIMESTAMP_COLUMN,
    target_columns=[TARGET_COLUMN],
    model_checkpoint="ttm_96_28",
    forecast_horizon=24,
)

status = getattr(result, 'status', None)
if status == 'success' and hasattr(result, 'results_file'):
    with open(result.results_file, "r") as fh:
        payload = json.load(fh)

    subset_df[TIMESTAMP_COLUMN] = pd.to_datetime(
        subset_df[TIMESTAMP_COLUMN], format="ISO8601", utc=True
    )
    subset_df[TARGET_COLUMN] = subset_df[TARGET_COLUMN].astype(float)
    subset_df = subset_df.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)

    prediction_df = pd.DataFrame(
        {
            TIMESTAMP_COLUMN: pd.to_datetime(payload["timestamp"][0], utc=True),
            "prediction": [float(step[0]) for step in payload["target_prediction"][0]],
        }
    )
    
    # Compare the predicted values against the actual values to ensure the model is working correctly.
    actual_values = pd.merge_asof(
        prediction_df.sort_values(TIMESTAMP_COLUMN),
        subset_df[[TIMESTAMP_COLUMN, TARGET_COLUMN]].sort_values(TIMESTAMP_COLUMN),
        on=TIMESTAMP_COLUMN,
        direction="nearest",
        tolerance=pd.Timedelta("8min"),
    )[TARGET_COLUMN].tolist()

    print("\n✓ STATUS: SUCCESS")
    print(f"Predicted Values: {[round(v, 4) for v in prediction_df['prediction'].tolist()]}")
    print(
        "Actual Values: "
        f"{[None if pd.isna(v) else round(float(v), 4) for v in actual_values]}"
    )
else:
    print("\n✗ STATUS: FAILED")
    error = getattr(result, 'error', str(result))
    print(f"  Error: {error}")
print("="*60)