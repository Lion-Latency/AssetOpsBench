from pathlib import Path
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

from src.servers.tsfm.main import run_tsfm_forecasting, run_tsad

print("\n" + "="*60)
print("ANOMALY DETECTION CHECK (Original Model with TTM)")
print("="*60)

forecast_result = run_tsfm_forecasting(
    dataset_path=str(SOURCE_DATASET_PATH),
    timestamp_column=TIMESTAMP_COLUMN,
    target_columns=[TARGET_COLUMN],
    model_checkpoint="ttm_96_28",
    forecast_horizon=24,
)

if getattr(forecast_result, "status", None) == "success" and hasattr(
    forecast_result, "results_file"
):
    tsad_result = run_tsad(
        dataset_path=str(SOURCE_DATASET_PATH),
        tsfm_output_json=forecast_result.results_file,
        timestamp_column=TIMESTAMP_COLUMN,
        target_columns=[TARGET_COLUMN],
        task="fit",
        false_alarm=0.05,
        n_calibration=0.2,
    )

    status = getattr(tsad_result, "status", None)
    if status == "success":
        print("\n✓ STATUS: SUCCESS")
        message = getattr(tsad_result, "message", "")
        if message:
            print(f"  Message: {message}")
        if hasattr(tsad_result, "anomaly_count") and hasattr(tsad_result, "total_records"):
            print(
                f"  Anomalies detected: {tsad_result.anomaly_count} "
                f"in {tsad_result.total_records} records"
            )
        if hasattr(tsad_result, "results_file"):
            print(f"  Results: {tsad_result.results_file}")
    else:
        print("\n✗ STATUS: FAILED")
        error = getattr(tsad_result, "error", str(tsad_result))
        print(f"  Error: {error}")
else:
    print("\n✗ STATUS: FAILED")
    error = getattr(forecast_result, "error", str(forecast_result))
    print(f"  Error: {error}")
print("="*60)