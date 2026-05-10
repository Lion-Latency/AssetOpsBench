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

from src.servers.tsfm.main import run_integrated_tsad_chronos

print("\n" + "="*75)
print("INTEGRATED ANOMALY DETECTION CHECK (Interchangeable Model with Chronos)")
print("="*75)

result = run_integrated_tsad_chronos(
    dataset_path=str(SOURCE_DATASET_PATH),
    timestamp_column=TIMESTAMP_COLUMN,
    target_columns=[TARGET_COLUMN],
    model_checkpoint="amazon/chronos-2",
    false_alarm=0.05,
    n_calibration=0.2,
    frequency_sampling="15_minutes",
)

status = getattr(result, "status", None)
if status == "success":
    print("\n✓ STATUS: SUCCESS")
    message = getattr(result, "message", "")
    if message:
        print(f"  Message: {message}")
    if hasattr(result, "anomaly_count") and hasattr(result, "total_records"):
        print(
            f"  Anomalies detected: {result.anomaly_count} in "
            f"{result.total_records} records"
        )
    if hasattr(result, "results_file"):
        print(f"  Results: {result.results_file}")
else:
    print("\n✗ STATUS: FAILED")
    error = getattr(result, "error", str(result))
    print(f"  Error: {error}")
print("="*60)