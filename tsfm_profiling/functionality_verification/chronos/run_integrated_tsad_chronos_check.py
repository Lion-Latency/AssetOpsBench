from pathlib import Path
import sys
import tempfile

from dotenv import load_dotenv
import pandas as pd

repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

load_dotenv(repo_root / ".env")

SOURCE_DATASET_PATH = repo_root / "tsfm_profiling" / "data" / "main.csv"
TARGET_COLUMN = "Chiller 4 Liquid Refrigerant Evaporator Temperature"
TIMESTAMP_COLUMN = "timestamp"
SUBSET_START_ROW = 789000
FORECAST_HORIZON = 24
SUBSET_NROWS = 2000

from src.servers.tsfm.main import run_integrated_tsad_chronos

print("\n" + "="*75)
print("INTEGRATED ANOMALY DETECTION CHECK (Interchangeable Model with Chronos)")
print("="*75)

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

result = run_integrated_tsad_chronos(
    dataset_path=str(subset_dataset_path),
    timestamp_column=TIMESTAMP_COLUMN,
    target_columns=[TARGET_COLUMN],
    model_checkpoint="amazon/chronos-2",
    false_alarm=0.05,
    n_calibration=0.2,
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