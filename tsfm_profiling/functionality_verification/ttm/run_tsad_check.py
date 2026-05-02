from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

BASE_DIR = Path(__file__).resolve().parent
FUNCTIONALITY_VERIFICATION_DIR = BASE_DIR.parent
DATA_DIR = FUNCTIONALITY_VERIFICATION_DIR / "synthetic_data"
MODELS_DIR = repo_root / "src" / "servers" / "tsfm" / "artifacts" / "tsfm_models"

from src.servers.tsfm.main import run_tsfm_forecasting, run_tsad

forecast_dataset_path = str(DATA_DIR / "chiller9_annotated_small_test.csv")
tsad_dataset_path = str(DATA_DIR / "chiller9_tsad.csv")

print("\n" + "="*60)
print("ANOMALY DETECTION CHECK")
print("="*60)
print("Step 1: Running forecasting (required for TSAD)...")

forecast_result = run_tsfm_forecasting(
    dataset_path=forecast_dataset_path,
    timestamp_column="Timestamp",
    target_columns=["Chiller 9 Condenser Water Flow"],
    model_checkpoint="ttm_96_28",
    forecast_horizon=24,
)

if getattr(forecast_result, "status", None) != "success":
    raise RuntimeError(f"Forecasting failed: {forecast_result}")

forecast_results_file = forecast_result.results_file
print(f"  ✓ Forecasting complete. Results: {forecast_results_file}")

print("\nStep 2: Running anomaly detection...")
tsad_result = run_tsad(
    dataset_path=tsad_dataset_path,
    tsfm_output_json=forecast_results_file,
    timestamp_column="Timestamp",
    target_columns=["Chiller 9 Condenser Water Flow"],
    task="fit",
    false_alarm=0.05,
    n_calibration=0.2,
)

status = getattr(tsad_result, 'status', None)
if status == 'success':
    print("\n✓ STATUS: SUCCESS")
    message = getattr(tsad_result, 'message', '')
    if message:
        print(f"  Message: {message}")
    if hasattr(tsad_result, 'anomaly_count'):
        print(f"  Anomalies detected: {tsad_result.anomaly_count} in {tsad_result.total_records} records")
    if hasattr(tsad_result, 'results_file'):
        print(f"  Results: {tsad_result.results_file}")
else:
    print("\n✗ STATUS: FAILED")
    error = getattr(tsad_result, 'error', str(tsad_result))
    print(f"  Error: {error}")
print("="*60)