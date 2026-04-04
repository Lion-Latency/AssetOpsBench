from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from src.servers.tsfm.main import (
    run_tsfm_forecasting,
    run_tsfm_finetuning,
    run_tsad,
    run_integrated_tsad,
)

def print_result(check_name, result):
    """Print formatted check result with clear success/failure indicators."""
    print(f"\n{'='*60}")
    print(f"{check_name}")
    print(f"{'='*60}")
    
    status = getattr(result, 'status', None)
    if status == 'success':
        print(f"✓ STATUS: SUCCESS")
        message = getattr(result, 'message', '')
        if message:
            print(f"  Message: {message}")
        
        # Print specific metrics based on the result type
        if hasattr(result, 'results_file'):
            print(f"  Results: {result.results_file}")
        if hasattr(result, 'model_checkpoint'):
            print(f"  Model: {result.model_checkpoint}")
        if hasattr(result, 'anomaly_count'):
            print(f"  Anomalies detected: {result.anomaly_count} in {result.total_records} records")
    else:
        print(f"✗ STATUS: FAILED")
        error = getattr(result, 'error', str(result))
        print(f"  Error: {error}")
    print()

print("\n" + "="*60)
print("TSFM FUNCTIONALITY VERIFICATION - RUNNING ALL CHECKS")
print("="*60)

# Check 1: Forecasting
forecast_result = run_tsfm_forecasting(
    dataset_path="synthetic_data/chiller9_annotated_small_test.csv",
    timestamp_column="Timestamp",
    target_columns=["Chiller 9 Condenser Water Flow"],
    model_checkpoint="ttm_96_28",
    forecast_horizon=24,
)
print_result("CHECK 1: Forecasting", forecast_result)

# Check 2: Fine-Tuning
finetune_result = run_tsfm_finetuning(
    dataset_path="synthetic_data/chiller9_finetuning_small.csv",
    timestamp_column="Timestamp",
    target_columns=["Chiller 9 Condenser Water Flow"],
    model_checkpoint="ttm_96_28",
    save_model_dir="tunedmodels",
    forecast_horizon=24,
    n_finetune=0.05,
    n_test=0.05,
)
print_result("CHECK 2: Fine-Tuning", finetune_result)

forecast_dataset_path = "/home/tp2758/tsfm_profiling_data/datasets/synthetic_data/chiller9_annotated_small_test.csv"
tsad_dataset_path = "/home/tp2758/tsfm_profiling_data/datasets/synthetic_data/chiller9_tsad.csv"

# Prepare for anomaly detection by running forecasting first
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

# Check 3: Anomaly Detection
tsad_result = run_tsad(
    dataset_path=tsad_dataset_path,
    tsfm_output_json=forecast_results_file,
    timestamp_column="Timestamp",
    target_columns=["Chiller 9 Condenser Water Flow"],
    task="fit",
    false_alarm=0.05,
    n_calibration=0.2,
)
print_result("CHECK 3: Anomaly Detection", tsad_result)

# Check 4: Integrated Anomaly Detection
integrated_tsad_result = run_integrated_tsad(
    dataset_path=tsad_dataset_path,
    timestamp_column="Timestamp",
    target_columns=["Chiller 9 Condenser Water Flow"],
    model_checkpoint="ttm_96_28",
    false_alarm=0.05,
    n_calibration=0.2,
)
print_result("CHECK 4: Integrated Anomaly Detection", integrated_tsad_result)

# Summary
print("="*60)
print("SUMMARY")
print("="*60)
results = [
    ("Forecasting (run_tsfm_forecasting)", forecast_result),
    ("Fine-Tuning (run_tsfm_finetuning)", finetune_result),
    ("Anomaly Detection (run_tsad)", tsad_result),
    ("Integrated Anomaly Detection (run_integrated_tsad)", integrated_tsad_result),
]

passed = sum(1 for _, r in results if getattr(r, 'status', None) == 'success')
total = len(results)

for name, result in results:
    status = getattr(result, 'status', None)
    symbol = "✓" if status == 'success' else "✗"
    print(f"{symbol} {name}")

print(f"\nTotal: {passed}/{total} checks passed")
if passed == total:
    print("✓ All checks PASSED!")
else:
    print(f"✗ {total - passed} check(s) FAILED")
print("="*60)