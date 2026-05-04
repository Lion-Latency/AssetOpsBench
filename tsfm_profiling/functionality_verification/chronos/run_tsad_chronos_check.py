from pathlib import Path
import runpy
import sys

repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

from src.servers.tsfm.interchangeable_model_interface.models.chronos import Chronos

print("\n" + "="*90)
print("ANOMALY DETECTION CHECK (Interchangeable Model with Chronos)")
print("This will run the forecasting check first, then perform anomaly detection using Chronos.")
print("="*90)

forecast = runpy.run_path(str(repo_root / "tsfm_profiling" / "functionality_verification" / "chronos" / "run_tsfm_forecasting_chronos_check.py"), run_name="__main__")

print("\n" + "="*60)
print("ANOMALY DETECTION (Interchangeable Model with Chronos)")
print("="*60)

if getattr(forecast["result"], "status", None) == "success" and hasattr(
    forecast["result"], "results_file"
):
    interchangeable_model = Chronos(
        model_checkpoint="amazon/chronos-2",
        context_length=0,
        prediction_filter_length=forecast["FORECAST_HORIZON"]
    )
    
    tsad_result = interchangeable_model.anomaly_detection(
        dataset_path=str(forecast["subset_dataset_path"]),
        tsfm_output_json=forecast["result"].results_file,
        timestamp_column=forecast["TIMESTAMP_COLUMN"],
        target_columns=[forecast["TARGET_COLUMN"]],
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
                f"  Anomalies detected: {tsad_result.anomaly_count} in "
                f"{tsad_result.total_records} records"
            )
        if hasattr(tsad_result, "results_file"):
            print(f"  Results: {tsad_result.results_file}")
    else:
        print("\n✗ STATUS: FAILED")
        error = getattr(tsad_result, "error", str(tsad_result))
        print(f"  Error: {error}")
else:
    print("\n✗ STATUS: FAILED")
    error = getattr(forecast["result"], "error", str(forecast["result"]))
    print(f"  Error: {error}")
print("="*60)