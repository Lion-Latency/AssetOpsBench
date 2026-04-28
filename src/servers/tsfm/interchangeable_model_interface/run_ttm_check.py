from pathlib import Path
import sys
import os
import numpy as np
import tempfile
import json
import pandas as pd
import traceback

repo_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(repo_root))

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "synthetic_data"
MODELS_DIR = repo_root / "src" / "servers" / "tsfm" / "artifacts" / "tsfm_models"
DATA_PATH = Path(__file__).resolve().parents[6] / "shared/tsfm_profiling_data/datasets/dhaval_data/main_flat.csv"

os.environ["PATH_TO_MODELS_DIR"] = str(MODELS_DIR) # to resolve the checkpoint path

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
    
    status = result if isinstance(result, str) else getattr(result, 'status', None)
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

print("\n" + "="*90)
print("INTERCHANGEABLE MODEL INTERFACE (with TTM) FUNCTIONALITY VERIFICATION - RUNNING ALL CHECKS")
print("="*90)

# Check 1: Forecasting
from src.servers.tsfm.interchangeable_model_interface.models.ttm import TTM

forecast_result = ""
try:
    data_path = DATA_PATH
    df = pd.read_csv(data_path, low_memory=False, nrows=2000)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
    df = df.dropna(axis=1, how="all")

    # Initialize model.
    model_path = MODELS_DIR / "ttm_96_28"
    interchangable_model = TTM(
        model_checkpoint=str(model_path),
        context_length=96,
        prediction_filter_length=28
    )

    # Run forecasting.
    forecast_data = interchangable_model.forecast(
        df_dataframe=df,
        column_specifiers={
            "timestamp_column": "timestamp",
            "target_columns": ["Chiller 6 Condenser Water Flow"]
        }
    )

    # Extract predictions
    predictions = np.array(forecast_data["target_prediction"])

    # Compute statistics
    pred_shape = predictions.shape
    pred_mean = float(np.mean(predictions))
    pred_std = float(np.std(predictions))

    forecast_result = "success"
except Exception as e:
    traceback.print_exc()
    forecast_result = f"EXCEPTION: {e}"
    
print_result("CHECK 1: Forecasting", forecast_result)

#--------------------------

# Check 2: Fine-Tuning
finetune_result = ""
try:
    data_path = DATA_PATH
    df = pd.read_csv(data_path, low_memory=False, nrows=2000)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
    df = df.dropna(axis=1, how="all")

    # Initialize model.
    model_path = MODELS_DIR / "ttm_96_28"
    interchangable_model = TTM(
        model_checkpoint=str(model_path),
        context_length=96,
        prediction_filter_length=28
    )

    # Run forecasting.
    forecast_data = interchangable_model.finetune(
        df_dataframe=df,
        column_specifiers={
            "timestamp_column": "timestamp",
            "target_columns": ["Chiller 6 Condenser Water Flow"]
        },
        save_model_dir=tempfile.mkdtemp()
    )

    finetune_result = "success"
except Exception as e:
    finetune_result = f"EXCEPTION: {e}"
    
print_result("CHECK 2: Fine-Tuning", finetune_result)

# Check 3: Anomaly Detection (Run forecasting using the interchangeable model, then run_tsad on the forecast_json_path.)
anomaly_detection_tsad_result = ""
try:
    # Run forecasting using the interchangeable model.
    try:
        data_path = DATA_PATH
        df = pd.read_csv(data_path, low_memory=False, nrows=2000)
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
        df = df.dropna(axis=1, how="all")

        # Initialize model.
        model_path = MODELS_DIR / "ttm_96_28"
        interchangable_model = TTM(
            model_checkpoint=str(model_path),
            context_length=96,
            prediction_filter_length=28
        )

        # Run forecasting.
        forecast_data = interchangable_model.forecast(
            df_dataframe=df,
            column_specifiers={
                "timestamp_column": "timestamp",
                "target_columns": ["Chiller 6 Condenser Water Flow"]
            }
        )

        # Extract predictions.
        predictions = np.array(forecast_data["target_prediction"])

        # Compute statistics.
        pred_shape = predictions.shape
        pred_mean = float(np.mean(predictions))
        pred_std = float(np.std(predictions))

        forecast_result = "success"
    except Exception as e:
        forecast_result = f"EXCEPTION: {e}"

    # Convert to JSON.
    json_data = {
        "target_prediction": np.array(forecast_data["target_prediction"]).tolist(),
        "timestamp": np.array(forecast_data["timestamp_prediction"]).astype(str).tolist(),
        "target_columns": forecast_data["target_columns"]
    }

    forecast_json_path = os.path.join(tempfile.mkdtemp(), "forecast_output.json")
    with open(forecast_json_path, 'w') as f:
        json.dump(json_data, f, indent=4)

    _df_tsad = pd.read_csv(DATA_PATH, low_memory=False, nrows=2000)
    _df_tsad = _df_tsad.dropna(axis=1, how="all")
    _df_tsad["timestamp"] = pd.to_datetime(_df_tsad["timestamp"], format="ISO8601", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S")
    _tsad_tmp = os.path.join(tempfile.mkdtemp(), "tsad_data.csv")
    _df_tsad.to_csv(_tsad_tmp, index=False)

    tsad_result = run_tsad(
        dataset_path=_tsad_tmp,
        tsfm_output_json=forecast_json_path,
        timestamp_column="timestamp",
        target_columns=["Chiller 6 Condenser Water Flow"],
        task="fit",
        false_alarm=0.05,
        n_calibration=0.2,
    )

    if getattr(tsad_result, 'status', None) != 'success':
        raise RuntimeError(f"run_tsad returned non-success: {getattr(tsad_result, 'error', tsad_result)}")

    anomaly_detection_tsad_result = tsad_result
except Exception as e:
    anomaly_detection_tsad_result = f"EXCEPTION: {e}"

print_result("CHECK 3: Anomaly Detection", anomaly_detection_tsad_result)

# Summary
print("="*60)
print("SUMMARY")
print("="*60)
results = [
    ("Forecasting (Interchangeable Forecast)", forecast_result),
    ("Fine-Tuning (Interchangeable Fine-Tuning)", finetune_result),
    ("Anomaly Detection (Interchangeable Anomaly Detection: run_tsad)", anomaly_detection_tsad_result),
    ]

passed = sum(1 for _, r in results if (r if isinstance(r, str) else getattr(r, 'status', None)) == 'success')
total = len(results)

for name, result in results:
    status = result if isinstance(result, str) else getattr(result, 'status', None)
    symbol = "✓" if status == 'success' else "✗"
    print(f"{symbol} {name}")

print(f"\nTotal: {passed}/{total} checks passed")
if passed == total:
    print("✓ All checks PASSED!")
else:
    print(f"✗ {total - passed} check(s) FAILED")
print("="*60)