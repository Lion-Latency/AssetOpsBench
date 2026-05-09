from pathlib import Path
import json
import tempfile
import sys

from dotenv import load_dotenv
import numpy as np
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

from src.servers.tsfm.main import run_tsfm_finetuning, run_tsfm_forecasting
from src.servers.tsfm.metrics import _MAE, _MAPE, _RMSE

print("\n" + "="*90)
print("FINE-TUNING CHECK (Original Model with TTM)")
print("="*90)

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

subset_df[TIMESTAMP_COLUMN] = pd.to_datetime(
    subset_df[TIMESTAMP_COLUMN], format="ISO8601", utc=True
)
subset_df[TARGET_COLUMN] = subset_df[TARGET_COLUMN].astype(float)
subset_df = subset_df.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)

forecast_input_df = subset_df.iloc[:-FORECAST_HORIZON].copy()
forecast_input_df[TIMESTAMP_COLUMN] = forecast_input_df[TIMESTAMP_COLUMN].dt.strftime(
    "%Y-%m-%dT%H:%M:%SZ"
)
forecast_dataset_path = Path(tempfile.mkdtemp()) / "dhaval_main_flat_forecast_subset.csv"
forecast_input_df.to_csv(forecast_dataset_path, index=False)

actual_values = subset_df.iloc[-FORECAST_HORIZON:][TARGET_COLUMN].to_numpy(dtype=float)
naive_errors = np.abs(
    np.diff(forecast_input_df[TARGET_COLUMN].to_numpy(dtype=float))
)
mase_denom = float(naive_errors.mean()) if len(naive_errors) > 0 else None

base_forecast_result = run_tsfm_forecasting(
    dataset_path=str(forecast_dataset_path),
    timestamp_column=TIMESTAMP_COLUMN,
    target_columns=[TARGET_COLUMN],
    model_checkpoint="ttm_96_28",
    forecast_horizon=FORECAST_HORIZON,
)

if getattr(base_forecast_result, 'status', None) == 'success' and hasattr(
    base_forecast_result, 'results_file'
):
    with open(base_forecast_result.results_file, "r") as fh:
        payload = json.load(fh)

    predicted_values = np.array(
        [float(step[0]) for step in payload["target_prediction"][0]], dtype=float
    )
    mape_value = _MAPE(actual_values, predicted_values)
    mase_value = (
        None
        if mase_denom in (None, 0.0)
        else float(np.mean(np.abs(actual_values - predicted_values)) / mase_denom)
    )
    print("\n\n\n")
    print("="*90)
    print(
        "Base checkpoint forecast metrics: "
        f"MAE={float(_MAE(actual_values, predicted_values)):.4f}, "
        f"RMSE={float(_RMSE(actual_values, predicted_values)):.4f}, "
        f"MAPE={None if mape_value is None else round(float(mape_value), 4)}, "
        f"MASE={None if mase_value is None else round(mase_value, 4)}"
    )
    print("="*90)
    print("\n\n\n")
else:
    print("\n\n\n")
    print("="*90)
    print(
        "Base checkpoint forecast metrics: "
        f"Error: {getattr(base_forecast_result, 'error', str(base_forecast_result))}"
    )
    print("="*90)
    print("\n\n\n")

result = run_tsfm_finetuning(
    dataset_path=str(subset_dataset_path),
    timestamp_column=TIMESTAMP_COLUMN,
    target_columns=[TARGET_COLUMN],
    model_checkpoint="ttm_96_28",
    save_model_dir="tunedmodels",
    forecast_horizon=FORECAST_HORIZON,
    n_finetune=0.05,
    n_test=0.05,
)

status = getattr(result, 'status', None)
if status == 'success':
    print("\n✓ STATUS: SUCCESS")
    message = getattr(result, 'message', '')
    if message:
        print(f"  Message: {message}")
    if hasattr(result, 'model_checkpoint'):
        print(f"  Model: {result.model_checkpoint}")
        finetuned_forecast_result = run_tsfm_forecasting(
            dataset_path=str(forecast_dataset_path),
            timestamp_column=TIMESTAMP_COLUMN,
            target_columns=[TARGET_COLUMN],
            model_checkpoint=result.model_checkpoint,
            forecast_horizon=FORECAST_HORIZON,
        )
        if getattr(finetuned_forecast_result, 'status', None) == 'success' and hasattr(
            finetuned_forecast_result, 'results_file'
        ):
            with open(finetuned_forecast_result.results_file, "r") as fh:
                payload = json.load(fh)

            predicted_values = np.array(
                [float(step[0]) for step in payload["target_prediction"][0]],
                dtype=float,
            )
            mape_value = _MAPE(actual_values, predicted_values)
            mase_value = (
                None
                if mase_denom in (None, 0.0)
                else float(np.mean(np.abs(actual_values - predicted_values)) / mase_denom)
            )
            print("\n\n\n")
            print("="*90)
            print(
                "Fine-tuned checkpoint forecast metrics: "
                f"MAE={float(_MAE(actual_values, predicted_values)):.4f}, "
                f"RMSE={float(_RMSE(actual_values, predicted_values)):.4f}, "
                f"MAPE={None if mape_value is None else round(float(mape_value), 4)}, "
                f"MASE={None if mase_value is None else round(mase_value, 4)}"
            )
            print("="*90)
            print("\n\n\n")
        else:
            print("\n\n\n")
            print("="*90)
            print(
                "Fine-tuned checkpoint forecast metrics: "
                f"Error: {getattr(finetuned_forecast_result, 'error', str(finetuned_forecast_result))}"
            )
            print("="*90)
            print("\n\n\n")
    if hasattr(result, 'results_file'):
        print(f"  Results: {result.results_file}")
else:
    print("\n✗ STATUS: FAILED")
    error = getattr(result, 'error', str(result))
    print(f"  Error: {error}")
print("="*90)