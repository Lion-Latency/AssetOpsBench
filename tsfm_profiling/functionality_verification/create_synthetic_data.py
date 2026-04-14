from pathlib import Path

import numpy as np
import pandas as pd


#output_dir = Path("/home/tp2758/tsfm_profiling_data/datasets/synthetic_data")
# Using a path relative to the script instead of hardcoded path
BASE_DIR = Path(__file__).resolve().parent
output_dir = BASE_DIR / "synthetic_data"
output_dir.mkdir(parents=True, exist_ok=True)

number_of_rows = 300
timestamps = pd.date_range("2020-01-01", periods=number_of_rows, freq="h")

random_number_generator = np.random.default_rng(42)

base_signal = 100 + 10 * np.sin(np.linspace(0, 20, number_of_rows))
noise = random_number_generator.normal(0, 1.5, number_of_rows)

dataframe = pd.DataFrame(
    {
        "Timestamp": timestamps,
        "Chiller 9 Condenser Water Flow": base_signal + noise,
        "Chiller 9 Liquid Refrigerant Evaporator Temperature": 42
        + 2 * np.sin(np.linspace(0, 15, number_of_rows)),
        "Chiller 9 Return Temperature": 55
        + 2 * np.cos(np.linspace(0, 12, number_of_rows)),
        "Chiller 9 Tonnage": 300 + 20 * np.sin(np.linspace(0, 8, number_of_rows)),
        "Chiller 9 Setpoint Temperature": np.full(number_of_rows, 44.0),
        "Chiller 9 Supply Temperature": 44
        + 1.5 * np.sin(np.linspace(0, 10, number_of_rows)),
        "Chiller 9 Chiller % Loaded": 70
        + 10 * np.sin(np.linspace(0, 7, number_of_rows)),
        "Chiller 9 Condenser Water Supply To Chiller Temperature": 68
        + 2 * np.cos(np.linspace(0, 10, number_of_rows)),
        "Chiller 9 Power Input": 120 + 10 * np.sin(np.linspace(0, 6, number_of_rows)),
        "Chiller 9 Chiller Efficiency": 0.75
        + 0.03 * np.sin(np.linspace(0, 5, number_of_rows)),
    }
)

forecast_path = output_dir / "chiller9_annotated_small_test.csv"
finetune_path = output_dir / "chiller9_finetuning_small.csv"
tsad_path = output_dir / "chiller9_tsad.csv"

forecast_dataframe = dataframe.iloc[:220].copy()
finetune_dataframe = dataframe.iloc[:260].copy()

tsad_dataframe = dataframe.copy()
tsad_dataframe["segment_id"] = 0
tsad_dataframe.loc[240:250, "Chiller 9 Condenser Water Flow"] += 25

forecast_dataframe.to_csv(forecast_path, index=False)
finetune_dataframe.to_csv(finetune_path, index=False)
tsad_dataframe.to_csv(tsad_path, index=False)

print("Created:")
print(forecast_path)
print(finetune_path)
print(tsad_path)