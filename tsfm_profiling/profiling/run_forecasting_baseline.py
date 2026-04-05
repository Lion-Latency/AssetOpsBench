from __future__ import annotations

import csv
import json
import os
import sys
import time
import random
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import psutil

# Make repo root importable
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Team-owned paths
BASE_DIR = REPO_ROOT / "tsfm_profiling" / "functionality_verification"
DATA_DIR = BASE_DIR / "synthetic_data"
MODELS_DIR = REPO_ROOT / "src" / "servers" / "tsfm" / "artifacts" / "tsfm_models"
OUTPUT_DIR = REPO_ROOT / "tsfm_profiling" / "profiling" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Override original repo default model dir without editing upstream code
os.environ["PATH_TO_MODELS_DIR"] = str(MODELS_DIR)

from src.servers.tsfm.main import run_tsfm_forecasting  # noqa: E402


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # Torch seeding is best-effort here; baseline should still run without it.
        pass


def bytes_to_mb(num_bytes: int) -> float:
    return num_bytes / (1024 * 1024)


def run_one_forecast(run_index: int, config: dict[str, Any]) -> dict[str, Any]:
    """Run one forecasting job and capture timing + memory stats."""
    process = psutil.Process(os.getpid())

    start_rss = process.memory_info().rss
    start_time = time.perf_counter()

    result = run_tsfm_forecasting(
        dataset_path=config["dataset_path"],
        timestamp_column=config["timestamp_column"],
        target_columns=config["target_columns"],
        model_checkpoint=config["model_checkpoint"],
        forecast_horizon=config["forecast_horizon"],
    )

    end_time = time.perf_counter()
    end_rss = process.memory_info().rss

    status = getattr(result, "status", None)
    error = getattr(result, "error", None)
    results_file = getattr(result, "results_file", None)
    message = getattr(result, "message", None)

    return {
        "workflow": "forecasting",
        "run_index": run_index,
        "run_type": "cold_start" if run_index == 1 else "steady_state",
        "status": status,
        "error": error,
        "message": message,
        "latency_sec": end_time - start_time,
        "cpu_rss_start_mb": bytes_to_mb(start_rss),
        "cpu_rss_end_mb": bytes_to_mb(end_rss),
        "cpu_rss_delta_mb": bytes_to_mb(end_rss - start_rss),
        "results_file": results_file,
        "dataset_path": config["dataset_path"],
        "model_checkpoint": config["model_checkpoint"],
        "forecast_horizon": config["forecast_horizon"],
        "seed": config["seed"],
    }


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_runs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cold_runs = [r for r in rows if r["run_type"] == "cold_start" and r["status"] == "success"]
    steady_runs = [r for r in rows if r["run_type"] == "steady_state" and r["status"] == "success"]

    summary: dict[str, Any] = {
        "total_runs": len(rows),
        "successful_runs": sum(r["status"] == "success" for r in rows),
        "failed_runs": sum(r["status"] != "success" for r in rows),
        "cold_start_latency_sec": cold_runs[0]["latency_sec"] if cold_runs else None,
        "steady_state_avg_latency_sec": mean([r["latency_sec"] for r in steady_runs]) if steady_runs else None,
        "steady_state_avg_cpu_rss_delta_mb": mean([r["cpu_rss_delta_mb"] for r in steady_runs]) if steady_runs else None,
    }
    return summary


def main() -> None:
    config = {
        "dataset_path": str(DATA_DIR / "chiller9_annotated_small_test.csv"),
        "timestamp_column": "Timestamp",
        "target_columns": ["Chiller 9 Condenser Water Flow"],
        "model_checkpoint": "ttm_96_28",
        "forecast_horizon": 24,
        "seed": 42,
        "repeats": 5,
    }

    dataset_path = Path(config["dataset_path"])
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. Run create_synthetic_data.py first."
        )

    model_path = MODELS_DIR / config["model_checkpoint"]
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint directory not found: {model_path}"
        )

    set_seed(config["seed"])

    rows: list[dict[str, Any]] = []
    print("=" * 72)
    print("FORECASTING BASELINE RUNNER")
    print("=" * 72)
    print(f"Dataset: {config['dataset_path']}")
    print(f"Model checkpoint: {config['model_checkpoint']}")
    print(f"Repeats: {config['repeats']}")
    print()

    for run_index in range(1, config["repeats"] + 1):
        row = run_one_forecast(run_index, config)
        rows.append(row)

        print(
            f"Run {run_index}/{config['repeats']} "
            f"[{row['run_type']}] "
            f"status={row['status']} "
            f"latency={row['latency_sec']:.4f}s "
            f"rss_delta={row['cpu_rss_delta_mb']:.2f}MB"
        )

        if row["status"] != "success":
            print(f"  error: {row['error']}")

    summary = summarize_runs(rows)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = OUTPUT_DIR / f"forecasting_baseline_{timestamp}.json"
    csv_path = OUTPUT_DIR / f"forecasting_baseline_{timestamp}.csv"
    summary_path = OUTPUT_DIR / f"forecasting_baseline_summary_{timestamp}.json"

    write_json(json_path, rows)
    write_csv(csv_path, rows)
    write_json(summary_path, summary)

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Successful runs: {summary['successful_runs']}/{summary['total_runs']}")
    print(f"Cold-start latency: {summary['cold_start_latency_sec']}")
    print(f"Steady-state avg latency: {summary['steady_state_avg_latency_sec']}")
    print(f"Steady-state avg RSS delta (MB): {summary['steady_state_avg_cpu_rss_delta_mb']}")
    print()
    print(f"Saved per-run JSON: {json_path}")
    print(f"Saved per-run CSV:  {csv_path}")
    print(f"Saved summary:      {summary_path}")


if __name__ == "__main__":
    main()