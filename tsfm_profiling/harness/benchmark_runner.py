from __future__ import annotations

import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import psutil
import wandb

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

BASE_DIR = REPO_ROOT / "tsfm_profiling" / "functionality_verification"
DATA_DIR = BASE_DIR / "synthetic_data"
MODELS_DIR = REPO_ROOT / "src" / "servers" / "tsfm" / "artifacts" / "tsfm_models"
OUTPUT_DIR = REPO_ROOT / "tsfm_profiling" / "harness" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

os.environ["PATH_TO_MODELS_DIR"] = str(MODELS_DIR)

from src.servers.tsfm.main import (
    _emit_metrics,
    run_integrated_tsad,
    run_tsad,
    run_tsfm_finetuning,
    run_tsfm_forecasting,
)

WANDB_ENTITY = "lion-latency"
WANDB_PROJECT = "hpml-project"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def bytes_to_mb(n):
    return n / (1024 * 1024)


def write_json(path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_csv(path, rows):
    if not rows:
        return
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    fieldnames = sorted(all_keys)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def timed_run(fn, **kwargs):
    process = psutil.Process(os.getpid())
    start_rss = process.memory_info().rss
    start = time.perf_counter()
    result = fn(**kwargs)
    latency = time.perf_counter() - start
    rss_delta = bytes_to_mb(process.memory_info().rss - start_rss)
    return result, latency, rss_delta


def make_row(workflow, run_index, result, latency, rss_delta, config):
    report = _emit_metrics._last_report or {}
    stages = report.get("stages", [])

    row = {
        "workflow": workflow,
        "run_index": run_index,
        "run_type": "cold_start" if run_index == 1 else "steady_state",
        "status": getattr(result, "status", "unknown"),
        "error": getattr(result, "error", None),
        "latency_sec": latency,
        "cpu_rss_delta_mb": rss_delta,
        "end_to_end_ms": report.get("end_to_end_ms"),
        "stage_total_ms": report.get("stage_total_ms"),
        "overhead_ms": report.get("overhead_ms"),
        "model_checkpoint": config.get("model_checkpoint"),
        "forecast_horizon": config.get("forecast_horizon"),
        "seed": config.get("seed"),
    }

    for stage in stages:
        name = stage["stage"]
        row[f"stage_{name}_ms"] = stage.get("wall_clock_ms")
        row[f"stage_{name}_rss_delta_mb"] = stage.get("rss_delta_mb")
        if "gpu_mem_delta_mb" in stage:
            row[f"stage_{name}_gpu_delta_mb"] = stage.get("gpu_mem_delta_mb")

    return row


def summarize(rows):
    cold = [r for r in rows if r["run_type"] == "cold_start" and r["status"] == "success"]
    steady = [r for r in rows if r["run_type"] == "steady_state" and r["status"] == "success"]
    return {
        "total_runs": len(rows),
        "successful_runs": sum(r["status"] == "success" for r in rows),
        "failed_runs": sum(r["status"] != "success" for r in rows),
        "cold_start_latency_sec": cold[0]["latency_sec"] if cold else None,
        "steady_state_avg_latency_sec": mean([r["latency_sec"] for r in steady]) if steady else None,
        "steady_state_avg_rss_delta_mb": mean([r["cpu_rss_delta_mb"] for r in steady]) if steady else None,
        "cold_start_end_to_end_ms": cold[0].get("end_to_end_ms") if cold else None,
        "steady_state_avg_end_to_end_ms": mean([r["end_to_end_ms"] for r in steady if r.get("end_to_end_ms")]) if steady else None,
        "steady_state_avg_overhead_ms": mean([r["overhead_ms"] for r in steady if r.get("overhead_ms")]) if steady else None,
    }


def bench_forecasting(config):
    rows = []
    for i in range(1, config["repeats"] + 1):
        result, latency, rss = timed_run(
            run_tsfm_forecasting,
            dataset_path=str(DATA_DIR / "chiller9_annotated_small_test.csv"),
            timestamp_column="Timestamp",
            target_columns=["Chiller 9 Condenser Water Flow"],
            model_checkpoint=config["model_checkpoint"],
            forecast_horizon=config["forecast_horizon"],
        )
        rows.append(make_row("forecasting", i, result, latency, rss, config))
    return rows


def bench_finetuning(config):
    rows = []
    for i in range(1, config["repeats"] + 1):
        result, latency, rss = timed_run(
            run_tsfm_finetuning,
            dataset_path=str(DATA_DIR / "chiller9_finetuning_small.csv"),
            timestamp_column="Timestamp",
            target_columns=["Chiller 9 Condenser Water Flow"],
            model_checkpoint=config["model_checkpoint"],
            save_model_dir="tunedmodels",
            forecast_horizon=config["forecast_horizon"],
            n_finetune=0.05,
            n_test=0.05,
        )
        rows.append(make_row("finetuning", i, result, latency, rss, config))
    return rows


def bench_tsad(config):
    forecast_result, _, _ = timed_run(
        run_tsfm_forecasting,
        dataset_path=str(DATA_DIR / "chiller9_annotated_small_test.csv"),
        timestamp_column="Timestamp",
        target_columns=["Chiller 9 Condenser Water Flow"],
        model_checkpoint=config["model_checkpoint"],
        forecast_horizon=config["forecast_horizon"],
    )
    if getattr(forecast_result, "status", None) != "success":
        raise RuntimeError(f"Forecasting pre-step failed: {forecast_result}")
    forecast_file = forecast_result.results_file

    rows = []
    for i in range(1, config["repeats"] + 1):
        result, latency, rss = timed_run(
            run_tsad,
            dataset_path=str(DATA_DIR / "chiller9_tsad.csv"),
            tsfm_output_json=forecast_file,
            timestamp_column="Timestamp",
            target_columns=["Chiller 9 Condenser Water Flow"],
            task="fit",
            false_alarm=0.05,
            n_calibration=0.2,
        )
        rows.append(make_row("tsad", i, result, latency, rss, config))
    return rows


def bench_integrated_tsad(config):
    rows = []
    for i in range(1, config["repeats"] + 1):
        result, latency, rss = timed_run(
            run_integrated_tsad,
            dataset_path=str(DATA_DIR / "chiller9_tsad.csv"),
            timestamp_column="Timestamp",
            target_columns=["Chiller 9 Condenser Water Flow"],
            model_checkpoint=config["model_checkpoint"],
            false_alarm=0.05,
            n_calibration=0.2,
        )
        rows.append(make_row("integrated_tsad", i, result, latency, rss, config))
    return rows


def log_to_wandb(workflow, rows, summary):
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=f"bench_{workflow}_{time.strftime('%Y%m%d_%H%M%S')}",
        config={
            "workflow": workflow,
            "model_checkpoint": rows[0]["model_checkpoint"] if rows else None,
            "repeats": len(rows),
        },
        reinit="finish_previous",
    )

    for row in rows:
        log_dict = {
            f"{workflow}/latency_sec": row["latency_sec"],
            f"{workflow}/rss_delta_mb": row["cpu_rss_delta_mb"],
            f"{workflow}/status": 1 if row["status"] == "success" else 0,
            "run_index": row["run_index"],
            "run_type": row["run_type"],
        }

        if row.get("end_to_end_ms") is not None:
            log_dict[f"{workflow}/end_to_end_ms"] = row["end_to_end_ms"]
        if row.get("stage_total_ms") is not None:
            log_dict[f"{workflow}/stage_total_ms"] = row["stage_total_ms"]
        if row.get("overhead_ms") is not None:
            log_dict[f"{workflow}/overhead_ms"] = row["overhead_ms"]

        for key, val in row.items():
            if key.startswith("stage_") and val is not None:
                log_dict[f"{workflow}/{key}"] = val

        run.log(log_dict)

    if summary["cold_start_latency_sec"] is not None:
        run.summary[f"{workflow}/cold_start_latency_sec"] = summary["cold_start_latency_sec"]
    if summary["steady_state_avg_latency_sec"] is not None:
        run.summary[f"{workflow}/steady_state_avg_latency_sec"] = summary["steady_state_avg_latency_sec"]
    if summary["steady_state_avg_rss_delta_mb"] is not None:
        run.summary[f"{workflow}/steady_state_avg_rss_delta_mb"] = summary["steady_state_avg_rss_delta_mb"]
    if summary.get("steady_state_avg_end_to_end_ms") is not None:
        run.summary[f"{workflow}/steady_state_avg_end_to_end_ms"] = summary["steady_state_avg_end_to_end_ms"]
    if summary.get("steady_state_avg_overhead_ms") is not None:
        run.summary[f"{workflow}/steady_state_avg_overhead_ms"] = summary["steady_state_avg_overhead_ms"]
    run.summary[f"{workflow}/successful_runs"] = summary["successful_runs"]
    run.summary[f"{workflow}/failed_runs"] = summary["failed_runs"]

    run.finish()
    print(f"  W&B: {run.url}")


BENCHMARKS = {
    "forecasting": bench_forecasting,
    "finetuning": bench_finetuning,
    "tsad": bench_tsad,
    "integrated_tsad": bench_integrated_tsad,
}


def main():
    config = {
        "model_checkpoint": "ttm_96_28",
        "forecast_horizon": 24,
        "seed": 42,
        "repeats": 5,
    }

    set_seed(config["seed"])
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    all_rows = []

    print("=" * 72)
    print("TSFM EXTERNAL BENCHMARKING HARNESS")
    print("=" * 72)
    print(f"Entity: {WANDB_ENTITY} | Project: {WANDB_PROJECT}")
    print(f"Repeats per workflow: {config['repeats']}")
    print()

    for workflow, bench_fn in BENCHMARKS.items():
        print(f"--- {workflow.upper()} ---")
        try:
            rows = bench_fn(config)
        except Exception as e:
            print(f"  SKIPPED ({e})")
            continue

        for r in rows:
            print(
                f"  Run {r['run_index']} [{r['run_type']}] "
                f"status={r['status']} "
                f"latency={r['latency_sec']:.4f}s "
                f"e2e={r.get('end_to_end_ms', 'N/A')}ms "
                f"overhead={r.get('overhead_ms', 'N/A')}ms"
            )
            if r["status"] != "success":
                print(f"    error: {r['error']}")

        summary = summarize(rows)
        print(f"  Cold-start latency:           {summary['cold_start_latency_sec']}")
        print(f"  Steady-state avg latency:     {summary['steady_state_avg_latency_sec']}")
        print(f"  Steady-state avg e2e ms:      {summary['steady_state_avg_end_to_end_ms']}")
        print(f"  Steady-state avg overhead ms: {summary['steady_state_avg_overhead_ms']}")

        log_to_wandb(workflow, rows, summary)

        write_json(OUTPUT_DIR / f"{workflow}_{timestamp}.json", rows)
        write_csv(OUTPUT_DIR / f"{workflow}_{timestamp}.csv", rows)
        write_json(OUTPUT_DIR / f"{workflow}_summary_{timestamp}.json", summary)

        all_rows.extend(rows)
        print()

    write_json(OUTPUT_DIR / f"all_benchmarks_{timestamp}.json", all_rows)
    write_csv(OUTPUT_DIR / f"all_benchmarks_{timestamp}.csv", all_rows)

    print("=" * 72)
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()

