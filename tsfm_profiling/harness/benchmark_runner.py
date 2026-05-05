from __future__ import annotations

import csv
import json
import os
from dotenv import load_dotenv
import random
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import psutil
import wandb

load_dotenv()
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = Path(os.getenv("PATH_TO_DATASETS_DIR", "/home/shared/tsfm_profiling_data/datasets")) / "dhaval_data"
MODELS_DIR = REPO_ROOT / "src" / "servers" / "tsfm" / "artifacts" / "tsfm_models"
OUTPUT_DIR = REPO_ROOT / "tsfm_profiling" / "harness" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

os.environ["PATH_TO_MODELS_DIR"] = str(MODELS_DIR)

from src.servers.tsfm import cache as _cache
from src.servers.tsfm.main import (
    _emit_metrics,
    run_integrated_tsad,
    run_tsad,
    run_tsfm_finetuning,
    run_tsfm_forecasting,
    run_tsfm_forecasting_chronos,
    run_tsfm_finetuning_chronos,
    run_tsad_chronos,
    run_integrated_tsad_chronos,
)

MODES = {
    "baseline":   {"TSFM_CACHE_ENABLED": "0", "TSFM_PREPROCESS_OPT": "0"},
    "cache_only": {"TSFM_CACHE_ENABLED": "1", "TSFM_PREPROCESS_OPT": "0"},
    "combined":   {"TSFM_CACHE_ENABLED": "1", "TSFM_PREPROCESS_OPT": "1",
                   "TSFM_PREPROCESS_WORKERS": "4",
                   "TSFM_PREPROCESS_EXECUTOR": "thread"},
    "parallelism_only": {"TSFM_CACHE_ENABLED": "0", "TSFM_PREPROCESS_OPT": "1",
                         "TSFM_PREPROCESS_WORKERS": "4",
                         "TSFM_PREPROCESS_EXECUTOR": "thread"},
}

def apply_mode(mode: str) -> None:
    for k in ("TSFM_CACHE_ENABLED", "TSFM_PREPROCESS_OPT",
              "TSFM_PREPROCESS_WORKERS", "TSFM_PREPROCESS_EXECUTOR"):
        os.environ.pop(k, None)
    for k, v in MODES[mode].items():
        os.environ[k] = v
    _cache.clear()


WANDB_ENTITY = "lion-latency"
WANDB_PROJECT = "hpml-project"

FORECAST_DATASET = str(DATA_DIR / "main_chiller9_small.csv")
FINETUNE_DATASET = str(DATA_DIR / "main_chiller9_small.csv")
TSAD_DATASET = str(DATA_DIR / "main_chiller9_small.csv")
TIMESTAMP_COLUMN = "timestamp"
TARGET_COLUMNS = ["Chiller 9 Condenser Water Flow"]


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


def _sanitize_label(s) -> str:
    return str(s).replace(" ", "_").replace("/", "_").replace(".", "_")


def _flatten_performance(perf) -> dict:
    out: dict = {}
    if not isinstance(perf, dict) or not perf:
        return out

    cols = list(perf.keys())
    all_subdicts = all(isinstance(perf[c], dict) for c in cols)

    if all_subdicts and "value" in cols:
        label_cols = [c for c in ("split", "target", "metric") if c in cols]
        h_col = "forecast" if "forecast" in cols else None

        for ridx in perf["value"].keys():
            try:
                val = float(perf["value"][ridx])
            except (TypeError, ValueError):
                continue

            parts = [_sanitize_label(perf[c].get(ridx)) for c in label_cols if perf[c].get(ridx) is not None]
            if h_col is not None and perf[h_col].get(ridx) is not None:
                parts.append(f"h{perf[h_col][ridx]}")

            key = "perf_" + "_".join(parts) if parts else f"perf_value_{ridx}"
            out[key] = val

        if "train_time" in cols:
            for ridx in perf["train_time"]:
                try:
                    out["perf_train_time"] = float(perf["train_time"][ridx])
                    break
                except (TypeError, ValueError):
                    continue
        return out

    for key, val in perf.items():
        if isinstance(val, dict):
            for k, v in val.items():
                try:
                    out[f"perf_{key}_{k}"] = float(list(v.values())[0]) if isinstance(v, dict) else float(v)
                except (TypeError, ValueError):
                    pass
        else:
            try:
                out[f"perf_{key}"] = float(val)
            except (TypeError, ValueError):
                pass
    return out


def extract_performance_metrics(result) -> dict:
    metrics = {}
    results_file = getattr(result, "results_file", None)
    if results_file and Path(results_file).exists():
        try:
            with open(results_file, "r") as f:
                data = json.load(f)
            if isinstance(data, dict) and "performance" in data:
                metrics.update(_flatten_performance(data["performance"]))
            if isinstance(data, dict) and "anomaly_count" in data:
                metrics["anomaly_count"] = data["anomaly_count"]
            if isinstance(data, dict) and "total_records" in data:
                metrics["total_records"] = data["total_records"]
        except Exception:
            pass
    anomaly_count = getattr(result, "anomaly_count", None)
    if anomaly_count is not None:
        metrics["anomaly_count"] = anomaly_count
    total_records = getattr(result, "total_records", None)
    if total_records is not None:
        metrics["total_records"] = total_records
    return metrics


def make_row(workflow, run_index, result, latency, rss_delta, config, mode=None):
    report = _emit_metrics._last_report or {}
    stages = report.get("stages", [])
    meta = report.get("metadata", {})

    row = {
        "workflow": workflow,
        "mode": mode,
        "run_index": run_index,
        "run_type": "cold_start" if run_index == 1 else "steady_state",
        "status": getattr(result, "status", "unknown"),
        "error": getattr(result, "error", None),
        "latency_sec": latency,
        "cpu_rss_delta_mb": rss_delta,
        "end_to_end_ms": report.get("end_to_end_ms"),
        "stage_total_ms": report.get("stage_total_ms"),
        "overhead_ms": report.get("overhead_ms"),
        "dq_cache_hit": meta.get("dq_cache_hit"),
        "prep_cache_hit": meta.get("prep_cache_hit"),
        "model_checkpoint": config.get("model_checkpoint"),
        "forecast_horizon": config.get("forecast_horizon"),
        "seed": config.get("seed"),
        "dataset": "real",
    }

    for stage in stages:
        name = stage["stage"]
        row[f"stage_{name}_ms"] = stage.get("wall_clock_ms")
        row[f"stage_{name}_rss_delta_mb"] = stage.get("rss_delta_mb")
        if "gpu_mem_delta_mb" in stage:
            row[f"stage_{name}_gpu_delta_mb"] = stage.get("gpu_mem_delta_mb")

    perf_metrics = extract_performance_metrics(result)
    row.update(perf_metrics)

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


def bench_forecasting(config, mode):
    rows = []
    for i in range(1, config["repeats"] + 1):
        result, latency, rss = timed_run(
            run_tsfm_forecasting,
            dataset_path=FORECAST_DATASET,
            timestamp_column=TIMESTAMP_COLUMN,
            target_columns=TARGET_COLUMNS,
            model_checkpoint=config["model_checkpoint"],
            forecast_horizon=config["forecast_horizon"],
        )
        rows.append(make_row("forecasting", i, result, latency, rss, config, mode))
    return rows


def bench_finetuning(config, mode):
    rows = []
    for i in range(1, config["repeats"] + 1):
        result, latency, rss = timed_run(
            run_tsfm_finetuning,
            dataset_path=FINETUNE_DATASET,
            timestamp_column=TIMESTAMP_COLUMN,
            target_columns=TARGET_COLUMNS,
            model_checkpoint=config["model_checkpoint"],
            save_model_dir="tunedmodels",
            forecast_horizon=config["forecast_horizon"],
            n_finetune=0.05,
            n_test=0.05,
        )
        rows.append(make_row("finetuning", i, result, latency, rss, config, mode))
    return rows


def bench_tsad(config, mode):
    forecast_result, _, _ = timed_run(
        run_tsfm_forecasting,
        dataset_path=FORECAST_DATASET,
        timestamp_column=TIMESTAMP_COLUMN,
        target_columns=TARGET_COLUMNS,
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
            dataset_path=TSAD_DATASET,
            tsfm_output_json=forecast_file,
            timestamp_column=TIMESTAMP_COLUMN,
            target_columns=TARGET_COLUMNS,
            task="fit",
            false_alarm=0.05,
            n_calibration=0.2,
        )
        rows.append(make_row("tsad", i, result, latency, rss, config, mode))
    return rows


def bench_integrated_tsad(config, mode):
    rows = []
    for i in range(1, config["repeats"] + 1):
        result, latency, rss = timed_run(
            run_integrated_tsad,
            dataset_path=TSAD_DATASET,
            timestamp_column=TIMESTAMP_COLUMN,
            target_columns=TARGET_COLUMNS,
            model_checkpoint=config["model_checkpoint"],
            false_alarm=0.05,
            n_calibration=0.2,
        )
        rows.append(make_row("integrated_tsad", i, result, latency, rss, config, mode))
    return rows


def log_to_wandb(workflow, rows, summary, run_tag="baseline", mode=None):
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=f"bench_{workflow}_{run_tag}_{time.strftime('%Y%m%d_%H%M%S')}",
        config={
            "workflow": workflow,
            "mode": mode,
            "cache_enabled": os.environ.get("TSFM_CACHE_ENABLED"),
            "preprocess_opt": os.environ.get("TSFM_PREPROCESS_OPT"),
            "preprocess_workers": os.environ.get("TSFM_PREPROCESS_WORKERS"),
            "model_checkpoint": rows[0]["model_checkpoint"] if rows else None,
            "repeats": len(rows),
            "dataset": "real",
            "run_tag": run_tag,
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

        for key in ["end_to_end_ms", "stage_total_ms", "overhead_ms"]:
            if row.get(key) is not None:
                log_dict[f"{workflow}/{key}"] = row[key]

        for key, val in row.items():
            if key.startswith("stage_") and val is not None:
                log_dict[f"{workflow}/{key}"] = val
            if key.startswith("perf_") and val is not None:
                log_dict[f"{workflow}/{key}"] = val
            if key in ("anomaly_count", "total_records") and val is not None:
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


def bench_forecasting_chronos(config, mode):
    rows = []
    for i in range(1, config["repeats"] + 1):
        result, latency, rss = timed_run(
            run_tsfm_forecasting_chronos,
            dataset_path=FORECAST_DATASET,
            timestamp_column=TIMESTAMP_COLUMN,
            target_columns=TARGET_COLUMNS,
            model_checkpoint="amazon/chronos-2",
            forecast_horizon=config["forecast_horizon"],
        )
        rows.append(make_row("forecasting_chronos", i, result, latency, rss, config, mode))
    return rows

def bench_finetuning_chronos(config, mode):
    rows = []
    for i in range(1, config["repeats"] + 1):
        result, latency, rss = timed_run(
            run_tsfm_finetuning_chronos,
            dataset_path=FINETUNE_DATASET,
            timestamp_column=TIMESTAMP_COLUMN,
            target_columns=TARGET_COLUMNS,
            model_checkpoint="amazon/chronos-2",
            forecast_horizon=config["forecast_horizon"],
        )
        rows.append(make_row("finetuning_chronos", i, result, latency, rss, config, mode))
    return rows

BENCHMARKS = {
    "forecasting": bench_forecasting,
    #"finetuning": bench_finetuning,
    "integrated_tsad": bench_integrated_tsad,
}

def bench_tsad_chronos(config, mode):
    rows = []
    for i in range(1, config["repeats"] + 1):
        result, latency, rss = timed_run(
            run_integrated_tsad_chronos,
            dataset_path=TSAD_DATASET,
            timestamp_column=TIMESTAMP_COLUMN,
            target_columns=TARGET_COLUMNS,
            model_checkpoint="amazon/chronos-2",
        )
        rows.append(make_row("tsad_chronos", i, result, latency, rss, config, mode))
    return rows

def bench_integrated_tsad_chronos(config, mode):
    rows = []
    for i in range(1, config["repeats"] + 1):
        result, latency, rss = timed_run(
            run_integrated_tsad_chronos,
            dataset_path=TSAD_DATASET,
            timestamp_column=TIMESTAMP_COLUMN,
            target_columns=TARGET_COLUMNS,
            model_checkpoint="amazon/chronos-2",
        )
        rows.append(make_row("integrated_tsad_chronos", i, result, latency, rss, config, mode))
    return rows

BENCHMARKS_CHRONOS = {
    "forecasting_chronos": bench_forecasting_chronos,
    "finetuning_chronos": bench_finetuning_chronos,
    "tsad_chronos": bench_tsad_chronos,
    "integrated_tsad_chronos": bench_integrated_tsad_chronos,
}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["baseline", "cache_only", "combined"],
                        choices=list(MODES.keys()))
    parser.add_argument("--model", default="ttm", choices=["ttm", "chronos"])
    args = parser.parse_args()
    modes = args.modes
    benchmarks = BENCHMARKS_CHRONOS if args.model == "chronos" else BENCHMARKS

    config = {
        "model_checkpoint": "ttm_96_28",
        "forecast_horizon": 24,
        "seed": 42,
        "repeats": 3,
    }
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    all_rows = []

    print("=" * 72)
    print("TSFM EXTERNAL BENCHMARKING HARNESS")
    print("=" * 72)
    print(f"Entity: {WANDB_ENTITY} | Project: {WANDB_PROJECT}")
    print(f"Dataset: real (main_chiller9_small.csv)")
    print(f"Repeats per workflow: {config['repeats']}")
    print(f"Modes: {modes}")
    print()

    for mode in modes:
        apply_mode(mode)
        set_seed(config["seed"])
        print("#" * 72)
        print(f"# MODE: {mode}")
        print("#" * 72)
        for workflow, bench_fn in benchmarks.items():
            print(f"--- {workflow.upper()} [{mode}] ---")
            try:
                rows = bench_fn(config, mode)
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
            log_to_wandb(workflow, rows, summary, run_tag=mode)
            write_json(OUTPUT_DIR / f"{workflow}_{mode}_{timestamp}.json", rows)
            write_csv(OUTPUT_DIR / f"{workflow}_{mode}_{timestamp}.csv", rows)
            write_json(OUTPUT_DIR / f"{workflow}_{mode}_summary_{timestamp}.json", summary)
            all_rows.extend(rows)
            print()

    write_json(OUTPUT_DIR / f"all_benchmarks_{timestamp}.json", all_rows)
    write_csv(OUTPUT_DIR / f"all_benchmarks_{timestamp}.csv", all_rows)

    print("=" * 72)
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
