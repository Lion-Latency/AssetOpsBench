from __future__ import annotations

import csv
import json
import os
from dotenv import load_dotenv
import random
import re
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import pandas as pd
import psutil
import wandb

load_dotenv()
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = Path(os.getenv("PATH_TO_DATASETS_DIR", "/home/shared/tsfm_profiling_data/datasets")) / "dhaval_data"
MODELS_DIR = REPO_ROOT / "src" / "servers" / "tsfm" / "artifacts" / "tsfm_models"
OUTPUT_DIR = REPO_ROOT / "tsfm_profiling" / "harness" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BENCH_ROWS_ENV = "TSFM_BENCH_ROWS"
BENCH_SWEEP_K = [100]
CHUNK_SIZE = 200_000
MISSING_ASSET_ID = "__missing_asset_id__"

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
WANDB_PROJECT = "hpml-project-final"

TIMESTAMP_COLUMN = "timestamp"
SOURCE_DATASET = DATA_DIR / "main_flat.csv"
ID_COLUMNS = ["asset_id"]
EXCLUDED_COLUMNS = {TIMESTAMP_COLUMN, *ID_COLUMNS, "selector"}


def _load_target_groups(dataset_path: Path) -> dict[str, list[str]]:
    header = pd.read_csv(dataset_path, nrows=0)
    candidate_columns = [
        column for column in header.columns if column not in EXCLUDED_COLUMNS
    ]

    asset_ids: set[str] = set()
    for chunk in pd.read_csv(dataset_path, usecols=ID_COLUMNS, chunksize=CHUNK_SIZE, dtype=str):
        asset_ids.update(chunk[ID_COLUMNS[0]].dropna().unique().tolist())

    target_groups: dict[str, list[str]] = {}
    for asset_id in sorted(asset_ids):
        group_columns = [
            column for column in candidate_columns if column.startswith(f"{asset_id} ")
        ]
        if group_columns:
            target_groups[asset_id] = group_columns
    return target_groups


def _prepare_sampled_dataset(source_path: Path, output_dir: Path, requested_rows: int) -> Path:
    row_label = f"{requested_rows // 1_000}k" if requested_rows % 1_000 == 0 else str(requested_rows)
    sample_path = output_dir / f"sample_{row_label}.csv"
    if sample_path.exists():
        with sample_path.open("r", encoding="utf-8", newline="") as handle:
            if max(sum(1 for _ in handle) - 1, 0) == requested_rows:
                return sample_path

    asset_counts: dict[str, int] = {}
    for chunk in pd.read_csv(
        source_path,
        usecols=ID_COLUMNS,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        asset_ids = chunk[ID_COLUMNS[0]].astype("string").fillna(MISSING_ASSET_ID)
        for asset_id, count in asset_ids.value_counts(sort=False).items():
            asset_key = str(asset_id)
            asset_counts[asset_key] = asset_counts.get(asset_key, 0) + int(count)

    total_rows = sum(asset_counts.values())
    if requested_rows > total_rows:
        raise ValueError(
            f"Requested {requested_rows} rows but dataset only has {total_rows} rows"
        )

    asset_ids = sorted(asset_counts)
    quotas = {asset_id: 0 for asset_id in asset_ids}
    remaining_rows = requested_rows

    if requested_rows >= len(asset_ids):
        for asset_id in asset_ids:
            quotas[asset_id] = 1
            remaining_rows -= 1

    remaining_capacity = {
        asset_id: asset_counts[asset_id] - quotas[asset_id]
        for asset_id in asset_ids
    }
    capacity_total = sum(remaining_capacity.values())
    if remaining_rows > 0 and capacity_total > 0:
        exact_additions = {
            asset_id: remaining_rows * remaining_capacity[asset_id] / capacity_total
            for asset_id in asset_ids
        }
        base_additions = {
            asset_id: int(exact_additions[asset_id])
            for asset_id in asset_ids
        }
        for asset_id in asset_ids:
            quotas[asset_id] += base_additions[asset_id]
        remaining_rows -= sum(base_additions.values())

        if remaining_rows > 0:
            ranked_assets = sorted(
                asset_ids,
                key=lambda asset_id: (
                    exact_additions[asset_id] - base_additions[asset_id],
                    asset_id,
                ),
                reverse=True,
            )
            for asset_id in ranked_assets:
                if remaining_rows == 0:
                    break
                if quotas[asset_id] >= asset_counts[asset_id]:
                    continue
                quotas[asset_id] += 1
                remaining_rows -= 1

    remaining_quotas = dict(quotas)

    tmp_path = sample_path.with_suffix(".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    rows_written = 0
    write_header = True
    for chunk in pd.read_csv(source_path, chunksize=CHUNK_SIZE, low_memory=False):
        chunk = chunk.copy()
        chunk["_bench_asset_id"] = (
            chunk[ID_COLUMNS[0]].astype("string").fillna(MISSING_ASSET_ID)
        )

        selected_indices: list[int] = []
        for asset_id, asset_rows in chunk.groupby("_bench_asset_id", sort=False):
            asset_key = str(asset_id)
            remaining = remaining_quotas.get(asset_key, 0)
            if remaining <= 0:
                continue
            take_count = min(remaining, len(asset_rows))
            selected_indices.extend(asset_rows.index[:take_count].tolist())
            remaining_quotas[asset_key] -= take_count

        if not selected_indices:
            continue

        selected_chunk = (
            chunk.loc[sorted(selected_indices)]
            .drop(columns=["_bench_asset_id"])
        )
        selected_chunk.to_csv(
            tmp_path,
            mode="a",
            header=write_header,
            index=False,
        )
        write_header = False
        rows_written += len(selected_chunk)

        if rows_written >= requested_rows:
            break

    if rows_written != requested_rows:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(
            f"Expected to write {requested_rows} rows to {sample_path}, wrote {rows_written}"
        )

    tmp_path.replace(sample_path)
    return sample_path


def _build_dataset_configs() -> list[dict[str, Any]]:
    raw_value = os.getenv(BENCH_ROWS_ENV)
    requested_rows_list: list[int | None] = []

    if raw_value is not None and raw_value.strip():
        try:
            requested_rows = int(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"{BENCH_ROWS_ENV} must be a positive integer, got {raw_value!r}"
            ) from exc
        if requested_rows <= 0:
            raise ValueError(
                f"{BENCH_ROWS_ENV} must be greater than 0, got {requested_rows}"
            )
        requested_rows_list = [requested_rows]
    else:
        for sweep_k in BENCH_SWEEP_K:
            sweep_size = int(sweep_k)
            if sweep_size <= 0:
                raise ValueError(
                    f"BENCH_SWEEP_K entries must be greater than 0, got {sweep_size}"
                )
            requested_rows_list.append(sweep_size * 1_000)

    dataset_configs: list[dict[str, Any]] = []
    for requested_rows in requested_rows_list or [None]:
        dataset_path = SOURCE_DATASET.resolve()
        dataset_label = "real"
        dataset_rows = None

        if requested_rows is not None:
            dataset_path = _prepare_sampled_dataset(
                SOURCE_DATASET, OUTPUT_DIR, requested_rows
            ).resolve()
            dataset_label = dataset_path.stem
            dataset_rows = requested_rows

        target_groups = _load_target_groups(dataset_path)
        if not target_groups:
            raise RuntimeError(f"No target groups found in dataset {dataset_path}")

        dataset_configs.append(
            {
                "dataset_label": dataset_label,
                "dataset_path": str(dataset_path),
                "dataset_rows": dataset_rows,
                "requested_rows": requested_rows,
                "target_groups": target_groups,
            }
        )

    return dataset_configs


def _iter_target_groups(config: dict[str, Any]):
    for target_group, target_columns in config["target_groups"].items():
        if target_group == "CQPA AHU 1": continue
        group_config = dict(config)
        group_config["target_group"] = target_group
        group_config["target_count"] = len(target_columns)
        yield group_config, target_columns


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


def _tsad_csv_metrics(path: Path) -> dict:
    out: dict = {}
    df = pd.read_csv(path)
    if df.empty:
        return out

    if "anomaly_label" in df.columns:
        out["perf_anomaly_rate"] = float(df["anomaly_label"].astype(float).mean())
    if "anomaly_score" in df.columns:
        out["perf_mean_anomaly_score"] = float(df["anomaly_score"].mean())
        out["perf_max_anomaly_score"] = float(df["anomaly_score"].max())

    has_bounds = {"value", "upper_bound", "lower_bound"}.issubset(df.columns)
    if has_bounds:
        within = (df["value"] >= df["lower_bound"]) & (df["value"] <= df["upper_bound"])
        out["perf_interval_coverage"] = float(within.mean())
        out["perf_mean_interval_width"] = float((df["upper_bound"] - df["lower_bound"]).mean())
        y_pred = (df["upper_bound"] + df["lower_bound"]) / 2.0
        diff = (y_pred - df["value"]).astype(float)
        out["perf_rmse"] = float(np.sqrt((diff ** 2).mean()))
        out["perf_mae"] = float(diff.abs().mean())
        denom = df["value"].astype(float).abs().replace(0, np.nan)
        mape = (diff.abs() / denom * 100).dropna()
        if len(mape):
            out["perf_mape"] = float(mape.mean())

    if "split" in df.columns:
        test_df = df[df["split"] == "test"]
        if not test_df.empty:
            if "anomaly_label" in test_df.columns:
                out["perf_test_anomaly_rate"] = float(test_df["anomaly_label"].astype(float).mean())
            if has_bounds:
                within = (test_df["value"] >= test_df["lower_bound"]) & (
                    test_df["value"] <= test_df["upper_bound"]
                )
                out["perf_test_interval_coverage"] = float(within.mean())

    return out


def extract_performance_metrics(result) -> dict:
    metrics = {}
    results_file = getattr(result, "results_file", None)
    if results_file and Path(results_file).exists():
        suffix = Path(results_file).suffix.lower()
        try:
            if suffix == ".json":
                with open(results_file, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "performance" in data:
                    metrics.update(_flatten_performance(data["performance"]))
                if isinstance(data, dict) and "anomaly_count" in data:
                    metrics["anomaly_count"] = data["anomaly_count"]
                if isinstance(data, dict) and "total_records" in data:
                    metrics["total_records"] = data["total_records"]
            elif suffix == ".csv":
                metrics.update(_tsad_csv_metrics(Path(results_file)))
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
        "target_group": config.get("target_group"),
        "target_count": config.get("target_count"),
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
        "dataset": config.get("dataset_label"),
        "dataset_path": config.get("dataset_path"),
        "dataset_rows": config.get("dataset_rows"),
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
    cold_end_to_end = [r["end_to_end_ms"] for r in cold if r.get("end_to_end_ms") is not None]
    return {
        "total_runs": len(rows),
        "successful_runs": sum(r["status"] == "success" for r in rows),
        "failed_runs": sum(r["status"] != "success" for r in rows),
        "cold_start_latency_sec": mean([r["latency_sec"] for r in cold]) if cold else None,
        "steady_state_avg_latency_sec": mean([r["latency_sec"] for r in steady]) if steady else None,
        "steady_state_avg_rss_delta_mb": mean([r["cpu_rss_delta_mb"] for r in steady]) if steady else None,
        "cold_start_end_to_end_ms": mean(cold_end_to_end) if cold_end_to_end else None,
        "steady_state_avg_end_to_end_ms": mean([r["end_to_end_ms"] for r in steady if r.get("end_to_end_ms")]) if steady else None,
        "steady_state_avg_overhead_ms": mean([r["overhead_ms"] for r in steady if r.get("overhead_ms")]) if steady else None,
    }


def bench_forecasting(config, mode):
    rows = []
    for group_config, target_columns in _iter_target_groups(config):
        for i in range(1, config["repeats"] + 1):
            result, latency, rss = timed_run(
                run_tsfm_forecasting,
                dataset_path=config["dataset_path"],
                timestamp_column=TIMESTAMP_COLUMN,
                target_columns=target_columns,
                model_checkpoint=config["model_checkpoint"],
                forecast_horizon=config["forecast_horizon"],
                id_columns=ID_COLUMNS,
            )
            rows.append(make_row("forecasting", i, result, latency, rss, group_config, mode))
    return rows


def bench_finetuning(config, mode):
    rows = []
    for group_config, target_columns in _iter_target_groups(config):
        for i in range(1, config["repeats"] + 1):
            result, latency, rss = timed_run(
                run_tsfm_finetuning,
                dataset_path=config["dataset_path"],
                timestamp_column=TIMESTAMP_COLUMN,
                target_columns=target_columns,
                model_checkpoint=config["model_checkpoint"],
                save_model_dir="tunedmodels",
                forecast_horizon=config["forecast_horizon"],
                n_finetune=0.05,
                n_test=0.05,
                id_columns=ID_COLUMNS,
            )
            rows.append(make_row("finetuning", i, result, latency, rss, group_config, mode))
    return rows


def bench_tsad(config, mode):
    rows = []
    for group_config, target_columns in _iter_target_groups(config):
        forecast_result, _, _ = timed_run(
            run_tsfm_forecasting,
            dataset_path=config["dataset_path"],
            timestamp_column=TIMESTAMP_COLUMN,
            target_columns=target_columns,
            model_checkpoint=config["model_checkpoint"],
            forecast_horizon=config["forecast_horizon"],
            id_columns=ID_COLUMNS,
        )
        if getattr(forecast_result, "status", None) != "success":
            raise RuntimeError(
                f"Forecasting pre-step failed for {group_config['target_group']}: {forecast_result}"
            )
        forecast_file = forecast_result.results_file

        for i in range(1, config["repeats"] + 1):
            result, latency, rss = timed_run(
                run_tsad,
                dataset_path=config["dataset_path"],
                tsfm_output_json=forecast_file,
                timestamp_column=TIMESTAMP_COLUMN,
                target_columns=target_columns,
                task="fit",
                false_alarm=0.05,
                n_calibration=0.2,
                id_columns=ID_COLUMNS,
            )
            rows.append(make_row("tsad", i, result, latency, rss, group_config, mode))
    return rows


def bench_integrated_tsad(config, mode):
    rows = []
    for group_config, target_columns in _iter_target_groups(config):
        for i in range(1, config["repeats"] + 1):
            result, latency, rss = timed_run(
                run_integrated_tsad,
                dataset_path=config["dataset_path"],
                timestamp_column=TIMESTAMP_COLUMN,
                target_columns=target_columns,
                model_checkpoint=config["model_checkpoint"],
                false_alarm=0.05,
                n_calibration=0.2,
                id_columns=ID_COLUMNS,
            )
            rows.append(make_row("integrated_tsad", i, result, latency, rss, group_config, mode))
    return rows


def log_to_wandb(workflow, rows, summary, config, run_tag="baseline", mode=None):
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=(
            f"bench_{workflow}_{config.get('dataset_label')}_{run_tag}_"
            f"{time.strftime('%Y%m%d_%H%M%S')}"
        ),
        config={
            "workflow": workflow,
            "mode": mode,
            "cache_enabled": os.environ.get("TSFM_CACHE_ENABLED"),
            "preprocess_opt": os.environ.get("TSFM_PREPROCESS_OPT"),
            "preprocess_workers": os.environ.get("TSFM_PREPROCESS_WORKERS"),
            "model_checkpoint": config.get("model_checkpoint"),
            "repeats": config.get("repeats"),
            "dataset": config.get("dataset_label"),
            "dataset_path": config.get("dataset_path"),
            "dataset_rows": config.get("dataset_rows"),
            "run_tag": run_tag,
        },
        reinit="finish_previous",
    )

    horizon_re = re.compile(r"^perf_(.+)_h(\d+)$")
    # base -> run_index -> horizon -> value (collected for line_series charts)
    horizon_series: dict = {}

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
            if val is None:
                continue
            if key.startswith("stage_"):
                log_dict[f"{workflow}/{key}"] = val
            elif key.startswith("perf_"):
                m = horizon_re.match(key)
                if m:
                    base, h = m.group(1), int(m.group(2))
                    try:
                        fv = float(val)
                    except (TypeError, ValueError):
                        continue
                    if not np.isfinite(fv):
                        continue
                    horizon_series.setdefault(base, {}).setdefault(
                        row["run_index"], {}
                    )[h] = fv
                else:
                    log_dict[f"{workflow}/{key}"] = val
            elif key in ("anomaly_count", "total_records"):
                log_dict[f"{workflow}/{key}"] = val

        run.log(log_dict)

    for base, by_run in horizon_series.items():
        all_h = sorted({h for run_d in by_run.values() for h in run_d})
        keys, ys = [], []
        for run_idx in sorted(by_run):
            series = [by_run[run_idx].get(h, float("nan")) for h in all_h]
            if not any(np.isfinite(v) for v in series):
                continue
            keys.append(f"run{run_idx}")
            ys.append(series)
        if not keys:
            continue
        run.log({
            f"{workflow}/perf_{base}": wandb.plot.line_series(
                xs=all_h,
                ys=ys,
                keys=keys,
                title=f"{workflow} {base} vs forecast horizon",
                xname="forecast_horizon",
            )
        })

    successful_rows = [row for row in rows if row["status"] == "success"]
    aggregate_metrics = {}

    latency_values = [row["latency_sec"] for row in successful_rows]
    if latency_values:
        aggregate_metrics[f"{workflow}/avg_latency_sec"] = mean(latency_values)
        aggregate_metrics[f"{workflow}/min_latency_sec"] = min(latency_values)
        aggregate_metrics[f"{workflow}/max_latency_sec"] = max(latency_values)

    end_to_end_values = [
        row["end_to_end_ms"]
        for row in successful_rows
        if row.get("end_to_end_ms") is not None
    ]
    if end_to_end_values:
        aggregate_metrics[f"{workflow}/avg_end_to_end_ms"] = mean(end_to_end_values)

    overhead_values = [
        row["overhead_ms"]
        for row in successful_rows
        if row.get("overhead_ms") is not None
    ]
    if overhead_values:
        aggregate_metrics[f"{workflow}/avg_overhead_ms"] = mean(overhead_values)

    if aggregate_metrics:
        run.log(aggregate_metrics)
        for key, value in aggregate_metrics.items():
            run.summary[key] = value

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
    for group_config, target_columns in _iter_target_groups(config):
        for i in range(1, config["repeats"] + 1):
            result, latency, rss = timed_run(
                run_tsfm_forecasting_chronos,
                dataset_path=config["dataset_path"],
                timestamp_column=TIMESTAMP_COLUMN,
                target_columns=target_columns,
                model_checkpoint="amazon/chronos-2",
                forecast_horizon=config["forecast_horizon"],
                id_columns=ID_COLUMNS,
            )
            rows.append(make_row("forecasting_chronos", i, result, latency, rss, group_config, mode))
    return rows

def bench_finetuning_chronos(config, mode):
    rows = []
    for group_config, target_columns in _iter_target_groups(config):
        for i in range(1, config["repeats"] + 1):
            result, latency, rss = timed_run(
                run_tsfm_finetuning_chronos,
                dataset_path=config["dataset_path"],
                timestamp_column=TIMESTAMP_COLUMN,
                target_columns=target_columns,
                model_checkpoint="amazon/chronos-2",
                forecast_horizon=config["forecast_horizon"],
                id_columns=ID_COLUMNS,
            )
            rows.append(make_row("finetuning_chronos", i, result, latency, rss, group_config, mode))
    return rows

BENCHMARKS = {
    "forecasting": bench_forecasting,
    "finetuning": bench_finetuning,
    "tsad": bench_tsad,
    "integrated_tsad": bench_integrated_tsad,
}

def bench_tsad_chronos(config, mode):
    rows = []
    for group_config, target_columns in _iter_target_groups(config):
        forecast_result, _, _ = timed_run(
            run_tsfm_forecasting_chronos,
            dataset_path=config["dataset_path"],
            timestamp_column=TIMESTAMP_COLUMN,
            target_columns=target_columns,
            model_checkpoint="amazon/chronos-2",
            forecast_horizon=config["forecast_horizon"],
            id_columns=ID_COLUMNS,
        )
        if getattr(forecast_result, "status", None) != "success":
            raise RuntimeError(
                f"Chronos forecasting pre-step failed for {group_config['target_group']}: {forecast_result}"
            )
        forecast_file = forecast_result.results_file

        for i in range(1, config["repeats"] + 1):
            result, latency, rss = timed_run(
                run_tsad_chronos,
                dataset_path=config["dataset_path"],
                tsfm_output_json=forecast_file,
                timestamp_column=TIMESTAMP_COLUMN,
                target_columns=target_columns,
                model_checkpoint="amazon/chronos-2",
                n_calibration=0.2,
                id_columns=ID_COLUMNS,
            )
            rows.append(make_row("tsad_chronos", i, result, latency, rss, group_config, mode))
    return rows

def bench_integrated_tsad_chronos(config, mode):
    rows = []
    for group_config, target_columns in _iter_target_groups(config):
        for i in range(1, config["repeats"] + 1):
            result, latency, rss = timed_run(
                run_integrated_tsad_chronos,
                dataset_path=config["dataset_path"],
                timestamp_column=TIMESTAMP_COLUMN,
                target_columns=target_columns,
                model_checkpoint="amazon/chronos-2",
                id_columns=ID_COLUMNS,
            )
            rows.append(make_row("integrated_tsad_chronos", i, result, latency, rss, group_config, mode))
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
    parser.add_argument(
        "--workflows",
        nargs="+",
        choices=sorted(set(BENCHMARKS) | set(BENCHMARKS_CHRONOS)),
    )
    args = parser.parse_args()
    modes = args.modes
    benchmarks = BENCHMARKS_CHRONOS if args.model == "chronos" else BENCHMARKS
    selected_workflows = set(args.workflows) if args.workflows else None

    dataset_configs = _build_dataset_configs()

    base_config = {
        "model_checkpoint": "ttm_96_28",
        "forecast_horizon": 24,
        "seed": 42,
        "repeats": 1,
    }
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    all_rows = []

    print("=" * 72)
    print("TSFM EXTERNAL BENCHMARKING HARNESS")
    print("=" * 72)
    print(f"Entity: {WANDB_ENTITY} | Project: {WANDB_PROJECT}")
    print(f"Model checkpoint: {base_config['model_checkpoint']}")
    if os.getenv(BENCH_ROWS_ENV):
        print(f"{BENCH_ROWS_ENV} override: {os.getenv(BENCH_ROWS_ENV)}")
    else:
        print(f"Sweep sizes (k rows): {BENCH_SWEEP_K}")
    print(f"Repeats per workflow: {base_config['repeats']}")
    print(f"Modes: {modes}")
    if selected_workflows is not None:
        print(f"Workflows: {sorted(selected_workflows)}")
    print()

    for dataset_config in dataset_configs:
        config = dict(base_config)
        config.update(dataset_config)
        dataset_rows: list[dict[str, Any]] = []

        print("~" * 72)
        print(f"Dataset: {config['dataset_label']}")
        print(f"Dataset path: {config['dataset_path']}")
        if config["requested_rows"] is not None:
            print(f"Requested sampled rows: {config['requested_rows']}")
        print(f"Target groups: {sorted(config['target_groups'])}")
        print("~" * 72)

        for mode in modes:
            apply_mode(mode)
            set_seed(config["seed"])
            print("#" * 72)
            print(f"# MODE: {mode}")
            print("#" * 72)
            for workflow, bench_fn in benchmarks.items():
                if selected_workflows is not None and workflow not in selected_workflows:
                    continue
                print(f"--- {workflow.upper()} [{mode}] ---")
                try:
                    rows = bench_fn(config, mode)
                except Exception as e:
                    print(f"  SKIPPED ({e})")
                    continue
                for r in rows:
                    group_label = f"[{r['target_group']}] " if r.get("target_group") else ""
                    print(
                        f"  {group_label}Run {r['run_index']} [{r['run_type']}] "
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
                log_to_wandb(workflow, rows, summary, config, run_tag=mode)
                write_json(
                    OUTPUT_DIR / f"{workflow}_{mode}_{config['dataset_label']}_{timestamp}.json",
                    rows,
                )
                write_csv(
                    OUTPUT_DIR / f"{workflow}_{mode}_{config['dataset_label']}_{timestamp}.csv",
                    rows,
                )
                write_json(
                    OUTPUT_DIR
                    / f"{workflow}_{mode}_summary_{config['dataset_label']}_{timestamp}.json",
                    summary,
                )
                dataset_rows.extend(rows)
                all_rows.extend(rows)
                print()

        write_json(
            OUTPUT_DIR / f"all_benchmarks_{config['dataset_label']}_{timestamp}.json",
            dataset_rows,
        )
        write_csv(
            OUTPUT_DIR / f"all_benchmarks_{config['dataset_label']}_{timestamp}.csv",
            dataset_rows,
        )

    write_json(OUTPUT_DIR / f"all_benchmarks_sweep_{timestamp}.json", all_rows)
    write_csv(OUTPUT_DIR / f"all_benchmarks_sweep_{timestamp}.csv", all_rows)

    print("=" * 72)
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
