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
from typing import Any, Optional

import numpy as np
import pandas as pd
import psutil
import wandb

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")
sys.path.insert(0, str(REPO_ROOT))

DATASETS_ROOT = Path(os.getenv("PATH_TO_DATASETS_DIR", str(REPO_ROOT.parent))).expanduser()
MODELS_DIR = Path(
    os.getenv(
        "PATH_TO_MODELS_DIR",
        str(REPO_ROOT / "src" / "servers" / "tsfm" / "artifacts" / "tsfm_models"),
    )
).expanduser()
OUTPUT_DIR = Path(
    os.getenv(
        "PATH_TO_OUTPUTS_DIR",
        str(REPO_ROOT / "tsfm_profiling" / "harness" / "results"),
    )
).expanduser()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 200_000

os.environ.setdefault("PATH_TO_DATASETS_DIR", str(DATASETS_ROOT))
os.environ.setdefault("PATH_TO_MODELS_DIR", str(MODELS_DIR))
os.environ.setdefault("PATH_TO_OUTPUTS_DIR", str(OUTPUT_DIR))

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
from tsfm_profiling.harness._stdio_client import StdioBenchClient, shim_result
from tsfm_profiling.harness import _plots

# Maps MCP tool name -> in-process callable. Used by `_invoke_tool` when
# `--transport in_process` so we keep one dispatch site for both transports.
_TOOL_REGISTRY = {
    "run_tsfm_forecasting": run_tsfm_forecasting,
    "run_tsfm_finetuning": run_tsfm_finetuning,
    "run_tsad": run_tsad,
    "run_integrated_tsad": run_integrated_tsad,
    "run_tsfm_forecasting_chronos": run_tsfm_forecasting_chronos,
    "run_tsfm_finetuning_chronos": run_tsfm_finetuning_chronos,
    "run_tsad_chronos": run_tsad_chronos,
    "run_integrated_tsad_chronos": run_integrated_tsad_chronos,
}

# Mutable state set by `apply_mode`. Module globals because the existing
# bench_* fns are sync and we don't want to thread a context through every
# signature just for transport selection.
_TRANSPORT: str = "in_process"
_STDIO_CLIENT: Optional["StdioBenchClient"] = None

MODES = {
    "baseline":   {"TSFM_CACHE_ENABLED": "0", "TSFM_PREPROCESS_OPT": "0"},
    "cache_only": {"TSFM_CACHE_ENABLED": "1", "TSFM_PREPROCESS_OPT": "0"},
    "combined":   {"TSFM_CACHE_ENABLED": "1", "TSFM_PREPROCESS_OPT": "1",
                   "TSFM_PREPROCESS_WORKERS": "4",
                   "TSFM_PREPROCESS_EXECUTOR": "thread"},
    "parallelism_only": {"TSFM_CACHE_ENABLED": "0", "TSFM_PREPROCESS_OPT": "1",
                         "TSFM_PREPROCESS_WORKERS": "4",
                         "TSFM_PREPROCESS_EXECUTOR": "thread"},
    # Inference-side opts ported from sam/AssetOpsBench. Disjoint env vars
    # so they stack cleanly on top of any preprocessing-cache mode above.
    "model_cache":   {"TSFM_CACHE_ENABLED": "1", "TSFM_MODEL_CACHE": "1"},
    "fast_trainer":  {"TSFM_CACHE_ENABLED": "1", "TSFM_MODEL_CACHE": "1",
                      "TSFM_FAST_TRAINER": "1"},
    "bf16":          {"TSFM_CACHE_ENABLED": "1", "TSFM_MODEL_CACHE": "1",
                      "TSFM_BF16": "1", "TSFM_FAST_TRAINER": "1"},
    "compile":       {"TSFM_CACHE_ENABLED": "1", "TSFM_MODEL_CACHE": "1",
                      "TSFM_COMPILE": "1", "TSFM_FAST_TRAINER": "1"},
    "all_inference": {"TSFM_CACHE_ENABLED": "1", "TSFM_MODEL_CACHE": "1",
                      "TSFM_BF16": "1", "TSFM_COMPILE": "1",
                      "TSFM_FAST_TRAINER": "1"},
}

_OPT_ENV_VARS = (
    "TSFM_CACHE_ENABLED", "TSFM_PREPROCESS_OPT",
    "TSFM_PREPROCESS_WORKERS", "TSFM_PREPROCESS_EXECUTOR",
    "TSFM_MODEL_CACHE", "TSFM_COMPILE", "TSFM_BF16",
    "TSFM_FAST_TRAINER",
)


def apply_mode(mode: str) -> None:
    """Apply env vars for the given mode in this process.

    For in_process transport this also clears the in-process preprocessing
    cache so each mode starts cold. For stdio transport the server gets the
    env vars at subprocess start (see `enter_stdio_for_mode`); the in-process
    cache here is irrelevant.
    """
    for k in _OPT_ENV_VARS:
        os.environ.pop(k, None)
    for k, v in MODES[mode].items():
        os.environ[k] = v
    if _TRANSPORT == "in_process":
        _cache.clear()


def mode_env_overrides(mode: str) -> dict:
    """Just the env-var overrides for a mode (used to seed the stdio server)."""
    return dict(MODES[mode])


def set_transport(transport: str) -> None:
    global _TRANSPORT
    if transport not in ("in_process", "stdio"):
        raise ValueError(f"unknown transport: {transport}")
    _TRANSPORT = transport


def enter_stdio_for_mode(
    mode: str,
    server_cmd: Optional[str] = None,
    server_args: Optional[list] = None,
) -> StdioBenchClient:
    """Open a fresh stdio session for the given mode and stash it globally.

    Defaults `server_cmd` to `sys.executable -m servers.tsfm.main` and prepends
    `<REPO_ROOT>/src` to `PYTHONPATH` so the subprocess imports MainProj's
    source rather than whatever `tsfm-mcp-server` console script the active
    venv was installed against. Override `--server-cmd` to point at an
    explicit entry point if you want a different server build.
    """
    global _STDIO_CLIENT
    if _STDIO_CLIENT is not None:
        _STDIO_CLIENT.__exit__(None, None, None)
        _STDIO_CLIENT = None
    overrides = mode_env_overrides(mode)
    src_dir = str(REPO_ROOT / "src")
    existing_pp = os.environ.get("PYTHONPATH", "")
    overrides["PYTHONPATH"] = (
        src_dir + (os.pathsep + existing_pp if existing_pp else "")
    )
    if server_cmd is None:
        server_cmd = sys.executable
        if server_args is None:
            server_args = ["-m", "servers.tsfm.main"]
    cli = StdioBenchClient(
        env_overrides=overrides,
        server_cmd=server_cmd,
        server_args=server_args,
    )
    cli.__enter__()
    _STDIO_CLIENT = cli
    return cli


def exit_stdio() -> None:
    global _STDIO_CLIENT
    if _STDIO_CLIENT is not None:
        _STDIO_CLIENT.__exit__(None, None, None)
        _STDIO_CLIENT = None


def _invoke_tool(tool_name: str, args: dict):
    """Dispatch one tool call and return (result_obj, latency_sec, rss_delta_mb).

    For stdio transport the server's stage-breakdown report is pulled from
    the per-call JSONL and stashed onto `_emit_metrics._last_report` (the
    same global `make_row` already reads), so downstream row construction
    stays transport-agnostic.
    """
    process = psutil.Process(os.getpid())
    start_rss = process.memory_info().rss
    start = time.perf_counter()
    if _TRANSPORT == "in_process":
        fn = _TOOL_REGISTRY[tool_name]
        result = fn(**args)
        latency = time.perf_counter() - start
        rss_delta = bytes_to_mb(process.memory_info().rss - start_rss)
        return result, latency, rss_delta

    # stdio
    assert _STDIO_CLIENT is not None, "enter_stdio_for_mode() not called"
    data = _STDIO_CLIENT.call(tool_name, args)
    latency = time.perf_counter() - start
    rss_delta = bytes_to_mb(process.memory_info().rss - start_rss)
    report = _STDIO_CLIENT.read_latest_report()
    _emit_metrics._last_report = report or {}
    return shim_result(data), latency, rss_delta


WANDB_ENTITY = os.getenv("WANDB_ENTITY", "lion-latency")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "hpml-project-final")

TIMESTAMP_COLUMN = "timestamp"
DATA_DIR = REPO_ROOT / "tsfm_profiling" / "data"
SOURCE_DATASET = DATA_DIR / "sample.csv"
ID_COLUMNS = ["asset_id"]
EXCLUDED_COLUMNS = {TIMESTAMP_COLUMN, *ID_COLUMNS, "selector"}
CHRONOS_TSAD_EXCLUDED_TARGET_PATTERN = re.compile(
    r"\b(status|schedule|command)\b", re.IGNORECASE
)


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


def _build_dataset_configs() -> list[dict[str, Any]]:
    if not SOURCE_DATASET.exists():
        raise FileNotFoundError(
            f"Dataset not found at {SOURCE_DATASET}. "
            f"Place sample.csv at tsfm_profiling/data/sample.csv to run the benchmark."
        )
    dataset_path = SOURCE_DATASET.resolve()
    target_groups = _load_target_groups(dataset_path)
    if not target_groups:
        raise RuntimeError(f"No target groups found in dataset {dataset_path}")
    return [
        {
            "dataset_label": SOURCE_DATASET.stem,
            "dataset_path": str(dataset_path),
            "target_groups": target_groups,
        }
    ]


TSAD_GROUPS_ALLOWLIST = {"Chiller 3"}


def _filter_chronos_tsad_target_columns(target_columns: list[str]) -> list[str]:
    return [
        column
        for column in target_columns
        if not CHRONOS_TSAD_EXCLUDED_TARGET_PATTERN.search(column)
    ]


def _iter_target_groups(config: dict[str, Any], target_columns_filter=None, groups_allowlist=None):
    for target_group, target_columns in config["target_groups"].items():
        if target_group == "CQPA AHU 1": continue
        if groups_allowlist is not None and target_group not in groups_allowlist:
            continue
        if target_columns_filter is not None:
            target_columns = target_columns_filter(target_columns)
        if not target_columns:
            continue
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

    nvml_keys = (
        "gpu_power_w_mean", "gpu_power_w_max",
        "gpu_util_pct_mean", "gpu_util_pct_max",
        "gpu_mem_util_pct_mean",
        "sm_clock_mhz_mean", "mem_clock_mhz_mean",
        "nvml_samples",
    )
    for stage in stages:
        name = stage["stage"]
        row[f"stage_{name}_ms"] = stage.get("wall_clock_ms")
        row[f"stage_{name}_rss_delta_mb"] = stage.get("rss_delta_mb")
        if "gpu_mem_delta_mb" in stage:
            row[f"stage_{name}_gpu_alloc_delta_mb"] = stage.get("gpu_mem_delta_mb")
        if "gpu_mem_peak_mb" in stage:
            row[f"stage_{name}_gpu_peak_mb"] = stage.get("gpu_mem_peak_mb")
        for k in nvml_keys:
            if k in stage:
                row[f"stage_{name}_{k}"] = stage[k]

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
            result, latency, rss = _invoke_tool("run_tsfm_forecasting", {
                "dataset_path": config["dataset_path"],
                "timestamp_column": TIMESTAMP_COLUMN,
                "target_columns": target_columns,
                "model_checkpoint": config["model_checkpoint"],
                "forecast_horizon": config["forecast_horizon"],
                "id_columns": ID_COLUMNS,
            })
            rows.append(make_row("forecasting", i, result, latency, rss, group_config, mode))
    return rows


def bench_finetuning(config, mode):
    rows = []
    for group_config, target_columns in _iter_target_groups(config):
        for i in range(1, config["repeats"] + 1):
            result, latency, rss = _invoke_tool("run_tsfm_finetuning", {
                "dataset_path": config["dataset_path"],
                "timestamp_column": TIMESTAMP_COLUMN,
                "target_columns": target_columns,
                "model_checkpoint": config["model_checkpoint"],
                "save_model_dir": "tunedmodels",
                "forecast_horizon": config["forecast_horizon"],
                "n_finetune": 0.05,
                "n_test": 0.05,
                "id_columns": ID_COLUMNS,
            })
            rows.append(make_row("finetuning", i, result, latency, rss, group_config, mode))
    return rows


def bench_tsad(config, mode):
    rows = []
    for group_config, target_columns in _iter_target_groups(config, groups_allowlist=TSAD_GROUPS_ALLOWLIST):
        for col in target_columns:
            col_config = dict(group_config)
            col_config["target_group"] = f"{group_config['target_group']} / {col}"
            col_config["target_count"] = 1

            forecast_result, _, _ = _invoke_tool("run_tsfm_forecasting", {
                "dataset_path": config["dataset_path"],
                "timestamp_column": TIMESTAMP_COLUMN,
                "target_columns": [col],
                "model_checkpoint": config["model_checkpoint"],
                "forecast_horizon": config["forecast_horizon"],
                "id_columns": ID_COLUMNS,
            })
            if getattr(forecast_result, "status", None) != "success":
                continue
            forecast_file = forecast_result.results_file

            for i in range(1, config["repeats"] + 1):
                result, latency, rss = _invoke_tool("run_tsad", {
                    "dataset_path": config["dataset_path"],
                    "tsfm_output_json": forecast_file,
                    "timestamp_column": TIMESTAMP_COLUMN,
                    "target_columns": [col],
                    "task": "fit",
                    "false_alarm": 0.05,
                    "n_calibration": 0.2,
                    "id_columns": ID_COLUMNS,
                })
                rows.append(make_row("tsad", i, result, latency, rss, col_config, mode))
    return rows


def bench_integrated_tsad(config, mode):
    rows = []
    for group_config, target_columns in _iter_target_groups(config, groups_allowlist=TSAD_GROUPS_ALLOWLIST):
        for i in range(1, config["repeats"] + 1):
            result, latency, rss = _invoke_tool("run_integrated_tsad", {
                "dataset_path": config["dataset_path"],
                "timestamp_column": TIMESTAMP_COLUMN,
                "target_columns": target_columns,
                "model_checkpoint": config["model_checkpoint"],
                "false_alarm": 0.05,
                "n_calibration": 0.2,
                "id_columns": ID_COLUMNS,
            })
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
            "model_cache": os.environ.get("TSFM_MODEL_CACHE"),
            "compile": os.environ.get("TSFM_COMPILE"),
            "bf16": os.environ.get("TSFM_BF16"),
            "fast_trainer": os.environ.get("TSFM_FAST_TRAINER"),
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

        if ys:
            mean_series = [
                float(np.nanmean([y[i] for y in ys])) for i in range(len(all_h))
            ]
            if any(np.isfinite(v) for v in mean_series):
                keys.append("mean")
                ys.append(mean_series)
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

   
    panels: Dict[str, Any] = {}
    img_stages = _plots.stage_breakdown_stacked(rows, workflow, mode or run_tag)
    if img_stages is not None:
        panels[f"{workflow}/stage_breakdown_stacked"] = img_stages
    img_cs = _plots.cold_vs_steady_grouped(rows, workflow, mode or run_tag)
    if img_cs is not None:
        panels[f"{workflow}/cold_vs_steady"] = img_cs
    if panels:
        run.log(panels)

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


def log_summary_to_wandb(
    all_rows: list,
    modes: list,
    config: dict,
    transport: str,
    run_tag: str = "summary",
) -> None:
    if not all_rows:
        return
    workflows = sorted({r["workflow"] for r in all_rows if r.get("workflow")})
    if not workflows:
        return

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=(
            f"summary_{config.get('dataset_label', 'dataset')}_"
            f"{transport}_{time.strftime('%Y%m%d_%H%M%S')}"
        ),
        config={
            "kind": "summary",
            "transport": transport,
            "modes": modes,
            "workflows": workflows,
            "model_checkpoint": config.get("model_checkpoint"),
            "dataset": config.get("dataset_label"),
            "dataset_path": config.get("dataset_path"),
            "run_tag": run_tag,
        },
        reinit="finish_previous",
        tags=["summary", transport],
    )

    for workflow in workflows:
        rows_by_mode = {}
        for m in modes:
            rs = [
                r for r in all_rows
                if r.get("workflow") == workflow and r.get("mode") == m
            ]
            if rs:
                rows_by_mode[m] = rs
        if not rows_by_mode:
            continue

        panels: Dict[str, Any] = {}
        img = _plots.cross_mode_latency(rows_by_mode, workflow)
        if img is not None:
            panels[f"{workflow}/cross_mode_latency"] = img
        img = _plots.cross_mode_stage_breakdown(rows_by_mode, workflow)
        if img is not None:
            panels[f"{workflow}/cross_mode_stage_breakdown"] = img
        img = _plots.cross_mode_gpu_power(rows_by_mode, workflow, stage="inference")
        if img is not None:
            panels[f"{workflow}/cross_mode_gpu_power_inference"] = img
        if panels:
            run.log(panels)

        # Horizon perf: one line per mode, one chart per metric.
        horizon_data = _plots.cross_mode_horizon_perf(rows_by_mode, workflow)
        for base, (xs, mode_series) in horizon_data.items():
            if not xs or not mode_series:
                continue
            keys = list(mode_series.keys())
            ys = [mode_series[k] for k in keys]
            run.log({
                f"{workflow}/cross_mode_perf_{base}": wandb.plot.line_series(
                    xs=xs,
                    ys=ys,
                    keys=keys,
                    title=f"{workflow} {base} across modes",
                    xname="forecast_horizon",
                )
            })

        # Mode -> mean latency scalars also written to summary so the W&B
        # auto-table compares them at a glance.
        for m, rs in rows_by_mode.items():
            succ = [r["latency_sec"] for r in rs if r.get("status") == "success"]
            if succ:
                run.summary[f"{workflow}/{m}/mean_latency_sec"] = mean(succ)
            e2e = [r["end_to_end_ms"] for r in rs
                   if r.get("status") == "success" and r.get("end_to_end_ms") is not None]
            if e2e:
                run.summary[f"{workflow}/{m}/mean_e2e_ms"] = mean(e2e)

    run.finish()
    print(f"  W&B summary: {run.url}")


def bench_forecasting_chronos(config, mode):
    rows = []
    for group_config, target_columns in _iter_target_groups(config):
        for i in range(1, config["repeats"] + 1):
            result, latency, rss = _invoke_tool("run_tsfm_forecasting_chronos", {
                "dataset_path": config["dataset_path"],
                "timestamp_column": TIMESTAMP_COLUMN,
                "target_columns": target_columns,
                "model_checkpoint": "amazon/chronos-2",
                "forecast_horizon": config["forecast_horizon"],
                "id_columns": ID_COLUMNS,
            })
            rows.append(make_row("forecasting_chronos", i, result, latency, rss, group_config, mode))
    return rows

def bench_finetuning_chronos(config, mode):
    rows = []
    for group_config, target_columns in _iter_target_groups(config):
        for i in range(1, config["repeats"] + 1):
            result, latency, rss = _invoke_tool("run_tsfm_finetuning_chronos", {
                "dataset_path": config["dataset_path"],
                "timestamp_column": TIMESTAMP_COLUMN,
                "target_columns": target_columns,
                "model_checkpoint": "amazon/chronos-2",
                "forecast_horizon": config["forecast_horizon"],
                "id_columns": ID_COLUMNS,
            })
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
    for group_config, target_columns in _iter_target_groups(
        config, _filter_chronos_tsad_target_columns, groups_allowlist=TSAD_GROUPS_ALLOWLIST
    ):
        for col in target_columns:
            col_config = dict(group_config)
            col_config["target_group"] = f"{group_config['target_group']} / {col}"
            col_config["target_count"] = 1

            forecast_result, _, _ = _invoke_tool("run_tsfm_forecasting_chronos", {
                "dataset_path": config["dataset_path"],
                "timestamp_column": TIMESTAMP_COLUMN,
                "target_columns": [col],
                "model_checkpoint": "amazon/chronos-2",
                "forecast_horizon": config["forecast_horizon"],
                "id_columns": ID_COLUMNS,
            })
            if getattr(forecast_result, "status", None) != "success":
                continue
            forecast_file = forecast_result.results_file

            for i in range(1, config["repeats"] + 1):
                result, latency, rss = _invoke_tool("run_tsad_chronos", {
                    "dataset_path": config["dataset_path"],
                    "tsfm_output_json": forecast_file,
                    "timestamp_column": TIMESTAMP_COLUMN,
                    "target_columns": [col],
                    "model_checkpoint": "amazon/chronos-2",
                    "n_calibration": 0.2,
                    "id_columns": ID_COLUMNS,
                    "frequency_sampling": "15_minutes",
                })
                rows.append(make_row("tsad_chronos", i, result, latency, rss, col_config, mode))
    return rows

def bench_integrated_tsad_chronos(config, mode):
    rows = []
    for group_config, target_columns in _iter_target_groups(
        config, _filter_chronos_tsad_target_columns, groups_allowlist=TSAD_GROUPS_ALLOWLIST
    ):
        for i in range(1, config["repeats"] + 1):
            result, latency, rss = _invoke_tool("run_integrated_tsad_chronos", {
                "dataset_path": config["dataset_path"],
                "timestamp_column": TIMESTAMP_COLUMN,
                "target_columns": target_columns,
                "model_checkpoint": "amazon/chronos-2",
                "id_columns": ID_COLUMNS,
                "frequency_sampling": "15_minutes",
            })
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
    parser.add_argument(
        "--transport",
        default="in_process",
        choices=["in_process", "stdio"],
        help="in_process: call tool fns directly; "
             "stdio: spawn `tsfm-mcp-server` subprocess and route via MCP client "
             "(captures FastMCP dispatch + JSON-RPC framing overhead).",
    )
    parser.add_argument(
        "--server-cmd",
        default=None,
        help="Server entry-point used for --transport stdio. Defaults to "
             "`<this-python> -m servers.tsfm.main` with PYTHONPATH=<repo>/src "
             "so the subprocess imports MainProj source. Override only if you "
             "want to bench a different server build.",
    )
    parser.add_argument(
        "--server-args",
        nargs="*",
        default=None,
        help="Args appended to --server-cmd. Default: ['-m', 'servers.tsfm.main'] "
             "when --server-cmd is also default; empty otherwise.",
    )
    parser.add_argument(
        "--torch-profile",
        default=None,
        metavar="DIR",
        help="Wrap each inference call with torch.profiler and write a "
             "tensorboard-format trace to DIR/trace_<timestamp>/. Works for "
             "both --transport in_process and stdio (env-var to server). "
             "Open with `tensorboard --logdir DIR`.",
    )
    parser.add_argument(
        "--cprofile",
        default=None,
        metavar="DIR",
        help="Capture cProfile stats per inference call into DIR/prof_<ts>.prof. "
             "Works for both transports. Inspect with "
             "`python -c \"import pstats; pstats.Stats('prof.prof').sort_stats('cumulative').print_stats(30)\"`.",
    )
    args = parser.parse_args()
    modes = args.modes
    benchmarks = BENCHMARKS_CHRONOS if args.model == "chronos" else BENCHMARKS
    selected_workflows = set(args.workflows) if args.workflows else None
    set_transport(args.transport)

    # Profiler dirs path to server 
    # in-process: same env; stdio: inherited by spawned subprocess via StdioBenchClient.
    # Absolute paths so cwd changes don't relocate output.
    if args.torch_profile:
        os.environ["TSFM_TORCH_PROFILE_DIR"] = str(Path(args.torch_profile).resolve())
        Path(args.torch_profile).resolve().mkdir(parents=True, exist_ok=True)
    if args.cprofile:
        os.environ["TSFM_CPROFILE_DIR"] = str(Path(args.cprofile).resolve())
        Path(args.cprofile).resolve().mkdir(parents=True, exist_ok=True)

    dataset_configs = _build_dataset_configs()

    base_config = {
        "model_checkpoint": "ttm_96_28",
        "forecast_horizon": -1,
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
    print(f"Dataset: {SOURCE_DATASET}")
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
        print(f"Target groups: {sorted(config['target_groups'])}")
        print("~" * 72)

        for mode in modes:
            apply_mode(mode)
            set_seed(config["seed"])
            print("#" * 72)
            print(f"# MODE: {mode}  (transport: {args.transport})")
            print("#" * 72)
            try:
                if args.transport == "stdio":
                    cli = enter_stdio_for_mode(
                        mode,
                        server_cmd=args.server_cmd,
                        server_args=args.server_args,
                    )
                    print(f"  stdio init: {cli.init_ms:.1f}ms")
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
            finally:
                if args.transport == "stdio":
                    exit_stdio()

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

    if len(modes) > 1:
        try:
            log_summary_to_wandb(
                all_rows=all_rows,
                modes=modes,
                config=base_config | (dataset_configs[-1] if dataset_configs else {}),
                transport=args.transport,
            )
        except Exception as exc:
            print(f"  [warn] summary W&B run failed: {exc!r}")

    print("=" * 72)
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
