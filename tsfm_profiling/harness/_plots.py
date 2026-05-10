"""Matplotlib → wandb.Image plot helpers for the bench harness.

"""

from __future__ import annotations

import io
import re
from collections import defaultdict
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # noqa: E402  (must precede pyplot import)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


# Stages we expect to see in a row, in display order. Anything else gets
# appended after these in alphabetical order.
_STAGE_DISPLAY_ORDER = (
    "data_retrieval",
    "data_quality_filter",
    "preprocessing",
    "model_loading",
    "inference",
    "serialization",
)


def _figure_to_image(fig) -> Any:
    """Encode a matplotlib Figure as a wandb.Image.

    """
    import wandb
    from PIL import Image

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return wandb.Image(Image.open(buf))


def _stage_keys(rows: Iterable[Dict[str, Any]]) -> List[str]:
    """Discover stage names from `stage_<name>_ms` columns across rows."""
    pat = re.compile(r"^stage_(.+)_ms$")
    found: set = set()
    for row in rows:
        for k, v in row.items():
            if v is None:
                continue
            m = pat.match(k)
            if m:
                found.add(m.group(1))
    found -= {"total"}  # `stage_total_ms` is the sum, not a real stage.
    ordered = [s for s in _STAGE_DISPLAY_ORDER if s in found]
    ordered += sorted(found - set(ordered))
    return ordered


def stage_breakdown_stacked(
    rows: List[Dict[str, Any]], workflow: str, mode: str
) -> Optional[Any]:
    """Stacked bar: x = target_group, stack = stage timing.

    Reveals where time is spent per asset within a single mode-run.
    """
    successful = [r for r in rows if r.get("status") == "success"]
    if not successful:
        return None
    stages = _stage_keys(successful)
    if not stages:
        return None

    groups = sorted({r.get("target_group", "?") for r in successful})
    # group -> stage -> mean ms across iters
    agg: Dict[str, Dict[str, float]] = {}
    for g in groups:
        agg[g] = {}
        for s in stages:
            vals = [
                r.get(f"stage_{s}_ms")
                for r in successful
                if r.get("target_group", "?") == g and r.get(f"stage_{s}_ms") is not None
            ]
            agg[g][s] = mean(vals) if vals else 0.0

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(groups)), 4.5))
    bottom = np.zeros(len(groups))
    cmap = plt.get_cmap("tab10")
    for i, s in enumerate(stages):
        heights = np.array([agg[g][s] for g in groups])
        ax.bar(groups, heights, bottom=bottom, label=s, color=cmap(i % 10))
        bottom += heights
    ax.set_ylabel("ms (mean across iters)")
    ax.set_xlabel("target group")
    ax.set_title(f"{workflow} stage breakdown — mode={mode}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return _figure_to_image(fig)


def cold_vs_steady_grouped(
    rows: List[Dict[str, Any]], workflow: str, mode: str
) -> Optional[Any]:
    """Grouped bar: x = target_group, two bars (cold | steady) of latency_sec.

    Hides nothing — if a target group only has cold runs (`repeats=1`), the
    steady bar shows zero. Caller should skip when steady-state is empty.
    """
    successful = [r for r in rows if r.get("status") == "success"]
    if not successful:
        return None
    cold = [r for r in successful if r.get("run_type") == "cold_start"]
    steady = [r for r in successful if r.get("run_type") == "steady_state"]
    if not steady:
        return None  # No second-iter data; nothing to compare.

    groups = sorted({r.get("target_group", "?") for r in successful})

    def _avg(rows_, group):
        vs = [r["latency_sec"] for r in rows_ if r.get("target_group", "?") == group]
        return mean(vs) if vs else 0.0

    cold_v = [_avg(cold, g) for g in groups]
    steady_v = [_avg(steady, g) for g in groups]

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(groups)), 4.5))
    x = np.arange(len(groups))
    w = 0.38
    ax.bar(x - w / 2, cold_v, w, label="cold_start", color="#d97706")
    ax.bar(x + w / 2, steady_v, w, label="steady_state", color="#0284c7")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.set_ylabel("latency (s)")
    ax.set_title(f"{workflow} cold vs steady — mode={mode}")
    ax.legend()
    return _figure_to_image(fig)


def cross_mode_latency(
    rows_by_mode: Dict[str, List[Dict[str, Any]]], workflow: str
) -> Optional[Any]:
    """Bar: x = mode, y = mean latency_sec across all groups + iters.

    The headline "did it get faster" chart for the report.
    """
    modes = list(rows_by_mode.keys())
    means: List[float] = []
    for m in modes:
        vals = [r["latency_sec"] for r in rows_by_mode[m] if r.get("status") == "success"]
        means.append(mean(vals) if vals else 0.0)
    if not any(means):
        return None
    fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(modes)), 4.5))
    bars = ax.bar(modes, means, color="#0f766e")
    ax.set_ylabel("mean latency (s)")
    ax.set_title(f"{workflow} latency across modes")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8)
    return _figure_to_image(fig)


def cross_mode_stage_breakdown(
    rows_by_mode: Dict[str, List[Dict[str, Any]]], workflow: str
) -> Optional[Any]:
    """Stacked bar: x = mode, stack = stage timing.

    Lets you see WHICH stage each opt actually compresses.
    """
    all_rows = [r for rows in rows_by_mode.values() for r in rows if r.get("status") == "success"]
    if not all_rows:
        return None
    stages = _stage_keys(all_rows)
    if not stages:
        return None
    modes = list(rows_by_mode.keys())

    # mode -> stage -> mean ms
    agg: Dict[str, Dict[str, float]] = {}
    for m in modes:
        agg[m] = {}
        rows = [r for r in rows_by_mode[m] if r.get("status") == "success"]
        for s in stages:
            vals = [r.get(f"stage_{s}_ms") for r in rows if r.get(f"stage_{s}_ms") is not None]
            agg[m][s] = mean(vals) if vals else 0.0

    fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(modes)), 4.5))
    bottom = np.zeros(len(modes))
    cmap = plt.get_cmap("tab10")
    for i, s in enumerate(stages):
        heights = np.array([agg[m][s] for m in modes])
        ax.bar(modes, heights, bottom=bottom, label=s, color=cmap(i % 10))
        bottom += heights
    ax.set_ylabel("ms (mean across runs)")
    ax.set_title(f"{workflow} stage breakdown across modes")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return _figure_to_image(fig)


def cross_mode_gpu_power(
    rows_by_mode: Dict[str, List[Dict[str, Any]]], workflow: str, stage: str = "inference"
) -> Optional[Any]:
    """Bar: x = mode, y = mean GPU power during the named stage.

    Skipped if NVML samples weren't collected (CPU runs, missing pynvml).
    """
    key = f"stage_{stage}_gpu_power_w_mean"
    modes = list(rows_by_mode.keys())
    means: List[float] = []
    for m in modes:
        vals = [
            r[key] for r in rows_by_mode[m]
            if r.get("status") == "success" and r.get(key) is not None
        ]
        means.append(mean(vals) if vals else 0.0)
    if not any(means):
        return None
    fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(modes)), 4.5))
    bars = ax.bar(modes, means, color="#b91c1c")
    ax.set_ylabel(f"mean GPU power (W) during {stage}")
    ax.set_title(f"{workflow} GPU power across modes — stage={stage}")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.1f}",
                ha="center", va="bottom", fontsize=8)
    return _figure_to_image(fig)


def cross_mode_horizon_perf(
    rows_by_mode: Dict[str, List[Dict[str, Any]]], workflow: str
) -> Dict[str, Tuple[List[int], Dict[str, List[float]]]]:
    """Build per-metric (xs, mode→ys) for cross-mode horizon line plots.

    Returns a dict keyed by metric base (e.g. "rmse_Chiller_6_Tonnage").
    Caller is responsible for converting to wandb.plot.line_series since
    that primitive is not matplotlib-based.
    """
    pat = re.compile(r"^perf_(.+)_h(\d+)$")
    # base -> mode -> horizon -> values
    data: Dict[str, Dict[str, Dict[int, List[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for mode, rows in rows_by_mode.items():
        for r in rows:
            if r.get("status") != "success":
                continue
            for k, v in r.items():
                if v is None or not isinstance(k, str):
                    continue
                m = pat.match(k)
                if not m:
                    continue
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(fv):
                    continue
                base, h = m.group(1), int(m.group(2))
                data[base][mode][h].append(fv)

    out: Dict[str, Tuple[List[int], Dict[str, List[float]]]] = {}
    for base, by_mode in data.items():
        all_h = sorted({h for d in by_mode.values() for h in d})
        mode_series: Dict[str, List[float]] = {}
        for mode, by_h in by_mode.items():
            mode_series[mode] = [
                mean(by_h[h]) if by_h.get(h) else float("nan") for h in all_h
            ]
        out[base] = (all_h, mode_series)
    return out
