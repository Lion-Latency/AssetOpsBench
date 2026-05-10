"""Per-stage profiling instrumentation for the TSFM MCP server.

Provides a lightweight context manager (`stage_timer`) and a request-scoped
metrics collector (`RequestMetrics`) that together capture wall-clock time,
GPU memory, and CPU/RSS usage at each pipeline stage. When `nvidia-ml-py`
is installed, also samples GPU power, utilization, and clock speeds via a
background thread.

Usage in instrumented code:

    from .profiling import RequestMetrics, stage_timer

    metrics = RequestMetrics(tool="run_tsfm_forecasting")

    with stage_timer("data_retrieval", metrics):
        data_df = _read_ts_data(...)

    with stage_timer("preprocessing", metrics):
        output_dq = _tsfm_data_quality_filter(...)

    # ... etc.

    report = metrics.finalize()   # dict ready for JSON / W&B logging
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("tsfm-mcp-server.profiling")


# ── NVML helper (sampled GPU power / util / clocks) ──────────────────────────


_NVML_INITIALIZED = False
_NVML_HANDLE: Any = None
_NVML_MODULE: Any = None


def _try_init_nvml() -> bool:
    """Initialize NVML once per process. Returns True if available."""
    global _NVML_INITIALIZED, _NVML_HANDLE, _NVML_MODULE
    if _NVML_INITIALIZED:
        return _NVML_HANDLE is not None
    _NVML_INITIALIZED = True
    try:
        import pynvml  # nvidia-ml-py exposes the `pynvml` import
        pynvml.nvmlInit()
        # Single-GPU assumption matches the rest of the harness.
        _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
        _NVML_MODULE = pynvml
        return True
    except Exception as exc:
        logger.debug("NVML unavailable; skipping GPU power/util sampling: %s", exc)
        _NVML_HANDLE = None
        _NVML_MODULE = None
        return False


class _NvmlSampler:
    """Background-thread NVML sampler.

    Polls power/util/clocks at `interval_s` cadence into in-memory lists.
    `start()` spins a daemon thread; `stop()` joins it and returns the
    aggregated dict. No-op when NVML isn't available — `aggregates()`
    returns an empty dict so callers don't need to branch.
    """

    __slots__ = (
        "interval_s", "_stop", "_thread",
        "_power_w", "_util_pct", "_mem_util_pct",
        "_sm_clock_mhz", "_mem_clock_mhz",
    )

    def __init__(self, interval_s: float = 0.1):
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._power_w: List[float] = []
        self._util_pct: List[int] = []
        self._mem_util_pct: List[int] = []
        self._sm_clock_mhz: List[int] = []
        self._mem_clock_mhz: List[int] = []

    def start(self) -> "_NvmlSampler":
        if _NVML_HANDLE is None:
            return self
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        nv = _NVML_MODULE
        h = _NVML_HANDLE
        if nv is None or h is None:
            return
        sm_clk = nv.NVML_CLOCK_SM
        mem_clk = nv.NVML_CLOCK_MEM
        get_power = nv.nvmlDeviceGetPowerUsage
        get_util = nv.nvmlDeviceGetUtilizationRates
        get_clock = nv.nvmlDeviceGetClockInfo
        wait = self._stop.wait
        while not wait(self.interval_s):
            try:
                self._power_w.append(get_power(h) / 1000.0)
                u = get_util(h)
                self._util_pct.append(u.gpu)
                self._mem_util_pct.append(u.memory)
                self._sm_clock_mhz.append(get_clock(h, sm_clk))
                self._mem_clock_mhz.append(get_clock(h, mem_clk))
            except Exception:
                # Don't let a single transient NVML hiccup poison the stage.
                continue

    def stop(self) -> "_NvmlSampler":
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        return self

    def aggregates(self) -> Dict[str, float]:
        if not self._power_w:
            return {}
        return {
            "gpu_power_w_mean": sum(self._power_w) / len(self._power_w),
            "gpu_power_w_max": max(self._power_w),
            "gpu_util_pct_mean": sum(self._util_pct) / len(self._util_pct),
            "gpu_util_pct_max": max(self._util_pct),
            "gpu_mem_util_pct_mean": sum(self._mem_util_pct) / len(self._mem_util_pct),
            "sm_clock_mhz_mean": sum(self._sm_clock_mhz) / len(self._sm_clock_mhz),
            "mem_clock_mhz_mean": sum(self._mem_clock_mhz) / len(self._mem_clock_mhz),
            "nvml_samples": len(self._power_w),
        }


# ── GPU helpers (safe when CUDA is unavailable) ──────────────────────────────


def _gpu_memory_mb() -> Optional[float]:
    """Return current GPU memory allocated in MB, or None if CUDA unavailable."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
    except ImportError:
        pass
    return None


def _gpu_max_memory_mb() -> Optional[float]:
    """Return peak GPU memory allocated in MB, or None if CUDA unavailable."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except ImportError:
        pass
    return None


def _reset_gpu_peak_memory() -> None:
    """Reset the peak GPU memory tracker so each stage gets its own peak."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


# ── CPU / RSS helper ─────────────────────────────────────────────────────────


def _rss_memory_mb() -> Optional[float]:
    """Return current process RSS in MB, or None if psutil unavailable."""
    try:
        import psutil

        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        pass
    return None


# ── Stage measurement record ─────────────────────────────────────────────────


@dataclass
class StageMeasurement:
    """Timing and resource snapshot for a single pipeline stage."""

    stage_name: str
    wall_clock_ms: float
    gpu_memory_before_mb: Optional[float] = None
    gpu_memory_after_mb: Optional[float] = None
    gpu_memory_peak_mb: Optional[float] = None
    rss_before_mb: Optional[float] = None
    rss_after_mb: Optional[float] = None
    nvml_aggregates: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "stage": self.stage_name,
            "wall_clock_ms": round(self.wall_clock_ms, 3),
        }
        if self.gpu_memory_before_mb is not None:
            d["gpu_mem_before_mb"] = round(self.gpu_memory_before_mb, 2)
            d["gpu_mem_after_mb"] = round(self.gpu_memory_after_mb or 0.0, 2)
            d["gpu_mem_delta_mb"] = round(
                (self.gpu_memory_after_mb or 0.0) - self.gpu_memory_before_mb, 2
            )
            d["gpu_mem_peak_mb"] = round(self.gpu_memory_peak_mb or 0.0, 2)
        if self.rss_before_mb is not None:
            d["rss_before_mb"] = round(self.rss_before_mb, 2)
            d["rss_after_mb"] = round(self.rss_after_mb or 0.0, 2)
            d["rss_delta_mb"] = round(
                (self.rss_after_mb or 0.0) - self.rss_before_mb, 2
            )
        for k, v in self.nvml_aggregates.items():
            d[k] = round(v, 3) if isinstance(v, float) else v
        return d


# ── Request-scoped metrics collector ─────────────────────────────────────────


@dataclass
class RequestMetrics:
    """Collects per-stage measurements for a single MCP tool invocation.

    Args:
        tool: Name of the MCP tool (e.g. "run_tsfm_forecasting").
        metadata: Optional dict of extra context (dataset, checkpoint, etc.).
    """

    tool: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    stages: List[StageMeasurement] = field(default_factory=list)
    _start_time: float = field(default=0.0, repr=False)

    def __post_init__(self):
        self._start_time = time.perf_counter()

    def add(self, measurement: StageMeasurement) -> None:
        self.stages.append(measurement)
        logger.debug(
            "Stage %-20s  %8.1f ms", measurement.stage_name, measurement.wall_clock_ms
        )

    def finalize(self) -> Dict[str, Any]:
        """Return a complete metrics report as a JSON-serializable dict."""
        end_to_end_ms = (time.perf_counter() - self._start_time) * 1000
        stage_total_ms = sum(s.wall_clock_ms for s in self.stages)

        return {
            "tool": self.tool,
            "metadata": self.metadata,
            "end_to_end_ms": round(end_to_end_ms, 3),
            "stage_total_ms": round(stage_total_ms, 3),
            "overhead_ms": round(end_to_end_ms - stage_total_ms, 3),
            "stages": [s.to_dict() for s in self.stages],
        }


# ── Context manager for timing a single stage ────────────────────────────────


@contextmanager
def stage_timer(stage_name: str, metrics: RequestMetrics):
    """Context manager that measures one pipeline stage.

    Records wall-clock time, GPU memory (before/after/peak), and CPU RSS
    (before/after), then appends a StageMeasurement to the metrics collector.

    Usage:
        with stage_timer("model_loading", metrics):
            model = TinyTimeMixerForPrediction.from_pretrained(...)
    """
    # --- entry snapshot ---
    gpu_before = _gpu_memory_mb()
    rss_before = _rss_memory_mb()
    _reset_gpu_peak_memory()
    sampler: Optional[_NvmlSampler] = None
    if _try_init_nvml():
        sampler = _NvmlSampler(interval_s=NVML_SAMPLE_INTERVAL_S).start()
    t0 = time.perf_counter()

    yield  # run the stage code

    # --- exit snapshot ---
    t1 = time.perf_counter()
    nvml_aggs: Dict[str, float] = {}
    if sampler is not None:
        sampler.stop()
        nvml_aggs = sampler.aggregates()
    gpu_after = _gpu_memory_mb()
    gpu_peak = _gpu_max_memory_mb()
    rss_after = _rss_memory_mb()

    measurement = StageMeasurement(
        stage_name=stage_name,
        wall_clock_ms=(t1 - t0) * 1000,
        gpu_memory_before_mb=gpu_before,
        gpu_memory_after_mb=gpu_after,
        gpu_memory_peak_mb=gpu_peak,
        rss_before_mb=rss_before,
        rss_after_mb=rss_after,
        nvml_aggregates=nvml_aggs,
    )
    metrics.add(measurement)


# ── Global toggles ────────────────────────────────────────────────────────────

# Set to False to disable all profiling with near-zero overhead.
# When disabled, stage_timer becomes a no-op passthrough.
PROFILING_ENABLED = True

# NVML sampling cadence in seconds. 100ms = 10 samples/sec, ~negligible
# overhead on a daemon thread. Stages shorter than this get 0-1 samples
# and report empty aggregates.
NVML_SAMPLE_INTERVAL_S = 0.1

_original_stage_timer = stage_timer


@contextmanager
def _noop_stage_timer(stage_name: str, metrics: RequestMetrics):
    yield


def set_profiling_enabled(enabled: bool) -> None:
    """Enable or disable profiling globally at runtime."""
    global PROFILING_ENABLED, stage_timer
    PROFILING_ENABLED = enabled
    if enabled:
        stage_timer = _original_stage_timer
    else:
        stage_timer = _noop_stage_timer
