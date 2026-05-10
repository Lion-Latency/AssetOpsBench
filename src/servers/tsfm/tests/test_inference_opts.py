"""Parity tests for TSFM inference optimization flags.

Covers the env-gated inference opts ported from `sam/AssetOpsBench`:
    TSFM_MODEL_CACHE   (#1) lru_cache the TTM `from_pretrained` load
    TSFM_COMPILE       (#2) torch.compile w/ mode="default"
    TSFM_BF16          (#3) cast cached model to bfloat16 on CUDA
    TSFM_FAST_TRAINER  (#4) bypass HF Trainer w/ direct inference loop

These are disjoint from MainProj's `cache.py` preprocessing-cache opts
(TSFM_CACHE_ENABLED, TSFM_PREPROCESS_OPT, ...) which have separate tests.

Each opt should produce predictions ≈ baseline within tolerance on the
same dataset. BF16 has its own looser tolerance.

Skipped automatically when tsfm_public isn't installed or when
PROFILE_TEST_DATASET / PROFILE_TEST_TARGET aren't set.

Run:
    PATH_TO_MODELS_DIR=/path/to/models \
    PROFILE_TEST_DATASET=/path/to/Chiller_6.smoke.csv \
    PROFILE_TEST_TARGET="Chiller 6 Tonnage" \
    pytest src/servers/tsfm/tests/test_inference_opts.py -v
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytest

from .conftest import requires_tsfm


_OPT_FLAGS = (
    "TSFM_MODEL_CACHE",
    "TSFM_COMPILE",
    "TSFM_BF16",
    "TSFM_FAST_TRAINER",
)


@pytest.fixture(scope="session")
def workload():
    dataset = os.environ.get("PROFILE_TEST_DATASET")
    target = os.environ.get("PROFILE_TEST_TARGET")
    checkpoint = os.environ.get("PROFILE_TEST_CHECKPOINT", "ttm_96_28")
    if not dataset or not target:
        pytest.skip("set PROFILE_TEST_DATASET + PROFILE_TEST_TARGET")
    if not Path(dataset).exists():
        pytest.skip(f"dataset missing: {dataset}")
    return {
        "dataset": str(Path(dataset).resolve()),
        "target_columns": target.split(","),
        "model_checkpoint": checkpoint,
        "timestamp_column": os.environ.get("PROFILE_TEST_TS_COLUMN", "timestamp"),
    }


@pytest.fixture(autouse=True)
def reset_opt_env(monkeypatch):
    for k in _OPT_FLAGS:
        monkeypatch.delenv(k, raising=False)
    # Clear the model lru_cache so prior tests don't leak compiled/bf16
    # variants into a fresh baseline run.
    from servers.tsfm import forecasting as f

    if hasattr(f, "_load_ttm_for_inference_cached"):
        f._load_ttm_for_inference_cached.cache_clear()
    yield


def _run_inference(workload: Dict, opts: Dict[str, str], clear_caches: bool = True) -> Dict:
    """Drive `_get_ttm_hf_inference` once with the requested opt flags set."""
    for k in _OPT_FLAGS:
        os.environ.pop(k, None)
    for k, v in opts.items():
        os.environ[k] = v

    from servers.tsfm import forecasting as f
    from servers.tsfm.io import _get_model_checkpoint_path, _read_ts_data

    if clear_caches:
        f._load_ttm_for_inference_cached.cache_clear()

    model_checkpoint = _get_model_checkpoint_path(workload["model_checkpoint"])
    with open(os.path.join(model_checkpoint, "config.json")) as cfg_f:
        model_config = json.load(cfg_f)

    dataset_config = {
        "column_specifiers": {
            "autoregressive_modeling": True,
            "timestamp_column": workload["timestamp_column"],
            "conditional_columns": [],
            "target_columns": workload["target_columns"],
        },
        "id_columns": [],
        "frequency_sampling": "oov",
    }
    cfg = copy.deepcopy(dataset_config)
    df = _read_ts_data(workload["dataset"], dataset_config_dictionary=cfg)
    dq = f._tsfm_data_quality_filter(df, cfg, model_config, task="inference")

    return f._get_ttm_hf_inference(
        dq["data"],
        dq["dataset_config_dictionary"],
        model_config,
        model_checkpoint,
        forecast_horizon=-1,
    )


def _close(a: np.ndarray, b: np.ndarray, rtol: float, atol: float) -> Tuple[bool, float]:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return False, float("inf")
    diff = np.abs(a - b)
    base = np.abs(b) * rtol + atol
    rel_err = float((diff / np.maximum(base, 1e-12)).max())
    return bool(np.all(diff <= base)), rel_err


@requires_tsfm
def test_baseline_runs(workload):
    out = _run_inference(workload, {})
    arr = np.array(out["target_prediction"])
    assert arr.size > 0
    assert np.isfinite(arr).all()


@requires_tsfm
def test_model_cache_parity(workload):
    base = _run_inference(workload, {})
    cached = _run_inference(workload, {"TSFM_MODEL_CACHE": "1"})
    ok, rel = _close(
        np.array(cached["target_prediction"]),
        np.array(base["target_prediction"]),
        rtol=1e-4, atol=1e-5,
    )
    assert ok, f"model cache parity failed: max rel-err {rel:.3g}"


@requires_tsfm
def test_fast_trainer_parity(workload):
    base = _run_inference(workload, {})
    fast = _run_inference(workload, {"TSFM_MODEL_CACHE": "1", "TSFM_FAST_TRAINER": "1"})
    # Fast path skips Trainer's pre-processing fluff and the redundant
    # second forward pass. Model maths identical.
    ok, rel = _close(
        np.array(fast["target_prediction"]),
        np.array(base["target_prediction"]),
        rtol=1e-3, atol=1e-4,
    )
    assert ok, f"fast trainer parity failed: max rel-err {rel:.3g}"


@requires_tsfm
def test_compile_parity(workload):
    """torch.compile (opt #2) must match baseline within float-reorder noise.

    CUDA-only: Inductor on CPU is fragile and has no perf upside. Compile
    cold-load is slow (minutes) — the autouse fixture clears the lru_cache
    so this test owns the full compile cost.
    """
    import torch

    if not torch.cuda.is_available():
        pytest.skip("compile parity test requires CUDA")
    base = _run_inference(workload, {})
    compiled = _run_inference(workload, {"TSFM_MODEL_CACHE": "1", "TSFM_COMPILE": "1"})
    ok, rel = _close(
        np.array(compiled["target_prediction"]),
        np.array(base["target_prediction"]),
        rtol=1e-3, atol=1e-4,
    )
    assert ok, f"compile parity failed: max rel-err {rel:.3g}"


@requires_tsfm
def test_bf16_within_tolerance(workload):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("bf16 requires CUDA")
    base = _run_inference(workload, {})
    bf = _run_inference(workload, {"TSFM_MODEL_CACHE": "1", "TSFM_BF16": "1"})
    base_arr = np.array(base["target_prediction"])
    bf_arr = np.array(bf["target_prediction"])
    assert base_arr.shape == bf_arr.shape
    # BF16 has ~7 mantissa bits → ~1% relative error is the norm for transformers.
    rel = float(
        np.abs(bf_arr - base_arr).max()
        / max(float(np.abs(base_arr).max()), 1e-12)
    )
    assert rel < 0.05, f"bf16 deviation too high: {rel:.3g}"


@requires_tsfm
def test_model_cache_hits_increment(workload):
    """Sanity: second call with same args = cache hit on the model loader."""
    from servers.tsfm import forecasting as f

    f._load_ttm_for_inference_cached.cache_clear()
    _run_inference(workload, {"TSFM_MODEL_CACHE": "1"})
    info1 = f._load_ttm_for_inference_cached.cache_info()
    # Second call must NOT clear the cache, else the hit can't register.
    _run_inference(workload, {"TSFM_MODEL_CACHE": "1"}, clear_caches=False)
    info2 = f._load_ttm_for_inference_cached.cache_info()
    assert info2.hits > info1.hits, f"expected cache hit; got {info1} → {info2}"
