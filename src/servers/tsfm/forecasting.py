"""TSFM inference, fine-tuning, and the data quality filter bridge.

Heavy ML dependencies (tsfm_public, transformers, torch) are imported lazily
so the module can be imported even when they are absent.

Inference optimization flags (env vars, default off → existing behavior):
    TSFM_MODEL_CACHE   "1" → lru_cache the TTM model load (opt #1)
    TSFM_COMPILE       "1" → wrap cached model w/ torch.compile (opt #2)
    TSFM_BF16          "1" → cast cached model to bfloat16 on CUDA (opt #3)
    TSFM_FAST_TRAINER  "1" → bypass HF Trainer w/ direct inference loop (opt #4)

Disjoint from the preprocessing/parallelism flags handled by `cache.py` and
`parallel.py` (TSFM_CACHE_ENABLED, TSFM_PREPROCESS_OPT, …). Finetuning path
is intentionally untouched by these inference flags.
"""

from __future__ import annotations

import math
import os
import pickle
import time
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from . import cache as _cache
from . import parallel as _parallel
from .dataquality import (
    _df_dt_stats,
    _df_nan_stats,
    _dq_timeseries_segmentation,
    _time_series_segment_quality_summary,
)
from .io import _make_json_compatible
from .metrics import _METRICS_FORECAST, _TSFREQUENCY_TOLERANCE, _freq_token_to_minutes


# ── Inference optimization flag readers ──────────────────────────────────────


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) == "1"


def _opt_model_cache() -> bool:
    return _flag("TSFM_MODEL_CACHE")


def _opt_compile() -> bool:
    return _flag("TSFM_COMPILE")


def _opt_bf16() -> bool:
    return _flag("TSFM_BF16")


def _opt_fast_trainer() -> bool:
    return _flag("TSFM_FAST_TRAINER")


# ── Cached model loader (opts #1 / #2 / #3) ──────────────────────────────────


@lru_cache(maxsize=4)
def _load_ttm_for_inference_cached(
    model_checkpoint: str,
    prediction_filter_length: int,
    bf16: bool,
    compile_mode: Optional[str],
    device: str,
):
    """Load a TTM model once per (checkpoint, horizon, dtype, compile, device).

    Cached output is treated as a read-only template; callers must not call
    Trainer training utilities on the returned instance. The fast-trainer
    path (opt #4) only reads.
    """
    import torch
    from tsfm_public import TinyTimeMixerForPrediction

    model = TinyTimeMixerForPrediction.from_pretrained(
        model_checkpoint, prediction_filter_length=prediction_filter_length
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if bf16 and torch.cuda.is_available():
        model = model.to(dtype=torch.bfloat16)
    if device == "cuda" and torch.cuda.is_available():
        model = model.to("cuda")
    if compile_mode is not None and torch.cuda.is_available():
        # mode="default": Inductor fusions only, no CUDA Graphs.
        # "reduce-overhead" captures graphs that reuse output buffers across
        # batches; HF Trainer concatenates per-batch outputs on GPU and trips
        # "accessing tensor output of CUDAGraphs that has been overwritten".
        model = torch.compile(model, mode=compile_mode, fullgraph=False)
    return model


def _resolve_inference_model(model_checkpoint: str, prediction_filter_length: int):
    """Return a TTM model honoring opt flags. No flags → fresh load (legacy)."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    any_opt = _opt_model_cache() or _opt_compile() or _opt_bf16()

    if not any_opt:
        from tsfm_public import TinyTimeMixerForPrediction

        model = TinyTimeMixerForPrediction.from_pretrained(
            model_checkpoint, prediction_filter_length=prediction_filter_length
        )
        if device == "cuda":
            model = model.to("cuda")
        return model

    compile_mode = "default" if _opt_compile() else None
    return _load_ttm_for_inference_cached(
        model_checkpoint=model_checkpoint,
        prediction_filter_length=prediction_filter_length,
        bf16=_opt_bf16(),
        compile_mode=compile_mode,
        device=device,
    )


def _bf16_autocast_ctx():
    """Autocast bf16 on CUDA when opt #3 is on; harmless no-op otherwise.

    Trainer's eval loop feeds fp32 input batches; the cached model has been
    cast to bf16 by `_load_ttm_for_inference_cached`, so without autocast
    the first F.linear call raises a dtype-mismatch error.
    """
    import contextlib

    import torch

    if _opt_bf16() and torch.cuda.is_available():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


# ── Shared collator (Trainer path + fast-trainer path) ───────────────────────


def _collate_tensor_only(batch):
    """Stack tensor-valued keys; drop non-tensor keys (id, timestamp, …).

    Required for the Trainer path when `remove_unused_columns=False`: the
    default collator would trip on string/datetime columns and TTM forward
    would receive surprise kwargs.
    """
    import torch

    out = {}
    for k in batch[0].keys():
        vals = [b[k] for b in batch]
        if all(isinstance(v, torch.Tensor) for v in vals):
            out[k] = torch.stack(vals)
    return out


# ── Opt #4: fast inference loop (bypass HF Trainer) ──────────────────────────


def _predict_fast(model, dataset, batch_size: int = 64) -> np.ndarray:
    """Single forward pass over `dataset` w/o constructing an HF Trainer.

    Returns predictions stacked as shape `[N, prediction_length, num_features]`,
    matching `Trainer.predict(...).predictions[0]`.
    """
    import torch
    from torch.utils.data import DataLoader

    eff_device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dtype = next(
        (p.dtype for p in model.parameters() if p.is_floating_point()),
        torch.float32,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(eff_device == "cuda"),
        collate_fn=_collate_tensor_only,
    )
    preds = []
    with torch.inference_mode():
        for batch in loader:
            for k, v in list(batch.items()):
                v = v.to(eff_device, non_blocking=(eff_device == "cuda"))
                if v.is_floating_point():
                    v = v.to(model_dtype)
                batch[k] = v
            out = model(**batch)
            tensor = getattr(out, "prediction_outputs", None)
            if tensor is None and isinstance(out, (tuple, list)):
                tensor = out[0]
            preds.append(tensor.detach().to(torch.float32).cpu())
    return torch.cat(preds, dim=0).numpy()


def _gt_pred_from_predictions(
    predictions_first,
    dataset,
    ix_target_features,
    inverse_transforms=None,
):
    """Mirror of `_get_gt_and_predictions` but reusing an already-computed
    output array. Avoids the redundant `trainer.predict` second pass that
    the Trainer-based path performs.
    """
    if inverse_transforms is None:
        inverse_transforms = []
    target_value_list = []
    pred_value_list = []
    timestamp_id_value_dic: dict = {}
    for i in range(len(dataset)):
        future_vals = dataset[i]["future_values"]
        if hasattr(future_vals, "detach"):
            future_vals = future_vals.detach().cpu().numpy()
        aux = future_vals[:, ix_target_features]
        if "timestamp" in dataset[i]:
            timestamp_id_value_dic.setdefault("timestamp", []).append(dataset[i]["timestamp"])
        if "id" in dataset[i]:
            timestamp_id_value_dic.setdefault("id", []).extend(list(dataset[i]["id"]))
        target_value_list.append(aux)
        forecast_h = aux.shape[0]
        aux_pred = predictions_first[i, :forecast_h, ix_target_features].transpose()
        pred_value_list.append(aux_pred)
    y_gt = np.array(target_value_list)
    y_pred = np.array(pred_value_list)
    for ix_fhorizon in range(y_gt.shape[1]):
        if inverse_transforms:
            y_gt[:, ix_fhorizon, :] = inverse_transforms[0](y_gt[:, ix_fhorizon, :])
            y_pred[:, ix_fhorizon, :] = inverse_transforms[0](y_pred[:, ix_fhorizon, :])
    return y_gt, y_pred, timestamp_id_value_dic


# ── TSFM data quality filter ──────────────────────────────────────────────────


def _tsfm_data_quality_filter(
    df_dataframe, dataset_config_dictionary, model_config, task="inference"
):
    timestamp_col = dataset_config_dictionary["column_specifiers"]["timestamp_column"]
    data_col = [timestamp_col]
    for columns_group in dataset_config_dictionary["column_specifiers"]:
        if "_columns" in columns_group:
            data_col.extend(
                dataset_config_dictionary["column_specifiers"][columns_group]
            )
    if "operation_on_column" in dataset_config_dictionary:
        data_col.extend(dataset_config_dictionary["operation_on_column"])

    df = df_dataframe[data_col].copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], format="mixed")
    for col in data_col:
        if col != timestamp_col:
            df[col] = df[col].astype(float)

    time_intervals_dic = _df_dt_stats(df, date_col=timestamp_col, intervals_dic=None)
    nans_dic = _df_nan_stats(df, perc_rows_less_than=[], perc_rows_more_than=[])

    FILTERING_PARAMS: dict = {"nans": {"efficient_removal": {"preference_tie": "row"}}}

    frequency_minutes = None
    if "frequency_sampling" in dataset_config_dictionary:
        freq_str = dataset_config_dictionary["frequency_sampling"]
        if freq_str:
            assert freq_str in _freq_token_to_minutes, (
                f" frequency_sampling input does not belong to {list(_freq_token_to_minutes.keys())}, "
                "select 'oov' to estimate it from the timestamps"
            )
            frequency_minutes = _freq_token_to_minutes[freq_str]

    if frequency_minutes is None:
        timestamps = pd.to_datetime(df[timestamp_col], format="mixed", utc=True)
        time_diffs = timestamps.diff().dropna()
        frequency_minutes = float(time_diffs.dt.total_seconds().div(60).median())

    freq_lower = frequency_minutes - _TSFREQUENCY_TOLERANCE * frequency_minutes
    freq_upper = frequency_minutes + _TSFREQUENCY_TOLERANCE * frequency_minutes
    FILTERING_PARAMS["dt"] = {"lower_bound": freq_lower, "upper_bound": freq_upper}

    df = _dq_timeseries_segmentation(
        df, filtering_params=FILTERING_PARAMS, timestamp_tag=timestamp_col
    )

    dataset_config = dataset_config_dictionary.copy()
    dataset_config["id_columns"] = ["segment_id"]
    for col_tag in dataset_config["column_specifiers"]:
        if col_tag not in ("timestamp_column", "autoregressive_modeling"):
            dataset_config["column_specifiers"][col_tag] = [
                c
                for c in dataset_config["column_specifiers"][col_tag]
                if c in df.columns
            ]

    n_minimum = 1
    if task == "inference":
        n_minimum = model_config["context_length"]
    if task == "finetuning":
        n_minimum = model_config["prediction_length"] + model_config["context_length"]

    group_sizes = df.groupby(dataset_config["id_columns"][0]).size()
    large_groups = group_sizes[group_sizes >= n_minimum].index
    df = df[df[dataset_config["id_columns"][0]].isin(large_groups)]

    ts_segments_quality_summary = _time_series_segment_quality_summary(
        df, timestamp_col, dataset_config["id_columns"][0]
    )
    ts_segments_quality_summary["removed_columns"] = [
        c for c in data_col if c not in df.columns
    ]
    ts_segments_quality_summary["frequency_sampling_min"] = frequency_minutes

    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    return {
        "data": df,
        "dataset_config_dictionary": dataset_config,
        "dataquality_summary": _make_json_compatible(
            {
                "original_data": {
                    "nans_summary": nans_dic,
                    "sampling_summary": time_intervals_dic,
                },
                "filtered_data_ts_segments": ts_segments_quality_summary,
            }
        ),
    }


# ── Inference helpers ─────────────────────────────────────────────────────────


def _get_gt_and_predictions(
    trainer, dataset, ix_target_features, inverse_transforms=None
):
    if inverse_transforms is None:
        inverse_transforms = []
    with _bf16_autocast_ctx():
        outputs = trainer.predict(dataset)
    target_value_list = []
    pred_value_list = []
    timestamp_id_value_dic: dict = {}
    for i in range(len(dataset)):
        aux = dataset[i]["future_values"][:, ix_target_features].detach().numpy()
        if "timestamp" in dataset[i]:
            timestamp_id_value_dic.setdefault("timestamp", []).append(
                dataset[i]["timestamp"]
            )
        if "id" in dataset[i]:
            timestamp_id_value_dic.setdefault("id", []).extend(list(dataset[i]["id"]))
        target_value_list.append(aux)
        forecast_h = aux.shape[0]
        aux_pred = outputs.predictions[0][
            i, :forecast_h, ix_target_features
        ].transpose()
        pred_value_list.append(aux_pred)
    y_gt = np.array(target_value_list)
    y_pred = np.array(pred_value_list)
    for ix_fhorizon in range(y_gt.shape[1]):
        if inverse_transforms:
            y_gt[:, ix_fhorizon, :] = inverse_transforms[0](y_gt[:, ix_fhorizon, :])
            y_pred[:, ix_fhorizon, :] = inverse_transforms[0](y_pred[:, ix_fhorizon, :])
    return y_gt, y_pred, timestamp_id_value_dic


def _get_performance(
    y_gt,
    y_pred,
    target_columns=None,
    prediction=True,
    inverse_transforms=None,
    ts_mask=None,
):
    if inverse_transforms is None:
        inverse_transforms = []
    if ts_mask is None:
        ts_mask = np.ones([y_gt.shape[0], y_gt.shape[1]])
    if not target_columns:
        target_columns = list(np.arange(y_gt.shape[2]))
    rows = []
    pd_prediction = pd.DataFrame()
    pd_performance = pd.DataFrame()
    for ix_target in range(y_gt.shape[2]):
        for ix_fhorizon in range(y_gt.shape[1]):
            if len(inverse_transforms) > ix_target:
                y_gt[:, ix_fhorizon, ix_target] = inverse_transforms[ix_target](
                    y_gt[:, ix_fhorizon, ix_target][:, np.newaxis]
                )[:, 0]
                y_pred[:, ix_fhorizon, ix_target] = inverse_transforms[ix_target](
                    y_pred[:, ix_fhorizon, ix_target][:, np.newaxis]
                )[:, 0]
            pd_aux = pd.DataFrame(
                {
                    "y_gt": y_gt[:, ix_fhorizon, ix_target],
                    "y_pred": y_pred[:, ix_fhorizon, ix_target],
                    "forecast_horizon": ix_fhorizon + 1,
                    "target": target_columns[ix_target],
                    "on_mask": ts_mask[:, ix_fhorizon],
                }
            )
            pd_prediction = pd.concat([pd_prediction, pd_aux], axis=0)
            y_gt_mask = y_gt[:, ix_fhorizon, ix_target][ts_mask[:, ix_fhorizon] > 0]
            y_pred_mask = y_pred[:, ix_fhorizon, ix_target][ts_mask[:, ix_fhorizon] > 0]
            valid_mask = np.isfinite(y_gt_mask) & np.isfinite(y_pred_mask)
            y_gt_mask = y_gt_mask[valid_mask]
            y_pred_mask = y_pred_mask[valid_mask]
            if y_gt_mask.shape[0] > 0:
                for metric in _METRICS_FORECAST:
                    value = _METRICS_FORECAST[metric](
                        y_gt[:, :ix_fhorizon, ix_target],
                        y_pred[:, :ix_fhorizon, ix_target],
                        axis=1,
                    )
                    stat = np.mean(value) if value is not None else None
                    rows.append(
                        [target_columns[ix_target], ix_fhorizon + 1, metric, stat]
                    )
    if rows:
        pd_performance = pd.DataFrame(
            data=rows, columns=["target", "forecast", "metric", "value"]
        )
    if prediction:
        return pd_performance, pd_prediction
    return pd_performance


def _get_ttm_hf_inference(
    df_dataframe,
    dataset_config_dictionary,
    model_config,
    model_checkpoint,
    scaling=False,
    tsp=None,
    forecast_horizon=-1,
    metrics=None,
):
    from tsfm_public.toolkit.time_series_preprocessor import (
        TimeSeriesPreprocessor,
        get_datasets,
        create_timestamps,
    )
    from transformers import Trainer, TrainingArguments

    from .profiling import RequestMetrics, stage_timer

    # If no metrics collector was passed, create a throwaway one so the
    # stage_timer calls still work without branching everywhere.
    if metrics is None:
        metrics = RequestMetrics(tool="_get_ttm_hf_inference")

    if forecast_horizon == -1:
        forecast_horizon = model_config["prediction_length"]
    else:
        assert forecast_horizon <= model_config["prediction_length"], (
            f" Selected forecast horizon is above what is supported by the model. "
            f"Set a forecast horizon smaller than {model_config['prediction_length']}"
        )
    context_length = model_config["context_length"]
    assert context_length <= len(df_dataframe), (
        " length of dataframe needs to be larger or equal to context length"
    )

    column_specifiers = dataset_config_dictionary["column_specifiers"]
    if (
        "id_columns" in dataset_config_dictionary
        and "id_columns" not in column_specifiers
    ):
        column_specifiers["id_columns"] = dataset_config_dictionary["id_columns"]

    # ── Stage: preprocessing ──────────────────────────────────────────────
    with stage_timer("preprocessing", metrics):
        encode_categorical = False

        prep_key = _cache.make_key(
            _parallel.mode_tag(), "prep_inference",
            _cache.df_fingerprint(df_dataframe),
            column_specifiers, scaling, encode_categorical,
            forecast_horizon, context_length,
        )
        cached_prep = _cache.get(prep_key)
        if cached_prep is not None:
            tsp, dataset_inference = cached_prep
            metrics.metadata["prep_cache_hit"] = True
        else:
            metrics.metadata["prep_cache_hit"] = False
            tsp = TimeSeriesPreprocessor(
                **column_specifiers,
                scaling=scaling,
                encode_categorical=encode_categorical,
                prediction_length=forecast_horizon,
                context_length=context_length,
            )
            dataset_dic = get_datasets(
                tsp,
                df_dataframe,
                split_config={"train": 1.0, "test": 0.0},
                use_frequency_token=True,
            )
            dataset_inference = dataset_dic[0]
            _cache.put(prep_key, (tsp, dataset_inference))

    # ── Stage: model_loading ──────────────────────────────────────────────
    with stage_timer("model_loading", metrics):
        # Opts #1/#2/#3: env-flag dispatched (cache, compile, bf16). With all
        # flags off this is a fresh `TinyTimeMixerForPrediction.from_pretrained`
        # equivalent to the legacy path.
        model = _resolve_inference_model(model_checkpoint, forecast_horizon)

        if not _opt_fast_trainer():
            # remove_unused_columns=False + tensor-only collator + label_names:
            # OptimizedModule (torch.compile, opt #2) wraps the model in a
            # forward signature of (*args, **kwargs). Trainer's column-pruning
            # would otherwise strip every TTM input key (past_values,
            # future_values, freq_token, …), producing an empty batch and
            # ValueError at predict time. Same opaque signature also defeats
            # label detection, so loss leaks into predictions[0]; naming
            # `future_values` as a label key restores the expected shape.
            args = TrainingArguments(
                output_dir="./output",
                logging_dir="./log",
                report_to="none",  # avoid wandb login conflict w harness
                remove_unused_columns=False,
                label_names=["future_values"],
            )
            trainer = Trainer(
                model=model,
                args=args,
                eval_dataset=dataset_inference,
                data_collator=_collate_tensor_only,
            )
        else:
            trainer = None

    # ── Stage: inference ──────────────────────────────────────────────────
    with stage_timer("inference", metrics):
        ix_target_features = list(
            np.arange(len(dataset_config_dictionary["column_specifiers"]["target_columns"]))
        )

        if _opt_fast_trainer():
            # Opt #4: bypass HF Trainer; one forward pass shared by predictions + perf.
            raw_pred = _predict_fast(model, dataset_inference)
            outputs_predictions_first = raw_pred
            y_pred = raw_pred[:, :forecast_horizon, ix_target_features]
        else:
            with _bf16_autocast_ctx():
                outputs = trainer.predict(dataset_inference)
            outputs_predictions_first = outputs.predictions[0]
            y_pred = outputs_predictions_first[:, :forecast_horizon, ix_target_features]

        if tsp.scaling:
            for ixf in range(y_pred.shape[1]):
                y_pred[:, ixf, :] = tsp.target_scaler_dict["0"].inverse_transform(
                    y_pred[:, ixf, :]
                )

    # ── Post-inference (timestamps, performance) — not a timed stage ─────
    timestamps_list = []
    timestamps_prediction_list = []
    for i in range(len(dataset_inference)):
        if "timestamp" in dataset_inference[i]:
            timestamps_list.append(dataset_inference[i]["timestamp"])
            timestamp_forecast = create_timestamps(
                last_timestamp=dataset_inference[i]["timestamp"],
                time_sequence=df_dataframe[
                    column_specifiers["timestamp_column"]
                ].values,
                periods=forecast_horizon,
            )
            timestamps_prediction_list.append(timestamp_forecast)

    output: dict = {
        "target_columns": dataset_config_dictionary["column_specifiers"][
            "target_columns"
        ],
        "target_prediction": y_pred,
        "timestamp": timestamps_list,
        "timestamp_prediction": timestamps_prediction_list,
    }

    inverse_transforms = []
    if scaling:
        inverse_transforms.append(tsp.target_scaler_dict["0"].inverse_transform)

    if _opt_fast_trainer():
        # Reuse the single forward pass from `_predict_fast` — no second
        # `trainer.predict` call (the legacy path runs forward twice).
        y_gt, y_pred_eval, timestamp_id_value_dic = _gt_pred_from_predictions(
            outputs_predictions_first,
            dataset_inference,
            ix_target_features=ix_target_features,
            inverse_transforms=inverse_transforms,
        )
    else:
        y_gt, y_pred_eval, timestamp_id_value_dic = _get_gt_and_predictions(
            trainer,
            dataset_inference,
            ix_target_features=ix_target_features,
            inverse_transforms=inverse_transforms,
        )
    target_columns = dataset_config_dictionary["column_specifiers"]["target_columns"]
    pd_performance = _get_performance(
        y_gt, y_pred_eval, target_columns=target_columns, prediction=False
    )
    output["performance"] = pd_performance

    return output


# ── Fine-tuning ───────────────────────────────────────────────────────────────

_DEFAULT_TRAINING_ARGUMENTS = {
    "overwrite_output_dir": True,
    "learning_rate": 0.0001,
    "num_train_epochs": 10,
    "do_eval": True,
    "evaluation_strategy": "epoch", 
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 32,
    "save_strategy": "epoch",
    "logging_strategy": "epoch",
    "save_total_limit": 3,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
}


def _ttm_main_config():
    return {
        "scaling": "",
        "p_validation": 0.1,
        "encode_categorical": False,
        "context_length": 512,
        "patch_length": 64,
        "forecast_horizon": 96,
        "batch_size": 32,
        "num_workers": 4,
        "seed": 42,
        "model_type": "ttm",
        "optim": "AdamW",
        "lr": 0.0,
        "epochs": 4,
        "scheduler": "OneCycleLR",
        "epochs_warmup": 5,
        "es_patience": 15.0,
        "es_th": 0.0001,
        "backbone_frozen": False,
        "decoder_mode": "mix_channel",
        "head_dropout": 0.7,
    }


def _finetune_ttm_hf(
    df_dataframe,
    dataset_config_dictionary,
    model_config,
    save_model_dir,
    n_finetune,
    n_calibration,
    n_test,
    model_checkpoint="",
    training_config_dic=None,
    metrics=None,
):
    from tsfm_public import (
        TinyTimeMixerConfig,
        TinyTimeMixerForPrediction,
        TrackingCallback,
    )
    from tsfm_public.toolkit.lr_finder import optimal_lr_finder
    from tsfm_public.toolkit.time_series_preprocessor import (
        TimeSeriesPreprocessor,
        get_datasets,
    )
    from tsfm_public.toolkit.util import select_by_index
    from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, set_seed

    from .profiling import RequestMetrics, stage_timer

    if metrics is None:
        metrics = RequestMetrics(tool="_finetune_ttm_hf")

    if training_config_dic is None:
        args_config_dic = _ttm_main_config()
    else:
        args_config_dic = training_config_dic.copy()
        default_config = _ttm_main_config()
        for k in default_config:
            if k not in args_config_dic:
                args_config_dic[k] = default_config[k]

    seed = args_config_dic["seed"]
    set_seed(seed)
    encode_categorical = args_config_dic["encode_categorical"]
    scaling_type = args_config_dic["scaling"]
    p_validation = args_config_dic["p_validation"]

    forecast_horizon = model_config["prediction_length"]
    context_length = model_config["context_length"]
    args_config_dic["forecast_horizon"] = forecast_horizon
    args_config_dic["context_length"] = context_length

    assert context_length <= len(df_dataframe), (
        " length of dataframe needs to be >= context length"
    )

    column_specifiers = dataset_config_dictionary["column_specifiers"]
    ix_target_features = list(np.arange(len(column_specifiers["target_columns"])))

    if (
        "id_columns" in dataset_config_dictionary
        and "id_columns" not in column_specifiers
    ):
        column_specifiers["id_columns"] = dataset_config_dictionary["id_columns"]

    n_data = len(df_dataframe)
    assert n_test >= 0
    p_test = n_test / n_data if n_test >= 1 else n_test
    n_train_total = int(np.floor((1 - p_test) * n_data))

    assert n_finetune > 0
    p_finetune = n_finetune / n_train_total if n_finetune > 1 else n_finetune
    n_validation = np.ceil(p_finetune * n_train_total * p_validation)
    p_train = (n_train_total - n_validation) / n_data
    n_train_effective = p_finetune * n_train_total - n_validation
    fewshot_fraction = n_train_effective / (n_train_total - n_validation)

    scaling = scaling_type == "standard"

    # ── Stage: preprocessing ──────────────────────────────────────────────
    with stage_timer("preprocessing", metrics):
        tsp = TimeSeriesPreprocessor(
            **column_specifiers,
            scaling=scaling,
            encode_categorical=encode_categorical,
            prediction_length=forecast_horizon,
            context_length=context_length,
        )
        dataset_dic = get_datasets(
            tsp,
            df_dataframe,
            split_config={"train": p_train, "test": p_test},
            use_frequency_token=True,
            fewshot_fraction=fewshot_fraction,
        )
        train_dataset = dataset_dic[0]
        valid_dataset = dataset_dic[1]
        test_dataset = dataset_dic[2]

    with open(os.path.join(save_model_dir, "args_config.yml"), "w") as outfile:
        yaml.dump(args_config_dic, outfile)
    with open(os.path.join(save_model_dir, "tsp.pickle"), "wb") as _f:
        pickle.dump(tsp, _f)

    # ── Stage: model_loading ──────────────────────────────────────────────
    with stage_timer("model_loading", metrics):
        if os.path.exists(model_checkpoint):
            finetune_forecast_model = TinyTimeMixerForPrediction.from_pretrained(
                model_checkpoint,
                head_dropout=args_config_dic["head_dropout"],
                num_input_channels=tsp.num_input_channels,
                exogenous_channel_indices=tsp.exogenous_channel_indices,
                prediction_channel_indices=tsp.prediction_channel_indices,
                decoder_mode=args_config_dic["decoder_mode"],
                enable_forecast_channel_mixing=False,
                fcm_use_mixer=False,
                ignore_mismatched_sizes=True,
                prediction_filter_length=forecast_horizon,
            )
        else:
            config_ttm_dic = model_config.copy()
            config_ttm_dic.update(
                {
                    "head_dropout": args_config_dic["head_dropout"],
                    "prediction_length": forecast_horizon,
                    "num_input_channels": tsp.num_input_channels,
                    "exogenous_channel_indices": tsp.exogenous_channel_indices,
                    "prediction_channel_indices": tsp.prediction_channel_indices,
                    "enable_forecast_channel_mixing": False,
                    "fcm_use_mixer": False,
                    "decoder_mode": args_config_dic["decoder_mode"],
                }
            )
            config = TinyTimeMixerConfig(**config_ttm_dic)
            finetune_forecast_model = TinyTimeMixerForPrediction(config)

    if args_config_dic["backbone_frozen"]:
        for param in finetune_forecast_model.backbone.parameters():
            param.requires_grad = False

    batch_size = args_config_dic["batch_size"]
    epochs = args_config_dic["epochs"]
    num_workers = args_config_dic["num_workers"]
    epochs_warmup = args_config_dic["epochs_warmup"]
    es_patience = args_config_dic["es_patience"]
    es_th = args_config_dic["es_th"]
    optim = args_config_dic["optim"]
    scheduler = args_config_dic["scheduler"]
    lr = args_config_dic["lr"]

    # Use a fresh copy of the defaults to avoid cross-call mutation
    training_config_dictionary = _DEFAULT_TRAINING_ARGUMENTS.copy()

    output_fewshot_dir = save_model_dir + "/fewshot/"
    logging_dir = save_model_dir + "/log/"
    os.makedirs(output_fewshot_dir, exist_ok=True)
    os.makedirs(logging_dir, exist_ok=True)

    training_config_dictionary.update(
        {
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "num_train_epochs": epochs,
            "learning_rate": lr,
            "output_dir": output_fewshot_dir,
            "logging_dir": logging_dir,
            "dataloader_num_workers": num_workers,
            "report_to": "none", # report_to var set to avoid wandb login conflict w harness
        }
    )
    if epochs_warmup > 0:
        training_config_dictionary["warmup_steps"] = math.ceil(
            epochs_warmup * len(train_dataset) / batch_size
        )
    with open(os.path.join(save_model_dir, "training_config.yml"), "w") as outfile:
        yaml.dump(training_config_dictionary, outfile)

    finetune_forecast_args = TrainingArguments(**training_config_dictionary)

    if n_finetune > 0:
        if lr <= 0:
            try:
                lr, finetune_forecast_model = optimal_lr_finder(
                    finetune_forecast_model, train_dataset, batch_size=batch_size
                )
                if lr <= 0:
                    lr = 0.0001
            except Exception:
                lr = 0.0001
    else:
        lr = 0.0001

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=es_patience,
        early_stopping_threshold=es_th,
    )

    optimizer = None
    if optim == "AdamW":
        from torch.optim import AdamW

        optimizer = AdamW(finetune_forecast_model.parameters(), lr=lr)

    scheduler_object = None
    if scheduler == "cosine_with_warmup":
        if optimizer is None:
            from torch.optim import AdamW

            optimizer = AdamW(finetune_forecast_model.parameters(), lr=lr)
        from transformers.optimization import get_cosine_schedule_with_warmup

        total_steps = math.ceil(len(train_dataset) * epochs / batch_size)
        num_warmup_steps = math.ceil(epochs_warmup * len(train_dataset) / batch_size)
        scheduler_object = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
        )
    if scheduler == "OneCycleLR":
        if optimizer is None:
            from torch.optim import AdamW

            optimizer = AdamW(finetune_forecast_model.parameters(), lr=lr)
        from torch.optim.lr_scheduler import OneCycleLR

        scheduler_object = OneCycleLR(
            optimizer,
            lr,
            epochs=epochs,
            steps_per_epoch=math.ceil(len(train_dataset) / batch_size),
        )

    tracking_callback = TrackingCallback()
    finetune_forecast_trainer = Trainer(
        model=finetune_forecast_model,
        args=finetune_forecast_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        callbacks=[early_stopping_callback, tracking_callback],
        optimizers=(optimizer, scheduler_object),
    )

    # ── Stage: training ──────────────────────────────────────────────────
    with stage_timer("training", metrics):
        if n_finetune > 0:
            finetune_forecast_trainer.train()

    # ── Stage: evaluation ─────────────────────────────────────────────────
    with stage_timer("evaluation", metrics):
        dataset_eval: dict = {}
        if n_finetune > 0:
            dataset_eval["train"] = train_dataset
            dataset_eval["valid"] = valid_dataset
        if n_test >= 1:
            dataset_eval["test"] = test_dataset

        pd_performance = pd.DataFrame()
        for dataset_key in dataset_eval:
            inverse_transforms_eval = []
            if scaling:
                inverse_transforms_eval.append(
                    tsp.target_scaler_dict["0"].inverse_transform
                )
            y_gt, y_pred_eval, _ = _get_gt_and_predictions(
                finetune_forecast_trainer,
                dataset_eval[dataset_key],
                ix_target_features=ix_target_features,
                inverse_transforms=inverse_transforms_eval,
            )
            target_columns = dataset_config_dictionary["column_specifiers"][
                "target_columns"
            ]
            pd_performance_i = _get_performance(
                y_gt, y_pred_eval, target_columns=target_columns, prediction=False
            )
            pd_performance_i["split"] = dataset_key
            pd_performance = pd.concat([pd_performance, pd_performance_i], axis=0)

    # Preserve train_time in output for backward compatibility
    train_stage = next(
        (s for s in metrics.stages if s.stage_name == "training"), None
    )
    train_time = train_stage.wall_clock_ms / 1000 if train_stage else 0.0
    pd_performance["train_time"] = train_time
    return {
        "performance": pd_performance,
        "save_model_dir": save_model_dir,
        "experiment_config_path": os.path.join(save_model_dir, "args_config.yml"),
    }


def _find_largest_tsfm_checkpoint_directory(root_dir: str) -> str:
    largest_checkpoint_dir = None
    largest_number = float("-inf")
    for f in os.listdir(root_dir):
        if "checkpoint" in f:
            number = int(f.split("-")[-1])
            if number > largest_number:
                largest_number = number
                largest_checkpoint_dir = os.path.join(root_dir, f)
    return largest_checkpoint_dir
