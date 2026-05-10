# Chronos - Adds support for Chronos-2 for the Interchangeable Model Interface.
# Reference: https://github.com/amazon-science/chronos-forecasting/blob/main/notebooks/chronos-2-quickstart.ipynb
import numpy as np
import os
import json
import tempfile
import uuid
import yaml
import pandas as pd

from ..interchangeable_model_interface import InterchangeableModelInterface
from ...metrics import _freq_token_to_minutes
from io import UnsupportedOperation

class Chronos(InterchangeableModelInterface):

    # Load the model (defaults to "amazon/chronos-2").
    def load_model(self, model_checkpoint="amazon/chronos-2"):
        import torch
        from chronos import Chronos2Pipeline

        self.model_checkpoint = model_checkpoint
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = torch.float32

        self.model = Chronos2Pipeline.from_pretrained(
            model_checkpoint,
            torch_dtype=torch_dtype,
        )
        self.model.model.to(device)

    # Run zero-shot forecasting on time series data.
    def forecast(self, df_dataframe, column_specifiers, metrics=None):
        from ...profiling import RequestMetrics, stage_timer

        if metrics is None:
            metrics = RequestMetrics(tool="forecast_chronos")

        forecast_horizon = self.prediction_filter_length
        timestamp_column = column_specifiers["timestamp_column"]
        target_columns = column_specifiers["target_columns"]

        # ── Stage: preprocessing ──────────────────────────────────────────────
        with stage_timer("preprocessing", metrics):
            predict_df = df_dataframe.copy()
            id_columns = column_specifiers.get("id_columns")
            if id_columns:
                id_column = id_columns[0]
            else:
                id_column = "item_id"
                if id_column not in predict_df.columns:
                    predict_df[id_column] = "0"

            parsed_timestamps = pd.to_datetime(
                predict_df[timestamp_column], format="ISO8601", utc=True
            )

            frequency_sampling = column_specifiers.get("frequency_sampling")
            if (
                frequency_sampling in _freq_token_to_minutes
                and _freq_token_to_minutes[frequency_sampling] is not None
            ):
                frequency_minutes = _freq_token_to_minutes[frequency_sampling]
            else:
                time_diffs = parsed_timestamps.sort_values().diff().dropna()
                frequency_minutes = float(
                    time_diffs.dt.total_seconds().div(60).median()
                )

            frequency_minutes = max(1, int(round(frequency_minutes)))
            freq = f"{frequency_minutes}min"
            predict_df[timestamp_column] = parsed_timestamps.dt.round(freq)
            predict_df = predict_df.sort_values([id_column, timestamp_column])

        # ── Stage: model_loading ──────────────────────────────────────────────
        with stage_timer("model_loading", metrics):
            self.load_model(self.model_checkpoint)

        # ── Stage: inference ──────────────────────────────────────────────────
        with stage_timer("inference", metrics):
            pred_df = self.model.predict_df(
                predict_df,
                future_df=None,
                prediction_length=forecast_horizon,
                quantile_levels=[0.1, 0.5, 0.9],
                id_column=id_column,
                timestamp_column=timestamp_column,
                target=target_columns,
                validate_inputs=False,
            )

            # Use the median quantile forecast to use as a point forecast output. (This is needed for proper comparison with TTM outputs, which are point forecasts.)
            point_forecast_column = "predictions"
            if "0.5" in pred_df.columns:
                point_forecast_column = "0.5"

            timestamps_list = []
            timestamps_prediction_list = []
            target_prediction_list = []

            for series_id, series_df in predict_df.groupby(id_column, sort=False):
                series_df = series_df.sort_values(timestamp_column)
                series_pred_df = pred_df[pred_df[id_column] == series_id]

                timestamps_list.append(series_df[timestamp_column].iloc[-1])
                timestamps_prediction_list.append(
                    series_pred_df[timestamp_column].drop_duplicates().to_numpy()
                )
                target_prediction_list.append(
                    np.stack(
                        [
                            series_pred_df[
                                series_pred_df["target_name"] == target_column
                            ]
                            .sort_values(timestamp_column)[point_forecast_column]
                            .to_numpy()
                            for target_column in target_columns
                        ],
                        axis=-1,
                    )
                )

            output: dict = {
                "target_columns": target_columns,
                "target_prediction": np.stack(target_prediction_list, axis=0),
                "timestamp": timestamps_list,
                "timestamp_prediction": timestamps_prediction_list,
            }

        return output
    
    # Fine-tune the model on a dataset. (See _get_ttm_hf_inference in forecasting.py for the original implementation.)
    def finetune(
        self,
        df_dataframe,
        column_specifiers,
        n_finetune=0.05,
        n_test=0.05,
        save_model_dir="chronos_finetuned",
        num_steps=1000,
    ):
        from ...profiling import RequestMetrics, stage_timer

        metrics = RequestMetrics(tool="_finetune_chronos")
        forecast_horizon = self.prediction_filter_length

        assert n_finetune > 0, "n_finetune needs to be positive"
        assert n_test >= 0, "n_test needs to be non-negative"

        os.makedirs(save_model_dir, exist_ok=True)

        args_config_dic = {
            "context_length": self.context_length,
            "forecast_horizon": forecast_horizon,
            "batch_size": 32,
            "num_steps": num_steps,
            "learning_rate": 1e-5,
            "logging_steps": 100,
            "finetune_mode": "full",
            "n_finetune": n_finetune,
            "n_test": n_test,
        }

        context_length = self.context_length
        if not context_length and self.model is not None:
            context_length = getattr(self.model, "model_context_length", forecast_horizon)
        if not context_length:
            self.load_model(self.model_checkpoint)
            context_length = getattr(self.model, "model_context_length", forecast_horizon)

        timestamp_column = column_specifiers["timestamp_column"]
        target_columns = column_specifiers["target_columns"]
        conditional_columns = column_specifiers.get("conditional_columns", [])
        id_columns = column_specifiers.get("id_columns") or []
        id_column = id_columns[0] if id_columns else "item_id"
        series_id_column = "__chronos_finetune_series_id"
        target_value_column = "__chronos_target_value"

        finetune_df = df_dataframe.copy()
        if id_column not in finetune_df.columns:
            finetune_df[id_column] = "0"

        required_columns = list(
            dict.fromkeys(
                [id_column, timestamp_column] + target_columns + conditional_columns
            )
        )
        finetune_df = finetune_df[required_columns].copy()
        finetune_df[timestamp_column] = pd.to_datetime(
            finetune_df[timestamp_column], format="ISO8601", utc=True, errors="coerce"
        )
        finetune_df = finetune_df.dropna(subset=[timestamp_column])

        for target_column in target_columns:
            finetune_df[target_column] = pd.to_numeric(
                finetune_df[target_column], errors="coerce"
            )

        # Expand sparse wide target columns into one series per target so sparse
        # asset datasets can still be fine-tuned without requiring dense rows.
        series_frames = []
        for target_column in target_columns:
            series_df = finetune_df[
                [id_column, timestamp_column, target_column] + conditional_columns
            ].copy()
            series_df = series_df.dropna(subset=[target_column] + conditional_columns)
            if series_df.empty:
                continue

            series_df = series_df.rename(columns={target_column: target_value_column})
            series_df[series_id_column] = (
                series_df[id_column].astype("string").fillna("0")
                + "::"
                + target_column
            )
            series_frames.append(series_df)

        if series_frames:
            finetune_df = pd.concat(series_frames, ignore_index=True)
            finetune_df = finetune_df.sort_values(
                [series_id_column, timestamp_column]
            ).reset_index(drop=True)
        else:
            finetune_df = pd.DataFrame(
                columns=[
                    series_id_column,
                    timestamp_column,
                    target_value_column,
                    *conditional_columns,
                ]
            )

        if len(finetune_df) > 0:
            max_series_length = int(
                finetune_df.groupby(series_id_column, sort=False).size().max()
            )
            max_supported_context = max(
                forecast_horizon, max_series_length - forecast_horizon
            )
            context_length = min(context_length, max_supported_context)
            self.context_length = context_length
            args_config_dic["context_length"] = context_length

        n_data = len(finetune_df)
        assert n_data > 0, "dataframe needs to contain rows for fine-tuning"

        p_test = n_test / n_data if n_test >= 1 else n_test
        p_test = float(np.clip(p_test, 0.0, 0.99))
        n_train_total = max(1, int(np.floor((1 - p_test) * n_data)))
        p_finetune = n_finetune / n_train_total if n_finetune > 1 else n_finetune
        p_finetune = float(np.clip(p_finetune, 0.0, 1.0))
        min_series_length = context_length + forecast_horizon

        # ── Stage: preprocessing ──────────────────────────────────────────────
        with stage_timer("preprocessing", metrics):
            train_inputs = []
            validation_inputs = []

            for _, series_df in finetune_df.groupby(series_id_column, sort=False):
                if len(series_df) < min_series_length:
                    continue

                n_series = len(series_df)
                n_test_series = int(np.floor(p_test * n_series))
                train_stop = n_series - n_test_series
                if train_stop < min_series_length:
                    train_stop = n_series
                    n_test_series = 0

                n_finetune_series = int(np.floor(p_finetune * train_stop))
                n_finetune_series = max(min_series_length, n_finetune_series)
                n_finetune_series = min(train_stop, n_finetune_series)
                if n_finetune_series < min_series_length:
                    continue

                train_start = train_stop - n_finetune_series
                train_series_df = series_df.iloc[train_start:train_stop]
                train_target_values = train_series_df[target_value_column].to_numpy(
                    dtype=np.float32, copy=True
                )

                train_series_input = {"target": train_target_values}
                if conditional_columns:
                    train_series_input["past_covariates"] = {
                        column: train_series_df[column].to_numpy(copy=True)
                        for column in conditional_columns
                    }
                train_inputs.append(train_series_input)

                validation_series_df = series_df.iloc[train_stop:]
                if len(validation_series_df) >= min_series_length:
                    validation_target_values = validation_series_df[
                        target_value_column
                    ].to_numpy(dtype=np.float32, copy=True)

                    validation_series_input = {"target": validation_target_values}
                    if conditional_columns:
                        validation_series_input["past_covariates"] = {
                            column: validation_series_df[column].to_numpy(copy=True)
                            for column in conditional_columns
                        }
                    validation_inputs.append(validation_series_input)

            if not train_inputs:
                raise ValueError(
                    "No time series remained after Chronos fine-tuning preprocessing. "
                    "Check the context length, forecast horizon, and missing values."
                )

        # ── Stage: model_loading ──────────────────────────────────────────────
        with stage_timer("model_loading", metrics):
            self.load_model(self.model_checkpoint)

        with open(os.path.join(save_model_dir, "args_config.yml"), "w") as outfile:
            yaml.dump(args_config_dic, outfile)

        # ── Stage: training ──────────────────────────────────────────────────
        with stage_timer("training", metrics):
            finetuned_pipeline = self.model.fit(
                inputs=train_inputs,
                validation_inputs=validation_inputs or None,
                prediction_length=forecast_horizon,
                finetune_mode=args_config_dic["finetune_mode"],
                context_length=context_length,
                learning_rate=args_config_dic["learning_rate"],
                num_steps=args_config_dic["num_steps"],
                batch_size=args_config_dic["batch_size"],
                output_dir=save_model_dir,
                min_past=context_length,
                logging_steps=args_config_dic["logging_steps"],
                disable_data_parallel=True,
            )

        self.model = finetuned_pipeline

        saved_checkpoint_dir = os.path.join(save_model_dir, "finetuned-ckpt")
        if os.path.isdir(saved_checkpoint_dir):
            self.model_checkpoint = saved_checkpoint_dir

        # ── Stage: evaluation ─────────────────────────────────────────────────
        with stage_timer("evaluation", metrics):
            pd_performance = pd.DataFrame(
                columns=["target", "forecast", "metric", "value", "split"]
            )

        train_stage = next(
            (stage for stage in metrics.stages if stage.stage_name == "training"), None
        )
        train_time = train_stage.wall_clock_ms / 1000 if train_stage else 0.0
        performance_columns = ["target", "forecast", "metric", "value", "split", "train_time"]
        pd_performance["train_time"] = train_time

        return {
            "performance": pd_performance[performance_columns],
            "save_model_dir": saved_checkpoint_dir if os.path.isdir(saved_checkpoint_dir) else save_model_dir,
            "experiment_config_path": os.path.join(save_model_dir, "args_config.yml"),
            "train_time": train_time,
        }
    
    # Run anomaly detection.
    def anomaly_detection(
        self,
        dataset_path,
        tsfm_output_json,
        timestamp_column,
        target_columns,
        task="fit",
        false_alarm=0.05,
        ad_model_type="timeseries_conformal_adaptive",
        ad_model_checkpoint=None,
        ad_model_save=None,
        n_calibration=0.2,
        conditional_columns=None,
        id_columns=None,
        frequency_sampling=None,
        autoregressive_modeling=True,
        metrics=None,
    ):
        from ...anomaly import _TimeSeriesAnomalyDetectionConformalWrapper
        from ...main import _build_dataset_config, _emit_metrics, _tsad_output_to_df, logger
        from ...models import ErrorResult, TSADResult
        from ...profiling import RequestMetrics, stage_timer

        if not dataset_path.strip():
            return ErrorResult(error="dataset_path is required")
        if not tsfm_output_json.strip():
            return ErrorResult(error="tsfm_output_json is required")
        if not target_columns:
            return ErrorResult(error="target_columns must not be empty")
        if task not in ("fit", "inference"):
            return ErrorResult(error="task must be 'fit' or 'inference'")

        try:
            import tsfm_public  # noqa: F401
        except ImportError as exc:
            return ErrorResult(error=f"tsfm dependencies unavailable: {exc}")

        created_metrics = metrics is None
        if metrics is None:
            metrics = RequestMetrics(
                tool="run_tsad",
                metadata={
                    "dataset_path": dataset_path,
                    "tsfm_output_json": tsfm_output_json,
                },
            )

        dataset_config = _build_dataset_config(
            timestamp_column,
            target_columns,
            conditional_columns,
            id_columns,
            frequency_sampling or "",
            autoregressive_modeling,
        )

        try:
            # ── Stage: data_retrieval ─────────────────────────────────────────
            with stage_timer("data_retrieval", metrics):
                with open(tsfm_output_json, "r") as fh:
                    tsmodel_pred = json.load(fh)

            # ── Stage: anomaly_detection ──────────────────────────────────────
            with stage_timer("anomaly_detection", metrics):
                prediction_rows = np.array(
                    tsmodel_pred.get("target_prediction", []), dtype=object
                ).shape[0]
                if task == "fit":
                    n_critical = int(np.ceil(1 / false_alarm))
                    if n_calibration is None:
                        minimum_prediction_rows = n_critical
                    elif n_calibration < 1:
                        minimum_prediction_rows = int(
                            np.ceil(n_critical / n_calibration)
                        )
                    else:
                        minimum_prediction_rows = int(n_calibration)
                else:
                    minimum_prediction_rows = 1

                if prediction_rows < max(1, minimum_prediction_rows):
                    tsmodel_pred = self.build_tsad_prediction_dictionary(
                        dataset_path,
                        dataset_config,
                        context_length=self.get_tsad_context_length(
                            ad_model_checkpoint=ad_model_checkpoint
                        ),
                        max_windows=minimum_prediction_rows,
                    )

                output = _TimeSeriesAnomalyDetectionConformalWrapper().run(
                    dataset_path,
                    dataset_config,
                    tsmodel_pred,
                    ad_model_checkpoint=ad_model_checkpoint,
                    ad_model_save=ad_model_save,
                    task=task,
                    ad_model_type=ad_model_type,
                    n_calibration=n_calibration,
                    false_alarm=false_alarm,
                )
        except Exception as exc:
            logger.error("run_tsad failed: %s", exc)
            return ErrorResult(error=str(exc))

        try:
            # ── Stage: serialization ──────────────────────────────────────────
            with stage_timer("serialization", metrics):
                df = _tsad_output_to_df(output)
                tmp_dir = tempfile.mkdtemp()
                csv_path = os.path.join(tmp_dir, f"tsad_output_{uuid.uuid4()}.csv")
                df.to_csv(csv_path, index=False)
                anomaly_count = (
                    int(df["anomaly_label"].sum()) if "anomaly_label" in df.columns else 0
                )
        except Exception as exc:
            logger.error("run_tsad result serialisation failed: %s", exc)
            return ErrorResult(error=f"Failed to serialise TSAD output: {exc}")

        if created_metrics:
            _emit_metrics(metrics)

        return TSADResult(
            status="success",
            results_file=csv_path,
            total_records=len(df),
            anomaly_count=anomaly_count,
            columns=list(df.columns),
            message=(
                f"Anomaly detection complete. {anomaly_count} anomalies in {len(df)} records. "
                f"Results saved to {csv_path}."
            ),
        )

    # Run integrated anomaly detection (forecasting + anomaly detection in one call).
    def integrated_anomaly_detection(
        self,
        dataset_path,
        timestamp_column,
        target_columns,
        model_checkpoint=None,
        false_alarm=0.05,
        ad_model_type="timeseries_conformal_adaptive",
        n_calibration=0.2,
        conditional_columns=None,
        id_columns=None,
        frequency_sampling="",
        autoregressive_modeling=True,
        metrics=None,
    ):
        from ...anomaly import _TimeSeriesAnomalyDetectionConformalWrapper
        from ...forecasting import _tsfm_data_quality_filter
        from ...io import (
            _get_dataset_path,
            _get_model_checkpoint_path,
            _get_outputs_path,
            _read_ts_data,
        )
        from ...main import _build_dataset_config, _emit_metrics, _tsad_output_to_df, logger
        from ...models import ErrorResult, TSADResult
        from ...profiling import RequestMetrics, stage_timer

        if not dataset_path.strip():
            return ErrorResult(error="dataset_path is required")
        if not target_columns:
            return ErrorResult(error="target_columns must not be empty")

        try:
            import chronos  # noqa: F401
            import tsfm_public  # noqa: F401
        except ImportError as exc:
            return ErrorResult(error=f"chronos/tsfm dependencies unavailable: {exc}")

        resolved_model_checkpoint = model_checkpoint or self.model_checkpoint
        candidate_model_checkpoint = _get_model_checkpoint_path(resolved_model_checkpoint)
        if not (os.path.isabs(resolved_model_checkpoint) or os.path.exists(resolved_model_checkpoint)):
            if os.path.exists(candidate_model_checkpoint):
                resolved_model_checkpoint = candidate_model_checkpoint

        created_metrics = metrics is None
        if metrics is None:
            metrics = RequestMetrics(
                tool="run_integrated_tsad_chronos",
                metadata={
                    "dataset_path": dataset_path,
                    "model_checkpoint": resolved_model_checkpoint,
                    "n_target_columns": len(target_columns),
                },
            )

        self.model_checkpoint = resolved_model_checkpoint
        dataset_path = _get_dataset_path(dataset_path)

        try:
            ad_model_save = _get_outputs_path("tsad_model_save/")
            os.makedirs(ad_model_save, exist_ok=True)

            prediction_filter_length = (
                self.prediction_filter_length
                if self.prediction_filter_length and self.prediction_filter_length > 0
                else 1
            )
            chronos_model_config = {
                "context_length": max(3, prediction_filter_length),
                "prediction_length": prediction_filter_length,
            }
            df_combined = pd.DataFrame()

            for col_idx, col in enumerate(target_columns):
                col_config = _build_dataset_config(
                    timestamp_column,
                    [col],
                    conditional_columns,
                    id_columns,
                    frequency_sampling,
                    autoregressive_modeling,
                )

                # 1. Load and quality-filter data for this column
                with stage_timer(f"data_retrieval_col{col_idx}", metrics):
                    data_df = _read_ts_data(
                        dataset_path, dataset_config_dictionary=col_config
                    )
                    selected_columns = list(
                        dict.fromkeys(
                            (id_columns or [])
                            + [timestamp_column]
                            + (conditional_columns or [])
                            + [col]
                        )
                    )
                    data_df = data_df[selected_columns].copy()

                with stage_timer(f"data_quality_filter_col{col_idx}", metrics):
                    output_dq = _tsfm_data_quality_filter(
                        data_df, col_config, chronos_model_config, task="inference"
                    )
                    data_df_filtered = output_dq["data"]
                    col_config_filtered = output_dq["dataset_config_dictionary"]

                if len(data_df_filtered) == 0:
                    logger.warning(
                        "Data quality filter removed all data for column %s; skipping.", col
                    )
                    continue

                # 2. Zero-shot forecasting for this column
                #    Stages: preprocessing, model_loading, inference are inside
                try:
                    column_specifiers = col_config_filtered["column_specifiers"].copy()
                    column_specifiers["id_columns"] = col_config_filtered["id_columns"]
                    column_specifiers["frequency_sampling"] = col_config_filtered[
                        "frequency_sampling"
                    ]
                    forecast_output = self.forecast(
                        data_df_filtered, column_specifiers, metrics=metrics
                    )
                except Exception as exc:
                    logger.warning("Forecasting failed for column %s: %s", col, exc)
                    continue

                inference_data = {
                    "target_prediction": forecast_output["target_prediction"].tolist(),
                    "timestamp": np.array(forecast_output["timestamp_prediction"])
                    .astype(str)
                    .tolist(),
                    "target_columns": forecast_output["target_columns"],
                }
                # 3. Conformal anomaly detection for this column
                tsmodel_pred = inference_data

                try:
                    col_config_for_tsad = {**col_config_filtered}
                    if "id_columns" in col_config_for_tsad:
                        col_config_for_tsad["id_columns"] = [
                            c for c in col_config_for_tsad["id_columns"] if c != "segment_id"
                        ]
                    if "id_columns" in col_config_for_tsad.get("column_specifiers", {}):
                        col_config_for_tsad["column_specifiers"] = {
                            **col_config_for_tsad["column_specifiers"],
                            "id_columns": [
                                c
                                for c in col_config_for_tsad["column_specifiers"]["id_columns"]
                                if c != "segment_id"
                            ],
                        }
                    with stage_timer(f"anomaly_detection_col{col_idx}", metrics):
                        prediction_rows = np.array(
                            tsmodel_pred.get("target_prediction", []), dtype=object
                        ).shape[0]
                        n_critical = int(np.ceil(1 / false_alarm))
                        if n_calibration is None:
                            minimum_prediction_rows = n_critical
                        elif n_calibration < 1:
                            minimum_prediction_rows = int(
                                np.ceil(n_critical / n_calibration)
                            )
                        else:
                            minimum_prediction_rows = int(n_calibration)

                        if prediction_rows < max(1, minimum_prediction_rows):
                            tsmodel_pred = self.build_tsad_prediction_dictionary(
                                dataset_path,
                                col_config_for_tsad,
                                context_length=self.get_tsad_context_length(),
                                max_windows=minimum_prediction_rows,
                            )

                        tsad_output = _TimeSeriesAnomalyDetectionConformalWrapper().run(
                            dataset_path,
                            col_config_for_tsad,
                            tsmodel_pred,
                            ad_model_checkpoint=None,
                            ad_model_save=ad_model_save,
                            task="fit",
                            ad_model_type=ad_model_type,
                            n_calibration=n_calibration,
                            false_alarm=false_alarm,
                        )
                except Exception as exc:
                    logger.warning("TSAD failed for column %s: %s", col, exc)
                    continue

                df_col = _tsad_output_to_df(tsad_output)
                df_combined = pd.concat([df_combined, df_col], axis=0, ignore_index=True)

            if df_combined.empty:
                return ErrorResult(error="No TSAD results produced for any target column.")

            # ── Stage: serialization ──────────────────────────────────────────
            with stage_timer("serialization", metrics):
                tmp_dir = tempfile.mkdtemp()
                csv_path = os.path.join(tmp_dir, f"integrated_tsad_{uuid.uuid4()}.csv")
                df_combined.to_csv(csv_path, index=False)
                anomaly_count = (
                    int(df_combined["anomaly_label"].sum())
                    if "anomaly_label" in df_combined.columns
                    else 0
                )

        except Exception as exc:
            logger.error("run_integrated_tsad_chronos failed: %s", exc)
            return ErrorResult(error=str(exc))

        if created_metrics:
            _emit_metrics(metrics)

        return TSADResult(
            status="success",
            results_file=csv_path,
            total_records=len(df_combined),
            anomaly_count=anomaly_count,
            columns=list(df_combined.columns),
            message=(
                f"Integrated TSAD complete. {anomaly_count} anomalies in {len(df_combined)} records "
                f"across {len(target_columns)} column(s). Results saved to {csv_path}."
            ),
        )

    def get_tsad_context_length(self, ad_model_checkpoint=None):
        if not ad_model_checkpoint:
            return 1

        config_path = os.path.join(ad_model_checkpoint, "config.json")
        if not os.path.exists(config_path):
            return 1

        with open(config_path, "r") as fh:
            return max(1, int(json.load(fh).get("context_length", 1)))

    def build_tsad_prediction_dictionary(self, dataset_path, dataset_config, context_length, max_windows=None):
        from ...io import _read_ts_data

        column_specifiers = dataset_config["column_specifiers"]
        timestamp_column = column_specifiers["timestamp_column"]
        target_columns = column_specifiers["target_columns"]
        id_columns = dataset_config.get("id_columns") or []

        data_df = _read_ts_data(dataset_path, dataset_config_dictionary=dataset_config)
        selected_columns = list(
            dict.fromkeys(id_columns + [timestamp_column] + target_columns)
        )
        predict_df = data_df[selected_columns].copy()

        if id_columns:
            source_id_column = id_columns[0]
        else:
            source_id_column = "item_id"
            if source_id_column not in predict_df.columns:
                predict_df[source_id_column] = "0"

        parsed_timestamps = pd.to_datetime(
            predict_df[timestamp_column], format="ISO8601", utc=True
        )

        frequency_sampling = dataset_config.get("frequency_sampling")
        if (
            frequency_sampling in _freq_token_to_minutes
            and _freq_token_to_minutes[frequency_sampling] is not None
        ):
            frequency_minutes = _freq_token_to_minutes[frequency_sampling]
        else:
            time_diffs = parsed_timestamps.sort_values().diff().dropna()
            frequency_minutes = float(
                time_diffs.dt.total_seconds().div(60).median()
            )

        frequency_minutes = max(1, int(round(frequency_minutes)))
        freq = f"{frequency_minutes}min"
        predict_df[timestamp_column] = parsed_timestamps.dt.round(freq)
        predict_df = predict_df.sort_values([source_id_column, timestamp_column])
        # Drop duplicate timestamps per series that can arise from frequency rounding;
        # pd.infer_freq returns None on windows with duplicate timestamps.
        predict_df = predict_df.drop_duplicates(
            subset=[source_id_column, timestamp_column], keep="last"
        ).reset_index(drop=True)

        if self.model is None:
            self.load_model(self.model_checkpoint)

        history_length = max(3, int(context_length))
        synthetic_id_column = "__chronos_tsad_window_id"
        rolling_frames = []
        rolling_window_ids = []

        for series_id, series_df in predict_df.groupby(source_id_column, sort=False):
            series_df = series_df.sort_values(timestamp_column).reset_index(drop=True)
            candidate_ends = range(history_length, len(series_df))
            if max_windows is not None and len(candidate_ends) > max_windows:
                # Evenly stride through the series instead of using every timestamp.
                step = len(candidate_ends) // max_windows
                candidate_ends = candidate_ends[::step][:max_windows]
            for end_idx in candidate_ends:
                window_id = f"{series_id}::{end_idx}"
                start_idx = max(0, end_idx - history_length)
                window_df = series_df.iloc[start_idx:end_idx].copy()
                window_df[synthetic_id_column] = window_id
                rolling_frames.append(
                    window_df[[synthetic_id_column, timestamp_column] + target_columns]
                )
                rolling_window_ids.append(window_id)

        if not rolling_frames:
            raise ValueError(
                "No rolling Chronos windows were available for anomaly detection."
            )

        chunk_size = int(os.environ.get("CHRONOS_TSAD_WINDOW_CHUNK", "128"))
        chunk_size = max(1, chunk_size)
        pred_chunks = []
        for chunk_start in range(0, len(rolling_frames), chunk_size):
            chunk_frames = rolling_frames[chunk_start : chunk_start + chunk_size]
            chunk_df = pd.concat(chunk_frames, ignore_index=True)
            pred_chunks.append(
                self.model.predict_df(
                    chunk_df,
                    future_df=None,
                    prediction_length=1,
                    quantile_levels=[0.1, 0.5, 0.9],
                    id_column=synthetic_id_column,
                    timestamp_column=timestamp_column,
                    target=target_columns,
                    validate_inputs=False,
                )
            )
        pred_df = pd.concat(pred_chunks, ignore_index=True)

        point_forecast_column = "predictions"
        if "0.5" in pred_df.columns:
            point_forecast_column = "0.5"

        timestamps_prediction_list = []
        target_prediction_list = []

        for window_id in rolling_window_ids:
            window_pred_df = pred_df[pred_df[synthetic_id_column] == window_id]
            timestamps_prediction_list.append(
                window_pred_df[timestamp_column].drop_duplicates().to_numpy()
            )
            target_prediction_list.append(
                np.stack(
                    [
                        window_pred_df[
                            window_pred_df["target_name"] == target_column
                        ]
                        .sort_values(timestamp_column)[point_forecast_column]
                        .to_numpy()
                        for target_column in target_columns
                    ],
                    axis=-1,
                )
            )

        return {
            "target_columns": target_columns,
            "target_prediction": np.stack(target_prediction_list, axis=0).tolist(),
            "timestamp": np.array(timestamps_prediction_list).astype(str).tolist(),
        }