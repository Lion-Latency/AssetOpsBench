# Chronos - Adds support for Chronos-2 for the Interchangeable Model Interface.
# Reference: https://github.com/amazon-science/chronos-forecasting/blob/main/notebooks/chronos-2-quickstart.ipynb
import numpy as np
import os
import yaml
import pandas as pd

from ..interchangeable_model_interface import InterchangeableModelInterface
from ...metrics import _freq_token_to_minutes

class Chronos(InterchangeableModelInterface):

    # Load the model (defaults to "amazon/chronos-2").
    def load_model(self, model_checkpoint="amazon/chronos-2"):
        import torch
        from chronos import Chronos2Pipeline

        self.model_checkpoint = model_checkpoint
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Chronos2Pipeline.from_pretrained(
            model_checkpoint,
            device_map=device,
            torch_dtype=torch.float32,
        )

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

        finetune_df = finetune_df.dropna(subset=target_columns + conditional_columns)
        finetune_df = finetune_df.sort_values([id_column, timestamp_column]).reset_index(
            drop=True
        )

        if len(finetune_df) > 0:
            max_series_length = int(finetune_df.groupby(id_column, sort=False).size().max())
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

            for _, series_df in finetune_df.groupby(id_column, sort=False):
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
                train_target_values = train_series_df[target_columns].to_numpy(
                    dtype=np.float32, copy=True
                ).T
                if train_target_values.shape[0] == 1:
                    train_target_values = train_target_values[0]

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
                        target_columns
                    ].to_numpy(dtype=np.float32, copy=True).T
                    if validation_target_values.shape[0] == 1:
                        validation_target_values = validation_target_values[0]

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