# Chronos - Adds support for Chronos-2 for the Interchangeable Model Interface.
# Reference: https://github.com/amazon-science/chronos-forecasting/blob/main/notebooks/chronos-2-quickstart.ipynb
import numpy as np
import os
import yaml
import pickle
import math
import pandas as pd

from ..interchangeable_model_interface import InterchangeableModelInterface
from ...metrics import _METRICS_FORECAST, _TSFREQUENCY_TOLERANCE, _freq_token_to_minutes

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
    def forecast(self, df_dataframe, column_specifiers):
        forecast_horizon = self.prediction_filter_length
        timestamp_column = column_specifiers["timestamp_column"]
        target_columns = column_specifiers["target_columns"]

        if self.model is None:
            self.load_model(self.model_checkpoint)

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
            frequency_minutes = float(time_diffs.dt.total_seconds().div(60).median())

        frequency_minutes = max(1, int(round(frequency_minutes)))
        freq = f"{frequency_minutes}min"
        predict_df[timestamp_column] = parsed_timestamps.dt.round(freq)
        predict_df = predict_df.sort_values([id_column, timestamp_column])

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
    def finetune(self, df_dataframe, column_specifiers, n_finetune=0.05, n_test=0.05, save_model_dir="chronos_finetuned"):
        return None
        # from tsfm_public import (
        # TinyTimeMixerConfig,
        # TinyTimeMixerForPrediction,
        # TrackingCallback,
        # )
        # from tsfm_public.toolkit.lr_finder import optimal_lr_finder
        # from tsfm_public.toolkit.time_series_preprocessor import (
        #     TimeSeriesPreprocessor,
        #     get_datasets,
        # )
        # from tsfm_public.toolkit.util import select_by_index
        # from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, set_seed

        # from ...profiling import RequestMetrics, stage_timer

        # metrics = RequestMetrics(tool="_finetune_ttm_hf")
        # forecast_horizon = self.prediction_filter_length
        # context_length = self.context_length

        # assert context_length <= len(df_dataframe), (
        #     " length of dataframe needs to be >= context length"
        # )

        # dataset_config_dictionary = {"column_specifiers": column_specifiers}
        # column_specifiers = dataset_config_dictionary["column_specifiers"]
        # ix_target_features = list(np.arange(len(column_specifiers["target_columns"])))

        # args_config_dic = {
        #     "scaling": "",
        #     "p_validation": 0.1,
        #     "encode_categorical": False,
        #     "context_length": 512,
        #     "patch_length": 64,
        #     "forecast_horizon": 96,
        #     "batch_size": 32,
        #     "num_workers": 4,
        #     "seed": 42,
        #     "model_type": "ttm",
        #     "optim": "AdamW",
        #     "lr": 0.0,
        #     "epochs": 4,
        #     "scheduler": "OneCycleLR",
        #     "epochs_warmup": 5,
        #     "es_patience": 15.0,
        #     "es_th": 0.0001,
        #     "backbone_frozen": False,
        #     "decoder_mode": "mix_channel",
        #     "head_dropout": 0.7,
        # }
        
        # # Override with forecast_horizon
        # args_config_dic["forecast_horizon"] = forecast_horizon
        # args_config_dic["context_length"] = self.context_length
        
        # seed = args_config_dic["seed"]
        # set_seed(seed)
        # encode_categorical = args_config_dic["encode_categorical"]
        # scaling_type = args_config_dic["scaling"]
        # p_validation = args_config_dic["p_validation"]

        # if (
        #     "id_columns" in dataset_config_dictionary
        #     and "id_columns" not in column_specifiers
        # ):
        #     column_specifiers["id_columns"] = dataset_config_dictionary["id_columns"]

        # n_data = len(df_dataframe)
        # assert n_test >= 0
        # p_test = n_test / n_data if n_test >= 1 else n_test
        # n_train_total = int(np.floor((1 - p_test) * n_data))

        # assert n_finetune > 0
        # p_finetune = n_finetune / n_train_total if n_finetune > 1 else n_finetune
        # n_validation = np.ceil(p_finetune * n_train_total * p_validation)
        # p_train = (n_train_total - n_validation) / n_data
        # n_train_effective = p_finetune * n_train_total - n_validation
        # fewshot_fraction = n_train_effective / (n_train_total - n_validation)

        # scaling = "standard"

        # # ── Stage: preprocessing ──────────────────────────────────────────────
        # with stage_timer("preprocessing", metrics):
        #     tsp = TimeSeriesPreprocessor(
        #         **column_specifiers,
        #         scaling=scaling,
        #         encode_categorical=None,
        #         prediction_length=forecast_horizon,
        #         context_length=context_length,
        #     )
        #     dataset_dic = get_datasets(
        #         tsp,
        #         df_dataframe,
        #         split_config={"train": p_train, "test": p_test},
        #         use_frequency_token=True,
        #         fewshot_fraction=fewshot_fraction,
        #     )
        #     train_dataset = dataset_dic[0]
        #     valid_dataset = dataset_dic[1]
        #     test_dataset = dataset_dic[2]

        # with open(os.path.join(save_model_dir, "args_config.yml"), "w") as outfile:
        #     yaml.dump(args_config_dic, outfile)
        # with open(os.path.join(save_model_dir, "tsp.pickle"), "wb") as _f:
        #     pickle.dump(tsp, _f)

        # # ── Stage: model_loading ──────────────────────────────────────────────
        # with stage_timer("model_loading", metrics):
        #     if os.path.exists(self.model_checkpoint):
        #         finetune_forecast_model = TinyTimeMixerForPrediction.from_pretrained(
        #             self.model_checkpoint,
        #             head_dropout=args_config_dic["head_dropout"],
        #             num_input_channels=tsp.num_input_channels,
        #             exogenous_channel_indices=tsp.exogenous_channel_indices,
        #             prediction_channel_indices=tsp.prediction_channel_indices,
        #             decoder_mode=args_config_dic["decoder_mode"],
        #             enable_forecast_channel_mixing=False,
        #             fcm_use_mixer=False,
        #             ignore_mismatched_sizes=True,
        #             prediction_filter_length=forecast_horizon,
        #         )
        #     else:
        #         model_config = {
        #             "context_length": self.context_length,
        #             "prediction_length": forecast_horizon,
        #         }
        #         config_ttm_dic = model_config.copy()
        #         config_ttm_dic.update(
        #             {
        #                 "head_dropout": args_config_dic["head_dropout"],
        #                 "prediction_length": forecast_horizon,
        #                 "num_input_channels": tsp.num_input_channels,
        #                 "exogenous_channel_indices": tsp.exogenous_channel_indices,
        #                 "prediction_channel_indices": tsp.prediction_channel_indices,
        #                 "enable_forecast_channel_mixing": False,
        #                 "fcm_use_mixer": False,
        #                 "decoder_mode": args_config_dic["decoder_mode"],
        #             }
        #         )
        #         config = TinyTimeMixerConfig(**config_ttm_dic)
        #         finetune_forecast_model = TinyTimeMixerForPrediction(config)

        # if args_config_dic["backbone_frozen"]:
        #     for param in finetune_forecast_model.backbone.parameters():
        #         param.requires_grad = False

        # batch_size = args_config_dic["batch_size"]
        # epochs = args_config_dic["epochs"]
        # num_workers = args_config_dic["num_workers"]
        # epochs_warmup = args_config_dic["epochs_warmup"]
        # es_patience = args_config_dic["es_patience"]
        # es_th = args_config_dic["es_th"]
        # optim = args_config_dic["optim"]
        # scheduler = args_config_dic["scheduler"]
        # lr = args_config_dic["lr"]

        # # Use a fresh copy of the defaults to avoid cross-call mutation
        # _DEFAULT_TRAINING_ARGUMENTS = {
        #     "overwrite_output_dir": True,
        #     "learning_rate": 0.0001,
        #     "num_train_epochs": 10,
        #     "do_eval": True,
        #     "evaluation_strategy": "epoch",
        #     "per_device_train_batch_size": 32,
        #     "per_device_eval_batch_size": 32,
        #     "save_strategy": "epoch",
        #     "logging_strategy": "epoch",
        #     "save_total_limit": 3,
        #     "load_best_model_at_end": True,
        #     "metric_for_best_model": "eval_loss",
        #     "greater_is_better": False,
        # }

        # training_config_dictionary = _DEFAULT_TRAINING_ARGUMENTS.copy()

        # output_fewshot_dir = save_model_dir + "/fewshot/"
        # logging_dir = save_model_dir + "/log/"
        # os.makedirs(output_fewshot_dir, exist_ok=True)
        # os.makedirs(logging_dir, exist_ok=True)

        # training_config_dictionary.update(
        #     {
        #         "per_device_train_batch_size": batch_size,
        #         "per_device_eval_batch_size": batch_size,
        #         "num_train_epochs": epochs,
        #         "learning_rate": lr,
        #         "output_dir": output_fewshot_dir,
        #         "logging_dir": logging_dir,
        #         "dataloader_num_workers": num_workers,
        #     }
        # )
        # if epochs_warmup > 0:
        #     training_config_dictionary["warmup_steps"] = math.ceil(
        #         epochs_warmup * len(train_dataset) / batch_size
        #     )
        # with open(os.path.join(save_model_dir, "training_config.yml"), "w") as outfile:
        #     yaml.dump(training_config_dictionary, outfile)

        # finetune_forecast_args = TrainingArguments(**training_config_dictionary)

        # if n_finetune > 0:
        #     if lr <= 0:
        #         try:
        #             lr, finetune_forecast_model = optimal_lr_finder(
        #                 finetune_forecast_model, train_dataset, batch_size=batch_size
        #             )
        #             if lr <= 0:
        #                 lr = 0.0001
        #         except Exception:
        #             lr = 0.0001
        # else:
        #     lr = 0.0001

        # early_stopping_callback = EarlyStoppingCallback(
        #     early_stopping_patience=es_patience,
        #     early_stopping_threshold=es_th,
        # )

        # optimizer = None
        # if optim == "AdamW":
        #     from torch.optim import AdamW

        #     optimizer = AdamW(finetune_forecast_model.parameters(), lr=lr)

        # scheduler_object = None
        # if scheduler == "cosine_with_warmup":
        #     if optimizer is None:
        #         from torch.optim import AdamW

        #         optimizer = AdamW(finetune_forecast_model.parameters(), lr=lr)
        #     from transformers.optimization import get_cosine_schedule_with_warmup

        #     total_steps = math.ceil(len(train_dataset) * epochs / batch_size)
        #     num_warmup_steps = math.ceil(epochs_warmup * len(train_dataset) / batch_size)
        #     scheduler_object = get_cosine_schedule_with_warmup(
        #         optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
        #     )
        # if scheduler == "OneCycleLR":
        #     if optimizer is None:
        #         from torch.optim import AdamW

        #         optimizer = AdamW(finetune_forecast_model.parameters(), lr=lr)
        #     from torch.optim.lr_scheduler import OneCycleLR

        #     scheduler_object = OneCycleLR(
        #         optimizer,
        #         lr,
        #         epochs=epochs,
        #         steps_per_epoch=math.ceil(len(train_dataset) / batch_size),
        #     )

        # tracking_callback = TrackingCallback()
        # finetune_forecast_trainer = Trainer(
        #     model=finetune_forecast_model,
        #     args=finetune_forecast_args,
        #     train_dataset=train_dataset,
        #     eval_dataset=valid_dataset,
        #     callbacks=[early_stopping_callback, tracking_callback],
        #     optimizers=(optimizer, scheduler_object),
        # )

        # # ── Stage: training ──────────────────────────────────────────────────
        # with stage_timer("training", metrics):
        #     if n_finetune > 0:
        #         finetune_forecast_trainer.train()

        # # ── Stage: evaluation ─────────────────────────────────────────────────
        # with stage_timer("evaluation", metrics):
        #     dataset_eval: dict = {}
        #     if n_finetune > 0:
        #         dataset_eval["train"] = train_dataset
        #         dataset_eval["valid"] = valid_dataset
        #     if n_test >= 1:
        #         dataset_eval["test"] = test_dataset

        #     pd_performance = pd.DataFrame()
        #     for dataset_key in dataset_eval:
        #         inverse_transforms_eval = []
        #         if scaling:
        #             inverse_transforms_eval.append(
        #                 tsp.target_scaler_dict["0"].inverse_transform
        #             )

        #         # Copied _get_gt_and_predictions in for access to function.
        #         def _get_gt_and_predictions(
        #             trainer, dataset, ix_target_features, inverse_transforms=None
        #         ):
        #             if inverse_transforms is None:
        #                 inverse_transforms = []
        #             outputs = trainer.predict(dataset)
        #             target_value_list = []
        #             pred_value_list = []
        #             timestamp_id_value_dic: dict = {}
        #             for i in range(len(dataset)):
        #                 aux = dataset[i]["future_values"][:, ix_target_features].detach().numpy()
        #                 if "timestamp" in dataset[i]:
        #                     timestamp_id_value_dic.setdefault("timestamp", []).append(
        #                         dataset[i]["timestamp"]
        #                     )
        #                 if "id" in dataset[i]:
        #                     timestamp_id_value_dic.setdefault("id", []).extend(list(dataset[i]["id"]))
        #                 target_value_list.append(aux)
        #                 forecast_h = aux.shape[0]
        #                 aux_pred = outputs.predictions[0][
        #                     i, :forecast_h, ix_target_features
        #                 ].transpose()
        #                 pred_value_list.append(aux_pred)
        #             y_gt = np.array(target_value_list)
        #             y_pred = np.array(pred_value_list)
        #             for ix_fhorizon in range(y_gt.shape[1]):
        #                 if inverse_transforms:
        #                     y_gt[:, ix_fhorizon, :] = inverse_transforms[0](y_gt[:, ix_fhorizon, :])
        #                     y_pred[:, ix_fhorizon, :] = inverse_transforms[0](y_pred[:, ix_fhorizon, :])
        #             return y_gt, y_pred, timestamp_id_value_dic

        #         y_gt, y_pred_eval, _ = _get_gt_and_predictions(
        #             finetune_forecast_trainer,
        #             dataset_eval[dataset_key],
        #             ix_target_features=ix_target_features,
        #             inverse_transforms=inverse_transforms_eval,
        #         )
        #         target_columns = dataset_config_dictionary["column_specifiers"][
        #             "target_columns"
        #         ]

        #         # Copied _get_performance in for access to function.
        #         def _get_performance(
        #             y_gt,
        #             y_pred,
        #             target_columns=None,
        #             prediction=True,
        #             inverse_transforms=None,
        #             ts_mask=None,
        #         ):
        #             if inverse_transforms is None:
        #                 inverse_transforms = []
        #             if ts_mask is None:
        #                 ts_mask = np.ones([y_gt.shape[0], y_gt.shape[1]])
        #             if not target_columns:
        #                 target_columns = list(np.arange(y_gt.shape[2]))
        #             rows = []
        #             pd_prediction = pd.DataFrame()
        #             pd_performance = pd.DataFrame()
        #             for ix_target in range(y_gt.shape[2]):
        #                 for ix_fhorizon in range(y_gt.shape[1]):
        #                     if len(inverse_transforms) > ix_target:
        #                         y_gt[:, ix_fhorizon, ix_target] = inverse_transforms[ix_target](
        #                             y_gt[:, ix_fhorizon, ix_target][:, np.newaxis]
        #                         )[:, 0]
        #                         y_pred[:, ix_fhorizon, ix_target] = inverse_transforms[ix_target](
        #                             y_pred[:, ix_fhorizon, ix_target][:, np.newaxis]
        #                         )[:, 0]
        #                     pd_aux = pd.DataFrame(
        #                         {
        #                             "y_gt": y_gt[:, ix_fhorizon, ix_target],
        #                             "y_pred": y_pred[:, ix_fhorizon, ix_target],
        #                             "forecast_horizon": ix_fhorizon + 1,
        #                             "target": target_columns[ix_target],
        #                             "on_mask": ts_mask[:, ix_fhorizon],
        #                         }
        #                     )
        #                     pd_prediction = pd.concat([pd_prediction, pd_aux], axis=0)
        #                     y_gt_mask = y_gt[:, ix_fhorizon, ix_target][ts_mask[:, ix_fhorizon] > 0]
        #                     y_pred_mask = y_pred[:, ix_fhorizon, ix_target][ts_mask[:, ix_fhorizon] > 0]
        #                     valid_mask = np.isfinite(y_gt_mask) & np.isfinite(y_pred_mask)
        #                     y_gt_mask = y_gt_mask[valid_mask]
        #                     y_pred_mask = y_pred_mask[valid_mask]
        #                     if y_gt_mask.shape[0] > 0:
        #                         for metric in _METRICS_FORECAST:
        #                             value = _METRICS_FORECAST[metric](
        #                                 y_gt[:, :ix_fhorizon, ix_target],
        #                                 y_pred[:, :ix_fhorizon, ix_target],
        #                                 axis=1,
        #                             )
        #                             stat = np.mean(value) if value is not None else None
        #                             rows.append(
        #                                 [target_columns[ix_target], ix_fhorizon + 1, metric, stat]
        #                             )
        #             if rows:
        #                 pd_performance = pd.DataFrame(
        #                     data=rows, columns=["target", "forecast", "metric", "value"]
        #                 )
        #             if prediction:
        #                 return pd_performance, pd_prediction
        #             return pd_performance

        #         pd_performance_i = _get_performance(
        #             y_gt, y_pred_eval, target_columns=target_columns, prediction=False
        #         )
        #         pd_performance_i["split"] = dataset_key
        #         pd_performance = pd.concat([pd_performance, pd_performance_i], axis=0)

        # # Preserve train_time in output for backward compatibility
        # train_stage = next(
        #     (s for s in metrics.stages if s.stage_name == "training"), None
        # )
        # train_time = train_stage.wall_clock_ms / 1000 if train_stage else 0.0
        # pd_performance["train_time"] = train_time
        # return {
        #     "performance": pd_performance,
        #     "save_model_dir": save_model_dir,
        #     "experiment_config_path": os.path.join(save_model_dir, "args_config.yml"),
        # }