from __future__ import annotations

import argparse
import datetime
import json
import traceback
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import torch

from HTAMP.data_processing.data_helpers import DataHelpers
from HTAMP.data_processing.processing_dataclasses import AnnotatedDataFiles
from HTAMP.prediction.data_provider.requests_number_dataset import (
    RequestNumberDataManager,
    RequestsNumberDataset,
    RequestsNumberTimeSeries,
)
from HTAMP.prediction.configs.request_number_config import (
    MedicalRequestDatasetConfig,
    TimeseriesModelConfig,
)
from HTAMP.prediction.module.request_number_module import RequestsNumberModule
from HTAMP.prediction.module.request_number_predictor import build_parser as build_training_parser

SPLITS = ("train", "val", "test")


def _build_dataset_config_from_args(args: argparse.Namespace) -> MedicalRequestDatasetConfig:
    dataset_config_kwargs = dict(
        annotated_data_files=AnnotatedDataFiles(
            annotated_visits="",
            annotated_admissions_discharges="",
            annotated_medications=args.medications_orders_file,
            annotated_blood_pressure=args.blood_pressure_orders_file,
            annotated_heart_rate=args.heart_rate_orders_file,
            annotated_respiratory_rate=args.respiratory_rate_orders_file,
            annotated_temperature=args.temperature_orders_file,
            annotated_oxygen_saturation=args.oxygen_saturation_orders_file,
        ),
        request_dir=args.request_dir,
        dataset_dir=args.dataset_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        time_step_minutes=args.time_step_minutes,
        patient_id_col=args.patient_id_col,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        use_saved_request_data=args.use_saved_request_data,
        input_padding_value=args.input_padding_value,
        target_padding_value=args.target_padding_value,
    )
    if args.test_iso_weeks:
        dataset_config_kwargs["test_iso_weeks"] = tuple(
            DataHelpers.parse_iso_week_args(args.test_iso_weeks)
        )

    return MedicalRequestDatasetConfig(**dataset_config_kwargs)


class RequestNumberPredictionManager:
    def __init__(
        self,
        dataset_config: MedicalRequestDatasetConfig,
        model_config: TimeseriesModelConfig,
        checkpoint_file_path: str,
        predictions_dir: Optional[str] = None,
        data_folders: object | None = None,
        load_predictions: bool = False,
        splits: Sequence[str] = SPLITS,
    ) -> None:
        self.dataset_config = dataset_config
        self.model_config = model_config
        self.checkpoint_path = Path(checkpoint_file_path)
        self.predictions_dir = self._resolve_predictions_dir(
            predictions_dir=predictions_dir,
            data_folders=data_folders,
        )
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.splits = self._normalize_splits(splits=splits)

        if load_predictions:
            self.prediction_df = self._load_request_predictions()
        else:
            self.prediction_df = self._initialize_prediction_df()

    def _resolve_predictions_dir(
        self,
        predictions_dir: Optional[str],
        data_folders: object | None,
    ) -> Path:
        if predictions_dir is not None:
            return Path(predictions_dir)

        predicted_requests_path = getattr(data_folders, "predicted_requests_folder_path", None)
        if predicted_requests_path is not None:
            return Path(predicted_requests_path)

        return Path(self.dataset_config.dataset_dir) / "predictions"

    def _normalize_splits(self, splits: Sequence[str]) -> tuple[str, ...]:
        normalized_splits = []
        for split in splits:
            if split not in SPLITS:
                raise ValueError(f"Unsupported split '{split}'. Expected one of {SPLITS}.")
            normalized_splits.append(split)
        return tuple(dict.fromkeys(normalized_splits))

    def _prediction_pickle_path(self) -> Path:
        return self.predictions_dir / "request_numbers.pkl"

    def _prediction_csv_path(self) -> Path:
        return self.predictions_dir / "request_numbers.csv"

    def _prediction_metadata_path(self) -> Path:
        return self.predictions_dir / "request_numbers_metadata.json"

    def _build_time_series(self) -> RequestsNumberTimeSeries:
        request_data_manager = RequestNumberDataManager(
            dataset_config=self.dataset_config,
            preprocess=self.model_config.preprocess_data,
            save_data=self.model_config.preprocess_data,
        )

        train_data_df, train_segments_df = request_data_manager.get_requests_numbers_training_data()
        val_data_df, val_segments_df = request_data_manager.get_requests_numbers_validation_data()
        test_data_df, test_segments_df = request_data_manager.get_requests_numbers_testing_data()

        time_series = RequestsNumberTimeSeries(
            train_data_df=train_data_df,
            val_data_df=val_data_df,
            test_data_df=test_data_df,
            train_segments_df=train_segments_df,
            val_segments_df=val_segments_df,
            test_segments_df=test_segments_df,
            metadata=request_data_manager.metadata,
            sequence_length=self.model_config.seq_len,
            label_length=self.model_config.label_len,
            prediction_length=self.model_config.pred_len,
        )
        self.model_config.sync_channel_dimensions(
            num_input_channels=len(time_series.input_feature_cols),
            num_output_channels=len(time_series.target_cols),
        )
        return time_series

    def _load_model(self, device: torch.device) -> RequestsNumberModule:
        print("Loading request interval predictor ...")
        requests_number_forecaster = RequestsNumberModule.load_from_checkpoint(
            checkpoint_path=str(self.checkpoint_path),
            map_location=device,
        )
        requests_number_forecaster.to(device)
        requests_number_forecaster.eval()
        print("Request interval predictor has been loaded!")
        return requests_number_forecaster

    def _build_dataset(
        self,
        time_series: RequestsNumberTimeSeries,
        split: str,
    ) -> RequestsNumberDataset:
        return RequestsNumberDataset(
            request_time_series=time_series,
            split=split,
            sequence_length=self.model_config.seq_len,
            label_length=self.model_config.label_len,
            prediction_length=self.model_config.pred_len,
        )

    def _apply_target_padding(
        self,
        time_differences,
        availability,
        target_padding_value: float,
    ):
        padded = time_differences.copy()
        padded[availability < 0.5] = target_padding_value
        return padded

    def _build_prediction_record(
        self,
        split: str,
        time_series: RequestsNumberTimeSeries,
        metadata_row: pd.Series,
        prediction_intervals,
        prediction_availability,
        true_intervals,
        true_availability,
    ) -> dict[str, object]:
        anchor_step = int(
            metadata_row["anchor_step"] if "anchor_step" in metadata_row else metadata_row["anchor_index"]
        )
        return {
            "split": split,
            "patient_id": str(metadata_row["patient_id"]),
            "segment_id": int(metadata_row["segment_id"]),
            "anchor_step": anchor_step,
            "anchor_timestamp": pd.Timestamp(metadata_row["anchor_timestamp"]),
            "history_timestamps_by_type": json.loads(metadata_row["history_timestamps_by_type"]),
            "last_observed_timestamps_by_type": json.loads(metadata_row["last_observed_timestamps_by_type"]),
            "future_timestamps_by_type": json.loads(metadata_row["future_timestamps_by_type"]),
            "task_columns": list(time_series.task_names),
            "prediction": prediction_intervals.tolist(),
            "prediction_available": prediction_availability.tolist(),
            "true_value": true_intervals.tolist(),
            "true_available": true_availability.tolist(),
            "target_padding_value": time_series.target_padding_value,
        }

    def _generate_request_predictions(
        self,
        request_number_dataset: RequestsNumberDataset,
        time_series: RequestsNumberTimeSeries,
        split: str,
        requests_number_forecaster: RequestsNumberModule,
    ) -> list[dict[str, object]]:
        prediction_records: list[dict[str, object]] = []
        metadata_df = time_series.get_split_metadata(split=split).reset_index(drop=True)

        print(f"Generating request-interval predictions on split '{split}' ...")
        for i in range(len(request_number_dataset)):
            metadata_row = metadata_df.iloc[i]
            seq_x, seq_y, seq_x_mark, seq_y_mark = request_number_dataset[i]

            true_targets = seq_y.unsqueeze(0)
            true_delta_scaled = true_targets[:, -self.model_config.pred_len :, time_series.delta_target_indices].cpu().numpy()
            true_availability = true_targets[:, -self.model_config.pred_len :, time_series.availability_target_indices].cpu().numpy()[0]
            true_intervals = time_series.inverse_transform_target_deltas(true_delta_scaled)[0]
            true_intervals = self._apply_target_padding(
                time_differences=true_intervals,
                availability=true_availability,
                target_padding_value=time_series.target_padding_value,
            )

            prediction_output = requests_number_forecaster.predict(
                x=seq_x.unsqueeze(0),
                x_mark=seq_x_mark.unsqueeze(0),
                y_mark=seq_y_mark.unsqueeze(0),
            )
            prediction_availability = prediction_output["availability"][0]
            prediction_intervals = self._apply_target_padding(
                time_differences=prediction_output["time_differences"][0],
                availability=prediction_availability,
                target_padding_value=time_series.target_padding_value,
            )

            prediction_records.append(
                self._build_prediction_record(
                    split=split,
                    time_series=time_series,
                    metadata_row=metadata_row,
                    prediction_intervals=prediction_intervals,
                    prediction_availability=prediction_availability,
                    true_intervals=true_intervals,
                    true_availability=true_availability,
                )
            )

            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1} windows for split '{split}'.")

        print(f"Request-interval predictions have been generated for split '{split}'!")
        return prediction_records

    def _build_prediction_metadata(self, time_series: RequestsNumberTimeSeries) -> dict[str, object]:
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "dataset_dir": self.dataset_config.dataset_dir,
            "request_dir": self.dataset_config.request_dir,
            "splits": list(self.splits),
            "seq_len": self.model_config.seq_len,
            "label_len": self.model_config.label_len,
            "pred_len": self.model_config.pred_len,
            "input_feature_columns": list(time_series.input_feature_cols),
            "target_columns": list(time_series.target_cols),
            "task_columns": list(time_series.task_names),
            "target_padding_value": time_series.target_padding_value,
            "metadata": time_series.metadata,
        }

    def _initialize_prediction_df(self) -> pd.DataFrame:
        time_series = self._build_time_series()
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        requests_number_forecaster = self._load_model(device=device)
        prediction_rows: list[dict[str, object]] = []

        for split in self.splits:
            request_number_dataset = self._build_dataset(time_series=time_series, split=split)
            prediction_rows.extend(
                self._generate_request_predictions(
                    request_number_dataset=request_number_dataset,
                    time_series=time_series,
                    split=split,
                    requests_number_forecaster=requests_number_forecaster,
                )
            )

        prediction_df = pd.DataFrame(prediction_rows)
        self._save_request_predictions(
            prediction_df=prediction_df,
            metadata=self._build_prediction_metadata(time_series=time_series),
        )
        return prediction_df

    def _save_request_predictions(
        self,
        prediction_df: pd.DataFrame,
        metadata: dict[str, object],
    ) -> None:
        print("Saving request-interval predictions ...")
        prediction_df.to_pickle(self._prediction_pickle_path())
        prediction_df.to_csv(self._prediction_csv_path(), index=False)
        with self._prediction_metadata_path().open("w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2, default=str)
        print("Request-interval predictions have been saved!")

    def _load_request_predictions(self) -> pd.DataFrame:
        print("Loading request-interval predictions ...")
        prediction_df = pd.read_pickle(self._prediction_pickle_path())
        print("Request-interval predictions have been loaded!")
        return prediction_df


class RequestLocationsPredictionManager:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "The location prediction handler still belongs to the old mobility project and "
            "has not been ported into the current medical prediction pipeline."
        )


Request_Number_Prediction_Manager = RequestNumberPredictionManager
Request_Locations_Prediction_Manager = RequestLocationsPredictionManager


def build_parser() -> argparse.ArgumentParser:
    parser = build_training_parser()
    parser.prog = "RequestPredictionHandler"
    parser.description = "Generate medical request-interval forecasts from a trained checkpoint."
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Lightning checkpoint for the trained request-interval forecaster.",
    )
    parser.add_argument(
        "--predictions_dir",
        type=str,
        default=None,
        help="Output folder for request_numbers.pkl/csv. Defaults to <dataset_dir>/predictions.",
    )
    parser.add_argument(
        "--load_predictions",
        action="store_true",
        default=False,
        help="Load previously saved predictions instead of regenerating them.",
    )
    parser.add_argument(
        "--prediction_splits",
        nargs="*",
        default=list(SPLITS),
        choices=SPLITS,
        help="Dataset splits to generate forecasts for.",
    )
    return parser


if __name__ == "__main__":
    p_start = datetime.datetime.now()
    try:
        parser = build_parser()
        args = parser.parse_args()

        dataset_config = _build_dataset_config_from_args(args=args)
        model_config = TimeseriesModelConfig.from_namespace(args=args)

        RequestNumberPredictionManager(
            dataset_config=dataset_config,
            model_config=model_config,
            checkpoint_file_path=args.checkpoint_path,
            predictions_dir=args.predictions_dir,
            load_predictions=args.load_predictions,
            splits=args.prediction_splits,
        )
    except Exception as error_main_context:
        print("Fail End Process: ", error_main_context)
        traceback.print_exc()
    p_stop = datetime.datetime.now()
    print("Execution time: " + str(p_stop - p_start))
