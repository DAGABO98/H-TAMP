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
from HTAMP.prediction.request_number_config import (
    MedicalRequestDatasetConfig,
    TimeseriesModelConfig,
)
from HTAMP.prediction.request_number_module import RequestsNumberModule
from HTAMP.prediction.request_number_predictor import build_parser as build_training_parser

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
        )
        self.model_config.sync_channel_dimensions(num_channels=len(time_series.feature_cols))
        return time_series

    def _load_model(self, device: torch.device) -> RequestsNumberModule:
        print("Loading request number predictor ...")
        requests_number_forecaster = RequestsNumberModule.load_from_checkpoint(
            checkpoint_path=str(self.checkpoint_path),
            map_location=device,
        )
        requests_number_forecaster.to(device)
        requests_number_forecaster.eval()
        print("Request number predictor has been loaded!")
        return requests_number_forecaster

    def _raw_split_df(self, time_series: RequestsNumberTimeSeries, split: str) -> pd.DataFrame:
        return {
            "train": time_series.train_data_df,
            "val": time_series.val_data_df,
            "test": time_series.test_data_df,
        }[split]

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

    def _build_prediction_record(
        self,
        split: str,
        time_series: RequestsNumberTimeSeries,
        split_raw_df: pd.DataFrame,
        start_point: int,
        prediction,
        true_value,
    ) -> dict[str, object]:
        target_start_idx = start_point + self.model_config.seq_len
        target_end_idx = target_start_idx + self.model_config.pred_len - 1
        input_start_idx = start_point
        input_end_idx = target_start_idx - 1

        input_start_row = split_raw_df.iloc[input_start_idx]
        input_end_row = split_raw_df.iloc[input_end_idx]
        target_start_row = split_raw_df.iloc[target_start_idx]
        target_end_row = split_raw_df.iloc[target_end_idx]
        horizon_slice = split_raw_df.iloc[target_start_idx : target_end_idx + 1]

        target_timestamp = pd.Timestamp(target_start_row[time_series.timestamp_col])

        return {
            "split": split,
            "patient_id": str(target_start_row[time_series.patient_id_col]),
            "input_start_timestamp": pd.Timestamp(input_start_row[time_series.timestamp_col]),
            "input_end_timestamp": pd.Timestamp(input_end_row[time_series.timestamp_col]),
            "forecast_start_timestamp": target_timestamp,
            "forecast_end_timestamp": pd.Timestamp(target_end_row[time_series.timestamp_col]),
            "forecast_start_year": int(target_timestamp.year),
            "forecast_start_month": int(target_start_row["month"]),
            "forecast_start_day": int(target_start_row["day"]),
            "forecast_start_weekday": int(target_start_row["weekday"]),
            "forecast_start_hour": int(target_start_row["hour"]),
            "forecast_start_minute": int(target_start_row["minute"]),
            "horizon_timestamps": horizon_slice[time_series.timestamp_col].astype(str).tolist(),
            "target_columns": list(time_series.target_cols),
            "prediction": prediction.tolist(),
            "true_value": true_value.tolist(),
        }

    def _generate_request_predictions(
        self,
        request_number_dataset: RequestsNumberDataset,
        time_series: RequestsNumberTimeSeries,
        split: str,
        requests_number_forecaster: RequestsNumberModule,
        device: torch.device,
    ) -> list[dict[str, object]]:
        split_raw_df = self._raw_split_df(time_series=time_series, split=split)
        prediction_records: list[dict[str, object]] = []

        print(f"Generating predictions for number of requests on split '{split}' ...")
        for i, start_point in enumerate(request_number_dataset.slice_start_points):
            seq_x, seq_y, seq_x_mark, seq_y_mark = request_number_dataset[i]

            seq_x = seq_x.unsqueeze(0)
            seq_y = seq_y.unsqueeze(0)
            seq_x_mark = seq_x_mark.unsqueeze(0)
            seq_y_mark = seq_y_mark.unsqueeze(0)

            true_value = seq_y[
                :,
                -self.model_config.pred_len :,
                time_series.target_channel_indices,
            ].cpu().numpy()
            true_value = time_series.inverse_transform(true_value)[0]

            prediction = requests_number_forecaster.predict(
                x=seq_x,
                x_mark=seq_x_mark,
                y_mark=seq_y_mark,
            )[0]

            prediction_records.append(
                self._build_prediction_record(
                    split=split,
                    time_series=time_series,
                    split_raw_df=split_raw_df,
                    start_point=start_point,
                    prediction=prediction,
                    true_value=true_value,
                )
            )

            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1} windows for split '{split}'.")

        print(f"Predicted number of requests has been generated for split '{split}'!")
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
            "feature_columns": list(time_series.feature_cols),
            "target_columns": list(time_series.target_cols),
            "auxiliary_feature_columns": list(time_series.auxiliary_feature_cols),
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
                    device=device,
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
        print("Saving number predictions ...")
        prediction_df.to_pickle(self._prediction_pickle_path())
        prediction_df.to_csv(self._prediction_csv_path(), index=False)
        with self._prediction_metadata_path().open("w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2, default=str)
        print("Number predictions have been saved!")

    def _load_request_predictions(self) -> pd.DataFrame:
        print("Loading number predictions ...")
        prediction_df = pd.read_pickle(self._prediction_pickle_path())
        print("Number predictions have been loaded!")
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
    parser.description = "Generate medical request-count forecasts from a trained checkpoint."
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Lightning checkpoint for the trained request-count forecaster.",
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
