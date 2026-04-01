from __future__ import annotations

import argparse
import datetime
import json
import re
from pathlib import Path
import traceback
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from HTAMP.data_processing.processing_dataclasses import AnnotatedDataFiles
from HTAMP.planning.request_handler import GlobalRequestHandler
from HTAMP.prediction.configs.request_config import MedicalRequestDatasetConfig
from HTAMP.prediction.data_provider.data_module import DataModule

TASK_SPECS: dict[str, str] = {
    "medication": "Medication Scheduled DTTM",
    "blood_pressure": "Scheduled DTTM",
    "heart_rate": "Scheduled DTTM",
    "respiratory_rate": "Scheduled DTTM",
    "temperature": "Scheduled DTTM",
    "oxygen_saturation": "Scheduled DTTM",
}

REQUEST_HANDLER_ATTRS: dict[str, str] = {
    "medication": "med_df",
    "blood_pressure": "bp_df",
    "heart_rate": "hr_df",
    "respiratory_rate": "rr_df",
    "temperature": "temp_df",
    "oxygen_saturation": "os_df",
}

VITAL_MEASUREMENT_COMPONENTS: dict[str, dict[str, list[str]]] = {
    "blood_pressure": {
        "systolic": [
            "Blood Pressure Systolic Value",
        ],
        "diastolic": [
            "Blood Pressure Diastolic Value",
        ],
        "combined": [
            "Blood Pressure Value",
        ],
    },
    "heart_rate": {
        "value": [
            "Heart Rate Value",
        ],
    },
    "respiratory_rate": {
        "value": [
            "Respiration Value",
        ],
    },
    "temperature": {
        "value": [
            "Temperature Value",
        ],
    },
    "oxygen_saturation": {
        "value": [
            "SP02 Value",
        ],
    },
}

VITAL_OUTPUT_COMPONENTS: dict[str, list[str]] = {
    "blood_pressure": ["systolic", "diastolic"],
    "heart_rate": ["value"],
    "respiratory_rate": ["value"],
    "temperature": ["value"],
    "oxygen_saturation": ["value"],
}

TIMESTAMP_COLUMN = "timestamp"
TIME_COLUMNS = ["month", "day", "weekday", "hour", "minute"]
SEGMENT_COLUMNS = ["patient_id", "start_idx", "end_idx", "num_rows"]
SPLITS = ("train", "val", "test")
TASK_NAMES = tuple(TASK_SPECS.keys())
TASK_TO_INDEX = {task_name: index for index, task_name in enumerate(TASK_NAMES)}


def _normalize_column_name(column_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(column_name).lower())


def _event_measurement_column(task_name: str, component: str) -> str:
    return f"{task_name}_{component}"


TASK_EVENT_MEASUREMENT_COLUMNS: dict[str, list[str]] = {
    task_name: [
        _event_measurement_column(task_name=task_name, component=component)
        for component in VITAL_OUTPUT_COMPONENTS.get(task_name, [])
    ]
    for task_name in TASK_NAMES
}

EVENT_MEASUREMENT_COLUMNS = tuple(
    feature_name
    for task_name in TASK_NAMES
    for feature_name in TASK_EVENT_MEASUREMENT_COLUMNS.get(task_name, [])
)


class RequestsDataManager:
    def __init__(
        self,
        dataset_config: MedicalRequestDatasetConfig
    ) -> None:
        self.dataset_config = dataset_config
        self.dataset_dir = Path(self.dataset_config.dataset_dir)
        self.metadata: dict[str, object] = {}
        self._warned_missing_measurements: set[str] = set()

        if dataset_config.preprocess_data:
            self.dataset_dir.mkdir(parents=True, exist_ok=True)
            request_handler = self._build_request_handler()
            split_data = self._preprocess_requests_data(request_handler=request_handler)
            self._unpack_preprocessed_data(split_data=split_data)
            if dataset_config.save_data:
                self._save_dataframes()
        else:
            self._load_dataframes()

    @property
    def patient_id_col(self) -> str:
        return self.dataset_config.patient_id_col

    @property
    def task_names(self) -> list[str]:
        return list(TASK_NAMES)

    @property
    def time_cols(self) -> list[str]:
        return TIME_COLUMNS.copy()

    @property
    def event_measurement_cols(self) -> list[str]:
        return list(EVENT_MEASUREMENT_COLUMNS)

    @property
    def event_cols(self) -> list[str]:
        return [
            self.patient_id_col,
            TIMESTAMP_COLUMN,
            "task_name",
            "task_index",
            *self.time_cols,
            "interval_available",
            "time_diff_minutes",
        ]

    def _build_request_handler(self) -> GlobalRequestHandler:
        return GlobalRequestHandler(
            annotated_data_files=self.dataset_config.annotated_data_files,
            request_dir=self.dataset_config.request_dir,
            start_date=self.dataset_config.start_date,
            end_date=self.dataset_config.end_date,
            use_saved_data=self.dataset_config.use_saved_request_data,
        )

    def _task_frames(self, request_handler: GlobalRequestHandler) -> dict[str, pd.DataFrame]:
        return {
            task_name: getattr(request_handler, handler_attr)
            for task_name, handler_attr in REQUEST_HANDLER_ATTRS.items()
        }

    def _empty_events_frame(self) -> pd.DataFrame:
        return pd.DataFrame(columns=self.event_cols)

    def _empty_segments_frame(self) -> pd.DataFrame:
        return pd.DataFrame(columns=SEGMENT_COLUMNS)

    def _match_first_column(self, columns: list[str], candidates: list[str]) -> Optional[str]:
        normalized_columns = {_normalize_column_name(column): column for column in columns}
        for candidate in candidates:
            matched_column = normalized_columns.get(_normalize_column_name(candidate))
            if matched_column is not None:
                return matched_column
        return None

    def _coerce_numeric_series(self, series: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(series):
            return pd.to_numeric(series, errors="coerce")
        extracted = series.astype(str).str.extract(r"(-?\d+(?:\.\d+)?)", expand=False)
        return pd.to_numeric(extracted, errors="coerce")

    def _extract_blood_pressure_pair(self, series: pd.Series) -> tuple[pd.Series, pd.Series]:
        extracted = series.astype(str).str.extract(
            r"(?P<systolic>-?\d+(?:\.\d+)?)\s*/\s*(?P<diastolic>-?\d+(?:\.\d+)?)"
        )
        systolic = pd.to_numeric(extracted["systolic"], errors="coerce")
        diastolic = pd.to_numeric(extracted["diastolic"], errors="coerce")
        return systolic, diastolic

    def _warn_missing_measurements(self, task_name: str) -> None:
        if task_name in self._warned_missing_measurements:
            return
        self._warned_missing_measurements.add(task_name)
        print(
            f"Warning: no measurement value column was detected for '{task_name}'. "
            "The request-interval features for that task will use padding values."
        )

    def _extract_measurement_components(
        self,
        task_name: str,
        task_df: pd.DataFrame,
    ) -> dict[str, pd.Series]:
        if task_name not in VITAL_MEASUREMENT_COMPONENTS:
            return {}

        columns = task_df.columns.tolist()
        candidates = VITAL_MEASUREMENT_COMPONENTS[task_name]

        if task_name == "blood_pressure":
            systolic_col = self._match_first_column(columns, candidates["systolic"])
            diastolic_col = self._match_first_column(columns, candidates["diastolic"])
            if systolic_col is not None and diastolic_col is not None:
                return {
                    "systolic": self._coerce_numeric_series(task_df[systolic_col]),
                    "diastolic": self._coerce_numeric_series(task_df[diastolic_col]),
                }

            combined_col = self._match_first_column(columns, candidates["combined"])
            if combined_col is not None:
                systolic, diastolic = self._extract_blood_pressure_pair(task_df[combined_col])
                return {
                    "systolic": systolic,
                    "diastolic": diastolic,
                }

            self._warn_missing_measurements(task_name=task_name)
            return {}

        matched_col = self._match_first_column(columns, candidates["value"])
        if matched_col is None:
            self._warn_missing_measurements(task_name=task_name)
            return {}

        return {"value": self._coerce_numeric_series(task_df[matched_col])}

    def _base_task_df(self, df: pd.DataFrame, time_col: str) -> pd.DataFrame:
        task_df = df.copy()
        task_df = task_df.dropna(subset=[self.patient_id_col, time_col])
        task_df[self.patient_id_col] = task_df[self.patient_id_col].astype(str)
        task_df[TIMESTAMP_COLUMN] = pd.to_datetime(task_df[time_col], errors="coerce")
        task_df = task_df.dropna(subset=[TIMESTAMP_COLUMN])
        task_df = task_df.sort_values([self.patient_id_col, TIMESTAMP_COLUMN]).reset_index(drop=True)
        return task_df

    def _build_task_events(
        self,
        df: pd.DataFrame,
        task_name: str,
        time_col: str,
    ) -> pd.DataFrame:
        if df.empty:
            return self._empty_events_frame()

        task_df = self._base_task_df(df=df, time_col=time_col)
        if task_df.empty:
            return self._empty_events_frame()

        event_df = task_df[[self.patient_id_col, TIMESTAMP_COLUMN]].copy()
        event_df["task_name"] = task_name
        event_df["task_index"] = TASK_TO_INDEX[task_name]
        event_df["month"] = event_df[TIMESTAMP_COLUMN].dt.month.astype(np.int64)
        event_df["day"] = event_df[TIMESTAMP_COLUMN].dt.day.astype(np.int64)
        event_df["weekday"] = event_df[TIMESTAMP_COLUMN].dt.weekday.astype(np.int64)
        event_df["hour"] = event_df[TIMESTAMP_COLUMN].dt.hour.astype(np.int64)
        event_df["minute"] = event_df[TIMESTAMP_COLUMN].dt.minute.astype(np.int64)
        event_df["interval_available"] = 0.0
        event_df["time_diff_minutes"] = np.nan

        for measurement_col in self.event_measurement_cols:
            event_df[measurement_col] = np.nan

        for component_name, component_series in self._extract_measurement_components(
            task_name=task_name,
            task_df=task_df,
        ).items():
            event_df[_event_measurement_column(task_name=task_name, component=component_name)] = component_series.astype(float)

        return event_df[self.event_cols]

    def _build_events_df(self, request_handler: GlobalRequestHandler) -> pd.DataFrame:
        task_frames = self._task_frames(request_handler=request_handler)
        event_frames = [
            self._build_task_events(
                df=task_frames[task_name],
                task_name=task_name,
                time_col=TASK_SPECS[task_name],
            )
            for task_name in TASK_NAMES
        ]

        events_df = pd.concat(event_frames, axis=0, ignore_index=True)
        if events_df.empty:
            raise ValueError(
                "No medical requests were found for the configured date range. "
                "Please check the annotated files and date filters."
            )

        events_df = events_df.sort_values(
            [self.patient_id_col, TIMESTAMP_COLUMN, "task_index"]
        ).reset_index(drop=True)
        return events_df[self.event_cols]

    def _derive_iso_week_fields(self, timestamp_series: pd.Series) -> pd.DataFrame:
        iso_calendar = pd.to_datetime(timestamp_series).dt.isocalendar()
        return pd.DataFrame(
            {
                "iso_year": iso_calendar["year"].astype(int),
                "iso_week": iso_calendar["week"].astype(int),
            }
        )

    def _validation_ratio_over_non_test_weeks(self) -> float:
        remaining_ratio = self.dataset_config.train_ratio + self.dataset_config.val_ratio
        if remaining_ratio <= 0.0:
            return 0.0
        return float(self.dataset_config.val_ratio / remaining_ratio)

    def _resolve_week_split_sets(
        self,
        patient_events: dict[str, pd.DataFrame],
    ) -> dict[str, set[tuple[int, int]]]:
        unique_weeks: set[tuple[int, int]] = set()

        for patient_df in patient_events.values():
            iso_fields = self._derive_iso_week_fields(patient_df[TIMESTAMP_COLUMN])
            unique_weeks.update(zip(iso_fields["iso_year"], iso_fields["iso_week"]))

        sorted_unique_weeks = sorted(unique_weeks)
        test_weeks = set(self.dataset_config.test_iso_weeks).intersection(sorted_unique_weeks)
        non_test_weeks = [week for week in sorted_unique_weeks if week not in test_weeks]

        val_ratio = self._validation_ratio_over_non_test_weeks()
        val_week_count = int(np.floor(len(non_test_weeks) * val_ratio))
        if val_ratio > 0.0 and val_week_count == 0 and non_test_weeks:
            val_week_count = 1
        val_week_count = min(val_week_count, len(non_test_weeks))

        val_weeks = set(non_test_weeks[-val_week_count:]) if val_week_count > 0 else set()
        train_weeks = set(non_test_weeks) - val_weeks

        return {
            "train": train_weeks,
            "val": val_weeks,
            "test": test_weeks,
        }

    def _split_patient_series_by_weeks(
        self,
        patient_df: pd.DataFrame,
        split_week_sets: dict[str, set[tuple[int, int]]],
    ) -> dict[str, list[pd.DataFrame]]:
        patient_with_iso = patient_df.copy()
        iso_fields = self._derive_iso_week_fields(patient_with_iso[TIMESTAMP_COLUMN])
        patient_with_iso["iso_year"] = iso_fields["iso_year"]
        patient_with_iso["iso_week"] = iso_fields["iso_week"]

        split_rows: dict[str, list[pd.DataFrame]] = {split: [] for split in SPLITS}
        current_split: Optional[str] = None
        run_start = 0

        def resolve_split(iso_year: int, iso_week: int) -> Optional[str]:
            week_key = (int(iso_year), int(iso_week))
            for split_name in SPLITS:
                if week_key in split_week_sets[split_name]:
                    return split_name
            return None

        resolved_splits = [
            resolve_split(iso_year=row.iso_year, iso_week=row.iso_week)
            for row in patient_with_iso[["iso_year", "iso_week"]].itertuples(index=False)
        ]

        for row_index, split_name in enumerate(resolved_splits):
            if split_name != current_split:
                if current_split is not None:
                    split_rows[current_split].append(
                        patient_with_iso.iloc[run_start:row_index]
                        .drop(columns=["iso_year", "iso_week"])
                        .reset_index(drop=True)
                    )
                current_split = split_name
                run_start = row_index

        if current_split is not None:
            split_rows[current_split].append(
                patient_with_iso.iloc[run_start:]
                .drop(columns=["iso_year", "iso_week"])
                .reset_index(drop=True)
            )

        return split_rows

    def _add_interval_features(
        self,
        split_df: pd.DataFrame,
        segments_df: pd.DataFrame,
    ) -> pd.DataFrame:
        if split_df.empty or segments_df.empty:
            return split_df

        enriched_df = split_df.copy(deep=True)
        enriched_df["interval_available"] = 0.0
        enriched_df["time_diff_minutes"] = np.nan

        for segment in segments_df.itertuples(index=False):
            start_idx = int(segment.start_idx)
            end_idx = int(segment.end_idx)
            segment_df = enriched_df.iloc[start_idx:end_idx].copy()
            segment_df["time_diff_minutes"] = (
                segment_df.groupby("task_name")[TIMESTAMP_COLUMN]
                .diff()
                .dt.total_seconds()
                .div(60.0)
            )
            segment_df["interval_available"] = segment_df["time_diff_minutes"].notna().astype(float)

            enriched_df.loc[start_idx:end_idx - 1, "time_diff_minutes"] = segment_df["time_diff_minutes"].to_numpy()
            enriched_df.loc[start_idx:end_idx - 1, "interval_available"] = segment_df["interval_available"].to_numpy()

        enriched_df["interval_available"] = enriched_df["interval_available"].fillna(0.0).astype(float)
        enriched_df["time_diff_minutes"] = pd.to_numeric(enriched_df["time_diff_minutes"], errors="coerce")
        return enriched_df

    def _build_split_frames(
        self,
        events_df: pd.DataFrame,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        patient_events = {
            patient_id: patient_df.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
            for patient_id, patient_df in events_df.groupby(self.patient_id_col, sort=False)
        }
        split_week_sets = self._resolve_week_split_sets(patient_events=patient_events)
        split_frame_parts: dict[str, list[pd.DataFrame]] = {split: [] for split in SPLITS}
        split_segments: dict[str, list[dict[str, object]]] = {split: [] for split in SPLITS}
        split_offsets = {split: 0 for split in SPLITS}

        for patient_id, patient_df in patient_events.items():
            patient_split_frames = self._split_patient_series_by_weeks(
                patient_df=patient_df,
                split_week_sets=split_week_sets,
            )

            for split_name, split_dfs in patient_split_frames.items():
                for split_df in split_dfs:
                    if split_df.empty:
                        continue

                    split_df = split_df.sort_values(
                        [TIMESTAMP_COLUMN, "task_index"]
                    ).reset_index(drop=True)

                    start_idx = split_offsets[split_name]
                    end_idx = start_idx + len(split_df)
                    split_offsets[split_name] = end_idx

                    split_frame_parts[split_name].append(split_df)
                    split_segments[split_name].append(
                        {
                            "patient_id": patient_id,
                            "start_idx": start_idx,
                            "end_idx": end_idx,
                            "num_rows": len(split_df),
                        }
                    )

        split_frames = {
            split_name: (
                pd.concat(frame_parts, ignore_index=True)
                if frame_parts
                else self._empty_events_frame()
            )
            for split_name, frame_parts in split_frame_parts.items()
        }
        split_segments_frames = {
            split_name: (
                pd.DataFrame(segment_rows, columns=SEGMENT_COLUMNS)
                if segment_rows
                else self._empty_segments_frame()
            )
            for split_name, segment_rows in split_segments.items()
        }

        for split_name in SPLITS:
            split_frames[split_name] = self._add_interval_features(
                split_df=split_frames[split_name],
                segments_df=split_segments_frames[split_name],
            )

        self.split_week_sets = split_week_sets
        return split_frames, split_segments_frames

    def _build_metadata(self) -> dict[str, object]:
        return {
            "dataset_representation": "time_aligned_multivariate_request_intervals",
            "sample_axis_semantics": "unique_anchor_timestamp_per_patient_segment",
            "sequence_axis_semantics": "per_task_history_depth",
            "patient_id_col": self.patient_id_col,
            "timestamp_col": TIMESTAMP_COLUMN,
            "task_names": list(TASK_NAMES),
            "task_to_index": TASK_TO_INDEX,
            "time_columns": self.time_cols,
            "event_measurement_columns": self.event_measurement_cols,
            "input_padding_value": self.dataset_config.input_padding_value,
            "target_padding_value": self.dataset_config.target_padding_value,
            "start_date": self.dataset_config.start_date,
            "end_date": self.dataset_config.end_date,
            "train_ratio": self.dataset_config.train_ratio,
            "val_ratio": self.dataset_config.val_ratio,
            "train_iso_weeks": [list(week) for week in sorted(getattr(self, "split_week_sets", {}).get("train", set()))],
            "val_iso_weeks": [list(week) for week in sorted(getattr(self, "split_week_sets", {}).get("val", set()))],
            "test_iso_weeks": [list(week) for week in sorted(getattr(self, "split_week_sets", {}).get("test", set()))],
        }

    def _preprocess_requests_data(
        self,
        request_handler: GlobalRequestHandler,
    ) -> tuple[
        tuple[pd.DataFrame, pd.DataFrame],
        tuple[pd.DataFrame, pd.DataFrame],
        tuple[pd.DataFrame, pd.DataFrame],
        dict[str, object],
    ]:
        events_df = self._build_events_df(request_handler=request_handler)
        split_frames, split_segment_frames = self._build_split_frames(events_df=events_df)
        metadata = self._build_metadata()

        return (
            (split_frames["train"], split_segment_frames["train"]),
            (split_frames["val"], split_segment_frames["val"]),
            (split_frames["test"], split_segment_frames["test"]),
            metadata,
        )

    def _unpack_preprocessed_data(
        self,
        split_data: tuple[
            tuple[pd.DataFrame, pd.DataFrame],
            tuple[pd.DataFrame, pd.DataFrame],
            tuple[pd.DataFrame, pd.DataFrame],
            dict[str, object],
        ],
    ) -> None:
        train_data, val_data, test_data, metadata = split_data
        self.train_requests_df, self.train_segments_df = train_data
        self.val_requests_df, self.val_segments_df = val_data
        self.test_requests_df, self.test_segments_df = test_data
        self.metadata = metadata

    def _metadata_path(self) -> Path:
        return self.dataset_dir / "metadata.json"

    def _save_split(
        self,
        split_name: str,
        data_df: pd.DataFrame,
        segments_df: pd.DataFrame,
    ) -> None:
        data_df.to_csv(self.dataset_dir / f"{split_name}_data.csv", index=False)
        segments_df.to_csv(self.dataset_dir / f"{split_name}_segments.csv", index=False)

    def _save_dataframes(self) -> None:
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self._save_split("train", self.train_requests_df, self.train_segments_df)
        self._save_split("val", self.val_requests_df, self.val_segments_df)
        self._save_split("test", self.test_requests_df, self.test_segments_df)
        with self._metadata_path().open("w", encoding="utf-8") as metadata_file:
            json.dump(self.metadata, metadata_file, indent=2)

    def _load_split(self, split_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        data_df = pd.read_csv(self.dataset_dir / f"{split_name}_data.csv")
        segments_df = pd.read_csv(self.dataset_dir / f"{split_name}_segments.csv")
        patient_id_col = str(self.metadata.get("patient_id_col", self.patient_id_col))

        if not data_df.empty:
            data_df[TIMESTAMP_COLUMN] = pd.to_datetime(data_df[TIMESTAMP_COLUMN], errors="coerce")
            data_df[patient_id_col] = data_df[patient_id_col].astype(str)
            data_df["task_name"] = data_df["task_name"].astype(str)

        if not segments_df.empty:
            segments_df["patient_id"] = segments_df["patient_id"].astype(str)

        return data_df, segments_df

    def _load_dataframes(self) -> None:
        with self._metadata_path().open("r", encoding="utf-8") as metadata_file:
            self.metadata = json.load(metadata_file)

        self.train_requests_df, self.train_segments_df = self._load_split("train")
        self.val_requests_df, self.val_segments_df = self._load_split("val")
        self.test_requests_df, self.test_segments_df = self._load_split("test")

    def get_requests_training_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self.train_requests_df, self.train_segments_df

    def get_requests_validation_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self.val_requests_df, self.val_segments_df

    def get_requests_testing_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self.test_requests_df, self.test_segments_df


class RequestsTimeSeries:
    def __init__(
        self,
        train_data_df: pd.DataFrame,
        val_data_df: pd.DataFrame,
        test_data_df: pd.DataFrame,
        train_segments_df: pd.DataFrame,
        val_segments_df: pd.DataFrame,
        test_segments_df: pd.DataFrame,
        metadata: dict[str, object],
        sequence_length: int,
        label_length: int,
        prediction_length: int,
    ) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be greater than zero.")
        if label_length < 0:
            raise ValueError("label_length must be non-negative.")
        if prediction_length <= 0:
            raise ValueError("prediction_length must be greater than zero.")

        self.train_data_df = train_data_df
        self.val_data_df = val_data_df
        self.test_data_df = test_data_df
        self.train_segments_df = train_segments_df
        self.val_segments_df = val_segments_df
        self.test_segments_df = test_segments_df
        self.metadata = metadata

        self.sequence_length = sequence_length
        self.label_length = label_length
        self.prediction_length = prediction_length
        self.target_sequence_length = self.label_length + self.prediction_length

        self.patient_id_col = str(metadata["patient_id_col"])
        self.timestamp_col = str(metadata["timestamp_col"])
        self.task_names = list(metadata["task_names"])
        self.time_cols = list(metadata["time_columns"])
        self.event_measurement_cols = list(metadata["event_measurement_columns"])
        self.input_padding_value = float(metadata.get("input_padding_value", -1.0))
        self.target_padding_value = float(metadata.get("target_padding_value", -1.0))

        (
            self.input_feature_cols,
            self.binary_input_feature_indices,
            self.continuous_input_feature_indices,
            self.task_input_specs,
        ) = self._build_input_feature_schema()

        (
            self.target_cols,
            self.delta_target_indices,
            self.availability_target_indices,
            self.task_target_specs,
        ) = self._build_target_feature_schema()

        self.samples: dict[str, dict[str, object]] = {
            "train": self._build_split_samples(
                split="train",
                events_df=self.train_data_df,
                segments_df=self.train_segments_df,
            ),
            "val": self._build_split_samples(
                split="val",
                events_df=self.val_data_df,
                segments_df=self.val_segments_df,
            ),
            "test": self._build_split_samples(
                split="test",
                events_df=self.test_data_df,
                segments_df=self.test_segments_df,
            ),
        }

        self.input_scaler_mean, self.input_scaler_scale = self._fit_input_scaler(
            train_x=self.samples["train"]["x"]
        )
        self.target_scaler_mean, self.target_scaler_scale = self._fit_target_scaler(
            train_y=self.samples["train"]["y"]
        )
        self._apply_scaling_to_all_splits()

    def _build_input_feature_schema(
        self,
    ) -> tuple[list[str], list[int], list[int], dict[str, dict[str, object]]]:
        input_feature_cols: list[str] = []
        binary_input_feature_indices: list[int] = []
        continuous_input_feature_indices: list[int] = []
        task_input_specs: dict[str, dict[str, object]] = {}
        cursor = 0

        for task_name in self.task_names:
            task_spec: dict[str, object] = {}

            task_spec["request_available_index"] = cursor
            input_feature_cols.append(f"{task_name}__request_available")
            binary_input_feature_indices.append(cursor)
            cursor += 1

            task_spec["interval_available_index"] = cursor
            input_feature_cols.append(f"{task_name}__interval_available")
            binary_input_feature_indices.append(cursor)
            cursor += 1

            task_spec["time_diff_index"] = cursor
            input_feature_cols.append(f"{task_name}__time_diff_minutes")
            continuous_input_feature_indices.append(cursor)
            cursor += 1

            measurement_indices: dict[str, int] = {}
            for measurement_col in TASK_EVENT_MEASUREMENT_COLUMNS.get(task_name, []):
                measurement_indices[measurement_col] = cursor
                feature_name = measurement_col.removeprefix(f"{task_name}_")
                input_feature_cols.append(f"{task_name}__{feature_name}")
                continuous_input_feature_indices.append(cursor)
                cursor += 1
            task_spec["measurement_indices"] = measurement_indices

            time_feature_indices: dict[str, int] = {}
            for time_col in self.time_cols:
                time_feature_indices[time_col] = cursor
                input_feature_cols.append(f"{task_name}__{time_col}")
                continuous_input_feature_indices.append(cursor)
                cursor += 1
            task_spec["time_feature_indices"] = time_feature_indices

            task_input_specs[task_name] = task_spec

        return (
            input_feature_cols,
            binary_input_feature_indices,
            continuous_input_feature_indices,
            task_input_specs,
        )

    def _build_target_feature_schema(
        self,
    ) -> tuple[list[str], list[int], list[int], dict[str, dict[str, int]]]:
        target_cols: list[str] = []
        delta_target_indices: list[int] = []
        availability_target_indices: list[int] = []
        task_target_specs: dict[str, dict[str, int]] = {}
        cursor = 0

        for task_name in self.task_names:
            delta_index = cursor
            target_cols.append(f"{task_name}__next_time_diff_minutes")
            delta_target_indices.append(delta_index)
            cursor += 1

            availability_index = cursor
            target_cols.append(f"{task_name}__next_interval_available")
            availability_target_indices.append(availability_index)
            cursor += 1

            task_target_specs[task_name] = {
                "delta_index": delta_index,
                "availability_index": availability_index,
            }

        return target_cols, delta_target_indices, availability_target_indices, task_target_specs

    def _metadata_columns(self) -> list[str]:
        return [
            "split",
            "patient_id",
            "segment_id",
            "anchor_step",
            "anchor_index",
            "anchor_timestamp",
            "history_timestamps_by_type",
            "last_observed_timestamps_by_type",
            "future_timestamps_by_type",
        ]

    def _empty_split_samples(self) -> dict[str, object]:
        return {
            "x": np.zeros(
                (0, self.sequence_length, len(self.input_feature_cols)),
                dtype=np.float32,
            ),
            "y": np.zeros(
                (0, self.target_sequence_length, len(self.target_cols)),
                dtype=np.float32,
            ),
            "x_mark": np.zeros(
                (0, self.sequence_length, len(self.time_cols)),
                dtype=np.float32,
            ),
            "y_mark": np.zeros(
                (0, self.target_sequence_length, len(self.time_cols)),
                dtype=np.float32,
            ),
            "metadata": pd.DataFrame(columns=self._metadata_columns()),
        }

    def _initialize_input_sample(self) -> np.ndarray:
        sample = np.full(
            (self.sequence_length, len(self.input_feature_cols)),
            self.input_padding_value,
            dtype=np.float32,
        )
        if self.binary_input_feature_indices:
            sample[:, self.binary_input_feature_indices] = 0.0
        return sample

    def _initialize_target_sample(self) -> np.ndarray:
        sample = np.full(
            (self.target_sequence_length, len(self.target_cols)),
            self.target_padding_value,
            dtype=np.float32,
        )
        if self.availability_target_indices:
            sample[:, self.availability_target_indices] = 0.0
        return sample

    def _history_timestamp_payload(
        self,
        history_records: list[dict[str, object]],
    ) -> tuple[list[Optional[str]], Optional[str]]:
        timestamps: list[Optional[str]] = [None] * self.sequence_length
        if not history_records:
            return timestamps, None

        pad_count = self.sequence_length - len(history_records)
        for history_offset, row in enumerate(history_records):
            timestamp = pd.Timestamp(row[self.timestamp_col]).isoformat()
            timestamps[pad_count + history_offset] = timestamp

        return timestamps, timestamps[-1]

    def _future_timestamp_payload(
        self,
        future_records: list[dict[str, object]],
    ) -> list[Optional[str]]:
        timestamps: list[Optional[str]] = [None] * self.prediction_length
        for future_offset, row in enumerate(future_records[: self.prediction_length]):
            timestamps[future_offset] = pd.Timestamp(row[self.timestamp_col]).isoformat()
        return timestamps

    def _segment_anchor_timestamps(
        self,
        segment_df: pd.DataFrame,
    ) -> list[pd.Timestamp]:
        raw_timestamps = pd.to_datetime(
            segment_df[self.timestamp_col],
            errors="coerce",
        ).dropna().unique()
        return [pd.Timestamp(timestamp_value) for timestamp_value in sorted(raw_timestamps)]

    def _slice_task_records(
        self,
        task_records: list[dict[str, object]],
        task_timestamps: pd.DatetimeIndex,
        anchor_timestamp: pd.Timestamp,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        observed_count = int(task_timestamps.searchsorted(anchor_timestamp, side="right"))
        history_start = max(0, observed_count - self.sequence_length)
        history_records = task_records[history_start:observed_count]
        future_records = task_records[observed_count : observed_count + self.prediction_length]
        return history_records, future_records
    
    def _build_segment_task_lookups(
        self,
        segment_df: pd.DataFrame,
    ) -> tuple[dict[str, list[dict[str, object]]], dict[str, pd.DatetimeIndex]]:
        task_records_by_type = {task_name: [] for task_name in self.task_names}
        task_timestamps_by_type = {
            task_name: pd.DatetimeIndex([])
            for task_name in self.task_names
        }

        for task_name, task_df in segment_df.groupby("task_name", sort=False):
            ordered_task_df = task_df.sort_values(self.timestamp_col).reset_index(drop=True)
            task_records_by_type[task_name] = ordered_task_df.to_dict(orient="records")
            task_timestamps_by_type[task_name] = pd.DatetimeIndex(ordered_task_df[self.timestamp_col])

        return task_records_by_type, task_timestamps_by_type

    def _fill_history_block(
        self,
        sample_x: np.ndarray,
        task_name: str,
        history_records: list[dict[str, object]],
    ) -> None:
        if not history_records:
            return

        task_spec = self.task_input_specs[task_name]
        pad_count = self.sequence_length - len(history_records)

        for history_offset, row in enumerate(history_records):
            slot = pad_count + history_offset

            sample_x[slot, task_spec["request_available_index"]] = 1.0
            sample_x[slot, task_spec["interval_available_index"]] = float(row["interval_available"])

            if float(row["interval_available"]) > 0.5 and pd.notna(row["time_diff_minutes"]):
                sample_x[slot, task_spec["time_diff_index"]] = float(row["time_diff_minutes"])

            for measurement_col, feature_index in task_spec["measurement_indices"].items():
                measurement_value = row.get(measurement_col)
                if pd.notna(measurement_value):
                    sample_x[slot, feature_index] = float(measurement_value)

            for time_col, feature_index in task_spec["time_feature_indices"].items():
                sample_x[slot, feature_index] = float(row[time_col])

    def _fill_decoder_context(
        self,
        sample_y: np.ndarray,
        task_name: str,
        history_records: list[dict[str, object]],
    ) -> None:
        if self.label_length == 0 or not history_records:
            return

        task_spec = self.task_target_specs[task_name]
        decoder_context_records = history_records[max(0, len(history_records) - self.label_length) :]
        pad_count = self.label_length - len(decoder_context_records)

        for context_offset, row in enumerate(decoder_context_records):
            y_slot = pad_count + context_offset
            if float(row["interval_available"]) > 0.5 and pd.notna(row["time_diff_minutes"]):
                sample_y[y_slot, task_spec["delta_index"]] = float(row["time_diff_minutes"])
                sample_y[y_slot, task_spec["availability_index"]] = 1.0

    def _fill_future_targets(
        self,
        sample_y: np.ndarray,
        task_name: str,
        history_records: list[dict[str, object]],
        future_records: list[dict[str, object]],
    ) -> None:
        task_spec = self.task_target_specs[task_name]
        previous_timestamp: Optional[pd.Timestamp] = None

        if history_records:
            last_history_row = history_records[-1]
            previous_timestamp = pd.Timestamp(last_history_row[self.timestamp_col])

        for future_offset, row in enumerate(future_records[: self.prediction_length]):
            current_timestamp = pd.Timestamp(row[self.timestamp_col])
            y_slot = self.label_length + future_offset

            if previous_timestamp is not None:
                interval_minutes = (current_timestamp - previous_timestamp).total_seconds() / 60.0
                sample_y[y_slot, task_spec["delta_index"]] = float(interval_minutes)
                sample_y[y_slot, task_spec["availability_index"]] = 1.0

            previous_timestamp = current_timestamp

    def _build_split_samples(
        self,
        split: str,
        events_df: pd.DataFrame,
        segments_df: pd.DataFrame,
    ) -> dict[str, object]:
        if events_df.empty or segments_df.empty:
            return self._empty_split_samples()

        x_samples: list[np.ndarray] = []
        y_samples: list[np.ndarray] = []
        metadata_rows: list[dict[str, object]] = []

        for segment_id, segment in enumerate(segments_df.itertuples(index=False)):
            segment_df = events_df.iloc[int(segment.start_idx) : int(segment.end_idx)].reset_index(drop=True)
            if segment_df.empty:
                continue

            task_records_by_type, task_timestamps_by_type = self._build_segment_task_lookups(
                segment_df=segment_df
            )
            anchor_timestamps = self._segment_anchor_timestamps(segment_df=segment_df)
            if not anchor_timestamps:
                continue

            for anchor_step, anchor_timestamp in enumerate(anchor_timestamps):
                sample_x = self._initialize_input_sample()
                sample_y = self._initialize_target_sample()

                history_timestamps_by_type: dict[str, list[Optional[str]]] = {}
                last_observed_timestamps_by_type: dict[str, Optional[str]] = {}
                future_timestamps_by_type: dict[str, list[Optional[str]]] = {}

                for task_name in self.task_names:
                    history_records, future_records = self._slice_task_records(
                        task_records=task_records_by_type[task_name],
                        task_timestamps=task_timestamps_by_type[task_name],
                        anchor_timestamp=anchor_timestamp,
                    )

                    history_timestamps, last_observed_timestamp = self._history_timestamp_payload(
                        history_records=history_records,
                    )
                    future_timestamps = self._future_timestamp_payload(
                        future_records=future_records,
                    )

                    history_timestamps_by_type[task_name] = history_timestamps
                    last_observed_timestamps_by_type[task_name] = last_observed_timestamp
                    future_timestamps_by_type[task_name] = future_timestamps

                    self._fill_history_block(
                        sample_x=sample_x,
                        task_name=task_name,
                        history_records=history_records,
                    )
                    self._fill_decoder_context(
                        sample_y=sample_y,
                        task_name=task_name,
                        history_records=history_records,
                    )
                    self._fill_future_targets(
                        sample_y=sample_y,
                        task_name=task_name,
                        history_records=history_records,
                        future_records=future_records,
                    )

                metadata_rows.append(
                    {
                        "split": split,
                        "patient_id": str(segment.patient_id),
                        "segment_id": int(segment_id),
                        "anchor_step": int(anchor_step),
                        "anchor_index": int(anchor_step),
                        "anchor_timestamp": pd.Timestamp(anchor_timestamp),
                        "history_timestamps_by_type": json.dumps(history_timestamps_by_type),
                        "last_observed_timestamps_by_type": json.dumps(last_observed_timestamps_by_type),
                        "future_timestamps_by_type": json.dumps(future_timestamps_by_type),
                    }
                )
                x_samples.append(sample_x)
                y_samples.append(sample_y)

        if not x_samples:
            return self._empty_split_samples()

        x_array = np.stack(x_samples, axis=0).astype(np.float32)
        y_array = np.stack(y_samples, axis=0).astype(np.float32)
        x_mark_array = np.zeros(
            (len(x_samples), self.sequence_length, len(self.time_cols)),
            dtype=np.float32,
        )
        y_mark_array = np.zeros(
            (len(x_samples), self.target_sequence_length, len(self.time_cols)),
            dtype=np.float32,
        )

        return {
            "x": x_array,
            "y": y_array,
            "x_mark": x_mark_array,
            "y_mark": y_mark_array,
            "metadata": pd.DataFrame(metadata_rows, columns=self._metadata_columns()),
        }

    def _fit_input_scaler(self, train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        means = np.zeros(len(self.continuous_input_feature_indices), dtype=np.float32)
        scales = np.ones(len(self.continuous_input_feature_indices), dtype=np.float32)

        for position, feature_index in enumerate(self.continuous_input_feature_indices):
            valid_mask = train_x[:, :, feature_index] != self.input_padding_value
            valid_values = train_x[:, :, feature_index][valid_mask]
            if valid_values.size == 0:
                continue

            means[position] = float(valid_values.mean())
            std_value = float(valid_values.std())
            scales[position] = 1.0 if std_value < 1e-6 else std_value

        return means, scales

    def _fit_target_scaler(self, train_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        means = np.zeros(len(self.delta_target_indices), dtype=np.float32)
        scales = np.ones(len(self.delta_target_indices), dtype=np.float32)

        for position, (delta_index, availability_index) in enumerate(
            zip(self.delta_target_indices, self.availability_target_indices)
        ):
            valid_mask = train_y[:, :, availability_index] > 0.5
            valid_values = train_y[:, :, delta_index][valid_mask]
            if valid_values.size == 0:
                continue

            means[position] = float(valid_values.mean())
            std_value = float(valid_values.std())
            scales[position] = 1.0 if std_value < 1e-6 else std_value

        return means, scales

    def _apply_input_scaling(self, x_array: np.ndarray) -> np.ndarray:
        scaled_array = x_array.copy()
        for position, feature_index in enumerate(self.continuous_input_feature_indices):
            valid_mask = scaled_array[:, :, feature_index] != self.input_padding_value
            if not valid_mask.any():
                continue

            scaled_array[:, :, feature_index][valid_mask] = (
                scaled_array[:, :, feature_index][valid_mask] - self.input_scaler_mean[position]
            ) / self.input_scaler_scale[position]
        return scaled_array

    def _apply_target_scaling(self, y_array: np.ndarray) -> np.ndarray:
        scaled_array = y_array.copy()
        for position, (delta_index, availability_index) in enumerate(
            zip(self.delta_target_indices, self.availability_target_indices)
        ):
            valid_mask = scaled_array[:, :, availability_index] > 0.5
            if not valid_mask.any():
                continue

            scaled_array[:, :, delta_index][valid_mask] = (
                scaled_array[:, :, delta_index][valid_mask] - self.target_scaler_mean[position]
            ) / self.target_scaler_scale[position]
        return scaled_array

    def _apply_scaling_to_all_splits(self) -> None:
        for split_name in SPLITS:
            self.samples[split_name]["x"] = self._apply_input_scaling(
                x_array=self.samples[split_name]["x"]
            )
            self.samples[split_name]["y"] = self._apply_target_scaling(
                y_array=self.samples[split_name]["y"]
            )

    def get_split_arrays(self, split: str) -> dict[str, object]:
        assert split in SPLITS
        return self.samples[split]

    def get_split_metadata(self, split: str) -> pd.DataFrame:
        assert split in SPLITS
        return self.samples[split]["metadata"]

    def inverse_transform_target_deltas(self, data: np.ndarray) -> np.ndarray:
        array = np.asarray(data, dtype=np.float32)
        return array * self.target_scaler_scale.reshape(1, 1, -1) + self.target_scaler_mean.reshape(1, 1, -1)

    def length(self, split: str) -> int:
        assert split in SPLITS
        return int(self.samples[split]["x"].shape[0])


class RequestsDataset(Dataset):
    def __init__(
        self,
        request_time_series: RequestsTimeSeries,
        slice_start_points_dict: Optional[dict[str, list[int]]] = None,
        split: str = "train",
        sequence_length: int = 5,
        label_length: int = 0,
        prediction_length: int = 3,
    ) -> None:
        assert split in SPLITS
        if sequence_length != request_time_series.sequence_length:
            raise ValueError("Dataset sequence_length must match the precomputed time-series sequence_length.")
        if label_length != request_time_series.label_length:
            raise ValueError("Dataset label_length must match the precomputed time-series label_length.")
        if prediction_length != request_time_series.prediction_length:
            raise ValueError("Dataset prediction_length must match the precomputed time-series prediction_length.")
        if slice_start_points_dict is not None:
            raise ValueError("slice_start_points_dict is no longer supported for the event-based dataset.")

        self.split = split
        self.series = request_time_series
        self.sequence_length = sequence_length
        self.label_length = label_length
        self.prediction_length = prediction_length
        self._split_arrays = self.series.get_split_arrays(split=split)

    def __len__(self) -> int:
        return int(self._split_arrays["x"].shape[0])

    @property
    def slice_start_points(self) -> list[int]:
        return list(range(len(self)))

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_x = torch.from_numpy(self._split_arrays["x"][i]).float()
        seq_y = torch.from_numpy(self._split_arrays["y"][i]).float()
        seq_x_mark = torch.from_numpy(self._split_arrays["x_mark"][i]).float()
        seq_y_mark = torch.from_numpy(self._split_arrays["y_mark"][i]).float()
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return self.series.inverse_transform_target_deltas(data)

def build_parser():
    parser = argparse.ArgumentParser(
        prog="RequestsDataset",
        description="Create a requests dataset for training and evaluation of forecasting models.",
    )
    parser.add_argument("--use_saved_request_data", action="store_true", default=False)
    parser.add_argument("--preprocess_data", action="store_true", default=False)
    parser.add_argument("--save_data", action="store_true", default=False)

    parser.add_argument("--seq_len", type=int, default=5)
    parser.add_argument("--label_len", type=int, default=0)
    parser.add_argument("--pred_len", type=int, default=3)

    parser.add_argument(
        "--medications_orders_file",
        type=str,
        default="data/processed/medication_orders_annotated.csv",
    )
    parser.add_argument(
        "--blood_pressure_orders_file",
        type=str,
        default="data/processed/blood_pressure_orders_annotated.csv",
    )
    parser.add_argument(
        "--heart_rate_orders_file",
        type=str,
        default="data/processed/heart_rate_orders_annotated.csv",
    )
    parser.add_argument(
        "--respiratory_rate_orders_file",
        type=str,
        default="data/processed/respiratory_rate_orders_annotated.csv",
    )
    parser.add_argument(
        "--temperature_orders_file",
        type=str,
        default="data/processed/temperature_orders_annotated.csv",
    )
    parser.add_argument(
        "--oxygen_saturation_orders_file",
        type=str,
        default="data/processed/oxygen_saturation_orders_annotated.csv",
    )
    return parser
    
if __name__ == '__main__':
    """Performs execution delta of the process."""
    # Unit tests
    pStart = datetime.datetime.now()
    try:
        parser = build_parser()
        args = parser.parse_args()
        annotated_data_files = AnnotatedDataFiles(
            annotated_visits=None,
            annotated_admissions_discharges=None,
            annotated_medications=args.medications_orders_file,
            annotated_blood_pressure=args.blood_pressure_orders_file,
            annotated_heart_rate=args.heart_rate_orders_file,
            annotated_respiratory_rate=args.respiratory_rate_orders_file,
            annotated_temperature=args.temperature_orders_file,
            annotated_oxygen_saturation=args.oxygen_saturation_orders_file,
        )
        dataset_config = MedicalRequestDatasetConfig(annotated_data_files=annotated_data_files,
                                                     use_saved_request_data=args.use_saved_request_data,
                                                     preprocess_data=args.preprocess_data,
                                                     save_data=args.save_data)
        
        request_data_manager = RequestsDataManager(dataset_config=dataset_config)

        train_data_df, train_segments_df = request_data_manager.get_requests_training_data()
        val_data_df, val_segments_df = request_data_manager.get_requests_validation_data()
        test_data_df, test_segments_df = request_data_manager.get_requests_testing_data()

        time_series = RequestsTimeSeries(train_data_df=train_data_df,
                                        val_data_df=val_data_df,
                                        test_data_df=test_data_df,
                                        train_segments_df=train_segments_df,
                                        val_segments_df=val_segments_df,
                                        test_segments_df=test_segments_df,
                                        metadata=request_data_manager.metadata,
                                        sequence_length=args.seq_len,
                                        label_length=args.label_len,
                                        prediction_length=args.pred_len)
        

        data_module = DataModule(
            dataset_cls=RequestsDataset,
            dataset_kwargs={
                "request_time_series": time_series,
                "split": "test",
                "sequence_length": args.seq_len,
                "label_length": args.label_len,
                "prediction_length": args.pred_len,
            },
            batch_size=32,
            workers=4,
            collate_fun=None,
        )

        test_data_loader = data_module.test_dataloader()

        for i, batch in enumerate(test_data_loader):
            seq_x, seq_y, seq_x_mark, seq_y_mark = batch
            print(i)
            print(seq_x.size())
            print(seq_y.size())
            print(seq_x_mark.size())
            print(seq_y_mark.size())
        print(seq_x)
        print(seq_y)
        print(seq_x_mark)
        print(seq_y_mark)
        print("Process completed successfully.")

    except Exception as errorMainContext:
        print("Fail End Process: ", errorMainContext)
        traceback.print_exc()
    qStop = datetime.datetime.now()
    print("Execution time: " + str(qStop-pStart)) 