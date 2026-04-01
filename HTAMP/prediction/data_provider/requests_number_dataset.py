from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from HTAMP.planning.request_handler import GlobalRequestHandler
from HTAMP.prediction.configs.request_number_config import MedicalRequestDatasetConfig

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


def _normalize_column_name(column_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(column_name).lower())


class RequestNumberDataManager:
    def __init__(
        self,
        dataset_config: MedicalRequestDatasetConfig,
        preprocess: bool = False,
        save_data: bool = False,
    ) -> None:
        self.dataset_config = dataset_config
        self.dataset_dir = Path(self.dataset_config.dataset_dir)
        self.metadata: dict[str, object] = {}
        self._warned_missing_measurements: set[str] = set()

        if preprocess:
            self.dataset_dir.mkdir(parents=True, exist_ok=True)
            request_handler = self._build_request_handler()
            split_data = self._preprocess_number_requests_data(request_handler=request_handler)
            self._unpack_preprocessed_data(split_data=split_data)
            if save_data:
                self._save_dataframes()
        else:
            self._load_dataframes()

    @property
    def patient_id_col(self) -> str:
        return self.dataset_config.patient_id_col

    @property
    def target_cols(self) -> list[str]:
        return list(TASK_SPECS.keys())

    @property
    def auxiliary_feature_cols(self) -> list[str]:
        feature_cols: list[str] = []
        for task_name, components in VITAL_OUTPUT_COMPONENTS.items():
            for component in components:
                min_col, max_col = self._measurement_feature_names(task_name=task_name, component=component)
                feature_cols.extend([min_col, max_col])
        return feature_cols

    @property
    def feature_cols(self) -> list[str]:
        return [*self.target_cols, *self.auxiliary_feature_cols]

    @property
    def time_cols(self) -> list[str]:
        return TIME_COLUMNS.copy()

    def _build_request_handler(self) -> GlobalRequestHandler:
        return GlobalRequestHandler(
            annotated_data_files=self.dataset_config.annotated_data_files,
            request_dir=self.dataset_config.request_dir,
            start_date=self.dataset_config.start_date,
            end_date=self.dataset_config.end_date,
            use_saved_data=self.dataset_config.use_saved_request_data,
        )

    def _freq(self) -> str:
        return f"{self.dataset_config.time_step_minutes}min"

    def _task_frames(self, request_handler: GlobalRequestHandler) -> dict[str, pd.DataFrame]:
        return {
            task_name: getattr(request_handler, handler_attr)
            for task_name, handler_attr in REQUEST_HANDLER_ATTRS.items()
        }

    def _empty_indexed_frame(self, columns: list[str]) -> pd.DataFrame:
        empty_index = pd.MultiIndex.from_arrays([[], []], names=[self.patient_id_col, TIMESTAMP_COLUMN])
        return pd.DataFrame(index=empty_index, columns=columns)

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

    def _measurement_feature_names(self, task_name: str, component: str) -> tuple[str, str]:
        prefix = task_name if component == "value" else f"{task_name}_{component}"
        return f"{prefix}_min_value", f"{prefix}_max_value"

    def _warn_missing_measurements(self, task_name: str) -> None:
        if task_name in self._warned_missing_measurements:
            return
        self._warned_missing_measurements.add(task_name)
        print(
            f"Warning: no measurement value column was detected for '{task_name}'. "
            "Its min/max auxiliary features will be missing."
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
        task_df[TIMESTAMP_COLUMN] = pd.to_datetime(task_df[time_col], errors="coerce").dt.floor(self._freq())
        task_df = task_df.dropna(subset=[TIMESTAMP_COLUMN])
        return task_df

    def _build_task_counts(
        self,
        df: pd.DataFrame,
        task_name: str,
        time_col: str,
    ) -> pd.DataFrame:
        if df.empty:
            return self._empty_indexed_frame(columns=[task_name])

        task_df = self._base_task_df(df=df[[self.patient_id_col, time_col]].copy(), time_col=time_col)
        return (
            task_df.groupby([self.patient_id_col, TIMESTAMP_COLUMN])
            .size()
            .rename(task_name)
            .to_frame()
        )

    def _build_task_measurement_features(
        self,
        df: pd.DataFrame,
        task_name: str,
        time_col: str,
    ) -> pd.DataFrame:
        feature_columns = [
            feature_name
            for component in VITAL_OUTPUT_COMPONENTS.get(task_name, [])
            for feature_name in self._measurement_feature_names(task_name=task_name, component=component)
        ]
        if not feature_columns:
            return self._empty_indexed_frame(columns=[])
        if df.empty:
            return self._empty_indexed_frame(columns=feature_columns)

        task_df = self._base_task_df(df=df, time_col=time_col)
        if task_df.empty:
            return self._empty_indexed_frame(columns=feature_columns)

        component_series = self._extract_measurement_components(task_name=task_name, task_df=task_df)
        grouped_index = (
            task_df.groupby([self.patient_id_col, TIMESTAMP_COLUMN])
            .size()
            .rename("__rows__")
            .to_frame()
            .drop(columns="__rows__")
        )

        for component in VITAL_OUTPUT_COMPONENTS.get(task_name, []):
            min_col, max_col = self._measurement_feature_names(task_name=task_name, component=component)
            if component not in component_series:
                grouped_index[min_col] = np.nan
                grouped_index[max_col] = np.nan
                continue

            stats = (
                task_df.assign(__value__=component_series[component])
                .groupby([self.patient_id_col, TIMESTAMP_COLUMN])["__value__"]
                .agg(["min", "max"])
                .rename(columns={"min": min_col, "max": max_col})
            )
            grouped_index = grouped_index.join(stats, how="left")

        return grouped_index[feature_columns]

    def _build_multivariate_counts(self, request_handler: GlobalRequestHandler) -> pd.DataFrame:
        task_frames = self._task_frames(request_handler=request_handler)
        indexed_frames: list[pd.DataFrame] = []

        for task_name, time_col in TASK_SPECS.items():
            indexed_frames.append(
                self._build_task_counts(
                    df=task_frames[task_name],
                    task_name=task_name,
                    time_col=time_col,
                )
            )
            indexed_frames.append(
                self._build_task_measurement_features(
                    df=task_frames[task_name],
                    task_name=task_name,
                    time_col=time_col,
                )
            )

        counts_df = pd.concat(indexed_frames, axis=1).reset_index()
        if counts_df.empty:
            raise ValueError(
                "No medical requests were found for the configured date range. "
                "Please check the annotated files and date filters."
            )

        for target_col in self.target_cols:
            if target_col not in counts_df.columns:
                counts_df[target_col] = 0
        for aux_col in self.auxiliary_feature_cols:
            if aux_col not in counts_df.columns:
                counts_df[aux_col] = np.nan

        counts_df[self.target_cols] = counts_df[self.target_cols].fillna(0).astype(np.int64)
        counts_df[self.auxiliary_feature_cols] = counts_df[self.auxiliary_feature_cols].astype(float)
        counts_df = counts_df.sort_values([self.patient_id_col, TIMESTAMP_COLUMN]).reset_index(drop=True)

        ordered_columns = [self.patient_id_col, TIMESTAMP_COLUMN, *self.feature_cols]
        return counts_df[ordered_columns]

    def _add_time_features(self, patient_df: pd.DataFrame, patient_id: str) -> pd.DataFrame:
        enriched_df = patient_df.copy()
        enriched_df[self.patient_id_col] = patient_id
        enriched_df["month"] = enriched_df[TIMESTAMP_COLUMN].dt.month.astype(np.int64)
        enriched_df["day"] = enriched_df[TIMESTAMP_COLUMN].dt.day.astype(np.int64)
        enriched_df["weekday"] = enriched_df[TIMESTAMP_COLUMN].dt.weekday.astype(np.int64)
        enriched_df["hour"] = enriched_df[TIMESTAMP_COLUMN].dt.hour.astype(np.int64)
        enriched_df["minute"] = enriched_df[TIMESTAMP_COLUMN].dt.minute.astype(np.int64)

        return enriched_df[
            [
                self.patient_id_col,
                TIMESTAMP_COLUMN,
                *self.time_cols,
                *self.feature_cols,
            ]
        ]

    def _build_patient_time_series(self, counts_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        patient_series: dict[str, pd.DataFrame] = {}

        for patient_id, patient_counts_df in counts_df.groupby(self.patient_id_col, sort=False):
            patient_counts_df = patient_counts_df.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
            start_time = patient_counts_df[TIMESTAMP_COLUMN].min()
            end_time = patient_counts_df[TIMESTAMP_COLUMN].max()

            reindexed_df = (
                patient_counts_df.set_index(TIMESTAMP_COLUMN)[self.feature_cols]
                .reindex(pd.date_range(start=start_time, end=end_time, freq=self._freq()))
                .reset_index()
                .rename(columns={"index": TIMESTAMP_COLUMN})
            )
            reindexed_df[self.target_cols] = reindexed_df[self.target_cols].fillna(0).astype(np.int64)
            reindexed_df[self.auxiliary_feature_cols] = reindexed_df[self.auxiliary_feature_cols].astype(float)

            patient_series[patient_id] = self._add_time_features(
                patient_df=reindexed_df,
                patient_id=patient_id,
            )

        return patient_series

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
        if remaining_ratio <= 0:
            return 0.0
        return float(self.dataset_config.val_ratio / remaining_ratio)

    def _resolve_week_split_sets(
        self,
        patient_series: dict[str, pd.DataFrame],
    ) -> dict[str, set[tuple[int, int]]]:
        unique_weeks: set[tuple[int, int]] = set()

        for patient_df in patient_series.values():
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

    def _empty_split_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                self.patient_id_col,
                TIMESTAMP_COLUMN,
                *self.time_cols,
                *self.feature_cols,
            ]
        )

    def _empty_segments_frame(self) -> pd.DataFrame:
        return pd.DataFrame(columns=SEGMENT_COLUMNS)

    def _build_split_frames(
        self,
        patient_series: dict[str, pd.DataFrame],
    ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        split_week_sets = self._resolve_week_split_sets(patient_series=patient_series)
        split_frame_parts: dict[str, list[pd.DataFrame]] = {split: [] for split in SPLITS}
        split_segments: dict[str, list[dict[str, object]]] = {split: [] for split in SPLITS}
        split_offsets = {split: 0 for split in SPLITS}

        for patient_id, patient_df in patient_series.items():
            patient_split_frames = self._split_patient_series_by_weeks(
                patient_df=patient_df,
                split_week_sets=split_week_sets,
            )

            for split_name, split_dfs in patient_split_frames.items():
                for split_df in split_dfs:
                    if split_df.empty:
                        continue

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
                else self._empty_split_frame()
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

        self.split_week_sets = split_week_sets

        return split_frames, split_segments_frames

    def _build_metadata(self) -> dict[str, object]:
        return {
            "patient_id_col": self.patient_id_col,
            "timestamp_col": TIMESTAMP_COLUMN,
            "task_columns": self.target_cols,
            "auxiliary_feature_columns": self.auxiliary_feature_cols,
            "feature_columns": self.feature_cols,
            "time_columns": self.time_cols,
            "time_step_minutes": self.dataset_config.time_step_minutes,
            "start_date": self.dataset_config.start_date,
            "end_date": self.dataset_config.end_date,
            "train_ratio": self.dataset_config.train_ratio,
            "val_ratio": self.dataset_config.val_ratio,
            "test_ratio": self.dataset_config.test_ratio,
            "train_iso_weeks": [list(week) for week in sorted(getattr(self, "split_week_sets", {}).get("train", set()))],
            "val_iso_weeks": [list(week) for week in sorted(getattr(self, "split_week_sets", {}).get("val", set()))],
            "test_iso_weeks": [list(week) for week in sorted(getattr(self, "split_week_sets", {}).get("test", set()))],
        }

    def _preprocess_number_requests_data(
        self,
        request_handler: GlobalRequestHandler,
    ) -> tuple[
        tuple[pd.DataFrame, pd.DataFrame],
        tuple[pd.DataFrame, pd.DataFrame],
        tuple[pd.DataFrame, pd.DataFrame],
        dict[str, object],
    ]:
        counts_df = self._build_multivariate_counts(request_handler=request_handler)
        patient_series = self._build_patient_time_series(counts_df=counts_df)
        split_frames, split_segment_frames = self._build_split_frames(patient_series=patient_series)
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

        self.train_number_of_requests_df, self.train_segments_df = train_data
        self.val_number_of_requests_df, self.val_segments_df = val_data
        self.test_number_of_requests_df, self.test_segments_df = test_data
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
        self._save_split("train", self.train_number_of_requests_df, self.train_segments_df)
        self._save_split("val", self.val_number_of_requests_df, self.val_segments_df)
        self._save_split("test", self.test_number_of_requests_df, self.test_segments_df)

        with self._metadata_path().open("w", encoding="utf-8") as metadata_file:
            json.dump(self.metadata, metadata_file, indent=2)

    def _load_split(self, split_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        data_df = pd.read_csv(self.dataset_dir / f"{split_name}_data.csv")
        segments_df = pd.read_csv(self.dataset_dir / f"{split_name}_segments.csv")
        patient_id_col = str(self.metadata.get("patient_id_col", self.patient_id_col))

        if not data_df.empty:
            data_df[TIMESTAMP_COLUMN] = pd.to_datetime(data_df[TIMESTAMP_COLUMN], errors="coerce")
            data_df[patient_id_col] = data_df[patient_id_col].astype(str)

        if not segments_df.empty:
            segments_df["patient_id"] = segments_df["patient_id"].astype(str)

        return data_df, segments_df

    def _load_dataframes(self) -> None:
        with self._metadata_path().open("r", encoding="utf-8") as metadata_file:
            self.metadata = json.load(metadata_file)

        self.train_number_of_requests_df, self.train_segments_df = self._load_split("train")
        self.val_number_of_requests_df, self.val_segments_df = self._load_split("val")
        self.test_number_of_requests_df, self.test_segments_df = self._load_split("test")

    def get_requests_numbers_training_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self.train_number_of_requests_df, self.train_segments_df

    def get_requests_numbers_validation_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self.val_number_of_requests_df, self.val_segments_df

    def get_requests_numbers_testing_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self.test_number_of_requests_df, self.test_segments_df


class RequestsNumberTimeSeries:
    def __init__(
        self,
        train_data_df: pd.DataFrame,
        val_data_df: pd.DataFrame,
        test_data_df: pd.DataFrame,
        train_segments_df: pd.DataFrame,
        val_segments_df: pd.DataFrame,
        test_segments_df: pd.DataFrame,
        metadata: dict[str, object],
    ) -> None:
        self.train_data_df = train_data_df
        self.val_data_df = val_data_df
        self.test_data_df = test_data_df
        self.train_segments_df = train_segments_df
        self.val_segments_df = val_segments_df
        self.test_segments_df = test_segments_df
        self.metadata = metadata

        self.patient_id_col = str(metadata["patient_id_col"])
        self.timestamp_col = str(metadata["timestamp_col"])
        self.time_cols = list(metadata["time_columns"])
        self.target_cols = list(metadata["task_columns"])
        self.auxiliary_feature_cols = list(metadata.get("auxiliary_feature_columns", []))
        self.feature_cols = list(metadata.get("feature_columns", self.target_cols))
        self.target_channel_indices = [self.feature_cols.index(column) for column in self.target_cols]

        self.feature_scaler = StandardScaler()
        self.feature_fill_values = self._compute_feature_fill_values(train_df=self.train_data_df)
        scaled_train_data, scaled_val_data, scaled_test_data = self.apply_feature_scaling_df(
            train_df=self.train_data_df,
            val_df=self.val_data_df,
            test_df=self.test_data_df,
        )

        self.target_scaler_mean = self.feature_scaler.mean_[self.target_channel_indices]
        self.target_scaler_scale = self.feature_scaler.scale_[self.target_channel_indices]

        self.scaled_train_data_df = self.apply_temporal_scaling_df(df=scaled_train_data)
        self.scaled_val_data_df = self.apply_temporal_scaling_df(df=scaled_val_data)
        self.scaled_test_data_df = self.apply_temporal_scaling_df(df=scaled_test_data)

    def _compute_feature_fill_values(self, train_df: pd.DataFrame) -> pd.Series:
        if train_df.empty:
            return pd.Series(0.0, index=self.feature_cols, dtype=float)

        fill_values = train_df[self.feature_cols].mean(numeric_only=True)
        fill_values = fill_values.reindex(self.feature_cols).fillna(0.0).astype(float)
        return fill_values

    def _fill_missing_feature_values(self, df: pd.DataFrame) -> pd.DataFrame:
        filled_df = df.copy(deep=True)
        if not filled_df.empty:
            filled_df[self.feature_cols] = filled_df[self.feature_cols].fillna(self.feature_fill_values.to_dict())
        return filled_df

    def apply_feature_scaling_df(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_scaled = self._fill_missing_feature_values(train_df)
        val_scaled = self._fill_missing_feature_values(val_df)
        test_scaled = self._fill_missing_feature_values(test_df)

        fit_frame = train_scaled[self.feature_cols]
        if fit_frame.empty:
            fit_frame = pd.DataFrame(np.zeros((1, len(self.feature_cols))), columns=self.feature_cols)

        self.feature_scaler.fit(fit_frame)

        if not train_scaled.empty:
            train_scaled[self.feature_cols] = self.feature_scaler.transform(train_scaled[self.feature_cols])
        if not val_scaled.empty:
            val_scaled[self.feature_cols] = self.feature_scaler.transform(val_scaled[self.feature_cols])
        if not test_scaled.empty:
            test_scaled[self.feature_cols] = self.feature_scaler.transform(test_scaled[self.feature_cols])

        return train_scaled, val_scaled, test_scaled

    def apply_temporal_scaling_df(self, df: pd.DataFrame) -> pd.DataFrame:
        scaled_df = df.copy(deep=True)
        time_bounds = {
            "month": (1.0, 12.0),
            "day": (1.0, 31.0),
            "weekday": (0.0, 6.0),
            "hour": (0.0, 23.0),
            "minute": (0.0, 59.0),
        }

        for column in self.time_cols:
            min_value, max_value = time_bounds[column]
            denominator = max(max_value - min_value, 1.0)
            scaled_df[column] = ((scaled_df[column] - min_value) / denominator) - 0.5

        return scaled_df

    def get_segments(self, split: str) -> pd.DataFrame:
        assert split in SPLITS
        return {
            "train": self.train_segments_df,
            "val": self.val_segments_df,
            "test": self.test_segments_df,
        }[split]

    def get_slice(self, split: str, start: int, stop: int) -> pd.DataFrame:
        assert split in SPLITS
        return {
            "train": self.train_data,
            "val": self.val_data,
            "test": self.test_data,
        }[split].iloc[start:stop]

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        array = np.asarray(data, dtype=np.float64)
        return array * self.target_scaler_scale + self.target_scaler_mean

    @property
    def train_data(self) -> pd.DataFrame:
        return self.scaled_train_data_df

    @property
    def val_data(self) -> pd.DataFrame:
        return self.scaled_val_data_df

    @property
    def test_data(self) -> pd.DataFrame:
        return self.scaled_test_data_df

    def length(self, split: str) -> int:
        return {
            "train": len(self.train_data),
            "val": len(self.val_data),
            "test": len(self.test_data),
        }[split]


class RequestsNumberDataset(Dataset):
    def __init__(
        self,
        request_time_series: RequestsNumberTimeSeries,
        slice_start_points_dict: Optional[dict[str, list[int]]] = None,
        split: str = "train",
        sequence_length: int = 60,
        label_length: int = 10,
        prediction_length: int = 60,
    ) -> None:
        assert split in SPLITS
        self.split = split
        self.series = request_time_series
        self.sequence_length = sequence_length
        self.label_length = label_length
        self.prediction_length = prediction_length

        if slice_start_points_dict is not None:
            self._slice_start_points = slice_start_points_dict[split]
        else:
            self._slice_start_points = self._build_slice_start_points()

    def _build_slice_start_points(self) -> list[int]:
        start_points: list[int] = []
        total_window = self.sequence_length + self.prediction_length
        split_segments_df = self.series.get_segments(split=self.split)

        for segment in split_segments_df.itertuples(index=False):
            max_start = int(segment.end_idx) - total_window
            if max_start < int(segment.start_idx):
                continue
            start_points.extend(range(int(segment.start_idx), max_start + 1))

        return start_points

    def __len__(self) -> int:
        return len(self._slice_start_points)

    @property
    def slice_start_points(self) -> list[int]:
        return list(self._slice_start_points)

    def _torch(self, *dfs: pd.DataFrame) -> tuple[torch.Tensor, ...]:
        return tuple(torch.from_numpy(df.to_numpy(dtype=np.float32)).float() for df in dfs)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        start = self._slice_start_points[i]
        stop = start + self.sequence_length + self.prediction_length

        series_slice = self.series.get_slice(
            split=self.split,
            start=start,
            stop=stop,
        )

        x_slice = series_slice.iloc[: self.sequence_length]
        y_slice = series_slice.iloc[self.sequence_length - self.label_length :]

        seq_x = x_slice[self.series.feature_cols]
        seq_x_mark = x_slice[self.series.time_cols]
        seq_y = y_slice[self.series.feature_cols]
        seq_y_mark = y_slice[self.series.time_cols]

        return self._torch(seq_x, seq_y, seq_x_mark, seq_y_mark)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return self.series.inverse_transform(data)
