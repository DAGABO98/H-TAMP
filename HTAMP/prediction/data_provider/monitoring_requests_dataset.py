from __future__ import annotations

import argparse
import datetime
import json
import pickle
import re
import tempfile
from pathlib import Path
import traceback
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from HTAMP.planning.request_handler import GlobalRequestHandler
from HTAMP.prediction.configs.monitoring_request_config import (
    MonitoringRequestDatasetConfig,
    MonitoringRequestTrainingConfig,
    SUPPORTED_REQUEST_TASKS,
    TimeseriesModelConfig,
)
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
ENCOUNTER_ID_COLUMN = "encounter_id"
TIME_MARK_COLUMNS = ["month", "day", "weekday", "hour", "minute"]
TIME_FEATURE_SPECS = (
    ("month", 12.0, 1.0),
    ("day", 31.0, 1.0),
    ("weekday", 7.0, 0.0),
    ("hour", 24.0, 0.0),
    ("minute", 60.0, 0.0),
)
TIME_COLUMNS = [
    f"{component_name}_{trig_component}"
    for component_name, _, _ in TIME_FEATURE_SPECS
    for trig_component in ("sin", "cos")
]
SEGMENT_COLUMNS = ["patient_id", "encounter_id", "start_idx", "end_idx", "num_rows"]
SPLITS = ("train", "val", "test")
TASK_NAMES = tuple(SUPPORTED_REQUEST_TASKS)
TASK_TO_INDEX = {task_name: index for index, task_name in enumerate(TASK_NAMES)}
REQUESTS_TIME_SERIES_CACHE_VERSION = 4
REQUESTS_TIME_SERIES_CACHE_DIRNAME = "time_series_cache"
REQUEST_CACHE_FILENAMES = {
    "medication": "medications_extended.csv",
    "blood_pressure": "blood_pressure_extended.csv",
    "heart_rate": "heart_rate_extended.csv",
    "respiratory_rate": "respiratory_rate_extended.csv",
    "temperature": "temperature_extended.csv",
    "oxygen_saturation": "oxygen_saturation_extended.csv",
}
DATASET_CACHE_FILENAMES = (
    "metadata.json",
    "train_data.csv",
    "train_segments.csv",
    "val_data.csv",
    "val_segments.csv",
    "test_data.csv",
    "test_segments.csv",
)
REQUESTS_TIME_SERIES_IGNORED_CONFIG_FIELDS = (
    "preprocess_data",
    "save_data",
    "use_saved_request_data",
    "use_saved_time_series",
)
ENCOUNTER_ID_CANDIDATES = (
    ENCOUNTER_ID_COLUMN,
    "Patient Encounter CSN",
    "PAT_ENC_CSN_ID",
)
ADMISSION_START_CANDIDATES = (
    "HOSPITAL_ADMISSION",
    "Hospital Admission",
)
DISCHARGE_END_CANDIDATES = (
    "HOSPITAL_DISCHARGE",
    "Hospital Discharge",
)
DEFAULT_ADMISSIONS_DISCHARGES_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "processed" / "admissions_discharges.csv"
)


def _normalize_column_name(column_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(column_name).lower())


def _event_measurement_column(task_name: str, component: str) -> str:
    return f"{task_name}_{component}"


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _json_safe_value(nested_value)
            for key, nested_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    return value


def _normalize_identifier_series(series: pd.Series) -> pd.Series:
    normalized = series.where(pd.notna(series), pd.NA).astype(str).str.strip()
    normalized = normalized.str.replace(r"(?<=\d)\.0+$", "", regex=True)
    return normalized.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})


def _resolved_admissions_discharges_path(
    dataset_config: MonitoringRequestDatasetConfig,
) -> Path | None:
    configured_path = str(
        getattr(dataset_config.annotated_data_files, "annotated_admissions_discharges", "")
        or ""
    ).strip()
    if configured_path:
        return Path(configured_path)
    if DEFAULT_ADMISSIONS_DISCHARGES_PATH.exists():
        return DEFAULT_ADMISSIONS_DISCHARGES_PATH
    return None


def _path_signature(path_value: str | Path) -> dict[str, object]:
    path_str = str(path_value)
    if not path_str:
        return {"path": "", "exists": False}

    path = Path(path_str)
    path_signature: dict[str, object] = {
        "path": str(path),
        "exists": path.exists(),
    }
    if path.exists():
        stat_result = path.stat()
        path_signature["size"] = int(stat_result.st_size)
        path_signature["mtime_ns"] = int(stat_result.st_mtime_ns)
    return path_signature


def _requests_time_series_signature(
    dataset_config: MonitoringRequestDatasetConfig,
    sequence_length: int,
    label_length: int,
    prediction_length: int,
    source_kind: str,
) -> dict[str, object]:
    dataset_payload = dataset_config.to_dict()
    for field_name in REQUESTS_TIME_SERIES_IGNORED_CONFIG_FIELDS:
        dataset_payload.pop(field_name, None)

    source_files: dict[str, object]
    if source_kind == "dataset_files":
        dataset_dir = Path(dataset_config.dataset_dir)
        source_files = {
            "dataset_dir": str(dataset_dir),
            "dataset_files": {
                filename: _path_signature(dataset_dir / filename)
                for filename in DATASET_CACHE_FILENAMES
            },
        }
    elif source_kind == "request_files":
        request_dir = Path(dataset_config.request_dir)
        source_files = {
            "request_dir": str(request_dir),
            "request_files": {
                task_name: _path_signature(request_dir / REQUEST_CACHE_FILENAMES[task_name])
                for task_name in dataset_config.included_tasks
            },
        }
    elif source_kind == "annotated_data_files":
        annotated_files = dataset_payload.get("annotated_data_files", {})
        source_files = {
            "annotated_data_files": {
                str(field_name): _path_signature(path_value)
                for field_name, path_value in dict(annotated_files).items()
            }
        }
    else:
        raise ValueError(f"Unsupported request time-series source kind '{source_kind}'.")

    resolved_admissions_path = _resolved_admissions_discharges_path(
        dataset_config=dataset_config,
    )

    return {
        "dataset_config": _json_safe_value(dataset_payload),
        "sequence_length": int(sequence_length),
        "label_length": int(label_length),
        "prediction_length": int(prediction_length),
        "source_kind": source_kind,
        "source_files": source_files,
        "admissions_discharges_file": _path_signature(
            resolved_admissions_path if resolved_admissions_path is not None else ""
        ),
    }


def _request_cache_files_exist(
    request_dir: Path,
    task_names: list[str] | tuple[str, ...],
) -> bool:
    return all(
        (request_dir / REQUEST_CACHE_FILENAMES[task_name]).exists()
        for task_name in task_names
    )


def _dataset_cache_files_exist(dataset_dir: Path) -> bool:
    return all((dataset_dir / filename).exists() for filename in DATASET_CACHE_FILENAMES)


def _resolve_requests_time_series_source_kind(
    dataset_config: MonitoringRequestDatasetConfig,
) -> str:
    dataset_dir = Path(dataset_config.dataset_dir)
    request_dir = Path(dataset_config.request_dir)

    if not dataset_config.preprocess_data:
        return "dataset_files"

    if dataset_config.save_data and _dataset_cache_files_exist(dataset_dir):
        return "dataset_files"

    if _request_cache_files_exist(request_dir, dataset_config.included_tasks):
        return "request_files"

    return "annotated_data_files"


def _requests_time_series_cache_path(
    dataset_config: MonitoringRequestDatasetConfig,
    sequence_length: int,
    label_length: int,
    prediction_length: int,
) -> Path:
    cache_dir = Path(dataset_config.dataset_dir) / REQUESTS_TIME_SERIES_CACHE_DIRNAME
    return cache_dir / (
        f"requests_timeseries_seq{sequence_length}_label{label_length}_pred{prediction_length}.pkl"
    )

def _build_cyclical_time_features(timestamp_series: pd.Series) -> pd.DataFrame:
    timestamp_series = pd.to_datetime(timestamp_series, errors="coerce")
    time_components = {
        "month": timestamp_series.dt.month.astype(float),
        "day": timestamp_series.dt.day.astype(float),
        "weekday": timestamp_series.dt.weekday.astype(float),
        "hour": timestamp_series.dt.hour.astype(float),
        "minute": timestamp_series.dt.minute.astype(float),
    }
    feature_values: dict[str, np.ndarray] = {}

    for component_name, period, offset in TIME_FEATURE_SPECS:
        normalized_values = time_components[component_name] - offset
        radians = 2.0 * np.pi * normalized_values / period
        feature_values[f"{component_name}_sin"] = np.sin(radians).astype(np.float32)
        feature_values[f"{component_name}_cos"] = np.cos(radians).astype(np.float32)

    return pd.DataFrame(feature_values, index=timestamp_series.index)


def _ensure_cyclical_time_columns(
    df: pd.DataFrame,
    timestamp_col: str = TIMESTAMP_COLUMN,
) -> pd.DataFrame:
    if df.empty:
        return df

    missing_time_cols = [time_col for time_col in TIME_COLUMNS if time_col not in df.columns]
    if not missing_time_cols:
        return df

    cyclical_time_df = _build_cyclical_time_features(df[timestamp_col])
    enriched_df = df.copy()
    for time_col in missing_time_cols:
        enriched_df[time_col] = cyclical_time_df[time_col].astype(np.float32)
    return enriched_df


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
        dataset_config: MonitoringRequestDatasetConfig
    ) -> None:
        self.dataset_config = dataset_config
        self.dataset_dir = Path(self.dataset_config.dataset_dir)
        self.metadata: dict[str, object] = {}
        self._warned_missing_measurements: set[str] = set()
        self.admission_windows_df = self._load_admission_windows()

        if dataset_config.preprocess_data:
            print("Preprocessing request data...")
            self.dataset_dir.mkdir(parents=True, exist_ok=True)
            request_handler = self._build_request_handler()
            split_data = self._preprocess_requests_data(request_handler=request_handler)
            self._unpack_preprocessed_data(split_data=split_data)
            print("Finished preprocessing request data.")
            if dataset_config.save_data:
                print("Saving preprocessed request data to dataset directory...")
                self._save_dataframes()
                print("Finished saving preprocessed request data.")
        else:
            print("Loading preprocessed request data from dataset directory...")
            self._load_dataframes()
            print("Finished loading preprocessed request data.")

    @property
    def patient_id_col(self) -> str:
        return self.dataset_config.patient_id_col

    @property
    def task_names(self) -> list[str]:
        return list(self.dataset_config.included_tasks)

    @property
    def task_to_index(self) -> dict[str, int]:
        return {
            task_name: index
            for index, task_name in enumerate(self.task_names)
        }

    @property
    def time_cols(self) -> list[str]:
        return TIME_COLUMNS.copy()

    @property
    def event_measurement_cols(self) -> list[str]:
        return [
            feature_name
            for task_name in self.task_names
            for feature_name in TASK_EVENT_MEASUREMENT_COLUMNS.get(task_name, [])
        ]

    @property
    def event_cols(self) -> list[str]:
        return [
            self.patient_id_col,
            ENCOUNTER_ID_COLUMN,
            TIMESTAMP_COLUMN,
            "task_name",
            "task_index",
            *self.time_cols,
            "interval_available",
            "time_diff_minutes",
        ]

    def _load_admission_windows(self) -> pd.DataFrame:
        resolved_path = _resolved_admissions_discharges_path(
            dataset_config=self.dataset_config,
        )
        empty_windows = pd.DataFrame(
            columns=[self.patient_id_col, ENCOUNTER_ID_COLUMN, "admission_start", "discharge_end"]
        )
        if resolved_path is None or not resolved_path.exists():
            return empty_windows

        try:
            admissions_df = pd.read_csv(resolved_path)
        except Exception as read_error:
            print(
                f"Warning: could not load admissions/discharges data from {resolved_path}: "
                f"{read_error}"
            )
            return empty_windows

        patient_col = self._match_first_column(
            admissions_df.columns.tolist(),
            [self.patient_id_col, "MRN", "PAT_ID"],
        )
        admission_col = self._match_first_column(
            admissions_df.columns.tolist(),
            list(ADMISSION_START_CANDIDATES),
        )
        discharge_col = self._match_first_column(
            admissions_df.columns.tolist(),
            list(DISCHARGE_END_CANDIDATES),
        )
        encounter_col = self._match_first_column(
            admissions_df.columns.tolist(),
            list(ENCOUNTER_ID_CANDIDATES),
        )

        if patient_col is None or admission_col is None:
            print(
                f"Warning: admissions/discharges file at {resolved_path} is missing "
                "the required patient or admission timestamp columns."
            )
            return empty_windows

        windows_df = pd.DataFrame(
            {
                self.patient_id_col: _normalize_identifier_series(admissions_df[patient_col]),
                "admission_start": pd.to_datetime(admissions_df[admission_col], errors="coerce"),
                "discharge_end": (
                    pd.to_datetime(admissions_df[discharge_col], errors="coerce")
                    if discharge_col is not None
                    else pd.Series(pd.NaT, index=admissions_df.index)
                ),
            }
        )
        if encounter_col is not None:
            windows_df[ENCOUNTER_ID_COLUMN] = _normalize_identifier_series(
                admissions_df[encounter_col]
            )
        else:
            windows_df[ENCOUNTER_ID_COLUMN] = (
                windows_df.groupby(self.patient_id_col).cumcount().add(1).map(
                    lambda encounter_index: f"admission_{int(encounter_index):04d}"
                )
            )

        windows_df = windows_df.dropna(subset=[self.patient_id_col, "admission_start"]).copy()
        if windows_df.empty:
            return empty_windows

        windows_df["discharge_end"] = windows_df["discharge_end"].fillna(pd.Timestamp.max)
        windows_df = windows_df.sort_values(
            [self.patient_id_col, "admission_start", "discharge_end", ENCOUNTER_ID_COLUMN],
            kind="mergesort",
        ).reset_index(drop=True)
        return windows_df

    def _assign_encounter_ids_from_admissions(
        self,
        task_df: pd.DataFrame,
        time_col: str,
    ) -> pd.Series:
        assigned_encounters = pd.Series(pd.NA, index=task_df.index, dtype="object")
        if task_df.empty or self.admission_windows_df.empty:
            return assigned_encounters

        event_windows_df = task_df[[self.patient_id_col, time_col]].copy()
        event_windows_df["_event_index"] = event_windows_df.index
        event_windows_df[self.patient_id_col] = _normalize_identifier_series(
            event_windows_df[self.patient_id_col]
        )
        event_windows_df[time_col] = pd.to_datetime(event_windows_df[time_col], errors="coerce")
        event_windows_df = event_windows_df.dropna(
            subset=[self.patient_id_col, time_col]
        ).sort_values([time_col, self.patient_id_col], kind="mergesort")
        if event_windows_df.empty:
            return assigned_encounters

        admissions_df = self.admission_windows_df.sort_values(
            ["admission_start", self.patient_id_col],
            kind="mergesort",
        ).reset_index(drop=True)
        merged_df = pd.merge_asof(
            event_windows_df,
            admissions_df,
            left_on=time_col,
            right_on="admission_start",
            by=self.patient_id_col,
            direction="backward",
            allow_exact_matches=True,
        )
        valid_window_mask = merged_df[time_col].lt(merged_df["discharge_end"])
        merged_df.loc[~valid_window_mask, ENCOUNTER_ID_COLUMN] = pd.NA
        assigned_encounters.loc[merged_df["_event_index"]] = merged_df[
            ENCOUNTER_ID_COLUMN
        ].to_numpy()
        return assigned_encounters

    def _assign_task_encounter_ids(
        self,
        task_df: pd.DataFrame,
        time_col: str,
    ) -> pd.Series:
        assigned_encounters = pd.Series(pd.NA, index=task_df.index, dtype="object")
        encounter_col = self._match_first_column(
            task_df.columns.tolist(),
            list(ENCOUNTER_ID_CANDIDATES),
        )
        if encounter_col is not None:
            assigned_encounters = _normalize_identifier_series(task_df[encounter_col])

        if assigned_encounters.isna().any():
            admissions_encounters = self._assign_encounter_ids_from_admissions(
                task_df=task_df,
                time_col=time_col,
            )
            assigned_encounters = assigned_encounters.where(
                assigned_encounters.notna(),
                admissions_encounters,
            )

        return assigned_encounters

    def _build_request_handler(self) -> GlobalRequestHandler:
        return GlobalRequestHandler(
            annotated_data_files=self.dataset_config.annotated_data_files,
            request_dir=self.dataset_config.request_dir,
            start_date=self.dataset_config.start_date,
            end_date=self.dataset_config.end_date,
            use_saved_data=self.dataset_config.use_saved_request_data,
            included_tasks=self.task_names,
        )

    def _task_frames(self, request_handler: GlobalRequestHandler) -> dict[str, pd.DataFrame]:
        return {
            task_name: getattr(request_handler, handler_attr)
            for task_name, handler_attr in REQUEST_HANDLER_ATTRS.items()
            if task_name in self.task_names
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
        task_df[self.patient_id_col] = _normalize_identifier_series(task_df[self.patient_id_col])
        task_df[TIMESTAMP_COLUMN] = pd.to_datetime(task_df[time_col], errors="coerce")
        task_df = task_df.dropna(subset=[TIMESTAMP_COLUMN])
        task_df[ENCOUNTER_ID_COLUMN] = self._assign_task_encounter_ids(
            task_df=task_df,
            time_col=time_col,
        )
        task_df = task_df.sort_values(
            [self.patient_id_col, TIMESTAMP_COLUMN, ENCOUNTER_ID_COLUMN],
            kind="mergesort",
        ).reset_index(drop=True)
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

        event_df = task_df[[self.patient_id_col, ENCOUNTER_ID_COLUMN, TIMESTAMP_COLUMN]].copy()
        event_df["task_name"] = task_name
        event_df["task_index"] = self.task_to_index[task_name]
        event_df = pd.concat(
            [event_df, _build_cyclical_time_features(event_df[TIMESTAMP_COLUMN])],
            axis=1,
        )
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
            for task_name in self.task_names
        ]

        events_df = pd.concat(event_frames, axis=0, ignore_index=True)
        if events_df.empty:
            raise ValueError(
                "No medical requests were found for the configured date range. "
                "Please check the annotated files and date filters."
            )

        events_df = events_df.sort_values(
            [self.patient_id_col, TIMESTAMP_COLUMN, "task_index", ENCOUNTER_ID_COLUMN],
            kind="mergesort",
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

    def _collect_sorted_unique_weeks(
        self,
        patient_events: dict[str, pd.DataFrame],
    ) -> list[tuple[int, int]]:
        unique_weeks: set[tuple[int, int]] = set()

        for patient_df in patient_events.values():
            if patient_df.empty:
                continue
            iso_fields = self._derive_iso_week_fields(patient_df[TIMESTAMP_COLUMN])
            unique_weeks.update(
                (int(iso_year), int(iso_week))
                for iso_year, iso_week in zip(iso_fields["iso_year"], iso_fields["iso_week"])
            )

        return sorted(unique_weeks)

    def _resolve_test_week_set(
        self,
        patient_events: dict[str, pd.DataFrame],
    ) -> set[tuple[int, int]]:
        sorted_unique_weeks = self._collect_sorted_unique_weeks(patient_events=patient_events)
        return set(self.dataset_config.test_iso_weeks).intersection(sorted_unique_weeks)

    def _resolve_week_split_sets(
        self,
        patient_events: dict[str, pd.DataFrame],
    ) -> dict[str, set[tuple[int, int]]]:
        sorted_unique_weeks = self._collect_sorted_unique_weeks(patient_events=patient_events)
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

    def _resolve_random_patient_split_sets(
        self,
        patient_events: dict[str, pd.DataFrame],
        test_weeks: set[tuple[int, int]],
    ) -> dict[str, set[str]]:
        eligible_patient_ids: list[str] = []

        for patient_id, patient_df in patient_events.items():
            if patient_df.empty:
                continue
            iso_fields = self._derive_iso_week_fields(patient_df[TIMESTAMP_COLUMN])
            patient_weeks = {
                (int(iso_year), int(iso_week))
                for iso_year, iso_week in zip(iso_fields["iso_year"], iso_fields["iso_week"])
            }
            if any(week not in test_weeks for week in patient_weeks):
                eligible_patient_ids.append(str(patient_id))

        eligible_patient_ids = sorted(set(eligible_patient_ids))
        val_ratio = self._validation_ratio_over_non_test_weeks()
        val_patient_count = int(np.floor(len(eligible_patient_ids) * val_ratio))
        if val_ratio > 0.0 and val_patient_count == 0 and eligible_patient_ids:
            val_patient_count = 1
        val_patient_count = min(val_patient_count, len(eligible_patient_ids))

        val_patients: set[str] = set()
        if val_patient_count > 0:
            rng = np.random.default_rng(self.dataset_config.validation_split_seed)
            selected_indices = np.atleast_1d(
                rng.choice(len(eligible_patient_ids), size=val_patient_count, replace=False)
            )
            val_patients = {
                eligible_patient_ids[int(patient_index)]
                for patient_index in selected_indices.tolist()
            }

        train_patients = set(eligible_patient_ids) - val_patients
        return {
            "train": train_patients,
            "val": val_patients,
            "test": set(),
        }

    def _resolve_row_split(
        self,
        patient_id: str,
        week_key: tuple[int, int],
        split_week_sets: dict[str, set[tuple[int, int]]],
        split_patient_sets: dict[str, set[str]],
    ) -> Optional[str]:
        if week_key in split_week_sets["test"]:
            return "test"

        if self.dataset_config.validation_split_strategy == "random_patients":
            if patient_id in split_patient_sets["val"]:
                return "val"
            if patient_id in split_patient_sets["train"]:
                return "train"
            return None

        for split_name in ("train", "val"):
            if week_key in split_week_sets[split_name]:
                return split_name
        return None

    def _split_patient_series(
        self,
        patient_id: str,
        patient_df: pd.DataFrame,
        split_week_sets: dict[str, set[tuple[int, int]]],
        split_patient_sets: dict[str, set[str]],
    ) -> dict[str, list[pd.DataFrame]]:
        patient_with_iso = patient_df.copy()
        iso_fields = self._derive_iso_week_fields(patient_with_iso[TIMESTAMP_COLUMN])
        patient_with_iso["iso_year"] = iso_fields["iso_year"]
        patient_with_iso["iso_week"] = iso_fields["iso_week"]

        split_rows: dict[str, list[pd.DataFrame]] = {split: [] for split in SPLITS}
        current_split: Optional[str] = None
        run_start = 0

        resolved_splits = [
            self._resolve_row_split(
                patient_id=str(patient_id),
                week_key=(int(row.iso_year), int(row.iso_week)),
                split_week_sets=split_week_sets,
                split_patient_sets=split_patient_sets,
            )
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

    def _derive_split_week_sets(
        self,
        split_frames: dict[str, pd.DataFrame],
    ) -> dict[str, set[tuple[int, int]]]:
        split_week_sets: dict[str, set[tuple[int, int]]] = {split: set() for split in SPLITS}

        for split_name, split_df in split_frames.items():
            if split_df.empty:
                continue
            iso_fields = self._derive_iso_week_fields(split_df[TIMESTAMP_COLUMN])
            split_week_sets[split_name] = {
                (int(iso_year), int(iso_week))
                for iso_year, iso_week in zip(iso_fields["iso_year"], iso_fields["iso_week"])
            }

        return split_week_sets

    def _derive_split_patient_sets(
        self,
        split_segments_frames: dict[str, pd.DataFrame],
    ) -> dict[str, set[str]]:
        split_patient_sets: dict[str, set[str]] = {split: set() for split in SPLITS}

        for split_name, segments_df in split_segments_frames.items():
            if segments_df.empty:
                continue
            split_patient_sets[split_name] = {
                str(patient_id)
                for patient_id in segments_df["patient_id"].astype(str).tolist()
            }

        return split_patient_sets

    def _group_episode_frames(
        self,
        events_df: pd.DataFrame,
    ) -> list[tuple[str, str, pd.DataFrame]]:
        if events_df.empty:
            return []

        patient_frames = [
            (
                str(patient_id),
                "",
                patient_df.sort_values(TIMESTAMP_COLUMN, kind="mergesort").reset_index(drop=True),
            )
            for patient_id, patient_df in events_df.groupby(self.patient_id_col, sort=False)
        ]
        if ENCOUNTER_ID_COLUMN not in events_df.columns:
            return patient_frames

        encounter_series = _normalize_identifier_series(events_df[ENCOUNTER_ID_COLUMN])
        if encounter_series.notna().sum() == 0:
            return patient_frames

        episode_df = events_df.copy()
        episode_df[ENCOUNTER_ID_COLUMN] = encounter_series
        episode_df["_episode_group"] = episode_df[ENCOUNTER_ID_COLUMN].fillna("__unknown_encounter__")

        grouped_frames: list[tuple[str, str, pd.DataFrame]] = []
        for (patient_id, encounter_group), group_df in episode_df.groupby(
            [self.patient_id_col, "_episode_group"],
            sort=False,
            dropna=False,
        ):
            grouped_frames.append(
                (
                    str(patient_id),
                    "" if encounter_group == "__unknown_encounter__" else str(encounter_group),
                    group_df.drop(columns=["_episode_group"])
                    .sort_values(TIMESTAMP_COLUMN, kind="mergesort")
                    .reset_index(drop=True),
                )
            )

        return grouped_frames

    def _build_split_frames(
        self,
        events_df: pd.DataFrame,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        patient_events = {
            patient_id: patient_df.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
            for patient_id, patient_df in events_df.groupby(self.patient_id_col, sort=False)
        }
        split_strategy = self.dataset_config.validation_split_strategy
        split_week_sets = {split: set() for split in SPLITS}
        split_patient_sets = {split: set() for split in SPLITS}
        split_week_sets["test"] = self._resolve_test_week_set(patient_events=patient_events)
        if split_strategy == "random_patients":
            split_patient_sets = self._resolve_random_patient_split_sets(
                patient_events=patient_events,
                test_weeks=split_week_sets["test"],
            )
        else:
            split_week_sets = self._resolve_week_split_sets(patient_events=patient_events)
        split_frame_parts: dict[str, list[pd.DataFrame]] = {split: [] for split in SPLITS}
        split_segments: dict[str, list[dict[str, object]]] = {split: [] for split in SPLITS}
        split_offsets = {split: 0 for split in SPLITS}

        for patient_id, encounter_id, patient_df in self._group_episode_frames(events_df=events_df):
            patient_split_frames = self._split_patient_series(
                patient_id=str(patient_id),
                patient_df=patient_df,
                split_week_sets=split_week_sets,
                split_patient_sets=split_patient_sets,
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
                            "encounter_id": encounter_id,
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

        self.split_week_sets = self._derive_split_week_sets(split_frames=split_frames)
        self.split_patient_sets = self._derive_split_patient_sets(
            split_segments_frames=split_segments_frames
        )
        return split_frames, split_segments_frames

    def _build_metadata(self) -> dict[str, object]:
        return {
            "dataset_representation": "time_aligned_multivariate_request_intervals",
            "sample_axis_semantics": "unique_anchor_timestamp_per_patient_encounter_segment",
            "segment_axis_semantics": "contiguous_rows_within_split_and_patient_encounter",
            "sequence_axis_semantics": "per_task_history_depth",
            "patient_id_col": self.patient_id_col,
            "encounter_id_col": ENCOUNTER_ID_COLUMN,
            "timestamp_col": TIMESTAMP_COLUMN,
            "task_names": self.task_names,
            "task_to_index": self.task_to_index,
            "time_columns": self.time_cols,
            "event_measurement_columns": self.event_measurement_cols,
            "start_date": self.dataset_config.start_date,
            "end_date": self.dataset_config.end_date,
            "train_ratio": self.dataset_config.train_ratio,
            "val_ratio": self.dataset_config.val_ratio,
            "validation_split_strategy": self.dataset_config.validation_split_strategy,
            "validation_split_seed": self.dataset_config.validation_split_seed,
            "train_patient_count": len(getattr(self, "split_patient_sets", {}).get("train", set())),
            "val_patient_count": len(getattr(self, "split_patient_sets", {}).get("val", set())),
            "test_patient_count": len(getattr(self, "split_patient_sets", {}).get("test", set())),
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

    def _rebuild_loaded_split(
        self,
        split_df: pd.DataFrame,
        segments_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if split_df.empty:
            return self._empty_events_frame(), self._empty_segments_frame()

        rebuilt_parts: list[pd.DataFrame] = []
        rebuilt_segments: list[dict[str, object]] = []
        split_offset = 0

        base_segments_df = segments_df
        if base_segments_df.empty:
            base_segments_df = pd.DataFrame(
                [
                    {
                        "patient_id": "",
                        "encounter_id": "",
                        "start_idx": 0,
                        "end_idx": len(split_df),
                        "num_rows": len(split_df),
                    }
                ],
                columns=SEGMENT_COLUMNS,
            )

        for segment in base_segments_df.itertuples(index=False):
            segment_slice_df = split_df.iloc[int(segment.start_idx) : int(segment.end_idx)].copy()
            if segment_slice_df.empty:
                continue

            for patient_id, encounter_id, episode_df in self._group_episode_frames(
                events_df=segment_slice_df
            ):
                ordered_episode_df = episode_df.sort_values(
                    [TIMESTAMP_COLUMN, "task_index"],
                    kind="mergesort",
                ).reset_index(drop=True)
                start_idx = split_offset
                end_idx = start_idx + len(ordered_episode_df)
                split_offset = end_idx
                rebuilt_parts.append(ordered_episode_df[self.event_cols])
                rebuilt_segments.append(
                    {
                        "patient_id": patient_id,
                        "encounter_id": encounter_id,
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "num_rows": len(ordered_episode_df),
                    }
                )

        rebuilt_data_df = (
            pd.concat(rebuilt_parts, ignore_index=True)
            if rebuilt_parts
            else self._empty_events_frame()
        )
        rebuilt_segments_df = (
            pd.DataFrame(rebuilt_segments, columns=SEGMENT_COLUMNS)
            if rebuilt_segments
            else self._empty_segments_frame()
        )
        rebuilt_data_df = self._add_interval_features(
            split_df=rebuilt_data_df,
            segments_df=rebuilt_segments_df,
        )
        return rebuilt_data_df, rebuilt_segments_df

    def _load_split(self, split_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        data_df = pd.read_csv(self.dataset_dir / f"{split_name}_data.csv")
        segments_df = pd.read_csv(self.dataset_dir / f"{split_name}_segments.csv")
        patient_id_col = str(self.metadata.get("patient_id_col", self.patient_id_col))

        if not data_df.empty:
            data_df[TIMESTAMP_COLUMN] = pd.to_datetime(data_df[TIMESTAMP_COLUMN], errors="coerce")
            data_df[patient_id_col] = _normalize_identifier_series(data_df[patient_id_col])
            if ENCOUNTER_ID_COLUMN not in data_df.columns:
                data_df[ENCOUNTER_ID_COLUMN] = pd.NA
            else:
                data_df[ENCOUNTER_ID_COLUMN] = _normalize_identifier_series(
                    data_df[ENCOUNTER_ID_COLUMN]
                )
            if data_df[ENCOUNTER_ID_COLUMN].isna().any():
                admissions_encounters = self._assign_encounter_ids_from_admissions(
                    task_df=data_df,
                    time_col=TIMESTAMP_COLUMN,
                )
                data_df[ENCOUNTER_ID_COLUMN] = data_df[ENCOUNTER_ID_COLUMN].where(
                    data_df[ENCOUNTER_ID_COLUMN].notna(),
                    admissions_encounters,
                )
            data_df["task_name"] = data_df["task_name"].astype(str)
            data_df = _ensure_cyclical_time_columns(data_df, timestamp_col=TIMESTAMP_COLUMN)

        if not segments_df.empty:
            segments_df["patient_id"] = _normalize_identifier_series(segments_df["patient_id"])
            if "encounter_id" not in segments_df.columns:
                segments_df["encounter_id"] = ""
            else:
                segments_df["encounter_id"] = _normalize_identifier_series(
                    segments_df["encounter_id"]
                ).fillna("")

        if data_df.empty:
            return data_df, segments_df

        return self._rebuild_loaded_split(split_df=data_df, segments_df=segments_df)

    def _filter_loaded_split_to_selected_tasks(
        self,
        data_df: pd.DataFrame,
        segments_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if data_df.empty or segments_df.empty:
            return self._empty_events_frame(), self._empty_segments_frame()
        if ENCOUNTER_ID_COLUMN not in data_df.columns:
            data_df = data_df.copy()
            data_df[ENCOUNTER_ID_COLUMN] = pd.NA

        selected_task_names = set(self.task_names)
        filtered_parts: list[pd.DataFrame] = []
        filtered_segments: list[dict[str, object]] = []
        split_offset = 0

        for segment in segments_df.itertuples(index=False):
            segment_df = data_df.iloc[int(segment.start_idx) : int(segment.end_idx)].copy()
            if segment_df.empty:
                continue

            segment_df = segment_df[segment_df["task_name"].isin(selected_task_names)].copy()
            if segment_df.empty:
                continue

            segment_df["task_index"] = segment_df["task_name"].map(self.task_to_index).astype(int)
            segment_df = segment_df.sort_values(
                [TIMESTAMP_COLUMN, "task_index"]
            ).reset_index(drop=True)
            encounter_id = ""
            if ENCOUNTER_ID_COLUMN in segment_df.columns:
                unique_encounters = (
                    _normalize_identifier_series(segment_df[ENCOUNTER_ID_COLUMN]).dropna().unique().tolist()
                )
                if len(unique_encounters) == 1:
                    encounter_id = str(unique_encounters[0])
            if not encounter_id and hasattr(segment, "encounter_id") and not pd.isna(segment.encounter_id):
                encounter_id = str(segment.encounter_id)

            start_idx = split_offset
            end_idx = start_idx + len(segment_df)
            split_offset = end_idx

            filtered_parts.append(segment_df[self.event_cols])
            filtered_segments.append(
                {
                    "patient_id": str(segment.patient_id),
                    "encounter_id": encounter_id,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "num_rows": len(segment_df),
                }
            )

        filtered_data_df = (
            pd.concat(filtered_parts, ignore_index=True)
            if filtered_parts
            else self._empty_events_frame()
        )
        filtered_segments_df = (
            pd.DataFrame(filtered_segments, columns=SEGMENT_COLUMNS)
            if filtered_segments
            else self._empty_segments_frame()
        )
        return filtered_data_df, filtered_segments_df

    def _load_dataframes(self) -> None:
        with self._metadata_path().open("r", encoding="utf-8") as metadata_file:
            self.metadata = json.load(metadata_file)

        loaded_time_cols = list(self.metadata.get("time_columns", []))
        if not loaded_time_cols or loaded_time_cols == TIME_MARK_COLUMNS:
            self.metadata["time_columns"] = TIME_COLUMNS.copy()

        self.metadata["task_names"] = self.task_names
        self.metadata["task_to_index"] = self.task_to_index
        self.metadata["event_measurement_columns"] = self.event_measurement_cols

        train_data_df, train_segments_df = self._load_split("train")
        val_data_df, val_segments_df = self._load_split("val")
        test_data_df, test_segments_df = self._load_split("test")

        self.train_requests_df, self.train_segments_df = self._filter_loaded_split_to_selected_tasks(
            data_df=train_data_df,
            segments_df=train_segments_df,
        )
        self.val_requests_df, self.val_segments_df = self._filter_loaded_split_to_selected_tasks(
            data_df=val_data_df,
            segments_df=val_segments_df,
        )
        self.test_requests_df, self.test_segments_df = self._filter_loaded_split_to_selected_tasks(
            data_df=test_data_df,
            segments_df=test_segments_df,
        )

    def get_requests_training_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self.train_requests_df, self.train_segments_df

    def get_requests_validation_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self.val_requests_df, self.val_segments_df

    def get_requests_testing_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self.test_requests_df, self.test_segments_df


class MonitoringRequestsTimeSeries:
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
        loaded_time_cols = list(metadata.get("time_columns", TIME_COLUMNS))
        self.time_cols = TIME_COLUMNS.copy() if loaded_time_cols == TIME_MARK_COLUMNS else loaded_time_cols
        self.event_measurement_cols = list(metadata["event_measurement_columns"])

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
            "encounter_id",
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
            "metadata": pd.DataFrame(columns=self._metadata_columns()),
        }

    def _initialize_input_sample(self) -> np.ndarray:
        sample = np.zeros(
            (self.sequence_length, len(self.input_feature_cols)),
            dtype=np.float32,
        )
        if self.continuous_input_feature_indices:
            sample[:, self.continuous_input_feature_indices] = np.nan
        return sample

    def _initialize_target_sample(self) -> np.ndarray:
        sample = np.zeros(
            (self.target_sequence_length, len(self.target_cols)),
            dtype=np.float32,
        )
        if self.delta_target_indices:
            sample[:, self.delta_target_indices] = np.nan
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
                        "encounter_id": (
                            ""
                            if pd.isna(getattr(segment, "encounter_id", ""))
                            else str(segment.encounter_id)
                        ),
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

        return {
            "x": x_array,
            "y": y_array,
            "metadata": pd.DataFrame(metadata_rows, columns=self._metadata_columns()),
        }

    def _fit_input_scaler(self, train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        means = np.zeros(len(self.continuous_input_feature_indices), dtype=np.float32)
        scales = np.ones(len(self.continuous_input_feature_indices), dtype=np.float32)

        for position, feature_index in enumerate(self.continuous_input_feature_indices):
            valid_mask = np.isfinite(train_x[:, :, feature_index])
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
            valid_mask = (train_y[:, :, availability_index] > 0.5) & np.isfinite(
                train_y[:, :, delta_index]
            )
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
            valid_mask = np.isfinite(scaled_array[:, :, feature_index])
            if not valid_mask.any():
                scaled_array[:, :, feature_index] = 0.0
                continue

            scaled_array[:, :, feature_index][valid_mask] = (
                scaled_array[:, :, feature_index][valid_mask] - self.input_scaler_mean[position]
            ) / self.input_scaler_scale[position]
            scaled_array[:, :, feature_index][~valid_mask] = 0.0
        return scaled_array

    def _apply_target_scaling(self, y_array: np.ndarray) -> np.ndarray:
        scaled_array = y_array.copy()
        for position, (delta_index, availability_index) in enumerate(
            zip(self.delta_target_indices, self.availability_target_indices)
        ):
            valid_mask = (scaled_array[:, :, availability_index] > 0.5) & np.isfinite(
                scaled_array[:, :, delta_index]
            )
            if not valid_mask.any():
                scaled_array[:, :, delta_index] = 0.0
                continue

            scaled_array[:, :, delta_index][valid_mask] = (
                scaled_array[:, :, delta_index][valid_mask] - self.target_scaler_mean[position]
            ) / self.target_scaler_scale[position]
            scaled_array[:, :, delta_index][~valid_mask] = 0.0
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

    def _cache_payload(
        self,
        dataset_config: MonitoringRequestDatasetConfig,
    ) -> dict[str, object]:
        source_kind = _resolve_requests_time_series_source_kind(
            dataset_config=dataset_config,
        )
        return {
            "_version": REQUESTS_TIME_SERIES_CACHE_VERSION,
            "source_kind": source_kind,
            "signature": _requests_time_series_signature(
                dataset_config=dataset_config,
                sequence_length=self.sequence_length,
                label_length=self.label_length,
                prediction_length=self.prediction_length,
                source_kind=source_kind,
            ),
            "time_series": self,
        }

    def save_cache(self, dataset_config: MonitoringRequestDatasetConfig) -> Path:
        cache_path = _requests_time_series_cache_path(
            dataset_config=dataset_config,
            sequence_length=self.sequence_length,
            label_length=self.label_length,
            prediction_length=self.prediction_length,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            mode="wb",
            delete=False,
            dir=cache_path.parent,
            prefix=f"{cache_path.stem}_",
            suffix=".tmp",
        ) as temp_file:
            pickle.dump(
                self._cache_payload(dataset_config=dataset_config),
                temp_file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            temp_path = Path(temp_file.name)

        temp_path.replace(cache_path)
        print(f"Saved request time-series cache to {cache_path}.")
        return cache_path

    @classmethod
    def load_cache(
        cls,
        dataset_config: MonitoringRequestDatasetConfig,
        sequence_length: int,
        label_length: int,
        prediction_length: int,
    ) -> "MonitoringRequestsTimeSeries":
        cache_path = _requests_time_series_cache_path(
            dataset_config=dataset_config,
            sequence_length=sequence_length,
            label_length=label_length,
            prediction_length=prediction_length,
        )

        with cache_path.open("rb") as cache_file:
            payload = pickle.load(cache_file)

        if not isinstance(payload, dict):
            raise ValueError("Request time-series cache is corrupted.")
        if payload.get("_version") != REQUESTS_TIME_SERIES_CACHE_VERSION:
            raise ValueError("Request time-series cache version mismatch.")
        source_kind = str(payload.get("source_kind", "")).strip()
        if not source_kind:
            raise ValueError("Request time-series cache source kind is missing.")
        expected_signature = _requests_time_series_signature(
            dataset_config=dataset_config,
            sequence_length=sequence_length,
            label_length=label_length,
            prediction_length=prediction_length,
            source_kind=source_kind,
        )
        if payload.get("signature") != expected_signature:
            raise ValueError("Request time-series cache does not match the current configuration.")

        time_series = payload.get("time_series")
        if not isinstance(time_series, cls):
            raise ValueError("Request time-series cache payload is corrupted.")

        print(f"Loaded request time-series cache from {cache_path}.")
        return time_series


class MonitoringRequestsDataset(Dataset):
    def __init__(
        self,
        request_time_series: MonitoringRequestsTimeSeries,
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

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq_x = torch.from_numpy(self._split_arrays["x"][i]).float()
        seq_y = torch.from_numpy(self._split_arrays["y"][i]).float()
        return seq_x, seq_y

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return self.series.inverse_transform_target_deltas(data)


def build_request_time_series(
    dataset_config: MonitoringRequestDatasetConfig,
    model_config: TimeseriesModelConfig,
) -> MonitoringRequestsTimeSeries:
    if dataset_config.use_saved_time_series:
        print("Checking for cached request time series...")
        cache_path = _requests_time_series_cache_path(
            dataset_config=dataset_config,
            sequence_length=model_config.seq_len,
            label_length=model_config.label_len,
            prediction_length=model_config.pred_len,
        )
        if cache_path.exists():
            print(f"Found cached request time series at {cache_path}. Attempting to load...")
            try:
                time_series = MonitoringRequestsTimeSeries.load_cache(
                    dataset_config=dataset_config,
                    sequence_length=model_config.seq_len,
                    label_length=model_config.label_len,
                    prediction_length=model_config.pred_len,
                )
                model_config.sync_channel_dimensions(
                    num_input_channels=len(time_series.input_feature_cols),
                    num_output_channels=len(time_series.target_cols),
                )
                print("Successfully loaded cached request time series.")
                return time_series
            except Exception as cache_error:
                print(
                    f"Could not reuse cached request time series at {cache_path}: {cache_error}. "
                    "Rebuilding the cache."
                )

    print("Building request time series from source data...")
    request_data_manager = RequestsDataManager(dataset_config=dataset_config)
    print("Loading request data for all splits...")

    train_data_df, train_segments_df = request_data_manager.get_requests_training_data()
    val_data_df, val_segments_df = request_data_manager.get_requests_validation_data()
    test_data_df, test_segments_df = request_data_manager.get_requests_testing_data()

    time_series = MonitoringRequestsTimeSeries(
        train_data_df=train_data_df,
        val_data_df=val_data_df,
        test_data_df=test_data_df,
        train_segments_df=train_segments_df,
        val_segments_df=val_segments_df,
        test_segments_df=test_segments_df,
        metadata=request_data_manager.metadata,
        sequence_length=model_config.seq_len,
        label_length=model_config.label_len,
        prediction_length=model_config.pred_len,
    )
    model_config.sync_channel_dimensions(
        num_input_channels=len(time_series.input_feature_cols),
        num_output_channels=len(time_series.target_cols),
    )

    if dataset_config.save_data:
        time_series.save_cache(dataset_config=dataset_config)

    return time_series

def build_parser():
    parser = argparse.ArgumentParser(
        prog="RequestsDataset",
        description="Create a requests dataset for training and evaluation from a JSON config file.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to a JSON file containing 'dataset_config' and 'model_config'.",
    )
    return parser
    
if __name__ == '__main__':
    """Performs execution delta of the process."""
    # Unit tests
    pStart = datetime.datetime.now()
    try:
        parser = build_parser()
        args = parser.parse_args()
        training_config = MonitoringRequestTrainingConfig.from_json_file(args.config_path)
        dataset_config = training_config.dataset_config
        model_config = training_config.model_config
        
        time_series = build_request_time_series(
            dataset_config=dataset_config,
            model_config=model_config,
        )
        

        data_module = DataModule(
            dataset_cls=MonitoringRequestsDataset,
            dataset_kwargs={
                "request_time_series": time_series,
                "split": "test",
                "sequence_length": model_config.seq_len,
                "label_length": model_config.label_len,
                "prediction_length": model_config.pred_len,
            },
            batch_size=model_config.batch_size,
            workers=model_config.num_workers,
            collate_fun=None,
        )

        test_data_loader = data_module.test_dataloader()

        for i, batch in enumerate(test_data_loader):
            seq_x, seq_y = batch
            print(i)
            print(seq_x.size())
            print(seq_y.size())
        print(seq_x)
        print(seq_y)
        print("Process completed successfully.")

    except Exception as errorMainContext:
        print("Fail End Process: ", errorMainContext)
        traceback.print_exc()
    qStop = datetime.datetime.now()
    print("Execution time: " + str(qStop-pStart)) 
