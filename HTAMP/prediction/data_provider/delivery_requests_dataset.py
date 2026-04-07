from __future__ import annotations

import argparse
import datetime
import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from HTAMP.planning.request_handler import GlobalRequestHandler
from HTAMP.prediction.configs.delivery_request_config import (
    DeliveryRequestDatasetConfig,
    DeliveryRequestTrainingConfig,
)
from HTAMP.prediction.medication_mapping import (
    MedicationMappingApplier,
    resolve_medication_name_column,
)

SPLITS = ("train", "val", "test")
TIMESTAMP_COLUMN = "event_time"
ENCOUNTER_ID_COLUMN = "encounter_id"
WORKFLOW_IGNORED_CONFIG_FIELDS = (
    "preprocess_data",
    "save_data",
    "use_saved_request_data",
    "use_saved_dataset",
)
DATASET_VERSION = 2
NPZ_FILENAMES = {split: f"{split}.npz" for split in SPLITS}
METADATA_FILENAMES = {split: f"{split}_metadata.csv" for split in SPLITS}
TIMELINE_COLUMNS = ["patient_id", ENCOUNTER_ID_COLUMN, TIMESTAMP_COLUMN, "source_type"]
SEGMENT_COLUMNS = [
    "split",
    "patient_id",
    ENCOUNTER_ID_COLUMN,
    "segment_id",
    "start_time",
    "end_time",
    "num_rows",
]
VITAL_MEASUREMENT_COMPONENTS: dict[str, dict[str, list[str]]] = {
    "blood_pressure": {
        "systolic": ["Blood Pressure Systolic Value"],
        "diastolic": ["Blood Pressure Diastolic Value"],
        "combined": ["Blood Pressure Value"],
    },
    "heart_rate": {
        "value": ["Heart Rate Value"],
    },
    "respiratory_rate": {
        "value": ["Respiration Value"],
    },
    "temperature": {
        "value": ["Temperature Value"],
    },
    "oxygen_saturation": {
        "value": ["SP02 Value"],
    },
}
ADMISSION_START_CANDIDATES = ("HOSPITAL_ADMISSION", "Hospital Admission")
DISCHARGE_END_CANDIDATES = ("HOSPITAL_DISCHARGE", "Hospital Discharge")
ENCOUNTER_ID_CANDIDATES = (
    ENCOUNTER_ID_COLUMN,
    "Patient Encounter CSN",
    "PAT_ENC_CSN_ID",
)
DEFAULT_MEDICATION_CODE_CANDIDATES = (
    "Medication Generic Name",
    "Medication Name",
    "Medication",
    "Medication Display Name",
    "Medication Description",
    "Order Med Name",
    "Order Medication",
    "Drug Name",
    "Generic Name",
    "med_code",
    "medication_code",
    "order_med_id",
    "Order Med ID",
)


def _normalize_column_name(column_name: str) -> str:
    return "".join(character for character in str(column_name).lower() if character.isalnum())


def _normalize_identifier_series(series: pd.Series) -> pd.Series:
    normalized = series.where(pd.notna(series), pd.NA).astype(str).str.strip()
    normalized = normalized.str.replace(r"(?<=\d)\.0+$", "", regex=True)
    return normalized.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})


def _normalize_string_series(series: pd.Series) -> pd.Series:
    normalized = series.where(pd.notna(series), pd.NA).astype(str).str.strip()
    return normalized.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})


def _dataset_config_snapshot(dataset_config: DeliveryRequestDatasetConfig) -> dict[str, object]:
    payload = dataset_config.to_dict()
    for field_name in WORKFLOW_IGNORED_CONFIG_FIELDS:
        payload.pop(field_name, None)
    return payload


def _hours_between(later: pd.Timestamp, earlier: pd.Timestamp) -> float:
    return float((later - earlier).total_seconds() / 3600.0)


def _duration_to_bin(duration_hours: float, bins: Sequence[float]) -> int:
    if duration_hours < 0:
        raise ValueError(f"Duration must be non-negative, got {duration_hours}.")
    bin_index = int(np.searchsorted(np.asarray(bins, dtype=np.float32), duration_hours, side="left"))
    if bin_index >= len(bins):
        bin_index = len(bins) - 1
    return bin_index


def _build_med_code_display_map(
    med_vocab: Sequence[str],
    *frames: pd.DataFrame,
) -> dict[str, str]:
    display_map = {str(med_code): str(med_code) for med_code in med_vocab}
    valid_frames = [
        frame[["med_code", "med_display_name"]].copy()
        for frame in frames
        if not frame.empty and {"med_code", "med_display_name"}.issubset(frame.columns)
    ]
    if not valid_frames:
        return display_map

    combined_df = pd.concat(valid_frames, ignore_index=True)
    combined_df = combined_df.dropna(subset=["med_code", "med_display_name"]).copy()
    if combined_df.empty:
        return display_map

    combined_df["med_code"] = combined_df["med_code"].astype(str)
    combined_df["med_display_name"] = combined_df["med_display_name"].astype(str)
    combined_df = combined_df[
        combined_df["med_code"].isin(display_map)
        & combined_df["med_display_name"].str.strip().ne("")
    ]
    if combined_df.empty:
        return display_map

    grouped = combined_df.groupby(["med_code", "med_display_name"]).size().reset_index(name="count")
    grouped = grouped.sort_values(["med_code", "count", "med_display_name"], ascending=[True, False, True])
    best_names = grouped.drop_duplicates(subset=["med_code"], keep="first")
    display_map.update(
        {
            str(row.med_code): str(row.med_display_name)
            for row in best_names.itertuples(index=False)
        }
    )
    return display_map


def _match_first_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    normalized_columns = {_normalize_column_name(column): column for column in columns}
    for candidate in candidates:
        matched_column = normalized_columns.get(_normalize_column_name(candidate))
        if matched_column is not None:
            return matched_column
    return None


def _coerce_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    extracted = series.astype(str).str.extract(r"(-?\d+(?:\.\d+)?)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def _extract_blood_pressure_pair(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    extracted = series.astype(str).str.extract(
        r"(?P<systolic>-?\d+(?:\.\d+)?)\s*/\s*(?P<diastolic>-?\d+(?:\.\d+)?)"
    )
    systolic = pd.to_numeric(extracted["systolic"], errors="coerce")
    diastolic = pd.to_numeric(extracted["diastolic"], errors="coerce")
    return systolic, diastolic


def _build_vocab(
    values: Sequence[str],
    *,
    top_k: Optional[int] = None,
    min_count: int = 1,
) -> list[str]:
    series = pd.Series(list(values)).dropna().astype(str).str.strip()
    series = series[series != ""]
    if series.empty:
        return []
    counts = series.value_counts()
    counts = counts[counts >= min_count]
    if top_k is not None:
        counts = counts.iloc[:top_k]
    return counts.index.tolist()


def _vocab_index(vocab: Sequence[str]) -> dict[str, int]:
    return {item: index for index, item in enumerate(vocab)}


def _compute_vital_means(vitals_df: pd.DataFrame, vital_vocab: Sequence[str]) -> dict[str, float]:
    subset = vitals_df[vitals_df["vital_name"].isin(vital_vocab)].copy()
    grouped_means = subset.groupby("vital_name")["value"].mean().to_dict()
    global_mean = float(subset["value"].mean()) if not subset.empty else 0.0
    return {
        vital_name: float(grouped_means.get(vital_name, global_mean))
        for vital_name in vital_vocab
    }


def _compute_vital_stds(vitals_df: pd.DataFrame, vital_vocab: Sequence[str]) -> dict[str, float]:
    subset = vitals_df[vitals_df["vital_name"].isin(vital_vocab)].copy()
    grouped_stds = subset.groupby("vital_name")["value"].std(ddof=0).to_dict()
    global_std = float(subset["value"].std(ddof=0)) if len(subset) > 1 else 1.0
    if not np.isfinite(global_std) or global_std < 1e-6:
        global_std = 1.0
    return {
        vital_name: (
            float(grouped_stds[vital_name])
            if np.isfinite(grouped_stds.get(vital_name, np.nan)) and float(grouped_stds[vital_name]) >= 1e-6
            else global_std
        )
        for vital_name in vital_vocab
    }


def _collect_triggers(
    vitals_group: pd.DataFrame,
    admin_group: pd.DataFrame,
    orders_group: pd.DataFrame,
    *,
    include_order_triggers: bool,
) -> list[pd.Timestamp]:
    trigger_times = list(vitals_group[TIMESTAMP_COLUMN].tolist()) + list(admin_group[TIMESTAMP_COLUMN].tolist())
    if include_order_triggers:
        trigger_times += list(orders_group[TIMESTAMP_COLUMN].tolist())
    return sorted(set(pd.Timestamp(value) for value in trigger_times))


@dataclass
class DeliveryRequestExample:
    split: str
    patient_id: str
    encounter_id: str
    segment_id: int
    trigger_time: pd.Timestamp
    x: np.ndarray
    m: np.ndarray
    d: np.ndarray
    step_mask: np.ndarray
    meds: np.ndarray
    duration_hours: float
    duration_idx: int
    event: float
    next_med_targets: np.ndarray
    med_target_available: float
    true_med_labels: list[str]


def _encode_vital_window(
    *,
    vitals_group: pd.DataFrame,
    trigger_time: pd.Timestamp,
    vital_vocab: Sequence[str],
    vital_means: dict[str, float],
    lookback_hours: float,
    max_seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    start_time = trigger_time - pd.Timedelta(hours=lookback_hours)
    vital_to_index = _vocab_index(vital_vocab)

    window = vitals_group[
        (vitals_group[TIMESTAMP_COLUMN] <= trigger_time)
        & (vitals_group[TIMESTAMP_COLUMN] > start_time)
        & (vitals_group["vital_name"].isin(vital_vocab))
    ].sort_values(TIMESTAMP_COLUMN)
    if window.empty:
        n_features = len(vital_vocab)
        return (
            np.zeros((max_seq_len, n_features), dtype=np.float32),
            np.zeros((max_seq_len, n_features), dtype=np.float32),
            np.zeros((max_seq_len, n_features), dtype=np.float32),
            np.zeros((max_seq_len,), dtype=np.float32),
        )

    unique_times = window[TIMESTAMP_COLUMN].drop_duplicates().sort_values().tolist()[-max_seq_len:]
    time_to_pos = {timestamp: position for position, timestamp in enumerate(unique_times)}
    n_steps = len(unique_times)
    n_features = len(vital_vocab)

    x = np.zeros((n_steps, n_features), dtype=np.float32)
    m = np.zeros((n_steps, n_features), dtype=np.float32)

    trimmed = window[window[TIMESTAMP_COLUMN].isin(unique_times)]
    for _, row in trimmed.iterrows():
        time_pos = time_to_pos[pd.Timestamp(row[TIMESTAMP_COLUMN])]
        feature_index = vital_to_index[str(row["vital_name"])]
        x[time_pos, feature_index] = float(row["value"])
        m[time_pos, feature_index] = 1.0

    d = np.zeros((n_steps, n_features), dtype=np.float32)
    history_before = vitals_group[
        (vitals_group[TIMESTAMP_COLUMN] <= trigger_time)
        & (vitals_group["vital_name"].isin(vital_vocab))
    ].sort_values(TIMESTAMP_COLUMN)
    last_obs_time: dict[str, Optional[pd.Timestamp]] = {vital_name: None for vital_name in vital_vocab}
    before_first_time = unique_times[0]
    older = history_before[history_before[TIMESTAMP_COLUMN] < before_first_time]
    if not older.empty:
        older_last = older.groupby("vital_name")[TIMESTAMP_COLUMN].max().to_dict()
        for vital_name in vital_vocab:
            if vital_name in older_last:
                last_obs_time[vital_name] = pd.Timestamp(older_last[vital_name])

    for time_pos, timestamp in enumerate(unique_times):
        current_time = pd.Timestamp(timestamp)
        for vital_name in vital_vocab:
            previous_time = last_obs_time[vital_name]
            if previous_time is None:
                delta_h = min(lookback_hours, _hours_between(current_time, start_time))
            else:
                delta_h = _hours_between(current_time, previous_time)
            d[time_pos, vital_to_index[vital_name]] = max(0.0, float(delta_h))

        current_rows = trimmed[trimmed[TIMESTAMP_COLUMN] == current_time]
        for _, row in current_rows.iterrows():
            last_obs_time[str(row["vital_name"])] = current_time

    x_pad = np.zeros((max_seq_len, n_features), dtype=np.float32)
    m_pad = np.zeros((max_seq_len, n_features), dtype=np.float32)
    d_pad = np.zeros((max_seq_len, n_features), dtype=np.float32)
    step_mask = np.zeros((max_seq_len,), dtype=np.float32)
    x_pad[-n_steps:] = x
    m_pad[-n_steps:] = m
    d_pad[-n_steps:] = d
    step_mask[-n_steps:] = 1.0
    return x_pad, m_pad, d_pad, step_mask


def _encode_medication_state(
    *,
    admin_group: pd.DataFrame,
    trigger_time: pd.Timestamp,
    med_vocab: Sequence[str],
    med_lookback_hours: float,
    med_decay_hours: float,
) -> np.ndarray:
    med_to_index = _vocab_index(med_vocab)
    start_time = trigger_time - pd.Timedelta(hours=med_lookback_hours)
    history = admin_group[
        (admin_group[TIMESTAMP_COLUMN] <= trigger_time)
        & (admin_group[TIMESTAMP_COLUMN] > start_time)
        & (admin_group["med_code"].isin(med_vocab))
    ].sort_values(TIMESTAMP_COLUMN)

    vector = np.zeros((len(med_vocab),), dtype=np.float32)
    if history.empty:
        return vector

    last_times = history.groupby("med_code")[TIMESTAMP_COLUMN].max().to_dict()
    counts = history.groupby("med_code").size().to_dict()
    for med_code, last_time in last_times.items():
        med_index = med_to_index[str(med_code)]
        delta_h = max(0.0, _hours_between(trigger_time, pd.Timestamp(last_time)))
        recency = float(np.exp(-delta_h / max(1e-6, med_decay_hours)))
        count_term = float(np.log1p(counts.get(med_code, 1)))
        vector[med_index] = recency * (1.0 + 0.15 * count_term)
    return vector


def _seen_medications(
    orders_group: pd.DataFrame,
    admin_group: pd.DataFrame,
    trigger_time: pd.Timestamp,
) -> set[str]:
    seen_orders = set(
        orders_group.loc[orders_group[TIMESTAMP_COLUMN] <= trigger_time, "med_code"]
        .dropna()
        .astype(str)
        .tolist()
    )
    seen_admins = set(
        admin_group.loc[admin_group[TIMESTAMP_COLUMN] <= trigger_time, "med_code"]
        .dropna()
        .astype(str)
        .tolist()
    )
    return seen_orders | seen_admins


def _find_next_new_delivery_request(
    *,
    orders_group: pd.DataFrame,
    admin_group: pd.DataFrame,
    trigger_time: pd.Timestamp,
    med_vocab: Sequence[str],
    prediction_horizon_hours: float,
    observation_end: pd.Timestamp,
) -> dict[str, object]:
    seen_medications = _seen_medications(
        orders_group=orders_group,
        admin_group=admin_group,
        trigger_time=trigger_time,
    )
    future_orders = orders_group[orders_group[TIMESTAMP_COLUMN] > trigger_time].sort_values(TIMESTAMP_COLUMN)
    future_orders = future_orders[~future_orders["med_code"].astype(str).isin(seen_medications)]

    max_followup_h = max(
        0.0,
        min(prediction_horizon_hours, _hours_between(observation_end, trigger_time)),
    )
    med_targets = np.zeros((len(med_vocab),), dtype=np.float32)

    if future_orders.empty or max_followup_h <= 0.0:
        return {
            "event": 0,
            "duration_hours": float(max_followup_h),
            "next_med_targets": med_targets,
            "med_target_available": 0.0,
            "true_med_labels": [],
        }

    first_time = pd.Timestamp(future_orders[TIMESTAMP_COLUMN].iloc[0])
    duration_hours = max(0.0, _hours_between(first_time, trigger_time))
    if duration_hours > prediction_horizon_hours:
        return {
            "event": 0,
            "duration_hours": float(prediction_horizon_hours),
            "next_med_targets": med_targets,
            "med_target_available": 0.0,
            "true_med_labels": [],
        }

    med_to_index = _vocab_index(med_vocab)
    first_rows = future_orders[future_orders[TIMESTAMP_COLUMN] == first_time]
    true_med_labels = sorted(set(first_rows["med_code"].dropna().astype(str).tolist()))
    med_target_available = 0.0

    for med_code in true_med_labels:
        med_index = med_to_index.get(med_code)
        if med_index is not None:
            med_targets[med_index] = 1.0
            med_target_available = 1.0

    return {
        "event": 1,
        "duration_hours": float(duration_hours),
        "next_med_targets": med_targets,
        "med_target_available": med_target_available,
        "true_med_labels": true_med_labels,
    }


class DeliveryRequestsDataManager:
    def __init__(self, dataset_config: DeliveryRequestDatasetConfig) -> None:
        self.dataset_config = dataset_config
        self.dataset_dir = Path(dataset_config.dataset_dir)
        self.patient_id_col = dataset_config.patient_id_col
        self._warned_missing_measurements: set[str] = set()
        self.admission_windows_df = self._load_admission_windows()
        self.resolved_medication_code_col: Optional[str] = None
        self.medication_mapping_applier = MedicationMappingApplier.from_dataset_config(dataset_config)
        self.samples: dict[str, dict[str, object]] = {}
        self.metadata: dict[str, object] = {}

        if dataset_config.use_saved_dataset:
            try:
                self._load_dataset()
                return
            except Exception as load_error:
                if not dataset_config.preprocess_data:
                    raise ValueError(
                        "use_saved_dataset=True was requested, but the saved delivery-request dataset "
                        "could not be loaded."
                    ) from load_error

        if dataset_config.preprocess_data:
            self.dataset_dir.mkdir(parents=True, exist_ok=True)
            request_handler = self._build_request_handler()
            split_samples, metadata = self._preprocess_from_request_handler(request_handler=request_handler)
            self.samples = split_samples
            self.metadata = metadata
            if dataset_config.save_data:
                self._save_dataset()
        else:
            self._load_dataset()

    def _load_admission_windows(self) -> pd.DataFrame:
        configured_path = str(
            getattr(self.dataset_config.annotated_data_files, "annotated_admissions_discharges", "")
            or ""
        ).strip()
        resolved_path = Path(configured_path) if configured_path else None
        if resolved_path is None or not resolved_path.exists():
            return pd.DataFrame(
                columns=[self.patient_id_col, ENCOUNTER_ID_COLUMN, "admission_start", "discharge_end"]
            )

        admissions_df = pd.read_csv(resolved_path)
        patient_col = _match_first_column(
            admissions_df.columns.tolist(),
            [self.patient_id_col, "MRN", "PAT_ID"],
        )
        admission_col = _match_first_column(
            admissions_df.columns.tolist(),
            list(ADMISSION_START_CANDIDATES),
        )
        discharge_col = _match_first_column(
            admissions_df.columns.tolist(),
            list(DISCHARGE_END_CANDIDATES),
        )
        encounter_col = _match_first_column(
            admissions_df.columns.tolist(),
            list(ENCOUNTER_ID_CANDIDATES),
        )
        if patient_col is None or admission_col is None:
            return pd.DataFrame(
                columns=[self.patient_id_col, ENCOUNTER_ID_COLUMN, "admission_start", "discharge_end"]
            )

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
            windows_df[ENCOUNTER_ID_COLUMN] = _normalize_identifier_series(admissions_df[encounter_col])
        else:
            windows_df[ENCOUNTER_ID_COLUMN] = (
                windows_df.groupby(self.patient_id_col).cumcount().add(1).map(
                    lambda encounter_index: f"admission_{int(encounter_index):04d}"
                )
            )

        windows_df = windows_df.dropna(subset=[self.patient_id_col, "admission_start"]).copy()
        if windows_df.empty:
            return pd.DataFrame(
                columns=[self.patient_id_col, ENCOUNTER_ID_COLUMN, "admission_start", "discharge_end"]
            )

        windows_df["discharge_end"] = windows_df["discharge_end"].fillna(pd.Timestamp.max)
        return windows_df.sort_values(
            [self.patient_id_col, "admission_start", "discharge_end", ENCOUNTER_ID_COLUMN],
            kind="mergesort",
        ).reset_index(drop=True)

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
        event_windows_df[self.patient_id_col] = _normalize_identifier_series(event_windows_df[self.patient_id_col])
        event_windows_df[time_col] = pd.to_datetime(event_windows_df[time_col], errors="coerce")
        event_windows_df = event_windows_df.dropna(subset=[self.patient_id_col, time_col]).sort_values(
            [time_col, self.patient_id_col],
            kind="mergesort",
        )
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
        assigned_encounters.loc[merged_df["_event_index"]] = merged_df[ENCOUNTER_ID_COLUMN].to_numpy()
        return assigned_encounters

    def _assign_task_encounter_ids(self, task_df: pd.DataFrame, time_col: str) -> pd.Series:
        encounter_col = _match_first_column(task_df.columns.tolist(), list(ENCOUNTER_ID_CANDIDATES))
        if encounter_col is not None:
            assigned_encounters = _normalize_identifier_series(task_df[encounter_col])
        else:
            assigned_encounters = pd.Series(pd.NA, index=task_df.index, dtype="object")

        if assigned_encounters.isna().any():
            admissions_encounters = self._assign_encounter_ids_from_admissions(task_df=task_df, time_col=time_col)
            assigned_encounters = assigned_encounters.where(assigned_encounters.notna(), admissions_encounters)
        return assigned_encounters

    def _prepare_base_event_df(self, df: pd.DataFrame, time_col: str) -> pd.DataFrame:
        task_df = df.copy()
        task_df = task_df.dropna(subset=[self.patient_id_col, time_col])
        task_df[self.patient_id_col] = _normalize_identifier_series(task_df[self.patient_id_col])
        task_df[TIMESTAMP_COLUMN] = pd.to_datetime(task_df[time_col], errors="coerce")
        task_df = task_df.dropna(subset=[TIMESTAMP_COLUMN])
        task_df[ENCOUNTER_ID_COLUMN] = self._assign_task_encounter_ids(task_df=task_df, time_col=time_col)
        return task_df.sort_values(
            [self.patient_id_col, TIMESTAMP_COLUMN, ENCOUNTER_ID_COLUMN],
            kind="mergesort",
        ).reset_index(drop=True)

    def _warn_missing_measurements(self, task_name: str) -> None:
        if task_name in self._warned_missing_measurements:
            return
        self._warned_missing_measurements.add(task_name)
        print(
            f"Warning: no measurement value column was detected for '{task_name}'. "
            "That task will not contribute delivery-request context features."
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
            systolic_col = _match_first_column(columns, candidates["systolic"])
            diastolic_col = _match_first_column(columns, candidates["diastolic"])
            if systolic_col is not None and diastolic_col is not None:
                return {
                    "systolic": _coerce_numeric_series(task_df[systolic_col]),
                    "diastolic": _coerce_numeric_series(task_df[diastolic_col]),
                }

            combined_col = _match_first_column(columns, candidates["combined"])
            if combined_col is not None:
                systolic, diastolic = _extract_blood_pressure_pair(task_df[combined_col])
                return {
                    "systolic": systolic,
                    "diastolic": diastolic,
                }

            self._warn_missing_measurements(task_name=task_name)
            return {}

        matched_col = _match_first_column(columns, candidates["value"])
        if matched_col is None:
            self._warn_missing_measurements(task_name=task_name)
            return {}
        return {"value": _coerce_numeric_series(task_df[matched_col])}

    def _build_request_handler(self) -> GlobalRequestHandler:
        included_tasks = tuple(
            dict.fromkeys(["medication", *self.dataset_config.included_tasks]).keys()
        )
        return GlobalRequestHandler(
            annotated_data_files=self.dataset_config.annotated_data_files,
            request_dir=self.dataset_config.request_dir,
            start_date=self.dataset_config.start_date,
            end_date=self.dataset_config.end_date,
            use_saved_data=self.dataset_config.use_saved_request_data,
            included_tasks=included_tasks,
        )

    def _resolve_vital_time_col(self, task_df: pd.DataFrame) -> str:
        if (
            self.dataset_config.use_admin_as_vital_time
            and "Administered DTTM" in task_df.columns
            and pd.to_datetime(task_df["Administered DTTM"], errors="coerce").notna().any()
        ):
            return "Administered DTTM"
        return "Scheduled DTTM"

    def _build_vitals_frame(self, request_handler: GlobalRequestHandler) -> pd.DataFrame:
        vital_rows: list[pd.DataFrame] = []
        handler_attrs = {
            "blood_pressure": "bp_df",
            "heart_rate": "hr_df",
            "respiratory_rate": "rr_df",
            "temperature": "temp_df",
            "oxygen_saturation": "os_df",
        }

        for task_name in self.dataset_config.included_tasks:
            task_df = getattr(request_handler, handler_attrs[task_name])
            if task_df.empty:
                continue
            time_col = self._resolve_vital_time_col(task_df=task_df)
            base_df = self._prepare_base_event_df(df=task_df, time_col=time_col)
            component_map = self._extract_measurement_components(task_name=task_name, task_df=base_df)
            for component_name, value_series in component_map.items():
                component_df = base_df[[self.patient_id_col, ENCOUNTER_ID_COLUMN, TIMESTAMP_COLUMN]].copy()
                component_df["vital_name"] = (
                    task_name if component_name == "value" else f"{task_name}_{component_name}"
                )
                component_df["value"] = pd.to_numeric(value_series, errors="coerce")
                component_df = component_df.dropna(subset=["value"]).copy()
                if component_df.empty:
                    continue
                component_df.rename(columns={self.patient_id_col: "patient_id"}, inplace=True)
                vital_rows.append(component_df)

        if not vital_rows:
            return pd.DataFrame(columns=["patient_id", ENCOUNTER_ID_COLUMN, TIMESTAMP_COLUMN, "vital_name", "value"])
        return pd.concat(vital_rows, ignore_index=True).sort_values(
            ["patient_id", TIMESTAMP_COLUMN, "vital_name", ENCOUNTER_ID_COLUMN],
            kind="mergesort",
        ).reset_index(drop=True)

    def _resolve_medication_code_col(self, med_df: pd.DataFrame) -> str:
        return resolve_medication_name_column(
            columns=med_df.columns.tolist(),
            explicit_col=self.dataset_config.medication_code_col,
        )

    def _build_medication_frames(
        self,
        request_handler: GlobalRequestHandler,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        med_df = request_handler.med_df
        if med_df.empty:
            empty_df = pd.DataFrame(
                columns=[
                    "patient_id",
                    ENCOUNTER_ID_COLUMN,
                    TIMESTAMP_COLUMN,
                    "med_code",
                    "med_code_type",
                    "med_display_name",
                ]
            )
            return empty_df, empty_df

        self.resolved_medication_code_col = self._resolve_medication_code_col(med_df=med_df)

        order_df = self._prepare_base_event_df(df=med_df, time_col="Medication Order DTTM")
        order_df = order_df.rename(columns={self.patient_id_col: "patient_id"})
        order_df = self.medication_mapping_applier.apply(
            order_df,
            med_name_col=self.resolved_medication_code_col,
        )
        order_df = order_df.dropna(subset=["med_code"]).copy()
        order_df = order_df[
            [
                "patient_id",
                ENCOUNTER_ID_COLUMN,
                TIMESTAMP_COLUMN,
                "med_code",
                "med_code_type",
                "med_display_name",
            ]
        ]

        admin_df = self._prepare_base_event_df(df=med_df, time_col="Administered DTTM")
        admin_df = admin_df.rename(columns={self.patient_id_col: "patient_id"})
        admin_df = self.medication_mapping_applier.apply(
            admin_df,
            med_name_col=self.resolved_medication_code_col,
        )
        admin_df = admin_df.dropna(subset=["med_code"]).copy()
        admin_df = admin_df[
            [
                "patient_id",
                ENCOUNTER_ID_COLUMN,
                TIMESTAMP_COLUMN,
                "med_code",
                "med_code_type",
                "med_display_name",
            ]
        ]

        order_df = order_df.sort_values(
            ["patient_id", TIMESTAMP_COLUMN, "med_code", ENCOUNTER_ID_COLUMN],
            kind="mergesort",
        ).reset_index(drop=True)
        admin_df = admin_df.sort_values(
            ["patient_id", TIMESTAMP_COLUMN, "med_code", ENCOUNTER_ID_COLUMN],
            kind="mergesort",
        ).reset_index(drop=True)
        return admin_df, order_df

    def _empty_timeline_frame(self) -> pd.DataFrame:
        return pd.DataFrame(columns=TIMELINE_COLUMNS)

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

    def _collect_sorted_unique_weeks(self, patient_events: dict[str, pd.DataFrame]) -> list[tuple[int, int]]:
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

    def _resolve_test_week_set(self, patient_events: dict[str, pd.DataFrame]) -> set[tuple[int, int]]:
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
        return {"train": train_weeks, "val": val_weeks, "test": test_weeks}

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
        return {"train": train_patients, "val": val_patients, "test": set()}

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

    def _group_episode_frames(
        self,
        timeline_df: pd.DataFrame,
    ) -> list[tuple[str, str, pd.DataFrame]]:
        if timeline_df.empty:
            return []

        encounter_series = _normalize_identifier_series(timeline_df[ENCOUNTER_ID_COLUMN])
        if encounter_series.notna().sum() == 0:
            return [
                (
                    str(patient_id),
                    "",
                    patient_df.sort_values(TIMESTAMP_COLUMN, kind="mergesort").reset_index(drop=True),
                )
                for patient_id, patient_df in timeline_df.groupby("patient_id", sort=False)
            ]

        grouped_timeline_df = timeline_df.copy()
        grouped_timeline_df[ENCOUNTER_ID_COLUMN] = encounter_series
        grouped_timeline_df["_episode_group"] = grouped_timeline_df[ENCOUNTER_ID_COLUMN].fillna(
            "__unknown_encounter__"
        )

        grouped_frames: list[tuple[str, str, pd.DataFrame]] = []
        for (patient_id, encounter_group), group_df in grouped_timeline_df.groupby(
            ["patient_id", "_episode_group"],
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

    def _derive_split_week_sets(self, split_frames: dict[str, pd.DataFrame]) -> dict[str, set[tuple[int, int]]]:
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

    def _derive_split_patient_sets(self, segments_df: pd.DataFrame) -> dict[str, set[str]]:
        split_patient_sets: dict[str, set[str]] = {split: set() for split in SPLITS}
        if segments_df.empty:
            return split_patient_sets
        for split_name in SPLITS:
            split_segments_df = segments_df[segments_df["split"] == split_name]
            split_patient_sets[split_name] = set(split_segments_df["patient_id"].astype(str).tolist())
        return split_patient_sets

    def _build_split_segments(
        self,
        timeline_df: pd.DataFrame,
    ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        patient_events = {
            patient_id: patient_df.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
            for patient_id, patient_df in timeline_df.groupby("patient_id", sort=False)
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
        segment_rows: list[dict[str, object]] = []
        segment_id = 0

        for patient_id, encounter_id, patient_df in self._group_episode_frames(timeline_df=timeline_df):
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
                    ordered_df = split_df.sort_values(TIMESTAMP_COLUMN, kind="mergesort").reset_index(drop=True)
                    split_frame_parts[split_name].append(ordered_df)
                    segment_rows.append(
                        {
                            "split": split_name,
                            "patient_id": patient_id,
                            ENCOUNTER_ID_COLUMN: encounter_id,
                            "segment_id": int(segment_id),
                            "start_time": pd.Timestamp(ordered_df[TIMESTAMP_COLUMN].iloc[0]),
                            "end_time": pd.Timestamp(ordered_df[TIMESTAMP_COLUMN].iloc[-1]),
                            "num_rows": int(len(ordered_df)),
                        }
                    )
                    segment_id += 1

        split_frames = {
            split_name: (
                pd.concat(frame_parts, ignore_index=True)
                if frame_parts
                else self._empty_timeline_frame()
            )
            for split_name, frame_parts in split_frame_parts.items()
        }
        segments_df = (
            pd.DataFrame(segment_rows, columns=SEGMENT_COLUMNS)
            if segment_rows
            else pd.DataFrame(columns=SEGMENT_COLUMNS)
        )
        self.split_week_sets = self._derive_split_week_sets(split_frames=split_frames)
        self.split_patient_sets = self._derive_split_patient_sets(segments_df=segments_df)
        return split_frames, segments_df

    def _build_timeline_df(
        self,
        vitals_df: pd.DataFrame,
        admin_df: pd.DataFrame,
        orders_df: pd.DataFrame,
    ) -> pd.DataFrame:
        frame_specs = [
            (vitals_df, "vital"),
            (admin_df, "med_admin"),
            (orders_df, "med_order"),
        ]
        timeline_parts: list[pd.DataFrame] = []
        for frame, source_type in frame_specs:
            if frame.empty:
                continue
            part_df = frame[["patient_id", ENCOUNTER_ID_COLUMN, TIMESTAMP_COLUMN]].copy()
            part_df["source_type"] = source_type
            timeline_parts.append(part_df)

        if not timeline_parts:
            return self._empty_timeline_frame()
        return pd.concat(timeline_parts, ignore_index=True).sort_values(
            ["patient_id", TIMESTAMP_COLUMN, ENCOUNTER_ID_COLUMN, "source_type"],
            kind="mergesort",
        ).reset_index(drop=True)

    def _segment_filter(self, frame: pd.DataFrame, segment: pd.Series) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()
        mask = frame["patient_id"].astype(str) == str(segment["patient_id"])
        if str(segment[ENCOUNTER_ID_COLUMN]).strip():
            mask &= (
                _normalize_identifier_series(frame[ENCOUNTER_ID_COLUMN])
                .fillna("")
                .astype(str)
                == str(segment[ENCOUNTER_ID_COLUMN])
            )
        mask &= frame[TIMESTAMP_COLUMN].between(
            pd.Timestamp(segment["start_time"]),
            pd.Timestamp(segment["end_time"]),
            inclusive="both",
        )
        return frame.loc[mask].copy()

    def _build_segment_examples(
        self,
        *,
        split: str,
        segment_id: int,
        patient_id: str,
        encounter_id: str,
        vitals_group: pd.DataFrame,
        admin_group: pd.DataFrame,
        orders_group: pd.DataFrame,
        vital_vocab: Sequence[str],
        vital_means: dict[str, float],
        med_vocab: Sequence[str],
    ) -> list[DeliveryRequestExample]:
        if vitals_group.empty and admin_group.empty and orders_group.empty:
            return []

        observation_end = max(
            [
                frame[TIMESTAMP_COLUMN].max()
                for frame in (vitals_group, admin_group, orders_group)
                if not frame.empty
            ]
        )
        examples: list[DeliveryRequestExample] = []
        trigger_times = _collect_triggers(
            vitals_group=vitals_group,
            admin_group=admin_group,
            orders_group=orders_group,
            include_order_triggers=self.dataset_config.include_order_triggers,
        )

        for trigger_time in trigger_times:
            x, m, d, step_mask = _encode_vital_window(
                vitals_group=vitals_group,
                trigger_time=pd.Timestamp(trigger_time),
                vital_vocab=vital_vocab,
                vital_means=vital_means,
                lookback_hours=self.dataset_config.lookback_hours,
                max_seq_len=self.dataset_config.max_seq_len,
            )
            if step_mask.sum() <= 0:
                continue

            med_state = _encode_medication_state(
                admin_group=admin_group,
                trigger_time=pd.Timestamp(trigger_time),
                med_vocab=med_vocab,
                med_lookback_hours=self.dataset_config.med_lookback_hours,
                med_decay_hours=self.dataset_config.med_decay_hours,
            )
            target_payload = _find_next_new_delivery_request(
                orders_group=orders_group,
                admin_group=admin_group,
                trigger_time=pd.Timestamp(trigger_time),
                med_vocab=med_vocab,
                prediction_horizon_hours=self.dataset_config.prediction_horizon_hours,
                observation_end=pd.Timestamp(observation_end),
            )
            duration_idx = _duration_to_bin(
                duration_hours=float(target_payload["duration_hours"]),
                bins=self.dataset_config.time_bins_hours,
            )
            examples.append(
                DeliveryRequestExample(
                    split=split,
                    patient_id=str(patient_id),
                    encounter_id=str(encounter_id),
                    segment_id=int(segment_id),
                    trigger_time=pd.Timestamp(trigger_time),
                    x=x,
                    m=m,
                    d=d,
                    step_mask=step_mask,
                    meds=med_state,
                    duration_hours=float(target_payload["duration_hours"]),
                    duration_idx=int(duration_idx),
                    event=float(target_payload["event"]),
                    next_med_targets=target_payload["next_med_targets"],
                    med_target_available=float(target_payload["med_target_available"]),
                    true_med_labels=list(target_payload["true_med_labels"]),
                )
            )
        return examples

    def _empty_split_payload(
        self,
        *,
        n_vitals: int,
        n_meds: int,
    ) -> dict[str, object]:
        return {
            "x": np.zeros((0, self.dataset_config.max_seq_len, n_vitals), dtype=np.float32),
            "m": np.zeros((0, self.dataset_config.max_seq_len, n_vitals), dtype=np.float32),
            "d": np.zeros((0, self.dataset_config.max_seq_len, n_vitals), dtype=np.float32),
            "step_mask": np.zeros((0, self.dataset_config.max_seq_len), dtype=np.float32),
            "meds": np.zeros((0, n_meds), dtype=np.float32),
            "duration_hours": np.zeros((0,), dtype=np.float32),
            "duration_idx": np.zeros((0,), dtype=np.int64),
            "event": np.zeros((0,), dtype=np.float32),
            "next_med_targets": np.zeros((0, n_meds), dtype=np.float32),
            "med_target_available": np.zeros((0,), dtype=np.float32),
            "metadata": pd.DataFrame(
                columns=[
                    "split",
                    "patient_id",
                    ENCOUNTER_ID_COLUMN,
                    "segment_id",
                    "trigger_time",
                    "true_med_labels",
                ]
            ),
        }

    def _examples_to_payload(
        self,
        examples: Sequence[DeliveryRequestExample],
        *,
        n_vitals: int,
        n_meds: int,
    ) -> dict[str, object]:
        if not examples:
            return self._empty_split_payload(n_vitals=n_vitals, n_meds=n_meds)

        metadata_rows = [
            {
                "split": example.split,
                "patient_id": example.patient_id,
                ENCOUNTER_ID_COLUMN: example.encounter_id,
                "segment_id": int(example.segment_id),
                "trigger_time": pd.Timestamp(example.trigger_time),
                "true_med_labels": json.dumps(example.true_med_labels),
            }
            for example in examples
        ]
        return {
            "x": np.stack([example.x for example in examples]).astype(np.float32),
            "m": np.stack([example.m for example in examples]).astype(np.float32),
            "d": np.stack([example.d for example in examples]).astype(np.float32),
            "step_mask": np.stack([example.step_mask for example in examples]).astype(np.float32),
            "meds": np.stack([example.meds for example in examples]).astype(np.float32),
            "duration_hours": np.asarray([example.duration_hours for example in examples], dtype=np.float32),
            "duration_idx": np.asarray([example.duration_idx for example in examples], dtype=np.int64),
            "event": np.asarray([example.event for example in examples], dtype=np.float32),
            "next_med_targets": np.stack([example.next_med_targets for example in examples]).astype(np.float32),
            "med_target_available": np.asarray(
                [example.med_target_available for example in examples],
                dtype=np.float32,
            ),
            "metadata": pd.DataFrame(metadata_rows),
        }

    def _apply_vital_scaling(
        self,
        split_payloads: dict[str, dict[str, object]],
        *,
        vital_vocab: Sequence[str],
        vital_means: dict[str, float],
        vital_stds: dict[str, float],
    ) -> None:
        if not vital_vocab:
            return

        mean_array = np.asarray([vital_means[vital_name] for vital_name in vital_vocab], dtype=np.float32)
        std_array = np.asarray([vital_stds[vital_name] for vital_name in vital_vocab], dtype=np.float32)
        for split_name in SPLITS:
            x_array = np.asarray(split_payloads[split_name]["x"], dtype=np.float32)
            m_array = np.asarray(split_payloads[split_name]["m"], dtype=np.float32)
            if x_array.size == 0:
                continue
            centered = (x_array - mean_array.reshape(1, 1, -1)) / std_array.reshape(1, 1, -1)
            split_payloads[split_name]["x"] = np.where(m_array > 0.5, centered, 0.0).astype(np.float32)

    def _build_metadata(
        self,
        *,
        vital_vocab: Sequence[str],
        med_vocab: Sequence[str],
        med_code_display_map: dict[str, str],
        vital_means: dict[str, float],
        vital_stds: dict[str, float],
        split_payloads: dict[str, dict[str, object]],
    ) -> dict[str, object]:
        return {
            "version": DATASET_VERSION,
            "dataset_representation": "delivery_request_point_process",
            "patient_id_col": self.patient_id_col,
            "encounter_id_col": ENCOUNTER_ID_COLUMN,
            "timestamp_col": TIMESTAMP_COLUMN,
            "included_tasks": list(self.dataset_config.included_tasks),
            "vital_vocab": list(vital_vocab),
            "med_vocab": list(med_vocab),
            "med_code_display_map": {str(key): str(value) for key, value in med_code_display_map.items()},
            "time_bins_hours": list(self.dataset_config.time_bins_hours),
            "vital_value_means": {name: float(vital_means[name]) for name in vital_vocab},
            "vital_value_stds": {name: float(vital_stds[name]) for name in vital_vocab},
            "scaled_x_mean": [0.0 for _ in vital_vocab],
            "lookback_hours": float(self.dataset_config.lookback_hours),
            "med_lookback_hours": float(self.dataset_config.med_lookback_hours),
            "med_decay_hours": float(self.dataset_config.med_decay_hours),
            "prediction_horizon_hours": float(self.dataset_config.prediction_horizon_hours),
            "max_seq_len": int(self.dataset_config.max_seq_len),
            "validation_split_strategy": self.dataset_config.validation_split_strategy,
            "validation_split_seed": int(self.dataset_config.validation_split_seed),
            "train_patient_count": len(getattr(self, "split_patient_sets", {}).get("train", set())),
            "val_patient_count": len(getattr(self, "split_patient_sets", {}).get("val", set())),
            "test_patient_count": len(getattr(self, "split_patient_sets", {}).get("test", set())),
            "train_iso_weeks": [list(week) for week in sorted(getattr(self, "split_week_sets", {}).get("train", set()))],
            "val_iso_weeks": [list(week) for week in sorted(getattr(self, "split_week_sets", {}).get("val", set()))],
            "test_iso_weeks": [list(week) for week in sorted(getattr(self, "split_week_sets", {}).get("test", set()))],
            "split_example_counts": {
                split_name: int(np.asarray(split_payloads[split_name]["event"]).shape[0])
                for split_name in SPLITS
            },
            "medication_code_col": self.resolved_medication_code_col,
            "medication_mapping": self.medication_mapping_applier.to_metadata(),
            "config_snapshot": _dataset_config_snapshot(self.dataset_config),
        }

    def _build_examples_from_event_frames(
        self,
        *,
        vitals_df: pd.DataFrame,
        admin_df: pd.DataFrame,
        orders_df: pd.DataFrame,
    ) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
        timeline_df = self._build_timeline_df(vitals_df=vitals_df, admin_df=admin_df, orders_df=orders_df)
        if timeline_df.empty:
            raise ValueError(
                "No delivery-request source events were found for the configured date range. "
                "Please check the annotated data files and filters."
            )

        _, segments_df = self._build_split_segments(timeline_df=timeline_df)
        split_source_frames: dict[str, dict[str, list[pd.DataFrame]]] = {
            split_name: {"vitals": [], "admins": [], "orders": []}
            for split_name in SPLITS
        }

        for segment in segments_df.itertuples(index=False):
            segment_series = pd.Series(segment._asdict())
            split_source_frames[segment.split]["vitals"].append(self._segment_filter(vitals_df, segment_series))
            split_source_frames[segment.split]["admins"].append(self._segment_filter(admin_df, segment_series))
            split_source_frames[segment.split]["orders"].append(self._segment_filter(orders_df, segment_series))

        train_vitals_df = (
            pd.concat(split_source_frames["train"]["vitals"], ignore_index=True)
            if split_source_frames["train"]["vitals"]
            else pd.DataFrame(columns=vitals_df.columns)
        )
        train_admin_df = (
            pd.concat(split_source_frames["train"]["admins"], ignore_index=True)
            if split_source_frames["train"]["admins"]
            else pd.DataFrame(columns=admin_df.columns)
        )
        train_orders_df = (
            pd.concat(split_source_frames["train"]["orders"], ignore_index=True)
            if split_source_frames["train"]["orders"]
            else pd.DataFrame(columns=orders_df.columns)
        )
        if train_vitals_df.empty:
            raise ValueError("The training split did not contain any vital-sign events.")
        if train_orders_df.empty and train_admin_df.empty:
            raise ValueError("The training split did not contain any medication history.")

        vital_vocab = _build_vocab(
            train_vitals_df["vital_name"].astype(str).tolist(),
            top_k=self.dataset_config.top_vitals,
            min_count=1,
        )
        med_vocab = _build_vocab(
            list(train_admin_df["med_code"].astype(str).tolist())
            + list(train_orders_df["med_code"].astype(str).tolist()),
            top_k=self.dataset_config.top_meds,
            min_count=self.dataset_config.min_med_count,
        )
        if not vital_vocab:
            raise ValueError("No vital features were available for the delivery-request model.")
        if not med_vocab:
            raise ValueError(
                "No medication identifiers met the training-vocabulary requirements. "
                "Adjust top_meds or min_med_count."
            )
        med_code_display_map = _build_med_code_display_map(
            med_vocab,
            train_admin_df,
            train_orders_df,
            admin_df,
            orders_df,
        )

        vital_means = _compute_vital_means(train_vitals_df, vital_vocab)
        vital_stds = _compute_vital_stds(train_vitals_df, vital_vocab)

        examples_by_split: dict[str, list[DeliveryRequestExample]] = {split: [] for split in SPLITS}
        for segment in segments_df.itertuples(index=False):
            segment_series = pd.Series(segment._asdict())
            examples_by_split[segment.split].extend(
                self._build_segment_examples(
                    split=str(segment.split),
                    segment_id=int(segment.segment_id),
                    patient_id=str(segment.patient_id),
                    encounter_id=str(getattr(segment, ENCOUNTER_ID_COLUMN)),
                    vitals_group=self._segment_filter(vitals_df, segment_series),
                    admin_group=self._segment_filter(admin_df, segment_series),
                    orders_group=self._segment_filter(orders_df, segment_series),
                    vital_vocab=vital_vocab,
                    vital_means=vital_means,
                    med_vocab=med_vocab,
                )
            )

        split_payloads = {
            split_name: self._examples_to_payload(
                examples_by_split[split_name],
                n_vitals=len(vital_vocab),
                n_meds=len(med_vocab),
            )
            for split_name in SPLITS
        }
        self._apply_vital_scaling(
            split_payloads=split_payloads,
            vital_vocab=vital_vocab,
            vital_means=vital_means,
            vital_stds=vital_stds,
        )

        for split_name in SPLITS:
            if int(np.asarray(split_payloads[split_name]["event"]).shape[0]) == 0:
                raise ValueError(
                    f"Split '{split_name}' has no delivery-request examples. "
                    "Adjust the date filters or split configuration."
                )

        metadata = self._build_metadata(
            vital_vocab=vital_vocab,
            med_vocab=med_vocab,
            med_code_display_map=med_code_display_map,
            vital_means=vital_means,
            vital_stds=vital_stds,
            split_payloads=split_payloads,
        )
        return split_payloads, metadata

    def _preprocess_from_request_handler(
        self,
        request_handler: GlobalRequestHandler,
    ) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
        vitals_df = self._build_vitals_frame(request_handler=request_handler)
        admin_df, orders_df = self._build_medication_frames(request_handler=request_handler)
        return self._build_examples_from_event_frames(
            vitals_df=vitals_df,
            admin_df=admin_df,
            orders_df=orders_df,
        )

    def _metadata_path(self) -> Path:
        return self.dataset_dir / "metadata.json"

    def _npz_path(self, split: str) -> Path:
        return self.dataset_dir / NPZ_FILENAMES[split]

    def _split_metadata_path(self, split: str) -> Path:
        return self.dataset_dir / METADATA_FILENAMES[split]

    def _save_dataset(self) -> None:
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        with self._metadata_path().open("w", encoding="utf-8") as metadata_file:
            json.dump(self.metadata, metadata_file, indent=2)

        for split_name in SPLITS:
            split_payload = self.samples[split_name]
            np.savez_compressed(
                self._npz_path(split_name),
                x=split_payload["x"],
                m=split_payload["m"],
                d=split_payload["d"],
                step_mask=split_payload["step_mask"],
                meds=split_payload["meds"],
                duration_hours=split_payload["duration_hours"],
                duration_idx=split_payload["duration_idx"],
                event=split_payload["event"],
                next_med_targets=split_payload["next_med_targets"],
                med_target_available=split_payload["med_target_available"],
            )
            split_payload["metadata"].to_csv(self._split_metadata_path(split_name), index=False)

    def _load_dataset(self) -> None:
        with self._metadata_path().open("r", encoding="utf-8") as metadata_file:
            self.metadata = json.load(metadata_file)

        if int(self.metadata.get("version", -1)) != DATASET_VERSION:
            raise ValueError("Saved delivery-request dataset version mismatch.")
        if self.metadata.get("config_snapshot") != _dataset_config_snapshot(self.dataset_config):
            raise ValueError("Saved delivery-request dataset does not match the current dataset configuration.")

        self.resolved_medication_code_col = self.metadata.get("medication_code_col")
        self.samples = {}
        for split_name in SPLITS:
            arrays = np.load(self._npz_path(split_name), allow_pickle=False)
            split_metadata_df = pd.read_csv(self._split_metadata_path(split_name))
            if not split_metadata_df.empty:
                split_metadata_df["trigger_time"] = pd.to_datetime(
                    split_metadata_df["trigger_time"],
                    errors="coerce",
                )

            self.samples[split_name] = {
                "x": arrays["x"].astype(np.float32),
                "m": arrays["m"].astype(np.float32),
                "d": arrays["d"].astype(np.float32),
                "step_mask": arrays["step_mask"].astype(np.float32),
                "meds": arrays["meds"].astype(np.float32),
                "duration_hours": arrays["duration_hours"].astype(np.float32),
                "duration_idx": arrays["duration_idx"].astype(np.int64),
                "event": arrays["event"].astype(np.float32),
                "next_med_targets": arrays["next_med_targets"].astype(np.float32),
                "med_target_available": arrays["med_target_available"].astype(np.float32),
                "metadata": split_metadata_df,
            }

    def get_dataset_bundle(self) -> "DeliveryRequestsDatasetBundle":
        return DeliveryRequestsDatasetBundle(samples=self.samples, metadata=self.metadata)


class DeliveryRequestsDatasetBundle:
    def __init__(self, samples: dict[str, dict[str, object]], metadata: dict[str, object]) -> None:
        self.samples = samples
        self.metadata = metadata
        self.vital_vocab = list(metadata.get("vital_vocab", []))
        self.med_vocab = list(metadata.get("med_vocab", []))
        self.med_code_display_map = {
            str(key): str(value)
            for key, value in dict(metadata.get("med_code_display_map", {})).items()
        }
        self.time_bins_hours = np.asarray(metadata.get("time_bins_hours", []), dtype=np.float32)
        self.x_mean = np.asarray(metadata.get("scaled_x_mean", []), dtype=np.float32)

    @property
    def n_vitals(self) -> int:
        return len(self.vital_vocab)

    @property
    def n_meds(self) -> int:
        return len(self.med_vocab)

    def get_split_arrays(self, split: str) -> dict[str, object]:
        if split not in SPLITS:
            raise ValueError(f"Unsupported split '{split}'.")
        return self.samples[split]

    def get_split_metadata(self, split: str) -> pd.DataFrame:
        return self.get_split_arrays(split)["metadata"]

    def length(self, split: str) -> int:
        return int(np.asarray(self.get_split_arrays(split)["event"]).shape[0])


class DeliveryRequestsDataset(Dataset):
    def __init__(
        self,
        dataset_bundle: DeliveryRequestsDatasetBundle,
        split: str = "train",
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"Unsupported split '{split}'. Expected one of {SPLITS}.")
        self.split = split
        self.bundle = dataset_bundle
        self._split_arrays = dataset_bundle.get_split_arrays(split)

    def __len__(self) -> int:
        return int(np.asarray(self._split_arrays["event"]).shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "x": torch.from_numpy(self._split_arrays["x"][index]).float(),
            "m": torch.from_numpy(self._split_arrays["m"][index]).float(),
            "d": torch.from_numpy(self._split_arrays["d"][index]).float(),
            "step_mask": torch.from_numpy(self._split_arrays["step_mask"][index]).float(),
            "meds": torch.from_numpy(self._split_arrays["meds"][index]).float(),
            "duration_hours": torch.tensor(
                float(self._split_arrays["duration_hours"][index]),
                dtype=torch.float32,
            ),
            "duration_idx": torch.tensor(
                int(self._split_arrays["duration_idx"][index]),
                dtype=torch.long,
            ),
            "event": torch.tensor(float(self._split_arrays["event"][index]), dtype=torch.float32),
            "next_med_targets": torch.from_numpy(
                self._split_arrays["next_med_targets"][index]
            ).float(),
            "med_target_available": torch.tensor(
                float(self._split_arrays["med_target_available"][index]),
                dtype=torch.float32,
            ),
        }


def build_delivery_request_dataset_bundle(
    dataset_config: DeliveryRequestDatasetConfig,
) -> DeliveryRequestsDatasetBundle:
    data_manager = DeliveryRequestsDataManager(dataset_config=dataset_config)
    return data_manager.get_dataset_bundle()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="DeliveryRequestsDataset",
        description="Create delivery-request prediction datasets from a JSON config file.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to a JSON file containing at least a 'dataset_config' section.",
    )
    return parser


if __name__ == "__main__":
    p_start = datetime.datetime.now()
    try:
        parser = build_parser()
        parsed_args = parser.parse_args()
        training_config = DeliveryRequestTrainingConfig.from_json_file(parsed_args.config_path)
        dataset_bundle = build_delivery_request_dataset_bundle(
            dataset_config=training_config.dataset_config,
        )
        for split_name in SPLITS:
            print(
                f"{split_name}: "
                f"{dataset_bundle.length(split_name)} examples, "
                f"x_shape={dataset_bundle.get_split_arrays(split_name)['x'].shape}"
            )
        print("Process completed successfully.")
    except Exception as error_main_context:
        print("Fail End Process: ", error_main_context)
        traceback.print_exc()
    p_stop = datetime.datetime.now()
    print("Execution time: " + str(p_stop - p_start))
