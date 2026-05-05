from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import math
import random
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from HTAMP.data_processing.data_helpers import DataHelpers
from HTAMP.data_processing.processing_dataclasses import AnnotatedDataFiles
from HTAMP.prediction import run_delivery_tpp_otd_evaluation as delivery_otd
from HTAMP.prediction import run_vital_sign_tpp_otd_evaluation as vital_otd
from HTAMP.prediction.data_provider.delivery_tpp_dataset import (
    DELIVERY_TASK_NAME,
    MEDICATION_CODE_PROPERTY,
)
from HTAMP.prediction.data_provider.request_events_dataset import (
    ENCOUNTER_ID_COLUMN,
    FLOOR_COLUMN,
    _normalize_identifier_series,
)
from HTAMP.prediction.data_provider.vital_sign_tpp_dataset import EOS_EVENT_TYPE_NAME
from HTAMP.prediction.metrics.otd_metric import Event

VITAL_TASK = "vital_sign"
DELIVERY_TASK = "delivery"
DEFAULT_OUTPUT_CSV = "data/prediction/offline_request_prediction_cache.csv"
DEFAULT_VITAL_EASY_CONFIG_PATH = (
    "HTAMP/prediction/configs/config_files/prediction/"
    "vital_sign_easy_tpp_training.json"
)
DEFAULT_DELIVERY_EASY_CONFIG_PATH = (
    "HTAMP/prediction/configs/config_files/prediction/"
    "delivery_easy_tpp_training.json"
)
SCHEMA_VERSION = 1

CSV_FIELDS = [
    "schema_version",
    "prediction_task",
    "family",
    "model_name",
    "variant",
    "run_name",
    "split",
    "sequence_index",
    "segment_id",
    "patient_id",
    "encounter_id",
    "mrn",
    "sequence_start_timestamp",
    "sequence_end_timestamp",
    "sequence_day",
    "prefix_id",
    "prefix_event_count",
    "prediction_anchor_timestamp",
    "observed_sequence_key",
    "observed_events_json",
    "sample_index",
    "sampled_sequence_id",
    "inference_seconds",
    "sampled_event_index",
    "row_kind",
    "predicted_request_id",
    "request_type",
    "predicted_mark_name",
    "predicted_relative_time_hours",
    "predicted_time_since_anchor_hours",
    "scheduled_dttm",
    "ordered_dttm",
    "Scheduled DTTM",
    "Ordered DTTM",
    "Medication Scheduled DTTM",
    "Medication Order DTTM",
    "Administered DTTM",
    "scheduled_room",
    "scheduled_space_id",
    "scheduled_space_supplies",
    "floor",
    "day",
    "medication_code",
    "medication_code_index",
    "medication_display_name",
    "properties_json",
]


@dataclass(frozen=True)
class LocationInfo:
    scheduled_room: str = ""
    scheduled_space_id: str = ""
    scheduled_space_supplies: str = ""
    floor: int | None = None


@dataclass(frozen=True)
class SampledRequest:
    relative_time_hours: float
    request_type: str
    mark_name: str
    properties: dict[str, Any]
    medication_code: str = ""
    medication_code_index: int | None = None
    medication_display_name: str = ""


def _log(message: str) -> None:
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", flush=True)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _hash_payload(value: Any) -> str:
    return hashlib.sha256(_json_dumps(value).encode("utf-8")).hexdigest()[:24]


def _resolve_repo_relative_path(path_str: str | Path | None) -> Path | None:
    return vital_otd._resolve_repo_relative_path(path_str)


def _parse_csv_set(raw_value: str | None) -> set[str] | None:
    if raw_value is None or not str(raw_value).strip():
        return None
    return {
        item.strip()
        for item in str(raw_value).split(",")
        if item.strip()
    }


def _parse_tasks(raw_value: str) -> tuple[str, ...]:
    aliases = {
        "vital": VITAL_TASK,
        "vitals": VITAL_TASK,
        "vital_sign": VITAL_TASK,
        "vital_signs": VITAL_TASK,
        "delivery": DELIVERY_TASK,
        "medicine": DELIVERY_TASK,
        "medication": DELIVERY_TASK,
        "medications": DELIVERY_TASK,
    }
    tasks: list[str] = []
    for raw_task in str(raw_value).split(","):
        task = aliases.get(raw_task.strip().lower())
        if task is None:
            raise ValueError(
                f"Unsupported task '{raw_task}'. Use any of: vital_sign,delivery."
            )
        if task not in tasks:
            tasks.append(task)
    return tuple(tasks)


def _tasks_in_safe_runtime_order(tasks: Sequence[str]) -> tuple[str, ...]:
    ordered_tasks = [task for task in (VITAL_TASK, DELIVERY_TASK) if task in tasks]
    if tuple(ordered_tasks) != tuple(tasks):
        _log(
            "Reordered tasks to run vital_sign before delivery because the "
            "delivery evaluator patches shared OTD helpers at runtime."
        )
    return tuple(ordered_tasks)


def _metadata_path_for_csv(output_csv: Path) -> Path:
    return output_csv.with_suffix(output_csv.suffix + ".metadata.json")


def _safe_identifier(value: Any) -> str:
    text = str(value or "").strip()
    return "".join(char if char.isalnum() else "_" for char in text).strip("_")


def _finite_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric_value):
        return None
    return numeric_value


def _int_or_blank(value: Any) -> int | str:
    numeric_value = _finite_float_or_none(value)
    if numeric_value is None:
        return ""
    return int(numeric_value)


def _normalize_identifier_value(value: Any) -> str:
    normalized = _normalize_identifier_series(pd.Series([value])).iloc[0]
    if pd.isna(normalized):
        return ""
    return str(normalized)


def _timestamp_from_record_time(record: Any, relative_time_hours: float) -> pd.Timestamp:
    return pd.Timestamp(record.sequence_start_timestamp) + pd.Timedelta(
        hours=float(relative_time_hours)
    )


def _record_non_eos_raw_events(record: Any) -> list[dict[str, Any]]:
    raw_events: list[dict[str, Any]] = []
    for raw_event, mark_name in zip(record.raw_events, record.mark_names):
        if mark_name == EOS_EVENT_TYPE_NAME:
            continue
        raw_events.append(dict(raw_event))
    return raw_events


def _observed_events_payload(record: Any, prefix_len: int) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for raw_event in _record_non_eos_raw_events(record)[:prefix_len]:
        payload.append(
            {
                "timestamp": str(raw_event.get("timestamp", "")),
                "task_name": str(raw_event.get("task_name", "")),
                "mark_name": str(
                    raw_event.get(
                        "delivery_event_type",
                        raw_event.get("flex_tpp_event_type", raw_event.get("task_name", "")),
                    )
                ),
                "medication_code": str(raw_event.get("medication_code", "")),
            }
        )
    return payload


def _anchor_timestamp(record: Any, prefix_len: int) -> pd.Timestamp:
    if prefix_len <= 0:
        return pd.Timestamp(record.sequence_start_timestamp)
    raw_events = _record_non_eos_raw_events(record)
    if prefix_len <= len(raw_events):
        return pd.Timestamp(raw_events[prefix_len - 1].get("timestamp"))
    return _timestamp_from_record_time(record, float(record.time_seqs[prefix_len - 1]))


def _record_floor_fallback(record: Any, prefix_len: int) -> int | None:
    floors: list[int] = []
    for raw_event in _record_non_eos_raw_events(record)[: max(prefix_len, 1)]:
        floor_value = _finite_float_or_none(raw_event.get("floor"))
        if floor_value is not None:
            floors.append(int(floor_value))
    if not floors:
        return None
    return max(set(floors), key=floors.count)


def _prefix_id(
    *,
    prediction_task: str,
    spec: Any,
    record: Any,
    sequence_index: int,
    prefix_len: int,
    observed_sequence_key: str,
) -> str:
    return ".".join(
        [
            "predctx",
            _safe_identifier(prediction_task),
            _safe_identifier(spec.run_name),
            f"seq{int(sequence_index):06d}",
            f"prefix{int(prefix_len):04d}",
            observed_sequence_key,
        ]
    )


class EncounterLocationLookup:
    def __init__(self, *, annotated_data_files: AnnotatedDataFiles, patient_id_col: str) -> None:
        self.patient_id_col = patient_id_col
        self.room_stays_df = self._load_room_stays(annotated_data_files.annotated_visits)
        self.admissions_df = self._load_admissions(
            annotated_data_files.annotated_admissions_discharges
        )
        self.space_supplies = self._load_space_supplies(
            annotated_data_files.annotated_medications
        )

    def _load_room_stays(self, path_value: str | None) -> pd.DataFrame:
        path = _resolve_repo_relative_path(path_value)
        columns = [self.patient_id_col, "location", "start", "end", "space_id"]
        if path is None or not path.exists():
            _log(f"Room-stay file not found at {path}; predicted locations may be blank.")
            return pd.DataFrame(columns=columns)

        room_df = pd.read_csv(path)
        patient_col = self._first_existing_column(room_df, [self.patient_id_col, "MRN", "PAT_ID"])
        if patient_col is None:
            _log(f"Room-stay file {path} has no patient id column; locations may be blank.")
            return pd.DataFrame(columns=columns)

        location_col = self._first_existing_column(room_df, ["location", "scheduled_room", "room"])
        space_col = self._first_existing_column(room_df, ["space_id", "scheduled_space_id"])
        start_col = self._first_existing_column(room_df, ["start", "scheduled_start"])
        end_col = self._first_existing_column(room_df, ["end", "scheduled_end"])
        if location_col is None or start_col is None or end_col is None:
            _log(f"Room-stay file {path} is missing location/start/end columns.")
            return pd.DataFrame(columns=columns)

        normalized = pd.DataFrame(
            {
                self.patient_id_col: _normalize_identifier_series(room_df[patient_col]),
                "location": room_df[location_col].astype(str),
                "start": pd.to_datetime(room_df[start_col], errors="coerce"),
                "end": pd.to_datetime(room_df[end_col], errors="coerce"),
                "space_id": (
                    room_df[space_col].astype(str)
                    if space_col is not None
                    else pd.Series("", index=room_df.index)
                ),
            }
        )
        normalized = normalized.dropna(subset=[self.patient_id_col, "start"]).copy()
        normalized["end"] = normalized["end"].fillna(pd.Timestamp.max)
        normalized = normalized.sort_values(
            [self.patient_id_col, "start", "end"],
            kind="mergesort",
        ).reset_index(drop=True)
        return normalized

    def _load_admissions(self, path_value: str | None) -> pd.DataFrame:
        path = _resolve_repo_relative_path(path_value)
        columns = [self.patient_id_col, ENCOUNTER_ID_COLUMN, "admission_start", "discharge_end"]
        if path is None or not path.exists():
            return pd.DataFrame(columns=columns)

        admissions_df = pd.read_csv(path)
        patient_col = self._first_existing_column(admissions_df, [self.patient_id_col, "MRN", "PAT_ID"])
        encounter_col = self._first_existing_column(
            admissions_df,
            [ENCOUNTER_ID_COLUMN, "Patient Encounter CSN", "PAT_ENC_CSN_ID"],
        )
        admission_col = self._first_existing_column(
            admissions_df,
            ["HOSPITAL_ADMISSION", "Hospital Admission"],
        )
        discharge_col = self._first_existing_column(
            admissions_df,
            ["HOSPITAL_DISCHARGE", "Hospital Discharge"],
        )
        if patient_col is None or admission_col is None:
            return pd.DataFrame(columns=columns)

        normalized = pd.DataFrame(
            {
                self.patient_id_col: _normalize_identifier_series(admissions_df[patient_col]),
                ENCOUNTER_ID_COLUMN: (
                    _normalize_identifier_series(admissions_df[encounter_col])
                    if encounter_col is not None
                    else pd.Series("", index=admissions_df.index)
                ),
                "admission_start": pd.to_datetime(admissions_df[admission_col], errors="coerce"),
                "discharge_end": (
                    pd.to_datetime(admissions_df[discharge_col], errors="coerce")
                    if discharge_col is not None
                    else pd.Series(pd.NaT, index=admissions_df.index)
                ),
            }
        )
        normalized = normalized.dropna(subset=[self.patient_id_col, "admission_start"]).copy()
        normalized["discharge_end"] = normalized["discharge_end"].fillna(pd.Timestamp.max)
        return normalized.sort_values(
            [self.patient_id_col, ENCOUNTER_ID_COLUMN, "admission_start"],
            kind="mergesort",
        ).reset_index(drop=True)

    def _load_space_supplies(self, medication_path_value: str | None) -> dict[str, str]:
        path = _resolve_repo_relative_path(medication_path_value)
        if path is None or not path.exists():
            return {}
        try:
            med_df = pd.read_csv(path, usecols=lambda col: col in {"scheduled_space_id", "scheduled_space_supplies"})
        except Exception:
            return {}
        if "scheduled_space_id" not in med_df.columns or "scheduled_space_supplies" not in med_df.columns:
            return {}
        med_df = med_df.dropna(subset=["scheduled_space_id", "scheduled_space_supplies"])
        return {
            str(row["scheduled_space_id"]): str(row["scheduled_space_supplies"])
            for row in med_df.to_dict(orient="records")
        }

    def _first_existing_column(self, frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
        normalized_columns = {
            str(column).strip().lower(): str(column)
            for column in frame.columns
        }
        for candidate in candidates:
            matched = normalized_columns.get(str(candidate).strip().lower())
            if matched is not None:
                return matched
        return None

    def _timestamp_in_encounter(
        self,
        *,
        patient_id: str,
        encounter_id: str,
        timestamp: pd.Timestamp,
    ) -> bool:
        if self.admissions_df.empty or not encounter_id:
            return True
        patient_mask = self.admissions_df[self.patient_id_col].astype(str).eq(str(patient_id))
        encounter_mask = self.admissions_df[ENCOUNTER_ID_COLUMN].astype(str).eq(str(encounter_id))
        candidates = self.admissions_df[patient_mask & encounter_mask]
        if candidates.empty:
            return True
        return bool(
            (
                candidates["admission_start"].le(timestamp)
                & candidates["discharge_end"].gt(timestamp)
            ).any()
        )

    def lookup(
        self,
        *,
        patient_id: str,
        encounter_id: str,
        timestamp: pd.Timestamp,
        floor_fallback: int | None = None,
    ) -> LocationInfo:
        timestamp = pd.Timestamp(timestamp)
        if self.room_stays_df.empty:
            return LocationInfo(floor=floor_fallback)
        patient_id = _normalize_identifier_value(patient_id)
        encounter_id = _normalize_identifier_value(encounter_id)
        patient_mask = self.room_stays_df[self.patient_id_col].astype(str).eq(patient_id)
        time_mask = self.room_stays_df["start"].le(timestamp) & self.room_stays_df["end"].gt(timestamp)
        candidates = self.room_stays_df[patient_mask & time_mask]
        if candidates.empty:
            return LocationInfo(floor=floor_fallback)

        if not self._timestamp_in_encounter(
            patient_id=patient_id,
            encounter_id=str(encounter_id),
            timestamp=timestamp,
        ):
            return LocationInfo(floor=floor_fallback)

        row = candidates.sort_values("start", kind="mergesort").iloc[-1]
        scheduled_room = str(row.get("location", ""))
        scheduled_space_id = str(row.get("space_id", ""))
        floor = floor_fallback
        extracted_floor = DataHelpers.extract_floor(scheduled_room)
        if extracted_floor is not None:
            floor = int(extracted_floor)
        scheduled_space_supplies = self.space_supplies.get(scheduled_space_id, "")
        return LocationInfo(
            scheduled_room=scheduled_room,
            scheduled_space_id=scheduled_space_id,
            scheduled_space_supplies=scheduled_space_supplies,
            floor=floor,
        )


class MedicationMetadata:
    def __init__(self, metadata: Mapping[str, Any]) -> None:
        source_metadata = dict(metadata.get("source_dataset_metadata", metadata))
        self.medication_code_vocab = [
            str(code)
            for code in source_metadata.get("medication_code_vocab", source_metadata.get("med_vocab", []))
        ]
        self.med_code_display_map = {
            str(key): str(value)
            for key, value in dict(source_metadata.get("med_code_display_map", {})).items()
        }
        self.event_type_to_med_code: dict[str, str] = {}
        for med_code, event_type in dict(source_metadata.get("med_code_to_event_type", {})).items():
            self.event_type_to_med_code.setdefault(str(event_type), str(med_code))

    def from_mark_or_properties(
        self,
        *,
        mark_name: str,
        properties: Mapping[str, Any],
    ) -> tuple[str, int | None, str]:
        med_code = self.event_type_to_med_code.get(str(mark_name), "")
        med_index = None
        numeric_index = _finite_float_or_none(properties.get(MEDICATION_CODE_PROPERTY))
        if numeric_index is not None:
            med_index = int(round(numeric_index))
            if not med_code and 0 <= med_index < len(self.medication_code_vocab):
                med_code = self.medication_code_vocab[med_index]
        if med_code and med_index is None and med_code in self.medication_code_vocab:
            med_index = self.medication_code_vocab.index(med_code)
        return med_code, med_index, self.med_code_display_map.get(med_code, "")


def _sampled_requests_from_events(
    *,
    events: Sequence[Event],
    prediction_task: str,
    metadata: Mapping[str, Any],
    medication_metadata: MedicationMetadata | None,
) -> list[SampledRequest]:
    requests: list[SampledRequest] = []
    for event in events:
        properties: dict[str, Any] = {}
        medication_code = ""
        medication_code_index = None
        medication_display_name = ""
        request_type = str(event.event_type)
        if prediction_task == DELIVERY_TASK and medication_metadata is not None:
            request_type = DELIVERY_TASK_NAME
            medication_code, medication_code_index, medication_display_name = (
                medication_metadata.from_mark_or_properties(
                    mark_name=str(event.mark),
                    properties=properties,
                )
            )
        requests.append(
            SampledRequest(
                relative_time_hours=float(event.time),
                request_type=request_type,
                mark_name=str(event.mark),
                properties=properties,
                medication_code=medication_code,
                medication_code_index=medication_code_index,
                medication_display_name=medication_display_name,
            )
        )
    return requests


def _sampled_requests_from_flex_events(
    *,
    flex_events: Sequence[tuple[float, float, int, Mapping[str, float]]],
    flex_bundle: Any,
    mark_encoder: Any,
    prediction_task: str,
    medication_metadata: MedicationMetadata | None,
) -> list[SampledRequest]:
    requests: list[SampledRequest] = []
    for start_time, _, event_type, event_props in flex_events:
        event_type_index = int(event_type)
        if event_type_index == getattr(flex_bundle, "eos_event_type", -1):
            break
        if event_type_index < 0 or event_type_index >= len(flex_bundle.event_types):
            continue
        event_type_name = str(flex_bundle.event_types[event_type_index])
        base_task = mark_encoder.base_task(event_type_name)
        if event_type_name in mark_encoder.mark_name_set:
            mark_name = event_type_name
        elif base_task in mark_encoder.mark_name_set:
            mark_name = base_task
        else:
            mark_name = mark_encoder.label_event(
                task_name=base_task,
                properties=event_props,
            )
        if mark_name is None or mark_name == EOS_EVENT_TYPE_NAME:
            continue
        properties = {
            str(key): value
            for key, value in dict(event_props or {}).items()
        }
        medication_code = ""
        medication_code_index = None
        medication_display_name = ""
        if prediction_task == DELIVERY_TASK and medication_metadata is not None:
            medication_code, medication_code_index, medication_display_name = (
                medication_metadata.from_mark_or_properties(
                    mark_name=str(mark_name),
                    properties=properties,
                )
            )
        requests.append(
            SampledRequest(
                relative_time_hours=float(start_time),
                request_type=str(base_task),
                mark_name=str(mark_name),
                properties=properties,
                medication_code=medication_code,
                medication_code_index=medication_code_index,
                medication_display_name=medication_display_name,
            )
        )
    return requests


def _base_row_payload(
    *,
    prediction_task: str,
    spec: Any,
    record: Any,
    sequence_index: int,
    prefix_len: int,
    prefix_id: str,
    observed_sequence_key: str,
    observed_events_json: str,
    sample_index: int,
    sampled_sequence_id: str,
    anchor_timestamp: pd.Timestamp,
) -> dict[str, Any]:
    sequence_day = pd.Timestamp(record.sequence_start_timestamp).date().isoformat()
    return {
        "schema_version": SCHEMA_VERSION,
        "prediction_task": prediction_task,
        "family": spec.family,
        "model_name": spec.model_name,
        "variant": spec.variant,
        "run_name": spec.run_name,
        "split": record.split,
        "sequence_index": int(sequence_index),
        "segment_id": _int_or_blank(record.segment_id),
        "patient_id": str(record.patient_id),
        "encounter_id": str(record.encounter_id),
        "mrn": str(record.patient_id),
        "sequence_start_timestamp": str(record.sequence_start_timestamp),
        "sequence_end_timestamp": str(record.sequence_end_timestamp),
        "sequence_day": sequence_day,
        "prefix_id": prefix_id,
        "prefix_event_count": int(prefix_len),
        "prediction_anchor_timestamp": anchor_timestamp.isoformat(),
        "observed_sequence_key": observed_sequence_key,
        "observed_events_json": observed_events_json,
        "sample_index": int(sample_index),
        "sampled_sequence_id": sampled_sequence_id,
        "inference_seconds": "",
    }


def _request_row(
    *,
    base_row: Mapping[str, Any],
    sampled_request: SampledRequest,
    sampled_event_index: int,
    location_lookup: EncounterLocationLookup,
    floor_fallback: int | None,
    anchor_timestamp: pd.Timestamp,
) -> dict[str, Any]:
    scheduled_timestamp = pd.Timestamp(base_row["sequence_start_timestamp"]) + pd.Timedelta(
        hours=float(sampled_request.relative_time_hours)
    )
    location_info = location_lookup.lookup(
        patient_id=str(base_row["patient_id"]),
        encounter_id=str(base_row["encounter_id"]),
        timestamp=scheduled_timestamp,
        floor_fallback=floor_fallback,
    )
    ordered_timestamp = anchor_timestamp
    scheduled_dttm = scheduled_timestamp.isoformat()
    ordered_dttm = ordered_timestamp.isoformat()
    request_type = str(sampled_request.request_type)
    row = dict(base_row)
    row.update(
        {
            "sampled_event_index": int(sampled_event_index),
            "row_kind": "sampled_request",
            "predicted_request_id": (
                f"{base_row['sampled_sequence_id']}.event{int(sampled_event_index):03d}"
            ),
            "request_type": request_type,
            "predicted_mark_name": sampled_request.mark_name,
            "predicted_relative_time_hours": float(sampled_request.relative_time_hours),
            "predicted_time_since_anchor_hours": (
                scheduled_timestamp - anchor_timestamp
            ).total_seconds() / 3600.0,
            "scheduled_dttm": scheduled_dttm,
            "ordered_dttm": ordered_dttm,
            "Scheduled DTTM": scheduled_dttm if request_type != DELIVERY_TASK_NAME else "",
            "Ordered DTTM": ordered_dttm if request_type != DELIVERY_TASK_NAME else "",
            "Medication Scheduled DTTM": scheduled_dttm if request_type == DELIVERY_TASK_NAME else "",
            "Medication Order DTTM": ordered_dttm if request_type == DELIVERY_TASK_NAME else "",
            "Administered DTTM": "",
            "scheduled_room": location_info.scheduled_room,
            "scheduled_space_id": location_info.scheduled_space_id,
            "scheduled_space_supplies": location_info.scheduled_space_supplies,
            "floor": "" if location_info.floor is None else int(location_info.floor),
            "day": scheduled_timestamp.date().isoformat(),
            "medication_code": sampled_request.medication_code,
            "medication_code_index": (
                "" if sampled_request.medication_code_index is None else sampled_request.medication_code_index
            ),
            "medication_display_name": sampled_request.medication_display_name,
            "properties_json": _json_dumps(sampled_request.properties),
        }
    )
    return row


def _empty_sample_row(base_row: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(base_row)
    row.update(
        {
            "sampled_event_index": -1,
            "row_kind": "empty_sample",
            "predicted_request_id": f"{base_row['sampled_sequence_id']}.empty",
            "request_type": "",
            "predicted_mark_name": "",
            "predicted_relative_time_hours": "",
            "predicted_time_since_anchor_hours": "",
            "scheduled_dttm": "",
            "ordered_dttm": "",
            "Scheduled DTTM": "",
            "Ordered DTTM": "",
            "Medication Scheduled DTTM": "",
            "Medication Order DTTM": "",
            "Administered DTTM": "",
            "scheduled_room": "",
            "scheduled_space_id": "",
            "scheduled_space_supplies": "",
            "floor": "",
            "day": "",
            "medication_code": "",
            "medication_code_index": "",
            "medication_display_name": "",
            "properties_json": "{}",
        }
    )
    return row


def _otd_args_for_task(
    *,
    args: argparse.Namespace,
    prediction_task: str,
    easy_config_path: str | None,
) -> argparse.Namespace:
    return argparse.Namespace(
        easy_config_path=easy_config_path,
        split=args.split,
        demand_level=args.demand_level,
        demand_sequence_assignment=args.demand_sequence_assignment,
        max_sequences=args.max_sequences,
        sequence_subset_strategy=args.sequence_subset_strategy,
        seed=args.seed,
        use_saved_datasets=args.use_saved_datasets,
        max_future_events=args.max_future_events,
        min_prefix_events=args.min_prefix_events,
        prefix_stride=1,
        max_prefixes_per_sequence=args.max_prefixes_per_sequence,
        prefix_subset_strategy=args.prefix_subset_strategy,
        num_samples=args.num_samples,
        device=args.device,
        flex_mean_of=args.flex_mean_of,
        easy_thinning_num_sample=args.easy_thinning_num_sample,
        easy_thinning_num_exp=args.easy_thinning_num_exp,
        easy_thinning_over_sample_rate=args.easy_thinning_over_sample_rate,
        easy_thinning_patience_counter=args.easy_thinning_patience_counter,
        easy_thinning_num_samples_boundary=args.easy_thinning_num_samples_boundary,
        easy_thinning_dtime_max=args.easy_thinning_dtime_max,
        progress_every=args.progress_every,
        progress_sample_interval=args.progress_sample_interval,
        prediction_task=prediction_task,
    )


def _write_rows_for_easy_model(
    *,
    writer: csv.DictWriter,
    prediction_task: str,
    otd: Any,
    spec: Any,
    context: Any,
    args: argparse.Namespace,
    device: torch.device,
    location_lookup: EncounterLocationLookup,
    medication_metadata: MedicationMetadata | None,
    include_observed_events_json: bool,
) -> int:
    model, model_easy_bundle = otd._load_easy_model_and_bundle(
        spec=spec,
        args=args,
        device=device,
    )
    first_event_pool = otd._first_event_pool(model_easy_bundle)
    model_records = otd._easy_records_by_key(model_easy_bundle, args.split)
    rng = random.Random(args.seed)
    row_count = 0
    for sequence_index, record in enumerate(context.easy_records):
        model_record = model_records.get(otd._easy_record_key(record))
        if model_record is None:
            raise ValueError(f"No matching EasyTPP record for sequence {sequence_index}.")
        true_events = otd._easy_record_events(record=record, mark_encoder=context.mark_encoder)

        def sample_easy_requests(prefix_len: int, future_count: int) -> list[SampledRequest]:
            return _sampled_requests_from_events(
                events=otd._sample_easy_rollout(
                    model=model,
                    record=model_record,
                    prefix_len=prefix_len,
                    max_future_events=future_count,
                    easy_bundle=model_easy_bundle,
                    first_event_pool=first_event_pool,
                    rng=rng,
                    device=device,
                ),
                prediction_task=prediction_task,
                metadata=model_easy_bundle.metadata,
                medication_metadata=medication_metadata,
            )

        for prefix_len in otd._prefix_lengths(args, true_events):
            future_count = otd._future_event_count(args, true_events, prefix_len)
            if future_count <= 0:
                continue
            row_count += _write_rows_for_prefix_samples(
                writer=writer,
                prediction_task=prediction_task,
                otd=otd,
                spec=spec,
                record=record,
                sequence_index=sequence_index,
                prefix_len=prefix_len,
                args=args,
                location_lookup=location_lookup,
                medication_metadata=medication_metadata,
                include_observed_events_json=include_observed_events_json,
                sample_fn=lambda prefix_len=prefix_len, future_count=future_count: (
                    sample_easy_requests(prefix_len, future_count)
                ),
            )
        if args.progress_every and (sequence_index + 1) % int(args.progress_every) == 0:
            _log(
                f"{prediction_task}/{spec.run_name}: processed "
                f"{sequence_index + 1}/{len(context.easy_records)} sequences; "
                f"rows={row_count}."
            )
    return row_count


def _write_rows_for_multittpp_model(
    *,
    writer: csv.DictWriter,
    prediction_task: str,
    otd: Any,
    spec: Any,
    context: Any,
    args: argparse.Namespace,
    device: torch.device,
    location_lookup: EncounterLocationLookup,
    medication_metadata: MedicationMetadata | None,
    include_observed_events_json: bool,
) -> int:
    model, multittpp_bundle = otd._load_multittpp_model_and_bundle(
        spec=spec,
        args=args,
        device=device,
    )
    model_records = otd._easy_records_by_key(multittpp_bundle, args.split)
    row_count = 0
    for sequence_index, record in enumerate(context.easy_records):
        model_record = model_records.get(otd._easy_record_key(record))
        if model_record is None:
            raise ValueError(f"No matching MultiTTPP record for sequence {sequence_index}.")
        true_events = otd._easy_record_events(record=record, mark_encoder=context.mark_encoder)

        def sample_multittpp_requests(prefix_len: int, future_count: int) -> list[SampledRequest]:
            sampled_times, sampled_types = otd._sample_multittpp_rollout(
                model=model,
                record=model_record,
                prefix_len=prefix_len,
                max_future_events=future_count,
                device=device,
            )
            return _sampled_requests_from_events(
                events=otd._discrete_events_from_multittpp_samples(
                    sampled_times=sampled_times,
                    sampled_types=sampled_types,
                    multittpp_bundle=multittpp_bundle,
                    mark_encoder=context.mark_encoder,
                ),
                prediction_task=prediction_task,
                metadata=multittpp_bundle.metadata,
                medication_metadata=medication_metadata,
            )

        for prefix_len in otd._prefix_lengths(args, true_events):
            future_count = otd._future_event_count(args, true_events, prefix_len)
            if future_count <= 0:
                continue
            row_count += _write_rows_for_prefix_samples(
                writer=writer,
                prediction_task=prediction_task,
                otd=otd,
                spec=spec,
                record=record,
                sequence_index=sequence_index,
                prefix_len=prefix_len,
                args=args,
                location_lookup=location_lookup,
                medication_metadata=medication_metadata,
                include_observed_events_json=include_observed_events_json,
                sample_fn=lambda prefix_len=prefix_len, future_count=future_count: (
                    sample_multittpp_requests(prefix_len, future_count)
                ),
            )
        if args.progress_every and (sequence_index + 1) % int(args.progress_every) == 0:
            _log(
                f"{prediction_task}/{spec.run_name}: processed "
                f"{sequence_index + 1}/{len(context.easy_records)} sequences; "
                f"rows={row_count}."
            )
    return row_count


def _write_rows_for_flex_model(
    *,
    writer: csv.DictWriter,
    prediction_task: str,
    otd: Any,
    spec: Any,
    context: Any,
    args: argparse.Namespace,
    device: torch.device,
    location_lookup: EncounterLocationLookup,
    medication_metadata: MedicationMetadata | None,
    include_observed_events_json: bool,
) -> int:
    model, flex_bundle = otd._load_flex_model_and_bundle(spec=spec, args=args, device=device)
    flex_records = otd._flex_records_by_key(flex_bundle, args.split)
    flex_dataset = flex_bundle.get_dataset(args.split)
    row_count = 0
    for sequence_index, record in enumerate(context.easy_records):
        flex_record = flex_records.get(otd._easy_record_key(record))
        if flex_record is None:
            raise ValueError(f"No matching FlexTPP record for sequence {sequence_index}.")
        matched_flex_events = otd._matched_flex_events_for_easy_record(
            easy_record=record,
            flex_record=flex_record,
        )
        true_events = otd._easy_record_events(record=record, mark_encoder=context.mark_encoder)

        def sample_flex_requests(
            prefix_events: Sequence[tuple[float, float, int, Mapping[str, float]]],
            future_count: int,
        ) -> list[SampledRequest]:
            return _sampled_requests_from_flex_events(
                flex_events=otd._sample_future_events_from_prefix(
                    model=model,
                    dataset_bundle=flex_bundle,
                    dataset=flex_dataset,
                    prefix_events=prefix_events,
                    condition=flex_record.condition,
                    max_future_events=future_count,
                    device=device,
                    argmax=False,
                    mean_of=int(args.flex_mean_of),
                    median=False,
                ),
                flex_bundle=flex_bundle,
                mark_encoder=context.mark_encoder,
                prediction_task=prediction_task,
                medication_metadata=medication_metadata,
            )

        for prefix_len in otd._prefix_lengths(args, true_events):
            future_count = otd._future_event_count(args, true_events, prefix_len)
            if future_count <= 0:
                continue
            prefix_events = matched_flex_events[:prefix_len]
            row_count += _write_rows_for_prefix_samples(
                writer=writer,
                prediction_task=prediction_task,
                otd=otd,
                spec=spec,
                record=record,
                sequence_index=sequence_index,
                prefix_len=prefix_len,
                args=args,
                location_lookup=location_lookup,
                medication_metadata=medication_metadata,
                include_observed_events_json=include_observed_events_json,
                sample_fn=lambda prefix_events=prefix_events, future_count=future_count: (
                    sample_flex_requests(prefix_events, future_count)
                ),
            )
        if args.progress_every and (sequence_index + 1) % int(args.progress_every) == 0:
            _log(
                f"{prediction_task}/{spec.run_name}: processed "
                f"{sequence_index + 1}/{len(context.easy_records)} sequences; "
                f"rows={row_count}."
            )
    return row_count


def _write_rows_for_prefix_samples(
    *,
    writer: csv.DictWriter,
    prediction_task: str,
    otd: Any,
    spec: Any,
    record: Any,
    sequence_index: int,
    prefix_len: int,
    args: argparse.Namespace,
    location_lookup: EncounterLocationLookup,
    medication_metadata: MedicationMetadata | None,
    include_observed_events_json: bool,
    sample_fn: Any,
) -> int:
    observed_events = _observed_events_payload(record, prefix_len)
    observed_sequence_key = _hash_payload(
        {
            "patient_id": str(record.patient_id),
            "encounter_id": str(record.encounter_id),
            "sequence_start_timestamp": str(record.sequence_start_timestamp),
            "prefix_event_count": int(prefix_len),
            "events": observed_events,
        }
    )
    observed_events_json = _json_dumps(observed_events) if include_observed_events_json else ""
    anchor_timestamp = _anchor_timestamp(record, prefix_len)
    prefix_identifier = _prefix_id(
        prediction_task=prediction_task,
        spec=spec,
        record=record,
        sequence_index=sequence_index,
        prefix_len=prefix_len,
        observed_sequence_key=observed_sequence_key,
    )
    floor_fallback = _record_floor_fallback(record, prefix_len)
    row_count = 0
    for sample_index in range(int(args.num_samples)):
        sampled_sequence_id = f"{prefix_identifier}.sample{int(sample_index):03d}"
        base_row = _base_row_payload(
            prediction_task=prediction_task,
            spec=spec,
            record=record,
            sequence_index=sequence_index,
            prefix_len=prefix_len,
            prefix_id=prefix_identifier,
            observed_sequence_key=observed_sequence_key,
            observed_events_json=observed_events_json,
            sample_index=sample_index,
            sampled_sequence_id=sampled_sequence_id,
            anchor_timestamp=anchor_timestamp,
        )
        sample_started_at = time.perf_counter()
        sampled_requests = sample_fn()
        inference_seconds = round(time.perf_counter() - sample_started_at, 6)
        base_row["inference_seconds"] = inference_seconds
        if not sampled_requests:
            writer.writerow(_empty_sample_row(base_row))
            row_count += 1
            continue
        for sampled_event_index, sampled_request in enumerate(sampled_requests):
            writer.writerow(
                _request_row(
                    base_row=base_row,
                    sampled_request=sampled_request,
                    sampled_event_index=sampled_event_index,
                    location_lookup=location_lookup,
                    floor_fallback=floor_fallback,
                    anchor_timestamp=anchor_timestamp,
                )
            )
            row_count += 1
    return row_count


def _prepare_task_runtime(
    *,
    prediction_task: str,
    args: argparse.Namespace,
) -> tuple[Any, argparse.Namespace, Path, list[Any], Any, EncounterLocationLookup, MedicationMetadata | None]:
    if prediction_task == VITAL_TASK:
        otd = vital_otd
        comparison_summary_path = otd._coerce_comparison_summary_path(
            args.vital_comparison_summary_path
        )
        selected_runs = _parse_csv_set(args.selected_vital_runs)
        easy_config_path = args.vital_easy_config_path
    elif prediction_task == DELIVERY_TASK:
        delivery_otd._patch_base_evaluator()
        otd = delivery_otd.base
        comparison_summary_path = delivery_otd._coerce_comparison_summary_path(
            args.delivery_comparison_summary_path
        )
        selected_runs = _parse_csv_set(args.selected_delivery_runs)
        easy_config_path = args.delivery_easy_config_path
    else:
        raise ValueError(f"Unsupported prediction task '{prediction_task}'.")

    otd_args = _otd_args_for_task(
        args=args,
        prediction_task=prediction_task,
        easy_config_path=easy_config_path,
    )
    specs = otd._load_model_specs(
        comparison_summary_path=comparison_summary_path,
        stf_log_dir=args.stf_log_dir,
        selected_runs=selected_runs,
    )
    context = otd._build_evaluation_context(args=otd_args, model_specs=specs)
    canonical_training_config = otd._canonical_easy_training_config(
        args=otd_args,
        model_specs=specs,
    )
    dataset_config = canonical_training_config.dataset_config
    location_lookup = EncounterLocationLookup(
        annotated_data_files=dataset_config.annotated_data_files,
        patient_id_col=dataset_config.patient_id_col,
    )
    medication_metadata = (
        MedicationMetadata(context.easy_bundle.metadata)
        if prediction_task == DELIVERY_TASK
        else None
    )
    return (
        otd,
        otd_args,
        comparison_summary_path,
        specs,
        context,
        location_lookup,
        medication_metadata,
    )


def _generate_task_cache(
    *,
    writer: csv.DictWriter,
    prediction_task: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    (
        otd,
        otd_args,
        comparison_summary_path,
        specs,
        context,
        location_lookup,
        medication_metadata,
    ) = _prepare_task_runtime(prediction_task=prediction_task, args=args)
    device = otd._device_from_args(otd_args)
    _log(
        f"{prediction_task}: generating cache from {comparison_summary_path} on {device} "
        f"for {len(specs)} model(s), {len(context.easy_records)} sequence(s)."
    )

    row_count = 0
    started_at = time.perf_counter()
    for spec_index, spec in enumerate(specs, start=1):
        _log(
            f"{prediction_task}: sampling model {spec_index}/{len(specs)} "
            f"{spec.run_name} [{spec.family}/{spec.variant}]"
        )
        family = str(spec.family).lower()
        if family == "easytpp":
            row_count += _write_rows_for_easy_model(
                writer=writer,
                prediction_task=prediction_task,
                otd=otd,
                spec=spec,
                context=context,
                args=otd_args,
                device=device,
                location_lookup=location_lookup,
                medication_metadata=medication_metadata,
                include_observed_events_json=not args.omit_observed_events_json,
            )
        elif family == "multittpp":
            row_count += _write_rows_for_multittpp_model(
                writer=writer,
                prediction_task=prediction_task,
                otd=otd,
                spec=spec,
                context=context,
                args=otd_args,
                device=device,
                location_lookup=location_lookup,
                medication_metadata=medication_metadata,
                include_observed_events_json=not args.omit_observed_events_json,
            )
        elif family == "flextpp":
            row_count += _write_rows_for_flex_model(
                writer=writer,
                prediction_task=prediction_task,
                otd=otd,
                spec=spec,
                context=context,
                args=otd_args,
                device=device,
                location_lookup=location_lookup,
                medication_metadata=medication_metadata,
                include_observed_events_json=not args.omit_observed_events_json,
            )
        else:
            raise ValueError(f"Unsupported model family '{spec.family}'.")
        _log(
            f"{prediction_task}: finished {spec.run_name}; cumulative rows={row_count}."
        )

    return {
        "prediction_task": prediction_task,
        "comparison_summary_path": str(comparison_summary_path),
        "model_count": len(specs),
        "sequence_count": len(context.easy_records),
        "row_count": row_count,
        "duration_seconds": round(time.perf_counter() - started_at, 3),
    }


def _write_metadata(
    *,
    output_csv: Path,
    metadata_path: Path,
    args: argparse.Namespace,
    task_summaries: Sequence[Mapping[str, Any]],
) -> None:
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "output_csv": str(output_csv),
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "csv_fields": CSV_FIELDS,
        "task_summaries": list(task_summaries),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, default=str)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="OfflineRequestPredictionCache",
        description=(
            "Sample future vital-sign and medication-delivery request sequences "
            "for test-set prefixes and store them as an offline CSV prediction cache."
        ),
    )
    parser.add_argument(
        "--tasks",
        default="vital_sign,delivery",
        help="Comma-separated tasks to cache: vital_sign,delivery.",
    )
    parser.add_argument("--output_csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--metadata_path", default=None)
    parser.add_argument("--vital_comparison_summary_path", default=None)
    parser.add_argument("--delivery_comparison_summary_path", default=None)
    parser.add_argument("--selected_vital_runs", default=None)
    parser.add_argument("--selected_delivery_runs", default=None)
    parser.add_argument("--vital_easy_config_path", default=DEFAULT_VITAL_EASY_CONFIG_PATH)
    parser.add_argument("--delivery_easy_config_path", default=DEFAULT_DELIVERY_EASY_CONFIG_PATH)
    parser.add_argument("--stf_log_dir", default=None)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument(
        "--max_future_events",
        type=int,
        default=5,
        help="Maximum sampled future requests per prefix; use 0 for all remaining true events.",
    )
    parser.add_argument(
        "--min_prefix_events",
        type=int,
        default=1,
        help="Minimum observed requests in a prefix. Default 1 means after each observed request.",
    )
    parser.add_argument("--max_prefixes_per_sequence", type=int, default=None)
    parser.add_argument(
        "--prefix_subset_strategy",
        choices=("evenly_spaced", "first"),
        default="evenly_spaced",
    )
    parser.add_argument("--max_sequences", type=int, default=None)
    parser.add_argument(
        "--sequence_subset_strategy",
        choices=("first", "random"),
        default="first",
    )
    parser.add_argument(
        "--demand_level",
        choices=vital_otd.DEMAND_LEVEL_CHOICES,
        default="all",
    )
    parser.add_argument(
        "--demand_sequence_assignment",
        choices=("majority", "any", "strict"),
        default="majority",
    )
    parser.add_argument(
        "--use_saved_datasets",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--flex_mean_of", type=int, default=1)
    parser.add_argument("--easy_thinning_num_sample", type=int, default=16)
    parser.add_argument("--easy_thinning_num_exp", type=int, default=200)
    parser.add_argument("--easy_thinning_over_sample_rate", type=float, default=5.0)
    parser.add_argument("--easy_thinning_patience_counter", type=int, default=5)
    parser.add_argument("--easy_thinning_num_samples_boundary", type=int, default=5)
    parser.add_argument("--easy_thinning_dtime_max", type=float, default=24.0)
    parser.add_argument("--progress_every", type=int, default=50)
    parser.add_argument("--progress_sample_interval", type=int, default=250)
    parser.add_argument(
        "--omit_observed_events_json",
        action="store_true",
        help="Write only observed_sequence_key, not the full observed prefix JSON.",
    )
    return parser


def main() -> int:
    started_at = time.perf_counter()
    args = build_parser().parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    random.seed(int(args.seed))

    output_csv = _resolve_repo_relative_path(args.output_csv)
    if output_csv is None:
        output_csv = vital_otd._repo_root() / DEFAULT_OUTPUT_CSV
    metadata_path = (
        _resolve_repo_relative_path(args.metadata_path)
        if args.metadata_path
        else _metadata_path_for_csv(output_csv)
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    tasks = _tasks_in_safe_runtime_order(_parse_tasks(args.tasks))
    task_summaries: list[dict[str, Any]] = []
    _log(f"Writing offline prediction cache to {output_csv}")
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for prediction_task in tasks:
            task_summaries.append(
                _generate_task_cache(
                    writer=writer,
                    prediction_task=prediction_task,
                    args=args,
                )
            )
            csv_file.flush()

    _write_metadata(
        output_csv=output_csv,
        metadata_path=metadata_path,
        args=args,
        task_summaries=task_summaries,
    )
    _log(f"Offline prediction cache saved to {output_csv}")
    _log(f"Offline prediction metadata saved to {metadata_path}")
    _log(f"Finished in {vital_otd._format_duration(time.perf_counter() - started_at)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
