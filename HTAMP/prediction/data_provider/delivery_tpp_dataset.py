from __future__ import annotations

import argparse
import datetime
import json
import logging
import re
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from HTAMP.planning.request_handler import GlobalRequestHandler
from HTAMP.prediction.configs.delivery_tpp_config import (
    DeliveryTPPDatasetConfig,
    DeliveryTPPModelConfig,
    DeliveryTPPTrainingConfig,
)
from HTAMP.prediction.data_provider.delivery_requests_dataset import (
    ENCOUNTER_ID_COLUMN,
    FLOOR_COLUMN,
    SPLITS,
    TIMESTAMP_COLUMN,
    DeliveryRequestsDataManager,
    _build_med_code_display_map,
    _build_vocab,
    _normalize_identifier_series,
    _vocab_index,
)
from HTAMP.prediction.data_provider.vital_sign_tpp_dataset import (
    EOS_EVENT_TYPE_NAME,
    VitalSignTPPDataset,
    VitalSignTPPSequenceRecord,
    _single_item_batch_spec,
    encode_events_as_item_spec,
)
from HTAMP.prediction.medication_mapping import MedicationMappingApplier
from HTAMP.prediction.point_process_models.flexTPP.dataset.base import (
    BatchSpec,
    ItemSpec,
    MODALITY_CATEGORICAL,
    MODALITY_CONTINUOUS,
)

DATASET_VERSION = 1
DATASET_REPRESENTATION = "delivery_request_flex_tpp"
DATASET_FILENAME = "delivery_tpp_dataset.pt"
METADATA_FILENAME = "metadata.json"
DELIVERY_TASK_NAME = "medication"
NO_CONDITIONING_MODE = "none"
PREVIOUS_DAY_SUMMARY_CONDITIONING_MODE = "previous_day_summary"
MEDICATION_CODE_PROPERTY = "medication_code_index"
LOGGER = logging.getLogger(__name__)
WORKFLOW_IGNORED_CONFIG_FIELDS = (
    "preprocess_data",
    "save_data",
    "use_saved_request_data",
    "use_saved_dataset",
)


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _json_safe_value(nested_value)
            for key, nested_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _json_safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    numeric_value = float(value)
    if not np.isfinite(numeric_value):
        return None
    return numeric_value


def _dataset_config_snapshot(dataset_config: DeliveryTPPDatasetConfig) -> dict[str, object]:
    payload = dataset_config.to_dict()
    for field_name in WORKFLOW_IGNORED_CONFIG_FIELDS:
        payload.pop(field_name, None)
    return _json_safe_value(payload)


def _uses_enhanced_event_types(dataset_config: DeliveryTPPDatasetConfig) -> bool:
    return str(dataset_config.event_type_mark_mode).strip().lower() != "task"


def _chunk_indices(length: int, max_chunk_size: Optional[int]) -> list[tuple[int, int]]:
    if length <= 0:
        return []
    if max_chunk_size is None or max_chunk_size >= length:
        return [(0, length)]
    return [
        (start_index, min(length, start_index + max_chunk_size))
        for start_index in range(0, length, max_chunk_size)
    ]


def _safe_mark_token(value: object) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip().lower())
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "unknown"


def _unique_medication_event_type_map(
    med_vocab: Sequence[str],
    *,
    unknown_medication_label: str,
) -> dict[str, str]:
    used_tokens: set[str] = set()
    mapping: dict[str, str] = {}
    for med_code in med_vocab:
        base_token = _safe_mark_token(med_code)
        token = base_token
        suffix = 2
        while token in used_tokens:
            token = f"{base_token}_{suffix}"
            suffix += 1
        used_tokens.add(token)
        mapping[str(med_code)] = f"{DELIVERY_TASK_NAME}__{token}"

    unknown_token = _safe_mark_token(unknown_medication_label)
    token = unknown_token
    suffix = 2
    while token in used_tokens:
        token = f"{unknown_token}_{suffix}"
        suffix += 1
    mapping[str(unknown_medication_label)] = f"{DELIVERY_TASK_NAME}__{token}"
    return mapping


def _episode_filter(
    frame: pd.DataFrame,
    *,
    patient_id: str,
    encounter_id: str,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    mask = frame["patient_id"].astype(str) == str(patient_id)
    if str(encounter_id).strip():
        mask &= (
            _normalize_identifier_series(frame[ENCOUNTER_ID_COLUMN])
            .fillna("")
            .astype(str)
            == str(encounter_id)
        )
    return frame.loc[mask].copy()


def _timed_log(start_time: float, message: str, *args: object) -> None:
    LOGGER.info("%s in %.1fs", message % args if args else message, time.perf_counter() - start_time)


def _normalize_encounter_value(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    normalized = _normalize_identifier_series(pd.Series([value])).iloc[0]
    if pd.isna(normalized):
        return ""
    return str(normalized)


def _build_episode_lookup(frame: pd.DataFrame) -> dict[str, dict[object, pd.DataFrame]]:
    empty_frame = frame.iloc[0:0].copy()
    if frame.empty:
        return {"patient": {}, "episode": {}, "empty": {"": empty_frame}}

    indexed_frame = frame.copy()
    indexed_frame["__patient_key"] = indexed_frame["patient_id"].astype(str)
    indexed_frame["__encounter_key"] = (
        _normalize_identifier_series(indexed_frame[ENCOUNTER_ID_COLUMN])
        .fillna("")
        .astype(str)
    )
    patient_lookup = {
        str(patient_id): group.drop(columns=["__patient_key", "__encounter_key"])
        .sort_values(TIMESTAMP_COLUMN, kind="mergesort")
        .reset_index(drop=True)
        for patient_id, group in indexed_frame.groupby("__patient_key", sort=False)
    }
    episode_lookup = {
        (str(patient_id), str(encounter_id)): group.drop(columns=["__patient_key", "__encounter_key"])
        .sort_values(TIMESTAMP_COLUMN, kind="mergesort")
        .reset_index(drop=True)
        for (patient_id, encounter_id), group in indexed_frame.groupby(
            ["__patient_key", "__encounter_key"],
            sort=False,
            dropna=False,
        )
        if str(encounter_id).strip()
    }
    return {"patient": patient_lookup, "episode": episode_lookup, "empty": {"": empty_frame}}


def _lookup_episode_frame(
    lookup: Mapping[str, Mapping[object, pd.DataFrame]],
    *,
    patient_id: str,
    encounter_id: str,
) -> pd.DataFrame:
    normalized_encounter_id = _normalize_encounter_value(encounter_id)
    empty_frame = lookup.get("empty", {}).get("", pd.DataFrame())
    if normalized_encounter_id:
        return lookup["episode"].get(
            (str(patient_id), normalized_encounter_id),
            empty_frame,
        )
    return lookup["patient"].get(str(patient_id), empty_frame)


def _filter_time_window(
    frame: pd.DataFrame,
    *,
    start_time: object,
    end_time: object,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    mask = frame[TIMESTAMP_COLUMN].between(
        pd.Timestamp(start_time),
        pd.Timestamp(end_time),
        inclusive="both",
    )
    return frame.loc[mask].copy()


def _segment_lookup_filter(
    lookup: Mapping[str, Mapping[object, pd.DataFrame]],
    segment: object,
) -> pd.DataFrame:
    frame = _lookup_episode_frame(
        lookup,
        patient_id=str(getattr(segment, "patient_id")),
        encounter_id=(
            ""
            if pd.isna(getattr(segment, ENCOUNTER_ID_COLUMN, ""))
            else str(getattr(segment, ENCOUNTER_ID_COLUMN, ""))
        ),
    )
    return _filter_time_window(
        frame,
        start_time=getattr(segment, "start_time"),
        end_time=getattr(segment, "end_time"),
    )


def _previous_day_condition_feature_names(
    *,
    vital_vocab: Sequence[str],
    med_vocab: Sequence[str],
) -> list[str]:
    feature_names = [
        "has_previous_day_context",
        "previous_day_total_vital_events",
        "previous_day_total_medication_admins",
    ]
    feature_names.extend(f"previous_day_vital_count_{_safe_mark_token(name)}" for name in vital_vocab)
    feature_names.extend(f"previous_day_vital_mean_{_safe_mark_token(name)}" for name in vital_vocab)
    feature_names.extend(f"previous_day_medication_admin_count_{_safe_mark_token(code)}" for code in med_vocab)
    return feature_names


def _build_previous_day_condition_vector(
    *,
    day_start: pd.Timestamp,
    vitals_group: pd.DataFrame,
    admin_group: pd.DataFrame,
    vital_vocab: Sequence[str],
    med_vocab: Sequence[str],
) -> list[float]:
    previous_day_start = pd.Timestamp(day_start) - pd.Timedelta(days=1)
    previous_day_end = pd.Timestamp(day_start)
    previous_vitals = vitals_group[
        (vitals_group[TIMESTAMP_COLUMN] >= previous_day_start)
        & (vitals_group[TIMESTAMP_COLUMN] < previous_day_end)
    ].copy()
    previous_admins = admin_group[
        (admin_group[TIMESTAMP_COLUMN] >= previous_day_start)
        & (admin_group[TIMESTAMP_COLUMN] < previous_day_end)
    ].copy()
    has_context = not previous_vitals.empty or not previous_admins.empty

    feature_values: list[float] = [
        1.0 if has_context else 0.0,
        float(len(previous_vitals)),
        float(len(previous_admins)),
    ]

    for vital_name in vital_vocab:
        feature_values.append(float((previous_vitals["vital_name"].astype(str) == str(vital_name)).sum()))

    for vital_name in vital_vocab:
        value_series = pd.to_numeric(
            previous_vitals.loc[
                previous_vitals["vital_name"].astype(str) == str(vital_name),
                "value",
            ],
            errors="coerce",
        ).dropna()
        feature_values.append(float(value_series.mean()) if not value_series.empty else 0.0)

    for med_code in med_vocab:
        feature_values.append(float((previous_admins["med_code"].astype(str) == str(med_code)).sum()))

    return feature_values


def _empty_medication_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "patient_id",
            ENCOUNTER_ID_COLUMN,
            TIMESTAMP_COLUMN,
            FLOOR_COLUMN,
            "med_code",
            "med_code_type",
            "med_display_name",
        ]
    )


def _build_delivery_helper(dataset_config: DeliveryTPPDatasetConfig) -> DeliveryRequestsDataManager:
    helper = DeliveryRequestsDataManager.__new__(DeliveryRequestsDataManager)
    helper.dataset_config = dataset_config
    helper.dataset_dir = Path(dataset_config.dataset_dir) / "_delivery_request_context_cache"
    helper.patient_id_col = dataset_config.patient_id_col
    helper._warned_missing_measurements = set()
    helper.admission_windows_df = helper._load_admission_windows()
    helper.resolved_medication_code_col = None
    helper.medication_mapping_applier = MedicationMappingApplier.from_dataset_config(dataset_config)
    helper.samples = {}
    helper.metadata = {}
    return helper


def _prepare_mapped_medication_frame(
    *,
    helper: DeliveryRequestsDataManager,
    med_df: pd.DataFrame,
    time_col: str,
) -> pd.DataFrame:
    if med_df.empty:
        return _empty_medication_frame()
    if time_col not in med_df.columns:
        raise ValueError(f"Medication time column '{time_col}' was not found in medication data.")

    if helper.resolved_medication_code_col is None:
        helper.resolved_medication_code_col = helper._resolve_medication_code_col(med_df=med_df)

    mapped_df = helper._prepare_base_event_df(df=med_df, time_col=time_col)
    if FLOOR_COLUMN not in mapped_df.columns:
        mapped_df[FLOOR_COLUMN] = pd.NA
    mapped_df = mapped_df.rename(columns={helper.patient_id_col: "patient_id"})
    mapped_df = helper.medication_mapping_applier.apply(
        mapped_df,
        med_name_col=helper.resolved_medication_code_col,
    )
    mapped_df = mapped_df.dropna(subset=["med_code"]).copy()
    if mapped_df.empty:
        return _empty_medication_frame()

    mapped_df = mapped_df[
        [
            "patient_id",
            ENCOUNTER_ID_COLUMN,
            TIMESTAMP_COLUMN,
            FLOOR_COLUMN,
            "med_code",
            "med_code_type",
            "med_display_name",
        ]
    ]
    return mapped_df.sort_values(
        ["patient_id", TIMESTAMP_COLUMN, "med_code", ENCOUNTER_ID_COLUMN],
        kind="mergesort",
    ).reset_index(drop=True)


class DeliveryTPPDataManager:
    def __init__(self, dataset_config: DeliveryTPPDatasetConfig) -> None:
        self.dataset_config = dataset_config
        self.dataset_dir = Path(dataset_config.dataset_dir)
        self.metadata: dict[str, object] = {}
        self.split_records: dict[str, list[VitalSignTPPSequenceRecord]] = {
            split: [] for split in SPLITS
        }
        self.resolved_medication_code_col: Optional[str] = None

        if dataset_config.use_saved_dataset:
            try:
                self._load_dataset()
                return
            except Exception as load_error:
                if not dataset_config.preprocess_data:
                    raise ValueError(
                        "use_saved_dataset=True was requested, but the saved delivery TPP "
                        "dataset could not be loaded."
                    ) from load_error

        if dataset_config.preprocess_data:
            self.dataset_dir.mkdir(parents=True, exist_ok=True)
            self._build_from_requests()
            if dataset_config.save_data:
                self._save_dataset()
        else:
            self._load_dataset()

    def _build_from_requests(self) -> None:
        build_start = time.perf_counter()
        LOGGER.info("Starting delivery TPP dataset build in %s", self.dataset_dir)
        helper = _build_delivery_helper(dataset_config=self.dataset_config)
        stage_start = time.perf_counter()
        request_handler = helper._build_request_handler()
        _timed_log(stage_start, "Built request handler")
        stage_start = time.perf_counter()
        vitals_df = helper._build_vitals_frame(request_handler=request_handler)
        _timed_log(stage_start, "Built vitals frame with %d rows", len(vitals_df))
        stage_start = time.perf_counter()
        admin_df, _ = helper._build_medication_frames(request_handler=request_handler)
        _timed_log(stage_start, "Built medication administration frame with %d rows", len(admin_df))
        stage_start = time.perf_counter()
        scheduled_df = _prepare_mapped_medication_frame(
            helper=helper,
            med_df=request_handler.med_df,
            time_col=self.dataset_config.medication_scheduled_time_col,
        )
        _timed_log(stage_start, "Built scheduled medication frame with %d rows", len(scheduled_df))
        self.resolved_medication_code_col = helper.resolved_medication_code_col

        if scheduled_df.empty:
            raise ValueError(
                "No scheduled medication delivery events were found for the configured "
                "date range and medication-time column."
            )

        stage_start = time.perf_counter()
        timeline_df = helper._build_timeline_df(
            vitals_df=vitals_df,
            admin_df=admin_df,
            orders_df=scheduled_df,
        )
        _timed_log(stage_start, "Built timeline frame with %d rows", len(timeline_df))
        if timeline_df.empty:
            raise ValueError(
                "No delivery TPP source events were found for the configured date range."
            )

        stage_start = time.perf_counter()
        _, segments_df = helper._build_split_segments(timeline_df=timeline_df)
        _timed_log(stage_start, "Built %d split segment rows", len(segments_df))
        stage_start = time.perf_counter()
        event_type_context = self._build_event_type_context(
            helper=helper,
            segments_df=segments_df,
            vitals_df=vitals_df,
            admin_df=admin_df,
            scheduled_df=scheduled_df,
        )
        _timed_log(stage_start, "Built event type context")
        stage_start = time.perf_counter()
        self.split_records = self._build_split_records(
            helper=helper,
            segments_df=segments_df,
            vitals_df=vitals_df,
            admin_df=admin_df,
            scheduled_df=scheduled_df,
            event_type_context=event_type_context,
        )
        _timed_log(stage_start, "Built split records")
        self.metadata = self._build_metadata(
            helper=helper,
            event_type_context=event_type_context,
        )
        _timed_log(build_start, "Finished delivery TPP dataset build")

    def _build_event_type_context(
        self,
        *,
        helper: DeliveryRequestsDataManager,
        segments_df: pd.DataFrame,
        vitals_df: pd.DataFrame,
        admin_df: pd.DataFrame,
        scheduled_df: pd.DataFrame,
    ) -> dict[str, object]:
        split_source_frames: dict[str, dict[str, list[pd.DataFrame]]] = {
            split_name: {"vitals": [], "admins": [], "scheduled": []}
            for split_name in SPLITS
        }
        lookup_start = time.perf_counter()
        vitals_lookup = _build_episode_lookup(vitals_df)
        admin_lookup = _build_episode_lookup(admin_df)
        scheduled_lookup = _build_episode_lookup(scheduled_df)
        _timed_log(lookup_start, "Built episode lookups for event type context")

        train_segments = segments_df[segments_df["split"].astype(str) == "train"]
        LOGGER.info(
            "Collecting train frames for vocabulary from %d/%d segments",
            len(train_segments),
            len(segments_df),
        )
        progress_every = max(1, len(train_segments) // 10) if len(train_segments) else 1
        for segment_index, segment in enumerate(train_segments.itertuples(index=False), start=1):
            split_source_frames["train"]["vitals"].append(
                _segment_lookup_filter(vitals_lookup, segment)
            )
            split_source_frames["train"]["admins"].append(
                _segment_lookup_filter(admin_lookup, segment)
            )
            split_source_frames["train"]["scheduled"].append(
                _segment_lookup_filter(scheduled_lookup, segment)
            )
            if segment_index % progress_every == 0 or segment_index == len(train_segments):
                LOGGER.info(
                    "Collected vocabulary source frames for %d/%d train segments",
                    segment_index,
                    len(train_segments),
                )

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
        train_scheduled_df = (
            pd.concat(split_source_frames["train"]["scheduled"], ignore_index=True)
            if split_source_frames["train"]["scheduled"]
            else pd.DataFrame(columns=scheduled_df.columns)
        )
        if train_scheduled_df.empty:
            raise ValueError("The training split did not contain any scheduled medication events.")

        med_vocab = _build_vocab(
            train_scheduled_df["med_code"].astype(str).tolist(),
            top_k=self.dataset_config.top_meds,
            min_count=self.dataset_config.min_med_count,
        )
        if not med_vocab:
            raise ValueError(
                "No medication identifiers met the training-vocabulary requirements. "
                "Adjust top_meds or min_med_count."
            )

        med_condition_source = train_admin_df
        if med_condition_source.empty:
            med_condition_source = train_scheduled_df
        med_condition_vocab = _build_vocab(
            med_condition_source["med_code"].astype(str).tolist(),
            top_k=self.dataset_config.top_meds,
            min_count=self.dataset_config.min_med_count,
        )
        vital_vocab = _build_vocab(
            train_vitals_df["vital_name"].astype(str).tolist(),
            top_k=self.dataset_config.top_vitals,
            min_count=1,
        )

        med_code_display_map = _build_med_code_display_map(
            med_vocab,
            train_scheduled_df,
            scheduled_df,
            train_admin_df,
            admin_df,
        )
        med_code_to_event_type = _unique_medication_event_type_map(
            med_vocab,
            unknown_medication_label=self.dataset_config.unknown_medication_label,
        )
        if _uses_enhanced_event_types(self.dataset_config):
            event_types = [
                med_code_to_event_type[str(med_code)]
                for med_code in med_vocab
            ]
            event_types.append(
                med_code_to_event_type[self.dataset_config.unknown_medication_label]
            )
        else:
            event_types = [DELIVERY_TASK_NAME]

        property_schema_by_task = {
            DELIVERY_TASK_NAME: (
                [MEDICATION_CODE_PROPERTY]
                if self.dataset_config.include_medication_code_as_property
                else []
            )
        }
        property_schema_by_event_type = {
            event_type_name: list(property_schema_by_task[DELIVERY_TASK_NAME])
            for event_type_name in event_types
        }
        property_types_by_event_type = {
            event_type_name: {
                property_name: "categorical"
                for property_name in property_schema_by_event_type[event_type_name]
            }
            for event_type_name in event_types
        }
        medication_code_vocab = list(med_vocab) + [self.dataset_config.unknown_medication_label]
        return {
            "event_types": event_types,
            "med_vocab": list(med_vocab),
            "medication_code_vocab": medication_code_vocab,
            "med_condition_vocab": list(med_condition_vocab),
            "vital_vocab": list(vital_vocab),
            "med_code_to_event_type": med_code_to_event_type,
            "med_code_display_map": med_code_display_map,
            "property_schema_by_task": property_schema_by_task,
            "property_schema_by_event_type": property_schema_by_event_type,
            "property_types_by_event_type": property_types_by_event_type,
        }

    def _event_type_for_med_code(
        self,
        *,
        med_code: str,
        event_type_context: Mapping[str, object],
    ) -> str:
        if not _uses_enhanced_event_types(self.dataset_config):
            return DELIVERY_TASK_NAME
        med_code_to_event_type = {
            str(key): str(value)
            for key, value in dict(event_type_context["med_code_to_event_type"]).items()
        }
        return med_code_to_event_type.get(
            str(med_code),
            med_code_to_event_type[self.dataset_config.unknown_medication_label],
        )

    def _build_split_records(
        self,
        *,
        helper: DeliveryRequestsDataManager,
        segments_df: pd.DataFrame,
        vitals_df: pd.DataFrame,
        admin_df: pd.DataFrame,
        scheduled_df: pd.DataFrame,
        event_type_context: Mapping[str, object],
    ) -> dict[str, list[VitalSignTPPSequenceRecord]]:
        event_types = [str(event_type) for event_type in event_type_context["event_types"]]
        event_type_to_index = {
            event_type: event_index
            for event_index, event_type in enumerate(event_types)
        }
        medication_code_vocab = [str(code) for code in event_type_context["medication_code_vocab"]]
        medication_code_to_index = _vocab_index(medication_code_vocab)
        unknown_medication_index = medication_code_to_index[self.dataset_config.unknown_medication_label]
        vital_vocab = [str(name) for name in event_type_context["vital_vocab"]]
        med_condition_vocab = [str(code) for code in event_type_context["med_condition_vocab"]]
        if _uses_enhanced_event_types(self.dataset_config):
            med_code_to_delivery_event_type = {
                str(key): str(value)
                for key, value in dict(event_type_context["med_code_to_event_type"]).items()
            }
            unknown_delivery_event_type = med_code_to_delivery_event_type[
                self.dataset_config.unknown_medication_label
            ]
        else:
            med_code_to_delivery_event_type = {}
            unknown_delivery_event_type = DELIVERY_TASK_NAME

        lookup_start = time.perf_counter()
        vitals_lookup = _build_episode_lookup(vitals_df)
        admin_lookup = _build_episode_lookup(admin_df)
        scheduled_lookup = _build_episode_lookup(scheduled_df)
        _timed_log(lookup_start, "Built episode lookups for split-record construction")

        split_records: dict[str, list[VitalSignTPPSequenceRecord]] = {
            split: [] for split in SPLITS
        }
        progress_every = max(1, len(segments_df) // 10) if len(segments_df) else 1
        for segment_index, segment in enumerate(segments_df.itertuples(index=False), start=1):
            split_name = str(segment.split)
            patient_id = str(segment.patient_id)
            encounter_id = (
                ""
                if pd.isna(getattr(segment, ENCOUNTER_ID_COLUMN, ""))
                else str(getattr(segment, ENCOUNTER_ID_COLUMN, ""))
            )
            segment_scheduled_df = _segment_lookup_filter(
                scheduled_lookup,
                segment,
            ).sort_values(
                [TIMESTAMP_COLUMN, "med_code", ENCOUNTER_ID_COLUMN],
                kind="mergesort",
            ).reset_index(drop=True)
            if segment_scheduled_df.empty:
                if segment_index % progress_every == 0 or segment_index == len(segments_df):
                    LOGGER.info(
                        "Built split records through %d/%d segments; counts=%s",
                        segment_index,
                        len(segments_df),
                        {split: len(records) for split, records in split_records.items()},
                    )
                continue

            patient_vitals_df = _lookup_episode_frame(
                vitals_lookup,
                patient_id=patient_id,
                encounter_id=encounter_id,
            )
            patient_admin_df = _lookup_episode_frame(
                admin_lookup,
                patient_id=patient_id,
                encounter_id=encounter_id,
            )
            segment_scheduled_df["_sequence_day"] = pd.to_datetime(
                segment_scheduled_df[TIMESTAMP_COLUMN]
            ).dt.normalize()
            for sequence_day, daily_df in segment_scheduled_df.groupby("_sequence_day", sort=False):
                day_start = pd.Timestamp(sequence_day)
                day_end = day_start + pd.Timedelta(days=1)
                daily_df = daily_df.drop(columns="_sequence_day").reset_index(drop=True)
                condition_vector = (
                    _build_previous_day_condition_vector(
                        day_start=day_start,
                        vitals_group=patient_vitals_df,
                        admin_group=patient_admin_df,
                        vital_vocab=vital_vocab,
                        med_vocab=med_condition_vocab,
                    )
                    if self.dataset_config.use_previous_day_summary_conditioning
                    else None
                )

                for chunk_start, chunk_end in _chunk_indices(
                    len(daily_df),
                    self.dataset_config.max_events_per_sequence,
                ):
                    chunk_df = daily_df.iloc[chunk_start:chunk_end].reset_index(drop=True)
                    if len(chunk_df) < self.dataset_config.min_events_per_sequence:
                        continue

                    encoded_events: list[tuple[float, float, int, dict[str, float]]] = []
                    raw_events: list[dict[str, object]] = []
                    for row in chunk_df.to_dict(orient="records"):
                        event_timestamp = pd.Timestamp(row[TIMESTAMP_COLUMN])
                        event_time_hours = float(
                            (event_timestamp - day_start).total_seconds() / 3600.0
                        )
                        med_code = str(row.get("med_code", self.dataset_config.unknown_medication_label))
                        medication_code_index = medication_code_to_index.get(
                            med_code,
                            unknown_medication_index,
                        )
                        delivery_event_type = med_code_to_delivery_event_type.get(
                            med_code,
                            unknown_delivery_event_type,
                        )
                        property_payload = (
                            {MEDICATION_CODE_PROPERTY: float(medication_code_index)}
                            if self.dataset_config.include_medication_code_as_property
                            else {}
                        )
                        encoded_events.append(
                            (
                                event_time_hours,
                                event_time_hours,
                                event_type_to_index[delivery_event_type],
                                property_payload,
                            )
                        )
                        raw_events.append(
                            {
                                "timestamp": event_timestamp.isoformat(),
                                "task_name": DELIVERY_TASK_NAME,
                                "delivery_event_type": delivery_event_type,
                                "medication_code": med_code,
                                "medication_code_index": int(medication_code_index),
                                "medication_code_type": str(row.get("med_code_type", "")),
                                "medication_display_name": str(row.get("med_display_name", "")),
                                "floor": (
                                    None
                                    if row.get(FLOOR_COLUMN) is None or pd.isna(row.get(FLOOR_COLUMN))
                                    else int(row[FLOOR_COLUMN])
                                ),
                                "properties": {
                                    MEDICATION_CODE_PROPERTY: int(medication_code_index)
                                }
                                if self.dataset_config.include_medication_code_as_property
                                else {},
                            }
                        )

                    if len(encoded_events) < self.dataset_config.min_events_per_sequence:
                        continue

                    split_records[split_name].append(
                        VitalSignTPPSequenceRecord(
                            split=split_name,
                            patient_id=patient_id,
                            encounter_id=encounter_id,
                            segment_id=len(split_records[split_name]),
                            sequence_start_timestamp=day_start.isoformat(),
                            sequence_end_timestamp=day_end.isoformat(),
                            events=encoded_events,
                            raw_events=raw_events,
                            condition=condition_vector,
                        )
                    )
            if segment_index % progress_every == 0 or segment_index == len(segments_df):
                LOGGER.info(
                    "Built split records through %d/%d segments; counts=%s",
                    segment_index,
                    len(segments_df),
                    {split: len(records) for split, records in split_records.items()},
                )

        for split_name in SPLITS:
            if not split_records[split_name]:
                raise ValueError(
                    f"Split '{split_name}' has no delivery TPP sequences. "
                    "Adjust date filters, split settings, or min_events_per_sequence."
                )
        return split_records

    def _build_metadata(
        self,
        *,
        helper: DeliveryRequestsDataManager,
        event_type_context: Mapping[str, object],
    ) -> dict[str, object]:
        vital_vocab = [str(name) for name in event_type_context["vital_vocab"]]
        med_condition_vocab = [str(code) for code in event_type_context["med_condition_vocab"]]
        return {
            "version": DATASET_VERSION,
            "dataset_representation": DATASET_REPRESENTATION,
            "patient_id_col": self.dataset_config.patient_id_col,
            "encounter_id_col": ENCOUNTER_ID_COLUMN,
            "timestamp_col": TIMESTAMP_COLUMN,
            "included_tasks": [DELIVERY_TASK_NAME],
            "event_types": [str(event_type) for event_type in event_type_context["event_types"]],
            "eos_event_type_name": EOS_EVENT_TYPE_NAME,
            "property_schema_by_task": _json_safe_value(
                dict(event_type_context["property_schema_by_task"])
            ),
            "property_schema_by_event_type": _json_safe_value(
                dict(event_type_context["property_schema_by_event_type"])
            ),
            "property_types_by_event_type": _json_safe_value(
                dict(event_type_context["property_types_by_event_type"])
            ),
            "event_type_mark_mode": self.dataset_config.event_type_mark_mode,
            "event_type_mark_schema": (
                "enhanced" if _uses_enhanced_event_types(self.dataset_config) else "task"
            ),
            "med_vocab": [str(code) for code in event_type_context["med_vocab"]],
            "medication_code_vocab": [
                str(code) for code in event_type_context["medication_code_vocab"]
            ],
            "med_condition_vocab": med_condition_vocab,
            "vital_vocab": vital_vocab,
            "med_code_display_map": {
                str(key): str(value)
                for key, value in dict(event_type_context["med_code_display_map"]).items()
            },
            "med_code_to_event_type": {
                str(key): str(value)
                for key, value in dict(event_type_context["med_code_to_event_type"]).items()
            },
            "sequence_boundary": "calendar_day",
            "conditioning_mode": (
                PREVIOUS_DAY_SUMMARY_CONDITIONING_MODE
                if self.dataset_config.use_previous_day_summary_conditioning
                else NO_CONDITIONING_MODE
            ),
            "condition_feature_names": (
                _previous_day_condition_feature_names(
                    vital_vocab=vital_vocab,
                    med_vocab=med_condition_vocab,
                )
                if self.dataset_config.use_previous_day_summary_conditioning
                else []
            ),
            "condition_dim": (
                len(
                    _previous_day_condition_feature_names(
                        vital_vocab=vital_vocab,
                        med_vocab=med_condition_vocab,
                    )
                )
                if self.dataset_config.use_previous_day_summary_conditioning
                else 0
            ),
            "eos_offset_minutes": float(self.dataset_config.eos_offset_minutes),
            "split_sequence_counts": {
                split_name: len(self.split_records[split_name])
                for split_name in SPLITS
            },
            "split_event_counts": {
                split_name: int(
                    sum(len(record.events) for record in self.split_records[split_name])
                )
                for split_name in SPLITS
            },
            "medication_code_col": self.resolved_medication_code_col,
            "medication_mapping": helper.medication_mapping_applier.to_metadata(),
            "config_snapshot": _dataset_config_snapshot(self.dataset_config),
        }

    def _dataset_path(self) -> Path:
        return self.dataset_dir / DATASET_FILENAME

    def _metadata_path(self) -> Path:
        return self.dataset_dir / METADATA_FILENAME

    def _save_dataset(self) -> None:
        payload = {
            "metadata": self.metadata,
            "split_records": {
                split_name: [record.to_dict() for record in records]
                for split_name, records in self.split_records.items()
            },
        }
        torch.save(payload, self._dataset_path())
        with self._metadata_path().open("w", encoding="utf-8") as metadata_file:
            json.dump(self.metadata, metadata_file, indent=2)

    def _load_dataset(self) -> None:
        payload = torch.load(self._dataset_path(), weights_only=False)
        self.metadata = dict(payload["metadata"])
        if int(self.metadata.get("version", -1)) != DATASET_VERSION:
            raise ValueError("Saved delivery TPP dataset version mismatch.")
        if self.metadata.get("config_snapshot") != _dataset_config_snapshot(self.dataset_config):
            raise ValueError(
                "Saved delivery TPP dataset does not match the current dataset configuration."
            )
        raw_split_records = dict(payload["split_records"])
        self.split_records = {
            split_name: [
                VitalSignTPPSequenceRecord.from_dict(record_payload)
                for record_payload in raw_split_records.get(split_name, [])
            ]
            for split_name in SPLITS
        }

    def get_dataset_bundle(
        self,
        model_config: DeliveryTPPModelConfig,
    ) -> "DeliveryTPPDatasetBundle":
        return DeliveryTPPDatasetBundle(
            split_records=self.split_records,
            metadata=self.metadata,
            model_config=model_config,
        )


def _property_type_value(raw_value: object) -> int:
    normalized = str(raw_value).strip().lower()
    if normalized in {"categorical", "category", "discrete", str(MODALITY_CATEGORICAL)}:
        return MODALITY_CATEGORICAL
    return MODALITY_CONTINUOUS


class DeliveryTPPDatasetBundle:
    def __init__(
        self,
        *,
        split_records: Mapping[str, Sequence[VitalSignTPPSequenceRecord | Mapping[str, Any]]],
        metadata: Mapping[str, Any],
        model_config: DeliveryTPPModelConfig | Mapping[str, Any],
    ) -> None:
        self.metadata = dict(metadata)
        self.model_config = DeliveryTPPModelConfig.from_dict(model_config)
        self.base_event_types = list(self.metadata.get("event_types", []))
        self.eos_event_type_name = str(self.metadata.get("eos_event_type_name", EOS_EVENT_TYPE_NAME))
        self.event_types = self.base_event_types + [self.eos_event_type_name]
        self.event_type_to_index = {
            event_type: event_index
            for event_index, event_type in enumerate(self.event_types)
        }
        self.eos_event_type = self.event_type_to_index[self.eos_event_type_name]
        self.property_schema_by_task = {
            str(task_name): list(property_names)
            for task_name, property_names in dict(
                self.metadata.get("property_schema_by_task", {})
            ).items()
        }
        self.property_schema_by_event_type = {
            str(event_type_name): list(property_names)
            for event_type_name, property_names in dict(
                self.metadata.get("property_schema_by_event_type", {})
            ).items()
        }
        raw_property_types_by_event_type = {
            str(event_type_name): {
                str(property_name): _property_type_value(property_type)
                for property_name, property_type in dict(property_types).items()
            }
            for event_type_name, property_types in dict(
                self.metadata.get("property_types_by_event_type", {})
            ).items()
        }
        self.property_types = {
            self.event_type_to_index[event_type_name]: {
                property_name: raw_property_types_by_event_type.get(
                    event_type_name,
                    {},
                ).get(property_name, MODALITY_CONTINUOUS)
                for property_name in property_names
            }
            for event_type_name, property_names in self.property_schema_by_event_type.items()
        }
        self.property_types[self.eos_event_type] = {}
        self.condition_feature_names = [
            str(feature_name)
            for feature_name in self.metadata.get("condition_feature_names", [])
        ]
        self.condition_dim = len(self.condition_feature_names)
        self.max_properties_per_event = max(
            (
                len(property_names)
                for property_names in self.property_schema_by_event_type.values()
            ),
            default=0,
        )
        self.dims = [
            1
            + 1
            + (
                self.max_properties_per_event
                if "P" in self.model_config.order
                else 0
            )
        ]
        self.max_num_classes = max(
            1,
            len(self.event_types),
            len(self.metadata.get("medication_code_vocab", [])),
        )
        self.split_records = {
            split_name: [
                VitalSignTPPSequenceRecord.from_dict(record)
                for record in split_records.get(split_name, [])
            ]
            for split_name in SPLITS
        }
        if self.condition_dim == 0:
            self.condition_dim = max(
                (
                    len(record.condition)
                    for records in self.split_records.values()
                    for record in records
                    if record.condition is not None
                ),
                default=0,
            )
        self.datasets = {
            split_name: VitalSignTPPDataset(
                sequence_records=self.split_records[split_name],
                property_types=self.property_types,
                eos_event_type=self.eos_event_type,
                eos_offset_minutes=float(self.metadata.get("eos_offset_minutes", 5.0)),
                order=self.model_config.order,
                gaussian_except_start_time=self.model_config.gaussian_except_start_time,
            )
            for split_name in SPLITS
        }

    def get_dataset(self, split: str) -> VitalSignTPPDataset:
        if split not in SPLITS:
            raise ValueError(f"Unsupported split '{split}'.")
        return self.datasets[split]

    def get_raw_records(self, split: str) -> list[VitalSignTPPSequenceRecord]:
        if split not in SPLITS:
            raise ValueError(f"Unsupported split '{split}'.")
        return list(self.split_records[split])

    def length(self, split: str) -> int:
        return len(self.get_raw_records(split))

    def encode_events(
        self,
        events: Sequence[tuple[float, float, int, Mapping[str, float]]],
        *,
        condition: Sequence[float] | np.ndarray | torch.Tensor | None = None,
    ) -> ItemSpec:
        return encode_events_as_item_spec(
            events=events,
            property_types=self.property_types,
            order=self.model_config.order,
            condition=condition,
        )

    def item_to_batch(self, item_spec: ItemSpec, device: torch.device) -> BatchSpec:
        return _single_item_batch_spec(item_spec=item_spec, device=device)


class DeliveryTPPSplitDataset(Dataset):
    def __init__(
        self,
        dataset_bundle: DeliveryTPPDatasetBundle,
        split: str = "train",
    ) -> None:
        self.split = split
        self.dataset_bundle = dataset_bundle
        self.dataset = dataset_bundle.get_dataset(split)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> ItemSpec:
        return self.dataset[index]

    def get_raw_record(self, index: int) -> VitalSignTPPSequenceRecord:
        return self.dataset.get_raw_record(index)


def build_delivery_tpp_dataset_bundle(
    dataset_config: DeliveryTPPDatasetConfig,
    model_config: DeliveryTPPModelConfig,
) -> DeliveryTPPDatasetBundle:
    data_manager = DeliveryTPPDataManager(dataset_config=dataset_config)
    return data_manager.get_dataset_bundle(model_config=model_config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="DeliveryTPPDataset",
        description="Create a delivery temporal point process dataset from a JSON config file.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to a JSON file containing at least 'dataset_config' and 'model_config'.",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    p_start = datetime.datetime.now()
    try:
        parser = build_parser()
        parsed_args = parser.parse_args()
        training_config = DeliveryTPPTrainingConfig.from_json_file(parsed_args.config_path)
        dataset_bundle = build_delivery_tpp_dataset_bundle(
            dataset_config=training_config.dataset_config,
            model_config=training_config.model_config,
        )
        for split_name in SPLITS:
            print(f"{split_name}: {dataset_bundle.length(split_name)} sequences")
        print("Process completed successfully.")
    except Exception as error_main_context:
        print("Fail End Process: ", error_main_context)
        traceback.print_exc()
    p_stop = datetime.datetime.now()
    print("Execution time: " + str(p_stop - p_start))
