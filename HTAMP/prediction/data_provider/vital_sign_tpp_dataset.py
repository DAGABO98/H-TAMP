from __future__ import annotations

import argparse
import datetime
import json
import traceback
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from HTAMP.prediction.configs.monitoring_request_config import MonitoringRequestDatasetConfig
from HTAMP.prediction.configs.vital_sign_tpp_config import (
    VitalSignTPPDatasetConfig,
    VitalSignTPPModelConfig,
    VitalSignTPPTrainingConfig,
)
from HTAMP.prediction.data_provider.monitoring_requests_dataset import (
    ENCOUNTER_ID_COLUMN,
    FLOOR_COLUMN,
    SPLITS,
    TIME_COLUMNS,
    TIMESTAMP_COLUMN,
    RequestsDataManager,
    VITAL_OUTPUT_COMPONENTS,
    _event_measurement_column,
)
from HTAMP.prediction.point_process_models.flexTPP.dataset.base import (
    BatchSpec,
    ItemSpec,
    MODALITY_CATEGORICAL,
    MODALITY_CONTINUOUS,
    log_and_log_abs_det,
)

from HTAMP.prediction.point_process_models.flexTPP.dataset.property_mtpp import PropertyMTTPDataset

DATASET_VERSION = 3
DATASET_REPRESENTATION = "vital_sign_request_flex_tpp"
DATASET_FILENAME = "vital_sign_tpp_dataset.pt"
METADATA_FILENAME = "metadata.json"
EOS_EVENT_TYPE_NAME = "{EOS}"
NO_CONDITIONING_MODE = "none"
PREVIOUS_DAY_SUMMARY_CONDITIONING_MODE = "previous_day_summary"
WORKFLOW_IGNORED_CONFIG_FIELDS = (
    "preprocess_data",
    "save_data",
    "use_saved_request_data",
    "use_saved_dataset",
)
STANDARD_EVENT_TYPE_MARK_CONFIG_FIELDS = (
    "event_type_mark_mode",
    "label_strategy",
    "label_names",
    "quantile_edges",
    "measurement_thresholds",
    "label_component_by_task",
    "missing_label",
    "drop_missing_measurement_events",
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


def _dataset_config_snapshot(dataset_config: VitalSignTPPDatasetConfig) -> dict[str, object]:
    payload = dataset_config.to_dict()
    for field_name in WORKFLOW_IGNORED_CONFIG_FIELDS:
        payload.pop(field_name, None)
    if str(payload.get("event_type_mark_mode", "task")).strip().lower() == "task":
        for field_name in STANDARD_EVENT_TYPE_MARK_CONFIG_FIELDS:
            payload.pop(field_name, None)
    return _json_safe_value(payload)


def _build_request_dataset_config(
    dataset_config: VitalSignTPPDatasetConfig,
) -> MonitoringRequestDatasetConfig:
    return MonitoringRequestDatasetConfig.from_dict(
        {
            "annotated_data_files": dataset_config.annotated_data_files,
            "request_dir": dataset_config.request_dir,
            "dataset_dir": str(Path(dataset_config.dataset_dir) / "_monitoring_requests_cache"),
            "start_date": dataset_config.start_date,
            "end_date": dataset_config.end_date,
            "patient_id_col": dataset_config.patient_id_col,
            "included_tasks": dataset_config.included_tasks,
            "train_ratio": dataset_config.train_ratio,
            "val_ratio": dataset_config.val_ratio,
            "test_iso_weeks": dataset_config.test_iso_weeks,
            "test_iso_weeks_by_floor": dataset_config.test_iso_weeks_by_floor,
            "validation_split_strategy": dataset_config.validation_split_strategy,
            "validation_split_seed": dataset_config.validation_split_seed,
            "use_saved_request_data": dataset_config.use_saved_request_data,
            "use_saved_time_series": False,
            "preprocess_data": True,
            "save_data": False,
        }
    )


def _task_property_columns(
    task_name: str,
    *,
    include_time_features_as_properties: bool,
) -> list[tuple[str, str]]:
    property_columns = [
        (
            property_name,
            _event_measurement_column(task_name=task_name, component=property_name),
        )
        for property_name in VITAL_OUTPUT_COMPONENTS[task_name]
    ]
    if include_time_features_as_properties:
        property_columns.extend((time_column, time_column) for time_column in TIME_COLUMNS)
    return property_columns


def _as_finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric_value):
        return None
    return numeric_value


def _uses_enhanced_event_types(dataset_config: VitalSignTPPDatasetConfig) -> bool:
    return str(getattr(dataset_config, "event_type_mark_mode", "task")).strip().lower() != "task"


def _components_for_event_type_task(
    *,
    task_name: str,
    dataset_config: VitalSignTPPDatasetConfig,
) -> tuple[str, ...]:
    configured_components = dataset_config.label_component_by_task.get(task_name)
    if configured_components:
        available_components = set(VITAL_OUTPUT_COMPONENTS.get(task_name, []))
        for configured_component in configured_components:
            if configured_component not in available_components:
                raise ValueError(
                    f"label_component_by_task for '{task_name}' references unsupported "
                    f"component '{configured_component}'."
                )
        return tuple(configured_components)

    components = tuple(VITAL_OUTPUT_COMPONENTS.get(task_name, []))
    if not components:
        raise ValueError(f"No vital measurement components are registered for '{task_name}'.")
    return (components[0],)


def _event_type_label_options(dataset_config: VitalSignTPPDatasetConfig) -> tuple[str, ...]:
    if dataset_config.drop_missing_measurement_events:
        return tuple(dataset_config.label_names)
    return tuple(dataset_config.label_names) + (dataset_config.missing_label,)


def _joint_event_type_label_combinations(
    *,
    component_names: Sequence[str],
    label_options: Sequence[str],
) -> list[list[tuple[str, str]]]:
    return [
        list(zip(component_names, label_combination))
        for label_combination in product(label_options, repeat=len(component_names))
    ]


def _event_type_name(
    *,
    task_name: str,
    component_labels: Sequence[tuple[str, str]],
    event_type_mark_mode: str,
) -> str:
    if event_type_mark_mode == "task":
        return task_name

    if len(component_labels) > 1:
        label_suffix = "__".join(
            f"{component_name}_{label_name}"
            for component_name, label_name in component_labels
        )
        return f"{task_name}__{label_suffix}"

    if len(component_labels) != 1:
        raise ValueError("component_labels must contain at least one component label.")

    component_name, label_name = component_labels[0]
    if event_type_mark_mode == "task_component_label":
        return f"{task_name}__{component_name}__{label_name}"
    return f"{task_name}__{label_name}"


def _base_task_from_event_type_name(
    *,
    event_type_name: str,
    included_tasks: Sequence[str],
) -> str:
    for task_name in sorted((str(task) for task in included_tasks), key=len, reverse=True):
        if event_type_name == task_name or event_type_name.startswith(f"{task_name}__"):
            return task_name
    return str(event_type_name).split("__", 1)[0]


def _parse_threshold_pair(raw_threshold: Any, *, field_name: str) -> tuple[float, float]:
    if isinstance(raw_threshold, Mapping):
        lower = raw_threshold.get("lower", raw_threshold.get("low"))
        upper = raw_threshold.get("upper", raw_threshold.get("high"))
        if lower is None or upper is None:
            raise ValueError(
                f"{field_name} threshold mappings must include lower/upper or low/high."
            )
        lower_threshold, upper_threshold = float(lower), float(upper)
    else:
        if len(raw_threshold) != 2:
            raise ValueError(f"{field_name} thresholds must contain exactly two values.")
        lower_threshold, upper_threshold = float(raw_threshold[0]), float(raw_threshold[1])

    if not lower_threshold < upper_threshold:
        raise ValueError(f"{field_name} thresholds must satisfy lower < upper.")
    return lower_threshold, upper_threshold


def _threshold_for_component(
    *,
    task_name: str,
    component_name: str,
    measurement_thresholds: Mapping[str, Any],
) -> tuple[float, float] | None:
    nested_thresholds = measurement_thresholds.get(task_name)
    if isinstance(nested_thresholds, Mapping):
        component_threshold = nested_thresholds.get(component_name)
        if component_threshold is not None:
            return _parse_threshold_pair(
                component_threshold,
                field_name=f"measurement_thresholds.{task_name}.{component_name}",
            )

    for key in (
        f"{task_name}.{component_name}",
        f"{task_name}__{component_name}",
        f"{task_name}_{component_name}",
        task_name,
    ):
        raw_threshold = measurement_thresholds.get(key)
        if raw_threshold is not None:
            return _parse_threshold_pair(
                raw_threshold,
                field_name=f"measurement_thresholds.{key}",
            )
    return None


def _label_for_value(
    *,
    value: float | None,
    lower_threshold: float,
    upper_threshold: float,
    label_names: tuple[str, str, str],
    missing_label: str,
) -> str:
    if value is None:
        return missing_label
    if not lower_threshold < upper_threshold:
        return label_names[1]
    if value <= lower_threshold:
        return label_names[0]
    if value >= upper_threshold:
        return label_names[2]
    return label_names[1]


def _chunk_indices(length: int, max_chunk_size: Optional[int]) -> list[tuple[int, int]]:
    if length <= 0:
        return []
    if max_chunk_size is None or max_chunk_size >= length:
        return [(0, length)]
    return [
        (start_index, min(length, start_index + max_chunk_size))
        for start_index in range(0, length, max_chunk_size)
    ]


def _eos_gap_hours(eos_offset_minutes: float) -> float:
    return float(eos_offset_minutes) / 60.0


def _split_segment_frame_by_day(segment_df: pd.DataFrame) -> list[pd.DataFrame]:
    if segment_df.empty:
        return []

    dated_segment_df = segment_df.copy()
    dated_segment_df["_sequence_day"] = pd.to_datetime(
        dated_segment_df[TIMESTAMP_COLUMN]
    ).dt.normalize()
    return [
        daily_df.drop(columns="_sequence_day").reset_index(drop=True)
        for _, daily_df in dated_segment_df.groupby("_sequence_day", sort=False)
    ]

def _current_sequence_day(sequence_df: pd.DataFrame) -> pd.Timestamp:
    return pd.Timestamp(sequence_df[TIMESTAMP_COLUMN].iloc[0]).normalize()

def _measurement_summary_component_columns(
    included_tasks: Sequence[str],
) -> list[tuple[str, str, str]]:
    return [
        (task_name, component_name, _event_measurement_column(task_name=task_name, component=component_name))
        for task_name in included_tasks
        for component_name in VITAL_OUTPUT_COMPONENTS[task_name]
    ]


def _previous_day_condition_feature_names(
    included_tasks: Sequence[str],
) -> list[str]:
    feature_names = [
        "has_previous_day",
        "hours_since_previous_day_last_request",
        "previous_day_total_requests",
    ]
    feature_names.extend(
        f"previous_day_request_count_{task_name}"
        for task_name in included_tasks
    )
    feature_names.extend(
        f"previous_day_last_request_hour_{task_name}"
        for task_name in included_tasks
    )
    feature_names.extend(
        f"previous_day_last_{task_name}_{component_name}"
        for task_name, component_name, _ in _measurement_summary_component_columns(included_tasks)
    )
    feature_names.extend(
        f"previous_day_mean_{task_name}_{component_name}"
        for task_name, component_name, _ in _measurement_summary_component_columns(included_tasks)
    )
    return feature_names


def _build_previous_day_condition_vector(
    *,
    current_day_df: pd.DataFrame,
    previous_day_df: pd.DataFrame | None,
    included_tasks: Sequence[str],
) -> list[float]:
    current_start_timestamp = pd.Timestamp(current_day_df[TIMESTAMP_COLUMN].iloc[0])
    has_previous_day = previous_day_df is not None and not previous_day_df.empty

    if has_previous_day:
        previous_day_df = previous_day_df.sort_values(
            [TIMESTAMP_COLUMN, "task_index"],
            kind="mergesort",
        ).reset_index(drop=True)
        previous_day_last_timestamp = pd.Timestamp(previous_day_df[TIMESTAMP_COLUMN].iloc[-1])
        hours_since_previous_day_last_request = float(
            max(
                0.0,
                (current_start_timestamp - previous_day_last_timestamp).total_seconds() / 3600.0,
            )
        )
        previous_day_total_requests = float(len(previous_day_df))
    else:
        hours_since_previous_day_last_request = 0.0
        previous_day_total_requests = 0.0

    feature_values: list[float] = [
        1.0 if has_previous_day else 0.0,
        hours_since_previous_day_last_request,
        previous_day_total_requests,
    ]

    previous_day_by_task = {}
    if has_previous_day:
        previous_day_by_task = {
            task_name: task_df.sort_values(
                [TIMESTAMP_COLUMN, "task_index"],
                kind="mergesort",
            ).reset_index(drop=True)
            for task_name, task_df in previous_day_df.groupby("task_name", sort=False)
        }

    for task_name in included_tasks:
        task_df = previous_day_by_task.get(task_name)
        feature_values.append(float(len(task_df)) if task_df is not None else 0.0)

    for task_name in included_tasks:
        task_df = previous_day_by_task.get(task_name)
        if task_df is None or task_df.empty:
            feature_values.append(0.0)
            continue
        task_last_timestamp = pd.Timestamp(task_df[TIMESTAMP_COLUMN].iloc[-1])
        feature_values.append(
            float(
                task_last_timestamp.hour
                + (task_last_timestamp.minute / 60.0)
                + (task_last_timestamp.second / 3600.0)
            )
        )

    for task_name, _, source_column in _measurement_summary_component_columns(included_tasks):
        task_df = previous_day_by_task.get(task_name)
        if task_df is None or task_df.empty or source_column not in task_df.columns:
            feature_values.append(0.0)
            continue
        value_series = pd.to_numeric(task_df[source_column], errors="coerce").dropna()
        feature_values.append(float(value_series.iloc[-1]) if not value_series.empty else 0.0)

    for task_name, _, source_column in _measurement_summary_component_columns(included_tasks):
        task_df = previous_day_by_task.get(task_name)
        if task_df is None or task_df.empty or source_column not in task_df.columns:
            feature_values.append(0.0)
            continue
        value_series = pd.to_numeric(task_df[source_column], errors="coerce").dropna()
        feature_values.append(float(value_series.mean()) if not value_series.empty else 0.0)

    return feature_values


@dataclass
class VitalSignTPPSequenceRecord:
    split: str
    patient_id: str
    encounter_id: str
    segment_id: int
    sequence_start_timestamp: str
    sequence_end_timestamp: str
    events: list[tuple[float, float, int, dict[str, float]]]
    raw_events: list[dict[str, object]]
    condition: list[float] | None = None

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any] | "VitalSignTPPSequenceRecord",
    ) -> "VitalSignTPPSequenceRecord":
        if isinstance(payload, cls):
            return payload

        events = []
        for start_time, end_time, event_type, props in payload["events"]:
            events.append(
                (
                    float(start_time),
                    float(end_time),
                    int(event_type),
                    {
                        str(key): (
                            float(value)
                            if value is not None and not pd.isna(value)
                            else float("nan")
                        )
                        for key, value in dict(props).items()
                    },
                )
            )

        return cls(
            split=str(payload["split"]),
            patient_id=str(payload["patient_id"]),
            encounter_id=str(payload.get("encounter_id", "")),
            segment_id=int(payload["segment_id"]),
            sequence_start_timestamp=str(payload["sequence_start_timestamp"]),
            sequence_end_timestamp=str(payload["sequence_end_timestamp"]),
            events=events,
            raw_events=[dict(event_payload) for event_payload in payload.get("raw_events", [])],
            condition=(
                None
                if payload.get("condition") is None
                else [float(value) for value in payload.get("condition", [])]
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def encode_events_as_item_spec(
    *,
    events: Sequence[tuple[float, float, int, Mapping[str, float]]],
    property_types: Mapping[int, Mapping[str, int]],
    order: str,
    condition: Sequence[float] | np.ndarray | torch.Tensor | None = None,
) -> ItemSpec:
    event_vector_parts: list[float] = []
    log_abs_det_vector_parts: list[float] = []
    type_vector_parts: list[int] = []
    position_vector_parts: list[int] = []
    event_index_vector_parts: list[int] = []
    previous_time = 0.0

    for event_idx, (start_time, end_time, event_type, event_props) in enumerate(events):
        start_time_log, start_time_vol_change = log_and_log_abs_det(start_time - previous_time)
        duration_log, duration_vol_change = log_and_log_abs_det(end_time - start_time)
        position = 0
        for entry in order:
            if entry == "S":
                data = [float(start_time_log)]
                log_prob_correction = [float(start_time_vol_change)]
                dtypes = [MODALITY_CONTINUOUS]
            elif entry == "D":
                data = [float(duration_log)]
                log_prob_correction = [float(duration_vol_change)]
                dtypes = [MODALITY_CONTINUOUS]
            elif entry == "T":
                data = [float(event_type)]
                log_prob_correction = [0.0]
                dtypes = [MODALITY_CATEGORICAL]
            elif entry == "P":
                data = [float(event_props[property_name]) for property_name in event_props]
                log_prob_correction = [0.0] * len(data)
                dtypes = [
                    int(property_types[int(event_type)][property_name])
                    for property_name in event_props
                ]
            else:
                raise ValueError(f"Unsupported order entry '{entry}'.")

            event_vector_parts.extend(data)
            log_abs_det_vector_parts.extend(log_prob_correction)
            type_vector_parts.extend(dtypes)
            position_vector_parts.extend(range(position, position + len(data)))
            event_index_vector_parts.extend([event_idx] * len(data))
            position += len(data)

        previous_time = start_time

    data = torch.tensor(event_vector_parts, dtype=torch.float32)
    return ItemSpec(
        data=data,
        types=torch.tensor(type_vector_parts, dtype=torch.long),
        log_prob_correction=torch.tensor(log_abs_det_vector_parts, dtype=torch.float32),
        condition=(
            None
            if condition is None
            else torch.as_tensor(condition, dtype=torch.float32)
        ),
        extras={
            "position_in_event": torch.tensor(position_vector_parts, dtype=torch.long),
            "event_index": torch.tensor(event_index_vector_parts, dtype=torch.long),
        },
    )


def _single_item_batch_spec(item_spec: ItemSpec, device: torch.device) -> BatchSpec:
    return BatchSpec(
        data=item_spec.data.unsqueeze(0).to(device),
        types=item_spec.types.unsqueeze(0).to(device),
        log_prob_correction=(
            item_spec.log_prob_correction.unsqueeze(0).to(device)
            if item_spec.log_prob_correction is not None
            else None
        ),
        condition=(
            None
            if item_spec.condition is None
            else item_spec.condition.unsqueeze(0).to(device)
        ),
        extras={
            key: value.unsqueeze(0).to(device)
            for key, value in dict(item_spec.extras or {}).items()
        },
    )


class VitalSignTPPDataManager:
    def __init__(self, dataset_config: VitalSignTPPDatasetConfig) -> None:
        self.dataset_config = dataset_config
        self.dataset_dir = Path(dataset_config.dataset_dir)
        self.metadata: dict[str, object] = {}
        self.split_records: dict[str, list[VitalSignTPPSequenceRecord]] = {split: [] for split in SPLITS}

        if dataset_config.use_saved_dataset:
            try:
                self._load_dataset()
                return
            except Exception as load_error:
                if not dataset_config.preprocess_data:
                    raise ValueError(
                        "use_saved_dataset=True was requested, but the saved vital-sign TPP "
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
        request_data_manager = RequestsDataManager(
            dataset_config=_build_request_dataset_config(self.dataset_config)
        )
        train_data_df, train_segments_df = request_data_manager.get_requests_training_data()
        val_data_df, val_segments_df = request_data_manager.get_requests_validation_data()
        test_data_df, test_segments_df = request_data_manager.get_requests_testing_data()

        split_frames = {
            "train": train_data_df,
            "val": val_data_df,
            "test": test_data_df,
        }
        split_segments = {
            "train": train_segments_df,
            "val": val_segments_df,
            "test": test_segments_df,
        }
        event_type_context = self._build_event_type_context(split_frames=split_frames)
        self.split_records = self._build_split_records(
            split_frames=split_frames,
            split_segments=split_segments,
            event_type_context=event_type_context,
        )
        self.metadata = self._build_metadata(
            request_metadata=request_data_manager.metadata,
            event_type_context=event_type_context,
        )

    def _build_event_type_context(
        self,
        *,
        split_frames: Mapping[str, pd.DataFrame],
    ) -> dict[str, object]:
        property_schema_by_task = {
            task_name: [
                property_name
                for property_name, _ in _task_property_columns(
                    task_name,
                    include_time_features_as_properties=self.dataset_config.include_time_features_as_properties,
                )
            ]
            for task_name in self.dataset_config.included_tasks
        }
        component_by_task = (
            {
                task_name: _components_for_event_type_task(
                    task_name=task_name,
                    dataset_config=self.dataset_config,
                )
                for task_name in self.dataset_config.included_tasks
            }
            if _uses_enhanced_event_types(self.dataset_config)
            else {task_name: tuple() for task_name in self.dataset_config.included_tasks}
        )
        thresholds = self._build_event_type_thresholds(
            split_frames=split_frames,
            component_by_task=component_by_task,
        )
        event_types = self._build_event_type_names(component_by_task=component_by_task)
        property_schema_by_event_type = {
            event_type_name: property_schema_by_task[
                _base_task_from_event_type_name(
                    event_type_name=event_type_name,
                    included_tasks=self.dataset_config.included_tasks,
                )
            ]
            for event_type_name in event_types
        }
        return {
            "component_by_task": component_by_task,
            "thresholds": thresholds,
            "event_types": event_types,
            "property_schema_by_task": property_schema_by_task,
            "property_schema_by_event_type": property_schema_by_event_type,
        }

    def _build_event_type_thresholds(
        self,
        *,
        split_frames: Mapping[str, pd.DataFrame],
        component_by_task: Mapping[str, Sequence[str]],
    ) -> dict[str, dict[str, tuple[float, float]]]:
        thresholds: dict[str, dict[str, tuple[float, float]]] = {}
        if not _uses_enhanced_event_types(self.dataset_config):
            return {
                task_name: {}
                for task_name in self.dataset_config.included_tasks
            }

        if self.dataset_config.label_strategy == "threshold":
            for task_name, component_names in component_by_task.items():
                for component_name in component_names:
                    threshold_pair = _threshold_for_component(
                        task_name=task_name,
                        component_name=component_name,
                        measurement_thresholds=self.dataset_config.measurement_thresholds,
                    )
                    if threshold_pair is None:
                        raise ValueError(
                            "label_strategy='threshold' requires measurement_thresholds for "
                            f"{task_name}.{component_name}."
                        )
                    thresholds.setdefault(task_name, {})[component_name] = threshold_pair
            return thresholds

        values_by_key: dict[tuple[str, str], list[float]] = {
            (task_name, component_name): []
            for task_name, component_names in component_by_task.items()
            for component_name in component_names
        }
        train_df = split_frames.get("train", pd.DataFrame())
        for row in train_df.to_dict(orient="records"):
            task_name = str(row.get("task_name", ""))
            component_names = component_by_task.get(task_name)
            if component_names is None:
                continue
            for component_name in component_names:
                numeric_value = _as_finite_float(
                    row.get(_event_measurement_column(task_name=task_name, component=component_name))
                )
                if numeric_value is not None:
                    values_by_key[(task_name, component_name)].append(numeric_value)

        lower_quantile, upper_quantile = self.dataset_config.quantile_edges
        for (task_name, component_name), values in values_by_key.items():
            if not values:
                thresholds.setdefault(task_name, {})[component_name] = (0.0, 0.0)
                continue
            finite_values = np.asarray(values, dtype=np.float64)
            lower_threshold, upper_threshold = np.quantile(
                finite_values,
                [lower_quantile, upper_quantile],
            )
            thresholds.setdefault(task_name, {})[component_name] = (
                float(lower_threshold),
                float(upper_threshold),
            )
        return thresholds

    def _build_event_type_names(
        self,
        *,
        component_by_task: Mapping[str, Sequence[str]],
    ) -> list[str]:
        if not _uses_enhanced_event_types(self.dataset_config):
            return list(self.dataset_config.included_tasks)

        event_type_names: list[str] = []
        for task_name in self.dataset_config.included_tasks:
            component_names = component_by_task[task_name]
            for component_labels in _joint_event_type_label_combinations(
                component_names=component_names,
                label_options=_event_type_label_options(self.dataset_config),
            ):
                event_type_names.append(
                    _event_type_name(
                        task_name=task_name,
                        component_labels=component_labels,
                        event_type_mark_mode=self.dataset_config.event_type_mark_mode,
                    )
                )
        return event_type_names

    def _label_flex_event_type(
        self,
        *,
        row: Mapping[str, object],
        component_by_task: Mapping[str, Sequence[str]],
        thresholds: Mapping[str, Mapping[str, tuple[float, float]]],
    ) -> str | None:
        task_name = str(row.get("task_name", ""))
        if not _uses_enhanced_event_types(self.dataset_config):
            return task_name if task_name in self.dataset_config.included_tasks else None

        component_names = component_by_task.get(task_name)
        if component_names is None:
            return None

        component_labels: list[tuple[str, str]] = []
        for component_name in component_names:
            numeric_value = _as_finite_float(
                row.get(_event_measurement_column(task_name=task_name, component=component_name))
            )
            if numeric_value is None and self.dataset_config.drop_missing_measurement_events:
                return None

            lower_threshold, upper_threshold = thresholds[task_name][component_name]
            label_name = _label_for_value(
                value=numeric_value,
                lower_threshold=lower_threshold,
                upper_threshold=upper_threshold,
                label_names=self.dataset_config.label_names,
                missing_label=self.dataset_config.missing_label,
            )
            component_labels.append((component_name, label_name))

        return _event_type_name(
            task_name=task_name,
            component_labels=component_labels,
            event_type_mark_mode=self.dataset_config.event_type_mark_mode,
        )

    def _build_split_records(
        self,
        *,
        split_frames: Mapping[str, pd.DataFrame],
        split_segments: Mapping[str, pd.DataFrame],
        event_type_context: Mapping[str, object],
    ) -> dict[str, list[VitalSignTPPSequenceRecord]]:
        event_types = [str(event_type) for event_type in event_type_context["event_types"]]
        event_type_to_index = {
            event_type: event_index
            for event_index, event_type in enumerate(event_types)
        }
        component_by_task = {
            str(task_name): tuple(component_names)
            for task_name, component_names in dict(event_type_context["component_by_task"]).items()
        }
        thresholds = {
            str(task_name): {
                str(component_name): tuple(threshold_pair)
                for component_name, threshold_pair in dict(component_thresholds).items()
            }
            for task_name, component_thresholds in dict(event_type_context["thresholds"]).items()
        }
        property_columns_by_task = {
            task_name: _task_property_columns(
                task_name,
                include_time_features_as_properties=self.dataset_config.include_time_features_as_properties,
            )
            for task_name in self.dataset_config.included_tasks
        }

        daily_frame_lookup: dict[tuple[str, str, pd.Timestamp], pd.DataFrame] = {}
        for split_name in SPLITS:
            split_df = split_frames[split_name]
            segments_df = split_segments[split_name]
            if split_df.empty or segments_df.empty:
                continue

            for segment in segments_df.itertuples(index=False):
                start_idx = int(segment.start_idx)
                end_idx = int(segment.end_idx)
                segment_df = (
                    split_df.iloc[start_idx:end_idx]
                    .sort_values([TIMESTAMP_COLUMN, "task_index"], kind="mergesort")
                    .reset_index(drop=True)
                )
                if segment_df.empty:
                    continue

                encounter_id = (
                    ""
                    if pd.isna(getattr(segment, ENCOUNTER_ID_COLUMN, ""))
                    else str(getattr(segment, ENCOUNTER_ID_COLUMN, ""))
                )
                for daily_df in _split_segment_frame_by_day(segment_df):
                    daily_frame_lookup[
                        (
                            str(segment.patient_id),
                            encounter_id,
                            _current_sequence_day(daily_df),
                        )
                    ] = daily_df

        split_records: dict[str, list[VitalSignTPPSequenceRecord]] = {split: [] for split in SPLITS}
        for split_name in SPLITS:
            split_df = split_frames[split_name]
            segments_df = split_segments[split_name]
            if split_df.empty or segments_df.empty:
                continue

            for segment in segments_df.itertuples(index=False):
                start_idx = int(segment.start_idx)
                end_idx = int(segment.end_idx)
                segment_df = (
                    split_df.iloc[start_idx:end_idx]
                    .sort_values([TIMESTAMP_COLUMN, "task_index"], kind="mergesort")
                    .reset_index(drop=True)
                )
                if segment_df.empty:
                    continue

                encounter_id = (
                    ""
                    if pd.isna(getattr(segment, ENCOUNTER_ID_COLUMN, ""))
                    else str(getattr(segment, ENCOUNTER_ID_COLUMN, ""))
                )
                daily_frames = _split_segment_frame_by_day(segment_df)
                for daily_df in daily_frames:
                    previous_day_df = None
                    if self.dataset_config.use_previous_day_summary_conditioning:
                        previous_day_df = daily_frame_lookup.get(
                            (
                                str(segment.patient_id),
                                encounter_id,
                                _current_sequence_day(daily_df) - pd.Timedelta(days=1),
                            )
                        )
                    condition_vector = (
                        _build_previous_day_condition_vector(
                            current_day_df=daily_df,
                            previous_day_df=previous_day_df,
                            included_tasks=self.dataset_config.included_tasks,
                        )
                        if self.dataset_config.use_previous_day_summary_conditioning
                        else None
                    )
                    for chunk_start, chunk_end in (
                        _chunk_indices(len(daily_df), self.dataset_config.max_events_per_sequence)
                    ):
                        chunk_df = daily_df.iloc[chunk_start:chunk_end].reset_index(drop=True)
                        if len(chunk_df) < self.dataset_config.min_events_per_sequence:
                            continue

                        chunk_start_timestamp = pd.Timestamp(chunk_df[TIMESTAMP_COLUMN].iloc[0])
                        chunk_end_timestamp = pd.Timestamp(chunk_df[TIMESTAMP_COLUMN].iloc[-1])
                        encoded_events: list[tuple[float, float, int, dict[str, float]]] = []
                        raw_events: list[dict[str, object]] = []

                        for row in chunk_df.to_dict(orient="records"):
                            task_name = str(row["task_name"])
                            flex_event_type = self._label_flex_event_type(
                                row=row,
                                component_by_task=component_by_task,
                                thresholds=thresholds,
                            )
                            if flex_event_type is None:
                                continue
                            event_timestamp = pd.Timestamp(row[TIMESTAMP_COLUMN])
                            start_time_hours = float(
                                (event_timestamp - chunk_start_timestamp).total_seconds() / 3600.0
                            )
                            property_payload = {
                                property_name: (
                                    float(row[source_column])
                                    if row.get(source_column) is not None
                                    and not pd.isna(row.get(source_column))
                                    else float("nan")
                                )
                                for property_name, source_column in property_columns_by_task[task_name]
                            }
                            encoded_events.append(
                                (
                                    start_time_hours,
                                    start_time_hours,
                                    event_type_to_index[flex_event_type],
                                    property_payload,
                                )
                            )
                            raw_events.append(
                                {
                                    "timestamp": event_timestamp.isoformat(),
                                    "task_name": task_name,
                                    "flex_tpp_event_type": flex_event_type,
                                    "task_index": int(row["task_index"]),
                                    "floor": (
                                        None
                                        if row.get(FLOOR_COLUMN) is None or pd.isna(row.get(FLOOR_COLUMN))
                                        else int(row[FLOOR_COLUMN])
                                    ),
                                    "properties": {
                                        property_name: _json_safe_float(row.get(source_column))
                                        for property_name, source_column in property_columns_by_task[task_name]
                                    },
                                }
                            )

                        if len(encoded_events) < self.dataset_config.min_events_per_sequence:
                            continue

                        split_records[split_name].append(
                            VitalSignTPPSequenceRecord(
                                split=split_name,
                                patient_id=str(segment.patient_id),
                                encounter_id=encounter_id,
                                segment_id=len(split_records[split_name]),
                                sequence_start_timestamp=chunk_start_timestamp.isoformat(),
                                sequence_end_timestamp=chunk_end_timestamp.isoformat(),
                                events=encoded_events,
                                raw_events=raw_events,
                                condition=condition_vector,
                            )
                        )

        for split_name in SPLITS:
            if not split_records[split_name]:
                raise ValueError(
                    f"Split '{split_name}' has no vital-sign TPP sequences. "
                    "Adjust the date filters, split strategy, or sequence-length settings."
                )

        return split_records

    def _build_metadata(
        self,
        *,
        request_metadata: Mapping[str, object],
        event_type_context: Mapping[str, object],
    ) -> dict[str, object]:
        property_schema_by_task = {
            str(task_name): list(property_names)
            for task_name, property_names in dict(
                event_type_context["property_schema_by_task"]
            ).items()
        }
        property_schema_by_event_type = {
            str(event_type_name): list(property_names)
            for event_type_name, property_names in dict(
                event_type_context["property_schema_by_event_type"]
            ).items()
        }
        component_by_task = {
            str(task_name): list(component_names)
            for task_name, component_names in dict(event_type_context["component_by_task"]).items()
        }
        thresholds = {
            str(task_name): {
                str(component_name): list(threshold_pair)
                for component_name, threshold_pair in dict(component_thresholds).items()
            }
            for task_name, component_thresholds in dict(event_type_context["thresholds"]).items()
        }
        event_types = [str(event_type) for event_type in event_type_context["event_types"]]
        return {
            "version": DATASET_VERSION,
            "dataset_representation": DATASET_REPRESENTATION,
            "patient_id_col": self.dataset_config.patient_id_col,
            "encounter_id_col": ENCOUNTER_ID_COLUMN,
            "timestamp_col": TIMESTAMP_COLUMN,
            "included_tasks": list(self.dataset_config.included_tasks),
            "event_types": event_types,
            "eos_event_type_name": EOS_EVENT_TYPE_NAME,
            "property_schema_by_task": property_schema_by_task,
            "property_schema_by_event_type": property_schema_by_event_type,
            "include_time_features_as_properties": bool(
                self.dataset_config.include_time_features_as_properties
            ),
            "event_type_mark_mode": self.dataset_config.event_type_mark_mode,
            "event_type_mark_schema": (
                "enhanced" if _uses_enhanced_event_types(self.dataset_config) else "task"
            ),
            "label_strategy": self.dataset_config.label_strategy,
            "label_names": list(self.dataset_config.label_names),
            "missing_label": self.dataset_config.missing_label,
            "drop_missing_measurement_events": bool(
                self.dataset_config.drop_missing_measurement_events
            ),
            "label_component_by_task": component_by_task,
            "thresholds_by_task_component": thresholds,
            "sequence_boundary": "calendar_day",
            "conditioning_mode": (
                PREVIOUS_DAY_SUMMARY_CONDITIONING_MODE
                if self.dataset_config.use_previous_day_summary_conditioning
                else NO_CONDITIONING_MODE
            ),
            "condition_feature_names": (
                _previous_day_condition_feature_names(self.dataset_config.included_tasks)
                if self.dataset_config.use_previous_day_summary_conditioning
                else []
            ),
            "condition_dim": (
                len(_previous_day_condition_feature_names(self.dataset_config.included_tasks))
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
            "request_metadata": _json_safe_value(dict(request_metadata)),
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
            raise ValueError("Saved vital-sign TPP dataset version mismatch.")
        if self.metadata.get("config_snapshot") != _dataset_config_snapshot(self.dataset_config):
            raise ValueError(
                "Saved vital-sign TPP dataset does not match the current dataset configuration."
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
        model_config: VitalSignTPPModelConfig,
    ) -> "VitalSignTPPDatasetBundle":
        return VitalSignTPPDatasetBundle(
            split_records=self.split_records,
            metadata=self.metadata,
            model_config=model_config,
        )


class VitalSignTPPDataset(PropertyMTTPDataset):
    def __init__(
        self,
        *,
        sequence_records: Sequence[VitalSignTPPSequenceRecord],
        property_types: Mapping[int, Mapping[str, int]],
        eos_event_type: int,
        eos_offset_minutes: float,
        order: str = "STP",
        gaussian_except_start_time: bool = False,
    ) -> None:
        self.sequence_records = [
            VitalSignTPPSequenceRecord.from_dict(record)
            for record in sequence_records
        ]
        self.eos_offset_hours = _eos_gap_hours(eos_offset_minutes)
        self.condition_dim = max(
            (
                len(record.condition)
                for record in self.sequence_records
                if record.condition is not None
            ),
            default=0,
        )
        conditional = self.condition_dim > 0

        time_series = [
            (
                (
                    np.asarray(record.condition, dtype=np.float32)
                    if conditional and record.condition is not None
                    else (
                        np.zeros(self.condition_dim, dtype=np.float32)
                        if conditional
                        else None
                    )
                ),
                self._events_with_eos(record.events, eos_event_type=eos_event_type),
            )
            for record in self.sequence_records
        ]
        super().__init__(
            time_series=time_series,
            property_types=dict(property_types),
            eos_event_type=eos_event_type,
            overlapping=True,
            order=order,
            conditional=conditional,
            gaussian_except_start_time=gaussian_except_start_time,
        )

    def _events_with_eos(
        self,
        events: Sequence[tuple[float, float, int, Mapping[str, float]]],
        *,
        eos_event_type: int,
    ) -> list[tuple[float, float, int, dict[str, float]]]:
        encoded_events = [
            (
                float(start_time),
                float(end_time),
                int(event_type),
                {
                    str(key): (
                        float(value)
                        if value is not None and not pd.isna(value)
                        else float("nan")
                    )
                    for key, value in dict(event_props).items()
                },
            )
            for start_time, end_time, event_type, event_props in events
        ]
        if encoded_events:
            eos_start = encoded_events[-1][0] + self.eos_offset_hours
        else:
            eos_start = self.eos_offset_hours
        encoded_events.append((eos_start, eos_start, eos_event_type, {}))
        return encoded_events

    def get_raw_record(self, index: int) -> VitalSignTPPSequenceRecord:
        return self.sequence_records[index]


class VitalSignTPPSplitDataset(Dataset):
    def __init__(
        self,
        dataset_bundle: "VitalSignTPPDatasetBundle",
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


class VitalSignTPPDatasetBundle:
    def __init__(
        self,
        *,
        split_records: Mapping[str, Sequence[VitalSignTPPSequenceRecord | Mapping[str, Any]]],
        metadata: Mapping[str, Any],
        model_config: VitalSignTPPModelConfig | Mapping[str, Any],
    ) -> None:
        self.metadata = dict(metadata)
        self.model_config = VitalSignTPPModelConfig.from_dict(model_config)
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
        if not self.property_schema_by_event_type:
            self.property_schema_by_event_type = {
                event_type_name: self.property_schema_by_task[
                    _base_task_from_event_type_name(
                        event_type_name=event_type_name,
                        included_tasks=self.metadata.get("included_tasks", self.base_event_types),
                    )
                ]
                for event_type_name in self.base_event_types
            }
        self.condition_feature_names = [
            str(feature_name)
            for feature_name in self.metadata.get("condition_feature_names", [])
        ]
        self.condition_dim = len(self.condition_feature_names)
        self.property_types = {
            self.event_type_to_index[event_type_name]: {
                property_name: MODALITY_CONTINUOUS
                for property_name in property_names
            }
            for event_type_name, property_names in self.property_schema_by_event_type.items()
        }
        self.property_types[self.eos_event_type] = {}
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
        self.max_num_classes = max(1, len(self.event_types))
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


def build_vital_sign_tpp_dataset_bundle(
    dataset_config: VitalSignTPPDatasetConfig,
    model_config: VitalSignTPPModelConfig,
) -> VitalSignTPPDatasetBundle:
    data_manager = VitalSignTPPDataManager(dataset_config=dataset_config)
    return data_manager.get_dataset_bundle(model_config=model_config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="VitalSignTPPDataset",
        description="Create a vital-sign temporal point process dataset from a JSON config file.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to a JSON file containing at least 'dataset_config' and 'model_config'.",
    )
    return parser


if __name__ == "__main__":
    p_start = datetime.datetime.now()
    try:
        parser = build_parser()
        parsed_args = parser.parse_args()
        training_config = VitalSignTPPTrainingConfig.from_json_file(parsed_args.config_path)
        dataset_bundle = build_vital_sign_tpp_dataset_bundle(
            dataset_config=training_config.dataset_config,
            model_config=training_config.model_config,
        )
        for split_name in SPLITS:
            print(
                f"{split_name}: "
                f"{dataset_bundle.length(split_name)} sequences"
            )
        print("Process completed successfully.")
    except Exception as error_main_context:
        print("Fail End Process: ", error_main_context)
        traceback.print_exc()
    p_stop = datetime.datetime.now()
    print("Execution time: " + str(p_stop - p_start))
