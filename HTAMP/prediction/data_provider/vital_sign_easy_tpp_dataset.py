from __future__ import annotations

import argparse
import datetime
import json
import traceback
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from HTAMP.prediction.configs.vital_sign_easy_tpp_config import (
    VitalSignEasyTPPDatasetConfig,
    VitalSignEasyTPPTrainingConfig,
)
from HTAMP.prediction.data_provider.monitoring_requests_dataset import (
    SPLITS,
    VITAL_OUTPUT_COMPONENTS,
)
from HTAMP.prediction.data_provider.vital_sign_tpp_dataset import (
    EOS_EVENT_TYPE_NAME,
    VitalSignTPPDataManager,
    VitalSignTPPSequenceRecord,
)

DATASET_VERSION = 1
DATASET_REPRESENTATION = "vital_sign_request_easy_tpp"
DATASET_FILENAME = "vital_sign_easy_tpp_dataset.pt"
METADATA_FILENAME = "metadata.json"
WORKFLOW_IGNORED_CONFIG_FIELDS = (
    "preprocess_data",
    "save_data",
    "use_saved_request_data",
    "use_saved_dataset",
    "event_type_mark_mode",
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


def _dataset_config_snapshot(dataset_config: VitalSignEasyTPPDatasetConfig) -> dict[str, object]:
    payload = dataset_config.to_dict()
    for field_name in WORKFLOW_IGNORED_CONFIG_FIELDS:
        payload.pop(field_name, None)
    return _json_safe_value(payload)


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


def _components_for_task(
    *,
    task_name: str,
    dataset_config: VitalSignEasyTPPDatasetConfig,
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


def _uses_enhanced_marks(dataset_config: VitalSignEasyTPPDatasetConfig) -> bool:
    return dataset_config.mark_label_mode != "task_only"


def _mark_name(
    *,
    task_name: str,
    component_labels: Sequence[tuple[str, str]],
    mark_label_mode: str,
) -> str:
    if mark_label_mode == "task_only":
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
    if mark_label_mode == "task_component_label":
        return f"{task_name}__{component_name}__{label_name}"
    return f"{task_name}__{label_name}"


def _label_options(dataset_config: VitalSignEasyTPPDatasetConfig) -> tuple[str, ...]:
    if dataset_config.drop_missing_measurement_events:
        return tuple(dataset_config.label_names)
    return tuple(dataset_config.label_names) + (dataset_config.missing_label,)


def _joint_mark_label_combinations(
    *,
    component_names: Sequence[str],
    label_options: Sequence[str],
) -> list[list[tuple[str, str]]]:
    return [
        list(zip(component_names, label_combination))
        for label_combination in product(label_options, repeat=len(component_names))
    ]


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


@dataclass
class VitalSignEasyTPPSequenceRecord:
    split: str
    patient_id: str
    encounter_id: str
    segment_id: int
    sequence_start_timestamp: str
    sequence_end_timestamp: str
    time_seqs: list[float]
    time_delta_seqs: list[float]
    type_seqs: list[int]
    mark_names: list[str]
    raw_events: list[dict[str, object]]

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any] | "VitalSignEasyTPPSequenceRecord",
    ) -> "VitalSignEasyTPPSequenceRecord":
        if isinstance(payload, cls):
            return payload
        return cls(
            split=str(payload["split"]),
            patient_id=str(payload["patient_id"]),
            encounter_id=str(payload.get("encounter_id", "")),
            segment_id=int(payload["segment_id"]),
            sequence_start_timestamp=str(payload["sequence_start_timestamp"]),
            sequence_end_timestamp=str(payload["sequence_end_timestamp"]),
            time_seqs=[float(value) for value in payload["time_seqs"]],
            time_delta_seqs=[float(value) for value in payload["time_delta_seqs"]],
            type_seqs=[int(value) for value in payload["type_seqs"]],
            mark_names=[str(value) for value in payload["mark_names"]],
            raw_events=[dict(event_payload) for event_payload in payload.get("raw_events", [])],
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class VitalSignEasyTPPDataManager:
    def __init__(self, dataset_config: VitalSignEasyTPPDatasetConfig) -> None:
        self.dataset_config = dataset_config
        self.dataset_dir = Path(dataset_config.dataset_dir)
        self.metadata: dict[str, object] = {}
        self.split_records: dict[str, list[VitalSignEasyTPPSequenceRecord]] = {
            split: [] for split in SPLITS
        }

        if dataset_config.use_saved_dataset:
            try:
                self._load_dataset()
                return
            except Exception as load_error:
                if not dataset_config.preprocess_data:
                    raise ValueError(
                        "use_saved_dataset=True was requested, but the saved EasyTPP "
                        "vital-sign dataset could not be loaded."
                    ) from load_error

        if dataset_config.preprocess_data:
            self.dataset_dir.mkdir(parents=True, exist_ok=True)
            self._build_from_vital_sign_sequences()
            if dataset_config.save_data:
                self._save_dataset()
        else:
            self._load_dataset()

    def _build_from_vital_sign_sequences(self) -> None:
        sequence_manager = VitalSignTPPDataManager(
            dataset_config=self.dataset_config.to_vital_sign_tpp_dataset_config(
                save_data=False
            )
        )
        component_by_task = (
            {
                task_name: _components_for_task(
                    task_name=task_name,
                    dataset_config=self.dataset_config,
                )
                for task_name in self.dataset_config.included_tasks
            }
            if _uses_enhanced_marks(self.dataset_config)
            else {task_name: tuple() for task_name in self.dataset_config.included_tasks}
        )
        thresholds = self._build_thresholds(
            split_records=sequence_manager.split_records,
            component_by_task=component_by_task,
        )
        mark_names = self._build_mark_names(component_by_task=component_by_task)
        mark_to_index = {
            mark_name: mark_index
            for mark_index, mark_name in enumerate(mark_names)
        }
        self.split_records = self._encode_split_records(
            split_records=sequence_manager.split_records,
            component_by_task=component_by_task,
            thresholds=thresholds,
            mark_to_index=mark_to_index,
        )
        self.metadata = self._build_metadata(
            source_metadata=sequence_manager.metadata,
            component_by_task=component_by_task,
            thresholds=thresholds,
            mark_names=mark_names,
        )

    def _build_thresholds(
        self,
        *,
        split_records: Mapping[str, Sequence[VitalSignTPPSequenceRecord]],
        component_by_task: Mapping[str, Sequence[str]],
    ) -> dict[str, dict[str, tuple[float, float]]]:
        thresholds: dict[str, dict[str, tuple[float, float]]] = {}
        if not _uses_enhanced_marks(self.dataset_config):
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
        for record in split_records.get("train", []):
            for raw_event in record.raw_events:
                task_name = str(raw_event.get("task_name", ""))
                component_names = component_by_task.get(task_name)
                if component_names is None:
                    continue
                properties = dict(raw_event.get("properties", {}))
                for component_name in component_names:
                    numeric_value = _as_finite_float(properties.get(component_name))
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

    def _build_mark_names(
        self,
        *,
        component_by_task: Mapping[str, Sequence[str]],
    ) -> list[str]:
        mark_names: list[str] = []
        if not _uses_enhanced_marks(self.dataset_config):
            mark_names.extend(self.dataset_config.included_tasks)
            if self.dataset_config.include_eos_event:
                mark_names.append(EOS_EVENT_TYPE_NAME)
            return mark_names

        for task_name in self.dataset_config.included_tasks:
            component_names = component_by_task[task_name]
            for component_labels in _joint_mark_label_combinations(
                component_names=component_names,
                label_options=_label_options(self.dataset_config),
            ):
                mark_names.append(
                    _mark_name(
                        task_name=task_name,
                        component_labels=component_labels,
                        mark_label_mode=self.dataset_config.mark_label_mode,
                    )
                )

        if self.dataset_config.include_eos_event:
            mark_names.append(EOS_EVENT_TYPE_NAME)
        return mark_names

    def _label_event_mark(
        self,
        *,
        raw_event: Mapping[str, object],
        component_by_task: Mapping[str, Sequence[str]],
        thresholds: Mapping[str, Mapping[str, tuple[float, float]]],
    ) -> str | None:
        task_name = str(raw_event.get("task_name", ""))
        if not _uses_enhanced_marks(self.dataset_config):
            return task_name if task_name in self.dataset_config.included_tasks else None

        component_names = component_by_task.get(task_name)
        if component_names is None:
            return None

        properties = dict(raw_event.get("properties", {}))
        component_labels: list[tuple[str, str]] = []
        for component_name in component_names:
            numeric_value = _as_finite_float(properties.get(component_name))
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

        return _mark_name(
            task_name=task_name,
            component_labels=component_labels,
            mark_label_mode=self.dataset_config.mark_label_mode,
        )

    def _encode_split_records(
        self,
        *,
        split_records: Mapping[str, Sequence[VitalSignTPPSequenceRecord]],
        component_by_task: Mapping[str, Sequence[str]],
        thresholds: Mapping[str, Mapping[str, tuple[float, float]]],
        mark_to_index: Mapping[str, int],
    ) -> dict[str, list[VitalSignEasyTPPSequenceRecord]]:
        encoded_split_records: dict[str, list[VitalSignEasyTPPSequenceRecord]] = {
            split: [] for split in SPLITS
        }
        eos_offset_hours = float(self.dataset_config.eos_offset_minutes) / 60.0

        for split_name in SPLITS:
            for source_record in split_records.get(split_name, []):
                time_seqs: list[float] = []
                type_seqs: list[int] = []
                mark_names: list[str] = []
                raw_events: list[dict[str, object]] = []

                for event_payload, raw_event in zip(
                    source_record.events,
                    source_record.raw_events,
                ):
                    mark_name = self._label_event_mark(
                        raw_event=raw_event,
                        component_by_task=component_by_task,
                        thresholds=thresholds,
                    )
                    if mark_name is None:
                        continue

                    event_time = float(event_payload[0])
                    time_seqs.append(event_time)
                    type_seqs.append(int(mark_to_index[mark_name]))
                    mark_names.append(mark_name)
                    enriched_event = dict(raw_event)
                    enriched_event["easy_tpp_mark"] = mark_name
                    raw_events.append(enriched_event)

                if not time_seqs:
                    continue

                if self.dataset_config.include_eos_event:
                    eos_time = float(time_seqs[-1]) + eos_offset_hours
                    time_seqs.append(eos_time)
                    type_seqs.append(int(mark_to_index[EOS_EVENT_TYPE_NAME]))
                    mark_names.append(EOS_EVENT_TYPE_NAME)
                    raw_events.append(
                        {
                            "timestamp": source_record.sequence_end_timestamp,
                            "task_name": EOS_EVENT_TYPE_NAME,
                            "easy_tpp_mark": EOS_EVENT_TYPE_NAME,
                            "properties": {},
                        }
                    )

                time_delta_seqs = [0.0]
                for previous_time, current_time in zip(time_seqs[:-1], time_seqs[1:]):
                    time_delta_seqs.append(float(max(0.0, current_time - previous_time)))

                if len(time_seqs) < 2:
                    continue

                encoded_split_records[split_name].append(
                    VitalSignEasyTPPSequenceRecord(
                        split=split_name,
                        patient_id=source_record.patient_id,
                        encounter_id=source_record.encounter_id,
                        segment_id=len(encoded_split_records[split_name]),
                        sequence_start_timestamp=source_record.sequence_start_timestamp,
                        sequence_end_timestamp=source_record.sequence_end_timestamp,
                        time_seqs=time_seqs,
                        time_delta_seqs=time_delta_seqs,
                        type_seqs=type_seqs,
                        mark_names=mark_names,
                        raw_events=raw_events,
                    )
                )

        for split_name in SPLITS:
            if not encoded_split_records[split_name]:
                raise ValueError(
                    f"Split '{split_name}' has no EasyTPP vital-sign sequences. "
                    "Adjust the date filters, split strategy, or missing-measurement settings."
                )
        return encoded_split_records

    def _build_metadata(
        self,
        *,
        source_metadata: Mapping[str, object],
        component_by_task: Mapping[str, Sequence[str]],
        thresholds: Mapping[str, Mapping[str, tuple[float, float]]],
        mark_names: Sequence[str],
    ) -> dict[str, object]:
        return {
            "version": DATASET_VERSION,
            "dataset_representation": DATASET_REPRESENTATION,
            "included_tasks": list(self.dataset_config.included_tasks),
            "event_types": list(mark_names),
            "mark_names": list(mark_names),
            "mark_to_index": {
                mark_name: mark_index
                for mark_index, mark_name in enumerate(mark_names)
            },
            "pad_token_id": len(mark_names),
            "num_event_types": len(mark_names),
            "num_event_types_pad": len(mark_names) + 1,
            "eos_event_type_name": EOS_EVENT_TYPE_NAME,
            "include_eos_event": bool(self.dataset_config.include_eos_event),
            "label_strategy": self.dataset_config.label_strategy,
            "mark_schema": (
                "enhanced" if _uses_enhanced_marks(self.dataset_config) else "task_only"
            ),
            "label_names": list(self.dataset_config.label_names),
            "missing_label": self.dataset_config.missing_label,
            "mark_label_mode": self.dataset_config.mark_label_mode,
            "label_component_by_task": {
                task_name: list(component_names)
                for task_name, component_names in component_by_task.items()
            },
            "thresholds_by_task_component": {
                task_name: {
                    component_name: list(threshold_pair)
                    for component_name, threshold_pair in component_thresholds.items()
                }
                for task_name, component_thresholds in thresholds.items()
            },
            "split_sequence_counts": {
                split_name: len(self.split_records[split_name])
                for split_name in SPLITS
            },
            "split_event_counts": {
                split_name: int(
                    sum(len(record.type_seqs) for record in self.split_records[split_name])
                )
                for split_name in SPLITS
            },
            "source_dataset_metadata": _json_safe_value(dict(source_metadata)),
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
            raise ValueError("Saved EasyTPP vital-sign dataset version mismatch.")
        if self.metadata.get("config_snapshot") != _dataset_config_snapshot(
            self.dataset_config
        ):
            raise ValueError(
                "Saved EasyTPP vital-sign dataset does not match the current "
                "dataset configuration."
            )

        raw_split_records = dict(payload["split_records"])
        self.split_records = {
            split_name: [
                VitalSignEasyTPPSequenceRecord.from_dict(record_payload)
                for record_payload in raw_split_records.get(split_name, [])
            ]
            for split_name in SPLITS
        }

    def get_dataset_bundle(self) -> "VitalSignEasyTPPDatasetBundle":
        return VitalSignEasyTPPDatasetBundle(
            split_records=self.split_records,
            metadata=self.metadata,
        )


class VitalSignEasyTPPDataset(Dataset):
    def __init__(
        self,
        *,
        sequence_records: Sequence[VitalSignEasyTPPSequenceRecord | Mapping[str, Any]],
    ) -> None:
        self.sequence_records = [
            VitalSignEasyTPPSequenceRecord.from_dict(record)
            for record in sequence_records
        ]

    def __len__(self) -> int:
        return len(self.sequence_records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.sequence_records[index]
        return {
            "time_seqs": list(record.time_seqs),
            "time_delta_seqs": list(record.time_delta_seqs),
            "type_seqs": list(record.type_seqs),
        }

    def get_raw_record(self, index: int) -> VitalSignEasyTPPSequenceRecord:
        return self.sequence_records[index]


class VitalSignEasyTPPSplitDataset(Dataset):
    def __init__(
        self,
        dataset_bundle: "VitalSignEasyTPPDatasetBundle",
        split: str = "train",
    ) -> None:
        self.split = split
        self.dataset_bundle = dataset_bundle
        self.dataset = dataset_bundle.get_dataset(split)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.dataset[index]

    def get_raw_record(self, index: int) -> VitalSignEasyTPPSequenceRecord:
        return self.dataset.get_raw_record(index)


class VitalSignEasyTPPCollator:
    def __init__(self, *, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, features: Sequence[Mapping[str, object]]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("Cannot collate an empty EasyTPP batch.")

        batch_size = len(features)
        max_len = max(len(feature["type_seqs"]) for feature in features)
        time_seqs = torch.zeros((batch_size, max_len), dtype=torch.float32)
        time_delta_seqs = torch.zeros((batch_size, max_len), dtype=torch.float32)
        type_seqs = torch.full(
            (batch_size, max_len),
            fill_value=self.pad_token_id,
            dtype=torch.long,
        )
        seq_non_pad_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

        for batch_index, feature in enumerate(features):
            feature_len = len(feature["type_seqs"])
            time_seqs[batch_index, :feature_len] = torch.as_tensor(
                feature["time_seqs"],
                dtype=torch.float32,
            )
            time_delta_seqs[batch_index, :feature_len] = torch.as_tensor(
                feature["time_delta_seqs"],
                dtype=torch.float32,
            )
            type_seqs[batch_index, :feature_len] = torch.as_tensor(
                feature["type_seqs"],
                dtype=torch.long,
            )
            seq_non_pad_mask[batch_index, :feature_len] = True

        subsequent_mask = torch.triu(
            torch.ones((max_len, max_len), dtype=torch.bool),
            diagonal=1,
        )
        attention_key_pad_mask = ~seq_non_pad_mask[:, None, :]
        attention_mask = subsequent_mask[None, :, :] | attention_key_pad_mask
        return {
            "time_seqs": time_seqs,
            "time_delta_seqs": time_delta_seqs,
            "type_seqs": type_seqs,
            "seq_non_pad_mask": seq_non_pad_mask,
            "attention_mask": attention_mask,
        }


class VitalSignEasyTPPDatasetBundle:
    def __init__(
        self,
        *,
        split_records: Mapping[
            str,
            Sequence[VitalSignEasyTPPSequenceRecord | Mapping[str, Any]],
        ],
        metadata: Mapping[str, Any],
    ) -> None:
        self.metadata = dict(metadata)
        self.mark_names = [
            str(mark_name)
            for mark_name in self.metadata.get("mark_names", self.metadata.get("event_types", []))
        ]
        self.mark_to_index = {
            str(mark_name): int(mark_index)
            for mark_name, mark_index in dict(
                self.metadata.get("mark_to_index", {})
            ).items()
        }
        if not self.mark_to_index:
            self.mark_to_index = {
                mark_name: mark_index
                for mark_index, mark_name in enumerate(self.mark_names)
            }
        self.pad_token_id = int(self.metadata.get("pad_token_id", len(self.mark_names)))
        self.num_event_types = int(self.metadata.get("num_event_types", len(self.mark_names)))
        self.num_event_types_pad = int(
            self.metadata.get("num_event_types_pad", self.num_event_types + 1)
        )
        self.split_records = {
            split_name: [
                VitalSignEasyTPPSequenceRecord.from_dict(record)
                for record in split_records.get(split_name, [])
            ]
            for split_name in SPLITS
        }
        self.datasets = {
            split_name: VitalSignEasyTPPDataset(
                sequence_records=self.split_records[split_name]
            )
            for split_name in SPLITS
        }

    def get_dataset(self, split: str) -> VitalSignEasyTPPDataset:
        if split not in SPLITS:
            raise ValueError(f"Unsupported split '{split}'.")
        return self.datasets[split]

    def get_raw_records(self, split: str) -> list[VitalSignEasyTPPSequenceRecord]:
        if split not in SPLITS:
            raise ValueError(f"Unsupported split '{split}'.")
        return list(self.split_records[split])

    def length(self, split: str) -> int:
        return len(self.get_raw_records(split))

    def collator(self) -> VitalSignEasyTPPCollator:
        return VitalSignEasyTPPCollator(pad_token_id=self.pad_token_id)

    def max_event_time(self, split: str = "train") -> float | None:
        times = [
            float(time_value)
            for record in self.get_raw_records(split)
            for time_value in record.time_seqs
        ]
        if not times:
            return None
        return max(times)

    def log_inter_time_stats(self, split: str = "train") -> tuple[float, float]:
        deltas = [
            float(delta_value)
            for record in self.get_raw_records(split)
            for delta_value in record.time_delta_seqs[1:]
            if float(delta_value) > 0.0
        ]
        if not deltas:
            return 0.0, 1.0
        log_deltas = np.log(np.asarray(deltas, dtype=np.float64))
        std_value = float(log_deltas.std())
        return float(log_deltas.mean()), std_value if std_value > 0.0 else 1.0


def build_vital_sign_easy_tpp_dataset_bundle(
    dataset_config: VitalSignEasyTPPDatasetConfig,
) -> VitalSignEasyTPPDatasetBundle:
    data_manager = VitalSignEasyTPPDataManager(dataset_config=dataset_config)
    return data_manager.get_dataset_bundle()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="VitalSignEasyTPPDataset",
        description="Create an EasyTPP vital-sign event dataset from a JSON config file.",
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
        training_config = VitalSignEasyTPPTrainingConfig.from_json_file(
            parsed_args.config_path
        )
        dataset_bundle = build_vital_sign_easy_tpp_dataset_bundle(
            dataset_config=training_config.dataset_config,
        )
        for split_name in SPLITS:
            print(f"{split_name}: {dataset_bundle.length(split_name)} sequences")
        print("Process completed successfully.")
    except Exception as error_main_context:
        print("Fail End Process: ", error_main_context)
        traceback.print_exc()
    p_stop = datetime.datetime.now()
    print("Execution time: " + str(p_stop - p_start))
