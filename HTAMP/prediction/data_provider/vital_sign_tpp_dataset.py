from __future__ import annotations

import argparse
import datetime
import json
import traceback
from dataclasses import asdict, dataclass
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

DATASET_VERSION = 1
DATASET_REPRESENTATION = "vital_sign_request_flex_tpp"
DATASET_FILENAME = "vital_sign_tpp_dataset.pt"
METADATA_FILENAME = "metadata.json"
EOS_EVENT_TYPE_NAME = "{EOS}"
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


def _dataset_config_snapshot(dataset_config: VitalSignTPPDatasetConfig) -> dict[str, object]:
    payload = dataset_config.to_dict()
    for field_name in WORKFLOW_IGNORED_CONFIG_FIELDS:
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
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def encode_events_as_item_spec(
    *,
    events: Sequence[tuple[float, float, int, Mapping[str, float]]],
    property_types: Mapping[int, Mapping[str, int]],
    order: str,
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
        condition=None,
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
        condition=None,
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
        self.split_records = self._build_split_records(split_frames=split_frames, split_segments=split_segments)
        self.metadata = self._build_metadata(request_metadata=request_data_manager.metadata)

    def _build_split_records(
        self,
        *,
        split_frames: Mapping[str, pd.DataFrame],
        split_segments: Mapping[str, pd.DataFrame],
    ) -> dict[str, list[VitalSignTPPSequenceRecord]]:
        event_types = list(self.dataset_config.included_tasks)
        event_type_to_index = {
            event_type: event_index
            for event_index, event_type in enumerate(event_types)
        }
        property_columns_by_task = {
            task_name: _task_property_columns(
                task_name,
                include_time_features_as_properties=self.dataset_config.include_time_features_as_properties,
            )
            for task_name in self.dataset_config.included_tasks
        }

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

                for chunk_start, chunk_end in (
                    _chunk_indices(len(segment_df), self.dataset_config.max_events_per_sequence)
                ):
                    chunk_df = segment_df.iloc[chunk_start:chunk_end].reset_index(drop=True)
                    if len(chunk_df) < self.dataset_config.min_events_per_sequence:
                        continue

                    chunk_start_timestamp = pd.Timestamp(chunk_df[TIMESTAMP_COLUMN].iloc[0])
                    chunk_end_timestamp = pd.Timestamp(chunk_df[TIMESTAMP_COLUMN].iloc[-1])
                    encoded_events: list[tuple[float, float, int, dict[str, float]]] = []
                    raw_events: list[dict[str, object]] = []

                    for row in chunk_df.to_dict(orient="records"):
                        task_name = str(row["task_name"])
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
                                event_type_to_index[task_name],
                                property_payload,
                            )
                        )
                        raw_events.append(
                            {
                                "timestamp": event_timestamp.isoformat(),
                                "task_name": task_name,
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

                    split_records[split_name].append(
                        VitalSignTPPSequenceRecord(
                            split=split_name,
                            patient_id=str(segment.patient_id),
                            encounter_id=(
                                ""
                                if pd.isna(getattr(segment, ENCOUNTER_ID_COLUMN, ""))
                                else str(getattr(segment, ENCOUNTER_ID_COLUMN, ""))
                            ),
                            segment_id=len(split_records[split_name]),
                            sequence_start_timestamp=chunk_start_timestamp.isoformat(),
                            sequence_end_timestamp=chunk_end_timestamp.isoformat(),
                            events=encoded_events,
                            raw_events=raw_events,
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
        return {
            "version": DATASET_VERSION,
            "dataset_representation": DATASET_REPRESENTATION,
            "patient_id_col": self.dataset_config.patient_id_col,
            "encounter_id_col": ENCOUNTER_ID_COLUMN,
            "timestamp_col": TIMESTAMP_COLUMN,
            "included_tasks": list(self.dataset_config.included_tasks),
            "event_types": list(self.dataset_config.included_tasks),
            "eos_event_type_name": EOS_EVENT_TYPE_NAME,
            "property_schema_by_task": property_schema_by_task,
            "include_time_features_as_properties": bool(
                self.dataset_config.include_time_features_as_properties
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

        time_series = [
            (
                None,
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
            conditional=False,
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
        self.property_types = {
            self.event_type_to_index[task_name]: {
                property_name: MODALITY_CONTINUOUS
                for property_name in property_names
            }
            for task_name, property_names in self.property_schema_by_task.items()
        }
        self.property_types[self.eos_event_type] = {}
        self.max_properties_per_event = max(
            (len(property_names) for property_names in self.property_schema_by_task.values()),
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

    def encode_events(self, events: Sequence[tuple[float, float, int, Mapping[str, float]]]) -> ItemSpec:
        return encode_events_as_item_spec(
            events=events,
            property_types=self.property_types,
            order=self.model_config.order,
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
