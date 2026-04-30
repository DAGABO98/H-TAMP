from __future__ import annotations

import argparse
import datetime
import json
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from HTAMP.prediction.configs.vital_sign_multittpp_config import (
    VitalSignMultiTTPPDatasetConfig,
    VitalSignMultiTTPPTrainingConfig,
)
from HTAMP.prediction.data_provider.request_events_dataset import SPLITS
from HTAMP.prediction.data_provider.vital_sign_easy_tpp_dataset import (
    EOS_EVENT_TYPE_NAME,
    VitalSignEasyTPPDataManager,
    VitalSignEasyTPPSequenceRecord,
)
from HTAMP.prediction.point_process_models.multittpp.data import Batch

DATASET_VERSION = 1
DATASET_REPRESENTATION = "vital_sign_request_multittpp"
DATASET_FILENAME = "vital_sign_multittpp_dataset.pt"
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


def _dataset_config_snapshot(dataset_config: VitalSignMultiTTPPDatasetConfig) -> dict[str, object]:
    payload = dataset_config.to_dict()
    for field_name in WORKFLOW_IGNORED_CONFIG_FIELDS:
        payload.pop(field_name, None)
    return _json_safe_value(payload)


def _record_without_eos(record: VitalSignEasyTPPSequenceRecord) -> VitalSignEasyTPPSequenceRecord:
    keep_count = len(record.type_seqs)
    for index, mark_name in enumerate(record.mark_names):
        if mark_name == EOS_EVENT_TYPE_NAME:
            keep_count = index
            break
    if keep_count == len(record.type_seqs):
        return record
    payload = record.to_dict()
    payload["time_seqs"] = record.time_seqs[:keep_count]
    payload["time_delta_seqs"] = record.time_delta_seqs[:keep_count]
    payload["type_seqs"] = record.type_seqs[:keep_count]
    payload["mark_names"] = record.mark_names[:keep_count]
    payload["raw_events"] = record.raw_events[:keep_count]
    return VitalSignEasyTPPSequenceRecord.from_dict(payload)


def _positive_max(values: Sequence[float], default: float = 1.0) -> float:
    finite_values = [
        float(value)
        for value in values
        if np.isfinite(float(value)) and float(value) > 0.0
    ]
    if not finite_values:
        return float(default)
    return float(max(finite_values))


class VitalSignMultiTTPPDataManager:
    def __init__(self, dataset_config: VitalSignMultiTTPPDatasetConfig) -> None:
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
                        "use_saved_dataset=True was requested, but the saved vital-sign "
                        "MultiTTPP dataset could not be loaded."
                    ) from load_error

        if dataset_config.preprocess_data:
            self.dataset_dir.mkdir(parents=True, exist_ok=True)
            self._build_from_easy_tpp_sequences()
            if dataset_config.save_data:
                self._save_dataset()
        else:
            self._load_dataset()

    def _source_dataset_config(self) -> VitalSignMultiTTPPDatasetConfig:
        payload = self.dataset_config.to_dict()
        payload["include_eos_event"] = False
        payload["use_saved_dataset"] = False
        payload["preprocess_data"] = True
        payload["save_data"] = False
        return VitalSignMultiTTPPDatasetConfig.from_dict(payload)

    def _build_from_easy_tpp_sequences(self) -> None:
        source_manager = VitalSignEasyTPPDataManager(
            dataset_config=self._source_dataset_config()
        )
        self.split_records = {
            split_name: [
                _record_without_eos(record)
                for record in source_manager.split_records.get(split_name, [])
                if len(_record_without_eos(record).type_seqs) >= 2
            ]
            for split_name in SPLITS
        }
        for split_name in SPLITS:
            if not self.split_records[split_name]:
                raise ValueError(
                    f"Split '{split_name}' has no MultiTTPP vital-sign sequences. "
                    "Adjust the date filters, split strategy, or sequence-length settings."
                )
        self.metadata = self._build_metadata(source_metadata=source_manager.metadata)

    def _build_metadata(self, *, source_metadata: Mapping[str, object]) -> dict[str, object]:
        mark_names = [
            str(mark_name)
            for mark_name in source_metadata.get(
                "mark_names",
                source_metadata.get("event_types", []),
            )
            if str(mark_name) != EOS_EVENT_TYPE_NAME
        ]
        if not mark_names:
            max_type = max(
                int(type_value)
                for records in self.split_records.values()
                for record in records
                for type_value in record.type_seqs
            )
            mark_names = [str(mark_index) for mark_index in range(max_type + 1)]
        mark_to_index = {mark_name: mark_index for mark_index, mark_name in enumerate(mark_names)}
        all_times = [
            float(time_value)
            for records in self.split_records.values()
            for record in records
            for time_value in record.time_seqs
        ]
        all_deltas = [
            float(delta_value)
            for records in self.split_records.values()
            for record in records
            for delta_value in record.time_delta_seqs[1:]
        ]
        return {
            "version": DATASET_VERSION,
            "dataset_representation": DATASET_REPRESENTATION,
            "included_tasks": list(self.dataset_config.included_tasks),
            "mark_names": mark_names,
            "event_types": mark_names,
            "mark_to_index": mark_to_index,
            "pad_token_id": len(mark_names),
            "num_event_types": len(mark_names),
            "num_event_types_pad": len(mark_names) + 1,
            "include_eos_event": False,
            "label_strategy": self.dataset_config.label_strategy,
            "mark_schema": (
                "task_only"
                if self.dataset_config.mark_label_mode == "task_only"
                else "enhanced"
            ),
            "label_names": list(self.dataset_config.label_names),
            "missing_label": self.dataset_config.missing_label,
            "mark_label_mode": self.dataset_config.mark_label_mode,
            "label_component_by_task": {
                task_name: list(component_names)
                for task_name, component_names in self.dataset_config.label_component_by_task.items()
            },
            "t_max_normalization": _positive_max(all_times),
            "dt_max_normalization": _positive_max(all_deltas),
            "n_events": max(
                len(record.type_seqs)
                for records in self.split_records.values()
                for record in records
            ),
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
            raise ValueError("Saved vital-sign MultiTTPP dataset version mismatch.")
        if self.metadata.get("config_snapshot") != _dataset_config_snapshot(self.dataset_config):
            raise ValueError(
                "Saved vital-sign MultiTTPP dataset does not match the current dataset configuration."
            )
        raw_split_records = dict(payload["split_records"])
        self.split_records = {
            split_name: [
                VitalSignEasyTPPSequenceRecord.from_dict(record_payload)
                for record_payload in raw_split_records.get(split_name, [])
            ]
            for split_name in SPLITS
        }

    def get_dataset_bundle(self) -> "VitalSignMultiTTPPDatasetBundle":
        return VitalSignMultiTTPPDatasetBundle(
            split_records=self.split_records,
            metadata=self.metadata,
        )


class VitalSignMultiTTPPDataset(Dataset):
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
            "timestamps": list(record.time_seqs),
            "intervals": list(record.time_delta_seqs),
            "types": list(record.type_seqs),
            "length": int(len(record.type_seqs)),
        }

    def get_raw_record(self, index: int) -> VitalSignEasyTPPSequenceRecord:
        return self.sequence_records[index]


class VitalSignMultiTTPPSplitDataset(Dataset):
    def __init__(
        self,
        dataset_bundle: "VitalSignMultiTTPPDatasetBundle",
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


class VitalSignMultiTTPPCollator:
    def __init__(self, *, n_marks: int, n_min: int) -> None:
        self.n_marks = int(n_marks)
        self.n_min = int(n_min)

    def __call__(self, features: Sequence[Mapping[str, object]]) -> Batch:
        if not features:
            raise ValueError("Cannot collate an empty MultiTTPP batch.")
        features = sorted(features, key=lambda feature: int(feature["length"]), reverse=True)
        seq_lengths = torch.tensor(
            [int(feature["length"]) for feature in features],
            dtype=torch.long,
        )
        max_in_len = max(int(seq_lengths.max().item()) - 1, 1)
        if max_in_len < self.n_min:
            max_in_len = self.n_min

        batch_size = len(features)
        in_dts = torch.zeros((batch_size, max_in_len), dtype=torch.float32)
        out_dts = torch.zeros((batch_size, max_in_len), dtype=torch.float32)
        in_types = torch.full((batch_size, max_in_len), self.n_marks, dtype=torch.long)
        out_types = torch.full((batch_size, max_in_len), self.n_marks, dtype=torch.long)
        in_times = torch.zeros((batch_size, max_in_len), dtype=torch.float32)
        last_times = torch.zeros((batch_size, self.n_marks), dtype=torch.float32)

        for batch_index, feature in enumerate(features):
            timestamps = torch.as_tensor(feature["timestamps"], dtype=torch.float32)
            intervals = torch.as_tensor(feature["intervals"], dtype=torch.float32)
            types = torch.as_tensor(feature["types"], dtype=torch.long)
            if len(timestamps) < 2:
                continue
            input_len = len(timestamps) - 1
            in_dts[batch_index, :input_len] = intervals[:-1]
            out_dts[batch_index, :input_len] = intervals[1:]
            in_types[batch_index, :input_len] = types[:-1]
            out_types[batch_index, :input_len] = types[1:]
            in_times[batch_index, :input_len] = timestamps[:-1]
            last_times[batch_index, :] = timestamps[-2]

        return Batch(
            in_dts=in_dts,
            in_types=in_types,
            in_times=in_times,
            seq_lengths=seq_lengths,
            last_times=last_times,
            out_dts=out_dts,
            out_types=out_types,
            N_min=self.n_min,
        )


class VitalSignMultiTTPPDatasetBundle:
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
        self.t_max_normalization = float(self.metadata.get("t_max_normalization", 1.0))
        self.dt_max_normalization = float(self.metadata.get("dt_max_normalization", 1.0))
        self.n_events = int(self.metadata.get("n_events", 1))
        self.split_records = {
            split_name: [
                VitalSignEasyTPPSequenceRecord.from_dict(record)
                for record in split_records.get(split_name, [])
            ]
            for split_name in SPLITS
        }
        self.datasets = {
            split_name: VitalSignMultiTTPPDataset(
                sequence_records=self.split_records[split_name]
            )
            for split_name in SPLITS
        }

    def get_dataset(self, split: str) -> VitalSignMultiTTPPDataset:
        if split not in SPLITS:
            raise ValueError(f"Unsupported split '{split}'.")
        return self.datasets[split]

    def get_raw_records(self, split: str) -> list[VitalSignEasyTPPSequenceRecord]:
        if split not in SPLITS:
            raise ValueError(f"Unsupported split '{split}'.")
        return list(self.split_records[split])

    def length(self, split: str) -> int:
        return len(self.get_raw_records(split))

    def collator(self, *, n_min: int) -> VitalSignMultiTTPPCollator:
        return VitalSignMultiTTPPCollator(
            n_marks=self.num_event_types,
            n_min=n_min,
        )


def build_vital_sign_multittpp_dataset_bundle(
    dataset_config: VitalSignMultiTTPPDatasetConfig,
) -> VitalSignMultiTTPPDatasetBundle:
    data_manager = VitalSignMultiTTPPDataManager(dataset_config=dataset_config)
    return data_manager.get_dataset_bundle()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="VitalSignMultiTTPPDataset",
        description="Create a MultiTTPP vital-sign event dataset from a JSON config file.",
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
        training_config = VitalSignMultiTTPPTrainingConfig.from_json_file(
            parsed_args.config_path
        )
        dataset_bundle = build_vital_sign_multittpp_dataset_bundle(
            dataset_config=training_config.dataset_config,
        )
        for split_name in SPLITS:
            print(f"{split_name}: {dataset_bundle.length(split_name)} sequences")
    except Exception as error:
        print("Fail End Process: ", error)
        traceback.print_exc()
    finally:
        p_stop = datetime.datetime.now()
        print("Execution time: " + str(p_stop - p_start))
