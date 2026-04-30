from __future__ import annotations

import argparse
import datetime
import json
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from HTAMP.prediction.configs.delivery_tpp_config import (
    DeliveryMultiTTPPDatasetConfig,
    DeliveryMultiTTPPTrainingConfig,
)
from HTAMP.prediction.data_provider.delivery_easy_tpp_dataset import (
    DeliveryEasyTPPDataManager,
)
from HTAMP.prediction.data_provider.delivery_requests_dataset import SPLITS
from HTAMP.prediction.data_provider.vital_sign_easy_tpp_dataset import (
    EOS_EVENT_TYPE_NAME,
    VitalSignEasyTPPSequenceRecord,
)
from HTAMP.prediction.data_provider.vital_sign_multittpp_dataset import (
    VitalSignMultiTTPPCollator,
    VitalSignMultiTTPPDataset,
    VitalSignMultiTTPPDatasetBundle,
    VitalSignMultiTTPPSplitDataset,
    _record_without_eos,
)

DATASET_VERSION = 1
DATASET_REPRESENTATION = "delivery_request_multittpp"
DATASET_FILENAME = "delivery_multittpp_dataset.pt"
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


def _dataset_config_snapshot(dataset_config: DeliveryMultiTTPPDatasetConfig) -> dict[str, object]:
    payload = dataset_config.to_dict()
    for field_name in WORKFLOW_IGNORED_CONFIG_FIELDS:
        payload.pop(field_name, None)
    return _json_safe_value(payload)


def _positive_max(values: Sequence[float], default: float = 1.0) -> float:
    finite_values = [
        float(value)
        for value in values
        if np.isfinite(float(value)) and float(value) > 0.0
    ]
    if not finite_values:
        return float(default)
    return float(max(finite_values))


class DeliveryMultiTTPPDataManager:
    def __init__(self, dataset_config: DeliveryMultiTTPPDatasetConfig) -> None:
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
                        "use_saved_dataset=True was requested, but the saved delivery "
                        "MultiTTPP dataset could not be loaded."
                    ) from load_error

        if dataset_config.preprocess_data:
            self.dataset_dir.mkdir(parents=True, exist_ok=True)
            self._build_from_easy_tpp_sequences()
            if dataset_config.save_data:
                self._save_dataset()
        else:
            self._load_dataset()

    def _source_dataset_config(self) -> DeliveryMultiTTPPDatasetConfig:
        payload = self.dataset_config.to_dict()
        payload["include_eos_event"] = False
        payload["use_saved_dataset"] = False
        payload["preprocess_data"] = True
        payload["save_data"] = False
        return DeliveryMultiTTPPDatasetConfig.from_dict(payload)

    def _build_from_easy_tpp_sequences(self) -> None:
        source_manager = DeliveryEasyTPPDataManager(
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
                    f"Split '{split_name}' has no MultiTTPP delivery sequences. "
                    "Adjust date filters, split settings, or sequence-length settings."
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
            "included_tasks": ["medication"],
            "mark_names": mark_names,
            "event_types": mark_names,
            "mark_to_index": mark_to_index,
            "pad_token_id": len(mark_names),
            "num_event_types": len(mark_names),
            "num_event_types_pad": len(mark_names) + 1,
            "include_eos_event": False,
            "mark_schema": (
                "task_only"
                if self.dataset_config.mark_label_mode == "task"
                else "enhanced"
            ),
            "mark_label_mode": self.dataset_config.mark_label_mode,
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
            raise ValueError("Saved delivery MultiTTPP dataset version mismatch.")
        if self.metadata.get("config_snapshot") != _dataset_config_snapshot(self.dataset_config):
            raise ValueError(
                "Saved delivery MultiTTPP dataset does not match the current dataset configuration."
            )
        raw_split_records = dict(payload["split_records"])
        self.split_records = {
            split_name: [
                VitalSignEasyTPPSequenceRecord.from_dict(record_payload)
                for record_payload in raw_split_records.get(split_name, [])
            ]
            for split_name in SPLITS
        }

    def get_dataset_bundle(self) -> VitalSignMultiTTPPDatasetBundle:
        return VitalSignMultiTTPPDatasetBundle(
            split_records=self.split_records,
            metadata=self.metadata,
        )


DeliveryMultiTTPPSequenceRecord = VitalSignEasyTPPSequenceRecord
DeliveryMultiTTPPDataset = VitalSignMultiTTPPDataset
DeliveryMultiTTPPSplitDataset = VitalSignMultiTTPPSplitDataset
DeliveryMultiTTPPCollator = VitalSignMultiTTPPCollator
DeliveryMultiTTPPDatasetBundle = VitalSignMultiTTPPDatasetBundle


def build_delivery_multittpp_dataset_bundle(
    dataset_config: DeliveryMultiTTPPDatasetConfig,
) -> VitalSignMultiTTPPDatasetBundle:
    data_manager = DeliveryMultiTTPPDataManager(dataset_config=dataset_config)
    return data_manager.get_dataset_bundle()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="DeliveryMultiTTPPDataset",
        description="Create a MultiTTPP delivery event dataset from a JSON config file.",
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
        training_config = DeliveryMultiTTPPTrainingConfig.from_json_file(
            parsed_args.config_path
        )
        dataset_bundle = build_delivery_multittpp_dataset_bundle(
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
