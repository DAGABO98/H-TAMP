from __future__ import annotations

import argparse
import datetime
import json
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from HTAMP.prediction.configs.delivery_tpp_config import (
    DeliveryEasyTPPDatasetConfig,
    DeliveryEasyTPPTrainingConfig,
)
from HTAMP.prediction.data_provider.delivery_tpp_dataset import (
    DATASET_REPRESENTATION as SOURCE_DATASET_REPRESENTATION,
    DELIVERY_TASK_NAME,
    DeliveryTPPDataManager,
)
from HTAMP.prediction.data_provider.delivery_requests_dataset import SPLITS
from HTAMP.prediction.data_provider.vital_sign_easy_tpp_dataset import (
    VitalSignEasyTPPCollator,
    VitalSignEasyTPPDataset,
    VitalSignEasyTPPDatasetBundle,
    VitalSignEasyTPPSplitDataset,
    VitalSignEasyTPPSequenceRecord,
)
from HTAMP.prediction.data_provider.vital_sign_tpp_dataset import (
    EOS_EVENT_TYPE_NAME,
    VitalSignTPPSequenceRecord,
)

DATASET_VERSION = 1
DATASET_REPRESENTATION = "delivery_request_easy_tpp"
DATASET_FILENAME = "delivery_easy_tpp_dataset.pt"
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


def _dataset_config_snapshot(dataset_config: DeliveryEasyTPPDatasetConfig) -> dict[str, object]:
    payload = dataset_config.to_dict()
    for field_name in WORKFLOW_IGNORED_CONFIG_FIELDS:
        payload.pop(field_name, None)
    return _json_safe_value(payload)


class DeliveryEasyTPPDataManager:
    def __init__(self, dataset_config: DeliveryEasyTPPDatasetConfig) -> None:
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
                        "delivery dataset could not be loaded."
                    ) from load_error

        if dataset_config.preprocess_data:
            self.dataset_dir.mkdir(parents=True, exist_ok=True)
            self._build_from_delivery_sequences()
            if dataset_config.save_data:
                self._save_dataset()
        else:
            self._load_dataset()

    def _build_from_delivery_sequences(self) -> None:
        source_manager = DeliveryTPPDataManager(
            dataset_config=self.dataset_config.to_delivery_tpp_dataset_config(
                save_data=False
            )
        )
        mark_names = self._build_mark_names(source_metadata=source_manager.metadata)
        mark_to_index = {
            mark_name: mark_index
            for mark_index, mark_name in enumerate(mark_names)
        }
        self.split_records = self._encode_split_records(
            split_records=source_manager.split_records,
            mark_to_index=mark_to_index,
        )
        self.metadata = self._build_metadata(
            source_metadata=source_manager.metadata,
            mark_names=mark_names,
        )

    def _build_mark_names(self, *, source_metadata: Mapping[str, object]) -> list[str]:
        mark_names = [str(mark_name) for mark_name in source_metadata.get("event_types", [])]
        if not mark_names:
            mark_names = [DELIVERY_TASK_NAME]
        if self.dataset_config.include_eos_event:
            mark_names.append(EOS_EVENT_TYPE_NAME)
        return mark_names

    def _encode_split_records(
        self,
        *,
        split_records: Mapping[str, Sequence[VitalSignTPPSequenceRecord]],
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
                    mark_name = str(
                        raw_event.get(
                            "delivery_event_type",
                            DELIVERY_TASK_NAME,
                        )
                    )
                    if mark_name not in mark_to_index:
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
                            "delivery_event_type": EOS_EVENT_TYPE_NAME,
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
                    f"Split '{split_name}' has no EasyTPP delivery sequences. "
                    "Adjust date filters, split settings, or sequence-length settings."
                )
        return encoded_split_records

    def _build_metadata(
        self,
        *,
        source_metadata: Mapping[str, object],
        mark_names: Sequence[str],
    ) -> dict[str, object]:
        return {
            "version": DATASET_VERSION,
            "dataset_representation": DATASET_REPRESENTATION,
            "source_dataset_representation": SOURCE_DATASET_REPRESENTATION,
            "included_tasks": [DELIVERY_TASK_NAME],
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
            "mark_schema": (
                "enhanced"
                if self.dataset_config.mark_label_mode == "medication_code"
                else "task_only"
            ),
            "mark_label_mode": self.dataset_config.mark_label_mode,
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
            raise ValueError("Saved EasyTPP delivery dataset version mismatch.")
        if self.metadata.get("config_snapshot") != _dataset_config_snapshot(
            self.dataset_config
        ):
            raise ValueError(
                "Saved EasyTPP delivery dataset does not match the current "
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

    def get_dataset_bundle(self) -> VitalSignEasyTPPDatasetBundle:
        return VitalSignEasyTPPDatasetBundle(
            split_records=self.split_records,
            metadata=self.metadata,
        )


DeliveryEasyTPPSequenceRecord = VitalSignEasyTPPSequenceRecord
DeliveryEasyTPPDataset = VitalSignEasyTPPDataset
DeliveryEasyTPPSplitDataset = VitalSignEasyTPPSplitDataset
DeliveryEasyTPPCollator = VitalSignEasyTPPCollator
DeliveryEasyTPPDatasetBundle = VitalSignEasyTPPDatasetBundle


def build_delivery_easy_tpp_dataset_bundle(
    dataset_config: DeliveryEasyTPPDatasetConfig,
) -> VitalSignEasyTPPDatasetBundle:
    data_manager = DeliveryEasyTPPDataManager(dataset_config=dataset_config)
    return data_manager.get_dataset_bundle()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="DeliveryEasyTPPDataset",
        description="Create an EasyTPP delivery event dataset from a JSON config file.",
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
        training_config = DeliveryEasyTPPTrainingConfig.from_json_file(
            parsed_args.config_path
        )
        dataset_bundle = build_delivery_easy_tpp_dataset_bundle(
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
