from __future__ import annotations

import argparse
import datetime
import json
import traceback
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

from HTAMP.prediction.configs.vital_sign_tpp_config import (
    VitalSignTPPDatasetConfig,
    VitalSignTPPModelConfig,
    VitalSignTPPPredictionJobConfig,
)
from HTAMP.prediction.data_provider.vital_sign_tpp_dataset import (
    SPLITS,
    BatchSpec,
    ItemSpec,
    VitalSignTPPDataset,
    VitalSignTPPDatasetBundle,
    VitalSignTPPSequenceRecord,
    build_vital_sign_tpp_dataset_bundle,
)
from HTAMP.prediction.module.vital_sign_tpp_module import VitalSignTPPModule


def _history_event_count(
    *,
    total_events: int,
    history_fraction: float,
    min_history_events: int,
    max_history_events: int | None,
) -> int:
    if total_events <= 1:
        return 0
    history_events = max(min_history_events, int(np.floor(total_events * history_fraction)))
    history_events = min(history_events, total_events - 1)
    if max_history_events is not None:
        history_events = min(history_events, int(max_history_events))
    return max(1, history_events)


def _single_item_batch(
    *,
    item_spec: ItemSpec,
    device: torch.device,
) -> BatchSpec:
    return BatchSpec(
        data=item_spec.data.unsqueeze(0).to(device),
        types=item_spec.types.unsqueeze(0).to(device),
        log_prob_correction=None,
        condition=(
            item_spec.condition.unsqueeze(0).to(device)
            if item_spec.condition is not None
            else None
        ),
        extras={
            key: value.unsqueeze(0).to(device)
            for key, value in dict(item_spec.extras or {}).items()
        },
    )


def _append_token(
    *,
    batch: BatchSpec,
    next_type: int,
    next_position: int,
    next_event_index: int,
) -> BatchSpec:
    device = batch.data.device
    next_data = torch.zeros((batch.data.shape[0], 1), device=device, dtype=batch.data.dtype)
    next_types = torch.tensor([[next_type]], device=device, dtype=batch.types.dtype)
    next_position_tensor = torch.tensor(
        [[next_position]],
        device=device,
        dtype=batch.extras["position_in_event"].dtype,
    )
    next_event_index_tensor = torch.tensor(
        [[next_event_index]],
        device=device,
        dtype=batch.extras["event_index"].dtype,
    )
    return BatchSpec(
        data=torch.cat([batch.data, next_data], dim=1),
        types=torch.cat([batch.types, next_types], dim=1),
        log_prob_correction=None,
        condition=batch.condition,
        extras={
            "position_in_event": torch.cat(
                [batch.extras["position_in_event"], next_position_tensor],
                dim=1,
            ),
            "event_index": torch.cat(
                [batch.extras["event_index"], next_event_index_tensor],
                dim=1,
            ),
        },
    )


def _sample_future_events_from_prefix(
    *,
    model: VitalSignTPPModule,
    dataset_bundle: VitalSignTPPDatasetBundle,
    dataset: VitalSignTPPDataset,
    prefix_events: Sequence[tuple[float, float, int, dict[str, float]]],
    condition: Sequence[float] | None,
    max_future_events: int,
    device: torch.device,
    argmax: bool,
    mean_of: int,
    median: bool,
) -> list[tuple[float, float, int, dict[str, float]]]:
    prefix_item = dataset_bundle.encode_events(prefix_events, condition=condition)
    batch = _single_item_batch(item_spec=prefix_item, device=device)
    predictor = dataset.determine_position_and_type()
    next_type_position = predictor.send(None)

    for prefix_value in prefix_item.data.detach().cpu().tolist():
        next_type_position = predictor.send(float(prefix_value))

    generated_events = 0
    sampled_token_count = 0
    max_token_steps = max_future_events * (max(dataset_bundle.dims) + 2) + 4
    event_type_position = dataset.order.index(dataset.EVENT_TYPE)

    while generated_events < max_future_events and sampled_token_count < max_token_steps:
        next_type, next_position = next_type_position
        next_event_index = (
            0
            if batch.extras["event_index"].shape[1] == 0
            else int(batch.extras["event_index"][0, -1].item()) + (1 if next_position == 0 else 0)
        )
        batch = _append_token(
            batch=batch,
            next_type=int(next_type),
            next_position=int(next_position),
            next_event_index=int(next_event_index),
        )
        sampled_value = model.flex_tpp_model._single_dim_sample(
            batch,
            arg_max=argmax,
            mean_of=mean_of,
            median=median,
        )[0]
        batch.data[0, -1] = sampled_value
        sampled_token_count += 1

        if (
            int(next_position) == int(event_type_position)
            and int(round(float(sampled_value.item()))) == int(dataset_bundle.eos_event_type)
        ):
            break

        try:
            next_type_position = predictor.send(float(sampled_value.item()))
        except StopIteration:
            break

        if int(next_type_position[1]) == 0:
            generated_events += 1

    parsed_batch = dataset.parse_batch(
        BatchSpec(
            data=batch.data.detach().cpu(),
            types=batch.types.detach().cpu(),
            log_prob_correction=None,
            condition=(
                None
                if batch.condition is None
                else batch.condition.detach().cpu()
            ),
            extras={
                key: value.detach().cpu()
                for key, value in batch.extras.items()
            },
        )
    )
    parsed_events = list(parsed_batch[0][1])
    future_events = parsed_events[len(prefix_events):]
    return [
        (
            float(start_time),
            float(end_time),
            int(event_type),
            {str(key): float(value) for key, value in event_props.items()},
        )
        for start_time, end_time, event_type, event_props in future_events
        if int(event_type) != dataset_bundle.eos_event_type
    ]


def _encoded_event_payload(
    *,
    event: tuple[float, float, int, dict[str, float]],
    dataset_bundle: VitalSignTPPDatasetBundle,
    sequence_start_timestamp: str,
) -> dict[str, object]:
    start_time_hours, _, event_type, event_props = event
    start_timestamp = pd.Timestamp(sequence_start_timestamp) + pd.Timedelta(hours=float(start_time_hours))
    return {
        "timestamp": start_timestamp.isoformat(),
        "relative_time_hours": float(start_time_hours),
        "task_name": dataset_bundle.event_types[int(event_type)],
        "properties": {
            str(key): (None if not np.isfinite(float(value)) else float(value))
            for key, value in dict(event_props).items()
        },
    }


def _true_future_payloads(
    *,
    record: VitalSignTPPSequenceRecord,
    history_event_count: int,
    prediction_event_count: int,
) -> list[dict[str, object]]:
    encoded_future = record.events[history_event_count : history_event_count + prediction_event_count]
    raw_future = record.raw_events[history_event_count : history_event_count + prediction_event_count]
    payloads: list[dict[str, object]] = []
    for encoded_event, raw_event in zip(encoded_future, raw_future):
        payloads.append(
            {
                "timestamp": str(raw_event["timestamp"]),
                "relative_time_hours": float(encoded_event[0]),
                "task_name": str(raw_event["task_name"]),
                "properties": dict(raw_event.get("properties", {})),
            }
        )
    return payloads


class VitalSignTPPPredictionManager:
    def __init__(
        self,
        *,
        dataset_config: VitalSignTPPDatasetConfig,
        model_config: VitalSignTPPModelConfig,
        checkpoint_file_path: str,
        predictions_dir: Optional[str] = None,
        load_predictions: bool = False,
        splits: Sequence[str] = SPLITS,
        history_fraction: float = 0.6,
        min_history_events: int = 3,
        max_history_events: Optional[int] = None,
        prediction_event_count: int = 5,
        argmax: bool = True,
        mean_of: int = 20,
        median: bool = False,
    ) -> None:
        self.dataset_config = dataset_config
        self.model_config = model_config
        self.checkpoint_path = Path(checkpoint_file_path)
        self.predictions_dir = (
            Path(predictions_dir)
            if predictions_dir is not None
            else Path(self.dataset_config.dataset_dir) / "predictions"
        )
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.splits = self._normalize_splits(splits=splits)
        self.history_fraction = float(history_fraction)
        self.min_history_events = int(min_history_events)
        self.max_history_events = max_history_events
        self.prediction_event_count = int(prediction_event_count)
        self.argmax = bool(argmax)
        self.mean_of = int(mean_of)
        self.median = bool(median)

        if load_predictions:
            self.prediction_df = self._load_predictions()
        else:
            self.prediction_df = self._generate_predictions()

    def _normalize_splits(self, splits: Sequence[str]) -> tuple[str, ...]:
        normalized_splits = []
        for split in splits:
            if split not in SPLITS:
                raise ValueError(f"Unsupported split '{split}'. Expected one of {SPLITS}.")
            normalized_splits.append(split)
        return tuple(dict.fromkeys(normalized_splits))

    def _prediction_pickle_path(self) -> Path:
        return self.predictions_dir / "vital_sign_tpp_predictions.pkl"

    def _prediction_csv_path(self) -> Path:
        return self.predictions_dir / "vital_sign_tpp_predictions.csv"

    def _prediction_metadata_path(self) -> Path:
        return self.predictions_dir / "vital_sign_tpp_predictions_metadata.json"

    def _load_model(self, device: torch.device) -> VitalSignTPPModule:
        print("Loading FlexTPP vital-sign predictor ...")
        predictor = VitalSignTPPModule.load_from_checkpoint(
            checkpoint_path=str(self.checkpoint_path),
            map_location=device,
        )
        predictor.to(device)
        predictor.eval()
        print("FlexTPP vital-sign predictor has been loaded!")
        return predictor

    def _build_dataset_bundle(self) -> VitalSignTPPDatasetBundle:
        return build_vital_sign_tpp_dataset_bundle(
            dataset_config=self.dataset_config,
            model_config=self.model_config,
        )

    def _prediction_record(
        self,
        *,
        split: str,
        record: VitalSignTPPSequenceRecord,
        history_event_count: int,
        predicted_future_events: list[tuple[float, float, int, dict[str, float]]],
        dataset_bundle: VitalSignTPPDatasetBundle,
    ) -> dict[str, object]:
        history_payload = record.raw_events[:history_event_count]
        predicted_payloads = [
            _encoded_event_payload(
                event=event,
                dataset_bundle=dataset_bundle,
                sequence_start_timestamp=record.sequence_start_timestamp,
            )
            for event in predicted_future_events
        ]
        true_future_payloads = _true_future_payloads(
            record=record,
            history_event_count=history_event_count,
            prediction_event_count=self.prediction_event_count,
        )
        return {
            "split": split,
            "patient_id": record.patient_id,
            "encounter_id": record.encounter_id,
            "segment_id": int(record.segment_id),
            "sequence_start_timestamp": record.sequence_start_timestamp,
            "sequence_end_timestamp": record.sequence_end_timestamp,
            "history_event_count": int(history_event_count),
            "history_events": history_payload,
            "predicted_future_events": predicted_payloads,
            "true_future_events": true_future_payloads,
        }

    def _generate_predictions(self) -> pd.DataFrame:
        dataset_bundle = self._build_dataset_bundle()
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        predictor = self._load_model(device=device)
        prediction_rows: list[dict[str, object]] = []

        for split in self.splits:
            split_dataset = dataset_bundle.get_dataset(split)
            split_records = dataset_bundle.get_raw_records(split)
            print(f"Generating predictions for split '{split}' ...")

            for record_index, record in enumerate(split_records):
                total_events = len(record.events)
                if total_events <= 1:
                    continue
                history_event_count = _history_event_count(
                    total_events=total_events,
                    history_fraction=self.history_fraction,
                    min_history_events=self.min_history_events,
                    max_history_events=self.max_history_events,
                )
                prefix_events = record.events[:history_event_count]
                predicted_future_events = _sample_future_events_from_prefix(
                    model=predictor,
                    dataset_bundle=dataset_bundle,
                    dataset=split_dataset,
                    prefix_events=prefix_events,
                    condition=record.condition,
                    max_future_events=self.prediction_event_count,
                    device=device,
                    argmax=self.argmax,
                    mean_of=self.mean_of,
                    median=self.median,
                )
                prediction_rows.append(
                    self._prediction_record(
                        split=split,
                        record=record,
                        history_event_count=history_event_count,
                        predicted_future_events=predicted_future_events,
                        dataset_bundle=dataset_bundle,
                    )
                )

                if (record_index + 1) % 100 == 0:
                    print(f"Generated {record_index + 1} sequence forecasts for split '{split}'.")

        prediction_df = pd.DataFrame(prediction_rows)
        self._save_predictions(
            prediction_df=prediction_df,
            dataset_bundle=dataset_bundle,
        )
        return prediction_df

    def _save_predictions(
        self,
        *,
        prediction_df: pd.DataFrame,
        dataset_bundle: VitalSignTPPDatasetBundle,
    ) -> None:
        export_df = prediction_df.copy()
        for nested_column in ("history_events", "predicted_future_events", "true_future_events"):
            if nested_column in export_df.columns:
                export_df[nested_column] = export_df[nested_column].map(json.dumps)

        prediction_df.to_pickle(self._prediction_pickle_path())
        export_df.to_csv(self._prediction_csv_path(), index=False)
        with self._prediction_metadata_path().open("w", encoding="utf-8") as metadata_file:
            json.dump(
                {
                    "checkpoint_path": str(self.checkpoint_path),
                    "dataset_dir": self.dataset_config.dataset_dir,
                    "splits": list(self.splits),
                    "history_fraction": self.history_fraction,
                    "min_history_events": self.min_history_events,
                    "max_history_events": self.max_history_events,
                    "prediction_event_count": self.prediction_event_count,
                    "argmax": self.argmax,
                    "mean_of": self.mean_of,
                    "median": self.median,
                    "event_types": dataset_bundle.event_types,
                    "training_config": {
                        "dataset_config": self.dataset_config.to_dict(),
                        "model_config": self.model_config.to_dict(),
                    },
                },
                metadata_file,
                indent=2,
            )

    def _load_predictions(self) -> pd.DataFrame:
        print("Loading FlexTPP vital-sign predictions ...")
        prediction_df = pd.read_pickle(self._prediction_pickle_path())
        print("FlexTPP vital-sign predictions have been loaded!")
        return prediction_df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="VitalSignTPPPredictionHandler",
        description="Generate FlexTPP vital-sign request forecasts from a JSON config file.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help=(
            "Path to a JSON file containing 'dataset_config', 'model_config', "
            "'checkpoint_path', and prediction settings."
        ),
    )
    return parser


if __name__ == "__main__":
    p_start = datetime.datetime.now()
    try:
        parser = build_parser()
        args = parser.parse_args()
        prediction_job_config = VitalSignTPPPredictionJobConfig.from_json_file(args.config_path)
        VitalSignTPPPredictionManager(
            dataset_config=prediction_job_config.dataset_config,
            model_config=prediction_job_config.model_config,
            checkpoint_file_path=prediction_job_config.checkpoint_path,
            predictions_dir=prediction_job_config.predictions_dir,
            load_predictions=prediction_job_config.load_predictions,
            splits=prediction_job_config.prediction_splits,
            history_fraction=prediction_job_config.history_fraction,
            min_history_events=prediction_job_config.min_history_events,
            max_history_events=prediction_job_config.max_history_events,
            prediction_event_count=prediction_job_config.prediction_event_count,
            argmax=prediction_job_config.argmax,
            mean_of=prediction_job_config.mean_of,
            median=prediction_job_config.median,
        )
    except Exception as error_main_context:
        print("Fail End Process: ", error_main_context)
        traceback.print_exc()
    p_stop = datetime.datetime.now()
    print("Execution time: " + str(p_stop - p_start))
