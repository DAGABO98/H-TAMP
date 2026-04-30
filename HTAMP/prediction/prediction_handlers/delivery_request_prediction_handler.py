from __future__ import annotations

import argparse
import datetime
import json
import traceback
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import torch
from torch.utils.data import DataLoader

from HTAMP.prediction.configs.delivery_request_config import (
    DeliveryPointProcessModelConfig,
    DeliveryRequestDatasetConfig,
    DeliveryRequestPredictionJobConfig,
)
from HTAMP.prediction.data_provider.delivery_requests_dataset import (
    DeliveryRequestsDataset,
    DeliveryRequestsDatasetBundle,
    build_delivery_request_dataset_bundle,
)
from HTAMP.prediction.module.delivery_request_module import DeliveryRequestModule

SPLITS = ("train", "val", "test")


class DeliveryRequestPredictionManager:
    def __init__(
        self,
        dataset_config: DeliveryRequestDatasetConfig,
        model_config: DeliveryPointProcessModelConfig,
        checkpoint_file_path: str,
        predictions_dir: Optional[str] = None,
        data_folders: object | None = None,
        load_predictions: bool = False,
        splits: Sequence[str] = SPLITS,
        top_k_labels: int = 5,
    ) -> None:
        self.dataset_config = dataset_config
        self.model_config = model_config
        self.checkpoint_path = Path(checkpoint_file_path)
        self.predictions_dir = self._resolve_predictions_dir(
            predictions_dir=predictions_dir,
            data_folders=data_folders,
        )
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.splits = self._normalize_splits(splits=splits)
        self.top_k_labels = max(1, int(top_k_labels))

        if load_predictions:
            self.prediction_df = self._load_predictions()
        else:
            self.prediction_df = self._initialize_prediction_df()

    def _resolve_predictions_dir(
        self,
        predictions_dir: Optional[str],
        data_folders: object | None,
    ) -> Path:
        if predictions_dir is not None:
            return Path(predictions_dir)

        predicted_requests_path = getattr(data_folders, "predicted_requests_folder_path", None)
        if predicted_requests_path is not None:
            return Path(predicted_requests_path)

        return Path(self.dataset_config.dataset_dir) / "predictions"

    def _normalize_splits(self, splits: Sequence[str]) -> tuple[str, ...]:
        normalized_splits = []
        for split in splits:
            if split not in SPLITS:
                raise ValueError(f"Unsupported split '{split}'. Expected one of {SPLITS}.")
            normalized_splits.append(split)
        return tuple(dict.fromkeys(normalized_splits))

    def _prediction_pickle_path(self) -> Path:
        return self.predictions_dir / "delivery_requests.pkl"

    def _prediction_csv_path(self) -> Path:
        return self.predictions_dir / "delivery_requests.csv"

    def _prediction_metadata_path(self) -> Path:
        return self.predictions_dir / "delivery_requests_metadata.json"

    def _build_dataset_bundle(self) -> DeliveryRequestsDatasetBundle:
        return build_delivery_request_dataset_bundle(dataset_config=self.dataset_config)

    def _load_model(self, device: torch.device) -> DeliveryRequestModule:
        delivery_model = DeliveryRequestModule.load_from_checkpoint(
            checkpoint_path=str(self.checkpoint_path),
            map_location=device,
        )
        delivery_model.to(device)
        delivery_model.eval()
        return delivery_model

    def _build_dataset(
        self,
        dataset_bundle: DeliveryRequestsDatasetBundle,
        split: str,
    ) -> DeliveryRequestsDataset:
        return DeliveryRequestsDataset(
            dataset_bundle=dataset_bundle,
            split=split,
        )

    def _top_labels(
        self,
        probabilities,
        labels: Sequence[str],
        display_map: Optional[dict[str, str]] = None,
    ) -> list[dict[str, object]]:
        if len(labels) == 0:
            return []
        top_k = min(self.top_k_labels, len(labels))
        top_indices = probabilities.argsort()[::-1][:top_k]
        top_labels: list[dict[str, object]] = []
        for index in top_indices:
            label = str(labels[int(index)])
            row = {"label": label, "probability": float(probabilities[int(index)])}
            if display_map is not None:
                row["display_name"] = str(display_map.get(label, label))
            top_labels.append(row)
        return top_labels

    def _generate_split_predictions(
        self,
        split: str,
        dataset_bundle: DeliveryRequestsDatasetBundle,
        delivery_model: DeliveryRequestModule,
    ) -> list[dict[str, object]]:
        dataset = self._build_dataset(dataset_bundle=dataset_bundle, split=split)
        metadata_df = dataset_bundle.get_split_metadata(split).reset_index(drop=True)
        dataloader = DataLoader(
            dataset,
            batch_size=self.model_config.batch_size,
            shuffle=False,
            num_workers=self.model_config.num_workers,
        )

        prediction_rows: list[dict[str, object]] = []
        sample_index = 0
        for batch in dataloader:
            outputs = delivery_model.predict_batch(batch=batch)
            batch_size = batch["event"].shape[0]
            for batch_offset in range(batch_size):
                metadata_row = metadata_df.iloc[sample_index + batch_offset]
                prediction_rows.append(
                    {
                        "split": split,
                        "patient_id": str(metadata_row["patient_id"]),
                        "encounter_id": str(metadata_row["encounter_id"]),
                        "segment_id": int(metadata_row["segment_id"]),
                        "trigger_time": pd.Timestamp(metadata_row["trigger_time"]),
                        "predicted_event_probability": float(outputs["event_probability"][batch_offset]),
                        "predicted_expected_time_hours": float(outputs["expected_time_hours"][batch_offset]),
                        "survival_by_time_bin": json.dumps(
                            {
                                str(bin_value): float(outputs["survival"][batch_offset][bin_index])
                                for bin_index, bin_value in enumerate(dataset_bundle.time_bins_hours.tolist())
                            }
                        ),
                        "cumulative_event_prob_by_time_bin": json.dumps(
                            {
                                str(bin_value): float(outputs["cumulative_event"][batch_offset][bin_index])
                                for bin_index, bin_value in enumerate(dataset_bundle.time_bins_hours.tolist())
                            }
                        ),
                        "top_candidate_meds": json.dumps(
                            self._top_labels(
                                probabilities=outputs["med_probs"][batch_offset],
                                labels=dataset_bundle.med_vocab,
                                display_map=dataset_bundle.med_code_display_map,
                            )
                        ),
                        "true_event": float(batch["event"][batch_offset].item()),
                        "true_duration_hours": float(batch["duration_hours"][batch_offset].item()),
                        "true_med_labels": metadata_row["true_med_labels"],
                    }
                )
            sample_index += batch_size
        return prediction_rows

    def _build_prediction_metadata(self, dataset_bundle: DeliveryRequestsDatasetBundle) -> dict[str, object]:
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "dataset_dir": self.dataset_config.dataset_dir,
            "splits": list(self.splits),
            "time_bins_hours": dataset_bundle.time_bins_hours.tolist(),
            "vital_vocab": list(dataset_bundle.vital_vocab),
            "med_vocab": list(dataset_bundle.med_vocab),
            "med_code_display_map": dict(dataset_bundle.med_code_display_map),
            "medication_mapping": dataset_bundle.metadata.get("medication_mapping", {}),
        }

    def _initialize_prediction_df(self) -> pd.DataFrame:
        dataset_bundle = self._build_dataset_bundle()
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        delivery_model = self._load_model(device=device)
        prediction_rows: list[dict[str, object]] = []
        for split in self.splits:
            prediction_rows.extend(
                self._generate_split_predictions(
                    split=split,
                    dataset_bundle=dataset_bundle,
                    delivery_model=delivery_model,
                )
            )

        prediction_df = pd.DataFrame(prediction_rows)
        self._save_predictions(
            prediction_df=prediction_df,
            metadata=self._build_prediction_metadata(dataset_bundle=dataset_bundle),
        )
        return prediction_df

    def _save_predictions(
        self,
        prediction_df: pd.DataFrame,
        metadata: dict[str, object],
    ) -> None:
        prediction_df.to_pickle(self._prediction_pickle_path())
        prediction_df.to_csv(self._prediction_csv_path(), index=False)
        with self._prediction_metadata_path().open("w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2, default=str)

    def _load_predictions(self) -> pd.DataFrame:
        return pd.read_pickle(self._prediction_pickle_path())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="DeliveryRequestPredictionHandler",
        description="Generate delivery-request predictions from a JSON config file.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help=(
            "Path to a JSON file containing 'dataset_config', 'model_config', "
            "'checkpoint_path', and optional prediction output settings."
        ),
    )
    return parser


if __name__ == "__main__":
    p_start = datetime.datetime.now()
    try:
        parser = build_parser()
        parsed_args = parser.parse_args()
        prediction_job_config = DeliveryRequestPredictionJobConfig.from_json_file(parsed_args.config_path)
        DeliveryRequestPredictionManager(
            dataset_config=prediction_job_config.dataset_config,
            model_config=prediction_job_config.model_config,
            checkpoint_file_path=prediction_job_config.checkpoint_path,
            predictions_dir=prediction_job_config.predictions_dir,
            load_predictions=prediction_job_config.load_predictions,
            splits=prediction_job_config.prediction_splits,
            top_k_labels=prediction_job_config.top_k_labels,
        )
    except Exception as error_main_context:
        print("Fail End Process: ", error_main_context)
        traceback.print_exc()
    p_stop = datetime.datetime.now()
    print("Execution time: " + str(p_stop - p_start))
