from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from HTAMP.data_processing.processing_dataclasses import AnnotatedDataFiles
from HTAMP.prediction.medication_mapping import (
    SUPPORTED_MEDICATION_CODE_STRATEGIES,
    SUPPORTED_MEDICATION_MAPPING_FALLBACKS,
)

SUPPORTED_DELIVERY_CONTEXT_TASKS = (
    "blood_pressure",
    "heart_rate",
    "respiratory_rate",
    "temperature",
    "oxygen_saturation",
)
SUPPORTED_VALIDATION_SPLIT_STRATEGIES = (
    "chronological_weeks",
    "random_patients",
)
VALIDATION_SPLIT_STRATEGY_ALIASES = {
    "grouped_patients": "random_patients",
}
DEFAULT_TIME_BINS_HOURS = (
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
    12.0,
    24.0,
)


def _default_test_iso_weeks() -> tuple[tuple[int, int], ...]:
    try:
        from HTAMP.assignment.run_test import ALLOWED_ISO_WEEKS

        return tuple(sorted(ALLOWED_ISO_WEEKS))
    except Exception:
        return (
            (2024, 27),
            (2024, 36),
            (2024, 40),
            (2024, 44),
            (2025, 5),
            (2025, 14),
        )


def _load_json_object(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as config_file:
        payload = json.load(config_file)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected the JSON config at '{path}' to contain an object at the top level.")
    return payload


def _coerce_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected '{field_name}' to be a JSON object, got {type(value).__name__}.")
    return value


def _deep_merge_dicts(base: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(base=merged[key], updates=value)
            continue
        merged[key] = value
    return merged


def _parse_iso_week_value(value: str) -> tuple[int, int]:
    cleaned = value.strip().replace("W", "-").replace(",", "-").replace("_", "-")
    parts = [part for part in cleaned.split("-") if part]
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        return int(parts[0]), int(parts[1])
    raise ValueError(
        f"Could not parse ISO week value '{value}'. "
        "Use formats like '2024W40', '2024-W40', or [2024, 40]."
    )


def _normalize_test_iso_weeks(
    raw_weeks: Sequence[str | Sequence[int]] | None,
) -> tuple[tuple[int, int], ...]:
    if raw_weeks is None:
        return _default_test_iso_weeks()
    if isinstance(raw_weeks, str):
        raw_weeks = [raw_weeks]

    normalized_weeks: list[tuple[int, int]] = []
    for entry in raw_weeks:
        if isinstance(entry, str):
            normalized_weeks.append(_parse_iso_week_value(entry))
            continue

        if len(entry) != 2:
            raise ValueError(
                "Each test_iso_weeks entry must be either a string like '2024W40' "
                "or a two-item sequence like [2024, 40]."
            )
        normalized_weeks.append((int(entry[0]), int(entry[1])))

    return tuple(normalized_weeks)


def _build_annotated_data_files(raw_value: Any) -> AnnotatedDataFiles:
    if isinstance(raw_value, AnnotatedDataFiles):
        return raw_value

    annotated_data = _coerce_mapping(value=raw_value, field_name="annotated_data_files")
    return AnnotatedDataFiles(**dict(annotated_data))


def _normalize_context_tasks(
    raw_tasks: Sequence[str] | str | None,
) -> tuple[str, ...]:
    if raw_tasks is None:
        return tuple(SUPPORTED_DELIVERY_CONTEXT_TASKS)
    if isinstance(raw_tasks, str):
        raw_tasks = [raw_tasks]

    normalized_tasks: list[str] = []
    seen_tasks: set[str] = set()
    for raw_task in raw_tasks:
        task_name = str(raw_task).strip()
        if not task_name:
            continue
        if task_name not in SUPPORTED_DELIVERY_CONTEXT_TASKS:
            raise ValueError(
                f"Unsupported monitoring context task '{task_name}'. "
                f"Expected one of {SUPPORTED_DELIVERY_CONTEXT_TASKS}."
            )
        if task_name in seen_tasks:
            continue
        normalized_tasks.append(task_name)
        seen_tasks.add(task_name)

    if not normalized_tasks:
        raise ValueError("included_tasks must contain at least one supported monitoring task.")

    return tuple(normalized_tasks)


def _normalize_validation_split_strategy(raw_strategy: str | None) -> str:
    if raw_strategy is None:
        return "chronological_weeks"

    normalized_strategy = VALIDATION_SPLIT_STRATEGY_ALIASES.get(
        str(raw_strategy).strip().lower(),
        str(raw_strategy).strip().lower(),
    )
    if not normalized_strategy:
        return "chronological_weeks"
    if normalized_strategy not in SUPPORTED_VALIDATION_SPLIT_STRATEGIES:
        raise ValueError(
            f"Unsupported validation_split_strategy '{raw_strategy}'. "
            f"Expected one of {SUPPORTED_VALIDATION_SPLIT_STRATEGIES}."
        )
    return normalized_strategy


def _normalize_validation_split_seed(raw_seed: Any) -> int:
    if isinstance(raw_seed, bool):
        raise TypeError("validation_split_seed must be an integer, not a boolean.")
    try:
        return int(raw_seed)
    except (TypeError, ValueError) as exc:
        raise TypeError("validation_split_seed must be an integer.") from exc


def _normalize_time_bins_hours(
    raw_bins: Sequence[float] | str | None,
) -> tuple[float, ...]:
    if raw_bins is None:
        return DEFAULT_TIME_BINS_HOURS
    if isinstance(raw_bins, str):
        raw_bins = [entry.strip() for entry in raw_bins.split(",") if entry.strip()]

    normalized_bins = sorted({float(bin_value) for bin_value in raw_bins})
    if not normalized_bins:
        raise ValueError("time_bins_hours must contain at least one positive value.")
    if normalized_bins[0] <= 0.0:
        raise ValueError("time_bins_hours must be strictly positive.")
    return tuple(normalized_bins)


@dataclass
class DeliveryRequestDatasetConfig:
    annotated_data_files: AnnotatedDataFiles
    request_dir: str = "data/requests"
    dataset_dir: str = "data/prediction/delivery_requests_dataset"
    start_date: str = "2024-06-24"
    end_date: str = "2025-06-29"
    patient_id_col: str = "MRN"
    included_tasks: tuple[str, ...] = field(default_factory=lambda: tuple(SUPPORTED_DELIVERY_CONTEXT_TASKS))
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_iso_weeks: tuple[tuple[int, int], ...] = field(default_factory=_default_test_iso_weeks)
    validation_split_strategy: str = "chronological_weeks"
    validation_split_seed: int = 42
    lookback_hours: float = 24.0
    med_lookback_hours: float = 48.0
    med_decay_hours: float = 12.0
    prediction_horizon_hours: float = 24.0
    max_seq_len: int = 48
    time_bins_hours: tuple[float, ...] = field(default_factory=lambda: DEFAULT_TIME_BINS_HOURS)
    top_vitals: int = 16
    top_meds: int = 128
    min_med_count: int = 1
    include_order_triggers: bool = True
    use_admin_as_vital_time: bool = True
    medication_code_col: Optional[str] = None
    medication_mapping_csv: Optional[str] = None
    medication_code_strategy: str = "raw_name"
    medication_mapping_fallback: str = "keep_clean_name"
    medication_location_col: str = "scheduled_space_id"
    use_saved_request_data: bool = False
    use_saved_dataset: bool = False
    preprocess_data: bool = True
    save_data: bool = True

    def __post_init__(self) -> None:
        self.annotated_data_files = _build_annotated_data_files(self.annotated_data_files)
        self.included_tasks = _normalize_context_tasks(raw_tasks=self.included_tasks)

        if min(self.train_ratio, self.val_ratio) < 0.0:
            raise ValueError("train_ratio and val_ratio must be non-negative.")
        if (self.train_ratio + self.val_ratio) <= 0.0:
            raise ValueError("train_ratio and val_ratio must sum to a positive value.")
        if self.lookback_hours <= 0.0:
            raise ValueError("lookback_hours must be greater than zero.")
        if self.med_lookback_hours <= 0.0:
            raise ValueError("med_lookback_hours must be greater than zero.")
        if self.med_decay_hours <= 0.0:
            raise ValueError("med_decay_hours must be greater than zero.")
        if self.prediction_horizon_hours <= 0.0:
            raise ValueError("prediction_horizon_hours must be greater than zero.")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be greater than zero.")
        if self.top_vitals <= 0:
            raise ValueError("top_vitals must be greater than zero.")
        if self.top_meds <= 0:
            raise ValueError("top_meds must be greater than zero.")
        if self.min_med_count <= 0:
            raise ValueError("min_med_count must be greater than zero.")

        self.test_iso_weeks = tuple(sorted(set(_normalize_test_iso_weeks(raw_weeks=self.test_iso_weeks))))
        self.validation_split_strategy = _normalize_validation_split_strategy(
            raw_strategy=self.validation_split_strategy
        )
        self.validation_split_seed = _normalize_validation_split_seed(
            raw_seed=self.validation_split_seed
        )
        self.time_bins_hours = _normalize_time_bins_hours(raw_bins=self.time_bins_hours)

        if self.medication_code_col is not None:
            self.medication_code_col = str(self.medication_code_col).strip() or None
        if self.medication_mapping_csv is not None:
            self.medication_mapping_csv = str(self.medication_mapping_csv).strip() or None
        self.medication_code_strategy = str(self.medication_code_strategy).strip().lower()
        if self.medication_code_strategy not in SUPPORTED_MEDICATION_CODE_STRATEGIES:
            raise ValueError(
                f"Unsupported medication_code_strategy '{self.medication_code_strategy}'. "
                f"Expected one of {SUPPORTED_MEDICATION_CODE_STRATEGIES}."
            )
        self.medication_mapping_fallback = str(self.medication_mapping_fallback).strip().lower()
        if self.medication_mapping_fallback not in SUPPORTED_MEDICATION_MAPPING_FALLBACKS:
            raise ValueError(
                f"Unsupported medication_mapping_fallback '{self.medication_mapping_fallback}'. "
                f"Expected one of {SUPPORTED_MEDICATION_MAPPING_FALLBACKS}."
            )
        if self.medication_code_strategy != "raw_name" and self.medication_mapping_csv is None:
            raise ValueError(
                "medication_mapping_csv must be provided when medication_code_strategy is not 'raw_name'."
            )
        self.medication_location_col = str(self.medication_location_col).strip()
        if not self.medication_location_col:
            raise ValueError("medication_location_col must not be empty.")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | "DeliveryRequestDatasetConfig") -> "DeliveryRequestDatasetConfig":
        if isinstance(payload, cls):
            return payload

        dataset_payload = dict(_coerce_mapping(value=payload, field_name="dataset_config"))
        if "annotated_data_files" not in dataset_payload:
            raise ValueError("dataset_config must include an 'annotated_data_files' object.")
        return cls(**dataset_payload)

    @classmethod
    def from_json_file(cls, config_path: str | Path) -> "DeliveryRequestDatasetConfig":
        return cls.from_dict(payload=_load_json_object(config_path=config_path))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class DeliveryPointProcessModelConfig:
    run_name: str = "delivery_request_point_process"
    wandb: bool = False
    batch_size: int = 64
    num_workers: int = 0
    max_epochs: int = 50
    patience: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    dropout: float = 0.1
    vital_hidden_size: int = 64
    med_hidden_size: int = 64
    fusion_hidden_size: int = 128
    hazard_loss_weight: float = 1.0
    med_loss_weight: float = 1.0
    gradient_clip_val: float = 1.0
    monitor_metric: str = "val_loss"
    monitor_mode: Optional[str] = "min"
    accelerator: str = "auto"
    devices: int = 1
    strategy: Optional[str] = None

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative.")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be greater than zero.")
        if self.patience < 0:
            raise ValueError("patience must be non-negative.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be greater than zero.")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative.")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout must be in the range [0, 1).")
        if min(self.vital_hidden_size, self.med_hidden_size, self.fusion_hidden_size) <= 0:
            raise ValueError("All hidden sizes must be greater than zero.")
        if self.hazard_loss_weight <= 0.0:
            raise ValueError("hazard_loss_weight must be greater than zero.")
        if self.med_loss_weight < 0.0:
            raise ValueError("med_loss_weight must be non-negative.")
        if self.gradient_clip_val < 0.0:
            raise ValueError("gradient_clip_val must be non-negative.")

        self.monitor_metric = str(self.monitor_metric).strip()
        if not self.monitor_metric:
            raise ValueError("monitor_metric must not be empty.")
        if self.monitor_mode is not None:
            self.monitor_mode = str(self.monitor_mode).strip().lower()
            if self.monitor_mode not in {"min", "max"}:
                raise ValueError("monitor_mode must be either 'min', 'max', or null.")

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any] | "DeliveryPointProcessModelConfig",
    ) -> "DeliveryPointProcessModelConfig":
        if isinstance(payload, cls):
            return payload
        return cls(**dict(_coerce_mapping(value=payload, field_name="model_config")))

    @classmethod
    def from_json_file(cls, config_path: str | Path) -> "DeliveryPointProcessModelConfig":
        return cls.from_dict(payload=_load_json_object(config_path=config_path))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class DeliveryRequestTrainingConfig:
    dataset_config: DeliveryRequestDatasetConfig
    model_config: DeliveryPointProcessModelConfig

    def __post_init__(self) -> None:
        self.dataset_config = DeliveryRequestDatasetConfig.from_dict(self.dataset_config)
        self.model_config = DeliveryPointProcessModelConfig.from_dict(self.model_config)

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any] | "DeliveryRequestTrainingConfig",
    ) -> "DeliveryRequestTrainingConfig":
        if isinstance(payload, cls):
            return payload

        config_payload = dict(_coerce_mapping(value=payload, field_name="training_config"))
        return cls(
            dataset_config=DeliveryRequestDatasetConfig.from_dict(config_payload["dataset_config"]),
            model_config=DeliveryPointProcessModelConfig.from_dict(config_payload["model_config"]),
        )

    @classmethod
    def from_json_file(cls, config_path: str | Path) -> "DeliveryRequestTrainingConfig":
        return cls.from_dict(payload=_load_json_object(config_path=config_path))

    def with_overrides(self, overrides: Mapping[str, Any] | None) -> "DeliveryRequestTrainingConfig":
        if overrides is None:
            return DeliveryRequestTrainingConfig.from_dict(self)

        merged_payload = self.to_dict()
        for section_name, section_updates in dict(_coerce_mapping(value=overrides, field_name="model_overrides")).items():
            section_mapping = dict(_coerce_mapping(value=section_updates, field_name=section_name))
            if section_name not in merged_payload:
                raise ValueError(
                    f"Unknown training config section '{section_name}'. "
                    "Expected 'dataset_config' or 'model_config'."
                )
            merged_payload[section_name] = _deep_merge_dicts(
                base=merged_payload[section_name],
                updates=section_mapping,
            )

        return DeliveryRequestTrainingConfig.from_dict(merged_payload)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class DeliveryRequestPredictionJobConfig:
    dataset_config: DeliveryRequestDatasetConfig
    model_config: DeliveryPointProcessModelConfig
    checkpoint_path: str
    predictions_dir: Optional[str] = None
    load_predictions: bool = False
    prediction_splits: tuple[str, ...] = ("train", "val", "test")
    top_k_labels: int = 5

    def __post_init__(self) -> None:
        self.dataset_config = DeliveryRequestDatasetConfig.from_dict(self.dataset_config)
        self.model_config = DeliveryPointProcessModelConfig.from_dict(self.model_config)
        if isinstance(self.prediction_splits, str):
            self.prediction_splits = (self.prediction_splits,)
        self.prediction_splits = tuple(self.prediction_splits)
        self.top_k_labels = max(1, int(self.top_k_labels))

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any] | "DeliveryRequestPredictionJobConfig",
    ) -> "DeliveryRequestPredictionJobConfig":
        if isinstance(payload, cls):
            return payload

        config_payload = dict(_coerce_mapping(value=payload, field_name="prediction_job_config"))
        return cls(
            dataset_config=DeliveryRequestDatasetConfig.from_dict(config_payload["dataset_config"]),
            model_config=DeliveryPointProcessModelConfig.from_dict(config_payload["model_config"]),
            checkpoint_path=str(config_payload["checkpoint_path"]),
            predictions_dir=config_payload.get("predictions_dir"),
            load_predictions=bool(config_payload.get("load_predictions", False)),
            prediction_splits=config_payload.get("prediction_splits", ("train", "val", "test")),
            top_k_labels=int(config_payload.get("top_k_labels", 5)),
        )

    @classmethod
    def from_json_file(cls, config_path: str | Path) -> "DeliveryRequestPredictionJobConfig":
        return cls.from_dict(payload=_load_json_object(config_path=config_path))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
