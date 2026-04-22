from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from HTAMP.data_processing.processing_dataclasses import AnnotatedDataFiles

SUPPORTED_REQUEST_MODELS = ("TimesNet", "TimeMixer", "iTransformer", "PatchTST")
SUPPORTED_REQUEST_TASKS = (
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
IsoWeek = tuple[int, int]


def _default_test_iso_weeks() -> tuple[IsoWeek, ...]:
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


def _deep_merge_dicts(base: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(base=merged[key], updates=value)
            continue
        merged[key] = value
    return merged


def _coerce_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected '{field_name}' to be a JSON object, got {type(value).__name__}.")
    return value


def _normalize_json_like(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_json_like(nested_value)
            for key, nested_value in value.items()
        }
    if isinstance(value, list):
        return [_normalize_json_like(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_json_like(item) for item in value]
    return value


def _parse_iso_week_value(value: str) -> IsoWeek:
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
) -> tuple[IsoWeek, ...]:
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


def _normalize_floor_key(raw_floor: Any, *, field_name: str) -> int:
    if isinstance(raw_floor, bool):
        raise TypeError(f"{field_name} floor keys must be integers, not booleans.")
    try:
        return int(raw_floor)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} floor keys must be integers.") from exc


def _default_test_iso_weeks_by_floor() -> dict[int, tuple[IsoWeek, ...]]:
    try:
        from HTAMP.assignment.run_test import ALLOWED_ISO_WEEKS_BY_FLOOR

        return {
            int(floor): tuple(sorted(set(_normalize_test_iso_weeks(raw_weeks=weeks))))
            for floor, weeks in ALLOWED_ISO_WEEKS_BY_FLOOR.items()
        }
    except Exception:
        return {}


def _normalize_test_iso_weeks_by_floor(
    raw_weeks_by_floor: Mapping[Any, Sequence[str | Sequence[int]] | None] | None,
) -> dict[int, tuple[IsoWeek, ...]]:
    if raw_weeks_by_floor is None:
        return _default_test_iso_weeks_by_floor()

    normalized_weeks_by_floor: dict[int, tuple[IsoWeek, ...]] = {}
    for raw_floor, raw_weeks in dict(
        _coerce_mapping(value=raw_weeks_by_floor, field_name="test_iso_weeks_by_floor")
    ).items():
        floor = _normalize_floor_key(raw_floor, field_name="test_iso_weeks_by_floor")
        normalized_weeks_by_floor[floor] = tuple(
            sorted(set(_normalize_test_iso_weeks(raw_weeks=raw_weeks)))
        )
    return dict(sorted(normalized_weeks_by_floor.items()))


def _build_annotated_data_files(raw_value: Any) -> AnnotatedDataFiles:
    if isinstance(raw_value, AnnotatedDataFiles):
        return raw_value

    annotated_data = _coerce_mapping(value=raw_value, field_name="annotated_data_files")
    return AnnotatedDataFiles(**dict(annotated_data))


def _default_included_tasks() -> tuple[str, ...]:
    return tuple(SUPPORTED_REQUEST_TASKS)


def _normalize_included_tasks(
    raw_tasks: Sequence[str] | str | None,
) -> tuple[str, ...]:
    if raw_tasks is None:
        return _default_included_tasks()
    if isinstance(raw_tasks, str):
        raw_tasks = [raw_tasks]

    normalized_tasks: list[str] = []
    seen_tasks: set[str] = set()
    for raw_task in raw_tasks:
        task_name = str(raw_task).strip()
        if not task_name:
            continue
        if task_name not in SUPPORTED_REQUEST_TASKS:
            raise ValueError(
                f"Unsupported request task '{task_name}'. "
                f"Expected one of {SUPPORTED_REQUEST_TASKS}."
            )
        if task_name in seen_tasks:
            continue
        normalized_tasks.append(task_name)
        seen_tasks.add(task_name)

    if not normalized_tasks:
        raise ValueError("included_tasks must contain at least one supported request task.")

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


@dataclass
class MonitoringRequestDatasetConfig:
    annotated_data_files: AnnotatedDataFiles
    request_dir: str = "data/requests"
    dataset_dir: str = "data/prediction/request_intervals"
    start_date: str = "2024-06-24"
    end_date: str = "2025-06-29"
    patient_id_col: str = "MRN"
    included_tasks: tuple[str, ...] = field(default_factory=_default_included_tasks)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_iso_weeks: tuple[IsoWeek, ...] = field(default_factory=_default_test_iso_weeks)
    test_iso_weeks_by_floor: dict[int, tuple[IsoWeek, ...]] = field(
        default_factory=_default_test_iso_weeks_by_floor
    )
    validation_split_strategy: str = "chronological_weeks"
    validation_split_seed: int = 42
    use_saved_request_data: bool = False
    use_saved_time_series: bool = False
    preprocess_data: bool = False
    save_data: bool = True

    def __post_init__(self) -> None:
        self.annotated_data_files = _build_annotated_data_files(self.annotated_data_files)
        self.included_tasks = _normalize_included_tasks(raw_tasks=self.included_tasks)

        if min(self.train_ratio, self.val_ratio) < 0.0:
            raise ValueError("train_ratio and val_ratio must be non-negative.")

        if (self.train_ratio + self.val_ratio) <= 0.0:
            raise ValueError("train_ratio and val_ratio must sum to a positive value.")

        self.test_iso_weeks = _normalize_test_iso_weeks(raw_weeks=self.test_iso_weeks)
        self.test_iso_weeks = tuple(sorted(set(self.test_iso_weeks)))
        self.test_iso_weeks_by_floor = _normalize_test_iso_weeks_by_floor(
            raw_weeks_by_floor=self.test_iso_weeks_by_floor
        )
        self.validation_split_strategy = _normalize_validation_split_strategy(
            raw_strategy=self.validation_split_strategy
        )
        self.validation_split_seed = _normalize_validation_split_seed(
            raw_seed=self.validation_split_seed
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | "MonitoringRequestDatasetConfig") -> "MonitoringRequestDatasetConfig":
        if isinstance(payload, cls):
            return payload

        dataset_payload = dict(_coerce_mapping(value=payload, field_name="dataset_config"))
        if "annotated_data_files" not in dataset_payload:
            raise ValueError("dataset_config must include an 'annotated_data_files' object.")
        return cls(**dataset_payload)

    @classmethod
    def from_json_file(cls, config_path: str | Path) -> "MonitoringRequestDatasetConfig":
        return cls.from_dict(payload=_load_json_object(config_path=config_path))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def resolved_test_iso_weeks_for_floor(self, floor: int | None) -> tuple[IsoWeek, ...]:
        if floor is None:
            return self.test_iso_weeks
        return self.test_iso_weeks_by_floor.get(int(floor), self.test_iso_weeks)


@dataclass
class TimeseriesModelConfig:
    model_name: str = "TimesNet"
    run_name: str = "TimesNet_medical_request_intervals"
    wandb: bool = False

    task_name: str = "long_term_forecast"
    seq_len: int = 8
    label_len: int = 0
    pred_len: int = 4

    enc_in: int = 0
    dec_in: int = 0
    c_out: int = 0
    d_model: int = 512
    n_heads: int = 8
    e_layers: int = 2
    d_layers: int = 1
    d_ff: int = 2048
    top_k: int = 5
    num_kernels: int = 6
    moving_avg: int = 25
    factor: int = 1
    dropout: float = 0.1
    patch_len: int = 4
    stride: int = 2
    activation: str = "gelu"
    output_attention: bool = False
    channel_independence: int = 0
    decomp_method: str = "moving_avg"
    use_norm: int = 1
    down_sampling_layers: int = 0
    down_sampling_window: int = 1
    down_sampling_method: Optional[str] = None
    embed: str = "timeF"
    freq: str = "h"
    num_class: int = 0

    num_workers: int = 0
    max_epochs: int = 300
    batch_size: int = 32
    patience: int = 40
    learning_rate: float = 0.0001
    loss: str = "MSE"
    huber_delta: float = 1.0
    delta_loss_weight: float = 1.0
    availability_loss_weight: float = 1.0
    monitor_metric: str = "val_delta_mae"
    monitor_mode: Optional[str] = None

    accelerator: str = "auto"
    devices: int = 1
    strategy: Optional[str] = None

    def __post_init__(self) -> None:
        if self.huber_delta <= 0:
            raise ValueError("huber_delta must be greater than zero.")
        if self.delta_loss_weight < 0:
            raise ValueError("delta_loss_weight must be non-negative.")
        if self.availability_loss_weight < 0:
            raise ValueError("availability_loss_weight must be non-negative.")
        if self.delta_loss_weight == 0 and self.availability_loss_weight == 0:
            raise ValueError("At least one loss weight must be greater than zero.")
        self.monitor_metric = str(self.monitor_metric).strip()
        if not self.monitor_metric:
            raise ValueError("monitor_metric must not be empty.")
        if self.monitor_mode is not None:
            self.monitor_mode = str(self.monitor_mode).strip().lower()
            if self.monitor_mode not in {"min", "max"}:
                raise ValueError("monitor_mode must be either 'min', 'max', or null.")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | "TimeseriesModelConfig") -> "TimeseriesModelConfig":
        if isinstance(payload, cls):
            return payload
        return cls(**dict(_coerce_mapping(value=payload, field_name="model_config")))

    @classmethod
    def from_json_file(cls, config_path: str | Path) -> "TimeseriesModelConfig":
        return cls.from_dict(payload=_load_json_object(config_path=config_path))

    @classmethod
    def from_namespace(cls, args) -> "TimeseriesModelConfig":
        field_names = {field.name for field in fields(cls)}
        values = {
            field_name: getattr(args, field_name)
            for field_name in field_names
            if hasattr(args, field_name)
        }
        return cls(**values)

    def sync_channel_dimensions(self, num_input_channels: int, num_output_channels: int) -> None:
        self.enc_in = num_input_channels
        self.dec_in = num_output_channels
        self.c_out = num_output_channels

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class MonitoringRequestTrainingConfig:
    dataset_config: MonitoringRequestDatasetConfig
    model_config: TimeseriesModelConfig

    def __post_init__(self) -> None:
        self.dataset_config = MonitoringRequestDatasetConfig.from_dict(self.dataset_config)
        self.model_config = TimeseriesModelConfig.from_dict(self.model_config)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | "MonitoringRequestTrainingConfig") -> "MonitoringRequestTrainingConfig":
        if isinstance(payload, cls):
            return payload

        config_payload = dict(_coerce_mapping(value=payload, field_name="training_config"))
        return cls(
            dataset_config=MonitoringRequestDatasetConfig.from_dict(config_payload["dataset_config"]),
            model_config=TimeseriesModelConfig.from_dict(config_payload["model_config"]),
        )

    @classmethod
    def from_json_file(cls, config_path: str | Path) -> "MonitoringRequestTrainingConfig":
        return cls.from_dict(payload=_load_json_object(config_path=config_path))

    def with_overrides(self, overrides: Mapping[str, Any] | None) -> "MonitoringRequestTrainingConfig":
        if overrides is None:
            return MonitoringRequestTrainingConfig.from_dict(self)

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

        return MonitoringRequestTrainingConfig.from_dict(merged_payload)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class MonitoringRequestPredictionJobConfig:
    dataset_config: MonitoringRequestDatasetConfig
    model_config: TimeseriesModelConfig
    checkpoint_path: str
    predictions_dir: Optional[str] = None
    load_predictions: bool = False
    prediction_splits: tuple[str, ...] = ("train", "val", "test")

    def __post_init__(self) -> None:
        self.dataset_config = MonitoringRequestDatasetConfig.from_dict(self.dataset_config)
        self.model_config = TimeseriesModelConfig.from_dict(self.model_config)
        if isinstance(self.prediction_splits, str):
            self.prediction_splits = (self.prediction_splits,)
        self.prediction_splits = tuple(self.prediction_splits)

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any] | "MonitoringRequestPredictionJobConfig",
    ) -> "MonitoringRequestPredictionJobConfig":
        if isinstance(payload, cls):
            return payload

        config_payload = dict(_coerce_mapping(value=payload, field_name="prediction_job_config"))
        return cls(
            dataset_config=MonitoringRequestDatasetConfig.from_dict(config_payload["dataset_config"]),
            model_config=TimeseriesModelConfig.from_dict(config_payload["model_config"]),
            checkpoint_path=str(config_payload["checkpoint_path"]),
            predictions_dir=config_payload.get("predictions_dir"),
            load_predictions=bool(config_payload.get("load_predictions", False)),
            prediction_splits=config_payload.get("prediction_splits", ("train", "val", "test")),
        )

    @classmethod
    def from_json_file(cls, config_path: str | Path) -> "MonitoringRequestPredictionJobConfig":
        return cls.from_dict(payload=_load_json_object(config_path=config_path))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class MonitoringRequestModelSweepConfig:
    predictor_config: MonitoringRequestTrainingConfig
    models: tuple[str, ...] = field(default_factory=lambda: SUPPORTED_REQUEST_MODELS)
    summary_path: str = "data/prediction/monitoring_request_model_sweep_summary.csv"
    comparison_plot_path: Optional[str] = None
    selection_metric: str = "val_delta_mae"
    fail_fast: bool = False
    model_overrides: dict[str, dict[str, dict[str, object]]] = field(default_factory=dict)
    search_space: dict[str, dict[str, object]] = field(default_factory=dict)
    max_parallel_runs: int = 1
    gpu_ids: tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.predictor_config = MonitoringRequestTrainingConfig.from_dict(self.predictor_config)
        if isinstance(self.models, str):
            self.models = (self.models,)
        self.models = tuple(self.models)
        self.model_overrides = {
            scope_name: {
                section_name: dict(_coerce_mapping(value=section_updates, field_name=section_name))
                for section_name, section_updates in dict(
                    _coerce_mapping(value=scope_updates, field_name=scope_name)
                ).items()
            }
            for scope_name, scope_updates in self.model_overrides.items()
        }
        self.search_space = {
            scope_name: dict(
                _coerce_mapping(value=_normalize_json_like(scope_updates), field_name=scope_name)
            )
            for scope_name, scope_updates in dict(
                _coerce_mapping(value=self.search_space, field_name="search_space")
            ).items()
        }
        if self.max_parallel_runs < 1:
            raise ValueError("max_parallel_runs must be at least 1.")
        if isinstance(self.gpu_ids, int):
            self.gpu_ids = (self.gpu_ids,)
        self.gpu_ids = tuple(int(gpu_id) for gpu_id in self.gpu_ids)
        if any(gpu_id < 0 for gpu_id in self.gpu_ids):
            raise ValueError("gpu_ids must contain non-negative integers.")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | "MonitoringRequestModelSweepConfig") -> "MonitoringRequestModelSweepConfig":
        if isinstance(payload, cls):
            return payload

        config_payload = dict(_coerce_mapping(value=payload, field_name="model_sweep_config"))
        return cls(
            predictor_config=MonitoringRequestTrainingConfig.from_dict(config_payload["predictor_config"]),
            models=config_payload.get("models", SUPPORTED_REQUEST_MODELS),
            summary_path=str(config_payload.get("summary_path", "data/prediction/monitoring_request_model_sweep_summary.csv")),
            comparison_plot_path=config_payload.get("comparison_plot_path"),
            selection_metric=str(config_payload.get("selection_metric", "val_delta_mae")),
            fail_fast=bool(config_payload.get("fail_fast", False)),
            model_overrides=dict(config_payload.get("model_overrides", {})),
            search_space=dict(config_payload.get("search_space", {})),
            max_parallel_runs=int(config_payload.get("max_parallel_runs", 1)),
            gpu_ids=tuple(config_payload.get("gpu_ids", ())),
        )

    @classmethod
    def from_json_file(cls, config_path: str | Path) -> "MonitoringRequestModelSweepConfig":
        return cls.from_dict(payload=_load_json_object(config_path=config_path))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)