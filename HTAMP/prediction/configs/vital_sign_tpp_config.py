from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from HTAMP.data_processing.processing_dataclasses import AnnotatedDataFiles

SUPPORTED_VITAL_SIGN_TASKS = (
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
SUPPORTED_EVENT_ORDERS = ("ST", "STP")
SUPPORTED_FLEX_EVENT_TYPE_MARK_MODES = ("task", "task_label", "task_component_label")
SUPPORTED_LABEL_STRATEGIES = ("quantile", "threshold")
DEFAULT_LABEL_NAMES = ("low", "medium", "high")
FLEX_EVENT_TYPE_MARK_MODE_ALIASES = {
    "standard": "task",
    "plain": "task",
    "task_only": "task",
    "base_task": "task",
    "enhanced": "task_label",
    "enhanced_marks": "task_label",
    "labels": "task_label",
    "labeled": "task_label",
}
SUPPORTED_NON_LINEARITIES = (
    "ELU",
    "GELU",
    "ReLU",
    "SiLU",
    "Softplus",
    "Tanh",
)
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
        raise ValueError(
            f"Expected the JSON config at '{path}' to contain an object at the top level."
        )
    return payload


def _coerce_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(
            f"Expected '{field_name}' to be a JSON object, got {type(value).__name__}."
        )
    return value


def _deep_merge_dicts(base: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(base=merged[key], updates=value)
            continue
        merged[key] = value
    return merged


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


def _normalize_tasks(raw_tasks: Sequence[str] | str | None) -> tuple[str, ...]:
    if raw_tasks is None:
        return tuple(SUPPORTED_VITAL_SIGN_TASKS)
    if isinstance(raw_tasks, str):
        raw_tasks = [raw_tasks]

    normalized_tasks: list[str] = []
    seen_tasks: set[str] = set()
    for raw_task in raw_tasks:
        task_name = str(raw_task).strip()
        if not task_name:
            continue
        if task_name not in SUPPORTED_VITAL_SIGN_TASKS:
            raise ValueError(
                f"Unsupported vital-sign task '{task_name}'. "
                f"Expected one of {SUPPORTED_VITAL_SIGN_TASKS}."
            )
        if task_name in seen_tasks:
            continue
        normalized_tasks.append(task_name)
        seen_tasks.add(task_name)

    if not normalized_tasks:
        raise ValueError("included_tasks must contain at least one supported vital-sign task.")
    return tuple(normalized_tasks)


def _default_label_component_by_task() -> dict[str, tuple[str, ...]]:
    return {
        "blood_pressure": ("systolic", "diastolic"),
    }


def _normalize_label_names(raw_label_names: Sequence[str] | None) -> tuple[str, str, str]:
    label_names = DEFAULT_LABEL_NAMES if raw_label_names is None else tuple(raw_label_names)
    if len(label_names) != 3:
        raise ValueError("label_names must contain exactly three labels.")
    normalized = tuple(str(label_name).strip().lower() for label_name in label_names)
    if any(not label_name for label_name in normalized):
        raise ValueError("label_names must not contain empty labels.")
    if len(set(normalized)) != 3:
        raise ValueError("label_names must contain three distinct labels.")
    return normalized  # type: ignore[return-value]


def _normalize_quantile_edges(raw_edges: Sequence[float]) -> tuple[float, float]:
    if len(raw_edges) != 2:
        raise ValueError("quantile_edges must contain exactly two numeric values.")
    lower_edge, upper_edge = (float(raw_edges[0]), float(raw_edges[1]))
    if not (0.0 < lower_edge < upper_edge < 1.0):
        raise ValueError("quantile_edges must satisfy 0 < lower < upper < 1.")
    return lower_edge, upper_edge


def _normalize_mapping_dict(raw_value: Mapping[str, Any] | None) -> dict[str, Any]:
    if raw_value is None:
        return {}
    return {
        str(key): value
        for key, value in dict(_coerce_mapping(value=raw_value, field_name="mapping")).items()
    }


def _normalize_component_names(raw_value: Any) -> tuple[str, ...]:
    if isinstance(raw_value, str):
        raw_components = [raw_value]
    elif isinstance(raw_value, Sequence):
        raw_components = list(raw_value)
    else:
        raise TypeError(
            "label_component_by_task values must be a component name or a list of names."
        )

    component_names: list[str] = []
    seen_components: set[str] = set()
    for raw_component in raw_components:
        component_name = str(raw_component).strip()
        if not component_name:
            continue
        if component_name in seen_components:
            continue
        component_names.append(component_name)
        seen_components.add(component_name)

    if not component_names:
        raise ValueError("label_component_by_task values must contain at least one component.")
    return tuple(component_names)


def _normalize_flex_event_type_mark_mode(raw_mode: str | None) -> str:
    if raw_mode is None:
        return "task"
    normalized_mode = str(raw_mode).strip().lower()
    normalized_mode = FLEX_EVENT_TYPE_MARK_MODE_ALIASES.get(normalized_mode, normalized_mode)
    if normalized_mode not in SUPPORTED_FLEX_EVENT_TYPE_MARK_MODES:
        raise ValueError(
            f"Unsupported event_type_mark_mode '{raw_mode}'. "
            f"Expected one of {SUPPORTED_FLEX_EVENT_TYPE_MARK_MODES}, or aliases "
            "'standard'/'plain' and 'enhanced'."
        )
    return normalized_mode


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


def _normalize_prediction_splits(
    raw_splits: Sequence[str] | str | None,
) -> tuple[str, ...]:
    allowed_splits = ("train", "val", "test")
    if raw_splits is None:
        return allowed_splits
    if isinstance(raw_splits, str):
        raw_splits = [raw_splits]

    normalized: list[str] = []
    seen: set[str] = set()
    for raw_split in raw_splits:
        split_name = str(raw_split).strip().lower()
        if split_name not in allowed_splits:
            raise ValueError(
                f"Unsupported prediction split '{raw_split}'. Expected one of {allowed_splits}."
            )
        if split_name in seen:
            continue
        normalized.append(split_name)
        seen.add(split_name)
    return tuple(normalized)


@dataclass
class VitalSignTPPDatasetConfig:
    annotated_data_files: AnnotatedDataFiles
    request_dir: str = "data/requests"
    dataset_dir: str = "data/prediction/vital_sign_tpp_dataset"
    start_date: str = "2024-06-24"
    end_date: str = "2025-06-29"
    patient_id_col: str = "MRN"
    included_tasks: tuple[str, ...] = field(default_factory=lambda: tuple(SUPPORTED_VITAL_SIGN_TASKS))
    event_type_mark_mode: str = "task"
    label_strategy: str = "quantile"
    label_names: tuple[str, str, str] = DEFAULT_LABEL_NAMES
    quantile_edges: tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0)
    measurement_thresholds: dict[str, Any] = field(default_factory=dict)
    label_component_by_task: dict[str, tuple[str, ...]] = field(
        default_factory=_default_label_component_by_task
    )
    missing_label: str = "unknown"
    drop_missing_measurement_events: bool = False
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_iso_weeks: tuple[IsoWeek, ...] = field(default_factory=_default_test_iso_weeks)
    test_iso_weeks_by_floor: dict[int, tuple[IsoWeek, ...]] = field(
        default_factory=_default_test_iso_weeks_by_floor
    )
    validation_split_strategy: str = "chronological_weeks"
    validation_split_seed: int = 42
    include_time_features_as_properties: bool = True
    use_previous_day_summary_conditioning: bool = True
    min_events_per_sequence: int = 2
    max_events_per_sequence: Optional[int] = None
    eos_offset_minutes: float = 5.0
    use_saved_request_data: bool = False
    use_saved_dataset: bool = False
    preprocess_data: bool = True
    save_data: bool = True

    def __post_init__(self) -> None:
        self.annotated_data_files = _build_annotated_data_files(self.annotated_data_files)
        self.included_tasks = _normalize_tasks(raw_tasks=self.included_tasks)

        if min(self.train_ratio, self.val_ratio) < 0.0:
            raise ValueError("train_ratio and val_ratio must be non-negative.")
        if (self.train_ratio + self.val_ratio) <= 0.0:
            raise ValueError("train_ratio and val_ratio must sum to a positive value.")
        if self.min_events_per_sequence < 1:
            raise ValueError("min_events_per_sequence must be at least 1.")
        if self.max_events_per_sequence is not None and self.max_events_per_sequence < 1:
            raise ValueError("max_events_per_sequence must be at least 1 when provided.")
        if self.eos_offset_minutes <= 0.0:
            raise ValueError("eos_offset_minutes must be greater than zero.")

        self.test_iso_weeks = tuple(sorted(set(_normalize_test_iso_weeks(self.test_iso_weeks))))
        self.test_iso_weeks_by_floor = _normalize_test_iso_weeks_by_floor(
            raw_weeks_by_floor=self.test_iso_weeks_by_floor
        )
        self.validation_split_strategy = _normalize_validation_split_strategy(
            raw_strategy=self.validation_split_strategy
        )
        self.validation_split_seed = _normalize_validation_split_seed(
            raw_seed=self.validation_split_seed
        )
        self.event_type_mark_mode = _normalize_flex_event_type_mark_mode(
            self.event_type_mark_mode
        )
        self.label_strategy = str(self.label_strategy).strip().lower()
        if self.label_strategy not in SUPPORTED_LABEL_STRATEGIES:
            raise ValueError(
                f"Unsupported label_strategy '{self.label_strategy}'. "
                f"Expected one of {SUPPORTED_LABEL_STRATEGIES}."
            )
        self.label_names = _normalize_label_names(self.label_names)
        self.quantile_edges = _normalize_quantile_edges(self.quantile_edges)
        self.measurement_thresholds = _normalize_mapping_dict(self.measurement_thresholds)
        self.label_component_by_task = {
            str(task_name): _normalize_component_names(component_names)
            for task_name, component_names in _normalize_mapping_dict(
                self.label_component_by_task
            ).items()
        }
        self.missing_label = str(self.missing_label).strip().lower()
        if not self.missing_label:
            raise ValueError("missing_label must not be empty.")
        self.drop_missing_measurement_events = bool(self.drop_missing_measurement_events)

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any] | "VitalSignTPPDatasetConfig",
    ) -> "VitalSignTPPDatasetConfig":
        if isinstance(payload, cls):
            return payload
        dataset_payload = dict(_coerce_mapping(value=payload, field_name="dataset_config"))
        if "annotated_data_files" not in dataset_payload:
            raise ValueError("dataset_config must include an 'annotated_data_files' object.")
        return cls(**dataset_payload)

    @classmethod
    def from_json_file(cls, config_path: str | Path) -> "VitalSignTPPDatasetConfig":
        return cls.from_dict(payload=_load_json_object(config_path=config_path))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def resolved_test_iso_weeks_for_floor(self, floor: int | None) -> tuple[IsoWeek, ...]:
        if floor is None:
            return self.test_iso_weeks
        return self.test_iso_weeks_by_floor.get(int(floor), self.test_iso_weeks)


@dataclass
class VitalSignTPPModelConfig:
    run_name: str = "vital_sign_flex_tpp"
    wandb: bool = False
    batch_size: int = 64
    num_workers: int = 0
    max_epochs: int = 50
    patience: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    gradient_clip_val: float = 1.0
    monitor_metric: str = "val_nll"
    monitor_mode: Optional[str] = "min"
    accelerator: str = "auto"
    devices: int = 1
    strategy: Optional[str] = None
    conditioning_network: Optional[dict[str, Any]] = None
    order: str = "STP"
    gaussian_except_start_time: bool = False
    depth: int = 6
    dim_k: int = 24
    n_head: int = 4
    dim_ff: int = 256
    non_linearity: str = "GELU"
    normalize: bool = True
    dropout: float = 0.15
    embed_event_index: bool = True
    monotonic_bins: int = 20
    param_nets_n_hidden_layer: int = 1
    param_nets_hidden_dim_factor: int = 2

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
        if self.gradient_clip_val < 0.0:
            raise ValueError("gradient_clip_val must be non-negative.")
        if self.order not in SUPPORTED_EVENT_ORDERS:
            raise ValueError(
                f"Unsupported event order '{self.order}'. Expected one of {SUPPORTED_EVENT_ORDERS}."
            )
        if "S" not in self.order or "T" not in self.order:
            raise ValueError("order must include both 'S' (time) and 'T' (event type).")
        if self.non_linearity not in SUPPORTED_NON_LINEARITIES:
            raise ValueError(
                f"Unsupported non_linearity '{self.non_linearity}'. "
                f"Expected one of {SUPPORTED_NON_LINEARITIES}."
            )
        if self.depth <= 0:
            raise ValueError("depth must be greater than zero.")
        if min(self.dim_k, self.n_head, self.dim_ff) <= 0:
            raise ValueError("dim_k, n_head, and dim_ff must be greater than zero.")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout must be in the range [0, 1).")
        if self.monotonic_bins <= 0:
            raise ValueError("monotonic_bins must be greater than zero.")
        if self.param_nets_n_hidden_layer < 0:
            raise ValueError("param_nets_n_hidden_layer must be non-negative.")
        if self.param_nets_hidden_dim_factor <= 0:
            raise ValueError("param_nets_hidden_dim_factor must be greater than zero.")

        self.monitor_metric = str(self.monitor_metric).strip()
        if not self.monitor_metric:
            raise ValueError("monitor_metric must not be empty.")
        if self.monitor_mode is not None:
            self.monitor_mode = str(self.monitor_mode).strip().lower()
            if self.monitor_mode not in {"min", "max"}:
                raise ValueError("monitor_mode must be either 'min', 'max', or null.")
        
        if self.conditioning_network is not None:
            self.conditioning_network = dict(
                _coerce_mapping(
                    value=self.conditioning_network,
                    field_name="conditioning_network",
                )
            )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any] | "VitalSignTPPModelConfig",
    ) -> "VitalSignTPPModelConfig":
        if isinstance(payload, cls):
            return payload
        return cls(**dict(_coerce_mapping(value=payload, field_name="model_config")))

    @classmethod
    def from_json_file(cls, config_path: str | Path) -> "VitalSignTPPModelConfig":
        return cls.from_dict(payload=_load_json_object(config_path=config_path))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class VitalSignTPPTrainingConfig:
    dataset_config: VitalSignTPPDatasetConfig
    model_config: VitalSignTPPModelConfig

    def __post_init__(self) -> None:
        self.dataset_config = VitalSignTPPDatasetConfig.from_dict(self.dataset_config)
        self.model_config = VitalSignTPPModelConfig.from_dict(self.model_config)

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any] | "VitalSignTPPTrainingConfig",
    ) -> "VitalSignTPPTrainingConfig":
        if isinstance(payload, cls):
            return payload
        config_payload = dict(_coerce_mapping(value=payload, field_name="training_config"))
        return cls(
            dataset_config=VitalSignTPPDatasetConfig.from_dict(config_payload["dataset_config"]),
            model_config=VitalSignTPPModelConfig.from_dict(config_payload["model_config"]),
        )

    @classmethod
    def from_json_file(cls, config_path: str | Path) -> "VitalSignTPPTrainingConfig":
        return cls.from_dict(payload=_load_json_object(config_path=config_path))

    def with_overrides(
        self,
        overrides: Mapping[str, Any] | None,
    ) -> "VitalSignTPPTrainingConfig":
        if overrides is None:
            return VitalSignTPPTrainingConfig.from_dict(self)

        merged_payload = self.to_dict()
        for section_name, section_updates in dict(
            _coerce_mapping(value=overrides, field_name="model_overrides")
        ).items():
            section_mapping = dict(
                _coerce_mapping(value=section_updates, field_name=section_name)
            )
            if section_name not in merged_payload:
                raise ValueError(
                    f"Unknown training config section '{section_name}'. "
                    "Expected 'dataset_config' or 'model_config'."
                )
            merged_payload[section_name] = _deep_merge_dicts(
                base=merged_payload[section_name],
                updates=section_mapping,
            )

        return VitalSignTPPTrainingConfig.from_dict(merged_payload)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class VitalSignTPPPredictionJobConfig:
    dataset_config: VitalSignTPPDatasetConfig
    model_config: VitalSignTPPModelConfig
    checkpoint_path: str
    predictions_dir: Optional[str] = None
    load_predictions: bool = False
    prediction_splits: tuple[str, ...] = ("train", "val", "test")
    history_fraction: float = 0.6
    min_history_events: int = 3
    max_history_events: Optional[int] = None
    prediction_event_count: int = 5
    argmax: bool = True
    mean_of: int = 20
    median: bool = False

    def __post_init__(self) -> None:
        self.dataset_config = VitalSignTPPDatasetConfig.from_dict(self.dataset_config)
        self.model_config = VitalSignTPPModelConfig.from_dict(self.model_config)
        self.prediction_splits = _normalize_prediction_splits(self.prediction_splits)
        if not (0.0 < float(self.history_fraction) < 1.0):
            raise ValueError("history_fraction must be strictly between 0 and 1.")
        if self.min_history_events < 1:
            raise ValueError("min_history_events must be at least 1.")
        if self.max_history_events is not None and self.max_history_events < 1:
            raise ValueError("max_history_events must be at least 1 when provided.")
        if self.prediction_event_count < 1:
            raise ValueError("prediction_event_count must be at least 1.")
        if self.mean_of < 1:
            raise ValueError("mean_of must be at least 1.")
        if self.median and self.mean_of != 1:
            raise ValueError("mean_of must be 1 when median=True.")

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any] | "VitalSignTPPPredictionJobConfig",
    ) -> "VitalSignTPPPredictionJobConfig":
        if isinstance(payload, cls):
            return payload
        config_payload = dict(
            _coerce_mapping(value=payload, field_name="prediction_job_config")
        )
        return cls(
            dataset_config=VitalSignTPPDatasetConfig.from_dict(config_payload["dataset_config"]),
            model_config=VitalSignTPPModelConfig.from_dict(config_payload["model_config"]),
            checkpoint_path=str(config_payload["checkpoint_path"]),
            predictions_dir=config_payload.get("predictions_dir"),
            load_predictions=bool(config_payload.get("load_predictions", False)),
            prediction_splits=config_payload.get("prediction_splits", ("train", "val", "test")),
            history_fraction=float(config_payload.get("history_fraction", 0.6)),
            min_history_events=int(config_payload.get("min_history_events", 3)),
            max_history_events=config_payload.get("max_history_events"),
            prediction_event_count=int(config_payload.get("prediction_event_count", 5)),
            argmax=bool(config_payload.get("argmax", True)),
            mean_of=int(config_payload.get("mean_of", 20)),
            median=bool(config_payload.get("median", False)),
        )

    @classmethod
    def from_json_file(cls, config_path: str | Path) -> "VitalSignTPPPredictionJobConfig":
        return cls.from_dict(payload=_load_json_object(config_path=config_path))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
