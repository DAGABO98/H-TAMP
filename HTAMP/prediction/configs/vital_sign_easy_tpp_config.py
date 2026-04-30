from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from HTAMP.prediction.configs.vital_sign_tpp_config import (
    SUPPORTED_VALIDATION_SPLIT_STRATEGIES,
    VALIDATION_SPLIT_STRATEGY_ALIASES,
    VitalSignTPPDatasetConfig,
    _build_annotated_data_files,
    _coerce_mapping,
    _deep_merge_dicts,
    _default_test_iso_weeks_by_floor,
    _load_json_object,
    _normalize_floor_key,
    _normalize_prediction_splits,
    _normalize_tasks,
    _normalize_test_iso_weeks,
    _normalize_validation_split_seed,
    IsoWeek,
)

SUPPORTED_EASY_TPP_MODELS = (
    "NHP",
    "AttNHP",
    "THP",
    "SAHP",
    "RMTPP",
    "FullyNN",
    "IntensityFree",
    "ODETPP",
    "ANHN",
    "S2P2",
    "WSMTHP",
)
SUPPORTED_LABEL_STRATEGIES = ("quantile", "threshold")
SUPPORTED_MARK_LABEL_MODES = ("task_label", "task_component_label", "task_only")
DEFAULT_LABEL_NAMES = ("low", "medium", "high")


def _default_label_component_by_task() -> dict[str, tuple[str, ...]]:
    return {
        "blood_pressure": ("systolic", "diastolic"),
    }


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


def _default_model_specs(model_id: str) -> dict[str, Any]:
    if model_id == "FullyNN":
        return {
            "num_mlp_layers": 2,
            "proper_marked_intensities": True,
        }
    if model_id == "IntensityFree":
        return {
            "num_mix_components": 64,
        }
    if model_id == "ODETPP":
        return {
            "ode_num_sample_per_step": 10,
        }
    if model_id == "S2P2":
        return {
            "P": 32,
        }
    if model_id == "WSMTHP":
        return {
            "T_mode": "train_global",
            "CE_coef": 10.0,
        }
    return {}


@dataclass
class VitalSignEasyTPPDatasetConfig(VitalSignTPPDatasetConfig):
    dataset_dir: str = "data/prediction/vital_sign_easy_tpp_dataset"
    label_strategy: str = "quantile"
    label_names: tuple[str, str, str] = DEFAULT_LABEL_NAMES
    quantile_edges: tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0)
    measurement_thresholds: dict[str, Any] = field(default_factory=dict)
    label_component_by_task: dict[str, tuple[str, ...]] = field(
        default_factory=_default_label_component_by_task
    )
    mark_label_mode: str = "task_label"
    missing_label: str = "unknown"
    drop_missing_measurement_events: bool = False
    include_eos_event: bool = True

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
        self.mark_label_mode = str(self.mark_label_mode).strip().lower()
        if self.mark_label_mode not in SUPPORTED_MARK_LABEL_MODES:
            raise ValueError(
                f"Unsupported mark_label_mode '{self.mark_label_mode}'. "
                f"Expected one of {SUPPORTED_MARK_LABEL_MODES}."
            )
        self.missing_label = str(self.missing_label).strip().lower()
        if not self.missing_label:
            raise ValueError("missing_label must not be empty.")

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any] | "VitalSignEasyTPPDatasetConfig",
    ) -> "VitalSignEasyTPPDatasetConfig":
        if isinstance(payload, cls):
            return payload
        dataset_payload = dict(_coerce_mapping(value=payload, field_name="dataset_config"))
        if "annotated_data_files" not in dataset_payload:
            raise ValueError("dataset_config must include an 'annotated_data_files' object.")
        return cls(**dataset_payload)

    @classmethod
    def from_json_file(cls, config_path: str | Path) -> "VitalSignEasyTPPDatasetConfig":
        return cls.from_dict(payload=_load_json_object(config_path=config_path))

    def to_vital_sign_tpp_dataset_config(
        self,
        *,
        save_data: bool = False,
    ) -> VitalSignTPPDatasetConfig:
        payload = self.to_dict()
        for field_name in (
            "label_strategy",
            "label_names",
            "quantile_edges",
            "measurement_thresholds",
            "label_component_by_task",
            "mark_label_mode",
            "missing_label",
            "drop_missing_measurement_events",
            "include_eos_event",
            "event_type_mark_mode",
        ):
            payload.pop(field_name, None)
        payload["use_saved_dataset"] = False
        payload["preprocess_data"] = True
        payload["save_data"] = bool(save_data)
        return VitalSignTPPDatasetConfig.from_dict(payload)


@dataclass
class VitalSignEasyTPPModelConfig:
    run_name: str = "vital_sign_easy_tpp_nhp"
    model_id: str = "NHP"
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
    precision: str | int = "32-true"
    accumulate_grad_batches: int = 1
    gpu: int = -1
    rnn_type: str = "LSTM"
    hidden_size: int = 64
    time_emb_size: int = 16
    num_layers: int = 2
    num_heads: int = 2
    sharing_param_layer: bool = False
    use_mc_samples: bool = True
    loss_integral_num_sample_per_step: int = 20
    dropout_rate: float = 0.0
    use_ln: bool = False
    thinning: Optional[dict[str, Any]] = None
    model_specs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.model_id = str(self.model_id).strip()
        if self.model_id not in SUPPORTED_EASY_TPP_MODELS:
            raise ValueError(
                f"Unsupported EasyTPP model_id '{self.model_id}'. "
                f"Expected one of {SUPPORTED_EASY_TPP_MODELS}."
            )
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
        if self.accumulate_grad_batches <= 0:
            raise ValueError("accumulate_grad_batches must be greater than zero.")
        if isinstance(self.precision, str):
            self.precision = self.precision.strip()
            if not self.precision:
                raise ValueError("precision must not be empty.")
        if min(self.hidden_size, self.time_emb_size, self.num_layers, self.num_heads) <= 0:
            raise ValueError(
                "hidden_size, time_emb_size, num_layers, and num_heads must be greater than zero."
            )
        if self.loss_integral_num_sample_per_step <= 0:
            raise ValueError("loss_integral_num_sample_per_step must be greater than zero.")
        if self.dropout_rate < 0.0 or self.dropout_rate >= 1.0:
            raise ValueError("dropout_rate must be in the range [0, 1).")

        self.monitor_metric = str(self.monitor_metric).strip()
        if not self.monitor_metric:
            raise ValueError("monitor_metric must not be empty.")
        if self.monitor_mode is not None:
            self.monitor_mode = str(self.monitor_mode).strip().lower()
            if self.monitor_mode not in {"min", "max"}:
                raise ValueError("monitor_mode must be either 'min', 'max', or null.")
        self.thinning = None if self.thinning is None else dict(
            _coerce_mapping(value=self.thinning, field_name="thinning")
        )
        supplied_specs = dict(_coerce_mapping(value=self.model_specs, field_name="model_specs"))
        self.model_specs = _deep_merge_dicts(
            base=_default_model_specs(self.model_id),
            updates=supplied_specs,
        )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any] | "VitalSignEasyTPPModelConfig",
    ) -> "VitalSignEasyTPPModelConfig":
        if isinstance(payload, cls):
            return payload
        return cls(**dict(_coerce_mapping(value=payload, field_name="model_config")))

    @classmethod
    def from_json_file(cls, config_path: str | Path) -> "VitalSignEasyTPPModelConfig":
        return cls.from_dict(payload=_load_json_object(config_path=config_path))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class VitalSignEasyTPPTrainingConfig:
    dataset_config: VitalSignEasyTPPDatasetConfig
    model_config: VitalSignEasyTPPModelConfig

    def __post_init__(self) -> None:
        self.dataset_config = VitalSignEasyTPPDatasetConfig.from_dict(self.dataset_config)
        self.model_config = VitalSignEasyTPPModelConfig.from_dict(self.model_config)

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any] | "VitalSignEasyTPPTrainingConfig",
    ) -> "VitalSignEasyTPPTrainingConfig":
        if isinstance(payload, cls):
            return payload
        config_payload = dict(_coerce_mapping(value=payload, field_name="training_config"))
        return cls(
            dataset_config=VitalSignEasyTPPDatasetConfig.from_dict(
                config_payload["dataset_config"]
            ),
            model_config=VitalSignEasyTPPModelConfig.from_dict(
                config_payload["model_config"]
            ),
        )

    @classmethod
    def from_json_file(cls, config_path: str | Path) -> "VitalSignEasyTPPTrainingConfig":
        return cls.from_dict(payload=_load_json_object(config_path=config_path))

    def with_overrides(
        self,
        overrides: Mapping[str, Any] | None,
    ) -> "VitalSignEasyTPPTrainingConfig":
        if overrides is None:
            return VitalSignEasyTPPTrainingConfig.from_dict(self)

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

        return VitalSignEasyTPPTrainingConfig.from_dict(merged_payload)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class VitalSignEasyTPPPredictionJobConfig:
    dataset_config: VitalSignEasyTPPDatasetConfig
    model_config: VitalSignEasyTPPModelConfig
    checkpoint_path: str
    predictions_dir: Optional[str] = None
    load_predictions: bool = False
    prediction_splits: tuple[str, ...] = ("train", "val", "test")
    prediction_event_count: int = 5

    def __post_init__(self) -> None:
        self.dataset_config = VitalSignEasyTPPDatasetConfig.from_dict(self.dataset_config)
        self.model_config = VitalSignEasyTPPModelConfig.from_dict(self.model_config)
        self.prediction_splits = _normalize_prediction_splits(self.prediction_splits)
        if self.prediction_event_count < 1:
            raise ValueError("prediction_event_count must be at least 1.")

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any] | "VitalSignEasyTPPPredictionJobConfig",
    ) -> "VitalSignEasyTPPPredictionJobConfig":
        if isinstance(payload, cls):
            return payload
        config_payload = dict(
            _coerce_mapping(value=payload, field_name="prediction_job_config")
        )
        return cls(
            dataset_config=VitalSignEasyTPPDatasetConfig.from_dict(
                config_payload["dataset_config"]
            ),
            model_config=VitalSignEasyTPPModelConfig.from_dict(
                config_payload["model_config"]
            ),
            checkpoint_path=str(config_payload["checkpoint_path"]),
            predictions_dir=config_payload.get("predictions_dir"),
            load_predictions=bool(config_payload.get("load_predictions", False)),
            prediction_splits=config_payload.get("prediction_splits", ("train", "val", "test")),
            prediction_event_count=int(config_payload.get("prediction_event_count", 5)),
        )

    @classmethod
    def from_json_file(cls, config_path: str | Path) -> "VitalSignEasyTPPPredictionJobConfig":
        return cls.from_dict(payload=_load_json_object(config_path=config_path))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def write_vital_sign_easy_tpp_training_config(
    training_config: VitalSignEasyTPPTrainingConfig,
    config_path: str | Path,
) -> None:
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as config_file:
        json.dump(training_config.to_dict(), config_file, indent=2)
