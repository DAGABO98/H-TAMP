from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from HTAMP.prediction.configs.delivery_request_config import (
    DeliveryRequestDatasetConfig,
    _coerce_mapping,
    _deep_merge_dicts,
    _load_json_object,
)
from HTAMP.prediction.configs.vital_sign_easy_tpp_config import (
    VitalSignEasyTPPModelConfig,
)
from HTAMP.prediction.configs.vital_sign_multittpp_config import (
    VitalSignMultiTTPPModelConfig,
)
from HTAMP.prediction.configs.vital_sign_tpp_config import VitalSignTPPModelConfig

SUPPORTED_DELIVERY_MARK_MODES = ("task", "medication_code")
DELIVERY_MARK_MODE_ALIASES = {
    "standard": "task",
    "plain": "task",
    "task_only": "task",
    "medication": "task",
    "enhanced": "medication_code",
    "enhanced_marks": "medication_code",
    "medication_type": "medication_code",
    "medicine_type": "medication_code",
    "code": "medication_code",
}


def _normalize_delivery_mark_mode(raw_value: str | None) -> str:
    normalized = str(raw_value or "task").strip().lower()
    normalized = DELIVERY_MARK_MODE_ALIASES.get(normalized, normalized)
    if normalized not in SUPPORTED_DELIVERY_MARK_MODES:
        raise ValueError(
            f"Unsupported delivery mark mode '{raw_value}'. "
            f"Expected one of {SUPPORTED_DELIVERY_MARK_MODES}."
        )
    return normalized


def _normalize_scheduled_time_col(raw_value: str | None) -> str:
    value = str(raw_value or "Medication Scheduled DTTM").strip()
    if not value:
        raise ValueError("medication_scheduled_time_col must not be empty.")
    return value


@dataclass
class DeliveryTPPDatasetConfig(DeliveryRequestDatasetConfig):
    dataset_dir: str = "data/prediction/delivery_tpp_dataset"
    event_type_mark_mode: str = "task"
    medication_scheduled_time_col: str = "Medication Scheduled DTTM"
    unknown_medication_label: str = "unknown_medication"
    include_medication_code_as_property: bool = True
    use_previous_day_summary_conditioning: bool = True
    min_events_per_sequence: int = 2
    max_events_per_sequence: Optional[int] = None
    eos_offset_minutes: float = 5.0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.event_type_mark_mode = _normalize_delivery_mark_mode(self.event_type_mark_mode)
        self.medication_scheduled_time_col = _normalize_scheduled_time_col(
            self.medication_scheduled_time_col
        )
        self.unknown_medication_label = str(self.unknown_medication_label).strip().lower()
        if not self.unknown_medication_label:
            raise ValueError("unknown_medication_label must not be empty.")
        if self.min_events_per_sequence < 1:
            raise ValueError("min_events_per_sequence must be at least 1.")
        if self.max_events_per_sequence is not None and self.max_events_per_sequence < 1:
            raise ValueError("max_events_per_sequence must be at least 1 when provided.")
        if self.eos_offset_minutes <= 0.0:
            raise ValueError("eos_offset_minutes must be greater than zero.")

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any] | "DeliveryTPPDatasetConfig",
    ) -> "DeliveryTPPDatasetConfig":
        if isinstance(payload, cls):
            return payload
        dataset_payload = dict(_coerce_mapping(value=payload, field_name="dataset_config"))
        if "annotated_data_files" not in dataset_payload:
            raise ValueError("dataset_config must include an 'annotated_data_files' object.")
        return cls(**dataset_payload)

    @classmethod
    def from_json_file(cls, config_path: str | Path) -> "DeliveryTPPDatasetConfig":
        return cls.from_dict(payload=_load_json_object(config_path=config_path))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class DeliveryTPPModelConfig(VitalSignTPPModelConfig):
    run_name: str = "delivery_flex_tpp"


@dataclass
class DeliveryTPPTrainingConfig:
    dataset_config: DeliveryTPPDatasetConfig
    model_config: DeliveryTPPModelConfig

    def __post_init__(self) -> None:
        self.dataset_config = DeliveryTPPDatasetConfig.from_dict(self.dataset_config)
        self.model_config = DeliveryTPPModelConfig.from_dict(self.model_config)

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any] | "DeliveryTPPTrainingConfig",
    ) -> "DeliveryTPPTrainingConfig":
        if isinstance(payload, cls):
            return payload
        config_payload = dict(_coerce_mapping(value=payload, field_name="training_config"))
        return cls(
            dataset_config=DeliveryTPPDatasetConfig.from_dict(config_payload["dataset_config"]),
            model_config=DeliveryTPPModelConfig.from_dict(config_payload["model_config"]),
        )

    @classmethod
    def from_json_file(cls, config_path: str | Path) -> "DeliveryTPPTrainingConfig":
        return cls.from_dict(payload=_load_json_object(config_path=config_path))

    def with_overrides(
        self,
        overrides: Mapping[str, Any] | None,
    ) -> "DeliveryTPPTrainingConfig":
        if overrides is None:
            return DeliveryTPPTrainingConfig.from_dict(self)

        merged_payload = self.to_dict()
        for section_name, section_updates in dict(
            _coerce_mapping(value=overrides, field_name="model_overrides")
        ).items():
            if section_name not in merged_payload:
                raise ValueError(
                    f"Unknown training config section '{section_name}'. "
                    "Expected 'dataset_config' or 'model_config'."
                )
            merged_payload[section_name] = _deep_merge_dicts(
                base=merged_payload[section_name],
                updates=dict(_coerce_mapping(value=section_updates, field_name=section_name)),
            )
        return DeliveryTPPTrainingConfig.from_dict(merged_payload)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class DeliveryEasyTPPDatasetConfig(DeliveryTPPDatasetConfig):
    dataset_dir: str = "data/prediction/delivery_easy_tpp_dataset"
    mark_label_mode: str = "medication_code"
    include_eos_event: bool = True

    def __post_init__(self) -> None:
        self.event_type_mark_mode = _normalize_delivery_mark_mode(self.mark_label_mode)
        super().__post_init__()
        self.mark_label_mode = self.event_type_mark_mode

    def to_delivery_tpp_dataset_config(self, *, save_data: bool = False) -> DeliveryTPPDatasetConfig:
        payload = self.to_dict()
        payload.pop("mark_label_mode", None)
        payload.pop("include_eos_event", None)
        payload["use_saved_dataset"] = False
        payload["preprocess_data"] = True
        payload["save_data"] = bool(save_data)
        return DeliveryTPPDatasetConfig.from_dict(payload)


@dataclass
class DeliveryEasyTPPModelConfig(VitalSignEasyTPPModelConfig):
    run_name: str = "delivery_easy_tpp_nhp"


@dataclass
class DeliveryEasyTPPTrainingConfig:
    dataset_config: DeliveryEasyTPPDatasetConfig
    model_config: DeliveryEasyTPPModelConfig

    def __post_init__(self) -> None:
        self.dataset_config = DeliveryEasyTPPDatasetConfig.from_dict(self.dataset_config)
        self.model_config = DeliveryEasyTPPModelConfig.from_dict(self.model_config)

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any] | "DeliveryEasyTPPTrainingConfig",
    ) -> "DeliveryEasyTPPTrainingConfig":
        if isinstance(payload, cls):
            return payload
        config_payload = dict(_coerce_mapping(value=payload, field_name="training_config"))
        return cls(
            dataset_config=DeliveryEasyTPPDatasetConfig.from_dict(config_payload["dataset_config"]),
            model_config=DeliveryEasyTPPModelConfig.from_dict(config_payload["model_config"]),
        )

    @classmethod
    def from_json_file(cls, config_path: str | Path) -> "DeliveryEasyTPPTrainingConfig":
        return cls.from_dict(payload=_load_json_object(config_path=config_path))

    def with_overrides(
        self,
        overrides: Mapping[str, Any] | None,
    ) -> "DeliveryEasyTPPTrainingConfig":
        if overrides is None:
            return DeliveryEasyTPPTrainingConfig.from_dict(self)
        merged_payload = self.to_dict()
        for section_name, section_updates in dict(
            _coerce_mapping(value=overrides, field_name="model_overrides")
        ).items():
            if section_name not in merged_payload:
                raise ValueError(
                    f"Unknown training config section '{section_name}'. "
                    "Expected 'dataset_config' or 'model_config'."
                )
            merged_payload[section_name] = _deep_merge_dicts(
                base=merged_payload[section_name],
                updates=dict(_coerce_mapping(value=section_updates, field_name=section_name)),
            )
        return DeliveryEasyTPPTrainingConfig.from_dict(merged_payload)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class DeliveryMultiTTPPDatasetConfig(DeliveryEasyTPPDatasetConfig):
    dataset_dir: str = "data/prediction/delivery_multittpp_dataset"
    include_eos_event: bool = False


@dataclass
class DeliveryMultiTTPPModelConfig(VitalSignMultiTTPPModelConfig):
    run_name: str = "delivery_multittpp_tritpp"


@dataclass
class DeliveryMultiTTPPTrainingConfig:
    dataset_config: DeliveryMultiTTPPDatasetConfig
    model_config: DeliveryMultiTTPPModelConfig

    def __post_init__(self) -> None:
        self.dataset_config = DeliveryMultiTTPPDatasetConfig.from_dict(self.dataset_config)
        self.model_config = DeliveryMultiTTPPModelConfig.from_dict(self.model_config)

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any] | "DeliveryMultiTTPPTrainingConfig",
    ) -> "DeliveryMultiTTPPTrainingConfig":
        if isinstance(payload, cls):
            return payload
        config_payload = dict(_coerce_mapping(value=payload, field_name="training_config"))
        return cls(
            dataset_config=DeliveryMultiTTPPDatasetConfig.from_dict(config_payload["dataset_config"]),
            model_config=DeliveryMultiTTPPModelConfig.from_dict(config_payload["model_config"]),
        )

    @classmethod
    def from_json_file(cls, config_path: str | Path) -> "DeliveryMultiTTPPTrainingConfig":
        return cls.from_dict(payload=_load_json_object(config_path=config_path))

    def with_overrides(
        self,
        overrides: Mapping[str, Any] | None,
    ) -> "DeliveryMultiTTPPTrainingConfig":
        if overrides is None:
            return DeliveryMultiTTPPTrainingConfig.from_dict(self)
        merged_payload = self.to_dict()
        for section_name, section_updates in dict(
            _coerce_mapping(value=overrides, field_name="model_overrides")
        ).items():
            if section_name not in merged_payload:
                raise ValueError(
                    f"Unknown training config section '{section_name}'. "
                    "Expected 'dataset_config' or 'model_config'."
                )
            merged_payload[section_name] = _deep_merge_dicts(
                base=merged_payload[section_name],
                updates=dict(_coerce_mapping(value=section_updates, field_name=section_name)),
            )
        return DeliveryMultiTTPPTrainingConfig.from_dict(merged_payload)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
