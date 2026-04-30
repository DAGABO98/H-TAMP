from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from HTAMP.prediction.configs.vital_sign_easy_tpp_config import (
    VitalSignEasyTPPDatasetConfig,
)
from HTAMP.prediction.configs.vital_sign_tpp_config import (
    _coerce_mapping,
    _deep_merge_dicts,
    _load_json_object,
)

SUPPORTED_MULTITTPP_MODELS = (
    "InhomogeneousPoisson",
    "Renewal",
    "ModulatedRenewal",
    "TriTPP",
    "SplineTransformer",
)


@dataclass
class VitalSignMultiTTPPDatasetConfig(VitalSignEasyTPPDatasetConfig):
    dataset_dir: str = "data/prediction/vital_sign_multittpp_dataset"
    include_eos_event: bool = False


@dataclass
class VitalSignMultiTTPPModelConfig:
    run_name: str = "vital_sign_multittpp_tritpp"
    model_name: str = "TriTPP"
    wandb: bool = False
    batch_size: int = 32
    val_batch_size: Optional[int] = None
    num_workers: int = 0
    max_epochs: int = 50
    patience: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    gradient_clip_val: float = 5.0
    monitor_metric: str = "val_nll"
    monitor_mode: Optional[str] = "min"
    accelerator: str = "auto"
    devices: int = 1
    strategy: Optional[str] = None
    precision: str | int = "32-true"
    accumulate_grad_batches: int = 1
    seed: int = 42
    n_samples: int = 100
    fixed_normalization: float = 50.0
    trainable_normalization: float = 1.0
    burn_in: int = 0
    n_blocks: int = 4
    n_embd: int = 8
    n_knots: int = 20
    spline_order: int = 2
    block_size: int = 16
    n_heads: int = 4
    dropout: float = 0.1
    enable_nan_check: bool = False
    use_jit: bool = False
    model_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.model_name = str(self.model_name).strip()
        if self.model_name not in SUPPORTED_MULTITTPP_MODELS:
            raise ValueError(
                f"Unsupported MultiTTPP model_name '{self.model_name}'. "
                f"Expected one of {SUPPORTED_MULTITTPP_MODELS}."
            )
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")
        if self.val_batch_size is not None and self.val_batch_size <= 0:
            raise ValueError("val_batch_size must be greater than zero when provided.")
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
        if self.n_samples <= 0:
            raise ValueError("n_samples must be greater than zero.")
        if self.fixed_normalization <= 0.0:
            raise ValueError("fixed_normalization must be greater than zero.")
        if self.trainable_normalization <= 0.0:
            raise ValueError("trainable_normalization must be greater than zero.")
        if min(self.n_blocks, self.n_embd, self.n_knots, self.spline_order, self.block_size) <= 0:
            raise ValueError(
                "n_blocks, n_embd, n_knots, spline_order, and block_size must be positive."
            )
        if self.n_heads <= 0:
            raise ValueError("n_heads must be greater than zero.")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout must be in the range [0, 1).")
        if self.monitor_mode is not None:
            self.monitor_mode = str(self.monitor_mode).strip().lower()
            if self.monitor_mode not in {"min", "max"}:
                raise ValueError("monitor_mode must be either 'min', 'max', or null.")
        self.model_kwargs = dict(
            _coerce_mapping(value=self.model_kwargs, field_name="model_kwargs")
        )

    @property
    def resolved_val_batch_size(self) -> int:
        return self.batch_size if self.val_batch_size is None else int(self.val_batch_size)

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any] | "VitalSignMultiTTPPModelConfig",
    ) -> "VitalSignMultiTTPPModelConfig":
        if isinstance(payload, cls):
            return payload
        return cls(**dict(_coerce_mapping(value=payload, field_name="model_config")))

    @classmethod
    def from_json_file(cls, config_path: str | Path) -> "VitalSignMultiTTPPModelConfig":
        return cls.from_dict(payload=_load_json_object(config_path=config_path))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_multittpp_kwargs(
        self,
        *,
        n_marks: int,
        n_events: int,
        t_max_normalization: float,
        dt_max_normalization: float,
    ) -> dict[str, Any]:
        kwargs = {
            "n_marks": int(n_marks),
            "n_events": int(n_events),
            "n_blocks": int(self.n_blocks),
            "n_embd": int(self.n_embd),
            "n_knots": int(self.n_knots),
            "spline_order": int(self.spline_order),
            "block_size": int(self.block_size),
            "n_heads": int(self.n_heads),
            "dropout": float(self.dropout),
            "fixed_normalization": float(self.fixed_normalization),
            "trainable_normalization": float(self.trainable_normalization),
            "t_max_normalization": float(t_max_normalization),
            "dt_max_normalization": float(dt_max_normalization),
            "burn_in": int(self.burn_in),
        }
        kwargs.update(self.model_kwargs)
        return kwargs


@dataclass
class VitalSignMultiTTPPTrainingConfig:
    dataset_config: VitalSignMultiTTPPDatasetConfig
    model_config: VitalSignMultiTTPPModelConfig

    def __post_init__(self) -> None:
        self.dataset_config = VitalSignMultiTTPPDatasetConfig.from_dict(self.dataset_config)
        self.model_config = VitalSignMultiTTPPModelConfig.from_dict(self.model_config)

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any] | "VitalSignMultiTTPPTrainingConfig",
    ) -> "VitalSignMultiTTPPTrainingConfig":
        if isinstance(payload, cls):
            return payload
        config_payload = dict(_coerce_mapping(value=payload, field_name="training_config"))
        return cls(
            dataset_config=VitalSignMultiTTPPDatasetConfig.from_dict(
                config_payload["dataset_config"]
            ),
            model_config=VitalSignMultiTTPPModelConfig.from_dict(
                config_payload["model_config"]
            ),
        )

    @classmethod
    def from_json_file(cls, config_path: str | Path) -> "VitalSignMultiTTPPTrainingConfig":
        return cls.from_dict(payload=_load_json_object(config_path=config_path))

    def with_overrides(
        self,
        overrides: Mapping[str, Any] | None,
    ) -> "VitalSignMultiTTPPTrainingConfig":
        if overrides is None:
            return VitalSignMultiTTPPTrainingConfig.from_dict(self)

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

        return VitalSignMultiTTPPTrainingConfig.from_dict(merged_payload)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def write_json_config(path: str | Path, config: VitalSignMultiTTPPTrainingConfig) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as config_file:
        json.dump(config.to_dict(), config_file, indent=2, default=str)
