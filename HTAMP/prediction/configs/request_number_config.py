from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Optional

from HTAMP.data_processing.processing_dataclasses import AnnotatedDataFiles


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


@dataclass
class MedicalRequestDatasetConfig:
    annotated_data_files: AnnotatedDataFiles
    request_dir: str = "data/requests"
    dataset_dir: str = "data/prediction/request_numbers"
    start_date: str = "2024-06-24"
    end_date: str = "2025-06-29"
    time_step_minutes: int = 60
    patient_id_col: str = "MRN"
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    test_iso_weeks: tuple[tuple[int, int], ...] = field(default_factory=_default_test_iso_weeks)
    use_saved_request_data: bool = False

    def __post_init__(self) -> None:
        if self.time_step_minutes <= 0:
            raise ValueError("time_step_minutes must be greater than zero.")

        if min(self.train_ratio, self.val_ratio, self.test_ratio) < 0.0:
            raise ValueError("train_ratio, val_ratio, and test_ratio must be non-negative.")

        if (self.train_ratio + self.val_ratio) <= 0.0:
            raise ValueError("train_ratio and val_ratio must sum to a positive value.")

        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if not self.test_iso_weeks and abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(
                "train_ratio, val_ratio, and test_ratio must sum to 1.0 when test_iso_weeks is not provided."
            )

        normalized_weeks = []
        for iso_year, iso_week in self.test_iso_weeks:
            normalized_weeks.append((int(iso_year), int(iso_week)))
        self.test_iso_weeks = tuple(sorted(set(normalized_weeks)))


@dataclass
class TimeseriesModelConfig:
    model_name: str = "TimesNet"
    run_name: str = "TimesNet_medical_requests"
    preprocess_data: bool = False
    wandb: bool = False

    task_name: str = "long_term_forecast"
    seq_len: int = 96
    label_len: int = 48
    pred_len: int = 96

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

    accelerator: str = "auto"
    devices: int = 1
    strategy: Optional[str] = None

    @classmethod
    def from_namespace(cls, args) -> "TimeseriesModelConfig":
        field_names = {field.name for field in fields(cls)}
        values = {
            field_name: getattr(args, field_name)
            for field_name in field_names
            if hasattr(args, field_name)
        }
        return cls(**values)

    def sync_channel_dimensions(self, num_channels: int) -> None:
        self.enc_in = num_channels
        self.dec_in = num_channels
        self.c_out = num_channels

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
