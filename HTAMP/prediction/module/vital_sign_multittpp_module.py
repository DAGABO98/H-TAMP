from __future__ import annotations

from typing import Any

import lightning as L
import torch

from HTAMP.prediction.configs.vital_sign_multittpp_config import (
    VitalSignMultiTTPPModelConfig,
)
from HTAMP.prediction.point_process_models.multittpp import models as multittpp_models
from HTAMP.prediction.point_process_models.multittpp.data import Batch


def _coerce_model_config(
    model_config: VitalSignMultiTTPPModelConfig | dict[str, Any],
) -> VitalSignMultiTTPPModelConfig:
    if isinstance(model_config, VitalSignMultiTTPPModelConfig):
        return model_config
    return VitalSignMultiTTPPModelConfig.from_dict(model_config)


def _move_batch_to_device(batch: Batch, device: torch.device) -> Batch:
    return Batch(
        in_dts=batch.in_dts.to(device),
        in_types=batch.in_types.to(device),
        in_times=batch.in_times.to(device),
        seq_lengths=batch.seq_lengths.to(device),
        last_times=batch.last_times.to(device),
        out_dts=batch.out_dts.to(device),
        out_types=batch.out_types.to(device),
        N_min=batch.N_min,
    )


class VitalSignMultiTTPPModule(L.LightningModule):
    def __init__(
        self,
        *,
        model_config: VitalSignMultiTTPPModelConfig | dict[str, Any],
        num_event_types: int,
        n_events: int,
        t_max_normalization: float,
        dt_max_normalization: float,
    ) -> None:
        super().__init__()
        self.model_config = _coerce_model_config(model_config=model_config)
        self.num_event_types = int(num_event_types)
        self.n_events = int(n_events)
        self.t_max_normalization = float(t_max_normalization)
        self.dt_max_normalization = float(dt_max_normalization)

        model_cls = getattr(multittpp_models, self.model_config.model_name)
        self.multittpp_model = model_cls(
            **self.model_config.to_multittpp_kwargs(
                n_marks=self.num_event_types,
                n_events=self.n_events,
                t_max_normalization=self.t_max_normalization,
                dt_max_normalization=self.dt_max_normalization,
            )
        )
        self.multittpp_model.update_n_max(self.n_events)
        if self.model_config.use_jit:
            self.multittpp_model = torch.jit.script(self.multittpp_model)

        self.save_hyperparameters(
            {
                "model_config": self.model_config.to_dict(),
                "num_event_types": self.num_event_types,
                "n_events": self.n_events,
                "t_max_normalization": self.t_max_normalization,
                "dt_max_normalization": self.dt_max_normalization,
            }
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.model_config.learning_rate,
            weight_decay=self.model_config.weight_decay,
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        return self._compute_metrics(batch)["nll"]

    def _compute_metrics(self, batch: Batch) -> dict[str, torch.Tensor]:
        batch = _move_batch_to_device(batch, self.device)
        loss = self.multittpp_model.loss(
            batch.in_times,
            batch.in_types,
            batch.last_times,
        )
        valid_event_count = (batch.out_types < self.num_event_types).sum().clamp_min(1)
        normalizer = valid_event_count.to(dtype=loss.dtype, device=loss.device)
        nll = loss / normalizer
        return {
            "loss": loss,
            "nll": nll,
            "num_events": normalizer,
        }

    def _log_metrics(
        self,
        prefix: str,
        metrics: dict[str, torch.Tensor],
        *,
        batch_size: int,
    ) -> None:
        for metric_name, metric_value in metrics.items():
            self.log(
                f"{prefix}_{metric_name}",
                metric_value,
                batch_size=batch_size,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
                prog_bar=(metric_name == "nll"),
            )

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        metrics = self._compute_metrics(batch)
        self._log_metrics("train", metrics, batch_size=int(batch.in_times.shape[0]))
        return metrics["nll"]

    def validation_step(self, batch: Batch, batch_idx: int) -> dict[str, torch.Tensor]:
        metrics = self._compute_metrics(batch)
        self._log_metrics("val", metrics, batch_size=int(batch.in_times.shape[0]))
        return metrics

    def test_step(self, batch: Batch, batch_idx: int) -> dict[str, torch.Tensor]:
        metrics = self._compute_metrics(batch)
        self._log_metrics("test", metrics, batch_size=int(batch.in_times.shape[0]))
        return metrics

    def generate_future(
        self,
        *,
        prefix_times: torch.Tensor,
        prefix_types: torch.Tensor,
        max_future_events: int,
        n_min: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.multittpp_model.eval()
        return self.multittpp_model.generate(
            prefix_times.to(self.device),
            prefix_types.to(self.device),
            int(max_future_events),
            int(n_min),
        )
