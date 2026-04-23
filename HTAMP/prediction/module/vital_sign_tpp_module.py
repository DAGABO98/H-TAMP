from __future__ import annotations

from typing import Any

import lightning as L
import torch

from HTAMP.prediction.configs.vital_sign_tpp_config import VitalSignTPPModelConfig

from HTAMP.prediction.point_process_models.flexTPP.dataset.base import (
    BatchSpec,
    MODALITY_CATEGORICAL,
    MODALITY_CONTINUOUS,
    MODALITY_ORDINAL,
)

from HTAMP.prediction.point_process_models.flexTPP.model import PropertyMTPPFlexTPPModel


def _coerce_model_config(
    model_config: VitalSignTPPModelConfig | dict[str, Any],
) -> VitalSignTPPModelConfig:
    if isinstance(model_config, VitalSignTPPModelConfig):
        return model_config
    return VitalSignTPPModelConfig.from_dict(model_config)

def _conditioning_network_config(
    *,
    model_config: VitalSignTPPModelConfig,
    condition_dim: int,
) -> dict[str, Any] | None:
    if condition_dim <= 0:
        return None

    if model_config.conditioning_network is not None:
        conditioning_network = dict(model_config.conditioning_network)
        conditioning_network.setdefault("window_size", condition_dim)
        return conditioning_network

    return {
        "_target_": "flex_tpp.model.EmbeddingTransformer",
        "dim_k": None,
        "window_size": condition_dim,
        "dim_ff": model_config.dim_ff,
        "depth": min(2, model_config.depth),
        "normalize": model_config.normalize,
        "non_linearity": model_config.non_linearity,
        "dropout": min(float(model_config.dropout), 0.1),
        "is_causal": False,
        "positional_encoding_migrated": True,
    }

class VitalSignTPPModule(L.LightningModule):
    def __init__(
        self,
        *,
        model_config: VitalSignTPPModelConfig | dict[str, Any],
        dims: list[int] | tuple[int, ...],
        max_num_classes: int,
        condition_dim: int = 0,
    ) -> None:
        super().__init__()
        self.model_config = _coerce_model_config(model_config=model_config)
        self.dims = [int(dim_value) for dim_value in dims]
        self.max_num_classes = int(max_num_classes)
        self.condition_dim = int(condition_dim)

        self.flex_tpp_model = PropertyMTPPFlexTPPModel(
            dim=sum(self.dims),
            dim_m=self.model_config.dim_k * self.model_config.n_head,
            dim_k=self.model_config.dim_k,
            dim_ff=self.model_config.dim_ff,
            depth=self.model_config.depth,
            normalize=self.model_config.normalize,
            non_linearity=self.model_config.non_linearity,
            dropout=self.model_config.dropout,
            embed_event_index=self.model_config.embed_event_index,
            monotonic_bins=self.model_config.monotonic_bins,
            param_nets_n_hidden_layer=self.model_config.param_nets_n_hidden_layer,
            param_nets_hidden_dim_factor=self.model_config.param_nets_hidden_dim_factor,
            max_num_classes=self.max_num_classes,
            conditioning_network=_conditioning_network_config(
                model_config=self.model_config,
                condition_dim=self.condition_dim,
            ),
        )

        self.save_hyperparameters(
            {
                "model_config": self.model_config.to_dict(),
                "dims": self.dims,
                "max_num_classes": self.max_num_classes,
                "condition_dim": self.condition_dim,
            }
        )

    def forward(self, batch: BatchSpec) -> torch.Tensor:
        return self.flex_tpp_model.log_prob(batch)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.model_config.learning_rate,
            weight_decay=self.model_config.weight_decay,
        )

    def _compute_metrics(self, batch: BatchSpec) -> dict[str, torch.Tensor]:
        log_prob = self.flex_tpp_model.log_prob(batch)
        finite_mask = torch.isfinite(batch.data)
        if not finite_mask.any():
            zero = log_prob.sum() * 0.0
            return {
                "nll": zero,
                "nll_cont": zero,
                "nll_disc": zero,
            }

        nll = -log_prob[finite_mask].mean()

        discrete_mask = finite_mask & (
            (batch.types == MODALITY_CATEGORICAL) | (batch.types == MODALITY_ORDINAL)
        )
        continuous_mask = finite_mask & (
            (batch.types == MODALITY_CONTINUOUS)
            | (~((batch.types == MODALITY_CATEGORICAL) | (batch.types == MODALITY_ORDINAL)))
        )

        if continuous_mask.any():
            nll_cont = -log_prob[continuous_mask].mean()
        else:
            nll_cont = nll * 0.0

        if discrete_mask.any():
            nll_disc = -log_prob[discrete_mask].mean()
        else:
            nll_disc = nll * 0.0

        return {
            "nll": nll,
            "nll_cont": nll_cont,
            "nll_disc": nll_disc,
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

    def training_step(self, batch: BatchSpec, batch_idx: int) -> torch.Tensor:
        metrics = self._compute_metrics(batch)
        self._log_metrics("train", metrics, batch_size=int(batch.data.shape[0]))
        return metrics["nll"]

    def validation_step(self, batch: BatchSpec, batch_idx: int) -> dict[str, torch.Tensor]:
        metrics = self._compute_metrics(batch)
        self._log_metrics("val", metrics, batch_size=int(batch.data.shape[0]))
        return metrics

    def test_step(self, batch: BatchSpec, batch_idx: int) -> dict[str, torch.Tensor]:
        metrics = self._compute_metrics(batch)
        self._log_metrics("test", metrics, batch_size=int(batch.data.shape[0]))
        return metrics
