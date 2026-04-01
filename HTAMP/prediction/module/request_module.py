from __future__ import annotations

from typing import Any

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F

from HTAMP.prediction.configs.request_config import TimeseriesModelConfig
from HTAMP.prediction.time_series_models.architectures.i_transformer import Model as ITransformerModel
from HTAMP.prediction.time_series_models.architectures.patch_tst import Model as PatchTSTModel
from HTAMP.prediction.time_series_models.architectures.time_mixer import Model as TimeMixerModel
from HTAMP.prediction.time_series_models.architectures.times_net import Model as TimesNetModel
from HTAMP.prediction.time_series_models.utils.metrics import metric


def _coerce_model_config(model_config: TimeseriesModelConfig | dict[str, Any]) -> TimeseriesModelConfig:
    if isinstance(model_config, TimeseriesModelConfig):
        return model_config
    return TimeseriesModelConfig(**model_config)


class RequestsModule(L.LightningModule):
    def __init__(
        self,
        model_config: TimeseriesModelConfig | dict[str, Any],
        target_scaler_mean: list[float] | np.ndarray,
        target_scaler_scale: list[float] | np.ndarray,
        delta_target_indices: list[int],
        availability_target_indices: list[int]
    ) -> None:
        super().__init__()

        model_dict = {
            "TimesNet": TimesNetModel,
            "TimeMixer": TimeMixerModel,
            "iTransformer": ITransformerModel,
            "PatchTST": PatchTSTModel,
        }

        self.model_config = _coerce_model_config(model_config=model_config)
        if self.model_config.model_name not in model_dict:
            raise ValueError(f"Unsupported model_name: {self.model_config.model_name}")

        self.forecaster = model_dict[self.model_config.model_name](self.model_config).float()
        self.delta_criterion = (
            torch.nn.MSELoss()
            if self.model_config.loss == "MSE"
            else torch.nn.L1Loss()
        )

        scaler_mean_tensor = torch.as_tensor(target_scaler_mean, dtype=torch.float32)
        scaler_scale_tensor = torch.as_tensor(target_scaler_scale, dtype=torch.float32)
        delta_target_indices_tensor = torch.as_tensor(delta_target_indices, dtype=torch.long)
        availability_target_indices_tensor = torch.as_tensor(availability_target_indices, dtype=torch.long)

        self.register_buffer("target_scaler_mean", scaler_mean_tensor)
        self.register_buffer("target_scaler_scale", scaler_scale_tensor)
        self.register_buffer("delta_target_indices", delta_target_indices_tensor)
        self.register_buffer("availability_target_indices", availability_target_indices_tensor)

        self.save_hyperparameters(
            {
                "model_config": self.model_config.to_dict(),
                "target_scaler_mean": scaler_mean_tensor.detach().cpu().tolist(),
                "target_scaler_scale": scaler_scale_tensor.detach().cpu().tolist(),
                "delta_target_indices": delta_target_indices_tensor.detach().cpu().tolist(),
                "availability_target_indices": availability_target_indices_tensor.detach().cpu().tolist(),
            }
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.model_config.learning_rate)

    def _log_stats(self, section: str, outs: dict[str, torch.Tensor | float]) -> None:
        for key, stat in outs.items():
            if isinstance(stat, np.ndarray):
                stat = float(stat.mean())
            if isinstance(stat, torch.Tensor):
                stat = stat.mean()
            self.log(
                f"{section}_{key}",
                stat,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )

    def _inverse_transform_delta_tensor(self, data: torch.Tensor) -> torch.Tensor:
        scale = self.target_scaler_scale.view(1, 1, -1)
        mean = self.target_scaler_mean.view(1, 1, -1)
        return data * scale + mean

    def _select_delta_targets(self, data: torch.Tensor) -> torch.Tensor:
        return torch.index_select(data, dim=-1, index=self.delta_target_indices)

    def _select_availability_targets(self, data: torch.Tensor) -> torch.Tensor:
        return torch.index_select(data, dim=-1, index=self.availability_target_indices)

    def _compute_delta_loss(
        self,
        pred_delta: torch.Tensor,
        true_delta: torch.Tensor,
        true_availability: torch.Tensor,
    ) -> torch.Tensor:
        valid_mask = true_availability > 0.5
        if valid_mask.any():
            return self.delta_criterion(pred_delta[valid_mask], true_delta[valid_mask])
        return pred_delta.sum() * 0.0

    def _compute_stats(
        self,
        pred_delta: torch.Tensor,
        true_delta: torch.Tensor,
        pred_availability_logits: torch.Tensor,
        true_availability: torch.Tensor,
    ) -> dict[str, float]:
        scaled_pred = self._inverse_transform_delta_tensor(pred_delta.detach()).cpu().numpy()
        scaled_true = self._inverse_transform_delta_tensor(true_delta.detach()).cpu().numpy()
        valid_mask = (true_availability.detach().cpu().numpy() > 0.5)

        if valid_mask.any():
            pred_flat = scaled_pred[valid_mask].reshape(-1, 1)
            true_flat = scaled_true[valid_mask].reshape(-1, 1)
            mae, mse, rmse = metric(pred_flat, true_flat)
        else:
            mae, mse, rmse = 0.0, 0.0, 0.0

        availability_probs = torch.sigmoid(pred_availability_logits.detach())
        availability_true = (true_availability > 0.5).float()
        availability_accuracy = float(
            ((availability_probs > 0.5).float() == availability_true).float().mean().item()
        )

        return {
            "delta_mae": float(mae),
            "delta_mse": float(mse),
            "delta_rmse": float(rmse),
            "availability_accuracy": availability_accuracy,
        }

    def _build_decoder_input(self, batch_y: torch.Tensor) -> torch.Tensor:
        decoder_zeros = torch.zeros(
            (batch_y.shape[0], self.model_config.pred_len, batch_y.shape[-1]),
            device=batch_y.device,
            dtype=torch.float32,
        )
        if self.model_config.label_len > 0:
            return torch.cat([batch_y[:, : self.model_config.label_len, :], decoder_zeros], dim=1).float()
        return decoder_zeros

    def _shared_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> dict[str, torch.Tensor | float]:
        batch_x, batch_y = batch

        decoder_input = self._build_decoder_input(batch_y=batch_y)
        outputs = self.forecaster(batch_x, None, decoder_input, None)
        predictions = outputs[:, -self.model_config.pred_len :, :]
        targets = batch_y[:, -self.model_config.pred_len :, :]

        pred_delta = self._select_delta_targets(predictions)
        true_delta = self._select_delta_targets(targets)
        pred_availability_logits = self._select_availability_targets(predictions)
        true_availability = self._select_availability_targets(targets)

        delta_loss = self._compute_delta_loss(
            pred_delta=pred_delta,
            true_delta=true_delta,
            true_availability=true_availability,
        )
        availability_loss = F.binary_cross_entropy_with_logits(
            pred_availability_logits,
            true_availability,
        )
        loss = delta_loss + availability_loss

        stats = self._compute_stats(
            pred_delta=pred_delta,
            true_delta=true_delta,
            pred_availability_logits=pred_availability_logits,
            true_availability=true_availability,
        )
        stats["delta_loss"] = delta_loss
        stats["availability_loss"] = availability_loss
        stats["loss"] = loss
        return stats

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor | float]:
        return self._shared_step(batch=batch)

    def on_train_batch_end(
        self,
        outputs: dict[str, torch.Tensor | float],
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor | float]:
        self._log_stats(section="train", outs=outputs)
        return outputs

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor | float]:
        return self._shared_step(batch=batch)

    def on_validation_batch_end(
        self,
        outputs: dict[str, torch.Tensor | float],
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor | float]:
        self._log_stats(section="val", outs=outputs)
        return outputs

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor | float]:
        return self._shared_step(batch=batch)

    def on_test_batch_end(
        self,
        outputs: dict[str, torch.Tensor | float],
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        self._log_stats(section="test", outs=outputs)
        return {"loss": outputs["loss"].mean()}

    def predict(
        self,
        x: torch.Tensor,
    ) -> dict[str, np.ndarray]:
        x = x.to(self.device).float()

        decoder_input = torch.zeros(
            (x.shape[0], self.model_config.label_len + self.model_config.pred_len, self.model_config.dec_in),
            device=self.device,
            dtype=torch.float32,
        )

        with torch.no_grad():
            outputs = self.forecaster(x, None, decoder_input, None)

        predictions = outputs[:, -self.model_config.pred_len :, :]
        pred_delta = self._select_delta_targets(predictions)
        pred_availability_logits = self._select_availability_targets(predictions)

        return {
            "time_differences": self._inverse_transform_delta_tensor(pred_delta).detach().cpu().numpy(),
            "availability": torch.sigmoid(pred_availability_logits).detach().cpu().numpy(),
        }
