from __future__ import annotations

from typing import Any

import numpy as np
import torch
import lightning as L

from HTAMP.prediction.request_number_config import TimeseriesModelConfig
from HTAMP.prediction.time_series_models.architectures.i_transformer import Model as ITransformerModel
from HTAMP.prediction.time_series_models.architectures.patch_tst import Model as PatchTSTModel
from HTAMP.prediction.time_series_models.architectures.time_mixer import Model as TimeMixerModel
from HTAMP.prediction.time_series_models.architectures.times_net import Model as TimesNetModel
from HTAMP.prediction.time_series_models.utils.metrics import metric


def _coerce_model_config(model_config: TimeseriesModelConfig | dict[str, Any]) -> TimeseriesModelConfig:
    if isinstance(model_config, TimeseriesModelConfig):
        return model_config
    return TimeseriesModelConfig(**model_config)


class RequestsNumberModule(L.LightningModule):
    def __init__(
        self,
        model_config: TimeseriesModelConfig | dict[str, Any],
        target_scaler_mean: list[float] | np.ndarray,
        target_scaler_scale: list[float] | np.ndarray,
        target_channel_indices: list[int],
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
        self.criterion = (
            torch.nn.MSELoss()
            if self.model_config.loss == "MSE"
            else torch.nn.L1Loss()
        )

        scaler_mean_tensor = torch.as_tensor(target_scaler_mean, dtype=torch.float32)
        scaler_scale_tensor = torch.as_tensor(target_scaler_scale, dtype=torch.float32)
        target_channel_indices_tensor = torch.as_tensor(target_channel_indices, dtype=torch.long)
        self.register_buffer("target_scaler_mean", scaler_mean_tensor)
        self.register_buffer("target_scaler_scale", scaler_scale_tensor)
        self.register_buffer("target_channel_indices", target_channel_indices_tensor)

        self.save_hyperparameters(
            {
                "model_config": self.model_config.to_dict(),
                "target_scaler_mean": scaler_mean_tensor.detach().cpu().tolist(),
                "target_scaler_scale": scaler_scale_tensor.detach().cpu().tolist(),
                "target_channel_indices": target_channel_indices_tensor.detach().cpu().tolist(),
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

    def _inverse_transform_tensor(self, data: torch.Tensor) -> torch.Tensor:
        scale = self.target_scaler_scale.view(1, 1, -1)
        mean = self.target_scaler_mean.view(1, 1, -1)
        return data * scale + mean

    def _select_target_channels(self, data: torch.Tensor) -> torch.Tensor:
        return torch.index_select(data, dim=-1, index=self.target_channel_indices)

    def _compute_stats(self, pred: torch.Tensor, true: torch.Tensor) -> dict[str, float]:
        scaled_pred = self._inverse_transform_tensor(pred.detach()).cpu().numpy()
        scaled_true = self._inverse_transform_tensor(true.detach()).cpu().numpy()

        mae, mse, rmse = metric(scaled_pred, scaled_true)
        return {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
        }

    def _shared_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> dict[str, torch.Tensor | float]:
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch

        decoder_zeros = torch.zeros_like(batch_y[:, -self.model_config.pred_len :, :]).float()
        decoder_input = torch.cat(
            [batch_y[:, : self.model_config.label_len, :], decoder_zeros],
            dim=1,
        ).float()

        outputs = self.forecaster(batch_x, batch_x_mark, decoder_input, batch_y_mark)
        predictions = outputs[:, -self.model_config.pred_len :, :]
        targets = batch_y[:, -self.model_config.pred_len :, :]

        predictions = self._select_target_channels(predictions)
        targets = self._select_target_channels(targets)

        loss = self.criterion(predictions, targets)
        stats = self._compute_stats(pred=predictions, true=targets)
        stats["loss"] = loss

        return stats

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor | float]:
        return self._shared_step(batch=batch)

    def on_train_batch_end(
        self,
        outputs: dict[str, torch.Tensor | float],
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor | float]:
        self._log_stats(section="train", outs=outputs)
        return outputs

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor | float]:
        return self._shared_step(batch=batch)

    def on_validation_batch_end(
        self,
        outputs: dict[str, torch.Tensor | float],
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor | float]:
        self._log_stats(section="val", outs=outputs)
        return outputs

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor | float]:
        return self._shared_step(batch=batch)

    def on_test_batch_end(
        self,
        outputs: dict[str, torch.Tensor | float],
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        self._log_stats(section="test", outs=outputs)
        return {"loss": outputs["loss"].mean()}

    def predict(
        self,
        x: torch.Tensor,
        x_mark: torch.Tensor,
        y_mark: torch.Tensor,
    ) -> np.ndarray:
        x = x.to(self.device).float()
        x_mark = x_mark.to(self.device).float()
        y_mark = y_mark.to(self.device).float()

        decoder_zeros = torch.zeros(
            (x.shape[0], self.model_config.pred_len, x.shape[-1]),
            device=self.device,
            dtype=torch.float32,
        )

        if self.model_config.label_len > 0:
            decoder_input = torch.cat(
                [x[:, -self.model_config.label_len :, :], decoder_zeros],
                dim=1,
            ).float()
        else:
            decoder_input = decoder_zeros

        with torch.no_grad():
            outputs = self.forecaster(x, x_mark, decoder_input, y_mark)

        predictions = outputs[:, -self.model_config.pred_len :, :]
        predictions = self._select_target_channels(predictions)
        scaled_predictions = self._inverse_transform_tensor(predictions)

        return scaled_predictions.detach().cpu().numpy()