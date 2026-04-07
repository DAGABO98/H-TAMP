from __future__ import annotations

from typing import Any

import lightning as L
import numpy as np
import torch

from HTAMP.prediction.configs.delivery_request_config import DeliveryPointProcessModelConfig
from HTAMP.prediction.point_process_models.delivery_point_process import (
    DiscreteTimeHazardLoss,
    EventConditionedMultilabelLoss,
    MultitaskDeliveryPointProcessModel,
    hazard_to_event_distribution,
    hazard_to_survival,
)


def _coerce_model_config(
    model_config: DeliveryPointProcessModelConfig | dict[str, Any],
) -> DeliveryPointProcessModelConfig:
    if isinstance(model_config, DeliveryPointProcessModelConfig):
        return model_config
    return DeliveryPointProcessModelConfig.from_dict(model_config)


class DeliveryRequestModule(L.LightningModule):
    def __init__(
        self,
        model_config: DeliveryPointProcessModelConfig | dict[str, Any],
        x_mean: list[float] | np.ndarray,
        time_bins_hours: list[float] | np.ndarray,
        n_vitals: int,
        n_meds: int,
    ) -> None:
        super().__init__()
        self.model_config = _coerce_model_config(model_config=model_config)
        self.point_process_model = MultitaskDeliveryPointProcessModel(
            n_vitals=int(n_vitals),
            n_meds=int(n_meds),
            time_bins=len(time_bins_hours),
            vital_hidden_size=self.model_config.vital_hidden_size,
            med_hidden_size=self.model_config.med_hidden_size,
            fusion_hidden_size=self.model_config.fusion_hidden_size,
            dropout=self.model_config.dropout,
        )
        self.hazard_loss_fn = DiscreteTimeHazardLoss()
        self.multilabel_loss_fn = EventConditionedMultilabelLoss()

        x_mean_tensor = torch.as_tensor(x_mean, dtype=torch.float32)
        time_bins_tensor = torch.as_tensor(time_bins_hours, dtype=torch.float32)
        self.register_buffer("x_mean", x_mean_tensor)
        self.register_buffer("time_bins_hours", time_bins_tensor)

        self.save_hyperparameters(
            {
                "model_config": self.model_config.to_dict(),
                "x_mean": x_mean_tensor.detach().cpu().tolist(),
                "time_bins_hours": time_bins_tensor.detach().cpu().tolist(),
                "n_vitals": int(n_vitals),
                "n_meds": int(n_meds),
            }
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.model_config.learning_rate,
            weight_decay=self.model_config.weight_decay,
        )

    def _move_batch_to_device(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    def _log_stats(self, section: str, outputs: dict[str, torch.Tensor | float]) -> None:
        for key, value in outputs.items():
            if isinstance(value, np.ndarray):
                value = float(value.mean())
            if isinstance(value, torch.Tensor):
                value = value.mean()
            self.log(
                f"{section}_{key}",
                value,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )

    def _expected_time_hours(self, hazard_logits: torch.Tensor) -> torch.Tensor:
        _, _, event_mass = hazard_to_event_distribution(hazard_logits)
        return (event_mass * self.time_bins_hours.view(1, -1)).sum(dim=-1)

    def _masked_exact_match(
        self,
        logits: torch.Tensor | None,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> float:
        if logits is None or logits.numel() == 0 or targets.numel() == 0 or not torch.any(mask):
            return 0.0
        predictions = (torch.sigmoid(logits[mask]) >= 0.5).float()
        exact_match = (predictions == targets[mask]).all(dim=-1).float().mean()
        return float(exact_match.item())

    def _shared_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | float]:
        outputs = self.point_process_model(
            x=batch["x"],
            m=batch["m"],
            d=batch["d"],
            step_mask=batch["step_mask"],
            meds=batch["meds"],
            x_mean=self.x_mean,
        )
        hazard_logits = outputs["hazard_logits"]
        med_logits = outputs["med_logits"]

        hazard_loss = self.hazard_loss_fn(
            hazard_logits=hazard_logits,
            duration_idx=batch["duration_idx"],
            event=batch["event"],
        )
        med_loss = self.multilabel_loss_fn(
            logits=med_logits,
            targets=batch["next_med_targets"],
            event=batch["event"],
            target_available=batch["med_target_available"],
        )

        weighted_hazard_loss = self.model_config.hazard_loss_weight * hazard_loss
        weighted_med_loss = self.model_config.med_loss_weight * med_loss
        loss = weighted_hazard_loss + weighted_med_loss

        _, cumulative_event = hazard_to_survival(hazard_logits)
        event_probability = cumulative_event[:, -1]
        event_prediction = (event_probability >= 0.5).float()
        event_accuracy = (event_prediction == batch["event"]).float().mean()
        event_brier = torch.mean((event_probability - batch["event"]) ** 2)

        event_mask = batch["event"] > 0.5
        duration_mae = loss.new_tensor(0.0)
        if torch.any(event_mask):
            duration_mae = torch.mean(
                torch.abs(
                    self._expected_time_hours(hazard_logits)[event_mask]
                    - batch["duration_hours"][event_mask]
                )
            )

        med_exact_match = self._masked_exact_match(
            logits=med_logits,
            targets=batch["next_med_targets"],
            mask=event_mask & (batch["med_target_available"] > 0.5),
        )

        return {
            "loss": loss,
            "hazard_loss": hazard_loss,
            "med_loss": med_loss,
            "weighted_hazard_loss": weighted_hazard_loss,
            "weighted_med_loss": weighted_med_loss,
            "event_accuracy": event_accuracy,
            "event_brier": event_brier,
            "duration_mae_hours": duration_mae,
            "med_exact_match": med_exact_match,
        }

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor | float]:
        return self._shared_step(batch=batch)

    def on_train_batch_end(
        self,
        outputs: dict[str, torch.Tensor | float],
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor | float]:
        self._log_stats(section="train", outputs=outputs)
        return outputs

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor | float]:
        return self._shared_step(batch=batch)

    def on_validation_batch_end(
        self,
        outputs: dict[str, torch.Tensor | float],
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor | float]:
        self._log_stats(section="val", outputs=outputs)
        return outputs

    def test_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor | float]:
        return self._shared_step(batch=batch)

    def on_test_batch_end(
        self,
        outputs: dict[str, torch.Tensor | float],
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        self._log_stats(section="test", outputs=outputs)
        return {"loss": outputs["loss"].mean()}

    def predict_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
        batch = self._move_batch_to_device(batch)
        with torch.no_grad():
            outputs = self.point_process_model(
                x=batch["x"],
                m=batch["m"],
                d=batch["d"],
                step_mask=batch["step_mask"],
                meds=batch["meds"],
                x_mean=self.x_mean,
            )
            hazard_logits = outputs["hazard_logits"]
            med_logits = outputs["med_logits"]
            _, survival, event_mass = hazard_to_event_distribution(hazard_logits)
            cumulative_event = 1.0 - survival
            expected_time_hours = (event_mass * self.time_bins_hours.view(1, -1)).sum(dim=-1)

        return {
            "event_probability": cumulative_event[:, -1].detach().cpu().numpy(),
            "survival": survival.detach().cpu().numpy(),
            "cumulative_event": cumulative_event.detach().cpu().numpy(),
            "expected_time_hours": expected_time_hours.detach().cpu().numpy(),
            "med_probs": torch.sigmoid(med_logits).detach().cpu().numpy(),
        }
