from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class GRUDEncoder(nn.Module):
    """Compact GRU-D encoder for irregular monitoring histories."""

    def __init__(self, n_features: int, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size

        self.gamma_x = nn.Linear(n_features, n_features)
        self.gamma_h = nn.Linear(n_features, hidden_size)
        self.gru_cell = nn.GRUCell(input_size=n_features * 2, hidden_size=hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        m: Tensor,
        d: Tensor,
        step_mask: Tensor,
        x_mean: Tensor,
    ) -> Tensor:
        batch_size, time_steps, _ = x.shape
        device = x.device

        h = torch.zeros(batch_size, self.hidden_size, device=device)
        x_last = x_mean.unsqueeze(0).expand(batch_size, -1).clone()
        x_mean_expanded = x_mean.unsqueeze(0).expand(batch_size, -1)

        for time_index in range(time_steps):
            x_t = x[:, time_index, :]
            m_t = m[:, time_index, :]
            d_t = d[:, time_index, :]
            real_t = step_mask[:, time_index].unsqueeze(-1)

            gamma_x = torch.exp(-torch.relu(self.gamma_x(d_t)))
            gamma_h = torch.exp(-torch.relu(self.gamma_h(d_t)))

            x_imputed = m_t * x_t + (1.0 - m_t) * (
                gamma_x * x_last + (1.0 - gamma_x) * x_mean_expanded
            )
            h_decay = gamma_h * h

            h_candidate = self.gru_cell(torch.cat([x_imputed, m_t], dim=-1), h_decay)
            h = real_t * h_candidate + (1.0 - real_t) * h
            x_last = real_t * (m_t * x_t + (1.0 - m_t) * x_last) + (1.0 - real_t) * x_last

        return self.dropout(h)


class MedicationStateEncoder(nn.Module):
    def __init__(self, n_meds: int, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_meds, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

    def forward(self, med_features: Tensor) -> Tensor:
        return self.net(med_features)


class MultitaskDeliveryPointProcessModel(nn.Module):
    def __init__(
        self,
        *,
        n_vitals: int,
        n_meds: int,
        time_bins: int,
        vital_hidden_size: int = 64,
        med_hidden_size: int = 64,
        fusion_hidden_size: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.grud = GRUDEncoder(
            n_features=n_vitals,
            hidden_size=vital_hidden_size,
            dropout=dropout,
        )
        self.med_encoder = MedicationStateEncoder(
            n_meds=n_meds,
            hidden_size=med_hidden_size,
            dropout=dropout,
        )
        self.fusion = nn.Sequential(
            nn.Linear(vital_hidden_size + med_hidden_size, fusion_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_size, fusion_hidden_size),
            nn.ReLU(),
        )
        self.hazard_head = nn.Linear(fusion_hidden_size, time_bins)
        self.med_head = nn.Linear(fusion_hidden_size, n_meds)

    def forward(
        self,
        *,
        x: Tensor,
        m: Tensor,
        d: Tensor,
        step_mask: Tensor,
        meds: Tensor,
        x_mean: Tensor,
    ) -> Dict[str, Optional[Tensor]]:
        vital_embedding = self.grud(x=x, m=m, d=d, step_mask=step_mask, x_mean=x_mean)
        med_embedding = self.med_encoder(meds)
        fused = self.fusion(torch.cat([vital_embedding, med_embedding], dim=-1))
        return {
            "hazard_logits": self.hazard_head(fused),
            "med_logits": self.med_head(fused),
            "embedding": fused,
        }


class DiscreteTimeHazardLoss(nn.Module):
    """Negative log-likelihood for a discrete-time hazard model."""

    def forward(self, hazard_logits: Tensor, duration_idx: Tensor, event: Tensor) -> Tensor:
        log_h = F.logsigmoid(hazard_logits)
        log_1mh = F.logsigmoid(-hazard_logits)
        cumulative_log_survival = torch.cumsum(log_1mh, dim=1)

        batch_index = torch.arange(hazard_logits.size(0), device=hazard_logits.device)
        clamped_duration_idx = duration_idx.long().clamp(min=0, max=hazard_logits.size(1) - 1)

        previous_survival = torch.zeros_like(event, dtype=hazard_logits.dtype)
        valid_previous_mask = clamped_duration_idx > 0
        previous_survival[valid_previous_mask] = cumulative_log_survival[
            batch_index[valid_previous_mask],
            clamped_duration_idx[valid_previous_mask] - 1,
        ]

        event = event.to(hazard_logits.dtype)
        event_log_likelihood = previous_survival + log_h[batch_index, clamped_duration_idx]
        censor_log_likelihood = cumulative_log_survival[batch_index, clamped_duration_idx]
        log_likelihood = event * event_log_likelihood + (1.0 - event) * censor_log_likelihood
        return -log_likelihood.mean()


class EventConditionedMultilabelLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(
        self,
        logits: Optional[Tensor],
        targets: Tensor,
        event: Tensor,
        target_available: Optional[Tensor] = None,
    ) -> Tensor:
        if logits is None or logits.numel() == 0 or targets.numel() == 0:
            reference_tensor = targets if targets.numel() > 0 else event
            return reference_tensor.new_tensor(0.0)

        event_mask = event > 0.5
        if target_available is not None:
            event_mask = event_mask & (target_available > 0.5)
        if not torch.any(event_mask):
            return logits.new_tensor(0.0)
        return self.loss_fn(logits[event_mask], targets[event_mask])


@torch.no_grad()
def hazard_to_event_distribution(hazard_logits: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    hazard = torch.sigmoid(hazard_logits)
    survival = torch.cumprod(1.0 - hazard, dim=-1)
    previous_survival = torch.ones_like(survival)
    previous_survival[..., 1:] = survival[..., :-1]
    event_mass = previous_survival * hazard
    return hazard, survival, event_mass


@torch.no_grad()
def hazard_to_survival(hazard_logits: Tensor) -> Tuple[Tensor, Tensor]:
    _, survival, _ = hazard_to_event_distribution(hazard_logits)
    cumulative_event = 1.0 - survival
    return survival, cumulative_event
