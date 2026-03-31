from __future__ import annotations

"""TimesNet.py

Code based on the implementation provided in
https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py
"""

from typing import Any, Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

from HTAMP.prediction.time_series_models.layers.Embed import DataEmbedding
from HTAMP.prediction.time_series_models.layers.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x: Tensor, k: int = 2) -> Tuple[Tensor, Tensor]:
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)

    # find period by amplitudes
    frequency_list = torch.abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)

    period = x.shape[1] // top_list
    return period, torch.abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs: Any) -> None:
        super().__init__()
        self.seq_len: int = configs.seq_len
        self.pred_len: int = configs.pred_len
        self.k: int = configs.top_k

        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(
                configs.d_model,
                configs.d_ff,
                num_kernels=configs.num_kernels,
            ),
            nn.GELU(),
            Inception_Block_V1(
                configs.d_ff,
                configs.d_model,
                num_kernels=configs.num_kernels,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, t, n = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res: list[Tensor] = []
        total_length = self.seq_len + self.pred_len

        for i in range(self.k):
            period = int(period_list[i].item())

            # padding
            if total_length % period != 0:
                length = ((total_length // period) + 1) * period
                padding = torch.zeros(
                    (x.shape[0], length - total_length, x.shape[2]),
                    device=x.device,
                    dtype=x.dtype,
                )
                out = torch.cat([x, padding], dim=1)
            else:
                length = total_length
                out = x

            # reshape
            out = out.reshape(b, length // period, period, n).permute(0, 3, 1, 2).contiguous()

            # 2D conv: from 1D variation to 2D variation
            out = self.conv(out)

            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(b, -1, n)
            res.append(out[:, :total_length, :])

        res_stacked = torch.stack(res, dim=-1)

        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, t, n, 1)
        res_out = torch.sum(res_stacked * period_weight, dim=-1)

        # residual connection
        res_out = res_out + x
        return res_out


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs: Any) -> None:
        super().__init__()
        self.configs = configs
        self.task_name: str = configs.task_name
        self.seq_len: int = configs.seq_len
        self.label_len: int = configs.label_len
        self.pred_len: int = configs.pred_len

        self.model = nn.ModuleList([TimesBlock(configs) for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )
        self.layer: int = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)

        if self.task_name in {"long_term_forecast", "short_term_forecast"}:
            self.predict_linear: nn.Module = nn.Linear(
                self.seq_len,
                self.pred_len + self.seq_len,
            )
            self.projection: nn.Module = nn.Linear(
                configs.d_model,
                configs.c_out,
                bias=True,
            )
        elif self.task_name in {"imputation", "anomaly_detection"}:
            self.projection = nn.Linear(
                configs.d_model,
                configs.c_out,
                bias=True,
            )
        elif self.task_name == "classification":
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len,
                configs.num_class,
            )
        else:
            raise ValueError(f"Unsupported task_name: {self.task_name}")

    def forecast(
        self,
        x_enc: Tensor,
        x_mark_enc: Optional[Tensor],
        x_dec: Tensor,
        x_mark_dec: Optional[Tensor],
    ) -> Tensor:
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(dim=1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B, T, C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)

        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # project back
        dec_out = self.projection(enc_out)

        # De-normalization from Non-stationary Transformer
        total_length = self.pred_len + self.seq_len
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, total_length, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, total_length, 1)
        return dec_out

    def imputation(
        self,
        x_enc: Tensor,
        x_mark_enc: Optional[Tensor],
        x_dec: Tensor,
        x_mark_dec: Optional[Tensor],
        mask: Tensor,
    ) -> Tensor:
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)

        stdev = torch.sqrt(
            torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5
        )
        stdev = stdev.unsqueeze(1).detach()
        x_enc = x_enc / stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B, T, C]

        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # project back
        dec_out = self.projection(enc_out)

        # De-normalization from Non-stationary Transformer
        total_length = self.pred_len + self.seq_len
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, total_length, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, total_length, 1)
        return dec_out

    def anomaly_detection(self, x_enc: Tensor) -> Tensor:
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(dim=1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B, T, C]

        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # project back
        dec_out = self.projection(enc_out)

        # De-normalization from Non-stationary Transformer
        total_length = self.pred_len + self.seq_len
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, total_length, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, total_length, 1)
        return dec_out

    def classification(self, x_enc: Tensor, x_mark_enc: Tensor) -> Tensor:
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B, T, C]

        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Output
        output = self.act(enc_out)
        output = self.dropout(output)

        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)

        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(
        self,
        x_enc: Tensor,
        x_mark_enc: Optional[Tensor],
        x_dec: Tensor,
        x_mark_dec: Optional[Tensor],
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        if self.task_name in {"long_term_forecast", "short_term_forecast"}:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len :, :]  # [B, L, D]

        if self.task_name == "imputation":
            if mask is None:
                raise ValueError("mask must not be None for imputation")
            return self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)

        if self.task_name == "anomaly_detection":
            return self.anomaly_detection(x_enc)

        if self.task_name == "classification":
            if x_mark_enc is None:
                raise ValueError("x_mark_enc must not be None for classification")
            return self.classification(x_enc, x_mark_enc)

        raise ValueError(f"Unsupported task_name: {self.task_name}")