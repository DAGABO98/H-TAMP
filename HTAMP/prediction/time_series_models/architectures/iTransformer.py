from __future__ import annotations

"""iTransformer.py

Code based on the implementation provided in
https://github.com/thuml/Time-Series-Library/blob/main/models/iTransformer.py
"""

from typing import Any, Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from HTAMP.prediction.time_series_models.layers.transformer_encdec import (
    Encoder,
    EncoderLayer,
)
from HTAMP.prediction.time_series_models.layers.self_attention_family import (
    AttentionLayer,
    FullAttention,
)
from HTAMP.prediction.time_series_models.layers.embed import DataEmbeddingInverted


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs: Any) -> None:
        super().__init__()
        self.task_name: str = configs.task_name
        self.seq_len: int = configs.seq_len
        self.pred_len: int = configs.pred_len
        self.output_attention: bool = configs.output_attention

        # Embedding
        self.enc_embedding = DataEmbeddingInverted(
            configs.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )

        # Decoder / head
        if self.task_name in {"long_term_forecast", "short_term_forecast"}:
            self.projection: nn.Module = nn.Linear(
                configs.d_model,
                configs.pred_len,
                bias=True,
            )
        elif self.task_name == "imputation":
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        elif self.task_name == "anomaly_detection":
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        elif self.task_name == "classification":
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.enc_in,
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
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        x_enc = x_enc / stdev

        _, _, n_vars = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :n_vars]

        # De-normalization from Non-stationary Transformer
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    def imputation(
        self,
        x_enc: Tensor,
        x_mark_enc: Optional[Tensor],
        x_dec: Tensor,
        x_mark_dec: Optional[Tensor],
        mask: Optional[Tensor],
    ) -> Tensor:
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(dim=1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        x_enc = x_enc / stdev

        _, seq_len, n_vars = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :n_vars]

        # De-normalization from Non-stationary Transformer
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, seq_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, seq_len, 1)
        return dec_out

    def anomaly_detection(self, x_enc: Tensor) -> Tensor:
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(dim=1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        x_enc = x_enc / stdev

        _, seq_len, n_vars = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :n_vars]

        # De-normalization from Non-stationary Transformer
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, seq_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, seq_len, 1)
        return dec_out

    def classification(
        self,
        x_enc: Tensor,
        x_mark_enc: Optional[Tensor],
    ) -> Tensor:
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output

    def forward(
        self,
        x_enc: Tensor,
        x_mark_enc: Optional[Tensor],
        x_dec: Tensor,
        x_mark_dec: Optional[Tensor],
        mask: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        if self.task_name in {"long_term_forecast", "short_term_forecast"}:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len :, :]  # [B, L, D]

        if self.task_name == "imputation":
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]

        if self.task_name == "anomaly_detection":
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]

        if self.task_name == "classification":
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]

        return None