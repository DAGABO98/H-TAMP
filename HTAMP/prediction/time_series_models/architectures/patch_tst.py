from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor, nn

from HTAMP.prediction.time_series_models.layers.transformer_encdec import (
    Encoder,
    EncoderLayer,
)
from HTAMP.prediction.time_series_models.layers.self_attention_family import (
    AttentionLayer,
    FullAttention,
)
from HTAMP.prediction.time_series_models.layers.embed import PatchEmbedding


class Transpose(nn.Module):
    def __init__(self, *dims: int, contiguous: bool = False) -> None:
        super().__init__()
        self.dims: tuple[int, ...] = dims
        self.contiguous = contiguous

    def forward(self, x: Tensor) -> Tensor:
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(
        self,
        n_vars: int,
        nf: int,
        target_window: int,
        head_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: [bs, nvars, d_model, patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(
        self,
        configs: Any,
        patch_len: int = 16,
        stride: int = 8,
    ) -> None:
        """
        patch_len: patch length for patch embedding
        stride: stride for patch embedding
        """
        super().__init__()
        self.task_name: str = configs.task_name
        self.seq_len: int = configs.seq_len
        self.pred_len: int = configs.pred_len
        padding = stride

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model,
            patch_len,
            stride,
            padding,
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
            norm_layer=nn.Sequential(
                Transpose(1, 2),
                nn.BatchNorm1d(configs.d_model),
                Transpose(1, 2),
            ),
        )

        # Prediction head
        self.head_nf: int = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)

        if self.task_name in {"long_term_forecast", "short_term_forecast"}:
            self.head: nn.Module = FlattenHead(
                configs.enc_in,
                self.head_nf,
                configs.pred_len,
                head_dropout=configs.dropout,
            )
        elif self.task_name in {"imputation", "anomaly_detection"}:
            self.head = FlattenHead(
                configs.enc_in,
                self.head_nf,
                configs.seq_len,
                head_dropout=configs.dropout,
            )
        elif self.task_name == "classification":
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection: nn.Module = nn.Linear(
                self.head_nf * configs.enc_in,
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

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # [bs * nvars, patch_num, d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        enc_out, attns = self.encoder(enc_out)

        # [bs, nvars, patch_num, d_model]
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # [bs, nvars, d_model, patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)

        # De-normalization
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
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

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        enc_out, attns = self.encoder(enc_out)

        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)

        # De-normalization
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1)
        return dec_out

    def anomaly_detection(self, x_enc: Tensor) -> Tensor:
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(dim=1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        enc_out, attns = self.encoder(enc_out)

        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)

        # De-normalization
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1)
        return dec_out

    def classification(
        self,
        x_enc: Tensor,
        x_mark_enc: Optional[Tensor],
    ) -> Tensor:
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(dim=1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        enc_out, attns = self.encoder(enc_out)

        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        output = self.flatten(enc_out)
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
            return dec_out[:, -self.pred_len :, :]

        if self.task_name == "imputation":
            if mask is None:
                raise ValueError("mask must not be None for imputation")
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out

        if self.task_name == "anomaly_detection":
            dec_out = self.anomaly_detection(x_enc)
            return dec_out

        if self.task_name == "classification":
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out

        return None