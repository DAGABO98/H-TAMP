from __future__ import annotations

"""TimeMixer.py

Code based on the implementation provided in
https://github.com/thuml/Time-Series-Library/blob/main/models/TimeMixer.py
"""

from typing import Any, Optional, Sequence, Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from HTAMP.prediction.time_series_models.layers.autoformer_encdec import SeriesDecomp
from HTAMP.prediction.time_series_models.layers.embed import DataEmbeddingWoPos
from HTAMP.prediction.time_series_models.layers.standard_norm import Normalize


class DFTSeriesDecomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, top_k: int = 5) -> None:
        super().__init__()
        self.top_k = top_k

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        xf = torch.fft.rfft(x, dim=1)
        freq = torch.abs(xf)
        freq[:, 0, :] = 0
        top_k_freq, _ = torch.topk(freq, self.top_k, dim=1)
        threshold = top_k_freq.min(dim=1, keepdim=True).values
        xf = xf.masked_fill(freq <= threshold, 0)
        x_season = torch.fft.irfft(xf, n=x.size(1), dim=1)
        x_trend = x - x_season
        return x_season, x_trend


class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(self, configs: Any) -> None:
        super().__init__()

        self.down_sampling_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),
                )
                for i in range(configs.down_sampling_layers)
            ]
        )

    def forward(self, season_list: Sequence[Tensor]) -> list[Tensor]:
        # mixing high -> low
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list: list[Tensor] = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(self, configs: Any) -> None:
        super().__init__()

        self.up_sampling_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                    nn.GELU(),
                    nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                )
                for i in reversed(range(configs.down_sampling_layers))
            ]
        )

    def forward(self, trend_list: Sequence[Tensor]) -> list[Tensor]:
        # mixing low -> high
        trend_list_reverse = list(trend_list).copy()
        trend_list_reverse.reverse()

        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list: list[Tensor] = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    def __init__(self, configs: Any) -> None:
        super().__init__()
        self.seq_len: int = configs.seq_len
        self.pred_len: int = configs.pred_len
        self.down_sampling_window: int = configs.down_sampling_window

        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.channel_independence: bool = configs.channel_independence

        self.decompsition: nn.Module
        if configs.decomp_method == "moving_avg":
            self.decompsition = SeriesDecomp(configs.moving_avg)
        elif configs.decomp_method == "dft_decomp":
            self.decompsition = DFTSeriesDecomp(configs.top_k)
        else:
            raise ValueError("decompsition is error")

        if not configs.channel_independence:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
                nn.GELU(),
                nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
            )

        # Mixing season
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)

        # Mixing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
            nn.GELU(),
            nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
        )

    def forward(self, x_list: Sequence[Tensor]) -> list[Tensor]:
        length_list: list[int] = []
        for x in x_list:
            _, t, _ = x.size()
            length_list.append(t)

        # Decompose to obtain the season and trend
        season_list: list[Tensor] = []
        trend_list: list[Tensor] = []
        for x in x_list:
            season, trend = self.decompsition(x)
            if not self.channel_independence:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        # bottom-up season mixing
        out_season_list = self.mixing_multi_scale_season(season_list)
        # top-down trend mixing
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list: list[Tensor] = []
        for ori, out_season, out_trend, length in zip(
            x_list, out_season_list, out_trend_list, length_list
        ):
            out = out_season + out_trend
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
        return out_list


class Model(nn.Module):
    def __init__(self, configs: Any) -> None:
        super().__init__()
        self.configs = configs
        self.task_name: str = configs.task_name
        self.seq_len: int = configs.seq_len
        self.label_len: int = configs.label_len
        self.pred_len: int = configs.pred_len
        self.down_sampling_window: int = configs.down_sampling_window
        self.channel_independence: bool = configs.channel_independence

        self.pdm_blocks = nn.ModuleList(
            [PastDecomposableMixing(configs) for _ in range(configs.e_layers)]
        )

        self.preprocess = SeriesDecomp(configs.moving_avg)
        self.enc_in: int = configs.enc_in

        if self.channel_independence:
            self.enc_embedding = DataEmbeddingWoPos(
                1,
                configs.d_model,
                configs.embed,
                configs.freq,
                configs.dropout,
            )
        else:
            self.enc_embedding = DataEmbeddingWoPos(
                configs.enc_in,
                configs.d_model,
                configs.embed,
                configs.freq,
                configs.dropout,
            )

        self.layer: int = configs.e_layers

        self.normalize_layers = nn.ModuleList(
            [
                Normalize(
                    self.configs.enc_in,
                    affine=True,
                    non_norm=True if configs.use_norm == 0 else False,
                )
                for _ in range(configs.down_sampling_layers + 1)
            ]
        )

        if self.task_name in {"long_term_forecast", "short_term_forecast"}:
            self.predict_layers = nn.ModuleList(
                [
                    nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.pred_len,
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )

            if self.channel_independence:
                self.projection_layer: nn.Module = nn.Linear(
                    configs.d_model, 1, bias=True
                )
            else:
                self.projection_layer = nn.Linear(
                    configs.d_model, configs.c_out, bias=True
                )

                self.out_res_layers = nn.ModuleList(
                    [
                        nn.Linear(
                            configs.seq_len // (configs.down_sampling_window ** i),
                            configs.seq_len // (configs.down_sampling_window ** i),
                        )
                        for i in range(configs.down_sampling_layers + 1)
                    ]
                )

                self.regression_layers = nn.ModuleList(
                    [
                        nn.Linear(
                            configs.seq_len // (configs.down_sampling_window ** i),
                            configs.pred_len,
                        )
                        for i in range(configs.down_sampling_layers + 1)
                    ]
                )

        if self.task_name in {"imputation", "anomaly_detection"}:
            if self.channel_independence:
                self.projection_layer = nn.Linear(configs.d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(
                    configs.d_model, configs.c_out, bias=True
                )

        if self.task_name == "classification":
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len,
                configs.num_class,
            )

    def out_projection(self, dec_out: Tensor, i: int, out_res: Tensor) -> Tensor:
        dec_out = self.projection_layer(dec_out)
        out_res = out_res.permute(0, 2, 1)
        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        dec_out = dec_out + out_res
        return dec_out

    def pre_enc(
        self, x_list: Sequence[Tensor]
    ) -> tuple[Sequence[Tensor], Optional[Sequence[Tensor]]]:
        if self.channel_independence:
            return x_list, None
        out1_list: list[Tensor] = []
        out2_list: list[Tensor] = []
        for x in x_list:
            x_1, x_2 = self.preprocess(x)
            out1_list.append(x_1)
            out2_list.append(x_2)
        return out1_list, out2_list

    def __multi_scale_process_inputs(
        self, x_enc: Tensor, x_mark_enc: Optional[Tensor]
    ) -> tuple[list[Tensor], Optional[list[Tensor]]]:
        if self.configs.down_sampling_method == "max":
            down_pool: nn.Module = nn.MaxPool1d(
                self.configs.down_sampling_window,
                return_indices=False,
            )
        elif self.configs.down_sampling_method == "avg":
            down_pool = nn.AvgPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == "conv":
            padding = 1 if torch.__version__ >= "1.5.0" else 2
            down_pool = nn.Conv1d(
                in_channels=self.configs.enc_in,
                out_channels=self.configs.enc_in,
                kernel_size=3,
                padding=padding,
                stride=self.configs.down_sampling_window,
                padding_mode="circular",
                bias=False,
            )
        else:
            return [x_enc], [x_mark_enc] if x_mark_enc is not None else None

        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_ori = x_mark_enc

        x_enc_sampling_list: list[Tensor] = [x_enc.permute(0, 2, 1)]
        x_mark_sampling_list: list[Tensor] = []
        if x_mark_enc is not None:
            x_mark_sampling_list.append(x_mark_enc)

        for _ in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc_ori is not None:
                x_mark_enc_ori = x_mark_enc_ori[
                    :, :: self.configs.down_sampling_window, :
                ]
                x_mark_sampling_list.append(x_mark_enc_ori)

        return (
            x_enc_sampling_list,
            x_mark_sampling_list if x_mark_enc is not None else None,
        )

    def forecast(
        self,
        x_enc: Tensor,
        x_mark_enc: Optional[Tensor],
        x_dec: Tensor,
        x_mark_dec: Optional[Tensor],
    ) -> Tensor:
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list: list[Tensor] = []
        x_mark_list: list[Tensor] = []

        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                b, t, n = x.size()
                x = self.normalize_layers[i](x, "norm")
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(b * n, t, 1)
                    x_mark = x_mark.repeat(n, 1, 1)
                x_list.append(x)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc):
                b, t, n = x.size()
                x = self.normalize_layers[i](x, "norm")
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(b * n, t, 1)
                x_list.append(x)

        # embedding
        enc_out_list: list[Tensor] = []
        x_list_pre, x_list_res = self.pre_enc(x_list)

        if x_mark_enc is not None:
            for x, x_mark in zip(x_list_pre, x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)
                enc_out_list.append(enc_out)
        else:
            for x in x_list_pre:
                enc_out = self.enc_embedding(x, None)
                enc_out_list.append(enc_out)

        # Past Decomposable Mixing as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        # Future Multipredictor Mixing as decoder for future
        dec_out_list = self.future_multi_mixing(
            b, enc_out_list, (x_list_pre, x_list_res)
        )

        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        dec_out = self.normalize_layers[0](dec_out, "denorm")
        return dec_out

    def future_multi_mixing(
        self,
        b: int,
        enc_out_list: Sequence[Tensor],
        x_list: tuple[Sequence[Tensor], Optional[Sequence[Tensor]]],
    ) -> list[Tensor]:
        dec_out_list: list[Tensor] = []

        if self.channel_independence:
            x_list_0 = x_list[0]
            for i, enc_out in zip(range(len(x_list_0)), enc_out_list):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1
                )
                dec_out = self.projection_layer(dec_out)
                dec_out = (
                    dec_out.reshape(b, self.configs.c_out, self.pred_len)
                    .permute(0, 2, 1)
                    .contiguous()
                )
                dec_out_list.append(dec_out)
        else:
            assert x_list[1] is not None
            for i, enc_out, out_res in zip(
                range(len(x_list[0])), enc_out_list, x_list[1]
            ):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1
                )
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)

        return dec_out_list

    def classification(self, x_enc: Tensor, x_mark_enc: Tensor) -> Tensor:
        x_enc_list, _ = self.__multi_scale_process_inputs(x_enc, None)
        x_list = x_enc_list

        # embedding
        enc_out_list: list[Tensor] = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)
            enc_out_list.append(enc_out)

        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        enc_out = enc_out_list[0]
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output

    def anomaly_detection(self, x_enc: Tensor) -> Tensor:
        b, _, _ = x_enc.size()
        x_enc_list, _ = self.__multi_scale_process_inputs(x_enc, None)

        x_list: list[Tensor] = []
        for i, x in zip(range(len(x_enc_list)), x_enc_list):
            b_local, t, n = x.size()
            x = self.normalize_layers[i](x, "norm")
            if self.channel_independence:
                x = x.permute(0, 2, 1).contiguous().reshape(b_local * n, t, 1)
            x_list.append(x)

        enc_out_list: list[Tensor] = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)
            enc_out_list.append(enc_out)

        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        dec_out = self.projection_layer(enc_out_list[0])
        dec_out = (
            dec_out.reshape(b, self.configs.c_out, -1)
            .permute(0, 2, 1)
            .contiguous()
        )

        dec_out = self.normalize_layers[0](dec_out, "denorm")
        return dec_out

    def imputation(
        self,
        x_enc: Tensor,
        x_mark_enc: Optional[Tensor],
        mask: Tensor,
    ) -> Tensor:
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(
            torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5
        )
        stdev = stdev.unsqueeze(1).detach()
        x_enc = x_enc / stdev

        b, _, _ = x_enc.size()
        x_enc_list, x_mark_enc_list = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list: list[Tensor] = []
        x_mark_list: list[Tensor] = []
        if x_mark_enc_list is not None:
            for x, x_mark in zip(x_enc_list, x_mark_enc_list):
                b_local, t, n = x.size()
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(b_local * n, t, 1)
                    x_mark = x_mark.repeat(n, 1, 1)
                x_list.append(x)
                x_mark_list.append(x_mark)
        else:
            for x in x_enc_list:
                b_local, t, n = x.size()
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(b_local * n, t, 1)
                x_list.append(x)

        enc_out_list: list[Tensor] = []
        if x_mark_enc_list is not None:
            for x, x_mark in zip(x_list, x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)
                enc_out_list.append(enc_out)
        else:
            for x in x_list:
                enc_out = self.enc_embedding(x, None)
                enc_out_list.append(enc_out)

        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        dec_out = self.projection_layer(enc_out_list[0])
        dec_out = (
            dec_out.reshape(b, self.configs.c_out, -1)
            .permute(0, 2, 1)
            .contiguous()
        )

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def forward(
        self,
        x_enc: Tensor,
        x_mark_enc: Optional[Tensor],
        x_dec: Tensor,
        x_mark_dec: Optional[Tensor],
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        if self.task_name in {"long_term_forecast", "short_term_forecast"}:
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if self.task_name == "imputation":
            if mask is None:
                raise ValueError("mask must not be None for imputation")
            return self.imputation(x_enc, x_mark_enc, mask)
        if self.task_name == "anomaly_detection":
            return self.anomaly_detection(x_enc)
        if self.task_name == "classification":
            if x_mark_enc is None:
                raise ValueError("x_mark_enc must not be None for classification")
            return self.classification(x_enc, x_mark_enc)
        raise ValueError("Other tasks implemented yet")