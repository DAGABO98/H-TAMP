from __future__ import annotations

from importlib import import_module
from typing import Any, Mapping

import lightning as L
import torch

from HTAMP.prediction.configs.vital_sign_easy_tpp_config import (
    VitalSignEasyTPPModelConfig,
)

EASY_TPP_MODEL_MODULES = (
    "torch_anhn",
    "torch_attnhp",
    "torch_fullynn",
    "torch_intensity_free",
    "torch_nhp",
    "torch_ode_tpp",
    "torch_rmtpp",
    "torch_s2p2",
    "torch_sahp",
    "torch_thp",
    "torch_wsm_thp",
)
EASY_TPP_PACKAGE = "HTAMP.prediction.point_process_models.easyTPP"


def _import_easy_tpp_model_registry() -> None:
    for module_name in EASY_TPP_MODEL_MODULES:
        import_module(f"{EASY_TPP_PACKAGE}.{module_name}")


class _ThinningConfigAdapter:
    def __init__(self, payload: Mapping[str, Any] | None) -> None:
        payload = dict(payload or {})
        self.num_seq = int(payload.get("num_seq", 10))
        self.num_sample = int(payload.get("num_sample", 1))
        self.num_exp = int(payload.get("num_exp", 500))
        self.look_ahead_time = float(payload.get("look_ahead_time", 10.0))
        self.patience_counter = int(payload.get("patience_counter", 5))
        self.over_sample_rate = float(payload.get("over_sample_rate", 5.0))
        self.num_samples_boundary = int(payload.get("num_samples_boundary", 5))
        self.dtime_max = float(payload.get("dtime_max", 5.0))
        self.num_step_gen = int(payload.get("num_step_gen", 1))


class _EasyTPPModelConfigAdapter:
    def __init__(
        self,
        *,
        model_config: VitalSignEasyTPPModelConfig,
        num_event_types: int,
        pad_token_id: int,
        mean_log_inter_time: float | None = None,
        std_log_inter_time: float | None = None,
        max_observed_time: float | None = None,
    ) -> None:
        payload = model_config.to_dict()
        self.model_id = str(payload["model_id"])
        self.rnn_type = str(payload["rnn_type"])
        self.hidden_size = int(payload["hidden_size"])
        self.time_emb_size = int(payload["time_emb_size"])
        self.num_layers = int(payload["num_layers"])
        self.num_heads = int(payload["num_heads"])
        self.sharing_param_layer = bool(payload["sharing_param_layer"])
        self.use_mc_samples = bool(payload["use_mc_samples"])
        self.loss_integral_num_sample_per_step = int(
            payload["loss_integral_num_sample_per_step"]
        )
        self.dropout_rate = float(payload["dropout_rate"])
        self.dropout = self.dropout_rate
        self.use_ln = bool(payload["use_ln"])
        self.is_training = True
        self.training = True
        self.num_event_types = int(num_event_types)
        self.num_event_types_pad = int(num_event_types) + 1
        self.pad_token_id = int(pad_token_id)
        self.gpu = int(payload.get("gpu", -1))
        self.pretrained_model_dir = None
        self.thinning = (
            None
            if payload.get("thinning") is None
            else _ThinningConfigAdapter(payload.get("thinning"))
        )
        self.model_specs = dict(payload.get("model_specs") or {})
        if mean_log_inter_time is not None:
            self.mean_log_inter_time = float(mean_log_inter_time)
        if std_log_inter_time is not None:
            self.std_log_inter_time = float(std_log_inter_time)
        if max_observed_time is not None and self.model_id == "WSMTHP":
            t_mode = str(self.model_specs.get("T_mode", "train_global")).lower()
            if t_mode == "train_global":
                self.model_specs["max_observed_time"] = float(max_observed_time)

    def __getitem__(self, key: str) -> Any:
        value = self.get(key, None)
        if value is None and key not in self.model_specs and not hasattr(self, key):
            raise KeyError(key)
        return value

    def get(self, key: str, default_var: Any = None) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        if key == "dropout":
            return self.dropout_rate
        return self.model_specs.get(key, default_var)


def _coerce_model_config(
    model_config: VitalSignEasyTPPModelConfig | dict[str, Any],
) -> VitalSignEasyTPPModelConfig:
    if isinstance(model_config, VitalSignEasyTPPModelConfig):
        return model_config
    return VitalSignEasyTPPModelConfig.from_dict(model_config)


class VitalSignEasyTPPModule(L.LightningModule):
    def __init__(
        self,
        *,
        model_config: VitalSignEasyTPPModelConfig | dict[str, Any],
        num_event_types: int,
        pad_token_id: int,
        mean_log_inter_time: float | None = None,
        std_log_inter_time: float | None = None,
        max_observed_time: float | None = None,
    ) -> None:
        super().__init__()
        self.model_config = _coerce_model_config(model_config=model_config)
        self.num_event_types = int(num_event_types)
        self.pad_token_id = int(pad_token_id)
        self.mean_log_inter_time = mean_log_inter_time
        self.std_log_inter_time = std_log_inter_time
        self.max_observed_time = max_observed_time

        _import_easy_tpp_model_registry()
        from HTAMP.prediction.point_process_models.easyTPP.torch_basemodel import TorchBaseModel

        adapter = _EasyTPPModelConfigAdapter(
            model_config=self.model_config,
            num_event_types=self.num_event_types,
            pad_token_id=self.pad_token_id,
            mean_log_inter_time=mean_log_inter_time,
            std_log_inter_time=std_log_inter_time,
            max_observed_time=max_observed_time,
        )
        self.easy_tpp_model = TorchBaseModel.generate_model_from_config(adapter)

        self.save_hyperparameters(
            {
                "model_config": self.model_config.to_dict(),
                "num_event_types": self.num_event_types,
                "pad_token_id": self.pad_token_id,
                "mean_log_inter_time": self.mean_log_inter_time,
                "std_log_inter_time": self.std_log_inter_time,
                "max_observed_time": self.max_observed_time,
            }
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.model_config.learning_rate,
            weight_decay=self.model_config.weight_decay,
        )

    def _batch_tuple(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        clone_tensors: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        batch_tuple = (
            batch["time_seqs"],
            batch["time_delta_seqs"],
            batch["type_seqs"],
            batch["seq_non_pad_mask"],
            batch["attention_mask"],
        )
        if clone_tensors:
            return tuple(tensor.clone() for tensor in batch_tuple)
        return batch_tuple

    def _sync_easy_tpp_device(self, batch: Mapping[str, torch.Tensor]) -> None:
        device = batch["time_seqs"].device
        if hasattr(self.easy_tpp_model, "device"):
            self.easy_tpp_model.device = device
        event_sampler = getattr(self.easy_tpp_model, "event_sampler", None)
        if event_sampler is not None and hasattr(event_sampler, "device"):
            event_sampler.device = device

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self._compute_metrics(batch)["nll"]

    def _loglike_loss(self, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, int]:
        if self.model_config.model_id == "FullyNN":
            with torch.inference_mode(False):
                with torch.enable_grad():
                    batch_tuple = self._batch_tuple(batch, clone_tensors=True)
                    return self.easy_tpp_model.loglike_loss(batch_tuple)
        batch_tuple = self._batch_tuple(batch)
        return self.easy_tpp_model.loglike_loss(batch_tuple)

    def _compute_metrics(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        detach_metrics: bool = False,
    ) -> dict[str, torch.Tensor]:
        self._sync_easy_tpp_device(batch)
        loss, num_events = self._loglike_loss(batch)
        normalizer = torch.as_tensor(
            max(float(num_events), 1.0),
            dtype=loss.dtype,
            device=loss.device,
        )
        nll = loss / normalizer
        metrics = {
            "loss": loss,
            "nll": nll,
            "num_events": normalizer,
        }
        if detach_metrics:
            metrics = {
                metric_name: metric_value.detach()
                for metric_name, metric_value in metrics.items()
            }
        return metrics

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

    def training_step(
        self,
        batch: Mapping[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        metrics = self._compute_metrics(batch)
        self._log_metrics("train", metrics, batch_size=int(batch["time_seqs"].shape[0]))
        return metrics["nll"]

    def validation_step(
        self,
        batch: Mapping[str, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        metrics = self._compute_metrics(batch, detach_metrics=True)
        self._log_metrics("val", metrics, batch_size=int(batch["time_seqs"].shape[0]))
        return metrics

    def test_step(
        self,
        batch: Mapping[str, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        metrics = self._compute_metrics(batch, detach_metrics=True)
        self._log_metrics("test", metrics, batch_size=int(batch["time_seqs"].shape[0]))
        return metrics
