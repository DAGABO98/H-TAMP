from __future__ import annotations

import argparse
import datetime
import json
import os
import traceback
from pathlib import Path
from typing import Any

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from HTAMP.prediction.configs.vital_sign_tpp_config import (
    VitalSignTPPModelConfig,
    VitalSignTPPDatasetConfig,
    VitalSignTPPTrainingConfig,
)
from HTAMP.prediction.data_provider.data_module import DataModule
from HTAMP.prediction.data_provider.vital_sign_tpp_dataset import (
    VitalSignTPPDatasetBundle,
    VitalSignTPPSplitDataset,
    build_vital_sign_tpp_dataset_bundle,
)
from HTAMP.prediction.point_process_models.flexTPP.dataset.base import batch_collate
from HTAMP.prediction.module.vital_sign_tpp_module import VitalSignTPPModule

METRICS_SUMMARY_FILENAME = "metrics_summary.json"


def _serialize_metric_value(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _serialize_metric_value(nested_value)
            for key, nested_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_serialize_metric_value(item) for item in value]
    return value


def _result_dict(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {}
    return {
        str(key): _serialize_metric_value(value)
        for key, value in results[0].items()
    }


def _metric_higher_is_better(metric_name: str) -> bool:
    metric_name_lower = str(metric_name).lower()
    return any(
        token in metric_name_lower
        for token in ("accuracy", "auc", "f1", "precision", "recall")
    )


class VitalSignTPPPredictor:
    def create_data_module_and_bundle(
        self,
        *,
        dataset_config: VitalSignTPPDatasetConfig,
        model_config: VitalSignTPPModelConfig,
    ) -> tuple[DataModule, VitalSignTPPDatasetBundle]:
        dataset_bundle = build_vital_sign_tpp_dataset_bundle(
            dataset_config=dataset_config,
            model_config=model_config,
        )
        data_module = DataModule(
            dataset_cls=VitalSignTPPSplitDataset,
            dataset_kwargs={"dataset_bundle": dataset_bundle},
            batch_size=model_config.batch_size,
            workers=model_config.num_workers,
            collate_fun=batch_collate,
        )
        return data_module, dataset_bundle

    def create_model(
        self,
        *,
        model_config: VitalSignTPPModelConfig,
        dataset_bundle: VitalSignTPPDatasetBundle,
    ) -> VitalSignTPPModule:
        return VitalSignTPPModule(
            model_config=model_config,
            dims=dataset_bundle.dims,
            max_num_classes=dataset_bundle.max_num_classes,
        )

    def create_callbacks(
        self,
        *,
        model_config: VitalSignTPPModelConfig,
        save_dir: str,
    ) -> list[object]:
        run_dir = Path(save_dir) / model_config.run_name
        monitor_metric = model_config.monitor_metric
        monitor_mode = (
            model_config.monitor_mode
            if model_config.monitor_mode is not None
            else ("max" if _metric_higher_is_better(metric_name=monitor_metric) else "min")
        )
        checkpoint = ModelCheckpoint(
            dirpath=run_dir,
            monitor=monitor_metric,
            mode=monitor_mode,
            filename=f"{model_config.run_name}" + "{epoch:02d}",
            save_top_k=1,
            auto_insert_metric_name=True,
        )
        return [
            checkpoint,
            EarlyStopping(
                monitor=monitor_metric,
                mode=monitor_mode,
                patience=model_config.patience,
            ),
            LearningRateMonitor(),
        ]

    def _create_logger(
        self,
        *,
        model_config: VitalSignTPPModelConfig,
        config_payload: dict[str, object],
        log_dir: str,
    ):
        if not model_config.wandb:
            logger = CSVLogger(
                save_dir=log_dir,
                name="vital_sign_tpp",
                version=model_config.run_name,
            )
            logger.log_hyperparams(config_payload)
            return logger

        import wandb
        from lightning.pytorch.loggers import WandbLogger

        experiment = wandb.init(
            project="vital_sign_tpp",
            config=config_payload,
            dir=log_dir,
        )
        wandb.run.name = model_config.run_name
        wandb.run.save(log_dir)
        logger = WandbLogger(experiment=experiment, save_dir=log_dir)
        logger.log_hyperparams(config_payload)
        return logger

    def _run_dir(self, *, log_dir: str, run_name: str) -> Path:
        return Path(log_dir) / run_name

    def _metrics_summary_path(self, *, log_dir: str, run_name: str) -> Path:
        return self._run_dir(log_dir=log_dir, run_name=run_name) / METRICS_SUMMARY_FILENAME

    def _write_metrics_summary(
        self,
        *,
        training_config: VitalSignTPPTrainingConfig,
        log_dir: str,
        checkpoint_callback: ModelCheckpoint,
        validation_metrics: dict[str, Any],
        test_metrics: dict[str, Any],
    ) -> Path:
        run_name = training_config.model_config.run_name
        summary_path = self._metrics_summary_path(log_dir=log_dir, run_name=run_name)
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        best_model_score = checkpoint_callback.best_model_score
        metrics_summary = {
            "run_name": run_name,
            "metrics_summary_path": str(summary_path),
            "best_checkpoint_path": checkpoint_callback.best_model_path,
            "best_checkpoint_score": (
                _serialize_metric_value(best_model_score)
                if best_model_score is not None
                else None
            ),
            "validation_metrics": _serialize_metric_value(validation_metrics),
            "test_metrics": _serialize_metric_value(test_metrics),
            "training_config": training_config.to_dict(),
        }

        with summary_path.open("w", encoding="utf-8") as summary_file:
            json.dump(metrics_summary, summary_file, indent=2)

        return summary_path

    def compile_and_train(
        self,
        training_config: VitalSignTPPTrainingConfig,
    ) -> dict[str, Any]:
        dataset_config = training_config.dataset_config
        model_config = training_config.model_config

        log_dir = os.getenv("STF_LOG_DIR", "./data/STF_LOG_DIR")
        os.makedirs(log_dir, exist_ok=True)

        logger = self._create_logger(
            model_config=model_config,
            config_payload=training_config.to_dict(),
            log_dir=log_dir,
        )
        data_module, dataset_bundle = self.create_data_module_and_bundle(
            dataset_config=dataset_config,
            model_config=model_config,
        )
        model = self.create_model(
            model_config=model_config,
            dataset_bundle=dataset_bundle,
        )
        callbacks = self.create_callbacks(model_config=model_config, save_dir=log_dir)
        checkpoint_callback = next(
            callback for callback in callbacks if isinstance(callback, ModelCheckpoint)
        )
        trainer = Trainer(
            accelerator=model_config.accelerator,
            devices=model_config.devices,
            strategy=model_config.strategy or "auto",
            callbacks=callbacks,
            logger=logger,
            default_root_dir=log_dir,
            gradient_clip_val=model_config.gradient_clip_val,
            max_epochs=model_config.max_epochs,
        )

        trainer.fit(model, datamodule=data_module)
        validation_results = trainer.validate(datamodule=data_module, ckpt_path="best", verbose=False)
        test_results = trainer.test(datamodule=data_module, ckpt_path="best", verbose=False)
        metrics_summary_path = self._write_metrics_summary(
            training_config=training_config,
            log_dir=log_dir,
            checkpoint_callback=checkpoint_callback,
            validation_metrics=_result_dict(validation_results),
            test_metrics=_result_dict(test_results),
        )
        print(f"Metrics summary saved to {metrics_summary_path}")
        return {
            "metrics_summary_path": str(metrics_summary_path),
            "validation_metrics": _result_dict(validation_results),
            "test_metrics": _result_dict(test_results),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="VitalSignTPPPredictor",
        description="Train a FlexTPP vital-sign request model from a JSON config file.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to a JSON file containing 'dataset_config' and 'model_config'.",
    )
    return parser


def main() -> int:
    p_start = datetime.datetime.now()
    try:
        parser = build_parser()
        parsed_args = parser.parse_args()
        training_config = VitalSignTPPTrainingConfig.from_json_file(parsed_args.config_path)
        predictor = VitalSignTPPPredictor()
        predictor.compile_and_train(training_config=training_config)
        return 0
    except Exception as error_main_context:
        print("Fail End Process: ", error_main_context)
        traceback.print_exc()
        return 1
    finally:
        p_stop = datetime.datetime.now()
        print("Execution time: " + str(p_stop - p_start))


if __name__ == "__main__":
    raise SystemExit(main())
