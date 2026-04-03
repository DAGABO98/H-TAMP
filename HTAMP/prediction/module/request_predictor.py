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

from HTAMP.prediction.data_provider.data_module import DataModule
from HTAMP.prediction.data_provider.requests_dataset import (
    RequestsDataset,
    RequestsTimeSeries,
    build_request_time_series,
)
from HTAMP.prediction.configs.request_config import (
    MedicalRequestDatasetConfig,
    RequestTrainingConfig,
    TimeseriesModelConfig,
)
from HTAMP.prediction.module.request_module import RequestsModule

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

class RequestsPredictor:
    def create_data_module_and_time_series(
        self,
        model_config: TimeseriesModelConfig,
        dataset_config: MedicalRequestDatasetConfig,
    ) -> tuple[DataModule, RequestsTimeSeries]:
        time_series = build_request_time_series(
            dataset_config=dataset_config,
            model_config=model_config,
        )

        data_module = DataModule(
            dataset_cls=RequestsDataset,
            dataset_kwargs={
                "request_time_series": time_series,
                "sequence_length": model_config.seq_len,
                "label_length": model_config.label_len,
                "prediction_length": model_config.pred_len,
            },
            batch_size=model_config.batch_size,
            workers=model_config.num_workers,
            collate_fun=None,
        )

        return data_module, time_series

    def create_model(
        self,
        model_config: TimeseriesModelConfig,
        time_series: RequestsTimeSeries,
    ) -> RequestsModule:
        return RequestsModule(
            model_config=model_config,
            target_scaler_mean=time_series.target_scaler_mean.tolist(),
            target_scaler_scale=time_series.target_scaler_scale.tolist(),
            delta_target_indices=time_series.delta_target_indices,
            availability_target_indices=time_series.availability_target_indices,
        )

    def create_callbacks(
        self,
        model_config: TimeseriesModelConfig,
        save_dir: str,
    ) -> list[object]:
        run_dir = Path(save_dir) / model_config.run_name
        checkpoint = ModelCheckpoint(
            dirpath=run_dir,
            monitor="val_loss",
            mode="min",
            filename=f"{model_config.run_name}" + "{epoch:02d}",
            save_top_k=1,
            auto_insert_metric_name=True,
        )

        return [
            checkpoint,
            EarlyStopping(monitor="val_loss", patience=model_config.patience),
            LearningRateMonitor(),
        ]

    def _create_logger(
        self,
        model_config: TimeseriesModelConfig,
        config_payload: dict[str, object],
        log_dir: str,
    ):
        if not model_config.wandb:
            return None

        import wandb
        from lightning.pytorch.loggers import WandbLogger

        experiment = wandb.init(
            project="medical_request_timeseries",
            config=config_payload,
            dir=log_dir,
        )
        wandb.run.name = model_config.run_name
        wandb.run.save(log_dir)

        logger = WandbLogger(experiment=experiment, save_dir=log_dir)
        logger.log_hyperparams(config_payload)
        return logger

    def _resolve_strategy(self, model_config: TimeseriesModelConfig) -> str | None:
        if model_config.strategy is not None:
            return model_config.strategy

        if model_config.model_name == "TimeMixer" and model_config.devices != 1:
            return "ddp_find_unused_parameters_true"

        return "auto"

    def _run_dir(self, log_dir: str, run_name: str) -> Path:
        return Path(log_dir) / run_name

    def _metrics_summary_path(self, log_dir: str, run_name: str) -> Path:
        return self._run_dir(log_dir=log_dir, run_name=run_name) / METRICS_SUMMARY_FILENAME

    def _write_metrics_summary(
        self,
        training_config: RequestTrainingConfig,
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
            "model_name": training_config.model_config.model_name,
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

    def compile_and_train(self, training_config: RequestTrainingConfig) -> dict[str, Any]:
        dataset_config = training_config.dataset_config
        model_config = training_config.model_config

        log_dir = os.getenv("STF_LOG_DIR", "./data/STF_LOG_DIR")
        os.makedirs(log_dir, exist_ok=True)

        logger = self._create_logger(
            model_config=model_config,
            config_payload=training_config.to_dict(),
            log_dir=log_dir,
        )
        data_module, time_series = self.create_data_module_and_time_series(
            model_config=model_config,
            dataset_config=dataset_config,
        )
        forecaster = self.create_model(model_config=model_config, time_series=time_series)
        callbacks = self.create_callbacks(model_config=model_config, save_dir=log_dir)
        strategy = self._resolve_strategy(model_config=model_config)
        checkpoint_callback = next(
            callback for callback in callbacks if isinstance(callback, ModelCheckpoint)
        )

        trainer = Trainer(
            accelerator=model_config.accelerator,
            devices=model_config.devices,
            strategy=strategy,
            callbacks=callbacks,
            logger=logger,
            max_epochs=model_config.max_epochs,
        )

        trainer.fit(forecaster, datamodule=data_module)
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
        prog="RequestsPredictor",
        description="Train a medical request-interval forecasting model from a JSON config file.",
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
        training_config = RequestTrainingConfig.from_json_file(parsed_args.config_path)
        request_predictor = RequestsPredictor()
        request_predictor.compile_and_train(training_config=training_config)
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
