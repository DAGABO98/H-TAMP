from __future__ import annotations

import argparse
import datetime
import os
import traceback
from pathlib import Path

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from HTAMP.prediction.data_provider.data_module import DataModule
from HTAMP.prediction.data_provider.requests_dataset import (
    RequestsDataManager,
    RequestsDataset,
    RequestsTimeSeries,
)
from HTAMP.prediction.configs.request_config import (
    MedicalRequestDatasetConfig,
    RequestTrainingConfig,
    TimeseriesModelConfig,
)
from HTAMP.prediction.module.request_module import RequestsModule

class RequestsPredictor:
    def create_data_module_and_time_series(
        self,
        model_config: TimeseriesModelConfig,
        dataset_config: MedicalRequestDatasetConfig,
    ) -> tuple[DataModule, RequestsTimeSeries]:
        request_data_manager = RequestsDataManager(dataset_config=dataset_config)

        train_data_df, train_segments_df = request_data_manager.get_requests_training_data()
        val_data_df, val_segments_df = request_data_manager.get_requests_validation_data()
        test_data_df, test_segments_df = request_data_manager.get_requests_testing_data()

        time_series = RequestsTimeSeries(
            train_data_df=train_data_df,
            val_data_df=val_data_df,
            test_data_df=test_data_df,
            train_segments_df=train_segments_df,
            val_segments_df=val_segments_df,
            test_segments_df=test_segments_df,
            metadata=request_data_manager.metadata,
            sequence_length=model_config.seq_len,
            label_length=model_config.label_len,
            prediction_length=model_config.pred_len,
        )

        model_config.sync_channel_dimensions(
            num_input_channels=len(time_series.input_feature_cols),
            num_output_channels=len(time_series.target_cols),
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
            reinit=True,
        )
        wandb.run.name = model_config.run_name
        wandb.run.save()

        logger = WandbLogger(experiment=experiment, save_dir=log_dir)
        logger.log_hyperparams(config_payload)
        return logger

    def _resolve_strategy(self, model_config: TimeseriesModelConfig) -> str | None:
        if model_config.strategy is not None:
            return model_config.strategy

        if model_config.model_name == "TimeMixer" and model_config.devices != 1:
            return "ddp_find_unused_parameters_true"

        return "auto"

    def compile_and_train(self, training_config: RequestTrainingConfig) -> None:
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

        trainer = Trainer(
            accelerator=model_config.accelerator,
            devices=model_config.devices,
            strategy=strategy,
            callbacks=callbacks,
            logger=logger,
            max_epochs=model_config.max_epochs,
        )

        trainer.fit(forecaster, datamodule=data_module)
        trainer.test(datamodule=data_module, ckpt_path="best")


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


if __name__ == "__main__":
    p_start = datetime.datetime.now()
    try:
        parser = build_parser()
        parsed_args = parser.parse_args()
        training_config = RequestTrainingConfig.from_json_file(parsed_args.config_path)
        request_predictor = RequestsPredictor()
        request_predictor.compile_and_train(training_config=training_config)
    except Exception as error_main_context:
        print("Fail End Process: ", error_main_context)
        traceback.print_exc()
    p_stop = datetime.datetime.now()
    print("Execution time: " + str(p_stop - p_start))
