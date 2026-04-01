from __future__ import annotations

import argparse
import datetime
import os
import traceback
from pathlib import Path

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from HTAMP.data_processing.data_helpers import DataHelpers
from HTAMP.data_processing.processing_dataclasses import AnnotatedDataFiles
from HTAMP.prediction.data_provider.data_module import DataModule
from HTAMP.prediction.data_provider.requests_dataset import (
    RequestsDataManager,
    RequestsDataset,
    RequestsTimeSeries,
)
from HTAMP.prediction.configs.request_config import (
    MedicalRequestDatasetConfig,
    TimeseriesModelConfig,
)
from HTAMP.prediction.module.request_module import RequestsModule

def build_dataset_config_from_args(args: argparse.Namespace) -> MedicalRequestDatasetConfig:
    dataset_config_kwargs = dict(
        annotated_data_files=AnnotatedDataFiles(
            annotated_visits="",
            annotated_admissions_discharges="",
            annotated_medications=args.medications_orders_file,
            annotated_blood_pressure=args.blood_pressure_orders_file,
            annotated_heart_rate=args.heart_rate_orders_file,
            annotated_respiratory_rate=args.respiratory_rate_orders_file,
            annotated_temperature=args.temperature_orders_file,
            annotated_oxygen_saturation=args.oxygen_saturation_orders_file,
        ),
        request_dir=args.request_dir,
        dataset_dir=args.dataset_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        patient_id_col=args.patient_id_col,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        use_saved_request_data=args.use_saved_request_data,
    )
    if args.test_iso_weeks:
        dataset_config_kwargs["test_iso_weeks"] = tuple(DataHelpers.parse_iso_week_args(args.test_iso_weeks))
    return MedicalRequestDatasetConfig(**dataset_config_kwargs)

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
            datasetCls=RequestsDataset,
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

    def _create_logger(self, args: argparse.Namespace, log_dir: str):
        if not args.wandb:
            return None

        import wandb
        from lightning.pytorch.loggers import WandbLogger

        experiment = wandb.init(
            project="medical_request_timeseries",
            config=vars(args),
            dir=log_dir,
            reinit=True,
        )
        wandb.run.name = args.run_name
        wandb.run.save()

        logger = WandbLogger(experiment=experiment, save_dir=log_dir)
        logger.log_hyperparams(vars(args))
        return logger

    def _resolve_strategy(self, model_config: TimeseriesModelConfig) -> str | None:
        if model_config.strategy is not None:
            return model_config.strategy

        if model_config.model_name == "TimeMixer" and model_config.devices != 1:
            return "ddp_find_unused_parameters_true"

        return "auto"

    def compile_and_train(self, args: argparse.Namespace) -> None:
        model_config = TimeseriesModelConfig.from_namespace(args=args)
        dataset_config = build_dataset_config_from_args(args=args)

        log_dir = os.getenv("STF_LOG_DIR", "./data/STF_LOG_DIR")
        os.makedirs(log_dir, exist_ok=True)

        logger = self._create_logger(args=args, log_dir=log_dir)
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
        description="Train a medical request-interval forecasting model.",
    )

    parser.add_argument("--model_name", type=str, default="TimesNet")
    parser.add_argument("--run_name", type=str, default="TimesNet_medical_request_intervals")
    parser.add_argument("--preprocess_data", action="store_true", default=False)
    parser.add_argument("--wandb", action="store_true", default=False)

    parser.add_argument("--request_dir", type=str, default="data/requests")
    parser.add_argument("--dataset_dir", type=str, default="data/prediction/request_intervals")
    parser.add_argument("--start_date", type=str, default="2024-06-24")
    parser.add_argument("--end_date", type=str, default="2025-06-29")
    parser.add_argument("--patient_id_col", type=str, default="MRN")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument(
        "--test_iso_weeks",
        nargs="*",
        default=None,
        help="Optional explicit test ISO weeks such as 2024W40 2025W05. Defaults to the weeks used in assignment/run_test.py.",
    )
    parser.add_argument("--use_saved_request_data", action="store_true", default=False)

    parser.add_argument(
        "--medications_orders_file",
        type=str,
        default="data/processed/medication_orders_annotated.csv",
    )
    parser.add_argument(
        "--blood_pressure_orders_file",
        type=str,
        default="data/processed/blood_pressure_orders_annotated.csv",
    )
    parser.add_argument(
        "--heart_rate_orders_file",
        type=str,
        default="data/processed/heart_rate_orders_annotated.csv",
    )
    parser.add_argument(
        "--respiratory_rate_orders_file",
        type=str,
        default="data/processed/respiratory_rate_orders_annotated.csv",
    )
    parser.add_argument(
        "--temperature_orders_file",
        type=str,
        default="data/processed/temperature_orders_annotated.csv",
    )
    parser.add_argument(
        "--oxygen_saturation_orders_file",
        type=str,
        default="data/processed/oxygen_saturation_orders_annotated.csv",
    )

    parser.add_argument("--seq_len", type=int, default=5)
    parser.add_argument("--label_len", type=int, default=0)
    parser.add_argument("--pred_len", type=int, default=3)

    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--num_kernels", type=int, default=6)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--d_layers", type=int, default=1)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--moving_avg", type=int, default=25)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--output_attention", action="store_true", default=False)
    parser.add_argument("--channel_independence", type=int, default=0)
    parser.add_argument("--decomp_method", type=str, default="moving_avg")
    parser.add_argument("--use_norm", type=int, default=1)
    parser.add_argument("--down_sampling_layers", type=int, default=0)
    parser.add_argument("--down_sampling_window", type=int, default=1)
    parser.add_argument("--down_sampling_method", type=str, default=None)
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--task_name", type=str, default="long_term_forecast")
    parser.add_argument("--num_class", type=int, default=0)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--loss", type=str, default="MSE")
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--strategy", type=str, default=None)

    return parser


if __name__ == "__main__":
    p_start = datetime.datetime.now()
    try:
        parser = build_parser()
        parsed_args = parser.parse_args()
        request_predictor = RequestsPredictor()
        request_predictor.compile_and_train(args=parsed_args)
    except Exception as error_main_context:
        print("Fail End Process: ", error_main_context)
        traceback.print_exc()
    p_stop = datetime.datetime.now()
    print("Execution time: " + str(p_stop - p_start))
