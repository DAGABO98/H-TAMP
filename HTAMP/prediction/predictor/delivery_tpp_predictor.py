from __future__ import annotations

import argparse
import datetime
import os
import traceback

from lightning.pytorch.loggers import CSVLogger

from HTAMP.prediction.configs.delivery_tpp_config import (
    DeliveryTPPDatasetConfig,
    DeliveryTPPModelConfig,
    DeliveryTPPTrainingConfig,
)
from HTAMP.prediction.data_provider.data_module import DataModule
from HTAMP.prediction.data_provider.delivery_tpp_dataset import (
    DeliveryTPPDatasetBundle,
    DeliveryTPPSplitDataset,
    build_delivery_tpp_dataset_bundle,
)
from HTAMP.prediction.module.vital_sign_tpp_module import VitalSignTPPModule
from HTAMP.prediction.point_process_models.flexTPP.dataset.base import batch_collate
from HTAMP.prediction.predictor.vital_sign_tpp_predictor import VitalSignTPPPredictor
from HTAMP.prediction.predictor.wandb_utils import wandb_init_settings


class DeliveryTPPPredictor(VitalSignTPPPredictor):
    def create_data_module_and_bundle(
        self,
        *,
        dataset_config: DeliveryTPPDatasetConfig,
        model_config: DeliveryTPPModelConfig,
    ) -> tuple[DataModule, DeliveryTPPDatasetBundle]:
        dataset_bundle = build_delivery_tpp_dataset_bundle(
            dataset_config=dataset_config,
            model_config=model_config,
        )
        data_module = DataModule(
            dataset_cls=DeliveryTPPSplitDataset,
            dataset_kwargs={"dataset_bundle": dataset_bundle},
            batch_size=model_config.batch_size,
            workers=model_config.num_workers,
            collate_fun=batch_collate,
        )
        return data_module, dataset_bundle

    def create_model(
        self,
        *,
        model_config: DeliveryTPPModelConfig,
        dataset_bundle: DeliveryTPPDatasetBundle,
    ) -> VitalSignTPPModule:
        return VitalSignTPPModule(
            model_config=model_config,
            dims=dataset_bundle.dims,
            max_num_classes=dataset_bundle.max_num_classes,
            condition_dim=dataset_bundle.condition_dim,
        )

    def _create_logger(
        self,
        *,
        model_config: DeliveryTPPModelConfig,
        config_payload: dict[str, object],
        log_dir: str,
    ):
        if not model_config.wandb:
            logger = CSVLogger(
                save_dir=log_dir,
                name="delivery_tpp",
                version=model_config.run_name,
            )
            logger.log_hyperparams(config_payload)
            return logger

        import wandb
        from lightning.pytorch.loggers import WandbLogger

        experiment = wandb.init(
            project=os.getenv("WANDB_PROJECT", "delivery_tpp"),
            group=os.getenv("WANDB_GROUP"),
            job_type=os.getenv("WANDB_JOB_TYPE", "train"),
            config=config_payload,
            dir=log_dir,
            settings=wandb_init_settings(wandb),
        )
        wandb.run.name = model_config.run_name
        wandb.run.save(log_dir)
        logger = WandbLogger(experiment=experiment, save_dir=log_dir)
        logger.log_hyperparams(config_payload)
        return logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="DeliveryTPPPredictor",
        description="Train a FlexTPP medication delivery request model from a JSON config file.",
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
        training_config = DeliveryTPPTrainingConfig.from_json_file(parsed_args.config_path)
        predictor = DeliveryTPPPredictor()
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
