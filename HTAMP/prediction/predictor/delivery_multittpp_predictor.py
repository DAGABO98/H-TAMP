from __future__ import annotations

import argparse
import datetime
import os
import traceback

from lightning.pytorch.loggers import CSVLogger

from HTAMP.prediction.configs.delivery_tpp_config import (
    DeliveryMultiTTPPDatasetConfig,
    DeliveryMultiTTPPModelConfig,
    DeliveryMultiTTPPTrainingConfig,
)
from HTAMP.prediction.data_provider.data_module import DataModule
from HTAMP.prediction.data_provider.delivery_multittpp_dataset import (
    DeliveryMultiTTPPDatasetBundle,
    DeliveryMultiTTPPSplitDataset,
    build_delivery_multittpp_dataset_bundle,
)
from HTAMP.prediction.module.vital_sign_multittpp_module import VitalSignMultiTTPPModule
from HTAMP.prediction.predictor.vital_sign_multittpp_predictor import (
    VitalSignMultiTTPPPredictor,
)


class DeliveryMultiTTPPPredictor(VitalSignMultiTTPPPredictor):
    def create_data_module_and_bundle(
        self,
        *,
        dataset_config: DeliveryMultiTTPPDatasetConfig,
        model_config: DeliveryMultiTTPPModelConfig,
    ) -> tuple[DataModule, DeliveryMultiTTPPDatasetBundle]:
        dataset_bundle = build_delivery_multittpp_dataset_bundle(
            dataset_config=dataset_config,
        )
        data_module = DataModule(
            dataset_cls=DeliveryMultiTTPPSplitDataset,
            dataset_kwargs={"dataset_bundle": dataset_bundle},
            batch_size=model_config.batch_size,
            workers=model_config.num_workers,
            collate_fun=dataset_bundle.collator(n_min=model_config.block_size),
        )
        return data_module, dataset_bundle

    def create_model(
        self,
        *,
        model_config: DeliveryMultiTTPPModelConfig,
        dataset_bundle: DeliveryMultiTTPPDatasetBundle,
    ) -> VitalSignMultiTTPPModule:
        return VitalSignMultiTTPPModule(
            model_config=model_config,
            num_event_types=dataset_bundle.num_event_types,
            n_events=dataset_bundle.n_events,
            t_max_normalization=dataset_bundle.t_max_normalization,
            dt_max_normalization=dataset_bundle.dt_max_normalization,
        )

    def _create_logger(
        self,
        *,
        model_config: DeliveryMultiTTPPModelConfig,
        config_payload: dict[str, object],
        log_dir: str,
    ):
        if not model_config.wandb:
            logger = CSVLogger(
                save_dir=log_dir,
                name="delivery_multittpp",
                version=model_config.run_name,
            )
            logger.log_hyperparams(config_payload)
            return logger

        import wandb
        from lightning.pytorch.loggers import WandbLogger

        experiment = wandb.init(
            project=os.getenv("WANDB_PROJECT", "delivery_multittpp"),
            group=os.getenv("WANDB_GROUP"),
            job_type=os.getenv("WANDB_JOB_TYPE", "train"),
            config=config_payload,
            dir=log_dir,
        )
        wandb.run.name = model_config.run_name
        wandb.run.save(log_dir)
        logger = WandbLogger(experiment=experiment, save_dir=log_dir)
        logger.log_hyperparams(config_payload)
        return logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="DeliveryMultiTTPPPredictor",
        description="Train a MultiTTPP medication delivery request model from a JSON config file.",
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
        training_config = DeliveryMultiTTPPTrainingConfig.from_json_file(
            parsed_args.config_path
        )
        predictor = DeliveryMultiTTPPPredictor()
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
