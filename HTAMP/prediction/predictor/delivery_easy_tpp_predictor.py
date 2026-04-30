from __future__ import annotations

import argparse
import datetime
import os
import traceback

from lightning.pytorch.loggers import CSVLogger

from HTAMP.prediction.configs.delivery_tpp_config import (
    DeliveryEasyTPPDatasetConfig,
    DeliveryEasyTPPModelConfig,
    DeliveryEasyTPPTrainingConfig,
)
from HTAMP.prediction.data_provider.data_module import DataModule
from HTAMP.prediction.data_provider.delivery_easy_tpp_dataset import (
    DeliveryEasyTPPDatasetBundle,
    DeliveryEasyTPPSplitDataset,
    build_delivery_easy_tpp_dataset_bundle,
)
from HTAMP.prediction.module.vital_sign_easy_tpp_module import VitalSignEasyTPPModule
from HTAMP.prediction.predictor.vital_sign_easy_tpp_predictor import (
    VitalSignEasyTPPPredictor,
)


class DeliveryEasyTPPPredictor(VitalSignEasyTPPPredictor):
    def create_data_module_and_bundle(
        self,
        *,
        dataset_config: DeliveryEasyTPPDatasetConfig,
        model_config: DeliveryEasyTPPModelConfig,
    ) -> tuple[DataModule, DeliveryEasyTPPDatasetBundle]:
        dataset_bundle = build_delivery_easy_tpp_dataset_bundle(
            dataset_config=dataset_config,
        )
        data_module = DataModule(
            dataset_cls=DeliveryEasyTPPSplitDataset,
            dataset_kwargs={"dataset_bundle": dataset_bundle},
            batch_size=model_config.batch_size,
            workers=model_config.num_workers,
            collate_fun=dataset_bundle.collator(),
        )
        return data_module, dataset_bundle

    def create_model(
        self,
        *,
        model_config: DeliveryEasyTPPModelConfig,
        dataset_bundle: DeliveryEasyTPPDatasetBundle,
    ) -> VitalSignEasyTPPModule:
        mean_log_inter_time = None
        std_log_inter_time = None
        if model_config.model_id == "IntensityFree":
            mean_log_inter_time, std_log_inter_time = dataset_bundle.log_inter_time_stats(
                "train"
            )

        max_observed_time = None
        if model_config.model_id == "WSMTHP":
            max_observed_time = dataset_bundle.max_event_time("train")

        return VitalSignEasyTPPModule(
            model_config=model_config,
            num_event_types=dataset_bundle.num_event_types,
            pad_token_id=dataset_bundle.pad_token_id,
            mean_log_inter_time=mean_log_inter_time,
            std_log_inter_time=std_log_inter_time,
            max_observed_time=max_observed_time,
        )

    def _create_logger(
        self,
        *,
        model_config: DeliveryEasyTPPModelConfig,
        config_payload: dict[str, object],
        log_dir: str,
    ):
        if not model_config.wandb:
            logger = CSVLogger(
                save_dir=log_dir,
                name="delivery_easy_tpp",
                version=model_config.run_name,
            )
            logger.log_hyperparams(config_payload)
            return logger

        import wandb
        from lightning.pytorch.loggers import WandbLogger

        experiment = wandb.init(
            project=os.getenv("WANDB_PROJECT", "delivery_easy_tpp"),
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
        prog="DeliveryEasyTPPPredictor",
        description="Train an EasyTPP medication delivery request model from a JSON config file.",
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
        training_config = DeliveryEasyTPPTrainingConfig.from_json_file(
            parsed_args.config_path
        )
        predictor = DeliveryEasyTPPPredictor()
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
