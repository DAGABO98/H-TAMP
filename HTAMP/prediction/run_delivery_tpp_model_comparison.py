from __future__ import annotations

import argparse
import copy
import datetime
import logging
import os
from pathlib import Path
from typing import Any, Mapping

from HTAMP.prediction import run_vital_sign_tpp_model_comparison as base

LOGGER = logging.getLogger(__name__)

DEFAULT_EASY_CONFIG_PATH = (
    "HTAMP/prediction/configs/config_files/prediction/"
    "delivery_easy_tpp_training.json"
)
DEFAULT_FLEX_CONFIG_PATH = (
    "HTAMP/prediction/configs/config_files/prediction/"
    "delivery_tpp_training.json"
)
DEFAULT_MULTITTPP_CONFIG_PATH = (
    "HTAMP/prediction/configs/config_files/prediction/"
    "delivery_multittpp_training.json"
)


def _run_prefix(raw_prefix: str | None) -> str:
    if raw_prefix:
        return raw_prefix
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"delivery_tpp_comparison_{timestamp}"


def _temp_config_dir(args: argparse.Namespace) -> Path:
    return base._log_dir(args) / "temp_configs" / "delivery_tpp_comparison"


def _default_summary_path(run_prefix: str) -> Path:
    return base._repo_root() / "data" / "prediction" / "delivery_tpp_comparison" / (
        f"{base._safe_name(run_prefix)}_summary.csv"
    )


def _summary_path(args: argparse.Namespace, run_prefix: str) -> Path:
    if args.summary_path:
        return base._resolve_repo_relative_path(args.summary_path)
    return _default_summary_path(run_prefix)


def _comparison_log_dir(args: argparse.Namespace) -> Path:
    return base._log_dir(args) / "delivery_comparison_logs"


def _comparison_log_path(args: argparse.Namespace, run_name: str) -> Path:
    return _comparison_log_dir(args) / f"{base._safe_name(run_name)}.log"


def _flex_st_schema_suffix(mark_schema: str) -> str:
    return "enhanced_marks" if mark_schema == "enhanced" else "standard_marks"


def _easy_schema_suffix(mark_schema: str) -> str:
    return "enhanced_marks" if mark_schema == "enhanced" else "plain_marks"


def _multittpp_schema_suffix(mark_schema: str) -> str:
    return "enhanced_marks" if mark_schema == "enhanced" else "standard_marks"


def _apply_flex_event_type_schema(
    payload: Mapping[str, Any],
    *,
    order: str,
    mark_schema: str,
) -> dict[str, Any]:
    updated_payload = copy.deepcopy(dict(payload))
    dataset_config = updated_payload.setdefault("dataset_config", {})
    if mark_schema == "standard":
        dataset_config["event_type_mark_mode"] = "task"
    elif mark_schema == "enhanced":
        if order != "ST":
            raise ValueError("Enhanced FlexTPP delivery marks are currently defined for ST jobs.")
        dataset_config["event_type_mark_mode"] = "medication_code"
        dataset_dir = Path(str(dataset_config.get("dataset_dir", "data/prediction/delivery_tpp_dataset")))
        dataset_config["dataset_dir"] = str(dataset_dir / f"flex_st_{_flex_st_schema_suffix(mark_schema)}")
    else:
        raise ValueError(f"Unsupported FlexTPP ST mark schema '{mark_schema}'.")
    return updated_payload


def _apply_flex_conditioning_mode(
    payload: Mapping[str, Any],
    *,
    conditioning_mode: str,
) -> dict[str, Any]:
    updated_payload = copy.deepcopy(dict(payload))
    dataset_config = updated_payload.setdefault("dataset_config", {})
    if conditioning_mode == "conditioned":
        dataset_config["use_previous_day_summary_conditioning"] = True
    elif conditioning_mode == "no_conditioning":
        dataset_config["use_previous_day_summary_conditioning"] = False
        dataset_dir = Path(str(dataset_config.get("dataset_dir", "data/prediction/delivery_tpp_dataset")))
        dataset_config["dataset_dir"] = str(dataset_dir / base._flex_conditioning_suffix(conditioning_mode))
    else:
        raise ValueError(f"Unsupported FlexTPP conditioning mode '{conditioning_mode}'.")
    return updated_payload


def _apply_flex_dataset_variant(
    payload: Mapping[str, Any],
    *,
    order: str,
    mark_schema: str,
    conditioning_mode: str,
) -> dict[str, Any]:
    return _apply_flex_conditioning_mode(
        _apply_flex_event_type_schema(
            payload,
            order=order,
            mark_schema=mark_schema,
        ),
        conditioning_mode=conditioning_mode,
    )


def _apply_easy_mark_schema(
    payload: Mapping[str, Any],
    *,
    mark_schema: str,
    separate_dataset_dir: bool,
) -> dict[str, Any]:
    updated_payload = copy.deepcopy(dict(payload))
    dataset_config = updated_payload.setdefault("dataset_config", {})
    if mark_schema == "plain":
        dataset_config["mark_label_mode"] = "task"
        dataset_config["event_type_mark_mode"] = "task"
    elif mark_schema == "enhanced":
        dataset_config["mark_label_mode"] = "medication_code"
        dataset_config["event_type_mark_mode"] = "medication_code"
    else:
        raise ValueError(f"Unsupported EasyTPP mark schema '{mark_schema}'.")

    if separate_dataset_dir:
        dataset_dir = Path(str(dataset_config.get("dataset_dir", "data/prediction/delivery_easy_tpp_dataset")))
        dataset_config["dataset_dir"] = str(dataset_dir / _easy_schema_suffix(mark_schema))
    return updated_payload


def _apply_multittpp_mark_schema(
    payload: Mapping[str, Any],
    *,
    mark_schema: str,
    separate_dataset_dir: bool,
) -> dict[str, Any]:
    updated_payload = copy.deepcopy(dict(payload))
    dataset_config = updated_payload.setdefault("dataset_config", {})
    if mark_schema == "plain":
        dataset_config["mark_label_mode"] = "task"
        dataset_config["event_type_mark_mode"] = "task"
    elif mark_schema == "enhanced":
        dataset_config["mark_label_mode"] = "medication_code"
        dataset_config["event_type_mark_mode"] = "medication_code"
    else:
        raise ValueError(f"Unsupported MultiTTPP mark schema '{mark_schema}'.")

    dataset_config["include_eos_event"] = False
    if separate_dataset_dir:
        dataset_dir = Path(str(dataset_config.get("dataset_dir", "data/prediction/delivery_multittpp_dataset")))
        dataset_config["dataset_dir"] = str(dataset_dir / _multittpp_schema_suffix(mark_schema))
    return updated_payload


def _build_flex_job_payload(
    *,
    base_payload: Mapping[str, Any],
    args: argparse.Namespace,
    run_prefix: str,
    order: str,
    mark_schema: str,
    conditioning_mode: str,
) -> base.ComparisonJob:
    schema_suffix = (
        f"_{_flex_st_schema_suffix(mark_schema)}"
        if order == "ST" and mark_schema == "enhanced"
        else ""
    )
    conditioning_suffix = (
        f"_{base._flex_conditioning_suffix(conditioning_mode)}"
        if conditioning_mode == "no_conditioning"
        else ""
    )
    run_name = f"{run_prefix}_flex_tpp_{order.lower()}{schema_suffix}{conditioning_suffix}"
    payload = base._apply_dataset_training_flags(
        _apply_flex_dataset_variant(
            base_payload,
            order=order,
            mark_schema=mark_schema,
            conditioning_mode=conditioning_mode,
        ),
        use_prepared_dataset=args.prepare_datasets,
    )
    payload["model_config"]["order"] = order
    payload = base._apply_architecture_defaults(
        payload,
        defaults=base.FLEX_TPP_VARIANT_DEFAULTS.get(order, {}),
        enabled=not args.no_model_defaults,
    )
    payload = base._apply_common_model_overrides(
        payload,
        args=args,
        run_name=run_name,
        wandb=not args.no_wandb,
    )
    return base.ComparisonJob(
        family="FlexTPP",
        model_name="FlexTPP",
        variant="_".join(
            variant_part
            for variant_part in (
                f"{order}_enhanced" if order == "ST" and mark_schema == "enhanced" else order,
                "no_conditioning" if conditioning_mode == "no_conditioning" else "",
            )
            if variant_part
        ),
        run_name=run_name,
        config_payload=payload,
        module_name="HTAMP.prediction.predictor.delivery_tpp_predictor",
    )


def _build_easy_job_payload(
    *,
    base_payload: Mapping[str, Any],
    args: argparse.Namespace,
    run_prefix: str,
    model_id: str,
    mark_schema: str,
    separate_dataset_dir: bool,
) -> base.ComparisonJob:
    schema_suffix = _easy_schema_suffix(mark_schema)
    run_name = f"{run_prefix}_easy_tpp_{base._safe_name(model_id)}_{schema_suffix}"
    payload = base._apply_dataset_training_flags(
        _apply_easy_mark_schema(
            base_payload,
            mark_schema=mark_schema,
            separate_dataset_dir=separate_dataset_dir,
        ),
        use_prepared_dataset=args.prepare_datasets,
    )
    payload["model_config"]["model_id"] = model_id
    payload = base._apply_architecture_defaults(
        payload,
        defaults=base.EASY_TPP_MODEL_DEFAULTS.get(model_id, {}),
        enabled=not args.no_model_defaults,
    )
    if args.easy_max_events_per_sequence is not None:
        payload.setdefault("dataset_config", {})["max_events_per_sequence"] = int(
            args.easy_max_events_per_sequence
        )
    payload = base._apply_sequence_cap_dataset_dir(payload)
    payload = base._apply_common_model_overrides(
        payload,
        args=args,
        run_name=run_name,
        wandb=not args.no_wandb,
    )
    return base.ComparisonJob(
        family="EasyTPP",
        model_name=model_id,
        variant=f"{model_id}_{mark_schema}",
        run_name=run_name,
        config_payload=payload,
        module_name="HTAMP.prediction.predictor.delivery_easy_tpp_predictor",
    )


def _build_multittpp_job_payload(
    *,
    base_payload: Mapping[str, Any],
    args: argparse.Namespace,
    run_prefix: str,
    model_name: str,
    mark_schema: str,
    separate_dataset_dir: bool,
) -> base.ComparisonJob:
    schema_suffix = _multittpp_schema_suffix(mark_schema)
    run_name = f"{run_prefix}_multittpp_{base._safe_name(model_name)}_{schema_suffix}"
    payload = base._apply_dataset_training_flags(
        _apply_multittpp_mark_schema(
            base_payload,
            mark_schema=mark_schema,
            separate_dataset_dir=separate_dataset_dir,
        ),
        use_prepared_dataset=args.prepare_datasets,
    )
    payload["model_config"]["model_name"] = model_name
    payload = base._apply_architecture_defaults(
        payload,
        defaults=base.MULTITTPP_MODEL_DEFAULTS.get(model_name, {}),
        enabled=not args.no_model_defaults,
    )
    payload = base._apply_common_model_overrides(
        payload,
        args=args,
        run_name=run_name,
        wandb=not args.no_wandb,
    )
    return base.ComparisonJob(
        family="MultiTTPP",
        model_name=model_name,
        variant=f"{model_name}_{mark_schema}",
        run_name=run_name,
        config_payload=payload,
        module_name="HTAMP.prediction.predictor.delivery_multittpp_predictor",
    )


def _build_jobs(args: argparse.Namespace, run_prefix: str) -> list[base.ComparisonJob]:
    flex_payload = base._load_json(args.flex_config_path)
    easy_payload = base._load_json(args.easy_config_path)
    multittpp_payload = base._load_json(args.multittpp_config_path)
    jobs: list[base.ComparisonJob] = []

    if not args.skip_flex:
        flex_orders = base._parse_csv_strings(args.flex_orders, allowed=base.DEFAULT_FLEX_ORDERS)
        flex_st_mark_schemas = base._parse_flex_st_mark_schemas(args.flex_st_mark_schemas)
        flex_conditioning_modes = base._parse_flex_conditioning_modes(args.flex_conditioning_modes)
        for order in flex_orders:
            order_mark_schemas = flex_st_mark_schemas if order == "ST" else ("standard",)
            for mark_schema in order_mark_schemas:
                for conditioning_mode in flex_conditioning_modes:
                    jobs.append(
                        _build_flex_job_payload(
                            base_payload=flex_payload,
                            args=args,
                            run_prefix=run_prefix,
                            order=order,
                            mark_schema=mark_schema,
                            conditioning_mode=conditioning_mode,
                        )
                    )

    if not args.skip_easy:
        easy_models = (
            base.SUPPORTED_EASY_TPP_MODELS
            if args.easy_models.strip().lower() == "all"
            else base._parse_csv_strings(args.easy_models, allowed=base.SUPPORTED_EASY_TPP_MODELS)
        )
        easy_mark_schemas = base._parse_easy_mark_schemas(args.easy_mark_schemas)
        separate_easy_dataset_dir = len(easy_mark_schemas) > 1
        for mark_schema in easy_mark_schemas:
            for model_id in easy_models:
                jobs.append(
                    _build_easy_job_payload(
                        base_payload=easy_payload,
                        args=args,
                        run_prefix=run_prefix,
                        model_id=model_id,
                        mark_schema=mark_schema,
                        separate_dataset_dir=separate_easy_dataset_dir,
                    )
                )

    if not args.skip_multittpp:
        multittpp_models = (
            base.SUPPORTED_MULTITTPP_MODELS
            if args.multittpp_models.strip().lower() == "all"
            else base._parse_csv_strings(args.multittpp_models, allowed=base.SUPPORTED_MULTITTPP_MODELS)
        )
        multittpp_mark_schemas = base._parse_multittpp_mark_schemas(args.multittpp_mark_schemas)
        separate_multittpp_dataset_dir = len(multittpp_mark_schemas) > 1
        for mark_schema in multittpp_mark_schemas:
            for model_name in multittpp_models:
                jobs.append(
                    _build_multittpp_job_payload(
                        base_payload=multittpp_payload,
                        args=args,
                        run_prefix=run_prefix,
                        model_name=model_name,
                        mark_schema=mark_schema,
                        separate_dataset_dir=separate_multittpp_dataset_dir,
                    )
                )

    if not jobs:
        raise ValueError(
            "No jobs were selected. Check --skip_flex/--skip_easy/--skip_multittpp settings."
        )
    return jobs


def _selected_flex_dataset_variants(args: argparse.Namespace) -> tuple[tuple[str, str], ...]:
    flex_orders = base._parse_csv_strings(args.flex_orders, allowed=base.DEFAULT_FLEX_ORDERS)
    flex_st_mark_schemas = base._parse_flex_st_mark_schemas(args.flex_st_mark_schemas)
    flex_conditioning_modes = base._parse_flex_conditioning_modes(args.flex_conditioning_modes)
    variants: list[tuple[str, str]] = []

    if "STP" in flex_orders or ("ST" in flex_orders and "standard" in flex_st_mark_schemas):
        variants.extend(("standard", conditioning_mode) for conditioning_mode in flex_conditioning_modes)
    if "ST" in flex_orders and "enhanced" in flex_st_mark_schemas:
        variants.extend(("enhanced", conditioning_mode) for conditioning_mode in flex_conditioning_modes)
    return tuple(variants)


DATASET_PREPARE_MODULE_BY_FAMILY = {
    "FlexTPP": "HTAMP.prediction.data_provider.delivery_tpp_dataset",
    "EasyTPP": "HTAMP.prediction.data_provider.delivery_easy_tpp_dataset",
    "MultiTTPP": "HTAMP.prediction.data_provider.delivery_multittpp_dataset",
}


def _prepare_datasets(
    args: argparse.Namespace,
    *,
    jobs: list[base.ComparisonJob],
    run_prefix: str,
) -> None:
    if args.skip_flex and args.skip_easy and args.skip_multittpp:
        LOGGER.info("Dataset pre-build skipped because every model family was skipped")
        return
    if not args.prepare_datasets:
        LOGGER.info("Dataset pre-build is disabled; training jobs will use config workflow flags")
        return

    base._prepare_datasets_for_jobs(
        args,
        jobs=jobs,
        run_prefix=run_prefix,
        module_by_family=DATASET_PREPARE_MODULE_BY_FAMILY,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = base.build_parser()
    parser.prog = "DeliveryTPPModelComparison"
    parser.description = (
        "Train selected medication-delivery EasyTPP, FlexTPP, and MultiTTPP "
        "models across multiple GPUs and summarize their metrics."
    )
    parser.set_defaults(
        easy_config_path=DEFAULT_EASY_CONFIG_PATH,
        flex_config_path=DEFAULT_FLEX_CONFIG_PATH,
        multittpp_config_path=DEFAULT_MULTITTPP_CONFIG_PATH,
        wandb_project="delivery_tpp_comparison",
    )
    return parser


def _patch_base_runner_paths() -> None:
    base._temp_config_dir = _temp_config_dir
    base._comparison_log_dir = _comparison_log_dir
    base._comparison_log_path = _comparison_log_path


def main() -> int:
    _patch_base_runner_paths()
    args = build_parser().parse_args()
    run_prefix = _run_prefix(args.run_prefix)
    controller_log_path = base.configure_comparison_logging(args, run_prefix=run_prefix)
    summary_path = _summary_path(args, run_prefix)
    jobs = _build_jobs(args, run_prefix=run_prefix)

    LOGGER.info("Starting delivery TPP model comparison")
    LOGGER.info("Controller log path: %s", controller_log_path)
    LOGGER.info("W&B project: %s", args.wandb_project)
    LOGGER.info("W&B group: %s", args.wandb_group or run_prefix)
    LOGGER.info("W&B enabled: %s", not args.no_wandb)
    LOGGER.info("Per-architecture defaults: %s", not args.no_model_defaults)
    LOGGER.info("Summary path: %s", summary_path)
    base._log_job_plan(jobs)

    if args.dry_run:
        LOGGER.info("Dry run requested; no datasets or models were trained")
        return 0

    _prepare_datasets(args, jobs=jobs, run_prefix=run_prefix)
    exit_code = base._run_jobs(
        args,
        jobs=jobs,
        run_prefix=run_prefix,
        summary_path=summary_path,
    )
    LOGGER.info("Delivery TPP comparison summary saved to %s", summary_path)
    if exit_code == 0:
        LOGGER.info("All selected delivery TPP comparison jobs completed successfully")
    else:
        LOGGER.error("At least one delivery TPP comparison job failed")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
