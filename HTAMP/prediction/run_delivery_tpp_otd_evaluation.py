from __future__ import annotations

import argparse
import datetime
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from HTAMP.prediction import run_vital_sign_tpp_otd_evaluation as base
from HTAMP.prediction.configs.delivery_tpp_config import (
    DeliveryEasyTPPTrainingConfig,
    DeliveryMultiTTPPTrainingConfig,
    DeliveryTPPTrainingConfig,
)
from HTAMP.prediction.data_provider.delivery_easy_tpp_dataset import (
    build_delivery_easy_tpp_dataset_bundle,
)
from HTAMP.prediction.data_provider.delivery_multittpp_dataset import (
    build_delivery_multittpp_dataset_bundle,
)
from HTAMP.prediction.data_provider.delivery_tpp_dataset import (
    DELIVERY_TASK_NAME,
    MEDICATION_CODE_PROPERTY,
    build_delivery_tpp_dataset_bundle,
)
from HTAMP.prediction.data_provider.vital_sign_tpp_dataset import EOS_EVENT_TYPE_NAME

DEFAULT_COMPARISON_SUMMARY_GLOB = "data/prediction/delivery_tpp_comparison/*_summary.csv"
DEFAULT_OUTPUT_DIR = "data/prediction/delivery_tpp_otd_evaluation"
DEFAULT_EASY_CONFIG_PATH = (
    "HTAMP/prediction/configs/config_files/prediction/"
    "delivery_easy_tpp_training.json"
)


def _finite_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric_value):
        return None
    return numeric_value


class DeliveryMarkEncoder:
    """Map delivery raw/model events into the canonical medication mark namespace."""

    def __init__(self, metadata: Mapping[str, Any]) -> None:
        self.metadata = dict(metadata)
        self.mark_names = [
            str(mark_name)
            for mark_name in self.metadata.get("mark_names", self.metadata.get("event_types", []))
        ]
        self.mark_name_set = set(self.mark_names)
        source_metadata = dict(self.metadata.get("source_dataset_metadata", {}))
        self.source_metadata = source_metadata
        self.medication_code_vocab = [
            str(code)
            for code in source_metadata.get("medication_code_vocab", [])
        ]
        med_code_to_event_type = {
            str(key): str(value)
            for key, value in dict(source_metadata.get("med_code_to_event_type", {})).items()
        }
        self.medication_index_to_mark = {
            index: med_code_to_event_type.get(str(med_code), DELIVERY_TASK_NAME)
            for index, med_code in enumerate(self.medication_code_vocab)
        }

    def base_task(self, mark_name: str) -> str:
        if mark_name == EOS_EVENT_TYPE_NAME:
            return EOS_EVENT_TYPE_NAME
        if mark_name == DELIVERY_TASK_NAME or str(mark_name).startswith(f"{DELIVERY_TASK_NAME}__"):
            return DELIVERY_TASK_NAME
        return str(mark_name).split("__", 1)[0]

    def label_event(self, *, task_name: str, properties: Mapping[str, Any]) -> str | None:
        if task_name == EOS_EVENT_TYPE_NAME:
            return EOS_EVENT_TYPE_NAME
        if task_name in self.mark_name_set:
            return task_name

        medication_code_index = _finite_float_or_none(properties.get(MEDICATION_CODE_PROPERTY))
        if medication_code_index is not None:
            mark_name = self.medication_index_to_mark.get(int(round(medication_code_index)))
            if mark_name in self.mark_name_set:
                return mark_name

        return DELIVERY_TASK_NAME if DELIVERY_TASK_NAME in self.mark_name_set else None


def _latest_comparison_summary() -> Path | None:
    candidates = [
        path
        for path in base._repo_root().glob(DEFAULT_COMPARISON_SUMMARY_GLOB)
        if path.is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _coerce_comparison_summary_path(raw_path: str | None) -> Path:
    if raw_path:
        resolved = base._resolve_repo_relative_path(raw_path)
        if resolved is None:
            raise ValueError("comparison_summary_path could not be resolved.")
        return resolved

    latest = _latest_comparison_summary()
    if latest is None:
        raise FileNotFoundError(
            "No delivery comparison summary was found. Provide --comparison_summary_path."
        )
    return latest


def _demand_child_args(
    *,
    parent_argv: Sequence[str],
    demand_level: str,
    device_arg: str,
    output_dir: Path,
) -> list[str]:
    parent_only_options = {
        "--run_demand_strata",
        "--demand_levels",
        "--demand_gpu_ids",
        "--demand_level",
        "--device",
        "--output_dir",
    }
    child_args = base._strip_option(list(parent_argv), parent_only_options)
    child_args.extend(
        [
            "--demand_level",
            demand_level,
            "--device",
            device_arg,
            "--output_dir",
            str(output_dir),
        ]
    )
    return child_args


def _run_demand_strata(args: argparse.Namespace, parent_argv: Sequence[str]) -> int:
    demand_levels = base._parse_csv_list(args.demand_levels, allowed_values=base.DEMAND_LEVELS)
    gpu_args = [
        base._normalize_gpu_device_arg(gpu_id)
        for gpu_id in base._parse_csv_list(args.demand_gpu_ids)
    ]
    if len(gpu_args) < len(demand_levels):
        raise ValueError(
            f"--run_demand_strata needs at least {len(demand_levels)} GPU ids, "
            f"but --demand_gpu_ids provided {len(gpu_args)}."
        )
    if len(set(gpu_args[: len(demand_levels)])) != len(demand_levels):
        raise ValueError("--demand_gpu_ids must assign a distinct GPU to each demand level.")

    base_output_dir = base._resolve_repo_relative_path(args.output_dir)
    if base_output_dir is None:
        base_output_dir = base._repo_root() / DEFAULT_OUTPUT_DIR
    base_output_dir.mkdir(parents=True, exist_ok=True)

    processes: list[tuple[str, str, Path, subprocess.Popen[Any], float]] = []
    for demand_level, device_arg in zip(demand_levels, gpu_args):
        child_output_dir = base_output_dir / demand_level
        child_output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "HTAMP.prediction.run_delivery_tpp_otd_evaluation",
            *_demand_child_args(
                parent_argv=parent_argv,
                demand_level=demand_level,
                device_arg=device_arg,
                output_dir=child_output_dir,
            ),
        ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        base._log(
            f"Launching delivery demand_level='{demand_level}' on {device_arg}; "
            f"outputs will be written to {child_output_dir}."
        )
        process = subprocess.Popen(command, cwd=base._repo_root(), env=env)
        processes.append((demand_level, device_arg, child_output_dir, process, time.perf_counter()))

    summary_rows: list[dict[str, Any]] = []
    return_code = 0
    for demand_level, device_arg, child_output_dir, process, started_at in processes:
        child_return_code = process.wait()
        duration_seconds = round(time.perf_counter() - started_at, 3)
        status = "success" if child_return_code == 0 else "failed"
        if child_return_code != 0:
            return_code = 1
        base._log(
            f"Finished delivery demand_level='{demand_level}' on {device_arg} with "
            f"status={status} in {base._format_duration(duration_seconds)}."
        )
        summary_rows.append(
            {
                "demand_level": demand_level,
                "device": device_arg,
                "status": status,
                "returncode": int(child_return_code),
                "duration_seconds": duration_seconds,
                "output_dir": str(child_output_dir),
                "summary_path": str(child_output_dir / "otd_summary.csv"),
                "detail_path": str(child_output_dir / "otd_prefix_samples.csv"),
            }
        )

    strata_summary_path = base_output_dir / "demand_strata_summary.csv"
    base._write_csv_rows(strata_summary_path, summary_rows)
    base._log(f"Delivery demand-stratified OTD launch summary saved to {strata_summary_path}")
    return return_code


def _patch_base_evaluator() -> None:
    base.DEFAULT_COMPARISON_SUMMARY_GLOB = DEFAULT_COMPARISON_SUMMARY_GLOB
    base.DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR
    base.DiscreteMarkEncoder = DeliveryMarkEncoder
    base.VitalSignEasyTPPTrainingConfig = DeliveryEasyTPPTrainingConfig
    base.VitalSignTPPTrainingConfig = DeliveryTPPTrainingConfig
    base.VitalSignMultiTTPPTrainingConfig = DeliveryMultiTTPPTrainingConfig
    base.build_vital_sign_easy_tpp_dataset_bundle = build_delivery_easy_tpp_dataset_bundle
    base.build_vital_sign_tpp_dataset_bundle = build_delivery_tpp_dataset_bundle
    base.build_vital_sign_multittpp_dataset_bundle = build_delivery_multittpp_dataset_bundle
    base._latest_comparison_summary = _latest_comparison_summary
    base._coerce_comparison_summary_path = _coerce_comparison_summary_path


def build_parser() -> argparse.ArgumentParser:
    _patch_base_evaluator()
    parser = base.build_parser()
    parser.prog = "DeliveryTPPOTDEvaluation"
    parser.description = (
        "Evaluate trained EasyTPP, FlexTPP, and MultiTTPP medication delivery "
        "request models with Monte Carlo type/time Optimal Transport Distance."
    )
    parser.set_defaults(
        easy_config_path=DEFAULT_EASY_CONFIG_PATH,
        output_dir=DEFAULT_OUTPUT_DIR,
        wandb_project="delivery_tpp_comparison",
    )
    return parser


def main() -> int:
    _patch_base_evaluator()
    args = build_parser().parse_args()
    if args.run_demand_strata:
        return _run_demand_strata(args=args, parent_argv=sys.argv[1:])

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    random.seed(int(args.seed))

    comparison_summary_path = _coerce_comparison_summary_path(args.comparison_summary_path)
    output_dir = base._resolve_repo_relative_path(args.output_dir)
    if output_dir is None:
        output_dir = base._repo_root() / DEFAULT_OUTPUT_DIR
    device = base._device_from_args(args)
    base._log(f"Using delivery comparison summary: {comparison_summary_path}")
    base._log(f"Using device: {device}")

    model_specs = base._load_model_specs(
        comparison_summary_path=comparison_summary_path,
        stf_log_dir=args.stf_log_dir,
        selected_runs=base._parse_selected_runs(args.selected_runs),
    )
    context = base._build_evaluation_context(args=args, model_specs=model_specs)
    otd_config = base._make_otd_config(args, default_tau=context.default_tau)
    total_prefixes, samples_per_model, rollout_event_budget_per_model = base._prefix_work_for_records(
        args=args,
        records=context.easy_records,
        mark_encoder=context.mark_encoder,
    )
    base._log(
        f"Prepared delivery OTD evaluation for {len(model_specs)} models on "
        f"{len(context.easy_records)} canonical {args.split} sequences: "
        f"{total_prefixes} prefixes/model, {samples_per_model} OTD samples/model, "
        f"{rollout_event_budget_per_model} rollout-event budget/model."
    )

    all_detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for model_index, spec in enumerate(model_specs, start=1):
        detail_rows, summary_row = base._evaluate_model(
            spec=spec,
            model_index=model_index,
            total_models=len(model_specs),
            context=context,
            args=args,
            device=device,
            otd_config=otd_config,
        )
        all_detail_rows.extend(detail_rows)
        summary_rows.append(summary_row)

    detail_path, summary_path, metadata_path = base._write_outputs(
        output_dir=output_dir,
        detail_rows=all_detail_rows,
        summary_rows=summary_rows,
        config_payload={
            "comparison_summary_path": str(comparison_summary_path),
            "output_dir": str(output_dir),
            "split": args.split,
            "demand_level": args.demand_level,
            "num_samples": args.num_samples,
            "max_future_events": args.max_future_events,
            "min_prefix_events": args.min_prefix_events,
            "prefix_stride": args.prefix_stride,
            "max_prefixes_per_sequence": args.max_prefixes_per_sequence,
            "max_sequences": args.max_sequences,
            "sequence_subset_strategy": args.sequence_subset_strategy,
            "seed": args.seed,
            "otd_config": otd_config,
            "model_count": len(model_specs),
            "detail_row_count": len(all_detail_rows),
        },
    )
    base._log(f"Delivery OTD detail rows saved to {detail_path}")
    base._log(f"Delivery OTD summary saved to {summary_path}")
    base._log(f"Delivery OTD metadata saved to {metadata_path}")

    plot_paths: list[Path] = []
    if not args.skip_plots:
        plot_paths = base._plot_results(
            output_dir=output_dir,
            detail_path=detail_path,
            summary_path=summary_path,
        )
        for plot_path in plot_paths:
            base._log(f"Plot saved to {plot_path}")

    if args.wandb:
        base._log_wandb(
            args=args,
            output_paths=[detail_path, summary_path, metadata_path],
            plot_paths=plot_paths,
            summary_rows=summary_rows,
            artifact_name="delivery_tpp_otd_evaluation",
        )
    return 0 if all(row.get("status") == "success" for row in summary_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
