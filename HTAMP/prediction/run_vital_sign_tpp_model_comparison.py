from __future__ import annotations

import argparse
import copy
import csv
import datetime
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

DEFAULT_EASY_CONFIG_PATH = (
    "HTAMP/prediction/configs/config_files/prediction/"
    "vital_sign_easy_tpp_training.json"
)
DEFAULT_FLEX_CONFIG_PATH = (
    "HTAMP/prediction/configs/config_files/prediction/"
    "vital_sign_tpp_training.json"
)
SUPPORTED_EASY_TPP_MODELS = (
    "NHP",
    "AttNHP",
    "THP",
    "SAHP",
    "RMTPP",
    "FullyNN",
    "IntensityFree",
    "ODETPP",
    "ANHN",
    "S2P2",
    "WSMTHP",
)
DEFAULT_FLEX_ORDERS = ("ST", "STP")
DEFAULT_SELECTION_METRIC = "val_nll"
METRICS_SUMMARY_FILENAME = "metrics_summary.json"
BASE_SUMMARY_FIELDS = [
    "family",
    "model_name",
    "variant",
    "run_name",
    "gpu_id",
    "status",
    "returncode",
    "start_time",
    "end_time",
    "duration_seconds",
    "selection_metric",
    "selection_metric_value",
    "selection_rank",
    "metrics_summary_path",
    "best_checkpoint_path",
    "best_checkpoint_score",
]


@dataclass
class ComparisonJob:
    family: str
    model_name: str
    variant: str
    run_name: str
    config_payload: dict[str, Any]
    module_name: str


@dataclass
class RunningJob:
    job: ComparisonJob
    process: subprocess.Popen
    temp_config_path: Path
    gpu_id: int | None
    start_time: datetime.datetime
    wall_clock_start: float


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_repo_relative_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return _repo_root() / path


def _load_json(path_str: str | Path) -> dict[str, Any]:
    path = _resolve_repo_relative_path(path_str)
    with path.open("r", encoding="utf-8") as config_file:
        payload = json.load(config_file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected '{path}' to contain a JSON object.")
    return payload


def _deep_merge_dicts(base: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(base=merged[key], updates=value)
            continue
        merged[key] = copy.deepcopy(value)
    return merged


def _parse_csv_strings(raw_value: str, *, allowed: Sequence[str] | None = None) -> tuple[str, ...]:
    values = tuple(
        item.strip()
        for item in str(raw_value).split(",")
        if item.strip()
    )
    if not values:
        raise ValueError("Expected at least one comma-separated value.")

    if allowed is not None:
        allowed_set = set(allowed)
        invalid_values = [value for value in values if value not in allowed_set]
        if invalid_values:
            raise ValueError(
                f"Unsupported value(s) {invalid_values}. Expected values from {tuple(allowed)}."
            )
    return values


def _parse_gpu_ids(raw_value: str) -> tuple[int, ...]:
    if not str(raw_value).strip():
        return ()
    gpu_ids = []
    for raw_gpu_id in str(raw_value).split(","):
        raw_gpu_id = raw_gpu_id.strip()
        if not raw_gpu_id:
            continue
        gpu_ids.append(int(raw_gpu_id))
    return tuple(gpu_ids)


def _run_prefix(raw_prefix: str | None) -> str:
    if raw_prefix:
        return raw_prefix
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"vital_sign_tpp_comparison_{timestamp}"


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value.lower()).strip("_")


def _log_dir(args: argparse.Namespace) -> Path:
    return Path(args.stf_log_dir or os.getenv("STF_LOG_DIR", "./data/STF_LOG_DIR"))


def _temp_config_dir(args: argparse.Namespace) -> Path:
    return _log_dir(args) / "temp_configs" / "vital_sign_tpp_comparison"


def _default_summary_path(run_prefix: str) -> Path:
    return _repo_root() / "data" / "prediction" / "vital_sign_tpp_comparison" / (
        f"{_safe_name(run_prefix)}_summary.csv"
    )


def _summary_path(args: argparse.Namespace, run_prefix: str) -> Path:
    if args.summary_path:
        return _resolve_repo_relative_path(args.summary_path)
    return _default_summary_path(run_prefix)


def _metrics_summary_path(args: argparse.Namespace, run_name: str) -> Path:
    return _log_dir(args) / run_name / METRICS_SUMMARY_FILENAME


def _write_temp_config(
    args: argparse.Namespace,
    *,
    payload: Mapping[str, Any],
    run_name: str,
) -> Path:
    temp_config_dir = _temp_config_dir(args)
    temp_config_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        prefix=f"{_safe_name(run_name)[:48]}_",
        dir=temp_config_dir,
        delete=False,
    ) as config_file:
        json.dump(payload, config_file, indent=2)
        return Path(config_file.name)


def _apply_dataset_training_flags(
    payload: dict[str, Any],
    *,
    use_prepared_dataset: bool,
) -> dict[str, Any]:
    updated_payload = copy.deepcopy(payload)
    dataset_config = updated_payload.setdefault("dataset_config", {})
    if use_prepared_dataset:
        dataset_config["use_saved_dataset"] = True
        dataset_config["preprocess_data"] = False
        dataset_config["save_data"] = False
    return updated_payload


def _apply_common_model_overrides(
    payload: dict[str, Any],
    *,
    args: argparse.Namespace,
    run_name: str,
    wandb: bool,
) -> dict[str, Any]:
    updated_payload = copy.deepcopy(payload)
    model_config = updated_payload.setdefault("model_config", {})
    model_config["run_name"] = run_name
    model_config["wandb"] = bool(wandb)
    model_config["accelerator"] = args.accelerator
    model_config["devices"] = 1
    model_config["strategy"] = args.strategy

    optional_overrides = {
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "learning_rate": args.learning_rate,
    }
    for field_name, field_value in optional_overrides.items():
        if field_value is not None:
            model_config[field_name] = field_value
    return updated_payload


def _build_flex_job_payload(
    *,
    base_payload: Mapping[str, Any],
    args: argparse.Namespace,
    run_prefix: str,
    order: str,
) -> ComparisonJob:
    run_name = f"{run_prefix}_flex_tpp_{order.lower()}"
    payload = _apply_dataset_training_flags(
        copy.deepcopy(dict(base_payload)),
        use_prepared_dataset=args.prepare_datasets,
    )
    payload = _apply_common_model_overrides(
        payload,
        args=args,
        run_name=run_name,
        wandb=not args.no_wandb,
    )
    payload["model_config"]["order"] = order
    return ComparisonJob(
        family="FlexTPP",
        model_name="FlexTPP",
        variant=order,
        run_name=run_name,
        config_payload=payload,
        module_name="HTAMP.prediction.vital_sign_tpp_predictor",
    )


def _build_easy_job_payload(
    *,
    base_payload: Mapping[str, Any],
    args: argparse.Namespace,
    run_prefix: str,
    model_id: str,
) -> ComparisonJob:
    run_name = f"{run_prefix}_easy_tpp_{_safe_name(model_id)}"
    payload = _apply_dataset_training_flags(
        copy.deepcopy(dict(base_payload)),
        use_prepared_dataset=args.prepare_datasets,
    )
    payload = _apply_common_model_overrides(
        payload,
        args=args,
        run_name=run_name,
        wandb=not args.no_wandb,
    )
    payload["model_config"]["model_id"] = model_id
    return ComparisonJob(
        family="EasyTPP",
        model_name=model_id,
        variant=model_id,
        run_name=run_name,
        config_payload=payload,
        module_name="HTAMP.prediction.vital_sign_easy_tpp_predictor",
    )


def _build_jobs(args: argparse.Namespace, run_prefix: str) -> list[ComparisonJob]:
    flex_payload = _load_json(args.flex_config_path)
    easy_payload = _load_json(args.easy_config_path)
    jobs: list[ComparisonJob] = []

    if not args.skip_flex:
        for order in _parse_csv_strings(args.flex_orders, allowed=DEFAULT_FLEX_ORDERS):
            jobs.append(
                _build_flex_job_payload(
                    base_payload=flex_payload,
                    args=args,
                    run_prefix=run_prefix,
                    order=order,
                )
            )

    if not args.skip_easy:
        easy_models = (
            SUPPORTED_EASY_TPP_MODELS
            if args.easy_models.strip().lower() == "all"
            else _parse_csv_strings(args.easy_models, allowed=SUPPORTED_EASY_TPP_MODELS)
        )
        for model_id in easy_models:
            jobs.append(
                _build_easy_job_payload(
                    base_payload=easy_payload,
                    args=args,
                    run_prefix=run_prefix,
                    model_id=model_id,
                )
            )

    if not jobs:
        raise ValueError("No jobs were selected. Check --skip_flex/--skip_easy settings.")
    return jobs


def _build_prepare_payload(base_payload: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(base_payload))
    dataset_config = payload.setdefault("dataset_config", {})
    dataset_config["use_saved_dataset"] = False
    dataset_config["preprocess_data"] = True
    dataset_config["save_data"] = True
    payload.setdefault("model_config", {})["wandb"] = False
    return payload


def _build_process_env(
    args: argparse.Namespace,
    *,
    gpu_id: int | None,
    run_prefix: str,
    job_type: str,
) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["WANDB_PROJECT"] = args.wandb_project
    env["WANDB_GROUP"] = args.wandb_group or run_prefix
    env["WANDB_JOB_TYPE"] = job_type
    if args.stf_log_dir:
        env["STF_LOG_DIR"] = str(_log_dir(args))
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return env


def _run_dataset_prepare(
    args: argparse.Namespace,
    *,
    label: str,
    module_name: str,
    payload: Mapping[str, Any],
    run_prefix: str,
) -> None:
    temp_config_path = _write_temp_config(
        args,
        payload=payload,
        run_name=f"{run_prefix}_{label}_dataset_prepare",
    )
    command = [
        sys.executable,
        "-m",
        module_name,
        "--config_path",
        str(temp_config_path),
    ]
    print(f"Preparing {label} dataset with command: {' '.join(command)}")
    try:
        result = subprocess.run(
            command,
            cwd=_repo_root(),
            env=_build_process_env(
                args,
                gpu_id=None,
                run_prefix=run_prefix,
                job_type="dataset_prepare",
            ),
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"{label} dataset preparation failed with return code {result.returncode}."
            )
    finally:
        temp_config_path.unlink(missing_ok=True)


def _prepare_datasets(args: argparse.Namespace, run_prefix: str) -> None:
    if args.skip_flex and args.skip_easy:
        return
    if not args.prepare_datasets:
        print("Dataset pre-build is disabled; training jobs will use config workflow flags.")
        return

    if not args.skip_flex:
        _run_dataset_prepare(
            args,
            label="FlexTPP",
            module_name="HTAMP.prediction.data_provider.vital_sign_tpp_dataset",
            payload=_build_prepare_payload(_load_json(args.flex_config_path)),
            run_prefix=run_prefix,
        )

    if not args.skip_easy:
        _run_dataset_prepare(
            args,
            label="EasyTPP",
            module_name="HTAMP.prediction.data_provider.vital_sign_easy_tpp_dataset",
            payload=_build_prepare_payload(_load_json(args.easy_config_path)),
            run_prefix=run_prefix,
        )


def _read_metrics_summary(args: argparse.Namespace, run_name: str) -> dict[str, Any]:
    metrics_path = _metrics_summary_path(args, run_name=run_name)
    if not metrics_path.exists():
        return {}
    with metrics_path.open("r", encoding="utf-8") as metrics_file:
        payload = json.load(metrics_file)
    return payload if isinstance(payload, dict) else {}


def _is_scalar_metric(value: Any) -> bool:
    return isinstance(value, (int, float, str)) or value is None


def _extract_metrics(metrics_summary: Mapping[str, Any]) -> dict[str, Any]:
    extracted = {
        "metrics_summary_path": str(metrics_summary.get("metrics_summary_path", "")),
        "best_checkpoint_path": str(metrics_summary.get("best_checkpoint_path", "")),
        "best_checkpoint_score": metrics_summary.get("best_checkpoint_score", ""),
    }
    for section_name in ("validation_metrics", "test_metrics"):
        section_payload = metrics_summary.get(section_name, {})
        if not isinstance(section_payload, Mapping):
            continue
        for metric_name, metric_value in section_payload.items():
            if _is_scalar_metric(metric_value):
                extracted[str(metric_name)] = metric_value
    return extracted


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool) or value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_higher_is_better(metric_name: str) -> bool:
    metric_name = metric_name.lower()
    return any(
        token in metric_name
        for token in ("accuracy", "auc", "f1", "precision", "recall")
    )


def _apply_selection_ranking(rows: list[dict[str, Any]], selection_metric: str) -> None:
    ranked_rows: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        row["selection_metric"] = selection_metric
        metric_value = _coerce_float(row.get(selection_metric))
        row["selection_metric_value"] = metric_value if metric_value is not None else ""
        row["selection_rank"] = ""
        if row.get("status") == "success" and metric_value is not None:
            ranked_rows.append((metric_value, row))

    ranked_rows.sort(
        key=lambda item: item[0],
        reverse=_metric_higher_is_better(selection_metric),
    )
    for rank, (_, row) in enumerate(ranked_rows, start=1):
        row["selection_rank"] = rank


def _write_summary(summary_path: Path, rows: list[dict[str, Any]]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    dynamic_fields = sorted(
        {
            field_name
            for row in rows
            for field_name in row
            if field_name not in BASE_SUMMARY_FIELDS
        }
    )
    fieldnames = BASE_SUMMARY_FIELDS + dynamic_fields
    with summary_path.open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _launch_job(
    args: argparse.Namespace,
    *,
    job: ComparisonJob,
    gpu_id: int | None,
    run_prefix: str,
) -> RunningJob:
    temp_config_path = _write_temp_config(
        args,
        payload=job.config_payload,
        run_name=job.run_name,
    )
    metrics_path = _metrics_summary_path(args, run_name=job.run_name)
    metrics_path.unlink(missing_ok=True)
    command = [
        sys.executable,
        "-m",
        job.module_name,
        "--config_path",
        str(temp_config_path),
    ]
    gpu_label = f"GPU {gpu_id}" if gpu_id is not None else "un-pinned GPU/CPU"
    print(f"Starting {job.run_name} on {gpu_label}: {' '.join(command)}")
    start_time = datetime.datetime.now()
    process = subprocess.Popen(
        command,
        cwd=_repo_root(),
        env=_build_process_env(
            args,
            gpu_id=gpu_id,
            run_prefix=run_prefix,
            job_type=job.family.lower(),
        ),
    )
    return RunningJob(
        job=job,
        process=process,
        temp_config_path=temp_config_path,
        gpu_id=gpu_id,
        start_time=start_time,
        wall_clock_start=time.perf_counter(),
    )


def _finalize_job(
    args: argparse.Namespace,
    *,
    running_job: RunningJob,
) -> dict[str, Any]:
    returncode = running_job.process.wait()
    duration_seconds = time.perf_counter() - running_job.wall_clock_start
    end_time = datetime.datetime.now()
    running_job.temp_config_path.unlink(missing_ok=True)
    status = "success" if returncode == 0 else "failed"
    metrics = _extract_metrics(
        _read_metrics_summary(args, run_name=running_job.job.run_name)
    )
    print(
        f"Finished {running_job.job.run_name} with status {status} "
        f"in {duration_seconds:.1f}s."
    )
    return {
        "family": running_job.job.family,
        "model_name": running_job.job.model_name,
        "variant": running_job.job.variant,
        "run_name": running_job.job.run_name,
        "gpu_id": running_job.gpu_id if running_job.gpu_id is not None else "",
        "status": status,
        "returncode": returncode,
        "start_time": running_job.start_time.isoformat(timespec="seconds"),
        "end_time": end_time.isoformat(timespec="seconds"),
        "duration_seconds": round(duration_seconds, 2),
        "selection_metric": "",
        "selection_metric_value": "",
        "selection_rank": "",
        **metrics,
    }


def _cancel_running_jobs(running_jobs: list[RunningJob]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for running_job in running_jobs:
        if running_job.process.poll() is None:
            running_job.process.terminate()

    for running_job in running_jobs:
        try:
            running_job.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            running_job.process.kill()
            running_job.process.wait()
        running_job.temp_config_path.unlink(missing_ok=True)
        rows.append(
            {
                "family": running_job.job.family,
                "model_name": running_job.job.model_name,
                "variant": running_job.job.variant,
                "run_name": running_job.job.run_name,
                "gpu_id": running_job.gpu_id if running_job.gpu_id is not None else "",
                "status": "cancelled",
                "returncode": running_job.process.returncode,
                "start_time": running_job.start_time.isoformat(timespec="seconds"),
                "end_time": datetime.datetime.now().isoformat(timespec="seconds"),
                "duration_seconds": "",
                "selection_metric": "",
                "selection_metric_value": "",
                "selection_rank": "",
                "metrics_summary_path": "",
                "best_checkpoint_path": "",
                "best_checkpoint_score": "",
            }
        )
    running_jobs.clear()
    return rows


def _run_jobs(
    args: argparse.Namespace,
    *,
    jobs: Sequence[ComparisonJob],
    run_prefix: str,
    summary_path: Path,
) -> int:
    pending_jobs = list(jobs)
    running_jobs: list[RunningJob] = []
    rows: list[dict[str, Any]] = []
    gpu_ids = list(_parse_gpu_ids(args.gpu_ids))
    max_parallel_runs = max(1, int(args.max_parallel_runs))
    if gpu_ids:
        max_parallel_runs = min(max_parallel_runs, len(gpu_ids))
    available_gpu_ids = list(gpu_ids[:max_parallel_runs])

    while pending_jobs or running_jobs:
        while pending_jobs and len(running_jobs) < max_parallel_runs:
            gpu_id = available_gpu_ids.pop(0) if available_gpu_ids else None
            running_jobs.append(
                _launch_job(
                    args,
                    job=pending_jobs.pop(0),
                    gpu_id=gpu_id,
                    run_prefix=run_prefix,
                )
            )

        time.sleep(1.0)
        finished_indices = [
            index
            for index, running_job in enumerate(running_jobs)
            if running_job.process.poll() is not None
        ]
        if not finished_indices:
            continue

        for finished_index in reversed(finished_indices):
            finished_job = running_jobs.pop(finished_index)
            rows.append(_finalize_job(args, running_job=finished_job))
            _apply_selection_ranking(rows, args.selection_metric)
            _write_summary(summary_path, rows)

            if finished_job.gpu_id is not None:
                available_gpu_ids.append(finished_job.gpu_id)
                available_gpu_ids.sort()

            if rows[-1]["status"] != "success" and args.fail_fast:
                rows.extend(_cancel_running_jobs(running_jobs))
                for pending_job in pending_jobs:
                    rows.append(
                        {
                            "family": pending_job.family,
                            "model_name": pending_job.model_name,
                            "variant": pending_job.variant,
                            "run_name": pending_job.run_name,
                            "gpu_id": "",
                            "status": "cancelled",
                            "returncode": "",
                            "start_time": "",
                            "end_time": "",
                            "duration_seconds": "",
                            "selection_metric": "",
                            "selection_metric_value": "",
                            "selection_rank": "",
                            "metrics_summary_path": "",
                            "best_checkpoint_path": "",
                            "best_checkpoint_score": "",
                        }
                    )
                _apply_selection_ranking(rows, args.selection_metric)
                _write_summary(summary_path, rows)
                return 1

    _apply_selection_ranking(rows, args.selection_metric)
    _write_summary(summary_path, rows)
    return 1 if any(row["status"] == "failed" for row in rows) else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="VitalSignTPPModelComparison",
        description=(
            "Train all selected vital-sign EasyTPP and FlexTPP models across "
            "multiple GPUs and summarize their metrics."
        ),
    )
    parser.add_argument("--easy_config_path", default=DEFAULT_EASY_CONFIG_PATH)
    parser.add_argument("--flex_config_path", default=DEFAULT_FLEX_CONFIG_PATH)
    parser.add_argument(
        "--easy_models",
        default="all",
        help=(
            "Comma-separated EasyTPP model ids, or 'all'. Available: "
            f"{', '.join(SUPPORTED_EASY_TPP_MODELS)}"
        ),
    )
    parser.add_argument(
        "--flex_orders",
        default=",".join(DEFAULT_FLEX_ORDERS),
        help="Comma-separated FlexTPP event orders. Defaults to ST,STP.",
    )
    parser.add_argument("--skip_easy", action="store_true")
    parser.add_argument("--skip_flex", action="store_true")
    parser.add_argument("--gpu_ids", default="0,1,2")
    parser.add_argument("--max_parallel_runs", type=int, default=3)
    parser.add_argument("--run_prefix", default=None)
    parser.add_argument("--summary_path", default=None)
    parser.add_argument("--stf_log_dir", default=None)
    parser.add_argument("--wandb_project", default="vital_sign_tpp_comparison")
    parser.add_argument("--wandb_group", default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--selection_metric", default=DEFAULT_SELECTION_METRIC)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--strategy", default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument(
        "--prepare_datasets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Pre-build and cache FlexTPP/EasyTPP datasets before launching "
            "parallel training jobs."
        ),
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--fail_fast", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_prefix = _run_prefix(args.run_prefix)
    summary_path = _summary_path(args, run_prefix)
    jobs = _build_jobs(args, run_prefix=run_prefix)

    print(f"Prepared {len(jobs)} training job(s).")
    print(f"W&B project: {args.wandb_project}")
    print(f"W&B group: {args.wandb_group or run_prefix}")
    print(f"Summary path: {summary_path}")
    for job in jobs:
        print(f"  - {job.run_name} [{job.family}/{job.variant}]")

    if args.dry_run:
        print("Dry run requested; no datasets or models were trained.")
        return 0

    _prepare_datasets(args, run_prefix=run_prefix)
    exit_code = _run_jobs(
        args,
        jobs=jobs,
        run_prefix=run_prefix,
        summary_path=summary_path,
    )
    print(f"Vital-sign TPP comparison summary saved to {summary_path}")
    if exit_code == 0:
        print("All selected vital-sign TPP comparison jobs completed successfully.")
    else:
        print("At least one vital-sign TPP comparison job failed.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
