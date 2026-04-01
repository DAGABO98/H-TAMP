from __future__ import annotations

import argparse
import csv
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

from HTAMP.prediction.module.request_predictor import build_parser as build_predictor_parser

AVAILABLE_MODELS = ("TimesNet", "TimeMixer", "iTransformer", "PatchTST")
DRIVER_ONLY_ARGS = {"models", "summary_path", "fail_fast", "model_overrides_file"}
DEFAULT_PREDICTOR_RUN_NAME = "TimesNet_medical_requests"
PREDICTOR_ARG_NAMES = {
    action.dest
    for action in build_predictor_parser()._actions
    if action.dest != "help"
}


def build_parser() -> argparse.ArgumentParser:
    parser = build_predictor_parser()
    parser.prog = "RequestModelSweep"
    parser.description = (
        "Run the request predictor across multiple time-series models "
        "and write a summary CSV for comparison."
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=list(AVAILABLE_MODELS),
        choices=AVAILABLE_MODELS,
        help="Models to train. Defaults to all supported request models.",
    )
    parser.add_argument(
        "--summary_path",
        type=str,
        default="data/prediction/request_model_sweep_summary.csv",
        help="CSV file where per-model run status and duration will be saved.",
    )
    parser.add_argument(
        "--fail_fast",
        action="store_true",
        default=False,
        help="Stop the sweep immediately if one model run fails.",
    )
    parser.add_argument(
        "--model_overrides_file",
        type=str,
        default=None,
        help=(
            "Optional JSON file with per-model predictor argument overrides. "
            "Expected shape: {'defaults': {...}, 'TimesNet': {...}, 'PatchTST': {...}}."
        ),
    )
    return parser


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_repo_relative_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return _repo_root() / path


def _resolve_summary_path(summary_path: str) -> Path:
    return _resolve_repo_relative_path(path_str=summary_path)


def _derive_run_name(base_run_name: str, model_name: str) -> str:
    if base_run_name == DEFAULT_PREDICTOR_RUN_NAME:
        return f"{model_name}_medical_requests"
    return f"{base_run_name}_{model_name}"


def _validate_override_mapping(scope_name: str, overrides: dict[str, object]) -> dict[str, object]:
    invalid_keys = sorted(
        key
        for key in overrides
        if key not in PREDICTOR_ARG_NAMES or key in {"model_name", "run_name"}
    )
    if invalid_keys:
        raise ValueError(
            f"Invalid override keys for '{scope_name}': {invalid_keys}. "
            "Use predictor argument names and do not override model_name or run_name."
        )
    return dict(overrides)


def _load_model_overrides(model_overrides_file: str | None) -> dict[str, dict[str, object]]:
    if model_overrides_file is None:
        return {}

    overrides_path = _resolve_repo_relative_path(path_str=model_overrides_file)
    with overrides_path.open("r", encoding="utf-8") as overrides_file:
        raw_overrides = json.load(overrides_file)

    if not isinstance(raw_overrides, dict):
        raise ValueError("model_overrides_file must contain a JSON object at the top level.")

    loaded_overrides: dict[str, dict[str, object]] = {}
    for scope_name, scope_overrides in raw_overrides.items():
        if scope_name != "defaults" and scope_name not in AVAILABLE_MODELS:
            raise ValueError(
                f"Unknown override scope '{scope_name}'. "
                f"Expected 'defaults' or one of {AVAILABLE_MODELS}."
            )
        if not isinstance(scope_overrides, dict):
            raise ValueError(f"Override scope '{scope_name}' must map to a JSON object.")

        loaded_overrides[scope_name] = _validate_override_mapping(
            scope_name=scope_name,
            overrides=scope_overrides,
        )

    return loaded_overrides


def _build_effective_predictor_args(
    args: argparse.Namespace,
    model_name: str,
    preprocess_data: bool,
    model_overrides: dict[str, dict[str, object]],
) -> tuple[dict[str, object], dict[str, object]]:
    effective_args = {
        key: value
        for key, value in vars(args).items()
        if key not in DRIVER_ONLY_ARGS and key not in {"model_name", "run_name"}
    }
    applied_overrides: dict[str, object] = {}

    for scope_name in ("defaults", model_name):
        scope_overrides = model_overrides.get(scope_name, {})
        effective_args.update(scope_overrides)
        applied_overrides.update(scope_overrides)

    effective_args["preprocess_data"] = preprocess_data
    if "preprocess_data" in applied_overrides:
        applied_overrides["preprocess_data"] = preprocess_data

    return effective_args, applied_overrides


def _serialize_predictor_args(
    effective_args: dict[str, object],
    model_name: str,
    base_run_name: str,
) -> list[str]:
    command_args: list[str] = []

    for key, value in effective_args.items():
        option = f"--{key}"
        if value is None:
            continue

        if isinstance(value, bool):
            if value:
                command_args.append(option)
            continue

        if isinstance(value, (list, tuple)):
            if not value:
                continue
            command_args.append(option)
            command_args.extend(str(item) for item in value)
            continue

        command_args.extend([option, str(value)])

    command_args.extend(["--model_name", model_name])
    command_args.extend(["--run_name", _derive_run_name(base_run_name=base_run_name, model_name=model_name)])
    return command_args


def _write_summary(summary_path: Path, summary_rows: list[dict[str, object]]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_name",
        "run_name",
        "status",
        "returncode",
        "start_time",
        "end_time",
        "duration_seconds",
        "applied_overrides",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def _run_single_model(
    args: argparse.Namespace,
    model_name: str,
    preprocess_data: bool,
    model_overrides: dict[str, dict[str, object]],
) -> dict[str, object]:
    start_time = datetime.datetime.now()
    run_name = _derive_run_name(base_run_name=args.run_name, model_name=model_name)
    effective_args, applied_overrides = _build_effective_predictor_args(
        args=args,
        model_name=model_name,
        preprocess_data=preprocess_data,
        model_overrides=model_overrides,
    )
    command = [
        sys.executable,
        "-m",
        "HTAMP.prediction.request_predictor",
        *_serialize_predictor_args(
            effective_args=effective_args,
            model_name=model_name,
            base_run_name=args.run_name,
        ),
    ]

    if applied_overrides:
        print(f"Applying overrides for {model_name}: {json.dumps(applied_overrides, sort_keys=True)}")
    print(f"Starting model sweep run for {model_name} with run name '{run_name}'")
    wall_clock_start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=_repo_root(),
        check=False,
    )
    duration_seconds = time.perf_counter() - wall_clock_start
    end_time = datetime.datetime.now()
    status = "success" if completed.returncode == 0 else "failed"

    print(
        f"Finished model sweep run for {model_name} with status '{status}' "
        f"in {duration_seconds:.2f} seconds."
    )

    return {
        "model_name": model_name,
        "run_name": run_name,
        "status": status,
        "returncode": completed.returncode,
        "start_time": start_time.isoformat(timespec="seconds"),
        "end_time": end_time.isoformat(timespec="seconds"),
        "duration_seconds": round(duration_seconds, 2),
        "applied_overrides": json.dumps(applied_overrides, sort_keys=True),
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    summary_path = _resolve_summary_path(summary_path=args.summary_path)
    model_overrides = _load_model_overrides(model_overrides_file=args.model_overrides_file)
    summary_rows: list[dict[str, object]] = []

    preprocess_data = bool(args.preprocess_data)
    for model_name in args.models:
        result = _run_single_model(
            args=args,
            model_name=model_name,
            preprocess_data=preprocess_data,
            model_overrides=model_overrides,
        )
        summary_rows.append(result)
        _write_summary(summary_path=summary_path, summary_rows=summary_rows)

        preprocess_data = False

        if result["status"] != "success" and args.fail_fast:
            print(f"Stopping early because {model_name} failed and --fail_fast was set.")
            return int(result["returncode"])

    failed_runs = [row for row in summary_rows if row["status"] != "success"]
    print(f"Model sweep summary saved to {summary_path}")
    if failed_runs:
        print(f"{len(failed_runs)} model run(s) failed.")
        return 1

    print("All model runs completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
