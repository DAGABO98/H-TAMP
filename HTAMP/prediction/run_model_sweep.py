from __future__ import annotations

import argparse
import copy
import csv
import datetime
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping

from HTAMP.data_processing.processing_dataclasses import AnnotatedDataFiles
from HTAMP.prediction.configs.request_config import (
    MedicalRequestDatasetConfig,
    RequestModelSweepConfig,
    RequestTrainingConfig,
    SUPPORTED_REQUEST_MODELS,
    TimeseriesModelConfig,
)

AVAILABLE_MODELS = SUPPORTED_REQUEST_MODELS
DEFAULT_PREDICTOR_RUN_NAME = "TimesNet_medical_request_intervals"
CONFIG_SECTION_FIELDS = {
    "dataset_config": {field.name for field in fields(MedicalRequestDatasetConfig)},
    "model_config": {field.name for field in fields(TimeseriesModelConfig)},
}
ANNOTATED_DATA_FILE_FIELDS = {field.name for field in fields(AnnotatedDataFiles)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="RequestModelSweep",
        description="Run the request predictor across multiple time-series models from a JSON config file.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help=(
            "Path to a JSON file containing 'predictor_config', optional 'model_overrides', "
            "and model sweep settings."
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
        return f"{model_name}_medical_request_intervals"
    return f"{base_run_name}_{model_name}"


def _deep_merge_dicts(base: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(base=merged[key], updates=value)
            continue
        merged[key] = copy.deepcopy(value)
    return merged


def _validate_override_mapping(
    scope_name: str,
    overrides: Mapping[str, Any],
) -> dict[str, dict[str, object]]:
    invalid_sections = sorted(section_name for section_name in overrides if section_name not in CONFIG_SECTION_FIELDS)
    if invalid_sections:
        raise ValueError(
            f"Invalid override sections for '{scope_name}': {invalid_sections}. "
            "Use 'dataset_config' and/or 'model_config'."
        )

    validated_overrides: dict[str, dict[str, object]] = {}
    for section_name, section_overrides in overrides.items():
        if not isinstance(section_overrides, Mapping):
            raise ValueError(f"Override section '{scope_name}.{section_name}' must be a JSON object.")

        invalid_fields = sorted(field_name for field_name in section_overrides if field_name not in CONFIG_SECTION_FIELDS[section_name])
        if invalid_fields:
            raise ValueError(
                f"Invalid override fields for '{scope_name}.{section_name}': {invalid_fields}."
            )

        if section_name == "model_config":
            protected_fields = sorted(
                field_name for field_name in section_overrides if field_name in {"model_name", "run_name"}
            )
            if protected_fields:
                raise ValueError(
                    f"Do not override {protected_fields} inside '{scope_name}.{section_name}'. "
                    "The sweep driver manages those values per model."
                )

        if section_name == "dataset_config" and "annotated_data_files" in section_overrides:
            annotated_data_files = section_overrides["annotated_data_files"]
            if not isinstance(annotated_data_files, Mapping):
                raise ValueError(
                    f"'{scope_name}.{section_name}.annotated_data_files' must be a JSON object."
                )

            invalid_annotated_fields = sorted(
                field_name for field_name in annotated_data_files if field_name not in ANNOTATED_DATA_FILE_FIELDS
            )
            if invalid_annotated_fields:
                raise ValueError(
                    f"Invalid annotated_data_files overrides for '{scope_name}.{section_name}': "
                    f"{invalid_annotated_fields}."
                )

        validated_overrides[section_name] = dict(section_overrides)

    return validated_overrides


def _validate_models(models: tuple[str, ...]) -> tuple[str, ...]:
    invalid_models = sorted(model_name for model_name in models if model_name not in AVAILABLE_MODELS)
    if invalid_models:
        raise ValueError(f"Unsupported model(s) in sweep config: {invalid_models}.")
    return models


def _validate_model_override_scopes(
    model_overrides: Mapping[str, Any],
    models: tuple[str, ...],
) -> None:
    allowed_scopes = {"defaults"} | set(models)
    invalid_scopes = sorted(scope_name for scope_name in model_overrides if scope_name not in allowed_scopes)
    if invalid_scopes:
        raise ValueError(
            f"Invalid model override scope(s): {invalid_scopes}. "
            f"Expected 'defaults' or one of {models}."
        )


def _build_effective_training_config(
    sweep_config: RequestModelSweepConfig,
    model_name: str,
    preprocess_data: bool,
) -> tuple[RequestTrainingConfig, dict[str, Any]]:
    effective_training_config = RequestTrainingConfig.from_dict(sweep_config.predictor_config)
    applied_overrides: dict[str, Any] = {}

    for scope_name in ("defaults", model_name):
        scope_overrides = sweep_config.model_overrides.get(scope_name, {})
        if not scope_overrides:
            continue

        validated_overrides = _validate_override_mapping(scope_name=scope_name, overrides=scope_overrides)
        effective_training_config = effective_training_config.with_overrides(validated_overrides)
        applied_overrides = _deep_merge_dicts(base=applied_overrides, updates=validated_overrides)

    effective_training_config.model_config.model_name = model_name
    effective_training_config.model_config.run_name = _derive_run_name(
        base_run_name=sweep_config.predictor_config.model_config.run_name,
        model_name=model_name,
    )
    effective_training_config.dataset_config.preprocess_data = preprocess_data

    if preprocess_data != sweep_config.predictor_config.dataset_config.preprocess_data:
        applied_overrides = _deep_merge_dicts(
            base=applied_overrides,
            updates={"dataset_config": {"preprocess_data": preprocess_data}},
        )

    return effective_training_config, applied_overrides


def _write_temp_training_config(training_config: RequestTrainingConfig, model_name: str) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        prefix=f"{model_name.lower()}_",
        dir=_repo_root(),
        delete=False,
    ) as config_file:
        json.dump(training_config.to_dict(), config_file, indent=2)
        return Path(config_file.name)


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
    sweep_config: RequestModelSweepConfig,
    model_name: str,
    preprocess_data: bool,
) -> dict[str, object]:
    start_time = datetime.datetime.now()
    effective_training_config, applied_overrides = _build_effective_training_config(
        sweep_config=sweep_config,
        model_name=model_name,
        preprocess_data=preprocess_data,
    )
    run_name = effective_training_config.model_config.run_name
    temp_config_path = _write_temp_training_config(
        training_config=effective_training_config,
        model_name=model_name,
    )
    command = [
        sys.executable,
        "-m",
        "HTAMP.prediction.module.request_predictor",
        "--config_path",
        str(temp_config_path),
    ]

    if applied_overrides:
        print(f"Applying overrides for {model_name}: {json.dumps(applied_overrides, sort_keys=True)}")
    print(f"Starting model sweep run for {model_name} with run name '{run_name}'")

    wall_clock_start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=_repo_root(),
            check=False,
        )
    finally:
        temp_config_path.unlink(missing_ok=True)

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
    sweep_config = RequestModelSweepConfig.from_json_file(args.config_path)
    sweep_config.models = _validate_models(models=sweep_config.models)
    _validate_model_override_scopes(
        model_overrides=sweep_config.model_overrides,
        models=sweep_config.models,
    )
    summary_path = _resolve_summary_path(summary_path=sweep_config.summary_path)
    summary_rows: list[dict[str, object]] = []

    preprocess_data = bool(sweep_config.predictor_config.dataset_config.preprocess_data)
    for model_name in sweep_config.models:
        result = _run_single_model(
            sweep_config=sweep_config,
            model_name=model_name,
            preprocess_data=preprocess_data,
        )
        summary_rows.append(result)
        _write_summary(summary_path=summary_path, summary_rows=summary_rows)

        preprocess_data = False

        if result["status"] != "success" and sweep_config.fail_fast:
            print(f"Stopping early because {model_name} failed and fail_fast was set in the config.")
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
