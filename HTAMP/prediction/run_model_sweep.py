from __future__ import annotations

import argparse
import copy
import csv
import datetime
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, fields
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

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
MANAGED_MODEL_FIELDS = {"model_name", "run_name"}
MULTI_DEVICE_STRATEGY_PREFIXES = ("ddp", "fsdp", "deepspeed")
DEFAULT_SELECTION_METRIC = "val_delta_mae"
METRICS_SUMMARY_FILENAME = "metrics_summary.json"
SUMMARY_METRIC_FIELDS = [
    "val_loss",
    "val_delta_loss",
    "val_availability_loss",
    "val_delta_mae",
    "val_delta_mse",
    "val_delta_rmse",
    "val_availability_accuracy",
    "test_loss",
    "test_delta_loss",
    "test_availability_loss",
    "test_delta_mae",
    "test_delta_mse",
    "test_delta_rmse",
    "test_availability_accuracy",
]
PREFERRED_TEST_PLOT_METRICS = [
    "test_delta_mae",
    "test_delta_rmse",
    "test_availability_accuracy",
    "test_loss",
]
CONFIG_SECTION_FIELDS = {
    "dataset_config": {field.name for field in fields(MedicalRequestDatasetConfig)},
    "model_config": {field.name for field in fields(TimeseriesModelConfig)},
}
ANNOTATED_DATA_FILE_FIELDS = {field.name for field in fields(AnnotatedDataFiles)}


@dataclass
class SweepTrial:
    model_name: str
    trial_index: int
    total_model_trials: int
    run_name: str
    training_config: RequestTrainingConfig
    applied_overrides: dict[str, Any]
    search_overrides: dict[str, Any]
    dataset_group_key: str
    wait_for_dataset_group: str | None = None
    is_dataset_group_leader: bool = False


@dataclass
class RunningTrial:
    trial: SweepTrial
    process: subprocess.Popen
    temp_config_path: Path
    gpu_id: int | None
    start_time: datetime.datetime
    wall_clock_start: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="RequestModelSweep",
        description="Run a request-prediction model sweep from a JSON config file.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help=(
            "Path to a JSON file containing 'predictor_config', optional 'model_overrides', "
            "optional per-model 'search_space', and sweep scheduling settings."
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


def _resolve_comparison_plot_path(
    comparison_plot_path: str | None,
    summary_path: Path,
) -> Path:
    if comparison_plot_path:
        return _resolve_repo_relative_path(path_str=comparison_plot_path)
    return summary_path.with_name(f"{summary_path.stem}_test_metrics.png")


def _log_dir() -> Path:
    return Path(os.getenv("STF_LOG_DIR", "./data/STF_LOG_DIR"))


def _temp_config_dir() -> Path:
    return _log_dir() / "temp_configs"


def _metrics_summary_path(run_name: str) -> Path:
    return _log_dir() / run_name / METRICS_SUMMARY_FILENAME


def _derive_model_run_name(base_run_name: str, model_name: str) -> str:
    if base_run_name == DEFAULT_PREDICTOR_RUN_NAME:
        return f"{model_name}_medical_request_intervals"
    return f"{base_run_name}_{model_name}"


def _derive_trial_run_name(
    base_run_name: str,
    model_name: str,
    trial_index: int,
    total_model_trials: int,
) -> str:
    model_run_name = _derive_model_run_name(base_run_name=base_run_name, model_name=model_name)
    if total_model_trials <= 1:
        return model_run_name
    return f"{model_run_name}_trial_{trial_index:03d}"


def _deep_merge_dicts(base: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(base=merged[key], updates=value)
            continue
        merged[key] = copy.deepcopy(value)
    return merged


def _set_nested_value(payload: dict[str, Any], path: Sequence[str], value: Any) -> None:
    current_level = payload
    for path_part in path[:-1]:
        nested_value = current_level.get(path_part)
        if not isinstance(nested_value, dict):
            nested_value = {}
            current_level[path_part] = nested_value
        current_level = nested_value
    current_level[path[-1]] = copy.deepcopy(value)


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool) or value in ("", None):
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric_value):
        return None
    return numeric_value


def _metric_higher_is_better(metric_name: str) -> bool:
    metric_name_lower = metric_name.lower()
    return any(token in metric_name_lower for token in ("accuracy", "auc", "f1", "precision", "recall"))


def _read_metrics_summary(run_name: str) -> dict[str, Any]:
    metrics_path = _metrics_summary_path(run_name=run_name)
    if not metrics_path.exists():
        return {}

    with metrics_path.open("r", encoding="utf-8") as metrics_file:
        payload = json.load(metrics_file)

    if not isinstance(payload, dict):
        return {}
    return payload


def _extract_summary_metrics(metrics_summary: Mapping[str, Any]) -> dict[str, Any]:
    extracted_metrics = {
        "metrics_summary_path": str(metrics_summary.get("metrics_summary_path", "")),
        "best_checkpoint_path": str(metrics_summary.get("best_checkpoint_path", "")),
        "best_checkpoint_score": metrics_summary.get("best_checkpoint_score", ""),
    }

    validation_metrics = metrics_summary.get("validation_metrics", {})
    if isinstance(validation_metrics, Mapping):
        for metric_name, metric_value in validation_metrics.items():
            extracted_metrics[str(metric_name)] = metric_value

    test_metrics = metrics_summary.get("test_metrics", {})
    if isinstance(test_metrics, Mapping):
        for metric_name, metric_value in test_metrics.items():
            extracted_metrics[str(metric_name)] = metric_value

    return extracted_metrics


def _write_summary(summary_path: Path, summary_rows: list[dict[str, object]]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_name",
        "trial_index",
        "total_model_trials",
        "run_name",
        "gpu_id",
        "status",
        "reason",
        "returncode",
        "start_time",
        "end_time",
        "duration_seconds",
        "selection_metric",
        "selection_metric_value",
        "selection_rank",
        "best_checkpoint_path",
        "best_checkpoint_score",
        "metrics_summary_path",
        "search_overrides",
        "applied_overrides",
        *SUMMARY_METRIC_FIELDS,
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


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

        invalid_fields = sorted(
            field_name for field_name in section_overrides if field_name not in CONFIG_SECTION_FIELDS[section_name]
        )
        if invalid_fields:
            raise ValueError(
                f"Invalid override fields for '{scope_name}.{section_name}': {invalid_fields}."
            )

        if section_name == "model_config":
            protected_fields = sorted(
                field_name for field_name in section_overrides if field_name in MANAGED_MODEL_FIELDS
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


def _validate_search_candidates(field_path: str, candidates: Any) -> list[Any]:
    if isinstance(candidates, (str, bytes, bytearray)) or not isinstance(candidates, Sequence):
        raise ValueError(
            f"Search space field '{field_path}' must be a JSON array of candidate values."
        )

    candidate_values = [copy.deepcopy(candidate) for candidate in candidates]
    if not candidate_values:
        raise ValueError(f"Search space field '{field_path}' must include at least one candidate value.")
    return candidate_values


def _validate_search_space_mapping(
    scope_name: str,
    search_space: Mapping[str, Any],
) -> dict[str, Any]:
    invalid_sections = sorted(section_name for section_name in search_space if section_name not in CONFIG_SECTION_FIELDS)
    if invalid_sections:
        raise ValueError(
            f"Invalid search-space sections for '{scope_name}': {invalid_sections}. "
            "Use 'dataset_config' and/or 'model_config'."
        )

    validated_search_space: dict[str, Any] = {}
    for section_name, section_search_space in search_space.items():
        if not isinstance(section_search_space, Mapping):
            raise ValueError(
                f"Search-space section '{scope_name}.{section_name}' must be a JSON object."
            )

        validated_section: dict[str, Any] = {}
        for field_name, field_candidates in section_search_space.items():
            if field_name not in CONFIG_SECTION_FIELDS[section_name]:
                raise ValueError(
                    f"Invalid search-space fields for '{scope_name}.{section_name}': {[field_name]}."
                )

            field_path = f"{scope_name}.{section_name}.{field_name}"
            if section_name == "model_config" and field_name in MANAGED_MODEL_FIELDS:
                raise ValueError(
                    f"Do not search over '{field_path}'. The sweep driver manages model names and run names."
                )

            if section_name == "dataset_config" and field_name == "annotated_data_files":
                if not isinstance(field_candidates, Mapping):
                    raise ValueError(
                        f"'{field_path}' must be a JSON object whose leaf values are candidate arrays."
                    )

                nested_candidates: dict[str, Any] = {}
                for nested_field_name, nested_field_candidates in field_candidates.items():
                    if nested_field_name not in ANNOTATED_DATA_FILE_FIELDS:
                        raise ValueError(
                            f"Invalid annotated_data_files search-space fields for '{field_path}': "
                            f"{[nested_field_name]}."
                        )
                    nested_candidates[nested_field_name] = _validate_search_candidates(
                        field_path=f"{field_path}.{nested_field_name}",
                        candidates=nested_field_candidates,
                    )
                validated_section[field_name] = nested_candidates
                continue

            if isinstance(field_candidates, Mapping):
                raise ValueError(
                    f"Search-space field '{field_path}' must be a JSON array of candidate values."
                )

            validated_section[field_name] = _validate_search_candidates(
                field_path=field_path,
                candidates=field_candidates,
            )

        validated_search_space[section_name] = validated_section

    return validated_search_space


def _validate_models(models: tuple[str, ...]) -> tuple[str, ...]:
    invalid_models = sorted(model_name for model_name in models if model_name not in AVAILABLE_MODELS)
    if invalid_models:
        raise ValueError(f"Unsupported model(s) in sweep config: {invalid_models}.")
    return models


def _validate_scoped_mapping_names(
    scoped_mapping: Mapping[str, Any],
    models: tuple[str, ...],
    mapping_name: str,
) -> None:
    allowed_scopes = {"defaults"} | set(models)
    invalid_scopes = sorted(scope_name for scope_name in scoped_mapping if scope_name not in allowed_scopes)
    if invalid_scopes:
        raise ValueError(
            f"Invalid {mapping_name} scope(s): {invalid_scopes}. "
            f"Expected 'defaults' or one of {models}."
        )


def _validate_selection_metric(selection_metric: str) -> str:
    normalized_metric = str(selection_metric).strip()
    if not normalized_metric:
        raise ValueError("selection_metric must not be empty.")
    return normalized_metric


def _validate_training_config_for_model(training_config: RequestTrainingConfig) -> None:
    model_config = training_config.model_config
    model_name = model_config.model_name

    if model_name in {"iTransformer", "PatchTST"} and model_config.d_model % model_config.n_heads != 0:
        raise ValueError(
            f"Invalid {model_name} configuration: d_model ({model_config.d_model}) must be "
            f"divisible by n_heads ({model_config.n_heads})."
        )

    if model_name != "PatchTST":
        return

    if model_config.patch_len <= 0:
        raise ValueError("Invalid PatchTST configuration: patch_len must be positive.")
    if model_config.stride <= 0:
        raise ValueError("Invalid PatchTST configuration: stride must be positive.")
    if model_config.patch_len > model_config.seq_len:
        raise ValueError(
            f"Invalid PatchTST configuration: patch_len ({model_config.patch_len}) must be "
            f"less than or equal to seq_len ({model_config.seq_len})."
        )
    if model_config.stride > model_config.patch_len:
        raise ValueError(
            f"Invalid PatchTST configuration: stride ({model_config.stride}) should be "
            f"less than or equal to patch_len ({model_config.patch_len})."
        )


def _build_effective_training_config(
    sweep_config: RequestModelSweepConfig,
    model_name: str,
    preprocess_data: bool,
) -> tuple[RequestTrainingConfig, dict[str, Any]]:
    effective_training_config = RequestTrainingConfig.from_dict(
        sweep_config.predictor_config.to_dict()
    )
    applied_overrides: dict[str, Any] = {}

    for scope_name in ("defaults", model_name):
        scope_overrides = sweep_config.model_overrides.get(scope_name, {})
        if not scope_overrides:
            continue

        validated_overrides = _validate_override_mapping(scope_name=scope_name, overrides=scope_overrides)
        effective_training_config = effective_training_config.with_overrides(validated_overrides)
        applied_overrides = _deep_merge_dicts(base=applied_overrides, updates=validated_overrides)

    effective_training_config.model_config.model_name = model_name
    effective_training_config.model_config.run_name = _derive_model_run_name(
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


def _build_effective_search_space(
    sweep_config: RequestModelSweepConfig,
    model_name: str,
) -> dict[str, Any]:
    effective_search_space: dict[str, Any] = {}

    for scope_name in ("defaults", model_name):
        scope_search_space = sweep_config.search_space.get(scope_name, {})
        if not scope_search_space:
            continue

        validated_search_space = _validate_search_space_mapping(
            scope_name=scope_name,
            search_space=scope_search_space,
        )
        effective_search_space = _deep_merge_dicts(
            base=effective_search_space,
            updates=validated_search_space,
        )

    return effective_search_space


def _collect_search_dimensions(
    path_prefix: tuple[str, ...],
    value: Any,
    dimensions: list[tuple[tuple[str, ...], list[Any]]],
) -> None:
    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            _collect_search_dimensions(
                path_prefix=(*path_prefix, str(key)),
                value=nested_value,
                dimensions=dimensions,
            )
        return

    dimensions.append((path_prefix, [copy.deepcopy(candidate) for candidate in value]))


def _expand_search_space_overrides(search_space: Mapping[str, Any]) -> list[dict[str, Any]]:
    dimensions: list[tuple[tuple[str, ...], list[Any]]] = []
    for section_name, section_search_space in search_space.items():
        _collect_search_dimensions(
            path_prefix=(section_name,),
            value=section_search_space,
            dimensions=dimensions,
        )

    if not dimensions:
        return [{}]

    search_overrides: list[dict[str, Any]] = []
    candidate_lists = [candidate_values for _, candidate_values in dimensions]
    for selected_values in product(*candidate_lists):
        selected_overrides: dict[str, Any] = {}
        for (path, _), selected_value in zip(dimensions, selected_values):
            _set_nested_value(payload=selected_overrides, path=path, value=selected_value)
        search_overrides.append(selected_overrides)

    return search_overrides


def _dataset_group_key(training_config: RequestTrainingConfig) -> str:
    dataset_payload = training_config.dataset_config.to_dict()
    dataset_payload.pop("preprocess_data", None)
    dataset_payload.pop("save_data", None)
    dataset_payload.pop("use_saved_time_series", None)
    return json.dumps(dataset_payload, sort_keys=True)


def _build_model_trials(
    sweep_config: RequestModelSweepConfig,
    model_name: str,
    preprocess_data: bool,
) -> list[SweepTrial]:
    effective_search_space = _build_effective_search_space(
        sweep_config=sweep_config,
        model_name=model_name,
    )
    search_overrides_list = _expand_search_space_overrides(search_space=effective_search_space)
    total_model_trials = len(search_overrides_list)
    trials: list[SweepTrial] = []

    for trial_index, search_overrides in enumerate(search_overrides_list, start=1):
        training_config, applied_overrides = _build_effective_training_config(
            sweep_config=sweep_config,
            model_name=model_name,
            preprocess_data=preprocess_data,
        )

        if search_overrides:
            training_config = training_config.with_overrides(search_overrides)
            applied_overrides = _deep_merge_dicts(
                base=applied_overrides,
                updates=search_overrides,
            )

        run_name = _derive_trial_run_name(
            base_run_name=sweep_config.predictor_config.model_config.run_name,
            model_name=model_name,
            trial_index=trial_index,
            total_model_trials=total_model_trials,
        )
        training_config.model_config.model_name = model_name
        training_config.model_config.run_name = run_name
        _validate_training_config_for_model(training_config=training_config)

        trials.append(
            SweepTrial(
                model_name=model_name,
                trial_index=trial_index,
                total_model_trials=total_model_trials,
                run_name=run_name,
                training_config=training_config,
                applied_overrides=applied_overrides,
                search_overrides=search_overrides,
                dataset_group_key=_dataset_group_key(training_config=training_config),
            )
        )

    return trials


def _assign_dataset_dependencies(trials: list[SweepTrial]) -> None:
    grouped_trials: dict[str, list[SweepTrial]] = {}
    for trial in trials:
        grouped_trials.setdefault(trial.dataset_group_key, []).append(trial)

    for group_trials in grouped_trials.values():
        if len(group_trials) > 1:
            for trial in group_trials:
                if trial.training_config.dataset_config.use_saved_time_series:
                    continue
                trial.training_config.dataset_config.use_saved_time_series = True
                trial.applied_overrides = _deep_merge_dicts(
                    base=trial.applied_overrides,
                    updates={"dataset_config": {"use_saved_time_series": True}},
                )

        leader = next(
            (
                trial
                for trial in group_trials
                if trial.training_config.dataset_config.preprocess_data
            ),
            None,
        )
        if leader is None:
            continue

        leader.is_dataset_group_leader = True
        if len(group_trials) > 1 and not leader.training_config.dataset_config.save_data:
            leader.training_config.dataset_config.save_data = True
            leader.applied_overrides = _deep_merge_dicts(
                base=leader.applied_overrides,
                updates={"dataset_config": {"save_data": True}},
            )

        for trial in group_trials:
            if trial is leader:
                continue

            trial.wait_for_dataset_group = leader.dataset_group_key
            if trial.training_config.dataset_config.preprocess_data:
                trial.training_config.dataset_config.preprocess_data = False
                trial.applied_overrides = _deep_merge_dicts(
                    base=trial.applied_overrides,
                    updates={"dataset_config": {"preprocess_data": False}},
                )


def _is_multi_device_strategy(strategy: str | None) -> bool:
    if strategy is None:
        return False
    return str(strategy).lower().startswith(MULTI_DEVICE_STRATEGY_PREFIXES)


def _normalize_parallel_trial_settings(
    trials: list[SweepTrial],
    parallelism: int,
    gpu_ids: Sequence[int],
) -> None:
    if parallelism <= 1 and not gpu_ids:
        return

    for trial in trials:
        if trial.training_config.model_config.devices != 1:
            trial.training_config.model_config.devices = 1
            trial.applied_overrides = _deep_merge_dicts(
                base=trial.applied_overrides,
                updates={"model_config": {"devices": 1}},
            )

        if _is_multi_device_strategy(trial.training_config.model_config.strategy):
            trial.training_config.model_config.strategy = None
            trial.applied_overrides = _deep_merge_dicts(
                base=trial.applied_overrides,
                updates={"model_config": {"strategy": None}},
            )


def _resolve_parallelism(sweep_config: RequestModelSweepConfig) -> int:
    parallelism = max(1, sweep_config.max_parallel_runs)
    if sweep_config.gpu_ids:
        parallelism = min(parallelism, len(sweep_config.gpu_ids))
    return max(1, parallelism)


def _build_sweep_trials(sweep_config: RequestModelSweepConfig) -> list[SweepTrial]:
    preprocess_data = bool(sweep_config.predictor_config.dataset_config.preprocess_data)
    trials: list[SweepTrial] = []
    for model_name in sweep_config.models:
        trials.extend(
            _build_model_trials(
                sweep_config=sweep_config,
                model_name=model_name,
                preprocess_data=preprocess_data,
            )
        )

    _assign_dataset_dependencies(trials=trials)
    parallelism = _resolve_parallelism(sweep_config=sweep_config)
    _normalize_parallel_trial_settings(
        trials=trials,
        parallelism=parallelism,
        gpu_ids=sweep_config.gpu_ids,
    )
    return trials


def _write_temp_training_config(training_config: RequestTrainingConfig, run_name: str) -> Path:
    safe_prefix = "".join(char if char.isalnum() else "_" for char in run_name.lower())
    temp_config_dir = _temp_config_dir()
    temp_config_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        prefix=f"{safe_prefix[:40]}_",
        dir=temp_config_dir,
        delete=False,
    ) as config_file:
        json.dump(training_config.to_dict(), config_file, indent=2)
        return Path(config_file.name)


def _build_process_env(gpu_id: int | None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return env


def _launch_trial(trial: SweepTrial, gpu_id: int | None) -> RunningTrial:
    metrics_path = _metrics_summary_path(run_name=trial.run_name)
    metrics_path.unlink(missing_ok=True)
    temp_config_path = _write_temp_training_config(
        training_config=trial.training_config,
        run_name=trial.run_name,
    )
    command = [
        sys.executable,
        "-m",
        "HTAMP.prediction.module.request_predictor",
        "--config_path",
        str(temp_config_path),
    ]

    if trial.applied_overrides:
        print(
            f"Applying overrides for {trial.run_name}: "
            f"{json.dumps(trial.applied_overrides, sort_keys=True)}"
        )

    launch_details = [
        f"Starting trial {trial.trial_index}/{trial.total_model_trials}",
        f"for {trial.model_name}",
        f"with run name '{trial.run_name}'",
    ]
    if gpu_id is not None:
        launch_details.append(f"on GPU {gpu_id}")
    if trial.is_dataset_group_leader and trial.training_config.dataset_config.preprocess_data:
        launch_details.append("(includes dataset preprocessing)")
    print(" ".join(launch_details))

    start_time = datetime.datetime.now()
    process = subprocess.Popen(
        command,
        cwd=_repo_root(),
        env=_build_process_env(gpu_id=gpu_id),
    )
    return RunningTrial(
        trial=trial,
        process=process,
        temp_config_path=temp_config_path,
        gpu_id=gpu_id,
        start_time=start_time,
        wall_clock_start=time.perf_counter(),
    )


def _finalize_running_trial(running_trial: RunningTrial) -> dict[str, object]:
    returncode = running_trial.process.wait()
    duration_seconds = time.perf_counter() - running_trial.wall_clock_start
    end_time = datetime.datetime.now()
    running_trial.temp_config_path.unlink(missing_ok=True)
    status = "success" if returncode == 0 else "failed"
    metrics_summary = _read_metrics_summary(run_name=running_trial.trial.run_name)
    extracted_metrics = _extract_summary_metrics(metrics_summary=metrics_summary)

    print(
        f"Finished trial {running_trial.trial.trial_index}/{running_trial.trial.total_model_trials} "
        f"for {running_trial.trial.model_name} with status '{status}' "
        f"in {duration_seconds:.2f} seconds."
    )

    return {
        "model_name": running_trial.trial.model_name,
        "trial_index": running_trial.trial.trial_index,
        "total_model_trials": running_trial.trial.total_model_trials,
        "run_name": running_trial.trial.run_name,
        "gpu_id": running_trial.gpu_id if running_trial.gpu_id is not None else "",
        "status": status,
        "reason": "",
        "returncode": returncode,
        "start_time": running_trial.start_time.isoformat(timespec="seconds"),
        "end_time": end_time.isoformat(timespec="seconds"),
        "duration_seconds": round(duration_seconds, 2),
        "selection_metric": "",
        "selection_metric_value": "",
        "selection_rank": "",
        "search_overrides": json.dumps(running_trial.trial.search_overrides, sort_keys=True),
        "applied_overrides": json.dumps(running_trial.trial.applied_overrides, sort_keys=True),
        **{metric_name: extracted_metrics.get(metric_name, "") for metric_name in SUMMARY_METRIC_FIELDS},
        "best_checkpoint_path": extracted_metrics.get("best_checkpoint_path", ""),
        "best_checkpoint_score": extracted_metrics.get("best_checkpoint_score", ""),
        "metrics_summary_path": extracted_metrics.get("metrics_summary_path", ""),
    }


def _build_non_run_trial_result(
    trial: SweepTrial,
    status: str,
    reason: str,
    returncode: int | str = "",
) -> dict[str, object]:
    return {
        "model_name": trial.model_name,
        "trial_index": trial.trial_index,
        "total_model_trials": trial.total_model_trials,
        "run_name": trial.run_name,
        "gpu_id": "",
        "status": status,
        "reason": reason,
        "returncode": returncode,
        "start_time": "",
        "end_time": "",
        "duration_seconds": "",
        "selection_metric": "",
        "selection_metric_value": "",
        "selection_rank": "",
        "best_checkpoint_path": "",
        "best_checkpoint_score": "",
        "metrics_summary_path": "",
        "search_overrides": json.dumps(trial.search_overrides, sort_keys=True),
        "applied_overrides": json.dumps(trial.applied_overrides, sort_keys=True),
        **{metric_name: "" for metric_name in SUMMARY_METRIC_FIELDS},
    }


def _collect_dependency_skips(
    pending_trials: list[SweepTrial],
    dataset_group_status: Mapping[str, str],
) -> list[dict[str, object]]:
    remaining_trials: list[SweepTrial] = []
    skipped_rows: list[dict[str, object]] = []

    for trial in pending_trials:
        dependency_status = dataset_group_status.get(trial.wait_for_dataset_group) if trial.wait_for_dataset_group else None
        if dependency_status == "failed":
            skipped_rows.append(
                _build_non_run_trial_result(
                    trial=trial,
                    status="skipped",
                    reason="Skipped because dataset preprocessing failed for this dataset configuration.",
                )
            )
            continue
        remaining_trials.append(trial)

    pending_trials[:] = remaining_trials
    return skipped_rows


def _find_next_ready_trial_index(
    pending_trials: Sequence[SweepTrial],
    dataset_group_status: Mapping[str, str],
) -> int | None:
    for index, trial in enumerate(pending_trials):
        if trial.wait_for_dataset_group is None:
            return index
        if dataset_group_status.get(trial.wait_for_dataset_group) == "ready":
            return index
    return None


def _cancel_running_trials(running_trials: list[RunningTrial]) -> list[dict[str, object]]:
    cancelled_rows: list[dict[str, object]] = []
    for running_trial in running_trials:
        if running_trial.process.poll() is None:
            running_trial.process.terminate()

    for running_trial in running_trials:
        try:
            running_trial.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            running_trial.process.kill()
            running_trial.process.wait()
        running_trial.temp_config_path.unlink(missing_ok=True)
        cancelled_rows.append(
            _build_non_run_trial_result(
                trial=running_trial.trial,
                status="cancelled",
                reason="Cancelled because fail_fast stopped the sweep after another trial failed.",
                returncode=running_trial.process.returncode,
            )
        )

    running_trials.clear()
    return cancelled_rows


def _run_sweep_trials(
    sweep_config: RequestModelSweepConfig,
    trials: list[SweepTrial],
    summary_path: Path,
) -> tuple[list[dict[str, object]], int]:
    parallelism = _resolve_parallelism(sweep_config=sweep_config)
    if parallelism > 1 and not sweep_config.gpu_ids:
        print(
            "Running sweep in parallel without explicit GPU pinning. "
            "Set 'gpu_ids' in the sweep config to pin one trial per GPU."
        )

    pending_trials = list(trials)
    running_trials: list[RunningTrial] = []
    summary_rows: list[dict[str, object]] = []
    dataset_group_status: dict[str, str] = {}
    available_gpu_ids = list(sweep_config.gpu_ids[:parallelism])

    while pending_trials or running_trials:
        skipped_rows = _collect_dependency_skips(
            pending_trials=pending_trials,
            dataset_group_status=dataset_group_status,
        )
        if skipped_rows:
            summary_rows.extend(skipped_rows)
            _write_summary(summary_path=summary_path, summary_rows=summary_rows)

        while len(running_trials) < parallelism:
            ready_index = _find_next_ready_trial_index(
                pending_trials=pending_trials,
                dataset_group_status=dataset_group_status,
            )
            if ready_index is None:
                break

            next_trial = pending_trials.pop(ready_index)
            gpu_id = available_gpu_ids.pop(0) if available_gpu_ids else None
            running_trials.append(_launch_trial(trial=next_trial, gpu_id=gpu_id))

        if not running_trials:
            if pending_trials:
                blocked_groups = sorted(
                    {
                        trial.wait_for_dataset_group
                        for trial in pending_trials
                        if trial.wait_for_dataset_group is not None
                    }
                )
                raise RuntimeError(
                    "No runnable trials remain. "
                    f"Blocked dataset group(s): {blocked_groups}"
                )
            break

        time.sleep(1.0)
        finished_indices = [
            index
            for index, running_trial in enumerate(running_trials)
            if running_trial.process.poll() is not None
        ]
        if not finished_indices:
            continue

        for finished_index in reversed(finished_indices):
            finished_trial = running_trials.pop(finished_index)
            result = _finalize_running_trial(running_trial=finished_trial)
            summary_rows.append(result)
            _write_summary(summary_path=summary_path, summary_rows=summary_rows)

            if finished_trial.gpu_id is not None:
                available_gpu_ids.append(finished_trial.gpu_id)
                available_gpu_ids.sort()

            if finished_trial.trial.is_dataset_group_leader:
                dataset_group_status[finished_trial.trial.dataset_group_key] = (
                    "ready" if result["status"] == "success" else "failed"
                )

            if result["status"] != "success" and sweep_config.fail_fast:
                print(
                    f"Stopping early because {finished_trial.trial.run_name} failed "
                    "and fail_fast was set in the config."
                )
                summary_rows.extend(_cancel_running_trials(running_trials=running_trials))
                for pending_trial in pending_trials:
                    summary_rows.append(
                        _build_non_run_trial_result(
                            trial=pending_trial,
                            status="cancelled",
                            reason="Cancelled because fail_fast stopped the sweep after another trial failed.",
                        )
                    )
                _write_summary(summary_path=summary_path, summary_rows=summary_rows)
                return summary_rows, int(result["returncode"])

    failed_rows = [row for row in summary_rows if row["status"] == "failed"]
    return summary_rows, 1 if failed_rows else 0


def _apply_selection_ranking(
    summary_rows: list[dict[str, object]],
    selection_metric: str,
) -> None:
    ranked_rows: list[tuple[float, dict[str, object]]] = []
    for row in summary_rows:
        row["selection_metric"] = selection_metric
        metric_value = _coerce_float(row.get(selection_metric))
        row["selection_metric_value"] = metric_value if metric_value is not None else ""
        row["selection_rank"] = ""
        if row.get("status") == "success" and metric_value is not None:
            ranked_rows.append((metric_value, row))

    ranked_rows.sort(
        key=lambda item: item[0],
        reverse=_metric_higher_is_better(metric_name=selection_metric),
    )
    for rank, (_, row) in enumerate(ranked_rows, start=1):
        row["selection_rank"] = rank


def _best_successful_rows_by_model(
    summary_rows: Sequence[dict[str, object]],
    selection_metric: str,
    model_order: Sequence[str],
) -> list[dict[str, object]]:
    best_rows_by_model: dict[str, dict[str, object]] = {}
    higher_is_better = _metric_higher_is_better(metric_name=selection_metric)

    for row in summary_rows:
        if row.get("status") != "success":
            continue
        metric_value = _coerce_float(row.get(selection_metric))
        if metric_value is None:
            continue

        model_name = str(row["model_name"])
        current_best = best_rows_by_model.get(model_name)
        if current_best is None:
            best_rows_by_model[model_name] = row
            continue

        current_value = _coerce_float(current_best.get(selection_metric))
        if current_value is None:
            best_rows_by_model[model_name] = row
            continue

        if higher_is_better:
            if metric_value > current_value:
                best_rows_by_model[model_name] = row
        elif metric_value < current_value:
            best_rows_by_model[model_name] = row

    ordered_rows = [best_rows_by_model[model_name] for model_name in model_order if model_name in best_rows_by_model]
    for model_name, row in sorted(best_rows_by_model.items()):
        if model_name not in model_order:
            ordered_rows.append(row)
    return ordered_rows


def _plot_test_metric_comparison(
    summary_rows: Sequence[dict[str, object]],
    selection_metric: str,
    model_order: Sequence[str],
    plot_path: Path,
) -> bool:
    best_rows = _best_successful_rows_by_model(
        summary_rows=summary_rows,
        selection_metric=selection_metric,
        model_order=model_order,
    )
    if not best_rows:
        return False

    metric_names = [
        metric_name
        for metric_name in PREFERRED_TEST_PLOT_METRICS
        if all(_coerce_float(row.get(metric_name)) is not None for row in best_rows)
    ]
    if not metric_names:
        return False

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib is not available, so the comparison plot was skipped.")
        return False

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        len(metric_names),
        1,
        figsize=(max(8, len(best_rows) * 1.8), max(4, len(metric_names) * 3.6)),
    )
    if len(metric_names) == 1:
        axes = [axes]

    labels = [
        f"{row['model_name']}\ntrial {int(row['trial_index']):03d}"
        for row in best_rows
    ]
    positions = np.arange(len(best_rows))

    for axis, metric_name in zip(axes, metric_names):
        values = [float(row[metric_name]) for row in best_rows]
        axis.bar(positions, values)
        axis.set_xticks(positions, labels, rotation=0)
        axis.set_ylabel(metric_name)
        direction = "higher is better" if _metric_higher_is_better(metric_name=metric_name) else "lower is better"
        axis.set_title(f"{metric_name} ({direction})")
        for index, value in enumerate(values):
            axis.annotate(
                f"{value:.4f}",
                xy=(positions[index], value),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.suptitle(f"Best test metrics by model, selected via {selection_metric}", fontsize=12)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    sweep_config = RequestModelSweepConfig.from_json_file(args.config_path)
    sweep_config.models = _validate_models(models=sweep_config.models)
    sweep_config.selection_metric = _validate_selection_metric(
        selection_metric=sweep_config.selection_metric
    )
    _validate_scoped_mapping_names(
        scoped_mapping=sweep_config.model_overrides,
        models=sweep_config.models,
        mapping_name="model override",
    )
    _validate_scoped_mapping_names(
        scoped_mapping=sweep_config.search_space,
        models=sweep_config.models,
        mapping_name="search-space",
    )

    summary_path = _resolve_summary_path(summary_path=sweep_config.summary_path)
    comparison_plot_path = _resolve_comparison_plot_path(
        comparison_plot_path=sweep_config.comparison_plot_path,
        summary_path=summary_path,
    )
    trials = _build_sweep_trials(sweep_config=sweep_config)
    total_trials = len(trials)
    parallelism = _resolve_parallelism(sweep_config=sweep_config)

    print(
        f"Prepared {total_trials} trial(s) across {len(sweep_config.models)} model(s). "
        f"Max parallel runs: {parallelism}."
    )
    if sweep_config.gpu_ids:
        print(f"Using GPU slots: {list(sweep_config.gpu_ids[:parallelism])}")

    summary_rows, exit_code = _run_sweep_trials(
        sweep_config=sweep_config,
        trials=trials,
        summary_path=summary_path,
    )
    _apply_selection_ranking(
        summary_rows=summary_rows,
        selection_metric=sweep_config.selection_metric,
    )
    _write_summary(summary_path=summary_path, summary_rows=summary_rows)

    plotted = _plot_test_metric_comparison(
        summary_rows=summary_rows,
        selection_metric=sweep_config.selection_metric,
        model_order=sweep_config.models,
        plot_path=comparison_plot_path,
    )

    print(f"Model sweep summary saved to {summary_path}")
    if plotted:
        print(f"Test-metric comparison plot saved to {comparison_plot_path}")

    top_ranked_rows = [
        row for row in summary_rows
        if row.get("selection_rank") == 1
    ]
    if top_ranked_rows:
        top_row = top_ranked_rows[0]
        print(
            f"Top trial by {sweep_config.selection_metric}: "
            f"{top_row['run_name']} ({top_row['selection_metric_value']})"
        )

    failed_rows = [row for row in summary_rows if row["status"] == "failed"]
    if failed_rows:
        print(f"{len(failed_rows)} trial(s) failed.")
        return exit_code or 1

    cancelled_rows = [row for row in summary_rows if row["status"] == "cancelled"]
    if cancelled_rows:
        print(f"{len(cancelled_rows)} trial(s) were cancelled.")
        return exit_code or 1

    skipped_rows = [row for row in summary_rows if row["status"] == "skipped"]
    if skipped_rows:
        print(f"{len(skipped_rows)} trial(s) were skipped.")

    print("All requested sweep trials completed successfully.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
