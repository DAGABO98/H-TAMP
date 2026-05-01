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
DEFAULT_MULTITTPP_CONFIG_PATH = (
    "HTAMP/prediction/configs/config_files/prediction/"
    "vital_sign_multittpp_training.json"
)
SUPPORTED_EASY_TPP_MODELS = (
    "NHP",
    "AttNHP",
    "THP",
    "SAHP",
    "RMTPP",
    "FullyNN",
    "IntensityFree",
    "ANHN",
    "S2P2",
    "WSMTHP",
)
DEFAULT_FLEX_ORDERS = ("ST", "STP")
DEFAULT_FLEX_ST_MARK_SCHEMAS = ("standard", "enhanced")
SUPPORTED_FLEX_ST_MARK_SCHEMAS = ("standard", "enhanced")
DEFAULT_FLEX_CONDITIONING_MODES = ("conditioned", "no_conditioning")
SUPPORTED_FLEX_CONDITIONING_MODES = ("conditioned", "no_conditioning")
SUPPORTED_MULTITTPP_MODELS = (
    "InhomogeneousPoisson",
    "Renewal",
    "ModulatedRenewal",
    "TriTPP",
    "SplineTransformer",
)
DEFAULT_MULTITTPP_MODELS = ("TriTPP", "SplineTransformer", "ModulatedRenewal", "Renewal", "InhomogeneousPoisson")
DEFAULT_MULTITTPP_MARK_SCHEMAS = ("enhanced", "plain")
SUPPORTED_MULTITTPP_MARK_SCHEMAS = ("enhanced", "plain")
DEFAULT_EASY_MARK_SCHEMAS = ("enhanced", "plain")
SUPPORTED_EASY_MARK_SCHEMAS = ("enhanced", "plain")
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
    "log_path",
    "metrics_summary_path",
    "best_checkpoint_path",
    "best_checkpoint_score",
]
EASY_TPP_MODEL_DEFAULTS: dict[str, dict[str, Any]] = {
    "NHP": {
        "model_config": {
            "hidden_size": 64,
            "num_layers": 1,
            "dropout_rate": 0.05,
            "learning_rate": 0.001,
            "loss_integral_num_sample_per_step": 20,
        },
    },
    "RMTPP": {
        "model_config": {
            "hidden_size": 64,
            "num_layers": 1,
            "dropout_rate": 0.05,
            "learning_rate": 0.001,
        },
    },
    "FullyNN": {
        "model_config": {
            "hidden_size": 64,
            "num_layers": 1,
            "dropout_rate": 0.05,
            "learning_rate": 0.001,
            "model_specs": {
                "num_mlp_layers": 2,
                "proper_marked_intensities": True,
            },
        },
    },
    "IntensityFree": {
        "model_config": {
            "hidden_size": 64,
            "num_layers": 1,
            "dropout_rate": 0.05,
            "learning_rate": 0.001,
            "model_specs": {
                "num_mix_components": 64,
            },
        },
    },
    "ODETPP": {
        "model_config": {
            "hidden_size": 64,
            "num_layers": 1,
            "dropout_rate": 0.05,
            "learning_rate": 0.0005,
            "model_specs": {
                "ode_num_sample_per_step": 10,
            },
        },
    },
    "THP": {
        "dataset_config": {
            "max_events_per_sequence": 96,
        },
        "model_config": {
            "batch_size": 16,
            "hidden_size": 128,
            "time_emb_size": 16,
            "num_layers": 2,
            "num_heads": 4,
            "dropout_rate": 0.1,
            "learning_rate": 0.0003,
            "loss_integral_num_sample_per_step": 12,
            "accumulate_grad_batches": 2,
        },
    },
    "AttNHP": {
        "dataset_config": {
            "max_events_per_sequence": 96,
        },
        "model_config": {
            "batch_size": 16,
            "hidden_size": 128,
            "time_emb_size": 16,
            "num_layers": 2,
            "num_heads": 4,
            "dropout_rate": 0.1,
            "learning_rate": 0.0003,
            "loss_integral_num_sample_per_step": 12,
            "accumulate_grad_batches": 2,
        },
    },
    "SAHP": {
        "dataset_config": {
            "max_events_per_sequence": 96,
        },
        "model_config": {
            "batch_size": 16,
            "hidden_size": 128,
            "time_emb_size": 16,
            "num_layers": 2,
            "num_heads": 4,
            "dropout_rate": 0.1,
            "learning_rate": 0.0003,
            "loss_integral_num_sample_per_step": 12,
            "accumulate_grad_batches": 2,
        },
    },
    "ANHN": {
        "dataset_config": {
            "max_events_per_sequence": 64,
        },
        "model_config": {
            "batch_size": 8,
            "hidden_size": 128,
            "time_emb_size": 16,
            "num_layers": 2,
            "num_heads": 4,
            "dropout_rate": 0.1,
            "learning_rate": 0.0003,
            "loss_integral_num_sample_per_step": 8,
            "accumulate_grad_batches": 4,
        },
    },
    "S2P2": {
        "model_config": {
            "hidden_size": 64,
            "num_layers": 2,
            "dropout_rate": 0.05,
            "learning_rate": 0.0005,
            "model_specs": {
                "P": 64,
            },
        },
    },
    "WSMTHP": {
        "dataset_config": {
            "max_events_per_sequence": 96,
        },
        "model_config": {
            "batch_size": 16,
            "hidden_size": 128,
            "num_layers": 2,
            "num_heads": 4,
            "dropout_rate": 0.1,
            "learning_rate": 0.0003,
            "loss_integral_num_sample_per_step": 12,
            "accumulate_grad_batches": 2,
            "model_specs": {
                "CE_coef": 10.0,
                "T_mode": "train_global",
            },
        },
    },
}
FLEX_TPP_VARIANT_DEFAULTS: dict[str, dict[str, Any]] = {
    "ST": {
        "model_config": {
            "depth": 4,
            "dim_k": 16,
            "n_head": 4,
            "dim_ff": 128,
            "dropout": 0.1,
            "learning_rate": 0.001,
        },
    },
    "STP": {
        "model_config": {
            "depth": 6,
            "dim_k": 24,
            "n_head": 4,
            "dim_ff": 256,
            "dropout": 0.15,
            "learning_rate": 0.001,
        },
    },
}
MULTITTPP_MODEL_DEFAULTS: dict[str, dict[str, Any]] = {
    "TriTPP": {
        "model_config": {
            "n_blocks": 4,
            "block_size": 16,
            "n_knots": 20,
            "learning_rate": 0.001,
        },
    },
    "SplineTransformer": {
        "model_config": {
            "n_embd": 32,
            "n_heads": 4,
            "n_knots": 20,
            "dropout": 0.1,
            "learning_rate": 0.001,
        },
    },
}


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
    log_path: Path
    log_file: Any
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


def _parse_easy_mark_schemas(raw_value: str) -> tuple[str, ...]:
    aliases = {
        "enhanced_marks": "enhanced",
        "enhanced": "enhanced",
        "labels": "enhanced",
        "labeled": "enhanced",
        "plain": "plain",
        "task_only": "plain",
        "no_labels": "plain",
        "unlabeled": "plain",
    }
    schemas: list[str] = []
    seen: set[str] = set()
    for raw_schema in _parse_csv_strings(raw_value):
        schema = aliases.get(raw_schema.strip().lower())
        if schema is None:
            raise ValueError(
                f"Unsupported EasyTPP mark schema '{raw_schema}'. "
                f"Expected one of {SUPPORTED_EASY_MARK_SCHEMAS}, with aliases "
                "'task_only' or 'no_labels' for plain."
            )
        if schema in seen:
            continue
        schemas.append(schema)
        seen.add(schema)
    return tuple(schemas)


def _parse_flex_st_mark_schemas(raw_value: str) -> tuple[str, ...]:
    aliases = {
        "standard": "standard",
        "plain": "standard",
        "task": "standard",
        "task_only": "standard",
        "base_task": "standard",
        "enhanced": "enhanced",
        "enhanced_marks": "enhanced",
        "labels": "enhanced",
        "labeled": "enhanced",
    }
    schemas: list[str] = []
    seen: set[str] = set()
    for raw_schema in _parse_csv_strings(raw_value):
        schema = aliases.get(raw_schema.strip().lower())
        if schema is None:
            raise ValueError(
                f"Unsupported FlexTPP ST mark schema '{raw_schema}'. "
                f"Expected one of {SUPPORTED_FLEX_ST_MARK_SCHEMAS}, with aliases "
                "'plain' for standard and 'labels' for enhanced."
            )
        if schema in seen:
            continue
        schemas.append(schema)
        seen.add(schema)
    return tuple(schemas)


def _parse_multittpp_mark_schemas(raw_value: str) -> tuple[str, ...]:
    schemas = _parse_easy_mark_schemas(raw_value)
    invalid_schemas = [
        schema for schema in schemas if schema not in SUPPORTED_MULTITTPP_MARK_SCHEMAS
    ]
    if invalid_schemas:
        raise ValueError(
            f"Unsupported MultiTTPP mark schema(s) {invalid_schemas}. "
            f"Expected one of {SUPPORTED_MULTITTPP_MARK_SCHEMAS}."
        )
    return schemas


def _parse_flex_conditioning_modes(raw_value: str) -> tuple[str, ...]:
    aliases = {
        "conditioned": "conditioned",
        "with_conditioning": "conditioned",
        "conditioning": "conditioned",
        "previous_day": "conditioned",
        "previous_day_summary": "conditioned",
        "no_conditioning": "no_conditioning",
        "unconditioned": "no_conditioning",
        "without_conditioning": "no_conditioning",
        "none": "no_conditioning",
        "off": "no_conditioning",
    }
    modes: list[str] = []
    seen: set[str] = set()
    for raw_mode in _parse_csv_strings(raw_value):
        mode = aliases.get(raw_mode.strip().lower())
        if mode is None:
            raise ValueError(
                f"Unsupported FlexTPP conditioning mode '{raw_mode}'. "
                f"Expected one of {SUPPORTED_FLEX_CONDITIONING_MODES}, with aliases "
                "'unconditioned' or 'none' for no_conditioning."
            )
        if mode in seen:
            continue
        modes.append(mode)
        seen.add(mode)
    return tuple(modes)


def _flex_st_schema_suffix(mark_schema: str) -> str:
    return "enhanced_marks" if mark_schema == "enhanced" else "standard_marks"


def _flex_conditioning_suffix(conditioning_mode: str) -> str:
    return "no_conditioning" if conditioning_mode == "no_conditioning" else "conditioned"


def _easy_schema_suffix(mark_schema: str) -> str:
    return "enhanced_marks" if mark_schema == "enhanced" else "plain_marks"


def _multittpp_schema_suffix(mark_schema: str) -> str:
    return "enhanced_marks" if mark_schema == "enhanced" else "standard_marks"


def _sequence_cap_cache_suffix(max_events_per_sequence: Any) -> str | None:
    if max_events_per_sequence is None:
        return None
    try:
        max_events = int(max_events_per_sequence)
    except (TypeError, ValueError):
        return f"max_events_{_safe_name(max_events_per_sequence)}"
    if max_events <= 0:
        return None
    return f"max_events_{max_events}"


def _apply_sequence_cap_dataset_dir(payload: Mapping[str, Any]) -> dict[str, Any]:
    updated_payload = copy.deepcopy(dict(payload))
    dataset_config = updated_payload.setdefault("dataset_config", {})
    suffix = _sequence_cap_cache_suffix(dataset_config.get("max_events_per_sequence"))
    if suffix is None:
        return updated_payload

    dataset_dir = Path(str(dataset_config.get("dataset_dir", "data/prediction/vital_sign_easy_tpp_dataset")))
    if dataset_dir.name != suffix:
        dataset_config["dataset_dir"] = str(dataset_dir / suffix)
    return updated_payload


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
            raise ValueError("Enhanced FlexTPP marks are currently defined for ST jobs.")
        dataset_config["event_type_mark_mode"] = "task_label"
        dataset_dir = Path(str(dataset_config.get("dataset_dir", "data/prediction/vital_sign_tpp_dataset")))
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
        dataset_dir = Path(str(dataset_config.get("dataset_dir", "data/prediction/vital_sign_tpp_dataset")))
        dataset_config["dataset_dir"] = str(dataset_dir / _flex_conditioning_suffix(conditioning_mode))
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
        dataset_config["mark_label_mode"] = "task_only"
    elif mark_schema == "enhanced":
        dataset_config.setdefault("mark_label_mode", "task_label")
    else:
        raise ValueError(f"Unsupported EasyTPP mark schema '{mark_schema}'.")

    if separate_dataset_dir:
        dataset_dir = Path(str(dataset_config.get("dataset_dir", "data/prediction/vital_sign_easy_tpp_dataset")))
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
        dataset_config["mark_label_mode"] = "task_only"
    elif mark_schema == "enhanced":
        dataset_config.setdefault("mark_label_mode", "task_label")
    else:
        raise ValueError(f"Unsupported MultiTTPP mark schema '{mark_schema}'.")

    dataset_config["include_eos_event"] = False
    if separate_dataset_dir:
        dataset_dir = Path(str(dataset_config.get("dataset_dir", "data/prediction/vital_sign_multittpp_dataset")))
        dataset_config["dataset_dir"] = str(dataset_dir / _multittpp_schema_suffix(mark_schema))

    return updated_payload


def _log_dir(args: argparse.Namespace) -> Path:
    return Path(args.stf_log_dir or os.getenv("STF_LOG_DIR", "./data/STF_LOG_DIR"))


def _temp_config_dir(args: argparse.Namespace) -> Path:
    return _log_dir(args) / "temp_configs" / "vital_sign_tpp_comparison"


def _comparison_log_dir(args: argparse.Namespace) -> Path:
    return _log_dir(args) / "comparison_logs"


def _comparison_log_path(args: argparse.Namespace, run_name: str) -> Path:
    return _comparison_log_dir(args) / f"{_safe_name(run_name)}.log"


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
        "precision": args.precision,
        "accumulate_grad_batches": args.accumulate_grad_batches,
    }
    for field_name, field_value in optional_overrides.items():
        if field_value is not None:
            model_config[field_name] = field_value
    return updated_payload


def _apply_architecture_defaults(
    payload: dict[str, Any],
    *,
    defaults: Mapping[str, Any],
    enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        return copy.deepcopy(payload)
    return _deep_merge_dicts(base=payload, updates=defaults)


def _build_flex_job_payload(
    *,
    base_payload: Mapping[str, Any],
    args: argparse.Namespace,
    run_prefix: str,
    order: str,
    mark_schema: str,
    conditioning_mode: str,
) -> ComparisonJob:
    schema_suffix = (
        f"_{_flex_st_schema_suffix(mark_schema)}"
        if order == "ST" and mark_schema == "enhanced"
        else ""
    )
    conditioning_suffix = (
        f"_{_flex_conditioning_suffix(conditioning_mode)}"
        if conditioning_mode == "no_conditioning"
        else ""
    )
    run_name = f"{run_prefix}_flex_tpp_{order.lower()}{schema_suffix}{conditioning_suffix}"
    payload = _apply_dataset_training_flags(
        _apply_flex_dataset_variant(
            base_payload,
            order=order,
            mark_schema=mark_schema,
            conditioning_mode=conditioning_mode,
        ),
        use_prepared_dataset=args.prepare_datasets,
    )
    payload["model_config"]["order"] = order
    payload = _apply_architecture_defaults(
        payload,
        defaults=FLEX_TPP_VARIANT_DEFAULTS.get(order, {}),
        enabled=not args.no_model_defaults,
    )
    payload = _apply_common_model_overrides(
        payload,
        args=args,
        run_name=run_name,
        wandb=not args.no_wandb,
    )
    return ComparisonJob(
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
        module_name="HTAMP.prediction.predictor.vital_sign_tpp_predictor",
    )


def _build_easy_job_payload(
    *,
    base_payload: Mapping[str, Any],
    args: argparse.Namespace,
    run_prefix: str,
    model_id: str,
    mark_schema: str,
    separate_dataset_dir: bool,
) -> ComparisonJob:
    schema_suffix = _easy_schema_suffix(mark_schema)
    run_name = f"{run_prefix}_easy_tpp_{_safe_name(model_id)}_{schema_suffix}"
    payload = _apply_dataset_training_flags(
        _apply_easy_mark_schema(
            base_payload,
            mark_schema=mark_schema,
            separate_dataset_dir=separate_dataset_dir,
        ),
        use_prepared_dataset=args.prepare_datasets,
    )
    payload["model_config"]["model_id"] = model_id
    payload = _apply_architecture_defaults(
        payload,
        defaults=EASY_TPP_MODEL_DEFAULTS.get(model_id, {}),
        enabled=not args.no_model_defaults,
    )
    if args.easy_max_events_per_sequence is not None:
        payload.setdefault("dataset_config", {})["max_events_per_sequence"] = int(
            args.easy_max_events_per_sequence
        )
    payload = _apply_sequence_cap_dataset_dir(payload)
    payload = _apply_common_model_overrides(
        payload,
        args=args,
        run_name=run_name,
        wandb=not args.no_wandb,
    )
    return ComparisonJob(
        family="EasyTPP",
        model_name=model_id,
        variant=f"{model_id}_{mark_schema}",
        run_name=run_name,
        config_payload=payload,
        module_name="HTAMP.prediction.predictor.vital_sign_easy_tpp_predictor",
    )


def _build_multittpp_job_payload(
    *,
    base_payload: Mapping[str, Any],
    args: argparse.Namespace,
    run_prefix: str,
    model_name: str,
    mark_schema: str,
    separate_dataset_dir: bool,
) -> ComparisonJob:
    schema_suffix = _multittpp_schema_suffix(mark_schema)
    run_name = f"{run_prefix}_multittpp_{_safe_name(model_name)}_{schema_suffix}"
    payload = _apply_dataset_training_flags(
        _apply_multittpp_mark_schema(
            base_payload,
            mark_schema=mark_schema,
            separate_dataset_dir=separate_dataset_dir,
        ),
        use_prepared_dataset=args.prepare_datasets,
    )
    payload["model_config"]["model_name"] = model_name
    payload = _apply_architecture_defaults(
        payload,
        defaults=MULTITTPP_MODEL_DEFAULTS.get(model_name, {}),
        enabled=not args.no_model_defaults,
    )
    payload = _apply_common_model_overrides(
        payload,
        args=args,
        run_name=run_name,
        wandb=not args.no_wandb,
    )
    return ComparisonJob(
        family="MultiTTPP",
        model_name=model_name,
        variant=f"{model_name}_{mark_schema}",
        run_name=run_name,
        config_payload=payload,
        module_name="HTAMP.prediction.predictor.vital_sign_multittpp_predictor",
    )


def _build_jobs(args: argparse.Namespace, run_prefix: str) -> list[ComparisonJob]:
    flex_payload = _load_json(args.flex_config_path)
    easy_payload = _load_json(args.easy_config_path)
    multittpp_payload = _load_json(args.multittpp_config_path)
    jobs: list[ComparisonJob] = []

    if not args.skip_flex:
        flex_orders = _parse_csv_strings(args.flex_orders, allowed=DEFAULT_FLEX_ORDERS)
        flex_st_mark_schemas = _parse_flex_st_mark_schemas(args.flex_st_mark_schemas)
        flex_conditioning_modes = _parse_flex_conditioning_modes(args.flex_conditioning_modes)
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
            SUPPORTED_EASY_TPP_MODELS
            if args.easy_models.strip().lower() == "all"
            else _parse_csv_strings(args.easy_models, allowed=SUPPORTED_EASY_TPP_MODELS)
        )
        easy_mark_schemas = _parse_easy_mark_schemas(args.easy_mark_schemas)
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
            SUPPORTED_MULTITTPP_MODELS
            if args.multittpp_models.strip().lower() == "all"
            else _parse_csv_strings(args.multittpp_models, allowed=SUPPORTED_MULTITTPP_MODELS)
        )
        multittpp_mark_schemas = _parse_multittpp_mark_schemas(args.multittpp_mark_schemas)
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


def _build_prepare_payload(base_payload: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(base_payload))
    dataset_config = payload.setdefault("dataset_config", {})
    dataset_config["use_saved_dataset"] = False
    dataset_config["preprocess_data"] = True
    dataset_config["save_data"] = True
    payload.setdefault("model_config", {})["wandb"] = False
    return payload


DATASET_PREPARE_MODULE_BY_FAMILY = {
    "FlexTPP": "HTAMP.prediction.data_provider.vital_sign_tpp_dataset",
    "EasyTPP": "HTAMP.prediction.data_provider.vital_sign_easy_tpp_dataset",
    "MultiTTPP": "HTAMP.prediction.data_provider.vital_sign_multittpp_dataset",
}


def _dataset_prepare_key(
    *,
    module_name: str,
    payload: Mapping[str, Any],
) -> tuple[str, str]:
    dataset_config = dict(payload.get("dataset_config", {}))
    for workflow_field in (
        "use_saved_dataset",
        "preprocess_data",
        "save_data",
        "use_saved_request_data",
    ):
        dataset_config.pop(workflow_field, None)
    return (
        module_name,
        json.dumps(dataset_config, sort_keys=True, default=str),
    )


def _prepare_datasets_for_jobs(
    args: argparse.Namespace,
    *,
    jobs: Sequence[ComparisonJob],
    run_prefix: str,
    module_by_family: Mapping[str, str],
) -> None:
    prepared_keys: set[tuple[str, str]] = set()
    for job in jobs:
        module_name = module_by_family.get(job.family)
        if module_name is None:
            raise ValueError(f"No dataset preparation module configured for family '{job.family}'.")

        prepare_payload = _build_prepare_payload(job.config_payload)
        prepare_key = _dataset_prepare_key(
            module_name=module_name,
            payload=prepare_payload,
        )
        if prepare_key in prepared_keys:
            continue
        prepared_keys.add(prepare_key)

        _run_dataset_prepare(
            args,
            label=f"{job.family}-{job.variant}",
            module_name=module_name,
            payload=prepare_payload,
            run_prefix=run_prefix,
        )


def _selected_flex_dataset_variants(args: argparse.Namespace) -> tuple[tuple[str, str], ...]:
    flex_orders = _parse_csv_strings(args.flex_orders, allowed=DEFAULT_FLEX_ORDERS)
    flex_st_mark_schemas = _parse_flex_st_mark_schemas(args.flex_st_mark_schemas)
    flex_conditioning_modes = _parse_flex_conditioning_modes(args.flex_conditioning_modes)
    variants: list[tuple[str, str]] = []

    if "STP" in flex_orders or ("ST" in flex_orders and "standard" in flex_st_mark_schemas):
        variants.extend(("standard", conditioning_mode) for conditioning_mode in flex_conditioning_modes)
    if "ST" in flex_orders and "enhanced" in flex_st_mark_schemas:
        variants.extend(("enhanced", conditioning_mode) for conditioning_mode in flex_conditioning_modes)
    return tuple(variants)


def _build_process_env(
    args: argparse.Namespace,
    *,
    gpu_id: int | None,
    run_prefix: str,
    job_type: str,
) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
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


def _prepare_datasets(
    args: argparse.Namespace,
    *,
    jobs: Sequence[ComparisonJob],
    run_prefix: str,
) -> None:
    if args.skip_flex and args.skip_easy and args.skip_multittpp:
        return
    if not args.prepare_datasets:
        print("Dataset pre-build is disabled; training jobs will use config workflow flags.")
        return

    _prepare_datasets_for_jobs(
        args,
        jobs=jobs,
        run_prefix=run_prefix,
        module_by_family=DATASET_PREPARE_MODULE_BY_FAMILY,
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
    log_path = _comparison_log_path(args, run_name=job.run_name)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    log_file.write(f"Command: {' '.join(command)}\n")
    log_file.write(f"GPU: {gpu_label}\n\n")
    log_file.flush()
    print(f"Starting {job.run_name} on {gpu_label}: {' '.join(command)}")
    print(f"  Log: {log_path}")
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
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    return RunningJob(
        job=job,
        process=process,
        temp_config_path=temp_config_path,
        log_path=log_path,
        log_file=log_file,
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
    running_job.log_file.close()
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
    if status != "success":
        print(f"  Failure log: {running_job.log_path}")
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
        "log_path": str(running_job.log_path),
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
        running_job.log_file.close()
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
                "log_path": str(running_job.log_path),
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
                            "log_path": "",
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
    parser.add_argument("--multittpp_config_path", default=DEFAULT_MULTITTPP_CONFIG_PATH)
    parser.add_argument(
        "--easy_models",
        default="all",
        help=(
            "Comma-separated EasyTPP model ids, or 'all'. Available: "
            f"{', '.join(SUPPORTED_EASY_TPP_MODELS)}"
        ),
    )
    parser.add_argument(
        "--easy_mark_schemas",
        default=",".join(DEFAULT_EASY_MARK_SCHEMAS),
        help=(
            "Comma-separated EasyTPP mark schemas to train. "
            "'enhanced' uses low/medium/high measurement labels; "
            "'plain' uses only request task marks. Defaults to enhanced,plain."
        ),
    )
    parser.add_argument(
        "--flex_orders",
        default=",".join(DEFAULT_FLEX_ORDERS),
        help="Comma-separated FlexTPP event orders. Defaults to ST,STP.",
    )
    parser.add_argument(
        "--flex_st_mark_schemas",
        default=",".join(DEFAULT_FLEX_ST_MARK_SCHEMAS),
        help=(
            "Comma-separated FlexTPP ST mark schemas to train. "
            "'standard' uses base request task types; 'enhanced' uses "
            "low/medium/high measurement labels in the categorical T mark. "
            "Defaults to standard,enhanced. STP always uses standard event types."
        ),
    )
    parser.add_argument(
        "--flex_conditioning_modes",
        default=",".join(DEFAULT_FLEX_CONDITIONING_MODES),
        help=(
            "Comma-separated FlexTPP conditioning modes to train. "
            "'conditioned' uses previous-day summary conditioning; "
            "'no_conditioning' disables it. Defaults to conditioned,no_conditioning."
        ),
    )
    parser.add_argument(
        "--multittpp_models",
        default=",".join(DEFAULT_MULTITTPP_MODELS),
        help=(
            "Comma-separated MultiTTPP model names, or 'all'. Available: "
            f"{', '.join(SUPPORTED_MULTITTPP_MODELS)}"
        ),
    )
    parser.add_argument(
        "--multittpp_mark_schemas",
        default=",".join(DEFAULT_MULTITTPP_MARK_SCHEMAS),
        help=(
            "Comma-separated MultiTTPP mark schemas to train. "
            "'enhanced' uses low/medium/high measurement labels; "
            "'plain' uses only request task marks. Defaults to enhanced,plain."
        ),
    )
    parser.add_argument("--skip_easy", action="store_true")
    parser.add_argument("--skip_flex", action="store_true")
    parser.add_argument("--skip_multittpp", action="store_true")
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
        "--precision",
        default=None,
        help=(
            "Optional Lightning precision override, e.g. 32-true, 16-mixed, "
            "or bf16-mixed."
        ),
    )
    parser.add_argument("--accumulate_grad_batches", type=int, default=None)
    parser.add_argument(
        "--easy_max_events_per_sequence",
        type=int,
        default=None,
        help=(
            "Optional cap for EasyTPP source sequence chunks. This is useful "
            "for attention-heavy EasyTPP models whose memory grows quadratically "
            "with sequence length."
        ),
    )
    parser.add_argument(
        "--no_model_defaults",
        action="store_true",
        help=(
            "Use the base JSON configs as-is, instead of applying the runner's "
            "per-architecture starting hyperparameters."
        ),
    )
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
    print(f"Per-architecture defaults: {not args.no_model_defaults}")
    print(f"Summary path: {summary_path}")
    for job in jobs:
        print(f"  - {job.run_name} [{job.family}/{job.variant}]")

    if args.dry_run:
        print("Dry run requested; no datasets or models were trained.")
        return 0

    _prepare_datasets(args, jobs=jobs, run_prefix=run_prefix)
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
