from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from HTAMP.prediction.configs.vital_sign_easy_tpp_config import (
    VitalSignEasyTPPTrainingConfig,
)
from HTAMP.prediction.configs.vital_sign_multittpp_config import (
    VitalSignMultiTTPPTrainingConfig,
)
from HTAMP.prediction.configs.vital_sign_tpp_config import (
    VitalSignTPPTrainingConfig,
)
from HTAMP.prediction.data_provider.vital_sign_easy_tpp_dataset import (
    VitalSignEasyTPPDatasetBundle,
    VitalSignEasyTPPSequenceRecord,
    build_vital_sign_easy_tpp_dataset_bundle,
)
from HTAMP.prediction.data_provider.vital_sign_tpp_dataset import (
    EOS_EVENT_TYPE_NAME,
    VitalSignTPPDatasetBundle,
    VitalSignTPPSequenceRecord,
    build_vital_sign_tpp_dataset_bundle,
)
from HTAMP.prediction.data_provider.vital_sign_multittpp_dataset import (
    VitalSignMultiTTPPDatasetBundle,
    build_vital_sign_multittpp_dataset_bundle,
)
from HTAMP.prediction.module.vital_sign_easy_tpp_module import (
    VitalSignEasyTPPModule,
    _ThinningConfigAdapter,
)
from HTAMP.prediction.module.vital_sign_multittpp_module import VitalSignMultiTTPPModule
from HTAMP.prediction.module.vital_sign_tpp_module import VitalSignTPPModule
from HTAMP.prediction.metrics.otd_metric import Event, MOTDConfig, marked_otd
from HTAMP.prediction.point_process_models.easyTPP.torch_intensity_free import (
    LogNormalMixtureDistribution,
    clamp_preserve_gradients,
)
from HTAMP.prediction.point_process_models.easyTPP.torch_thinning import EventSampler
from HTAMP.prediction.prediction_handlers.vital_sign_tpp_prediction_handler import (
    _sample_future_events_from_prefix,
)

DEFAULT_COMPARISON_SUMMARY_GLOB = "data/prediction/vital_sign_tpp_comparison/*_summary.csv"
DEFAULT_OUTPUT_DIR = "data/prediction/vital_sign_tpp_otd_evaluation"
DEMAND_LEVELS = ("high", "medium", "low")
DEMAND_LEVEL_CHOICES = ("all", *DEMAND_LEVELS)
DEFAULT_DEMAND_WEEK_SETS_BY_FLOOR: dict[str, dict[int | None, set[tuple[int, int]]]] = {
    "high": {
        None: {(2024, 40), (2025, 5)},
        2: {(2025, 6), (2025, 8)},
        3: {(2025, 5), (2025, 10)},
        7: {(2024, 39), (2024, 40)},
        9: {(2024, 43), (2025, 6)},
    },
    "medium": {
        None: {(2024, 44), (2025, 14)},
        2: {(2024, 44), (2025, 14)},
        3: {(2024, 44), (2025, 14)},
        7: {(2024, 44), (2025, 14)},
        9: {(2024, 44), (2025, 14)},
    },
    "low": {
        None: {(2024, 27), (2024, 36)},
        2: {(2024, 27), (2024, 28)},
        3: {(2024, 46), (2025, 2)},
        7: {(2024, 46), (2025, 26)},
        9: {(2025, 19), (2025, 21)},
    },
}


@dataclass(frozen=True)
class ModelSpec:
    family: str
    model_name: str
    variant: str
    run_name: str
    checkpoint_path: Path
    metrics_summary_path: Path | None
    training_config: dict[str, Any]


@dataclass(frozen=True)
class EvaluationContext:
    easy_bundle: VitalSignEasyTPPDatasetBundle
    easy_records: list[VitalSignEasyTPPSequenceRecord]
    mark_encoder: "DiscreteMarkEncoder"
    time_scales: dict[str, list[float]]
    default_tau: float


class DiscreteMarkEncoder:
    """Label raw vital-sign events with the EasyTPP discrete joint mark schema."""

    def __init__(self, metadata: Mapping[str, Any]) -> None:
        self.metadata = dict(metadata)
        self.included_tasks = [
            str(task_name)
            for task_name in self.metadata.get("included_tasks", [])
        ]
        self.mark_names = [
            str(mark_name)
            for mark_name in self.metadata.get("mark_names", [])
        ]
        self.mark_name_set = set(self.mark_names)
        self.label_names = tuple(
            str(label_name)
            for label_name in self.metadata.get("label_names", ("low", "medium", "high"))
        )
        if len(self.label_names) != 3:
            raise ValueError("Expected exactly three EasyTPP label names.")
        self.missing_label = str(self.metadata.get("missing_label", "unknown"))
        self.mark_label_mode = str(self.metadata.get("mark_label_mode", "task_label"))
        config_snapshot = dict(self.metadata.get("config_snapshot", {}))
        self.drop_missing_measurement_events = bool(
            self.metadata.get(
                "drop_missing_measurement_events",
                config_snapshot.get("drop_missing_measurement_events", False),
            )
        )
        self.label_component_by_task = {
            str(task_name): [str(component) for component in component_names]
            for task_name, component_names in dict(
                self.metadata.get("label_component_by_task", {})
            ).items()
        }
        self.thresholds_by_task_component = {
            str(task_name): {
                str(component): (float(pair[0]), float(pair[1]))
                for component, pair in dict(component_thresholds).items()
            }
            for task_name, component_thresholds in dict(
                self.metadata.get("thresholds_by_task_component", {})
            ).items()
        }

    def base_task(self, mark_name: str) -> str:
        for task_name in sorted(self.included_tasks, key=len, reverse=True):
            if mark_name == task_name or mark_name.startswith(f"{task_name}__"):
                return task_name
        return str(mark_name).split("__", 1)[0]

    def label_value(self, value: Any, lower: float, upper: float) -> str:
        numeric_value = _finite_float_or_none(value)
        if numeric_value is None:
            return self.missing_label
        if not lower < upper:
            return self.label_names[1]
        if numeric_value <= lower:
            return self.label_names[0]
        if numeric_value >= upper:
            return self.label_names[2]
        return self.label_names[1]

    def mark_name(self, *, task_name: str, component_labels: Sequence[tuple[str, str]]) -> str:
        if len(component_labels) > 1:
            label_suffix = "__".join(
                f"{component_name}_{label_name}"
                for component_name, label_name in component_labels
            )
            return f"{task_name}__{label_suffix}"

        component_name, label_name = component_labels[0]
        if self.mark_label_mode == "task_component_label":
            return f"{task_name}__{component_name}__{label_name}"
        return f"{task_name}__{label_name}"

    def label_event(self, *, task_name: str, properties: Mapping[str, Any]) -> str | None:
        if task_name == EOS_EVENT_TYPE_NAME:
            return EOS_EVENT_TYPE_NAME

        component_names = self.label_component_by_task.get(task_name)
        component_thresholds = self.thresholds_by_task_component.get(task_name)
        if not component_names or not component_thresholds:
            return None

        component_labels: list[tuple[str, str]] = []
        for component_name in component_names:
            numeric_value = _finite_float_or_none(properties.get(component_name))
            if numeric_value is None and self.drop_missing_measurement_events:
                return None
            lower, upper = component_thresholds[component_name]
            component_labels.append(
                (
                    component_name,
                    self.label_value(numeric_value, lower=lower, upper=upper),
                )
            )

        mark_name = self.mark_name(
            task_name=task_name,
            component_labels=component_labels,
        )
        return mark_name if mark_name in self.mark_name_set else None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_repo_relative_path(path_str: str | Path | None) -> Path | None:
    if path_str in (None, ""):
        return None
    path = Path(path_str)
    if path.is_absolute():
        return path
    return _repo_root() / path


def _load_json(path: str | Path) -> dict[str, Any]:
    resolved_path = _resolve_repo_relative_path(path)
    if resolved_path is None:
        raise ValueError("Missing JSON path.")
    with resolved_path.open("r", encoding="utf-8") as json_file:
        payload = json.load(json_file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected '{resolved_path}' to contain a JSON object.")
    return payload


def _finite_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric_value):
        return None
    return numeric_value


def _floor_or_none(value: Any) -> int | None:
    numeric_value = _finite_float_or_none(value)
    if numeric_value is None:
        return None
    return int(numeric_value)


def _timestamp_iso_week(value: Any) -> tuple[int, int] | None:
    timestamp_text = str(value or "").strip()
    if not timestamp_text:
        return None
    if timestamp_text.endswith("Z"):
        timestamp_text = f"{timestamp_text[:-1]}+00:00"
    try:
        timestamp = datetime.datetime.fromisoformat(timestamp_text)
    except ValueError:
        try:
            timestamp = datetime.datetime.combine(
                datetime.date.fromisoformat(timestamp_text[:10]),
                datetime.time.min,
            )
        except ValueError:
            return None
    iso_calendar = timestamp.isocalendar()
    return int(iso_calendar.year), int(iso_calendar.week)


def _demand_weeks_for_floor(demand_level: str, floor: int | None) -> set[tuple[int, int]]:
    weeks_by_floor = DEFAULT_DEMAND_WEEK_SETS_BY_FLOOR[str(demand_level)]
    if floor is not None and floor in weeks_by_floor:
        return weeks_by_floor[floor]
    return weeks_by_floor[None]


def _raw_event_demand_level(raw_event: Mapping[str, Any]) -> str | None:
    if str(raw_event.get("task_name", "")) == EOS_EVENT_TYPE_NAME:
        return None
    week_key = _timestamp_iso_week(raw_event.get("timestamp"))
    if week_key is None:
        return None
    floor = _floor_or_none(raw_event.get("floor"))
    for demand_level in DEMAND_LEVELS:
        if week_key in _demand_weeks_for_floor(demand_level=demand_level, floor=floor):
            return demand_level
    return None


def _record_demand_event_counts(
    record: VitalSignEasyTPPSequenceRecord,
) -> dict[str, int]:
    counts = {demand_level: 0 for demand_level in DEMAND_LEVELS}
    for raw_event in record.raw_events:
        demand_level = _raw_event_demand_level(raw_event)
        if demand_level is not None:
            counts[demand_level] += 1
    return counts


def _record_primary_demand_level(record: VitalSignEasyTPPSequenceRecord) -> str | None:
    counts = _record_demand_event_counts(record)
    if not any(counts.values()):
        return None
    return max(DEMAND_LEVELS, key=lambda demand_level: counts[demand_level])


def _record_matches_demand_level(
    record: VitalSignEasyTPPSequenceRecord,
    *,
    demand_level: str,
    assignment_strategy: str,
) -> bool:
    if demand_level == "all":
        return True
    counts = _record_demand_event_counts(record)
    if assignment_strategy == "any":
        return counts[demand_level] > 0
    if assignment_strategy == "strict":
        return counts[demand_level] > 0 and all(
            count == 0
            for other_level, count in counts.items()
            if other_level != demand_level
        )
    return _record_primary_demand_level(record) == demand_level


def _demand_sequence_counts(
    records: Sequence[VitalSignEasyTPPSequenceRecord],
) -> dict[str, int]:
    counts = {demand_level: 0 for demand_level in DEMAND_LEVELS}
    counts["unmatched"] = 0
    for record in records:
        demand_level = _record_primary_demand_level(record)
        if demand_level is None:
            counts["unmatched"] += 1
        else:
            counts[demand_level] += 1
    return counts


def _serializable_demand_week_sets_by_floor() -> dict[str, dict[str, list[str]]]:
    return {
        demand_level: {
            ("default" if floor is None else str(floor)): [
                f"{year:04d}W{week:02d}"
                for year, week in sorted(weeks)
            ]
            for floor, weeks in weeks_by_floor.items()
        }
        for demand_level, weeks_by_floor in DEFAULT_DEMAND_WEEK_SETS_BY_FLOOR.items()
    }


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value.lower()).strip("_")


def _latest_comparison_summary() -> Path | None:
    candidates = sorted(
        _repo_root().glob(DEFAULT_COMPARISON_SUMMARY_GLOB),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _coerce_comparison_summary_path(raw_path: str | None) -> Path:
    if raw_path:
        resolved = _resolve_repo_relative_path(raw_path)
        if resolved is None:
            raise ValueError("Missing comparison summary path.")
        return resolved
    latest_path = _latest_comparison_summary()
    if latest_path is None:
        raise ValueError(
            "No comparison summary was provided and no default summary was found under "
            f"'{DEFAULT_COMPARISON_SUMMARY_GLOB}'."
        )
    return latest_path


def _metrics_summary_path_from_row(row: Mapping[str, str], stf_log_dir: str | None) -> Path | None:
    raw_metrics_path = row.get("metrics_summary_path")
    if raw_metrics_path:
        return _resolve_repo_relative_path(raw_metrics_path)
    run_name = row.get("run_name")
    if not run_name:
        return None
    log_dir = Path(stf_log_dir or os.getenv("STF_LOG_DIR", "./data/STF_LOG_DIR"))
    return _resolve_repo_relative_path(log_dir / run_name / "metrics_summary.json")


def _read_training_config_from_metrics(metrics_summary_path: Path | None) -> dict[str, Any]:
    if metrics_summary_path is None or not metrics_summary_path.exists():
        return {}
    payload = _load_json(metrics_summary_path)
    training_config = payload.get("training_config", {})
    return dict(training_config) if isinstance(training_config, Mapping) else {}


def _checkpoint_path_from_row(
    row: Mapping[str, str],
    metrics_summary_path: Path | None,
) -> Path | None:
    raw_checkpoint_path = row.get("best_checkpoint_path")
    if raw_checkpoint_path:
        return _resolve_repo_relative_path(raw_checkpoint_path)
    if metrics_summary_path is None or not metrics_summary_path.exists():
        return None
    payload = _load_json(metrics_summary_path)
    raw_checkpoint_path = payload.get("best_checkpoint_path")
    return _resolve_repo_relative_path(raw_checkpoint_path)


def _load_model_specs(
    *,
    comparison_summary_path: Path,
    stf_log_dir: str | None,
    selected_runs: set[str] | None,
) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    with comparison_summary_path.open("r", newline="", encoding="utf-8") as summary_file:
        reader = csv.DictReader(summary_file)
        for row in reader:
            run_name = str(row.get("run_name", "")).strip()
            if not run_name:
                continue
            if selected_runs is not None and run_name not in selected_runs:
                continue
            if str(row.get("status", "")).strip().lower() != "success":
                continue

            metrics_summary_path = _metrics_summary_path_from_row(row, stf_log_dir)
            training_config = _read_training_config_from_metrics(metrics_summary_path)
            checkpoint_path = _checkpoint_path_from_row(row, metrics_summary_path)
            if checkpoint_path is None or not checkpoint_path.exists():
                _log(f"Skipping {run_name}: checkpoint not found at {checkpoint_path}.")
                continue
            if not training_config:
                _log(f"Skipping {run_name}: metrics summary has no training_config.")
                continue

            specs.append(
                ModelSpec(
                    family=str(row.get("family", "")),
                    model_name=str(row.get("model_name", "")),
                    variant=str(row.get("variant", "")),
                    run_name=run_name,
                    checkpoint_path=checkpoint_path,
                    metrics_summary_path=metrics_summary_path,
                    training_config=training_config,
                )
            )
    if not specs:
        raise ValueError(f"No successful model specs were found in {comparison_summary_path}.")
    return specs


def _apply_dataset_load_flags(dataset_config: Any, *, use_saved_datasets: bool) -> Any:
    dataset_config.use_saved_dataset = bool(use_saved_datasets)
    dataset_config.preprocess_data = not bool(use_saved_datasets)
    dataset_config.save_data = not bool(use_saved_datasets)
    return dataset_config


def _canonical_easy_training_config(
    *,
    args: argparse.Namespace,
    model_specs: Sequence[ModelSpec],
) -> VitalSignEasyTPPTrainingConfig:
    if args.easy_config_path:
        return VitalSignEasyTPPTrainingConfig.from_json_file(args.easy_config_path)

    for spec in model_specs:
        if spec.family.lower() == "easytpp":
            return VitalSignEasyTPPTrainingConfig.from_dict(spec.training_config)

    raise ValueError(
        "The OTD evaluator needs an EasyTPP dataset config to define the discrete "
        "low/medium/high joint marks. Provide --easy_config_path when evaluating "
        "FlexTPP-only summaries."
    )


def _build_evaluation_context(
    *,
    args: argparse.Namespace,
    model_specs: Sequence[ModelSpec],
) -> EvaluationContext:
    easy_training_config = _canonical_easy_training_config(
        args=args,
        model_specs=model_specs,
    )
    easy_dataset_config = _apply_dataset_load_flags(
        easy_training_config.dataset_config,
        use_saved_datasets=args.use_saved_datasets,
    )
    easy_bundle = build_vital_sign_easy_tpp_dataset_bundle(
        dataset_config=easy_dataset_config,
    )
    easy_records = easy_bundle.get_raw_records(args.split)
    demand_counts_before_filter = _demand_sequence_counts(easy_records)
    if args.demand_level != "all":
        easy_records = [
            record
            for record in easy_records
            if _record_matches_demand_level(
                record,
                demand_level=args.demand_level,
                assignment_strategy=args.demand_sequence_assignment,
            )
        ]
        if not easy_records:
            raise ValueError(
                f"No {args.split} sequences matched demand_level='{args.demand_level}' "
                f"with demand_sequence_assignment='{args.demand_sequence_assignment}'."
            )
    if args.max_sequences is not None:
        max_sequences = int(args.max_sequences)
        if max_sequences < 0:
            raise ValueError("max_sequences must be non-negative when provided.")
        if args.sequence_subset_strategy == "random":
            rng = random.Random(int(args.seed))
            easy_records = list(easy_records)
            rng.shuffle(easy_records)
            easy_records = easy_records[:max_sequences]
        else:
            easy_records = easy_records[:max_sequences]
    _log(
        "Demand sequence counts before filtering: "
        + ", ".join(
            f"{level}={count}"
            for level, count in demand_counts_before_filter.items()
        )
    )
    _log(
        f"Selected {len(easy_records)} {args.split} sequences for "
        f"demand_level='{args.demand_level}'."
    )
    mark_encoder = DiscreteMarkEncoder(easy_bundle.metadata)
    time_scales, default_tau = _build_time_scales(
        easy_bundle=easy_bundle,
        mark_encoder=mark_encoder,
    )
    return EvaluationContext(
        easy_bundle=easy_bundle,
        easy_records=easy_records,
        mark_encoder=mark_encoder,
        time_scales=time_scales,
        default_tau=default_tau,
    )


def _easy_record_events(
    *,
    record: VitalSignEasyTPPSequenceRecord,
    mark_encoder: DiscreteMarkEncoder,
) -> list[Event]:
    events: list[Event] = []
    for event_time, mark_name in zip(record.time_seqs, record.mark_names):
        if mark_name == EOS_EVENT_TYPE_NAME:
            continue
        events.append(
            Event(
                time=float(event_time),
                event_type=mark_encoder.base_task(mark_name),
                mark=str(mark_name),
            )
        )
    return events


def _easy_record_non_eos_raw_events(
    record: VitalSignEasyTPPSequenceRecord,
) -> list[dict[str, object]]:
    return [
        dict(raw_event)
        for raw_event, mark_name in zip(record.raw_events, record.mark_names)
        if mark_name != EOS_EVENT_TYPE_NAME
    ]


def _build_time_scales(
    *,
    easy_bundle: VitalSignEasyTPPDatasetBundle,
    mark_encoder: DiscreteMarkEncoder,
) -> tuple[dict[str, list[float]], float]:
    scales: dict[str, list[float]] = {}
    all_gaps: list[float] = []
    for record in easy_bundle.get_raw_records("train"):
        events = _easy_record_events(record=record, mark_encoder=mark_encoder)
        previous_time_by_type: dict[str, float] = {}
        previous_time: float | None = None
        for event in events:
            if previous_time is not None:
                gap = max(0.0, float(event.time) - previous_time)
                if gap > 0.0:
                    all_gaps.append(gap)
            previous_type_time = previous_time_by_type.get(str(event.event_type))
            if previous_type_time is not None:
                gap = max(0.0, float(event.time) - previous_type_time)
                if gap > 0.0:
                    scales.setdefault(str(event.event_type), []).append(gap)
            previous_time_by_type[str(event.event_type)] = float(event.time)
            previous_time = float(event.time)

    default_tau = float(np.median(np.asarray(all_gaps, dtype=np.float64))) if all_gaps else 1.0
    return scales, max(default_tau, 1e-12)


def _make_otd_config(args: argparse.Namespace, default_tau: float) -> MOTDConfig:
    return MOTDConfig(
        alpha=float(args.time_weight),
        beta=float(args.type_weight),
        gamma=0.0,
        c_del=float(args.delete_cost),
        c_ins=float(args.insert_cost),
        default_tau=float(default_tau),
        mark_mode="categorical",
        hard_type=not bool(args.soft_type_matching),
    )


def _log(message: str) -> None:
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", flush=True)


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(float(seconds)) or float(seconds) < 0.0:
        return "unknown"
    total_seconds = int(round(float(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def _progress_fraction(completed: int, total: int) -> float:
    if total <= 0:
        return 1.0
    return min(1.0, max(0.0, float(completed) / float(total)))


def _eta_seconds(*, completed: int, total: int, elapsed_seconds: float) -> float | None:
    fraction = _progress_fraction(completed, total)
    if fraction <= 0.0 or total <= 0:
        return None
    return elapsed_seconds * (1.0 - fraction) / fraction


def _log_model_progress(
    *,
    run_name: str,
    completed_sequences: int,
    total_sequences: int,
    completed_prefixes: int,
    total_prefixes: int,
    completed_samples: int,
    total_samples: int,
    completed_rollout_event_budget: int,
    total_rollout_event_budget: int,
    started_at: float,
) -> None:
    elapsed_seconds = time.perf_counter() - started_at
    fraction = _progress_fraction(completed_samples, total_samples)
    eta = _eta_seconds(
        completed=completed_samples,
        total=total_samples,
        elapsed_seconds=elapsed_seconds,
    )
    _log(
        f"{run_name}: {fraction:.1%} complete | "
        f"sequences {completed_sequences}/{total_sequences}, "
        f"prefixes {completed_prefixes}/{total_prefixes}, "
        f"OTD samples {completed_samples}/{total_samples}, "
        f"rollout-event budget {completed_rollout_event_budget}/{total_rollout_event_budget} | "
        f"elapsed {_format_duration(elapsed_seconds)}, ETA {_format_duration(eta)}"
    )


def _progress_intervals(args: argparse.Namespace) -> tuple[int, int]:
    return (
        max(1, int(args.progress_every)),
        max(0, int(args.progress_sample_interval)),
    )


def _sync_device_for_timing(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize(device)
    except Exception:
        torch.cuda.synchronize()


def _start_inference_timer(device: torch.device) -> float:
    _sync_device_for_timing(device)
    return time.perf_counter()


def _stop_inference_timer(device: torch.device, started_at: float) -> float:
    _sync_device_for_timing(device)
    return float(time.perf_counter() - started_at)


def _summary_stats(values: Sequence[float]) -> tuple[float, float]:
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        return float("nan"), float("nan")
    if len(finite_values) == 1:
        return finite_values[0], 0.0
    return mean(finite_values), stdev(finite_values)


def _raw_event_key(raw_event: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(raw_event.get("timestamp", "")),
        str(raw_event.get("task_name", "")),
        str(raw_event.get("task_index", "")),
    )


def _flex_record_key(record: VitalSignTPPSequenceRecord) -> tuple[str, str, str, str]:
    return (
        str(record.patient_id),
        str(record.encounter_id),
        str(record.sequence_start_timestamp),
        str(record.sequence_end_timestamp),
    )


def _easy_record_key(record: VitalSignEasyTPPSequenceRecord) -> tuple[str, str, str, str]:
    return (
        str(record.patient_id),
        str(record.encounter_id),
        str(record.sequence_start_timestamp),
        str(record.sequence_end_timestamp),
    )


def _easy_records_by_key(
    easy_bundle: VitalSignEasyTPPDatasetBundle,
    split: str,
) -> dict[tuple[str, str, str, str], VitalSignEasyTPPSequenceRecord]:
    return {
        _easy_record_key(record): record
        for record in easy_bundle.get_raw_records(split)
    }


def _flex_records_by_key(
    flex_bundle: VitalSignTPPDatasetBundle,
    split: str,
) -> dict[tuple[str, str, str, str], VitalSignTPPSequenceRecord]:
    return {
        _flex_record_key(record): record
        for record in flex_bundle.get_raw_records(split)
    }


def _matched_flex_events_for_easy_record(
    *,
    easy_record: VitalSignEasyTPPSequenceRecord,
    flex_record: VitalSignTPPSequenceRecord,
) -> list[tuple[float, float, int, dict[str, float]]]:
    matched_events: list[tuple[float, float, int, dict[str, float]]] = []
    target_events = _easy_record_non_eos_raw_events(easy_record)
    flex_index = 0
    for target_event in target_events:
        target_key = _raw_event_key(target_event)
        found_match = False
        while flex_index < len(flex_record.raw_events):
            if _raw_event_key(flex_record.raw_events[flex_index]) == target_key:
                start_time, end_time, event_type, event_props = flex_record.events[flex_index]
                matched_events.append(
                    (
                        float(start_time),
                        float(end_time),
                        int(event_type),
                        {
                            str(key): float(value)
                            for key, value in dict(event_props).items()
                        },
                    )
                )
                flex_index += 1
                found_match = True
                break
            flex_index += 1
        if not found_match:
            raise ValueError(
                "Could not align EasyTPP canonical event to the FlexTPP record: "
                f"{target_key}."
            )
    return matched_events


def _future_event_count(args: argparse.Namespace, true_events: Sequence[Event], prefix_len: int) -> int:
    remaining = max(0, len(true_events) - prefix_len)
    if remaining <= 0:
        return 0
    if args.max_future_events is None or int(args.max_future_events) <= 0:
        return remaining
    return min(remaining, int(args.max_future_events))


def _prefix_lengths(args: argparse.Namespace, true_events: Sequence[Event]) -> list[int]:
    max_prefix = len(true_events) - 1
    if max_prefix < args.min_prefix_events:
        return []
    prefix_lengths = list(range(int(args.min_prefix_events), max_prefix + 1))
    prefix_stride = max(1, int(args.prefix_stride))
    if prefix_stride > 1 and prefix_lengths:
        last_prefix = prefix_lengths[-1]
        prefix_lengths = prefix_lengths[::prefix_stride]
        if prefix_lengths[-1] != last_prefix:
            prefix_lengths.append(last_prefix)

    if args.max_prefixes_per_sequence is not None:
        max_prefix_count = int(args.max_prefixes_per_sequence)
        if max_prefix_count <= 0:
            return []
        if len(prefix_lengths) > max_prefix_count:
            if args.prefix_subset_strategy == "first":
                prefix_lengths = prefix_lengths[:max_prefix_count]
            else:
                selected_indices = np.linspace(
                    0,
                    len(prefix_lengths) - 1,
                    num=max_prefix_count,
                    dtype=int,
                )
                prefix_lengths = [
                    prefix_lengths[index]
                    for index in sorted(set(int(index) for index in selected_indices))
                ]
    return prefix_lengths


def _prefix_work_for_records(
    *,
    args: argparse.Namespace,
    records: Sequence[VitalSignEasyTPPSequenceRecord],
    mark_encoder: DiscreteMarkEncoder,
) -> tuple[int, int, int]:
    total_prefixes = 0
    total_rollout_event_budget = 0
    for record in records:
        true_events = _easy_record_events(record=record, mark_encoder=mark_encoder)
        for prefix_len in _prefix_lengths(args, true_events):
            future_count = _future_event_count(args, true_events, prefix_len)
            if future_count <= 0:
                continue
            total_prefixes += 1
            total_rollout_event_budget += int(future_count) * int(args.num_samples)
    return (
        total_prefixes,
        total_prefixes * int(args.num_samples),
        total_rollout_event_budget,
    )


def _first_event_pool(easy_bundle: VitalSignEasyTPPDatasetBundle) -> list[tuple[float, int]]:
    first_events: list[tuple[float, int]] = []
    for record in easy_bundle.get_raw_records("train"):
        for event_time, type_id, mark_name in zip(
            record.time_seqs,
            record.type_seqs,
            record.mark_names,
        ):
            if mark_name == EOS_EVENT_TYPE_NAME:
                continue
            first_events.append((float(event_time), int(type_id)))
            break
    if not first_events:
        raise ValueError("The EasyTPP training split has no first events to seed empty prefixes.")
    return first_events


def _easy_batch_from_prefix(
    *,
    times: Sequence[float],
    type_ids: Sequence[int],
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    if not times or not type_ids:
        raise ValueError("EasyTPP sampling needs at least one conditioning event.")
    if len(times) != len(type_ids):
        raise ValueError("times and type_ids must have the same length.")

    time_values = [float(value) for value in times]
    deltas = [0.0]
    for previous_time, current_time in zip(time_values[:-1], time_values[1:]):
        deltas.append(max(0.0, float(current_time) - float(previous_time)))

    time_seqs = torch.as_tensor([time_values], dtype=torch.float32, device=device)
    time_delta_seqs = torch.as_tensor([deltas], dtype=torch.float32, device=device)
    type_seqs = torch.as_tensor([[int(value) for value in type_ids]], dtype=torch.long, device=device)
    seq_non_pad_mask = torch.ones_like(type_seqs, dtype=torch.bool, device=device)
    seq_len = int(type_seqs.shape[1])
    attention_mask = torch.triu(
        torch.ones((seq_len, seq_len), dtype=torch.bool, device=device),
        diagonal=1,
    ).unsqueeze(0)
    return time_seqs, time_delta_seqs, type_seqs, seq_non_pad_mask, attention_mask


def _ensure_easy_event_sampler(
    *,
    model: VitalSignEasyTPPModule,
    thinning_payload: Mapping[str, Any],
    device: torch.device,
) -> None:
    easy_model = model.easy_tpp_model
    thinning = _ThinningConfigAdapter(thinning_payload)
    easy_model.gen_config = thinning
    easy_model.device = device
    easy_model.event_sampler = EventSampler(
        num_sample=thinning.num_sample,
        num_exp=thinning.num_exp,
        over_sample_rate=thinning.over_sample_rate,
        patience_counter=thinning.patience_counter,
        num_samples_boundary=thinning.num_samples_boundary,
        dtime_max=thinning.dtime_max,
        device=device,
    )


@torch.no_grad()
def _sample_intensity_free_next(
    *,
    easy_model: torch.nn.Module,
    times: Sequence[float],
    type_ids: Sequence[int],
    device: torch.device,
) -> tuple[float, int]:
    _, time_delta_seqs, type_seqs, _, _ = _easy_batch_from_prefix(
        times=times,
        type_ids=type_ids,
        device=device,
    )
    context = easy_model.forward(time_delta_seqs, type_seqs)
    raw_params = easy_model.linear(context[:, -1:, :])
    locs = raw_params[..., : easy_model.num_mix_components]
    log_scales = raw_params[
        ...,
        easy_model.num_mix_components : (2 * easy_model.num_mix_components),
    ]
    log_weights = raw_params[..., (2 * easy_model.num_mix_components) :]
    log_scales = clamp_preserve_gradients(log_scales, -5.0, 3.0)
    log_weights = torch.log_softmax(log_weights, dim=-1)
    inter_time_dist = LogNormalMixtureDistribution(
        locs=locs,
        log_scales=log_scales,
        log_weights=log_weights,
        mean_log_inter_time=easy_model.mean_log_inter_time,
        std_log_inter_time=easy_model.std_log_inter_time,
    )
    dtime = float(inter_time_dist.sample().reshape(-1)[0].item())
    mark_logits = easy_model.mark_linear(context[:, -1, :])[:, : easy_model.num_event_types]
    probs = torch.softmax(mark_logits, dim=-1).squeeze(0)
    type_id = int(torch.multinomial(probs, 1).item())
    return max(0.0, dtime), type_id


def _sample_easy_next_event(
    *,
    model: VitalSignEasyTPPModule,
    times: Sequence[float],
    type_ids: Sequence[int],
    device: torch.device,
) -> tuple[float, int]:
    easy_model = model.easy_tpp_model
    if easy_model.__class__.__name__ == "IntensityFree":
        return _sample_intensity_free_next(
            easy_model=easy_model,
            times=times,
            type_ids=type_ids,
            device=device,
        )

    # FullyNN computes intensity by differentiating its cumulative hazard with
    # respect to sampled time. Most EasyTPP models should run under no_grad for
    # cheap inference, but FullyNN needs a small grad-enabled island.
    needs_grad_for_intensity = easy_model.__class__.__name__ == "FullyNN"
    parameters = list(easy_model.parameters()) if needs_grad_for_intensity else []
    original_requires_grad = [parameter.requires_grad for parameter in parameters]
    if needs_grad_for_intensity:
        for parameter in parameters:
            parameter.requires_grad_(False)

    try:
        grad_context = torch.enable_grad() if needs_grad_for_intensity else torch.no_grad()
        with torch.inference_mode(False):
            with grad_context:
                time_seq, time_delta_seq, event_seq, _, _ = _easy_batch_from_prefix(
                    times=times,
                    type_ids=type_ids,
                    device=device,
                )
                dtime_boundary = time_delta_seq + easy_model.event_sampler.dtime_max
                accepted_dtimes, _ = easy_model.event_sampler.draw_next_time_one_step(
                    time_seq,
                    time_delta_seq,
                    event_seq,
                    dtime_boundary,
                    easy_model.compute_intensities_at_sample_times,
                    compute_last_step_only=True,
                )
                accepted_last = accepted_dtimes[0, -1]
                accepted_index = torch.randint(
                    low=0,
                    high=int(accepted_last.numel()),
                    size=(1,),
                    device=device,
                )
                dtime_tensor = accepted_last[accepted_index].reshape(1, 1, 1).clamp_min(0.0)
                intensities = easy_model.compute_intensities_at_sample_times(
                    time_seq,
                    time_delta_seq,
                    event_seq,
                    dtime_tensor,
                    max_steps=int(event_seq.shape[1]),
                    compute_last_step_only=True,
                )
                mark_scores = intensities[0, -1, 0, :].clamp_min(0.0)
                if not torch.isfinite(mark_scores).all() or float(mark_scores.sum().item()) <= 0.0:
                    mark_scores = torch.ones_like(mark_scores)
                probs = (mark_scores / mark_scores.sum()).detach()
                type_id = int(torch.multinomial(probs, 1).item())
                dtime_value = float(dtime_tensor.detach().reshape(-1)[0].item())
                return dtime_value, type_id
    finally:
        for parameter, requires_grad in zip(parameters, original_requires_grad):
            parameter.requires_grad_(requires_grad)


def _sample_easy_rollout(
    *,
    model: VitalSignEasyTPPModule,
    record: VitalSignEasyTPPSequenceRecord,
    prefix_len: int,
    max_future_events: int,
    easy_bundle: VitalSignEasyTPPDatasetBundle,
    first_event_pool: Sequence[tuple[float, int]],
    rng: random.Random,
    device: torch.device,
) -> list[Event]:
    non_eos = [
        (float(event_time), int(type_id), str(mark_name))
        for event_time, type_id, mark_name in zip(
            record.time_seqs,
            record.type_seqs,
            record.mark_names,
        )
        if mark_name != EOS_EVENT_TYPE_NAME
    ]
    times = [event_time for event_time, _, _ in non_eos[:prefix_len]]
    type_ids = [type_id for _, type_id, _ in non_eos[:prefix_len]]
    generated: list[Event] = []

    for _ in range(max_future_events):
        if not times:
            sampled_time, sampled_type_id = rng.choice(list(first_event_pool))
        else:
            sampled_dtime, sampled_type_id = _sample_easy_next_event(
                model=model,
                times=times,
                type_ids=type_ids,
                device=device,
            )
            if not math.isfinite(sampled_dtime):
                break
            sampled_time = times[-1] + max(0.0, sampled_dtime)

        if sampled_type_id < 0 or sampled_type_id >= len(easy_bundle.mark_names):
            break
        mark_name = easy_bundle.mark_names[sampled_type_id]
        if mark_name == EOS_EVENT_TYPE_NAME:
            break

        generated.append(
            Event(
                time=float(sampled_time),
                event_type=_base_task_from_mark_name(mark_name, easy_bundle.metadata),
                mark=mark_name,
            )
        )
        times.append(float(sampled_time))
        type_ids.append(int(sampled_type_id))

    return generated


def _base_task_from_mark_name(mark_name: str, metadata: Mapping[str, Any]) -> str:
    included_tasks = [str(task_name) for task_name in metadata.get("included_tasks", [])]
    for task_name in sorted(included_tasks, key=len, reverse=True):
        if mark_name == task_name or mark_name.startswith(f"{task_name}__"):
            return task_name
    return str(mark_name).split("__", 1)[0]


def _easy_training_config(spec: ModelSpec, args: argparse.Namespace) -> VitalSignEasyTPPTrainingConfig:
    training_config = VitalSignEasyTPPTrainingConfig.from_dict(spec.training_config)
    training_config.dataset_config = _apply_dataset_load_flags(
        training_config.dataset_config,
        use_saved_datasets=args.use_saved_datasets,
    )
    return training_config


def _load_easy_model_and_bundle(
    *,
    spec: ModelSpec,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[VitalSignEasyTPPModule, VitalSignEasyTPPDatasetBundle]:
    training_config = _easy_training_config(spec=spec, args=args)
    easy_bundle = build_vital_sign_easy_tpp_dataset_bundle(
        dataset_config=training_config.dataset_config,
    )
    model = VitalSignEasyTPPModule.load_from_checkpoint(
        checkpoint_path=str(spec.checkpoint_path),
        map_location=device,
    )
    model.to(device)
    model.eval()
    _ensure_easy_event_sampler(
        model=model,
        thinning_payload={
            "num_sample": args.easy_thinning_num_sample,
            "num_exp": args.easy_thinning_num_exp,
            "over_sample_rate": args.easy_thinning_over_sample_rate,
            "patience_counter": args.easy_thinning_patience_counter,
            "num_samples_boundary": args.easy_thinning_num_samples_boundary,
            "dtime_max": args.easy_thinning_dtime_max,
        },
        device=device,
    )
    return model, easy_bundle


def _flex_training_config(spec: ModelSpec, args: argparse.Namespace) -> VitalSignTPPTrainingConfig:
    training_config = VitalSignTPPTrainingConfig.from_dict(spec.training_config)
    training_config.dataset_config = _apply_dataset_load_flags(
        training_config.dataset_config,
        use_saved_datasets=args.use_saved_datasets,
    )
    return training_config


def _load_flex_model_and_bundle(
    *,
    spec: ModelSpec,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[VitalSignTPPModule, VitalSignTPPDatasetBundle]:
    training_config = _flex_training_config(spec, args)
    flex_bundle = build_vital_sign_tpp_dataset_bundle(
        dataset_config=training_config.dataset_config,
        model_config=training_config.model_config,
    )
    model = VitalSignTPPModule.load_from_checkpoint(
        checkpoint_path=str(spec.checkpoint_path),
        map_location=device,
        model_config=training_config.model_config,
        dims=flex_bundle.dims,
        max_num_classes=flex_bundle.max_num_classes,
        condition_dim=flex_bundle.condition_dim,
    )
    model.to(device)
    model.eval()
    return model, flex_bundle


def _multittpp_training_config(
    spec: ModelSpec,
    args: argparse.Namespace,
) -> VitalSignMultiTTPPTrainingConfig:
    training_config = VitalSignMultiTTPPTrainingConfig.from_dict(spec.training_config)
    training_config.dataset_config = _apply_dataset_load_flags(
        training_config.dataset_config,
        use_saved_datasets=args.use_saved_datasets,
    )
    return training_config


def _load_multittpp_model_and_bundle(
    *,
    spec: ModelSpec,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[VitalSignMultiTTPPModule, VitalSignMultiTTPPDatasetBundle]:
    training_config = _multittpp_training_config(spec=spec, args=args)
    multittpp_bundle = build_vital_sign_multittpp_dataset_bundle(
        dataset_config=training_config.dataset_config,
    )
    model = VitalSignMultiTTPPModule.load_from_checkpoint(
        checkpoint_path=str(spec.checkpoint_path),
        map_location=device,
        model_config=training_config.model_config,
        num_event_types=multittpp_bundle.num_event_types,
        n_events=multittpp_bundle.n_events,
        t_max_normalization=multittpp_bundle.t_max_normalization,
        dt_max_normalization=multittpp_bundle.dt_max_normalization,
    )
    model.to(device)
    model.eval()
    return model, multittpp_bundle


def _discrete_events_from_flex_events(
    *,
    events: Sequence[tuple[float, float, int, Mapping[str, float]]],
    flex_bundle: VitalSignTPPDatasetBundle,
    mark_encoder: DiscreteMarkEncoder,
) -> list[Event]:
    discrete_events: list[Event] = []
    for start_time, _, event_type, event_props in events:
        event_type_index = int(event_type)
        if event_type_index == flex_bundle.eos_event_type:
            break
        if event_type_index < 0 or event_type_index >= len(flex_bundle.event_types):
            continue
        event_type_name = flex_bundle.event_types[event_type_index]
        base_task = mark_encoder.base_task(event_type_name)
        if event_type_name in mark_encoder.mark_name_set:
            mark_name = event_type_name
        elif base_task in mark_encoder.mark_name_set:
            mark_name = base_task
        else:
            mark_name = mark_encoder.label_event(
                task_name=base_task,
                properties=event_props,
            )
        if mark_name is None or mark_name == EOS_EVENT_TYPE_NAME:
            continue
        discrete_events.append(
            Event(
                time=float(start_time),
                event_type=base_task,
                mark=mark_name,
            )
        )
    return discrete_events


def _sample_multittpp_rollout(
    *,
    model: VitalSignMultiTTPPModule,
    record: VitalSignEasyTPPSequenceRecord,
    prefix_len: int,
    max_future_events: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    prefix_times = torch.as_tensor(
        record.time_seqs[:prefix_len],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    prefix_types = torch.as_tensor(
        record.type_seqs[:prefix_len],
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    if prefix_len == 0:
        prefix_times = torch.zeros((1, 0), dtype=torch.float32, device=device)
        prefix_types = torch.zeros((1, 0), dtype=torch.long, device=device)
    with torch.no_grad():
        return model.generate_future(
            prefix_times=prefix_times,
            prefix_types=prefix_types,
            max_future_events=max_future_events,
            n_min=model.model_config.block_size,
        )


def _discrete_events_from_multittpp_samples(
    *,
    sampled_times: torch.Tensor,
    sampled_types: torch.Tensor,
    multittpp_bundle: VitalSignMultiTTPPDatasetBundle,
    mark_encoder: DiscreteMarkEncoder,
) -> list[Event]:
    times = sampled_times.detach().cpu().reshape(-1).tolist()
    types = sampled_types.detach().cpu().reshape(-1).tolist()
    events: list[Event] = []
    for event_time, event_type in zip(times, types):
        event_type_index = int(event_type)
        if event_type_index < 0 or event_type_index >= multittpp_bundle.num_event_types:
            continue
        mark_name = str(multittpp_bundle.mark_names[event_type_index])
        base_task = mark_encoder.base_task(mark_name)
        events.append(
            Event(
                time=float(event_time),
                event_type=base_task,
                mark=mark_name,
            )
        )
    return events


def _evaluate_easy_model(
    *,
    spec: ModelSpec,
    context: EvaluationContext,
    args: argparse.Namespace,
    device: torch.device,
    otd_config: MOTDConfig,
) -> list[dict[str, Any]]:
    model, model_easy_bundle = _load_easy_model_and_bundle(
        spec=spec,
        args=args,
        device=device,
    )
    first_event_pool = _first_event_pool(model_easy_bundle)
    model_records = _easy_records_by_key(model_easy_bundle, args.split)
    rng = random.Random(args.seed)
    rows: list[dict[str, Any]] = []
    total_prefixes, total_samples, total_rollout_event_budget = _prefix_work_for_records(
        args=args,
        records=context.easy_records,
        mark_encoder=context.mark_encoder,
    )
    total_sequences = len(context.easy_records)
    started_at = time.perf_counter()
    completed_prefixes = 0
    completed_samples = 0
    completed_rollout_event_budget = 0
    last_sample_log = 0
    sequence_interval = max(1, int(args.progress_every))
    sample_interval = max(0, int(args.progress_sample_interval))
    _log(
        f"{spec.run_name}: starting EasyTPP OTD evaluation with "
        f"{total_sequences} sequences, {total_prefixes} prefixes, "
        f"{total_samples} Monte Carlo OTD samples."
    )

    for sequence_index, record in enumerate(context.easy_records):
        model_record = model_records.get(_easy_record_key(record))
        if model_record is None:
            raise ValueError(
                f"No matching EasyTPP model-schema record for canonical sequence {sequence_index}."
            )
        true_events = _easy_record_events(record=record, mark_encoder=context.mark_encoder)
        for prefix_len in _prefix_lengths(args, true_events):
            future_count = _future_event_count(args, true_events, prefix_len)
            if future_count <= 0:
                continue
            true_future = true_events[prefix_len : prefix_len + future_count]
            for sample_index in range(int(args.num_samples)):
                inference_started_at = _start_inference_timer(device)
                predicted = _sample_easy_rollout(
                    model=model,
                    record=model_record,
                    prefix_len=prefix_len,
                    max_future_events=future_count,
                    easy_bundle=model_easy_bundle,
                    first_event_pool=first_event_pool,
                    rng=rng,
                    device=device,
                )
                inference_seconds = _stop_inference_timer(device, inference_started_at)
                result = marked_otd(
                    pred_seq=predicted,
                    true_seq=true_future,
                    config=otd_config,
                    time_scales=context.time_scales,
                    return_alignment=False,
                )
                rows.append(
                    _detail_row(
                        spec=spec,
                        record=record,
                        sequence_index=sequence_index,
                        prefix_len=prefix_len,
                        sample_index=sample_index,
                        true_future=true_future,
                        predicted=predicted,
                        inference_seconds=inference_seconds,
                        result=result,
                    )
                )
                completed_samples += 1
                completed_rollout_event_budget += int(future_count)
                if (
                    sample_interval > 0
                    and completed_samples - last_sample_log >= sample_interval
                ):
                    _log_model_progress(
                        run_name=spec.run_name,
                        completed_sequences=sequence_index,
                        total_sequences=total_sequences,
                        completed_prefixes=completed_prefixes,
                        total_prefixes=total_prefixes,
                        completed_samples=completed_samples,
                        total_samples=total_samples,
                        completed_rollout_event_budget=completed_rollout_event_budget,
                        total_rollout_event_budget=total_rollout_event_budget,
                        started_at=started_at,
                    )
                    last_sample_log = completed_samples
            completed_prefixes += 1
        if (
            (sequence_index + 1) % sequence_interval == 0
            or (sequence_index + 1) == total_sequences
        ):
            _log_model_progress(
                run_name=spec.run_name,
                completed_sequences=sequence_index + 1,
                total_sequences=total_sequences,
                completed_prefixes=completed_prefixes,
                total_prefixes=total_prefixes,
                completed_samples=completed_samples,
                total_samples=total_samples,
                completed_rollout_event_budget=completed_rollout_event_budget,
                total_rollout_event_budget=total_rollout_event_budget,
                started_at=started_at,
            )
    return rows


def _evaluate_multittpp_model(
    *,
    spec: ModelSpec,
    context: EvaluationContext,
    args: argparse.Namespace,
    device: torch.device,
    otd_config: MOTDConfig,
) -> list[dict[str, Any]]:
    model, model_multittpp_bundle = _load_multittpp_model_and_bundle(
        spec=spec,
        args=args,
        device=device,
    )
    model_records = _easy_records_by_key(model_multittpp_bundle, args.split)
    rows: list[dict[str, Any]] = []
    total_prefixes, total_samples, total_rollout_event_budget = _prefix_work_for_records(
        args=args,
        records=context.easy_records,
        mark_encoder=context.mark_encoder,
    )
    total_sequences = len(context.easy_records)
    sequence_interval, sample_interval = _progress_intervals(args)
    completed_prefixes = 0
    completed_samples = 0
    completed_rollout_event_budget = 0
    last_sample_log = 0
    started_at = time.perf_counter()
    _log(
        f"{spec.run_name}: starting MultiTTPP OTD evaluation with "
        f"{total_sequences} sequences, {total_prefixes} prefixes, "
        f"{total_samples} sampled rollouts."
    )

    for sequence_index, record in enumerate(context.easy_records):
        model_record = model_records.get(_easy_record_key(record))
        if model_record is None:
            raise ValueError(
                f"No matching MultiTTPP model-schema record for canonical sequence {sequence_index}."
            )
        true_events = _easy_record_events(record=record, mark_encoder=context.mark_encoder)
        for prefix_len in _prefix_lengths(args, true_events):
            future_count = _future_event_count(args, true_events, prefix_len)
            if future_count <= 0:
                continue
            true_future = true_events[prefix_len : prefix_len + future_count]
            for sample_index in range(int(args.num_samples)):
                inference_started_at = _start_inference_timer(device)
                sampled_times, sampled_types = _sample_multittpp_rollout(
                    model=model,
                    record=model_record,
                    prefix_len=prefix_len,
                    max_future_events=future_count,
                    device=device,
                )
                inference_seconds = _stop_inference_timer(device, inference_started_at)
                predicted = _discrete_events_from_multittpp_samples(
                    sampled_times=sampled_times,
                    sampled_types=sampled_types,
                    multittpp_bundle=model_multittpp_bundle,
                    mark_encoder=context.mark_encoder,
                )
                result = marked_otd(
                    pred_seq=predicted,
                    true_seq=true_future,
                    config=otd_config,
                    time_scales=context.time_scales,
                    return_alignment=False,
                )
                rows.append(
                    _detail_row(
                        spec=spec,
                        record=record,
                        sequence_index=sequence_index,
                        prefix_len=prefix_len,
                        sample_index=sample_index,
                        true_future=true_future,
                        predicted=predicted,
                        inference_seconds=inference_seconds,
                        result=result,
                    )
                )
                completed_samples += 1
                completed_rollout_event_budget += int(future_count)
                if (
                    sample_interval > 0
                    and completed_samples - last_sample_log >= sample_interval
                ):
                    _log_model_progress(
                        run_name=spec.run_name,
                        completed_sequences=sequence_index,
                        total_sequences=total_sequences,
                        completed_prefixes=completed_prefixes,
                        total_prefixes=total_prefixes,
                        completed_samples=completed_samples,
                        total_samples=total_samples,
                        completed_rollout_event_budget=completed_rollout_event_budget,
                        total_rollout_event_budget=total_rollout_event_budget,
                        started_at=started_at,
                    )
                    last_sample_log = completed_samples
            completed_prefixes += 1
        if (
            (sequence_index + 1) % sequence_interval == 0
            or (sequence_index + 1) == total_sequences
        ):
            _log_model_progress(
                run_name=spec.run_name,
                completed_sequences=sequence_index + 1,
                total_sequences=total_sequences,
                completed_prefixes=completed_prefixes,
                total_prefixes=total_prefixes,
                completed_samples=completed_samples,
                total_samples=total_samples,
                completed_rollout_event_budget=completed_rollout_event_budget,
                total_rollout_event_budget=total_rollout_event_budget,
                started_at=started_at,
            )
    return rows


def _evaluate_flex_model(
    *,
    spec: ModelSpec,
    context: EvaluationContext,
    args: argparse.Namespace,
    device: torch.device,
    otd_config: MOTDConfig,
) -> list[dict[str, Any]]:
    model, flex_bundle = _load_flex_model_and_bundle(spec=spec, args=args, device=device)
    flex_records = _flex_records_by_key(flex_bundle, args.split)
    flex_dataset = flex_bundle.get_dataset(args.split)
    rows: list[dict[str, Any]] = []
    total_prefixes, total_samples, total_rollout_event_budget = _prefix_work_for_records(
        args=args,
        records=context.easy_records,
        mark_encoder=context.mark_encoder,
    )
    total_sequences = len(context.easy_records)
    started_at = time.perf_counter()
    completed_prefixes = 0
    completed_samples = 0
    completed_rollout_event_budget = 0
    last_sample_log = 0
    sequence_interval = max(1, int(args.progress_every))
    sample_interval = max(0, int(args.progress_sample_interval))
    _log(
        f"{spec.run_name}: starting FlexTPP OTD evaluation with "
        f"{total_sequences} sequences, {total_prefixes} prefixes, "
        f"{total_samples} Monte Carlo OTD samples."
    )

    for sequence_index, easy_record in enumerate(context.easy_records):
        flex_record = flex_records.get(_easy_record_key(easy_record))
        if flex_record is None:
            raise ValueError(
                f"No matching FlexTPP record for EasyTPP test sequence {sequence_index}."
            )
        matched_flex_events = _matched_flex_events_for_easy_record(
            easy_record=easy_record,
            flex_record=flex_record,
        )
        true_events = _easy_record_events(record=easy_record, mark_encoder=context.mark_encoder)
        for prefix_len in _prefix_lengths(args, true_events):
            future_count = _future_event_count(args, true_events, prefix_len)
            if future_count <= 0:
                continue
            true_future = true_events[prefix_len : prefix_len + future_count]
            prefix_events = matched_flex_events[:prefix_len]
            for sample_index in range(int(args.num_samples)):
                inference_started_at = _start_inference_timer(device)
                sampled_flex_events = _sample_future_events_from_prefix(
                    model=model,
                    dataset_bundle=flex_bundle,
                    dataset=flex_dataset,
                    prefix_events=prefix_events,
                    condition=flex_record.condition,
                    max_future_events=future_count,
                    device=device,
                    argmax=False,
                    mean_of=int(args.flex_mean_of),
                    median=False,
                )
                inference_seconds = _stop_inference_timer(device, inference_started_at)
                predicted = _discrete_events_from_flex_events(
                    events=sampled_flex_events,
                    flex_bundle=flex_bundle,
                    mark_encoder=context.mark_encoder,
                )
                result = marked_otd(
                    pred_seq=predicted,
                    true_seq=true_future,
                    config=otd_config,
                    time_scales=context.time_scales,
                    return_alignment=False,
                )
                rows.append(
                    _detail_row(
                        spec=spec,
                        record=easy_record,
                        sequence_index=sequence_index,
                        prefix_len=prefix_len,
                        sample_index=sample_index,
                        true_future=true_future,
                        predicted=predicted,
                        inference_seconds=inference_seconds,
                        result=result,
                    )
                )
                completed_samples += 1
                completed_rollout_event_budget += int(future_count)
                if (
                    sample_interval > 0
                    and completed_samples - last_sample_log >= sample_interval
                ):
                    _log_model_progress(
                        run_name=spec.run_name,
                        completed_sequences=sequence_index,
                        total_sequences=total_sequences,
                        completed_prefixes=completed_prefixes,
                        total_prefixes=total_prefixes,
                        completed_samples=completed_samples,
                        total_samples=total_samples,
                        completed_rollout_event_budget=completed_rollout_event_budget,
                        total_rollout_event_budget=total_rollout_event_budget,
                        started_at=started_at,
                    )
                    last_sample_log = completed_samples
            completed_prefixes += 1
        if (
            (sequence_index + 1) % sequence_interval == 0
            or (sequence_index + 1) == total_sequences
        ):
            _log_model_progress(
                run_name=spec.run_name,
                completed_sequences=sequence_index + 1,
                total_sequences=total_sequences,
                completed_prefixes=completed_prefixes,
                total_prefixes=total_prefixes,
                completed_samples=completed_samples,
                total_samples=total_samples,
                completed_rollout_event_budget=completed_rollout_event_budget,
                total_rollout_event_budget=total_rollout_event_budget,
                started_at=started_at,
            )
    return rows


def _detail_row(
    *,
    spec: ModelSpec,
    record: VitalSignEasyTPPSequenceRecord,
    sequence_index: int,
    prefix_len: int,
    sample_index: int,
    true_future: Sequence[Event],
    predicted: Sequence[Event],
    inference_seconds: float,
    result: Any,
) -> dict[str, Any]:
    true_event_count = int(len(true_future))
    predicted_event_count = int(len(predicted))
    inference_seconds = float(inference_seconds)
    inference_seconds_per_true_event = (
        inference_seconds / float(true_event_count)
        if true_event_count > 0
        else float("nan")
    )
    inference_seconds_per_predicted_event = (
        inference_seconds / float(predicted_event_count)
        if predicted_event_count > 0
        else float("nan")
    )
    return {
        "family": spec.family,
        "model_name": spec.model_name,
        "variant": spec.variant,
        "run_name": spec.run_name,
        "split": record.split,
        "sequence_index": int(sequence_index),
        "patient_id": record.patient_id,
        "encounter_id": record.encounter_id,
        "segment_id": int(record.segment_id),
        "demand_level": _record_primary_demand_level(record) or "unmatched",
        "prefix_event_count": int(prefix_len),
        "sample_index": int(sample_index),
        "true_event_count": true_event_count,
        "predicted_event_count": predicted_event_count,
        "inference_seconds": inference_seconds,
        "inference_milliseconds": inference_seconds * 1000.0,
        "inference_seconds_per_true_event": inference_seconds_per_true_event,
        "inference_milliseconds_per_true_event": inference_seconds_per_true_event * 1000.0,
        "inference_seconds_per_predicted_event": inference_seconds_per_predicted_event,
        "inference_milliseconds_per_predicted_event": (
            inference_seconds_per_predicted_event * 1000.0
        ),
        "otd_total": float(result.cost),
        "otd_time": float(result.time_cost),
        "otd_type": float(result.type_cost),
        "otd_edit": float(result.edit_cost),
        "otd_delete": float(getattr(result, "delete_cost", 0.0)),
        "otd_insert": float(getattr(result, "insert_cost", 0.0)),
        "otd_other": float(result.other_cost),
    }


def _evaluate_model(
    *,
    spec: ModelSpec,
    model_index: int,
    total_models: int,
    context: EvaluationContext,
    args: argparse.Namespace,
    device: torch.device,
    otd_config: MOTDConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started_at = datetime.datetime.now()
    try:
        _log(
            f"Evaluating model {model_index}/{total_models}: "
            f"{spec.run_name} [{spec.family}/{spec.variant}]"
        )
        if spec.family.lower() == "easytpp":
            detail_rows = _evaluate_easy_model(
                spec=spec,
                context=context,
                args=args,
                device=device,
                otd_config=otd_config,
            )
        elif spec.family.lower() == "multittpp":
            detail_rows = _evaluate_multittpp_model(
                spec=spec,
                context=context,
                args=args,
                device=device,
                otd_config=otd_config,
            )
        elif spec.family.lower() == "flextpp":
            detail_rows = _evaluate_flex_model(
                spec=spec,
                context=context,
                args=args,
                device=device,
                otd_config=otd_config,
            )
        else:
            raise ValueError(f"Unsupported model family '{spec.family}'.")
        summary_row = _summary_row_from_details(
            spec=spec,
            detail_rows=detail_rows,
            started_at=started_at,
            evaluation_demand_level=args.demand_level,
            status="success",
            error_message="",
        )
        _log(f"Finished {spec.run_name}: mean OTD={summary_row['otd_total_mean']:.4f}.")
        return detail_rows, summary_row
    except Exception as exc:
        traceback.print_exc()
        return [], _summary_row_from_details(
            spec=spec,
            detail_rows=[],
            started_at=started_at,
            evaluation_demand_level=args.demand_level,
            status="failed",
            error_message=str(exc),
        )


def _summary_row_from_details(
    *,
    spec: ModelSpec,
    detail_rows: Sequence[Mapping[str, Any]],
    started_at: datetime.datetime,
    evaluation_demand_level: str,
    status: str,
    error_message: str,
) -> dict[str, Any]:
    ended_at = datetime.datetime.now()
    row: dict[str, Any] = {
        "family": spec.family,
        "model_name": spec.model_name,
        "variant": spec.variant,
        "run_name": spec.run_name,
        "demand_level": str(evaluation_demand_level),
        "status": status,
        "error_message": error_message,
        "checkpoint_path": str(spec.checkpoint_path),
        "metrics_summary_path": "" if spec.metrics_summary_path is None else str(spec.metrics_summary_path),
        "started_at": started_at.isoformat(timespec="seconds"),
        "ended_at": ended_at.isoformat(timespec="seconds"),
        "duration_seconds": round((ended_at - started_at).total_seconds(), 3),
        "num_otd_samples": len(detail_rows),
    }
    row["inference_seconds_total"] = float(
        sum(
            float(detail_row.get("inference_seconds", 0.0))
            for detail_row in detail_rows
            if math.isfinite(float(detail_row.get("inference_seconds", 0.0)))
        )
    )
    for column in (
        "otd_total",
        "otd_time",
        "otd_other",
        "otd_type",
        "otd_edit",
        "otd_delete",
        "otd_insert",
        "predicted_event_count",
        "inference_seconds",
        "inference_milliseconds",
        "inference_seconds_per_true_event",
        "inference_milliseconds_per_true_event",
        "inference_seconds_per_predicted_event",
        "inference_milliseconds_per_predicted_event",
    ):
        values = [float(detail_row[column]) for detail_row in detail_rows if column in detail_row]
        value_mean, value_std = _summary_stats(values)
        row[f"{column}_mean"] = value_mean
        row[f"{column}_std"] = value_std
    return row


def _write_outputs(
    *,
    output_dir: Path,
    detail_rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    config_payload: Mapping[str, Any],
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "otd_prefix_samples.csv"
    summary_path = output_dir / "otd_summary.csv"
    metadata_path = output_dir / "otd_evaluation_metadata.json"

    _write_csv_rows(detail_path, detail_rows)
    _write_csv_rows(summary_path, summary_rows)
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(dict(config_payload), metadata_file, indent=2, default=str)
    return detail_path, summary_path, metadata_path


def _write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        for row in rows:
            for field_name in row.keys():
                if field_name not in fieldnames:
                    fieldnames.append(field_name)
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_results(*, output_dir: Path, detail_path: Path, summary_path: Path) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception as exc:
        _log(f"Skipping plots because plotting dependencies could not be imported: {exc}")
        return []

    detail_df = pd.read_csv(detail_path)
    summary_df = pd.read_csv(summary_path)
    summary_df = summary_df[summary_df["status"] == "success"].copy()
    if detail_df.empty or summary_df.empty:
        return []

    plot_paths: list[Path] = []
    label_column = "run_name"
    summary_df = summary_df.sort_values("otd_total_mean", ascending=True)

    component_path = output_dir / "otd_component_bar.png"
    labels = summary_df[label_column].tolist()
    x_positions = np.arange(len(labels))
    time_values = summary_df["otd_time_mean"].to_numpy(dtype=float)
    type_values = summary_df["otd_type_mean"].to_numpy(dtype=float)
    delete_values = summary_df.get(
        "otd_delete_mean",
        summary_df["otd_edit_mean"] * 0.0,
    ).to_numpy(dtype=float)
    insert_values = summary_df.get(
        "otd_insert_mean",
        summary_df["otd_edit_mean"] * 0.0,
    ).to_numpy(dtype=float)
    residual_values = np.maximum(
        0.0,
        summary_df["otd_total_mean"].to_numpy(dtype=float)
        - time_values
        - type_values
        - delete_values
        - insert_values,
    )
    total_std = summary_df["otd_total_std"].to_numpy(dtype=float)
    fig_width = max(10.0, 0.55 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, 6.0))
    bottoms = np.zeros_like(time_values)
    ax.bar(x_positions, time_values, bottom=bottoms, label="Time")
    bottoms = bottoms + time_values
    ax.bar(x_positions, type_values, bottom=bottoms, label="Type")
    bottoms = bottoms + type_values
    ax.bar(x_positions, delete_values, bottom=bottoms, label="Deletion")
    bottoms = bottoms + delete_values
    ax.bar(x_positions, insert_values, bottom=bottoms, label="Addition")
    bottoms = bottoms + insert_values
    if np.any(residual_values > 1e-12):
        ax.bar(x_positions, residual_values, bottom=bottoms, label="Other")
        bottoms = bottoms + residual_values
    ax.errorbar(
        x_positions,
        bottoms,
        yerr=total_std,
        fmt="none",
        ecolor="black",
        elinewidth=1,
        capsize=2,
    )
    ax.set_ylabel("Mean type/time OTD")
    ax.set_title("Vital-sign TPP type/time OTD by model")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(component_path, dpi=180)
    plt.close(fig)
    plot_paths.append(component_path)

    boxplot_path = output_dir / "otd_distribution_boxplot.png"
    detail_df = detail_df[detail_df["run_name"].isin(labels)].copy()
    grouped_values = [
        detail_df.loc[detail_df["run_name"] == run_name, "otd_total"].to_numpy(dtype=float)
        for run_name in labels
    ]
    fig, ax = plt.subplots(figsize=(fig_width, 6.0))
    ax.boxplot(grouped_values, labels=labels, showfliers=False)
    ax.set_ylabel("Sample type/time OTD")
    ax.set_title("Type/time OTD distribution across prefixes and samples")
    ax.tick_params(axis="x", rotation=45)
    for tick_label in ax.get_xticklabels():
        tick_label.set_ha("right")
    fig.tight_layout()
    fig.savefig(boxplot_path, dpi=180)
    plt.close(fig)
    plot_paths.append(boxplot_path)

    if "inference_milliseconds_mean" in summary_df.columns:
        inference_path = output_dir / "inference_latency_bar.png"
        inference_values = summary_df["inference_milliseconds_mean"].to_numpy(dtype=float)
        inference_std = summary_df["inference_milliseconds_std"].to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(fig_width, 6.0))
        ax.bar(x_positions, inference_values, yerr=inference_std, capsize=2)
        ax.set_ylabel("Mean rollout inference latency (ms)")
        ax.set_title("Vital-sign TPP rollout latency by model")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(inference_path, dpi=180)
        plt.close(fig)
        plot_paths.append(inference_path)

    return plot_paths


def _log_wandb(
    *,
    args: argparse.Namespace,
    output_paths: Sequence[Path],
    plot_paths: Sequence[Path],
    summary_rows: Sequence[Mapping[str, Any]],
) -> None:
    if not args.wandb:
        return
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        group=args.wandb_group,
        job_type="otd_eval",
        config={key: value for key, value in vars(args).items()},
        dir=str(_resolve_repo_relative_path(args.output_dir) or Path(args.output_dir)),
    )
    for summary_row in summary_rows:
        if summary_row.get("status") != "success":
            continue
        metric_prefix = _safe_name(str(summary_row["run_name"]))
        for metric_name in (
            "otd_total_mean",
            "otd_total_std",
            "otd_time_mean",
            "otd_time_std",
            "otd_other_mean",
            "otd_other_std",
            "inference_milliseconds_mean",
            "inference_milliseconds_std",
            "inference_milliseconds_per_true_event_mean",
            "inference_milliseconds_per_true_event_std",
            "inference_seconds_total",
        ):
            if metric_name in summary_row:
                wandb.summary[f"{metric_prefix}/{metric_name}"] = summary_row[metric_name]
    for plot_path in plot_paths:
        wandb.log({plot_path.stem: wandb.Image(str(plot_path))})
    artifact = wandb.Artifact("vital_sign_tpp_otd_evaluation", type="evaluation")
    for output_path in output_paths:
        artifact.add_file(str(output_path))
    for plot_path in plot_paths:
        artifact.add_file(str(plot_path))
    run.log_artifact(artifact)
    run.finish()


def _parse_selected_runs(raw_value: str | None) -> set[str] | None:
    if raw_value is None or not raw_value.strip():
        return None
    return {
        run_name.strip()
        for run_name in raw_value.split(",")
        if run_name.strip()
    }


def _device_from_args(args: argparse.Namespace) -> torch.device:
    if args.device:
        return torch.device(args.device)
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def _parse_csv_list(
    raw_value: str,
    *,
    allowed_values: Sequence[str] | None = None,
) -> list[str]:
    values = [
        value.strip()
        for value in str(raw_value or "").split(",")
        if value.strip()
    ]
    if not values:
        raise ValueError("Expected at least one comma-separated value.")
    if allowed_values is not None:
        allowed = set(allowed_values)
        unsupported = [value for value in values if value not in allowed]
        if unsupported:
            raise ValueError(
                f"Unsupported values {unsupported}. Expected values from {tuple(allowed_values)}."
            )
    return values


def _normalize_gpu_device_arg(raw_gpu: str) -> str:
    gpu = str(raw_gpu).strip()
    if not gpu:
        raise ValueError("GPU ids must not be empty.")
    if gpu.lower() == "cpu" or ":" in gpu:
        return gpu
    return f"cuda:{gpu}"


def _strip_option(args: list[str], option_names: set[str]) -> list[str]:
    stripped: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        matched_option = None
        for option_name in option_names:
            if token == option_name or token.startswith(f"{option_name}="):
                matched_option = option_name
                break
        if matched_option is None:
            stripped.append(token)
            index += 1
            continue
        if token == matched_option and index + 1 < len(args) and not args[index + 1].startswith("--"):
            index += 2
        else:
            index += 1
    return stripped


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
    child_args = _strip_option(list(parent_argv), parent_only_options)
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
    demand_levels = _parse_csv_list(args.demand_levels, allowed_values=DEMAND_LEVELS)
    gpu_args = [
        _normalize_gpu_device_arg(gpu_id)
        for gpu_id in _parse_csv_list(args.demand_gpu_ids)
    ]
    if len(gpu_args) < len(demand_levels):
        raise ValueError(
            f"--run_demand_strata needs at least {len(demand_levels)} GPU ids, "
            f"but --demand_gpu_ids provided {len(gpu_args)}."
        )
    if len(set(gpu_args[: len(demand_levels)])) != len(demand_levels):
        raise ValueError("--demand_gpu_ids must assign a distinct GPU to each demand level.")

    base_output_dir = _resolve_repo_relative_path(args.output_dir)
    if base_output_dir is None:
        base_output_dir = _repo_root() / DEFAULT_OUTPUT_DIR
    base_output_dir.mkdir(parents=True, exist_ok=True)

    processes: list[tuple[str, str, Path, subprocess.Popen[Any], float]] = []
    for demand_level, device_arg in zip(demand_levels, gpu_args):
        child_output_dir = base_output_dir / demand_level
        child_output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "HTAMP.prediction.run_vital_sign_tpp_otd_evaluation",
            *_demand_child_args(
                parent_argv=parent_argv,
                demand_level=demand_level,
                device_arg=device_arg,
                output_dir=child_output_dir,
            ),
        ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        _log(
            f"Launching demand_level='{demand_level}' on {device_arg}; "
            f"outputs will be written to {child_output_dir}."
        )
        process = subprocess.Popen(command, cwd=_repo_root(), env=env)
        processes.append((demand_level, device_arg, child_output_dir, process, time.perf_counter()))

    summary_rows: list[dict[str, Any]] = []
    return_code = 0
    for demand_level, device_arg, child_output_dir, process, started_at in processes:
        child_return_code = process.wait()
        duration_seconds = round(time.perf_counter() - started_at, 3)
        status = "success" if child_return_code == 0 else "failed"
        if child_return_code != 0:
            return_code = 1
        _log(
            f"Finished demand_level='{demand_level}' on {device_arg} with "
            f"status={status} in {_format_duration(duration_seconds)}."
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
    _write_csv_rows(strata_summary_path, summary_rows)
    _log(f"Demand-stratified OTD launch summary saved to {strata_summary_path}")
    return return_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="VitalSignTPPOTDEvaluation",
        description=(
            "Evaluate trained EasyTPP, FlexTPP, and MultiTTPP vital-sign request models with "
            "Monte Carlo marked Optimal Transport Distance."
        ),
    )
    parser.add_argument(
        "--comparison_summary_path",
        default=None,
        help=(
            "CSV summary produced by run_vital_sign_tpp_model_comparison. "
            "Defaults to the most recent summary under data/prediction."
        ),
    )
    parser.add_argument(
        "--easy_config_path",
        default=None,
        help=(
            "Optional EasyTPP training JSON to define the canonical discrete "
            "low/medium/high mark schema."
        ),
    )
    parser.add_argument("--selected_runs", default=None, help="Comma-separated run names to evaluate.")
    parser.add_argument("--stf_log_dir", default=None)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument(
        "--demand_level",
        choices=DEMAND_LEVEL_CHOICES,
        default="all",
        help=(
            "Filter canonical sequences by the demand tier implied by the "
            "test floor/week mapping before applying --max_sequences."
        ),
    )
    parser.add_argument(
        "--demand_sequence_assignment",
        choices=("majority", "any", "strict"),
        default="majority",
        help=(
            "How to assign a sequence with events from multiple demand tiers: "
            "majority uses the tier with the most events, any allows overlap, "
            "and strict keeps only sequences with no events from other tiers."
        ),
    )
    parser.add_argument(
        "--run_demand_strata",
        action="store_true",
        help=(
            "Launch one child OTD evaluation per demand tier, assigning each "
            "tier to a GPU from --demand_gpu_ids and writing separate outputs."
        ),
    )
    parser.add_argument(
        "--demand_levels",
        default=",".join(DEMAND_LEVELS),
        help="Comma-separated demand tiers to launch with --run_demand_strata.",
    )
    parser.add_argument(
        "--demand_gpu_ids",
        default="0,1,2",
        help=(
            "Comma-separated GPU ids or device strings used by --run_demand_strata. "
            "Examples: 0,1,2 or cuda:0,cuda:1,cuda:2."
        ),
    )
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument(
        "--max_future_events",
        type=int,
        default=5,
        help="Maximum rollout length per prefix; use 0 to compare against all remaining events.",
    )
    parser.add_argument("--min_prefix_events", type=int, default=0)
    parser.add_argument(
        "--prefix_stride",
        type=int,
        default=1,
        help=(
            "Evaluate every Nth prefix within each sequence, while still including "
            "the final prefix. Increase this for a direct runtime reduction."
        ),
    )
    parser.add_argument("--max_prefixes_per_sequence", type=int, default=None)
    parser.add_argument(
        "--prefix_subset_strategy",
        choices=("evenly_spaced", "first"),
        default="evenly_spaced",
        help=(
            "When --max_prefixes_per_sequence is set, choose prefixes across the "
            "whole day or take only the earliest prefixes."
        ),
    )
    parser.add_argument("--max_sequences", type=int, default=None)
    parser.add_argument(
        "--sequence_subset_strategy",
        choices=("first", "random"),
        default="first",
        help=(
            "When --max_sequences is set, choose the first N canonical sequences "
            "or a reproducible random subset controlled by --seed."
        ),
    )
    parser.add_argument(
        "--use_saved_datasets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load the cached FlexTPP/EasyTPP/MultiTTPP datasets by default.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time_weight", type=float, default=1.0)
    parser.add_argument("--type_weight", type=float, default=2.0)
    parser.add_argument(
        "--mark_weight",
        type=float,
        default=0.0,
        help="Deprecated and ignored; OTD evaluation is type/time-only.",
    )
    parser.add_argument("--delete_cost", type=float, default=1.0)
    parser.add_argument("--insert_cost", type=float, default=1.0)
    parser.add_argument(
        "--soft_type_matching",
        action="store_true",
        help="Allow cross-task substitutions with --type_weight instead of hard task matching.",
    )
    parser.add_argument("--flex_mean_of", type=int, default=1)
    parser.add_argument("--easy_thinning_num_sample", type=int, default=16)
    parser.add_argument("--easy_thinning_num_exp", type=int, default=200)
    parser.add_argument("--easy_thinning_over_sample_rate", type=float, default=5.0)
    parser.add_argument("--easy_thinning_patience_counter", type=int, default=5)
    parser.add_argument("--easy_thinning_num_samples_boundary", type=int, default=5)
    parser.add_argument("--easy_thinning_dtime_max", type=float, default=24.0)
    parser.add_argument("--progress_every", type=int, default=50)
    parser.add_argument(
        "--progress_sample_interval",
        type=int,
        default=250,
        help=(
            "Log model progress every N completed Monte Carlo OTD samples. "
            "Set to 0 to disable sample-interval progress logs."
        ),
    )
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="vital_sign_tpp_comparison")
    parser.add_argument("--wandb_group", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.run_demand_strata:
        return _run_demand_strata(args=args, parent_argv=sys.argv[1:])

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    random.seed(int(args.seed))

    comparison_summary_path = _coerce_comparison_summary_path(args.comparison_summary_path)
    output_dir = _resolve_repo_relative_path(args.output_dir)
    if output_dir is None:
        output_dir = _repo_root() / DEFAULT_OUTPUT_DIR
    device = _device_from_args(args)
    _log(f"Using comparison summary: {comparison_summary_path}")
    _log(f"Using device: {device}")

    model_specs = _load_model_specs(
        comparison_summary_path=comparison_summary_path,
        stf_log_dir=args.stf_log_dir,
        selected_runs=_parse_selected_runs(args.selected_runs),
    )
    context = _build_evaluation_context(args=args, model_specs=model_specs)
    otd_config = _make_otd_config(args, default_tau=context.default_tau)
    total_prefixes, samples_per_model, rollout_event_budget_per_model = _prefix_work_for_records(
        args=args,
        records=context.easy_records,
        mark_encoder=context.mark_encoder,
    )
    _log(
        f"Prepared OTD evaluation for {len(model_specs)} models on "
        f"{len(context.easy_records)} canonical {args.split} sequences: "
        f"{total_prefixes} prefixes/model, {samples_per_model} OTD samples/model, "
        f"{rollout_event_budget_per_model} rollout-event budget/model."
    )

    all_detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for model_index, spec in enumerate(model_specs, start=1):
        detail_rows, summary_row = _evaluate_model(
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

    config_payload = {
        "comparison_summary_path": str(comparison_summary_path),
        "output_dir": str(output_dir),
        "args": vars(args),
        "otd_config": otd_config.__dict__,
        "num_models": len(model_specs),
        "num_sequences": len(context.easy_records),
        "demand_level": args.demand_level,
        "demand_sequence_assignment": args.demand_sequence_assignment,
        "demand_sequence_counts": _demand_sequence_counts(context.easy_records),
        "demand_week_sets_by_floor": _serializable_demand_week_sets_by_floor(),
        "mark_names": context.easy_bundle.mark_names,
    }
    detail_path, summary_path, metadata_path = _write_outputs(
        output_dir=output_dir,
        detail_rows=all_detail_rows,
        summary_rows=summary_rows,
        config_payload=config_payload,
    )
    plot_paths: list[Path] = []
    if not args.skip_plots and all_detail_rows:
        plot_paths = _plot_results(
            output_dir=output_dir,
            detail_path=detail_path,
            summary_path=summary_path,
        )
    _log_wandb(
        args=args,
        output_paths=[detail_path, summary_path, metadata_path],
        plot_paths=plot_paths,
        summary_rows=summary_rows,
    )
    _log(f"OTD detail rows saved to {detail_path}")
    _log(f"OTD summary saved to {summary_path}")
    for plot_path in plot_paths:
        _log(f"Plot saved to {plot_path}")
    return 1 if any(row["status"] == "failed" for row in summary_rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())