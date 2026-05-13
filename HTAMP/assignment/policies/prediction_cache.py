from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import pandas as pd

from HTAMP.environment.traversal_graph_gen import TraversalGraphGenerator
from HTAMP.planning.planning_dataclasses import AllTaskProperties, RequestsLists, TaskProperties, TaskRequest
from HTAMP.planning.state import PlanningState


MONITORING_REQUEST_FIELDS = (
    "blood_pressure_requests",
    "heart_rate_requests",
    "respiratory_rate_requests",
    "temperature_requests",
    "oxygen_saturation_requests",
)
REQUEST_FIELD_BY_TYPE = {
    "blood_pressure": "blood_pressure_requests",
    "heart_rate": "heart_rate_requests",
    "respiratory_rate": "respiratory_rate_requests",
    "temperature": "temperature_requests",
    "oxygen_saturation": "oxygen_saturation_requests",
    "medication": "medications_requests",
}


def empty_requests_lists() -> RequestsLists:
    return RequestsLists(
        blood_pressure_requests=[],
        heart_rate_requests=[],
        respiratory_rate_requests=[],
        temperature_requests=[],
        oxygen_saturation_requests=[],
        medications_requests=[],
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_path(path_value: str | Path | None) -> Path | None:
    if path_value is None or not str(path_value).strip():
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return _repo_root() / path


def _parse_csv_set(raw_value: str | Sequence[str] | None) -> set[str] | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        values = raw_value.split(",")
    else:
        values = [str(value) for value in raw_value]
    parsed = {value.strip() for value in values if value and value.strip()}
    return parsed or None


def _column_or_default(frame: pd.DataFrame, column: str, default: Any = "") -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series(default, index=frame.index)


def _normalize_identifier(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text


def _int_or_none(value: Any) -> int | None:
    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric_value):
        return None
    return int(numeric_value)


def _safe_cache_component(value: Any, *, unknown: str = "unknown") -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "<na>", "nat"}:
        text = unknown
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in text)
    return cleaned.strip("_") or unknown


def _split_cache_shard_path(
    *,
    cache_dir: Path,
    floor_number: int,
    date_stamp: pd.Timestamp,
) -> Path | None:
    floor = _safe_cache_component(str(int(floor_number)))
    day = pd.Timestamp(date_stamp).date().isoformat()
    for suffix in (".csv", ".csv.gz"):
        shard_path = cache_dir / f"floor_{floor}" / f"day_{day}{suffix}"
        if shard_path.exists():
            return shard_path

    manifest_path = cache_dir / "manifest.csv"
    if not manifest_path.exists():
        return None
    try:
        manifest = pd.read_csv(manifest_path, dtype=str)
    except Exception:
        return None
    if manifest.empty or not {"floor", "day", "relative_path"}.issubset(manifest.columns):
        return None
    matches = manifest[
        manifest["floor"].astype(str).eq(str(int(floor_number)))
        & manifest["day"].astype(str).eq(day)
    ]
    if matches.empty:
        return None
    shard_path = cache_dir / str(matches.iloc[0]["relative_path"])
    return shard_path if shard_path.exists() else None


def _task_properties_for_request_type(
    all_task_properties: AllTaskProperties,
    request_type: str,
) -> TaskProperties | None:
    if request_type == "medication":
        return all_task_properties.medications
    return getattr(all_task_properties, request_type, None)


def _iter_requests(requests_lists: Optional[RequestsLists]) -> Iterable[TaskRequest]:
    if requests_lists is None:
        return
    for data_field in requests_lists.__dataclass_fields__:
        yield from getattr(requests_lists, data_field)


def _append_request(requests_lists: RequestsLists, request: TaskRequest) -> None:
    data_field = REQUEST_FIELD_BY_TYPE.get(request.request_type)
    if data_field is None:
        return
    getattr(requests_lists, data_field).append(request)


def _monitoring_requests(requests_lists: RequestsLists) -> list[TaskRequest]:
    requests: list[TaskRequest] = []
    for data_field in MONITORING_REQUEST_FIELDS:
        requests.extend(getattr(requests_lists, data_field))
    return requests


def _remove_one_close_prediction(
    predicted_requests: list[TaskRequest],
    real_request: TaskRequest,
    tolerance_seconds: float,
) -> bool:
    best_index = None
    best_delta = None
    for index, predicted_request in enumerate(predicted_requests):
        delta = abs(float(predicted_request.scheduled_time) - float(real_request.scheduled_time))
        if delta <= tolerance_seconds and (best_delta is None or delta < best_delta):
            best_index = index
            best_delta = delta
    if best_index is None:
        return False
    predicted_requests.pop(best_index)
    return True


def remove_real_request_matches(
    *,
    predicted_requests_lists: RequestsLists,
    real_requests_lists: Optional[RequestsLists],
    tolerance_minutes: float,
) -> RequestsLists:
    filtered = copy.deepcopy(predicted_requests_lists)
    tolerance_seconds = float(tolerance_minutes) * 60.0
    if real_requests_lists is None:
        return filtered

    for real_request in real_requests_lists.medications_requests:
        _remove_one_close_prediction(
            predicted_requests=filtered.medications_requests,
            real_request=real_request,
            tolerance_seconds=tolerance_seconds,
        )

    predicted_monitoring = _monitoring_requests(filtered)
    for data_field in MONITORING_REQUEST_FIELDS:
        for real_request in getattr(real_requests_lists, data_field):
            if _remove_one_close_prediction(
                predicted_requests=predicted_monitoring,
                real_request=real_request,
                tolerance_seconds=tolerance_seconds,
            ):
                for field_name in MONITORING_REQUEST_FIELDS:
                    setattr(filtered, field_name, [])
                for predicted_request in predicted_monitoring:
                    _append_request(filtered, predicted_request)
    return filtered


def requests_lists_to_time_buckets(requests_lists: RequestsLists) -> dict[float, RequestsLists]:
    buckets: dict[float, RequestsLists] = {}
    for request in _iter_requests(requests_lists):
        time_key = float(request.scheduled_time)
        bucket = buckets.setdefault(time_key, empty_requests_lists())
        _append_request(bucket, request)
    return buckets


class OfflinePredictionCache:
    def __init__(
        self,
        *,
        csv_path: str | Path | None,
        date_stamp: pd.Timestamp,
        floor_number: int,
        selected_run_names: str | Sequence[str] | None = None,
    ) -> None:
        self.csv_path = _resolve_path(csv_path)
        self.date_stamp = pd.Timestamp(date_stamp).date()
        self.floor_number = int(floor_number)
        self.selected_run_names = _parse_csv_set(selected_run_names)
        self.frame = self._load_frame()

    @property
    def enabled(self) -> bool:
        return self.csv_path is not None and not self.frame.empty

    def _load_frame(self) -> pd.DataFrame:
        if self.csv_path is None:
            return pd.DataFrame()
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Prediction cache CSV not found: {self.csv_path}")

        cache_csv_path = self.csv_path
        if self.csv_path.is_dir():
            cache_csv_path = _split_cache_shard_path(
                cache_dir=self.csv_path,
                floor_number=self.floor_number,
                date_stamp=pd.Timestamp(self.date_stamp),
            )
            if cache_csv_path is None:
                return pd.DataFrame()

        frame = pd.read_csv(cache_csv_path, compression="infer")
        if frame.empty:
            return frame
        if self.selected_run_names is not None and "run_name" in frame.columns:
            frame = frame[frame["run_name"].astype(str).isin(self.selected_run_names)].copy()
        if frame.empty:
            return frame

        frame["prediction_anchor_timestamp"] = pd.to_datetime(
            _column_or_default(frame, "prediction_anchor_timestamp"),
            errors="coerce",
        )
        frame["scheduled_timestamp"] = pd.to_datetime(
            _column_or_default(frame, "scheduled_dttm"),
            errors="coerce",
        )
        frame["sequence_day_date"] = pd.to_datetime(
            _column_or_default(frame, "sequence_day"),
            errors="coerce",
        ).dt.date
        frame["request_day_date"] = pd.to_datetime(
            _column_or_default(frame, "day"),
            errors="coerce",
        ).dt.date
        frame["floor_number"] = pd.to_numeric(
            _column_or_default(frame, "floor"),
            errors="coerce",
        )
        frame["prefix_event_count"] = pd.to_numeric(
            _column_or_default(frame, "prefix_event_count", 0),
            errors="coerce",
        ).fillna(0)
        frame["sample_index"] = pd.to_numeric(
            _column_or_default(frame, "sample_index", -1),
            errors="coerce",
        ).fillna(-1)
        frame["patient_key"] = _column_or_default(frame, "patient_id").map(_normalize_identifier)
        frame["encounter_key"] = _column_or_default(frame, "encounter_id").map(_normalize_identifier)

        day_mask = (frame["sequence_day_date"] == self.date_stamp) | (
            frame["request_day_date"] == self.date_stamp
        )
        frame = frame[day_mask].copy()
        if frame.empty:
            return frame
        return frame.reset_index(drop=True)

    def _observed_patient_keys(self, state: PlanningState) -> set[tuple[str, str]]:
        keys: set[tuple[str, str]] = set()
        for request in state.requests.values():
            patient_id = _normalize_identifier(getattr(request, "patient_id", ""))
            if not patient_id:
                continue
            encounter_id = _normalize_identifier(getattr(request, "encounter_id", ""))
            keys.add((patient_id, encounter_id))
        return keys

    def _latest_prefix_rows(
        self,
        *,
        current_timestamp: pd.Timestamp,
        state: PlanningState,
    ) -> pd.DataFrame:
        frame = self.frame
        if frame.empty:
            return frame
        frame = frame[frame["prediction_anchor_timestamp"].le(current_timestamp)].copy()
        if frame.empty:
            return frame

        observed_keys = self._observed_patient_keys(state)
        if observed_keys:
            observed_by_patient: dict[str, set[str]] = {}
            for patient_id, encounter_id in observed_keys:
                observed_by_patient.setdefault(patient_id, set()).add(encounter_id)

            def row_matches_observed_key(row: pd.Series) -> bool:
                patient_id = str(row["patient_key"])
                encounter_id = str(row["encounter_key"])
                observed_encounters = observed_by_patient.get(patient_id)
                if observed_encounters is None:
                    return False
                return not encounter_id or "" in observed_encounters or encounter_id in observed_encounters

            key_mask = frame.apply(
                row_matches_observed_key,
                axis=1,
            )
            frame = frame[key_mask].copy()
            if frame.empty:
                return frame

        group_columns = [
            "prediction_task",
            "family",
            "model_name",
            "variant",
            "run_name",
            "sequence_index",
            "patient_key",
            "encounter_key",
        ]
        present_group_columns = [column for column in group_columns if column in frame.columns]
        latest_prefix_ids = (
            frame.sort_values(
                ["prediction_anchor_timestamp", "prefix_event_count"],
                kind="mergesort",
            )
            .groupby(present_group_columns, dropna=False)["prefix_id"]
            .last()
            .dropna()
            .astype(str)
        )
        if latest_prefix_ids.empty:
            return frame.iloc[0:0].copy()
        return frame[frame["prefix_id"].astype(str).isin(set(latest_prefix_ids))].copy()

    def _row_to_task_request(
        self,
        *,
        row: Mapping[str, Any],
        initial_time: pd.Timestamp,
        all_task_properties: AllTaskProperties,
        traversal_graph_generator: TraversalGraphGenerator,
    ) -> TaskRequest | None:
        request_type = str(row.get("request_type", "")).strip()
        task_properties = _task_properties_for_request_type(
            all_task_properties=all_task_properties,
            request_type=request_type,
        )
        if task_properties is None:
            return None

        scheduled_timestamp = pd.to_datetime(row.get("scheduled_timestamp"), errors="coerce")
        if pd.isna(scheduled_timestamp):
            return None
        ordered_timestamp = pd.to_datetime(row.get("ordered_dttm"), errors="coerce")
        if pd.isna(ordered_timestamp):
            ordered_timestamp = pd.to_datetime(
                row.get("prediction_anchor_timestamp"),
                errors="coerce",
            )
        if pd.isna(ordered_timestamp):
            ordered_timestamp = scheduled_timestamp

        scheduled_time = (scheduled_timestamp - initial_time).total_seconds()
        ordered_time = (ordered_timestamp - initial_time).total_seconds()
        space_id = _normalize_identifier(row.get("scheduled_space_id", ""))
        supplies_space_id = _normalize_identifier(row.get("scheduled_space_supplies", ""))

        try:
            if request_type == "medication":
                supplies_node_label = traversal_graph_generator.doorway_to_node_dict[str(supplies_space_id)]
                room_node_label = traversal_graph_generator.doorway_to_node_dict[str(space_id)]
                goal_nodes = [supplies_node_label, room_node_label]
                wait_times_at_goals_seconds = [
                    task_properties.wait_time_seconds,
                    task_properties.wait_time_seconds,
                ]
            else:
                room_node_label = traversal_graph_generator.doorway_to_node_dict[str(space_id)]
                goal_nodes = [room_node_label]
                wait_times_at_goals_seconds = [task_properties.wait_time_seconds]
        except KeyError:
            return None

        request = TaskRequest(
            request_id=f"pred.{row.get('predicted_request_id')}",
            request_type=request_type,
            goal_nodes=goal_nodes,
            wait_times_at_goals_seconds=wait_times_at_goals_seconds,
            time_for_rejection_minutes=task_properties.time_for_rejection_minutes,
            ordered_time=ordered_time,
            scheduled_time=scheduled_time,
            administered_time=None,
            patient_id=_normalize_identifier(row.get("patient_id", "")),
            mrn=_normalize_identifier(row.get("mrn", row.get("patient_id", ""))),
            encounter_id=_normalize_identifier(row.get("encounter_id", "")),
            scheduled_room=str(row.get("scheduled_room", "")),
            scheduled_space_id=space_id,
            scheduled_space_supplies=supplies_space_id,
            floor=_int_or_none(row.get("floor_number", row.get("floor"))),
            scheduled_day=str(row.get("day", "")),
        )
        request.prediction_sample_index = int(row.get("sample_index", -1))
        request.prediction_prefix_id = str(row.get("prefix_id", ""))
        return request

    def prediction_sample_sets(
        self,
        *,
        state: PlanningState,
        real_requests_lists: Optional[RequestsLists],
        initial_time: pd.Timestamp,
        all_task_properties: AllTaskProperties,
        traversal_graph_generator: TraversalGraphGenerator,
        lookahead_minutes: float,
        match_tolerance_minutes: float,
        planning_horizon_end_timestamp: pd.Timestamp | None = None,
    ) -> list[dict[float, RequestsLists]]:
        if not self.enabled:
            return []

        current_timestamp = pd.Timestamp(initial_time) + pd.Timedelta(
            seconds=float(state.simulator_time)
        )
        horizon_timestamp = current_timestamp + pd.Timedelta(minutes=float(lookahead_minutes))
        if planning_horizon_end_timestamp is not None:
            horizon_timestamp = min(
                horizon_timestamp,
                pd.Timestamp(planning_horizon_end_timestamp),
            )
        if horizon_timestamp <= current_timestamp:
            return []
        prefix_rows = self._latest_prefix_rows(
            current_timestamp=current_timestamp,
            state=state,
        )
        if prefix_rows.empty:
            return []

        sample_sets: dict[int, RequestsLists] = {}
        for sample_index in pd.to_numeric(prefix_rows["sample_index"], errors="coerce").dropna().unique():
            sample_sets[int(sample_index)] = empty_requests_lists()

        request_rows = prefix_rows[
            _column_or_default(prefix_rows, "row_kind").astype(str).eq("sampled_request")
        ].copy()
        if not request_rows.empty:
            time_mask = request_rows["scheduled_timestamp"].ge(current_timestamp) & request_rows[
                "scheduled_timestamp"
            ].lt(horizon_timestamp)
            floor_mask = request_rows["floor_number"].eq(float(self.floor_number))
            request_rows = request_rows[time_mask & floor_mask].copy()

        for _, row in request_rows.iterrows():
            sample_index = int(row.get("sample_index", -1))
            sample_requests = sample_sets.setdefault(sample_index, empty_requests_lists())
            request = self._row_to_task_request(
                row=row,
                initial_time=initial_time,
                all_task_properties=all_task_properties,
                traversal_graph_generator=traversal_graph_generator,
            )
            if request is not None:
                _append_request(sample_requests, request)

        sample_buckets: list[dict[float, RequestsLists]] = []
        for sample_index in sorted(sample_sets):
            filtered_requests = remove_real_request_matches(
                predicted_requests_lists=sample_sets[sample_index],
                real_requests_lists=real_requests_lists,
                tolerance_minutes=match_tolerance_minutes,
            )
            sample_buckets.append(requests_lists_to_time_buckets(filtered_requests))
        return sample_buckets
