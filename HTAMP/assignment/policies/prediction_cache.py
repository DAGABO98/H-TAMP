from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import pandas as pd

from HTAMP.data_processing.data_helpers import DataHelpers
from HTAMP.environment.traversal_graph_gen import TraversalGraphGenerator
from HTAMP.planning.planning_dataclasses import AllTaskProperties, RequestsLists, TaskProperties, TaskRequest
from HTAMP.planning.state import PlanningState


DEFAULT_ANNOTATED_VISITS_PATH = "data/processed/patient_room_stays.csv"
DEFAULT_ANNOTATED_ADMISSIONS_DISCHARGES_PATH = "data/processed/admissions_discharges.csv"
ENCOUNTER_ID_COLUMN = "Patient Encounter CSN"
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


def _first_existing_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    normalized_columns = {
        str(column).strip().lower(): str(column)
        for column in frame.columns
    }
    for candidate in candidates:
        matched = normalized_columns.get(str(candidate).strip().lower())
        if matched is not None:
            return matched
    return None


def _resolve_path_with_default(path_value: str | Path | None, default_path: str) -> Path | None:
    return _resolve_path(path_value) or _resolve_path(default_path)


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


class ActivePatientFloorFilter:
    def __init__(
        self,
        *,
        floor_number: int,
        annotated_visits_path: str | Path | None = None,
        annotated_admissions_discharges_path: str | Path | None = None,
    ) -> None:
        self.floor_number = int(floor_number)
        self.room_stays = self._load_room_stays(annotated_visits_path)
        self.admissions = self._load_admissions(annotated_admissions_discharges_path)

    @property
    def enabled(self) -> bool:
        return not self.room_stays.empty

    def _load_room_stays(self, path_value: str | Path | None) -> pd.DataFrame:
        columns = ["patient_key", "encounter_key", "location", "start", "end", "floor_number"]
        path = _resolve_path_with_default(path_value, DEFAULT_ANNOTATED_VISITS_PATH)
        if path is None or not path.exists():
            return pd.DataFrame(columns=columns)

        room_df = pd.read_csv(path)
        patient_col = _first_existing_column(room_df, ["patient_id", "MRN", "PAT_ID"])
        location_col = _first_existing_column(room_df, ["location", "scheduled_room", "room"])
        start_col = _first_existing_column(
            room_df,
            ["start", "scheduled_start", "IN_TIME", "HOSPITAL_ADMISSION"],
        )
        end_col = _first_existing_column(
            room_df,
            ["end", "scheduled_end", "OUT_TIME", "HOSPITAL_DISCHARGE"],
        )
        if patient_col is None or location_col is None or start_col is None or end_col is None:
            return pd.DataFrame(columns=columns)

        encounter_col = _first_existing_column(
            room_df,
            ["encounter_id", ENCOUNTER_ID_COLUMN, "PAT_ENC_CSN_ID"],
        )
        floor_col = _first_existing_column(room_df, ["floor", "floor_number", "__floor__"])
        normalized = pd.DataFrame(
            {
                "patient_key": room_df[patient_col].map(_normalize_identifier),
                "encounter_key": (
                    room_df[encounter_col].map(_normalize_identifier)
                    if encounter_col is not None
                    else pd.Series("", index=room_df.index)
                ),
                "location": room_df[location_col].astype(str),
                "start": pd.to_datetime(room_df[start_col], errors="coerce"),
                "end": pd.to_datetime(room_df[end_col], errors="coerce"),
            }
        )
        location_floor = normalized["location"].map(DataHelpers.extract_floor)
        if floor_col is not None:
            normalized["floor_number"] = pd.to_numeric(
                room_df[floor_col],
                errors="coerce",
            ).fillna(location_floor)
        else:
            normalized["floor_number"] = location_floor
        normalized = normalized.dropna(subset=["patient_key", "start", "floor_number"]).copy()
        normalized["end"] = normalized["end"].fillna(pd.Timestamp.max)
        normalized["floor_number"] = normalized["floor_number"].astype(int)
        return normalized.sort_values(
            ["patient_key", "start", "end"],
            kind="mergesort",
        ).reset_index(drop=True)

    def _load_admissions(self, path_value: str | Path | None) -> pd.DataFrame:
        columns = ["patient_key", "encounter_key", "admission_start", "discharge_end"]
        path = _resolve_path_with_default(
            path_value,
            DEFAULT_ANNOTATED_ADMISSIONS_DISCHARGES_PATH,
        )
        if path is None or not path.exists():
            return pd.DataFrame(columns=columns)

        admissions_df = pd.read_csv(path)
        patient_col = _first_existing_column(admissions_df, ["patient_id", "MRN", "PAT_ID"])
        encounter_col = _first_existing_column(
            admissions_df,
            ["encounter_id", ENCOUNTER_ID_COLUMN, "PAT_ENC_CSN_ID"],
        )
        admission_col = _first_existing_column(
            admissions_df,
            ["admission_start", "admission", "HOSPITAL_ADMISSION", "Hospital Admission"],
        )
        discharge_col = _first_existing_column(
            admissions_df,
            ["discharge_end", "discharge", "HOSPITAL_DISCHARGE", "Hospital Discharge"],
        )
        if patient_col is None or admission_col is None:
            return pd.DataFrame(columns=columns)

        normalized = pd.DataFrame(
            {
                "patient_key": admissions_df[patient_col].map(_normalize_identifier),
                "encounter_key": (
                    admissions_df[encounter_col].map(_normalize_identifier)
                    if encounter_col is not None
                    else pd.Series("", index=admissions_df.index)
                ),
                "admission_start": pd.to_datetime(admissions_df[admission_col], errors="coerce"),
                "discharge_end": (
                    pd.to_datetime(admissions_df[discharge_col], errors="coerce")
                    if discharge_col is not None
                    else pd.Series(pd.NaT, index=admissions_df.index)
                ),
            }
        )
        normalized = normalized.dropna(subset=["patient_key", "admission_start"]).copy()
        normalized["discharge_end"] = normalized["discharge_end"].fillna(pd.Timestamp.max)
        return normalized.sort_values(
            ["patient_key", "encounter_key", "admission_start"],
            kind="mergesort",
        ).reset_index(drop=True)

    def _active_encounter_keys(
        self,
        *,
        patient_key: str,
        room_encounter_key: str,
        timestamp: pd.Timestamp,
    ) -> set[str]:
        if self.admissions.empty:
            return {room_encounter_key}
        patient_admissions = self.admissions[self.admissions["patient_key"].eq(patient_key)]
        if patient_admissions.empty:
            return {room_encounter_key}
        active_admissions = patient_admissions[
            patient_admissions["admission_start"].le(timestamp)
            & patient_admissions["discharge_end"].gt(timestamp)
        ]
        if room_encounter_key:
            active_admissions = active_admissions[
                active_admissions["encounter_key"].eq(room_encounter_key)
            ]
        if active_admissions.empty:
            return set()
        encounter_keys = {
            _normalize_identifier(encounter_key)
            for encounter_key in active_admissions["encounter_key"].tolist()
        }
        return encounter_keys or {room_encounter_key}

    def active_patient_keys(self, *, timestamp: pd.Timestamp) -> set[tuple[str, str]] | None:
        if not self.enabled:
            return None
        timestamp = pd.Timestamp(timestamp)
        active_stays = self.room_stays[
            self.room_stays["floor_number"].eq(int(self.floor_number))
            & self.room_stays["start"].le(timestamp)
            & self.room_stays["end"].gt(timestamp)
        ]
        active_keys: set[tuple[str, str]] = set()
        for row in active_stays.itertuples(index=False):
            patient_key = _normalize_identifier(getattr(row, "patient_key", ""))
            if not patient_key:
                continue
            room_encounter_key = _normalize_identifier(getattr(row, "encounter_key", ""))
            encounter_keys = self._active_encounter_keys(
                patient_key=patient_key,
                room_encounter_key=room_encounter_key,
                timestamp=timestamp,
            )
            for encounter_key in encounter_keys:
                active_keys.add((patient_key, encounter_key))
        return active_keys


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

    def _filter_rows_to_patient_keys(
        self,
        *,
        frame: pd.DataFrame,
        patient_keys: set[tuple[str, str]],
    ) -> pd.DataFrame:
        if not patient_keys:
            return frame.iloc[0:0].copy()

        allowed_by_patient: dict[str, set[str]] = {}
        for patient_id, encounter_id in patient_keys:
            allowed_by_patient.setdefault(patient_id, set()).add(encounter_id)

        def row_matches_patient_key(row: pd.Series) -> bool:
            patient_id = str(row["patient_key"])
            encounter_id = str(row["encounter_key"])
            allowed_encounters = allowed_by_patient.get(patient_id)
            if allowed_encounters is None:
                return False
            return not encounter_id or "" in allowed_encounters or encounter_id in allowed_encounters

        return frame[
            frame.apply(
                row_matches_patient_key,
                axis=1,
            )
        ].copy()

    def _latest_prefix_rows(
        self,
        *,
        current_timestamp: pd.Timestamp,
        state: PlanningState,
        filter_to_observed_patients: bool = True,
        patient_key_filter: set[tuple[str, str]] | None = None,
    ) -> pd.DataFrame:
        frame = self.frame
        if frame.empty:
            return frame
        frame = frame[frame["prediction_anchor_timestamp"].le(current_timestamp)].copy()
        if frame.empty:
            return frame

        observed_keys = self._observed_patient_keys(state) if filter_to_observed_patients else set()
        if observed_keys:
            frame = self._filter_rows_to_patient_keys(
                frame=frame,
                patient_keys=observed_keys,
            )
            if frame.empty:
                return frame

        if patient_key_filter is not None:
            frame = self._filter_rows_to_patient_keys(
                frame=frame,
                patient_keys=patient_key_filter,
            )
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
        filter_to_observed_patients: bool = True,
        patient_key_filter: set[tuple[str, str]] | None = None,
        remove_real_matches: bool = True,
        sample_limit: int | None = None,
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
            filter_to_observed_patients=filter_to_observed_patients,
            patient_key_filter=patient_key_filter,
        )
        if prefix_rows.empty:
            return []

        sample_indices = sorted(
            int(sample_index)
            for sample_index in pd.to_numeric(prefix_rows["sample_index"], errors="coerce").dropna().unique()
        )
        if sample_limit is not None and int(sample_limit) > 0 and len(sample_indices) > int(sample_limit):
            limit = int(sample_limit)
            if limit == 1:
                sample_indices = [sample_indices[0]]
            else:
                selected_indices = {
                    round(i * (len(sample_indices) - 1) / (limit - 1))
                    for i in range(limit)
                }
                sample_indices = [sample_indices[i] for i in sorted(selected_indices)]

        sample_index_set = set(sample_indices)
        sample_sets: dict[int, RequestsLists] = {
            sample_index: empty_requests_lists()
            for sample_index in sample_indices
        }

        request_rows = prefix_rows[
            _column_or_default(prefix_rows, "row_kind").astype(str).eq("sampled_request")
        ].copy()
        if not request_rows.empty:
            request_rows = request_rows[
                pd.to_numeric(request_rows["sample_index"], errors="coerce").isin(sample_index_set)
            ].copy()
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
            if remove_real_matches:
                filtered_requests = remove_real_request_matches(
                    predicted_requests_lists=sample_sets[sample_index],
                    real_requests_lists=real_requests_lists,
                    tolerance_minutes=match_tolerance_minutes,
                )
            else:
                filtered_requests = copy.deepcopy(sample_sets[sample_index])
            sample_buckets.append(requests_lists_to_time_buckets(filtered_requests))
        return sample_buckets
