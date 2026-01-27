

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from HTAMP.environment.loc_dataclasses import Coordinate, GridIndex, TimeInterval
from HTAMP.environment.robot_dataclasses import RobotProfile
from HTAMP.environment.traversal_dataclasses import TraversalNode

@dataclass
class TimeReservation:
    interval: TimeInterval
    robot_id: int

    def overlaps(self, other: "TimeReservation") -> bool:
        return round(self.interval.start, 4) <= round(other.interval.end, 4) and \
            round(self.interval.end, 4) >= round(other.interval.start, 4)
    
    def merge(self, other: "TimeReservation") -> "TimeReservation":
        if not self.overlaps(other):
            raise ValueError("Intervals do not overlap and cannot be merged.")
        
        if self.robot_id != other.robot_id:
            raise ValueError("Cannot merge reservations for different robots.")
        
        new_start = min(self.interval.start, other.interval.start)
        new_end = max(self.interval.end, other.interval.end)
        return TimeReservation(TimeInterval(new_start, new_end), self.robot_id)
    
    def duration(self) -> float:
        return self.interval.end - self.interval.start

    def complement(self, horizon: float) -> list["TimeReservation"]:
        """Return the complement intervals within [0, horizon)."""
        complements: list[TimeReservation] = []
        if self.interval.start > 0:
            complements.append(TimeReservation(TimeInterval(0, self.interval.start), self.robot_id))
        if self.interval.end < horizon:
            complements.append(TimeReservation(TimeInterval(self.interval.end, horizon), self.robot_id))
        return complements
    
    def __repr__(self) -> str:
        return f"TimeReservation(start={self.interval.start}, end={self.interval.end}, robot_id={self.robot_id})"

@dataclass
class ReservationTable:
    reservations: dict[GridIndex, list[TimeReservation]]
    robot_cell_dict: dict[int, list[GridIndex]] = field(default_factory=dict)

    def get_reservations(self, cell: GridIndex) -> list[TimeReservation]:
        return self.reservations.get(cell, [])

    def add_reservation(self, cell: GridIndex, reservation: TimeReservation) -> TimeReservation:
        if cell in self.reservations:
            for i, existing in enumerate(self.reservations[cell]):
                if existing.overlaps(reservation):
                    if existing.robot_id == reservation.robot_id:
                        merged = existing.merge(reservation)
                        self.reservations[cell][i] = merged
                        return merged
                    else:
                        raise ValueError(f"Conflict detected for cell {cell} between robot {existing.robot_id} and robot {reservation.robot_id}.")
            self.reservations[cell].append(reservation)
            self.robot_cell_dict.setdefault(reservation.robot_id, []).append(cell)
        else:
            self.reservations[cell] = [reservation]
            self.robot_cell_dict.setdefault(reservation.robot_id, []).append(cell)
        return reservation

    def _remove_cell_reservation_for_robot(self, cell: GridIndex, robot_id: int) -> None:
        if cell in self.reservations:
            self.reservations[cell] = [r for r in self.reservations[cell] if r.robot_id != robot_id]
            if not self.reservations[cell]:
                del self.reservations[cell]
    
    def remove_reservations_for_robot(self, robot_id: int) -> None:
        if robot_id in self.robot_cell_dict:
            for cell in self.robot_cell_dict[robot_id]:
                self._remove_cell_reservation_for_robot(cell, robot_id)
            del self.robot_cell_dict[robot_id]

    def check_conflict(self, cell: GridIndex, reservation: TimeReservation) -> bool:
        for existing in self.get_reservations(cell):
            if existing.overlaps(reservation) and existing.robot_id != reservation.robot_id:
                return True
        return False

    def get_safe_intervals(self, 
                           cell: GridIndex,
                           horizon: float = float('inf'), 
                           robot_id: int = 1) -> list[TimeReservation]:
        blocked = sorted([r for r in self.get_reservations(cell) if r.robot_id != robot_id], 
                         key=lambda r: r.interval.start)
        safe_intervals: list[TimeReservation] = []
        current_time = 0.0
        for reservation in blocked:
            if reservation.interval.start > current_time:
                safe_intervals.append(TimeReservation(TimeInterval(current_time, reservation.interval.start), robot_id=robot_id))
            current_time = max(current_time, reservation.interval.end)
        if current_time < horizon:
            safe_intervals.append(TimeReservation(TimeInterval(current_time, horizon), robot_id=robot_id))
        return safe_intervals

class DateOperationalRange:
    def __init__(self, year: int, month: int, day: int, start_hour: int, end_hour: int, month_lengths=None):
        self.start_date = (year, month, day, start_hour)
        self.end_date = (year, month, day, end_hour)
        if month_lengths is None:
            month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        self.month_lengths = month_lengths

    def __repr__(self):
        return f"DateOperationalRange(start_date={self.start_date}, end_date={self.end_date})"

class SimulatorConfig:
    def __init__(self, 
                 fps: int,
                 robot_profiles: dict[int, RobotProfile], 
                 rejection_penalty: float, 
                 initial_time: pd.Timestamp,
                 horizon: float,
                 initial_robot_positions: dict[int, Coordinate],
                 initial_nodes: dict[int, TraversalNode]):
        self.robot_profiles = robot_profiles
        self.rejection_penalty = rejection_penalty
        self.initial_time = initial_time
        self.initial_robot_positions = initial_robot_positions
        self.initial_nodes = initial_nodes
        self.fps = fps
        self.time_step = 1.0 / fps
        self.horizon = horizon

class DateStamp:
    def __init__(self, year: int, month, day):
        self.year = year
        self.month = month
        self.day = day
        self.time_stamp = pd.Timestamp(year=year,
                                  month=month,
                                  day=day)
        self.weekday = self.time_stamp.weekday()
    
    def __repr__(self):
        date_stamp_str = str(self.year) + "-" + str(self.month) + "-" + str(self.day)

        return date_stamp_str

class TimeSignal:
    def __init__(self, year: int, month, day, hour, minute):
        self.year = year
        self.month = month
        self.day = day
        self.hour = hour
        self.minute = minute
        self.time_stamp = pd.Timestamp(year=year,
                                  month=month,
                                  day=day,
                                  hour=hour,
                                  minute=minute)
        self.weekday = self.time_stamp.weekday()
    
    def __repr__(self):
        time_signal_str = str(self.year) + "-" + str(self.month) + "-" + str(self.day)+ " " + str(self.hour) + ":" + str(self.minute)

        return time_signal_str

@dataclass
class TaskProperties:
    task_type: str
    wait_time_seconds: float
    time_for_rejection_minutes: float

@dataclass
class AllTaskProperties:
    blood_pressure: TaskProperties
    heart_rate: TaskProperties
    respiratory_rate: TaskProperties
    temperature: TaskProperties
    oxygen_saturation: TaskProperties
    medications: TaskProperties

class TaskRequest:
    def __init__(self, 
                 request_id: str, 
                 request_type: str, 
                 goal_nodes: list[str], 
                 wait_times_at_goals_seconds: list[float],
                 time_for_rejection_minutes: float,
                 ordered_time: float,
                 scheduled_time: float,
                 assigned_robot_id: Optional[int] = None,
                 started: bool = False,
                 completed_goals: int = 0,
                 completed: bool = False,
                 rejected: bool = False,
                 planned_time: Optional[float] = None,
                 planned_goal_indices: Optional[list[int]] = None):
        self.request_id = request_id
        self.request_type = request_type
        self.goal_nodes = goal_nodes
        self.wait_times_at_goals_seconds = wait_times_at_goals_seconds
        self.ordered_time = ordered_time
        self.scheduled_time = scheduled_time
        self.time_for_service = scheduled_time + (60.0 * time_for_rejection_minutes)
        self.started = started
        self.completed_goals = completed_goals
        self.completed = completed
        self.rejected = rejected
        self.total_cost = 0.0
        self.planned_goal_indices = planned_goal_indices if planned_goal_indices is not None else []
        self.planned_time = planned_time if planned_time is not None else -1.0
        self.assigned_robot_id = assigned_robot_id

    def mark_completed(self, completion_time: float) -> None:
        self.completed = True
        self.total_cost = completion_time - self.scheduled_time
        print(f"Request {self.request_id} completed at time {completion_time:.2f} with total cost {self.total_cost:.2f}.")
    
    def mark_rejected(self, rejection_penalty: float) -> None:
        self.rejected = True
        self.total_cost = rejection_penalty
    
    def mark_started(self) -> None:
        self.started = True

    def schedule_task(self, planned_time: float, planned_goal_indices: list[int], assigned_robot_id: Optional[int] = None) -> None:
        self.completed_goals = 0
        self.planned_time = planned_time    
        self.planned_goal_indices = planned_goal_indices
        if assigned_robot_id is not None:
            self.assigned_robot_id = assigned_robot_id
    
    def reset_assignment(self) -> None:
        self.planned_time = -1.0
        self.planned_goal_indices = []
        self.assigned_robot_id = None
        self.completed_goals = 0
        self.started = False
    
    def is_expired(self, current_time: float) -> bool:
        return current_time >= self.time_for_service
    
    def is_started(self) -> bool:
        return self.started
    
    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "request_type": self.request_type,
            "goal_nodes": self.goal_nodes,
            "wait_times_at_goals_seconds": self.wait_times_at_goals_seconds,
            "ordered_time": self.ordered_time,
            "scheduled_time": self.scheduled_time,
            "time_for_service": self.time_for_service,
            "started": self.started,
            "completed_goals": self.completed_goals,
            "completed": self.completed,
            "rejected": self.rejected,
            "total_cost": self.total_cost,
            "planned_time": self.planned_time,
            "planned_goal_indices": self.planned_goal_indices,
            "assigned_robot_id": self.assigned_robot_id,
        }

    def __repr__(self):
        return f"Request(request_id={self.request_id}, request_type='{self.request_type}'," + \
            f"ordered_time={self.ordered_time}, scheduled_time={self.scheduled_time}, time_for_service={self.time_for_service}, " + \
            f"goal_nodes={self.goal_nodes}, wait_times_at_goals_seconds={self.wait_times_at_goals_seconds}, " + \
            f"started={self.started}, completed_goals={self.completed_goals}, completed={self.completed}, rejected={self.rejected}, " + \
            f"planned_time={self.planned_time}, planned_goal_indices={self.planned_goal_indices}, total_cost={self.total_cost}, assigned_robot_id={self.assigned_robot_id})"
    
@dataclass
class RequestsLists:
    blood_pressure_requests: list[TaskRequest]
    heart_rate_requests: list[TaskRequest]
    respiratory_rate_requests: list[TaskRequest]
    temperature_requests: list[TaskRequest]
    oxygen_saturation_requests: list[TaskRequest]
    medications_requests: list[TaskRequest]

class FrameData:
    def __init__(self):
        self.robot_positions_seq: list[dict[int, Coordinate]] = []
        self.robots_current_node_index_seq: list[dict[int, int]] =[]
        self.point_indices_on_edge_seq: list[dict[int, int]] =[]
        self.robot_paths_seq: list[dict[int, list[tuple[TraversalNode, TimeInterval]]]] = []
        self.planned_goal_indices_seq: list[dict[int, list[int]]] = []
        self.completed_goals_seq: list[dict[int, int]] = []