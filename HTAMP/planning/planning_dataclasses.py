

from dataclasses import dataclass, field
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
                 robot_profiles: list[RobotProfile], 
                 rejection_penalty: float, 
                 date_range: DateOperationalRange,
                 horizon: float,
                 initial_robot_positions: dict[int, Coordinate]):
        self.robot_profiles = robot_profiles
        self.rejection_penalty = rejection_penalty
        self.date_range = date_range
        self.initial_robot_positions = initial_robot_positions
        self.fps = fps
        self.time_step = 1.0 / fps
        self.horizon = horizon

class TaskRequest:
    def __init__(self, 
                 request_id: int, 
                 request_type: str, 
                 goal_nodes: list[str], 
                 wait_times_at_goals: list[float],
                 start_time: float = 0.0,
                 end_time: float = 0.0,
                 desired_time_for_service: float = 0.0,
                 planned_time_for_service: float = 0.0,
                 started: bool = False,
                 completed_goals: int = 0,
                 completed: bool = False,
                 completion_time: float = 0.0,
                 planned_goal_indices: list[int] = None
                 ):
        self.request_id = request_id
        self.request_type = request_type
        self.goal_nodes = goal_nodes
        self.wait_times_at_goals = wait_times_at_goals
        self.start_time = start_time
        self.end_time = end_time
        self.desired_time_for_service = desired_time_for_service
        self.planned_time_for_service = planned_time_for_service
        self.started = started
        self.planned_goal_indices = planned_goal_indices
        self.completed_goals = completed_goals
        self.completed = completed
        self.completion_time = completion_time
        self.total_cost = 0.0
    
    def mark_completed(self, completion_time: float) -> None:
        self.completed = True
        self.completion_time = completion_time
        self.total_cost = max(self.completion_time - self.desired_time_for_service, 0.0)
    
    def mark_started(self) -> None:
        self.started = True

    def schedule_task(self, planned_time: float, planned_goal_indices: list[int]) -> None:
        self.completed_goals = 0
        self.planned_time_for_service = planned_time
        self.planned_goal_indices = planned_goal_indices

    def __repr__(self):
        return f"Request(request_id={self.request_id}, request_type='{self.request_type}')"