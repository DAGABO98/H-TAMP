import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple, Dict, Any
from matplotlib.patches import Rectangle, Circle

from HTAMP.grid_world import Coordinate, PosChange, TimeInterval, GridWorld, GridIndex

@dataclass
class TimeReservation:
    interval: TimeInterval
    robot_id: int

    def overlaps(self, other: "TimeReservation") -> bool:
        return self.interval.start < other.interval.end and self.interval.end > other.interval.start
    
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

    def complement(self, horizon: float) -> List["TimeReservation"]:
        """Return the complement intervals within [0, horizon)."""
        complements: List[TimeReservation] = []
        if self.interval.start > 0:
            complements.append(TimeReservation(TimeInterval(0, self.interval.start), self.robot_id))
        if self.interval.end < horizon:
            complements.append(TimeReservation(TimeInterval(self.interval.end, horizon), self.robot_id))
        return complements
    
    def __repr__(self) -> str:
        return f"TimeReservation(start={self.interval.start}, end={self.interval.end}, robot_id={self.robot_id})"

@dataclass
class ReservationTable:
    grid: GridWorld
    reservations: Dict[GridIndex, List[TimeReservation]]

    def get_reservations(self, cell: GridIndex) -> List[TimeReservation]:
        return self.reservations.get(cell, [])

    def add_reservation(self, cell: GridIndex, reservation: TimeReservation) -> None:
        if cell in self.reservations:
            for i, existing in enumerate(self.reservations[cell]):
                if existing.overlaps(reservation) and existing.robot_id == reservation.robot_id:
                    merged = existing.merge(reservation)
                    self.reservations[cell][i] = merged
                    return merged
            self.reservations[cell].append(reservation)
        else:
            self.reservations[cell] = [reservation]
        return reservation
    
    def remove_reservation(self, cell: GridIndex, reservation: TimeReservation) -> None:
        if cell in self.reservations:
            self.reservations[cell] = [r for r in self.reservations[cell] if r != reservation]
            if not self.reservations[cell]:
                del self.reservations[cell]
    
    def check_conflict(self, cell: GridIndex, reservation: TimeReservation) -> bool:
        for existing in self.get_reservations(cell):
            if existing.overlaps(reservation) and existing.robot_id != reservation.robot_id:
                return True
        return False

    def get_safe_intervals(self, cell: GridIndex,
                           horizon: float = float('inf'), 
                           robot_id: int = 1) -> List[TimeReservation]:
        blocked = sorted(self.get_reservations(cell), key=lambda r: r.interval.start)
        safe_intervals: List[TimeReservation] = []
        current_time = 0.0
        for reservation in blocked:
            if reservation.interval.start > current_time:
                safe_intervals.append(TimeReservation(TimeInterval(current_time, reservation.interval.start), robot_id=robot_id))
            current_time = max(current_time, reservation.interval.end)
        if current_time < horizon:
            safe_intervals.append(TimeReservation(TimeInterval(current_time, horizon), robot_id=robot_id))
        return safe_intervals

@dataclass(order=True)
class PQItem:
    f: int
    g: int
    node: "SIPPNode" = field(compare=False)

@dataclass
class SIPPNode:
    pos: Coordinate
    interval: TimeInterval
    arrival: int
    parent: Optional["SIPPNode"] = None

class MotionPlanner:
    def __init__(self, grid: GridWorld):
        self.grid = grid
        self.reservation_table = ReservationTable(grid=grid, reservations={})
    
    def heuristic(self, pos: Coordinate, goal: Coordinate) -> float:
        return np.linalg.norm(np.array([pos.x - goal.x, pos.y - goal.y]))
    
    def _intersect_intervals(self, 
                             list1: List[TimeReservation],
                             list2: List[TimeReservation]) -> List[TimeReservation]:
        result: List[TimeReservation] = []
        i, j = 0, 0
        while i < len(list1) and j < len(list2):
            a, b = list1[i], list2[j]
            if a.interval.end <= b.interval.start:
                i += 1
            elif b.interval.end <= a.interval.start:
                j += 1
            else:
                start = max(a.interval.start, b.interval.start)
                end = min(a.interval.end, b.interval.end)
                if start < end:
                    result.append(TimeReservation(TimeInterval(start, end), robot_id=-1))
                if a.interval.end < b.interval.end:
                    i += 1
                else:
                    j += 1
        return result

    def _get_safe_intervals_for_path(self, start_pos: Coordinate, end_pos: Coordinate,
                                      robot_radius: float, robot_velocity: float,
                                      horizon: float = float('inf'),
                                 robot_id: int = 1) -> List[Tuple[GridIndex, List[TimeReservation]]]:
        direction = np.array([end_pos.x - start_pos.x, end_pos.y - start_pos.y])
        distance = np.linalg.norm(direction)
        if distance == 0:
            return []
        direction /= distance
        
        major_axis = np.argmax(np.abs(direction))
        minor_axis = 1 - major_axis
        major_displacement = direction[major_axis] * self.grid.cell_size
        minor_displacement = direction[minor_axis] * self.grid.cell_size
        
        num_steps = int(np.ceil(distance / self.grid.cell_size))
        safe_intervals_along_path: List[Tuple[GridIndex, List[TimeReservation]]] = []
        
        for step in range(num_steps + 1):
            if step == num_steps:
                curr_pos = end_pos
            else:
                curr_pos = Coordinate(
                    x=start_pos.x + step * major_displacement if major_axis == 0 else start_pos.x + step * minor_displacement,
                    y=start_pos.y + step * minor_displacement if minor_axis == 1 else start_pos.y + step * major_displacement
                )
            
            if abs(major_displacement) > abs(minor_displacement):
                pos_change = PosChange(dev_x=minor_displacement, dev_y=major_displacement)
            else:
                pos_change = PosChange(dev_x=major_displacement, dev_y=minor_displacement)
            curr_cells: Set[GridIndex] = self.grid.get_occupied_cells_for_partial_move(robot_start_pos=start_pos, 
                                                                                       robot_radius=robot_radius, 
                                                                                       pos_change=pos_change)
            total_displacement = np.sqrt(minor_displacement**2 + major_displacement**2)
            time_to_end = total_displacement / robot_velocity if robot_velocity > 0 else 0
            
            step_safe_intervals: List[TimeReservation] = []
            for cell in curr_cells:
                cell_safe_intervals = self.reservation_table.get_safe_intervals(cell, horizon=horizon, robot_id=robot_id)
                if not step_safe_intervals:
                    step_safe_intervals = cell_safe_intervals
                else:
                    step_safe_intervals = self._intersect_intervals(step_safe_intervals, cell_safe_intervals)

            if step_safe_intervals:
                safe_intervals_along_path.append((self.grid.get_cell_index(curr_pos), step_safe_intervals))
            current_time += time_to_end
        return safe_intervals_along_path
    
    def _get_earliest_departure(self,
                                curr_node: SIPPNode,
                                start_pos: Coordinate,
                                end_pos: Coordinate) -> Optional[float]:
        # Find the earliest departure time for the current node
        earliest_departure = None
        for interval in self.reservation_table.get_reserved_intervals(start_pos, end_pos):
            if interval.start > curr_node.arrival_time:
                earliest_departure = interval.start
                break
        return earliest_departure

    def plan_path(self, 
                  start: Coordinate, 
                  goal: Coordinate, 
                  robot_radius: float, 
                  robot_velocity: float,
                  start_time: float = 0.0,
                  horizon: float = float('inf'),
                  robot_id: int = 1) -> Optional[List[Tuple[Coordinate, TimeInterval]]]:
        # Plan the path from start to goal while avoiding obstacles
        safe_intervals = self._get_safe_intervals_for_path(start, goal, robot_radius, robot_velocity, horizon, robot_id)
        if not safe_intervals:
            return None

        # Find a valid path through the safe intervals
        path = self._find_path_through_intervals(start, goal, safe_intervals)
        if not path:
            return None

        # Reserve the path
        if not self._reserve_path(path, robot_id):
            return None

        return path