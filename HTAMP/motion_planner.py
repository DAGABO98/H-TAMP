import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Set, Tuple, Dict, Any
from matplotlib.patches import Rectangle, Circle

from HTAMP.grid_world import TimeInterval, GridWorld, GridIndex

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

    def complement(self, horizon: float, current_time: float) -> List["TimeReservation"]:
        """Return the complement intervals within [0, horizon)."""
        if self.interval.start > current_time:
            return [TimeReservation(TimeInterval(current_time, self.interval.start), robot_id=self.robot_id),
                    TimeReservation(TimeInterval(self.interval.end, horizon), robot_id=self.robot_id)]
        elif self.interval.start <= current_time < self.interval.end:
            return [TimeReservation(TimeInterval(self.interval.end, horizon), robot_id=self.robot_id)]
        else:
            return [TimeReservation(TimeInterval(current_time, horizon), robot_id=self.robot_id)]
    
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
    
    def get_safe_intervals(self, cell: GridIndex, horizon: float = float('inf')) -> List[TimeReservation]:
        blocked = sorted(self.get_reservations(cell), key=lambda r: r.interval.start)
        safe_intervals: List[TimeReservation] = []
        current_time = 0.0
        for reservation in blocked:
            if reservation.interval.start > current_time:
                safe_intervals.append(TimeReservation(TimeInterval(current_time, reservation.interval.start), robot_id=-1))
            current_time = max(current_time, reservation.interval.end)
        if current_time < horizon:
            safe_intervals.append(TimeReservation(TimeInterval(current_time, horizon), robot_id=-1))
        return safe_intervals


