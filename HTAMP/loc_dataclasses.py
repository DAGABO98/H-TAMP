from dataclasses import dataclass
from typing import Set
import numpy as np

@dataclass(frozen=True, slots=True)
class Coordinate:
    x: float
    y: float
    tol: float = 1e-3

    def __post_init__(self):
        if self.tol <= 0:
            raise ValueError("tol must be > 0")

    def _key(self):
        # snap-to-grid; points in the same cell are "equal"
        return (round(self.x / self.tol), round(self.y / self.tol), self.tol)

    def __eq__(self, other):
        if not isinstance(other, Coordinate):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self):
        return hash(self._key())

@dataclass(frozen=True)
class GridIndex:
    index_x: int
    index_y: int

@dataclass(frozen=True)
class Cell: 
    lower_x: float
    lower_y: float
    upper_x: float
    upper_y: float

@dataclass(frozen=True)
class PosChange:
    dev_x: float
    dev_y: float

@dataclass(frozen=True)
class BoundingIndices:
    lower_x: int
    lower_y: int
    upper_x: int
    upper_y: int

@dataclass(frozen=True, slots=True)
class TimeInterval:
    start: float
    end: float
    tol: float = 1e-3

    def __post_init__(self):
        if self.tol <= 0:
            raise ValueError("tol must be > 0")

    def _key(self):
        # snap-to-grid; points in the same cell are "equal"
        return (round(self.start / self.tol), round(self.end / self.tol), self.tol)

    def __eq__(self, other):
        if not isinstance(other, TimeInterval):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self):
        return hash(self._key())

@dataclass
class RobotOccupancy:
    occupied_cells: Set[GridIndex]
    robot_center: Coordinate

@dataclass
class MotionReservation:
    time_interval: TimeInterval
    robot_occupancy: RobotOccupancy

@dataclass
class OrientationVector:
    x: float
    y: float

    def __repr__(self):
        if self.x == 0.0 and self.y == 1.0:
            return "Down"
        elif self.x == 0.0 and self.y == -1.0:
            return "Up"
        elif self.x == 1.0 and self.y == 0.0:
            return "Right"
        elif self.x == -1.0 and self.y == 0.0:
            return "Left"
        else:
            return f"({self.x}, {self.y})"
    
    def __eq__(self, other):
        if not isinstance(other, OrientationVector):
            return NotImplemented
        return np.isclose(self.x, other.x) and np.isclose(self.y, other.y)