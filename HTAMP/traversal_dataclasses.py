from dataclasses import dataclass
from typing import List

from HTAMP.loc_dataclasses import Coordinate, OrientationVector
from HTAMP.geometry_helpers import CurvedConnector

@dataclass
class Lane:
    start_point: Coordinate
    end_point: Coordinate

@dataclass
class Corridor:
    corridor_id: str
    direction: str
    width_start: Coordinate
    width_end: Coordinate
    lanes: List[Lane]
    intersections: List[str]

@dataclass
class DriveThrough:
    entry_start: Coordinate
    entry_end: Coordinate
    exit_start: Coordinate
    exit_end: Coordinate
    entry_corridor_id: str
    exit_corridor_id: str
    lanes: List[Lane]

@dataclass
class Doorway:
    start: Coordinate
    end: Coordinate
    lanes: List[Lane]
    corridor_id: str

@dataclass
class TraversalNode:
    label: str
    position: Coordinate
    orientation_vec: OrientationVector
    connections: list["TraversalNode"]

@dataclass
class TraversalEdge:
    from_node: str
    to_node: str
    action: str
    edge_connector: CurvedConnector

@dataclass
class IntersectionSubgraph:
    upper_nodes: list[TraversalNode]
    lower_nodes: list[TraversalNode]
    left_nodes: list[TraversalNode]
    right_nodes: list[TraversalNode]
    edges: list[TraversalEdge]

@dataclass
class DoorwaySubgraph:
    room_nodes: list[TraversalNode]
    doorway_nodes: list[TraversalNode]
    left_nodes: list[TraversalNode]
    right_nodes: list[TraversalNode]
    edges: list[TraversalEdge]

@dataclass
class EndPointSubgraph:
    corridor_nodes: list[TraversalNode]
    edges: list[TraversalEdge]

@dataclass
class DriveThroughSubgraph:
    entry_nodes: list[TraversalNode]
    left_entry_nodes: list[TraversalNode]
    right_entry_nodes: list[TraversalNode]
    exit_nodes: list[TraversalNode]
    left_exit_nodes: list[TraversalNode]
    right_exit_nodes: list[TraversalNode]
    edges: list[TraversalEdge]

@dataclass
class SwitchingPointSubgraph:
    left_nodes: list[TraversalNode]
    right_nodes: list[TraversalNode]
    edges: list[TraversalEdge]

@dataclass
class TraversalGraph:
    nodes: list[TraversalNode]
    edges: list[TraversalEdge]