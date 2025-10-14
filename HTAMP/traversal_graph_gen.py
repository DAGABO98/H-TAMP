from collections.abc import Set
from typing import List, Tuple
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

from HTAMP.grid_world import Coordinate

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
    orientation_vec: Tuple[float, float]
    connections: list["TraversalNode"]

@dataclass
class TraversalEdge:
    from_node: str
    to_node: str
    action: str

@dataclass
class IntersectionSubgraph:
    upper_nodes: list[TraversalNode]
    lower_nodes: list[TraversalNode]
    left_nodes: list[TraversalNode]
    right_nodes: list[TraversalNode]
    edges: list[TraversalEdge]

@dataclass
class TraversalGraph:
    nodes: list[TraversalNode]
    edges: list[TraversalEdge]

class TraversalGraphGenerator:
    def __init__(self, occupancy_map_path: str, config_path: str, meters_per_pixel: float = 0.036, factor: int = 1,
                 num_lanes_per_corridor: int = 3, num_lanes_per_drive_through: int = 1, num_lanes_per_doorway: int = 2,
                 doorway_lane_threshold: float = 30.0):
        self.occupancy_map_path = occupancy_map_path
        self.config_path = config_path
        self.meters_per_cell = meters_per_pixel * factor
        self.num_lanes_per_corridor = num_lanes_per_corridor
        self.num_lanes_per_drive_through = num_lanes_per_drive_through
        self.num_lanes_per_doorway = num_lanes_per_doorway
        self.doorway_lane_threshold = doorway_lane_threshold
        self.occupancy_map = self._load_map(occupancy_map_path)
        self.config = self._load_config(config_path)
        self.origin_x = self.config.get('origin', [0, 0])[0]
        self.origin_y = self.config.get('origin', [0, 0])[1]
        self.resolution = self.config.get('resolution', meters_per_pixel * factor)
        self.corridors = self._extract_corridors_from_config()
        self.drive_throughs = self._extract_drive_throughs_from_config()
        self.doorways = self._extract_doorways_from_config()
        self.corridor_intersections = self._generate_corridor_intersection_traversal_subgraph()

        print(f"Coarse map shape: {self.occupancy_map.shape}, meters per cell: {self.meters_per_cell:.4f}")

    def _load_map(self, path: str) -> np.ndarray:
        try:
            return np.load(path)
        except Exception as e:
            print(f"Error loading map: {e}")
            return np.zeros((0, 0), dtype=np.int8)
    
    def _load_config(self, path: str) -> dict:
        try:
            with open(path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}

    def _extract_corridors_from_config(self) -> List[Corridor]:
        corridors = []
        if 'corridors' in self.config:
            for corridor in self.config['corridors']:
                corridor_lanes = []
                corridor_direction = corridor.get('direction', None)
                if corridor_direction is None:
                    print("Warning: Corridor without direction found in config.")
                    continue
                elif corridor_direction == 'horizontal':
                    corridor_width = self.meters_per_cell * abs(corridor['width_end'][1] - corridor['width_start'][1])
                    lane_separation = corridor_width / (self.num_lanes_per_corridor + 1)
                    for lane_idx in range(1, self.num_lanes_per_corridor + 1):
                        lane_y = (self.meters_per_cell * corridor['width_start'][1]) + (lane_idx * lane_separation)
                        lane_struct = Lane(start_point=Coordinate(x=corridor['length_start'][0]*self.meters_per_cell, y=lane_y),
                                            end_point=Coordinate(x=corridor['length_end'][0]*self.meters_per_cell, y=lane_y))
                        corridor_lanes.append(lane_struct)
                elif corridor_direction == 'vertical':
                    corridor_width = self.meters_per_cell * abs(corridor['width_end'][0] - corridor['width_start'][0])
                    lane_separation = corridor_width / (self.num_lanes_per_corridor + 1)
                    for lane_idx in range(1, self.num_lanes_per_corridor + 1):
                        lane_x = (self.meters_per_cell * corridor['width_start'][0]) + (lane_idx * lane_separation)
                        lane_struct = Lane(start_point=Coordinate(x=lane_x, y=corridor['length_start'][1]*self.meters_per_cell),
                                            end_point=Coordinate(x=lane_x, y=corridor['length_end'][1]*self.meters_per_cell))
                        corridor_lanes.append(lane_struct)
                else:
                    print(f"Warning: Unknown corridor direction '{corridor_direction}' found in config.")
                    continue
                width_start_x = (corridor['width_start'][0])
                width_start_y = (corridor['width_start'][1])
                width_end_x = (corridor['width_end'][0])
                width_end_y = (corridor['width_end'][1])
                corridor_intersections = corridor.get('intersections', [])

                corridor_struct = Corridor(corridor_id=corridor.get('id', 'unknown'),
                    direction=corridor.get('direction', 'unknown'),
                    width_start=Coordinate(x=(width_start_x*self.meters_per_cell), y=(width_start_y*self.meters_per_cell)),
                    width_end=Coordinate(x=(width_end_x*self.meters_per_cell), y=(width_end_y*self.meters_per_cell)),
                    lanes=corridor_lanes,
                    intersections=corridor_intersections)

                corridors.append(corridor_struct)
        return corridors
    
    def _extract_drive_throughs_from_config(self) -> List[DriveThrough]:
        drive_throughs = []
        if 'drive_through_spaces' in self.config:
            for dt in self.config['drive_through_spaces']:
                dt_lanes = []
                dt_direction = dt.get('direction', None)
                if dt_direction is None:
                    print("Warning: Drive-through without direction found in config.")
                    continue
                elif dt_direction == 'horizontal':
                    entry_width = self.meters_per_cell * abs(dt['entry_end_point'][1] - dt['entry_start_point'][1])
                    exit_width = self.meters_per_cell * abs(dt['exit_end_point'][1] - dt['exit_start_point'][1])
                    entry_lane_separation = entry_width / (self.num_lanes_per_drive_through + 1)
                    exit_lane_separation = exit_width / (self.num_lanes_per_drive_through + 1)
                    for lane_idx in range(1, self.num_lanes_per_drive_through + 1):
                        entry_lane_y = (self.meters_per_cell * dt['entry_start_point'][1]) + (lane_idx * entry_lane_separation)
                        exit_lane_y = (self.meters_per_cell * dt['exit_start_point'][1]) + (lane_idx * exit_lane_separation)
                        lane_struct = Lane(start_point=Coordinate(x=dt['entry_start_point'][0]*self.meters_per_cell, y=entry_lane_y),
                                            end_point=Coordinate(x=dt['exit_end_point'][0]*self.meters_per_cell, y=exit_lane_y))
                        dt_lanes.append(lane_struct)
                elif dt_direction == 'vertical':
                    entry_width = self.meters_per_cell * abs(dt['entry_end_point'][0] - dt['entry_start_point'][0])
                    exit_width = self.meters_per_cell * abs(dt['exit_end_point'][0] - dt['exit_start_point'][0])
                    entry_lane_separation = entry_width / (self.num_lanes_per_drive_through + 1)
                    exit_lane_separation = exit_width / (self.num_lanes_per_drive_through + 1)
                    for lane_idx in range(1, self.num_lanes_per_drive_through + 1):
                        entry_lane_x = (self.meters_per_cell * dt['entry_start_point'][0]) + (lane_idx * entry_lane_separation)
                        exit_lane_x = (self.meters_per_cell * dt['exit_start_point'][0]) + (lane_idx * exit_lane_separation)
                        lane_struct = Lane(start_point=Coordinate(x=entry_lane_x, y=dt['entry_start_point'][1]*self.meters_per_cell),
                                            end_point=Coordinate(x=exit_lane_x, y=dt['exit_end_point'][1]*self.meters_per_cell))
                        dt_lanes.append(lane_struct)
                else:
                    print(f"Warning: Unknown drive-through direction '{dt_direction}' found in config.")
                    continue

                entry_start = Coordinate(x=(dt['entry_start_point'][0]*self.meters_per_cell), 
                                         y=(dt['entry_start_point'][1]*self.meters_per_cell))
                entry_end = Coordinate(x=(dt['entry_end_point'][0]*self.meters_per_cell), 
                                        y=(dt['entry_end_point'][1]*self.meters_per_cell))
                exit_start = Coordinate(x=(dt['exit_start_point'][0]*self.meters_per_cell), 
                                        y=(dt['exit_start_point'][1]*self.meters_per_cell))
                exit_end = Coordinate(x=(dt['exit_end_point'][0]*self.meters_per_cell), 
                                        y=(dt['exit_end_point'][1]*self.meters_per_cell))
                
                current_drive_through = DriveThrough(entry_start=entry_start,
                                                     entry_end=entry_end,
                                                     exit_start=exit_start,
                                                     exit_end=exit_end,
                                                     entry_corridor_id=dt.get('entry_corridor_id', 'unknown'),
                                                     exit_corridor_id=dt.get('exit_corridor_id', 'unknown'),
                                                     lanes=dt_lanes)

                drive_throughs.append(current_drive_through)
        return drive_throughs

    def _extract_doorways_from_config(self) -> List[Doorway]:
        doorways = []
        if 'doorways' in self.config:
            for dw in self.config['doorways']:
                doorway_lanes = []
                doorway_corridor_id = dw.get('corridor_id', None)
                doorway_direction = None

                if "H" in doorway_corridor_id:
                    doorway_direction = 'horizontal'
                elif "V" in doorway_corridor_id:
                    doorway_direction = 'vertical'
                else:
                    print(f"Warning: Unknown doorway corridor ID '{doorway_corridor_id}' found in config.")
                    continue

                if doorway_direction is None:
                    print("Warning: Doorway without direction found in config.")
                    continue
                elif doorway_direction == 'horizontal':
                    doorway_width = self.meters_per_cell * abs(dw['end_point'][0] - dw['start_point'][0])
                    if doorway_width < self.doorway_lane_threshold * self.meters_per_cell:
                        lane_separation = doorway_width / (self.num_lanes_per_doorway)
                        for lane_idx in range(1, self.num_lanes_per_doorway):
                            lane_x = (self.meters_per_cell * dw['start_point'][0]) + (lane_idx * lane_separation)
                            lane_struct = Lane(start_point=Coordinate(x=lane_x, y=dw['start_point'][1]*self.meters_per_cell),
                                                end_point=Coordinate(x=lane_x, y=dw['end_point'][1]*self.meters_per_cell))
                            doorway_lanes.append(lane_struct)
                    else:
                        lane_separation = doorway_width / (self.num_lanes_per_doorway + 1)
                        for lane_idx in range(1, self.num_lanes_per_doorway + 1):
                            lane_x = (self.meters_per_cell * dw['start_point'][0]) + (lane_idx * lane_separation)
                            lane_struct = Lane(start_point=Coordinate(x=lane_x, y=dw['start_point'][1]*self.meters_per_cell),
                                                end_point=Coordinate(x=lane_x, y=dw['end_point'][1]*self.meters_per_cell))
                            doorway_lanes.append(lane_struct)
                elif doorway_direction == 'vertical':
                    doorway_width = self.meters_per_cell * abs(dw['end_point'][1] - dw['start_point'][1])
                    if doorway_width < self.doorway_lane_threshold * self.meters_per_cell:
                        lane_separation = doorway_width / (self.num_lanes_per_doorway)
                        for lane_idx in range(1, self.num_lanes_per_doorway):
                            lane_y = (self.meters_per_cell * dw['start_point'][1]) + (lane_idx * lane_separation)
                            lane_struct = Lane(start_point=Coordinate(x=dw['start_point'][0]*self.meters_per_cell, y=lane_y),
                                                end_point=Coordinate(x=dw['end_point'][0]*self.meters_per_cell, y=lane_y))
                            doorway_lanes.append(lane_struct)
                    else:
                        lane_separation = doorway_width / (self.num_lanes_per_doorway + 1)
                        for lane_idx in range(1, self.num_lanes_per_doorway + 1):
                            lane_y = (self.meters_per_cell * dw['start_point'][1]) + (lane_idx * lane_separation)
                            lane_struct = Lane(start_point=Coordinate(x=dw['start_point'][0]*self.meters_per_cell, y=lane_y),
                                                end_point=Coordinate(x=dw['end_point'][0]*self.meters_per_cell, y=lane_y))
                            doorway_lanes.append(lane_struct)
                else:
                    print(f"Warning: Unknown doorway direction '{doorway_direction}' found in config.")
                    continue

                start = Coordinate(x=(dw['start_point'][0]*self.meters_per_cell), 
                                    y=(dw['start_point'][1]*self.meters_per_cell))
                end = Coordinate(x=(dw['end_point'][0]*self.meters_per_cell), 
                                    y=(dw['end_point'][1]*self.meters_per_cell))
                current_doorway = Doorway(start=start,
                                          end=end,
                                          lanes=doorway_lanes,
                                          corridor_id=doorway_corridor_id)
                doorways.append(current_doorway)
        return doorways
    
    def _extract_corridor_intersection_nodes(self, 
                                             corridor: Corridor,
                                             seen_corridors: Set[int]) -> Tuple[List[TraversalNode], List[TraversalNode], List[TraversalNode], List[TraversalNode]]:
        upper_nodes = []
        lower_nodes = []
        left_nodes = []
        right_nodes = []
        corridor_dict = {corridor.corridor_id: corridor for corridor in self.corridors}

        for intersection_id in corridor.intersections:
            if intersection_id not in corridor_dict:
                print(f"Warning: Intersection ID '{intersection_id}' not found among corridors.")
                continue

            intersection_corridor = corridor_dict[intersection_id]

            if intersection_id in seen_corridors:
                continue

            if corridor.direction == "horizontal":
                vertical_start_y = corridor.width_start.y
                vertical_end_y = corridor.width_end.y

                for i, lane in enumerate(intersection_corridor.lanes):
                    lane_x = lane.start_point.x
                    start_node_location = Coordinate(x=lane_x, y=vertical_start_y)
                    end_node_location = Coordinate(x=lane_x, y=vertical_end_y)
                    if i == 0:
                        orientation_vec = (0.0, 1.0)  # Facing down
                    elif i == len(intersection_corridor.lanes) - 1:
                        orientation_vec = (0.0, -1.0)  # Facing up
                    else:
                        orientation_vec = None
                    
                    if orientation_vec is not None:
                        upper_node = TraversalNode(label=f"{intersection_id}_upper_{i}",
                                                    position=start_node_location,
                                                    orientation_vec=orientation_vec,
                                                    connections=[])
                        lower_node = TraversalNode(label=f"{intersection_id}_lower_{i}",
                                                    position=end_node_location,
                                                    orientation_vec=orientation_vec,
                                                    connections=[])
                        upper_nodes.append(upper_node)
                        lower_nodes.append(lower_node)
                    else:
                        possible_orientations = [(0.0, 1.0), (0.0, -1.0)]
                        for possible_orientation in possible_orientations:
                            upper_node = TraversalNode(label=f"{intersection_id}_upper_{i}",
                                                        position=start_node_location,
                                                        orientation_vec=possible_orientation,
                                                        connections=[])
                            lower_node = TraversalNode(label=f"{intersection_id}_lower_{i}",
                                                        position=end_node_location,
                                                        orientation_vec=possible_orientation,
                                                        connections=[])
                            upper_nodes.append(upper_node)
                            lower_nodes.append(lower_node)

                horizontal_start_x = intersection_corridor.width_start.x
                horizontal_end_x = intersection_corridor.width_end.x

                for j, lane in enumerate(corridor.lanes):
                    lane_y = lane.start_point.y
                    start_node_location = Coordinate(x=horizontal_start_x, y=lane_y)
                    end_node_location = Coordinate(x=horizontal_end_x, y=lane_y)
                    if j == 0:
                        orientation_vec = (-1.0, 0.0)  # Facing  left
                    elif j == len(corridor.lanes) - 1:
                        orientation_vec = (1.0, 0.0)  # Facing right
                    else:
                        orientation_vec = None
                    
                    if orientation_vec is not None:
                        left_node = TraversalNode(label=f"{corridor.corridor_id}_left_{j}",
                                                    position=start_node_location,
                                                    orientation_vec=orientation_vec,
                                                    connections=[])
                        right_node = TraversalNode(label=f"{corridor.corridor_id}_right_{j}",
                                                    position=end_node_location,
                                                    orientation_vec=orientation_vec,
                                                    connections=[])
                        left_nodes.append(left_node)
                        right_nodes.append(right_node)
                    else:
                        possible_orientations = [(1.0, 0.0), (-1.0, 0.0)]
                        for possible_orientation in possible_orientations:
                            left_node = TraversalNode(label=f"{corridor.corridor_id}_left_{j}",
                                                        position=start_node_location,
                                                        orientation_vec=possible_orientation,
                                                        connections=[])
                            right_node = TraversalNode(label=f"{corridor.corridor_id}_right_{j}",
                                                        position=end_node_location,
                                                        orientation_vec=possible_orientation,
                                                        connections=[])
                            left_nodes.append(left_node)
                            right_nodes.append(right_node)

            elif corridor.direction == "vertical":
                horizontal_start_x = corridor.width_start.x
                horizontal_end_x = corridor.width_end.x

                for i, lane in enumerate(intersection_corridor.lanes):
                    lane_y = lane.start_point.y
                    start_node_location = Coordinate(x=horizontal_start_x, y=lane_y)
                    end_node_location = Coordinate(x=horizontal_end_x, y=lane_y)
                    if i == 0:
                        orientation_vec = (-1.0, 0.0)  # Facing left
                    elif i == len(intersection_corridor.lanes) - 1:
                        orientation_vec = (1.0, 0.0)  # Facing right
                    else:
                        orientation_vec = None
                    
                    if orientation_vec is not None:
                        left_node = TraversalNode(label=f"{intersection_id}_left_{i}",
                                                    position=start_node_location,
                                                    orientation_vec=orientation_vec,
                                                    connections=[])
                        right_node = TraversalNode(label=f"{intersection_id}_right_{i}",
                                                    position=end_node_location,
                                                    orientation_vec=orientation_vec,
                                                    connections=[])
                        left_nodes.append(left_node)
                        right_nodes.append(right_node)
                    else:
                        possible_orientations = [(1.0, 0.0), (-1.0, 0.0)]
                        for possible_orientation in possible_orientations:
                            left_node = TraversalNode(label=f"{intersection_id}_left_{i}",
                                                        position=start_node_location,
                                                        orientation_vec=possible_orientation,
                                                        connections=[])
                            right_node = TraversalNode(label=f"{intersection_id}_right_{i}",
                                                        position=end_node_location,
                                                        orientation_vec=possible_orientation,
                                                        connections=[])
                            left_nodes.append(left_node)
                            right_nodes.append(right_node)

                vertical_start_y = intersection_corridor.width_start.y
                vertical_end_y = intersection_corridor.width_end.y

                for j, lane in enumerate(corridor.lanes):
                    lane_x = lane.start_point.x
                    start_node_location = Coordinate(x=lane_x, y=vertical_start_y)
                    end_node_location = Coordinate(x=lane_x, y=vertical_end_y)
                    if j == 0:
                        orientation_vec = (0.0, -1.0)  # Facing up
                    elif j == len(corridor.lanes) - 1:
                        orientation_vec = (0.0, 1.0)  # Facing down
                    else:
                        orientation_vec = None

                    if orientation_vec is not None:
                        upper_node = TraversalNode(label=f"{corridor.corridor_id}_upper_{j}",
                                                    position=start_node_location,
                                                    orientation_vec=orientation_vec,
                                                    connections=[])
                        lower_node = TraversalNode(label=f"{corridor.corridor_id}_lower_{j}",
                                                    position=end_node_location,
                                                    orientation_vec=orientation_vec,
                                                    connections=[])
                        upper_nodes.append(upper_node)
                        lower_nodes.append(lower_node)
                    else:
                        possible_orientations = [(0.0, 1.0), (0.0, -1.0)]
                        for possible_orientation in possible_orientations:
                            upper_node = TraversalNode(label=f"{corridor.corridor_id}_upper_{j}",
                                                        position=start_node_location,
                                                        orientation_vec=possible_orientation,
                                                        connections=[])
                            lower_node = TraversalNode(label=f"{corridor.corridor_id}_lower_{j}",
                                                        position=end_node_location,
                                                        orientation_vec=possible_orientation,
                                                        connections=[])
                            upper_nodes.append(upper_node)
                            lower_nodes.append(lower_node)
            else:
                print(f"Warning: Unknown corridor direction '{corridor.direction}' found in config.")
                continue
        return upper_nodes, lower_nodes, left_nodes, right_nodes
    
    def _create_intersection_connections_for_nodes(self, 
                                                   reference_nodes: List[TraversalNode], 
                                                   ref_orientation: Tuple[float, float],
                                                   opposite_nodes: List[TraversalNode],
                                                   left_nodes: List[TraversalNode],
                                                   left_orientation: Tuple[float, float],
                                                   right_nodes: List[TraversalNode],
                                                   right_orientation: Tuple[float, float],
                                                   invert: bool) -> List[TraversalEdge]:
        edges = []
        if invert:
            ref_index = len(reference_nodes) - 1
        else:
            ref_index = 0

        for i, ref_node in enumerate(reference_nodes):
            if ref_node.orientation_vec == ref_orientation:
                if i == ref_index:
                    opp_node = opposite_nodes[i]
                    ref_node.connections.append(opp_node)
                    edge = TraversalEdge(from_node=ref_node.label, to_node=opp_node.label, action="go_straight")
                    edges.append(edge)

                    for left_node in left_nodes:
                        if left_node.orientation_vec == left_orientation:
                            ref_node.connections.append(left_node)
                            edge = TraversalEdge(from_node=ref_node.label, to_node=left_node.label, action="turn_left")
                            edges.append(edge)
                else:
                    for opp_node in opposite_nodes:
                        if opp_node.orientation_vec == ref_orientation and opp_node.position.x == ref_node.position.x:
                            ref_node.connections.append(opp_node)
                            edge = TraversalEdge(from_node=ref_node.label, to_node=opp_node.label, action="go_straight")
                            edges.append(edge)
                    
                    for right_node in right_nodes:
                        if right_node.orientation_vec == right_orientation:
                            ref_node.connections.append(right_node)
                            edge = TraversalEdge(from_node=ref_node.label, to_node=right_node.label, action="turn_right")
                            edges.append(edge)
        
        return edges

    def _generate_corridor_intersection_traversal_subgraph(self) -> IntersectionSubgraph:
        edges = []
        seen_corridors = set()
        for corridor in self.corridors:
            upper_nodes, lower_nodes, left_nodes, right_nodes  = self._extract_corridor_intersection_nodes(corridor=corridor, 
                                                                                                           seen_corridors=seen_corridors)
            seen_corridors.add(corridor.corridor_id)

            upper_nodes.sort(key=lambda node: node.position.x)
            lower_nodes.sort(key=lambda node: node.position.x)
            left_nodes.sort(key=lambda node: node.position.y)
            right_nodes.sort(key=lambda node: node.position.y)

            for i, upper_node in enumerate(upper_nodes):
                if upper_node.orientation_vec == (0.0, 1.0):  # Facing down
                    if i == 0:
                        lower_node = lower_nodes[i]
                        upper_node.connections.append(lower_node)
                        edge = TraversalEdge(from_node=upper_node, to_node=lower_node, action="go_straight")
                        edges.append(edge)

                        for left_node in left_nodes:
                            if left_node.orientation_vec == (-1.0, 0.0):
                                upper_node.connections.append(left_node)
                                edge = TraversalEdge(from_node=upper_node, to_node=left_node, action="turn_left")
                                edges.append(edge)
                    else:
                        for lower_node in lower_nodes:
                            if lower_node.orientation_vec == (0.0, 1.0) and lower_node.position.x == upper_node.position.x:
                                upper_node.connections.append(lower_node)
                                edge = TraversalEdge(from_node=upper_node, to_node=lower_node, action="go_straight")
                                edges.append(edge)
                        
                        for right_node in right_nodes:
                            if right_node.orientation_vec == (1.0, 0.0):
                                upper_node.connections.append(right_node)
                                edge = TraversalEdge(from_node=upper_node, to_node=right_node, action="turn_right")
                                edges.append(edge)
            
            for j, lower_node in enumerate(lower_nodes):
                if lower_node.orientation_vec == (0.0, -1.0):  # Facing up
                    if j == len(lower_nodes) - 1:
                        upper_node = upper_nodes[j]
                        lower_node.connections.append(upper_node)
                        edge = TraversalEdge(from_node=lower_node, to_node=upper_node, action="go_straight")
                        edges.append(edge)

                        for right_node in right_nodes:
                            if right_node.orientation_vec == (1.0, 0.0):
                                lower_node.connections.append(right_node)
                                edge = TraversalEdge(from_node=lower_node, to_node=right_node, action="turn_right")
                                edges.append(edge)
                    else:
                        for upper_node in upper_nodes:
                            if upper_node.orientation_vec == (0.0, -1.0) and upper_node.position.x == lower_node.position.x:
                                lower_node.connections.append(upper_node)
                                edge = TraversalEdge(from_node=lower_node, to_node=upper_node, action="go_straight")
                                edges.append(edge)
                        
                        for left_node in left_nodes:
                            if left_node.orientation_vec == (-1.0, 0.0):
                                lower_node.connections.append(left_node)
                                edge = TraversalEdge(from_node=lower_node, to_node=left_node, action="turn_left")
                                edges.append(edge)

            for m, right_node in enumerate(right_nodes):
                if right_node.orientation_vec == (-1.0, 0.0):  # Facing left
                    if m == 0:
                        left_node = left_nodes[m]
                        right_node.connections.append(left_node)
                        edge = TraversalEdge(from_node=right_node, to_node=left_node, action="go_straight")
                        edges.append(edge)

                        for upper_node in upper_nodes:
                            if upper_node.orientation_vec == (0.0, -1.0):
                                right_node.connections.append(upper_node)
                                edge = TraversalEdge(from_node=right_node, to_node=upper_node, action="turn_right")
                                edges.append(edge)
                    else:
                        for left_node in left_nodes:
                            if left_node.orientation_vec == (-1.0, 0.0) and right_node.position.y == left_node.position.y:
                                right_node.connections.append(left_node)
                                edge = TraversalEdge(from_node=right_node, to_node=left_node, action="go_straight")
                                edges.append(edge)

                        for lower_node in lower_nodes:
                            if lower_node.orientation_vec == (0.0, 1.0):
                                right_node.connections.append(lower_node)
                                edge = TraversalEdge(from_node=right_node, to_node=lower_node, action="turn_left")
                                edges.append(edge)


            for m, left_node in enumerate(left_nodes):
                if left_node.orientation_vec == (1.0, 0.0):  # Facing right
                    if m == len(left_nodes) - 1:
                        right_node = right_nodes[m]
                        left_node.connections.append(right_node)
                        edge = TraversalEdge(from_node=left_node, to_node=right_node, action="go_straight")
                        edges.append(edge)

                        for lower_node in lower_nodes:
                            if lower_node.orientation_vec == (0.0, 1.0):
                                left_node.connections.append(lower_node)
                                edge = TraversalEdge(from_node=left_node, to_node=lower_node, action="turn_right")
                                edges.append(edge)
                    else:
                        for right_node in right_nodes:
                            if right_node.orientation_vec == (1.0, 0.0) and left_node.position.y == right_node.position.y:
                                left_node.connections.append(right_node)
                                edge = TraversalEdge(from_node=left_node, to_node=right_node, action="go_straight")
                                edges.append(edge)

                        for upper_node in upper_nodes:
                            if upper_node.orientation_vec == (0.0, -1.0):
                                left_node.connections.append(upper_node)
                                edge = TraversalEdge(from_node=left_node, to_node=upper_node, action="turn_left")
                                edges.append(edge)

        return IntersectionSubgraph(upper_nodes=upper_nodes, lower_nodes=lower_nodes, left_nodes=left_nodes, right_nodes=right_nodes, edges=edges)

    def plot_extracted_structs(self):
        rows, cols = self.occupancy_map.shape
        xmin, xmax = self.origin_x, self.origin_x + cols * self.resolution
        ymin, ymax = self.origin_y, self.origin_y + rows * self.resolution

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(
            self.occupancy_map,
            cmap="gray_r",
            origin="upper",              # flip so (0,0) is top-left
            extent=[xmin, xmax, ymax, ymin],  # still in meters
            aspect="equal"
        )

        for corridor in self.corridors:
            for lane in corridor.lanes:
                ax.plot([lane.start_point.x, lane.end_point.x],
                        [lane.start_point.y, lane.end_point.y],
                        color='blue', linewidth=1)
                
        for dt in self.drive_throughs:
            for lane in dt.lanes:
                ax.plot([lane.start_point.x, lane.end_point.x],
                        [lane.start_point.y, lane.end_point.y],
                        color='green', linewidth=1)
                
        for dw in self.doorways:
            for lane in dw.lanes:
                ax.scatter([lane.start_point.x, lane.end_point.x],
                        [lane.start_point.y, lane.end_point.y],
                        color='red', s=1)
                
        ax.set_title("Extracted Structs Overlay")
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        plt.savefig("results/extracted_structs.png")
        plt.show()
    

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="maps/FA3/FA3_lanes.yaml", help="Path to the configuration file")
    parser.add_argument("--occupancy_map_path", type=str, default="maps/FA3/occupancy_map.npy", help="Path to the input occupancy map")
    parser.add_argument("--factor", type=int, default=1, help="Downsampling factor")
    parser.add_argument("--meters_per_pixel", type=float, default=0.036, help="Meters per pixel in the original image")
    args = parser.parse_args()

    tg_generator = TraversalGraphGenerator(occupancy_map_path=args.occupancy_map_path,
                                           config_path=args.config_path,
                                           meters_per_pixel=args.meters_per_pixel,
                                           factor=args.factor)
    
    tg_generator.plot_extracted_structs()
