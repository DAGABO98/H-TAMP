from collections.abc import Set
from typing import List, Tuple
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

from HTAMP.geometry_helpers import CurvedConnector
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
class TraversalGraph:
    nodes: list[TraversalNode]
    edges: list[TraversalEdge]

class TraversalGraphGenerator:
    def __init__(self, occupancy_map_path: str, config_path: str, meters_per_pixel: float = 0.036, factor: int = 1,
                 num_lanes_per_corridor: int = 3, num_lanes_per_drive_through: int = 1, num_lanes_per_doorway: int = 2,
                 doorway_lane_threshold: float = 30.0, tangent_scaling_factor: float = 1.0, num_samples: int = 10, threshold: float = 10.0):
        self.occupancy_map_path = occupancy_map_path
        self.config_path = config_path
        self.tangent_scaling_factor = tangent_scaling_factor
        self.num_samples = num_samples
        self.meters_per_cell = meters_per_pixel * factor
        self.threshold = threshold * self.meters_per_cell
        self.num_lanes_per_corridor = num_lanes_per_corridor
        self.num_lanes_per_drive_through = num_lanes_per_drive_through
        self.doorway_node_offset = 2 * doorway_lane_threshold * self.meters_per_cell  # meters
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
        self.corridor_intersection_subgraphs, self.corridor_intersection_subgraph_indices = self._generate_corridor_intersection_traversal_subgraph()
        self.doorway_subgraphs, self.doorway_subgraph_indices = self._generate_doorway_traversal_subgraph()

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
                                             intersection_corridor: Corridor) -> Tuple[List[TraversalNode], List[TraversalNode], List[TraversalNode], List[TraversalNode]]:
        upper_nodes = []
        lower_nodes = []
        left_nodes = []
        right_nodes = []

        if corridor.direction == "horizontal":
            vertical_start_y = corridor.width_start.y
            vertical_end_y = corridor.width_end.y

            for i, lane in enumerate(intersection_corridor.lanes):
                lane_x = lane.start_point.x
                if abs(lane.start_point.y - vertical_start_y) < self.threshold:
                    start_node_location = None
                else:
                    start_node_location = Coordinate(x=lane_x, y=vertical_start_y)

                if abs(lane.end_point.y - vertical_end_y) < self.threshold:
                    end_node_location = None
                else:
                    end_node_location = Coordinate(x=lane_x, y=vertical_end_y)

                if i == 0:
                    orientation_vec = (0.0, 1.0)  # Facing down
                elif i == len(intersection_corridor.lanes) - 1:
                    orientation_vec = (0.0, -1.0)  # Facing up
                else:
                    orientation_vec = None
                
                if orientation_vec is not None:
                    if start_node_location is not None:
                        upper_node = TraversalNode(label=f"{corridor.corridor_id}_{intersection_corridor.corridor_id}_upper_{i}",
                                                    position=start_node_location,
                                                    orientation_vec=orientation_vec,
                                                    connections=[])
                        upper_nodes.append(upper_node)
                    if end_node_location is not None:
                        lower_node = TraversalNode(label=f"{corridor.corridor_id}_{intersection_corridor.corridor_id}_lower_{i}",
                                                    position=end_node_location,
                                                    orientation_vec=orientation_vec,
                                                    connections=[])
                        lower_nodes.append(lower_node)
                else:
                    possible_orientations = [(0.0, 1.0), (0.0, -1.0)]
                    for possible_orientation in possible_orientations:
                        if start_node_location is not None:
                            upper_node = TraversalNode(label=f"{corridor.corridor_id}_{intersection_corridor.corridor_id}_upper_{i}",
                                                        position=start_node_location,
                                                        orientation_vec=possible_orientation,
                                                        connections=[])
                            upper_nodes.append(upper_node)
                        if end_node_location is not None:
                            lower_node = TraversalNode(label=f"{corridor.corridor_id}_{intersection_corridor.corridor_id}_lower_{i}",
                                                    position=end_node_location,
                                                    orientation_vec=possible_orientation,
                                                    connections=[])
                            lower_nodes.append(lower_node)

            horizontal_start_x = intersection_corridor.width_start.x
            horizontal_end_x = intersection_corridor.width_end.x

            for j, lane in enumerate(corridor.lanes):
                lane_y = lane.start_point.y

                if abs(lane.start_point.x - horizontal_start_x) < self.threshold:
                    start_node_location = None
                else:
                    start_node_location = Coordinate(x=horizontal_start_x, y=lane_y)

                if abs(lane.end_point.x - horizontal_end_x) < self.threshold:
                    end_node_location = None
                else:
                    end_node_location = Coordinate(x=horizontal_end_x, y=lane_y)

                if j == 0:
                    orientation_vec = (-1.0, 0.0)  # Facing  left
                elif j == len(corridor.lanes) - 1:
                    orientation_vec = (1.0, 0.0)  # Facing right
                else:
                    orientation_vec = None
                
                if orientation_vec is not None:
                    if start_node_location is not None:
                        left_node = TraversalNode(label=f"{corridor.corridor_id}_{intersection_corridor.corridor_id}_left_{j}",
                                                    position=start_node_location,
                                                    orientation_vec=orientation_vec,
                                                    connections=[])
                        left_nodes.append(left_node)
                    if end_node_location is not None:
                        right_node = TraversalNode(label=f"{corridor.corridor_id}_{intersection_corridor.corridor_id}_right_{j}",
                                                    position=end_node_location,
                                                    orientation_vec=orientation_vec,
                                                    connections=[])
                        right_nodes.append(right_node)
                else:
                    possible_orientations = [(-1.0, 0.0), (1.0, 0.0)]
                    for possible_orientation in possible_orientations:
                        if start_node_location is not None:
                            left_node = TraversalNode(label=f"{corridor.corridor_id}_{intersection_corridor.corridor_id}_left_{j}",
                                                        position=start_node_location,
                                                        orientation_vec=possible_orientation,
                                                        connections=[])
                            left_nodes.append(left_node)
                        if end_node_location is not None:
                            right_node = TraversalNode(label=f"{corridor.corridor_id}_{intersection_corridor.corridor_id}_right_{j}",
                                                    position=end_node_location,
                                                    orientation_vec=possible_orientation,
                                                    connections=[])
                            right_nodes.append(right_node)

        return upper_nodes, lower_nodes, left_nodes, right_nodes
    
    def _create_edge_between_nodes(self, 
                                   from_node: TraversalNode, 
                                   to_node: TraversalNode, 
                                   action: str) -> TraversalEdge:
        from_node.connections.append(to_node)
        edge_connector = CurvedConnector(origin=from_node.position, 
                                        destination=to_node.position, 
                                        vec_origin=from_node.orientation_vec,
                                        vec_destination=to_node.orientation_vec,
                                        tangent_scaling_factor=self.tangent_scaling_factor,
                                        num_samples=self.num_samples)
        edge = TraversalEdge(from_node=from_node.label, 
                            to_node=to_node.label, 
                            action=action, 
                            edge_connector=edge_connector)
        return edge
    
    def _create_intersection_connections_for_nodes(self, 
                                                   reference_nodes: List[TraversalNode], 
                                                   ref_orientation: Tuple[float, float],
                                                   opposite_nodes: List[TraversalNode],
                                                   right_nodes: List[TraversalNode],
                                                   right_orientation: Tuple[float, float],
                                                   left_nodes: List[TraversalNode],
                                                   left_orientation: Tuple[float, float],
                                                   horizontal: bool,
                                                   invert: bool) -> List[TraversalEdge]:
        edges = []
        if invert:
            ref_index = len(reference_nodes) - 1
        else:
            ref_index = 0

        for i, ref_node in enumerate(reference_nodes):
            if ref_node.orientation_vec == ref_orientation:
                if i == ref_index:
                    if len(opposite_nodes) > 0:
                        opp_node = opposite_nodes[i]
                        edge = self._create_edge_between_nodes(from_node=ref_node, 
                                                               to_node=opp_node, 
                                                               action="go_straight")
                        edges.append(edge)
                    
                    if len(right_nodes) > 0:
                        for right_node in right_nodes:
                            if right_node.orientation_vec == right_orientation:
                                edge = self._create_edge_between_nodes(from_node=ref_node, 
                                                                       to_node=right_node, 
                                                                       action="turn_right")
                                edges.append(edge)
                else:
                    if len(opposite_nodes) > 0:
                        for opp_node in opposite_nodes:
                            if horizontal:
                                directionality_flag = opp_node.position.x == ref_node.position.x
                            else:
                                directionality_flag = opp_node.position.y == ref_node.position.y

                            if opp_node.orientation_vec == ref_orientation and directionality_flag:
                                edge = self._create_edge_between_nodes(from_node=ref_node, 
                                                                       to_node=opp_node, 
                                                                       action="go_straight")
                                edges.append(edge)
                    
                    if len(left_nodes) > 0:
                        for left_node in left_nodes:
                            if left_node.orientation_vec == left_orientation:
                                edge = self._create_edge_between_nodes(from_node=ref_node,
                                                                       to_node=left_node,
                                                                       action="turn_left")
                                edges.append(edge)
        return edges

    def _generate_corridor_intersection_traversal_subgraph(self) -> Tuple[List[IntersectionSubgraph], dict[str, List[int]]]:
        seen_corridors = set()
        subgraphs = []
        subgraph_indices = {}
        current_index = 0
        for corridor in self.corridors:
            corridor_dict = {corridor.corridor_id: corridor for corridor in self.corridors}

            for intersection_id in corridor.intersections:
                edges = []
                if intersection_id not in corridor_dict:
                    print(f"Warning: Intersection ID '{intersection_id}' not found among corridors.")
                    continue

                intersection_corridor = corridor_dict[intersection_id]

                if intersection_corridor.corridor_id in seen_corridors:
                    continue

                upper_nodes, lower_nodes, left_nodes, right_nodes  = self._extract_corridor_intersection_nodes(corridor=corridor, 
                                                                                                               intersection_corridor=intersection_corridor)
                
                seen_corridors.add(corridor.corridor_id)

                if len(upper_nodes) > 0:
                    upper_nodes.sort(key=lambda node: node.position.x)
                if len(lower_nodes) > 0:
                    lower_nodes.sort(key=lambda node: node.position.x)
                if len(left_nodes) > 0:
                    left_nodes.sort(key=lambda node: node.position.y)
                if len(right_nodes) > 0:
                    right_nodes.sort(key=lambda node: node.position.y)

                upper_edges = self._create_intersection_connections_for_nodes(reference_nodes=upper_nodes,
                                                                            ref_orientation=(0.0, 1.0),
                                                                            opposite_nodes=lower_nodes,
                                                                            right_nodes=left_nodes,
                                                                            right_orientation=(-1.0, 0.0),
                                                                            left_nodes=right_nodes,
                                                                            left_orientation=(1.0, 0.0),
                                                                            horizontal=True,
                                                                            invert=False)
                edges.extend(upper_edges)

                lower_edges = self._create_intersection_connections_for_nodes(reference_nodes=lower_nodes,
                                                                            ref_orientation=(0.0, -1.0),
                                                                            opposite_nodes=upper_nodes,
                                                                            right_nodes=right_nodes,
                                                                            right_orientation=(1.0, 0.0),
                                                                            left_nodes=left_nodes,
                                                                            left_orientation=(-1.0, 0.0),
                                                                            horizontal=True,
                                                                            invert=True)
                edges.extend(lower_edges)

                right_edges = self._create_intersection_connections_for_nodes(reference_nodes=right_nodes,
                                                                            ref_orientation=(-1.0, 0.0),
                                                                            opposite_nodes=left_nodes,
                                                                            right_nodes=upper_nodes,
                                                                            right_orientation=(0.0, -1.0),
                                                                            left_nodes=lower_nodes,
                                                                            left_orientation=(0.0, 1.0),
                                                                            horizontal=False,
                                                                            invert=False)
                edges.extend(right_edges)

                left_edges = self._create_intersection_connections_for_nodes(reference_nodes=left_nodes,
                                                                            ref_orientation=(1.0, 0.0),
                                                                            opposite_nodes=right_nodes,
                                                                            right_nodes=lower_nodes,
                                                                            right_orientation=(0.0, 1.0),
                                                                            left_nodes=upper_nodes,
                                                                            left_orientation=(0.0, -1.0),
                                                                            horizontal=False,
                                                                            invert=True)
                edges.extend(left_edges)
                current_subgraph = IntersectionSubgraph(upper_nodes=upper_nodes, 
                                                        lower_nodes=lower_nodes, 
                                                        left_nodes=left_nodes, 
                                                        right_nodes=right_nodes, 
                                                        edges=edges)

                subgraphs.append(current_subgraph)
                subgraph_indices.setdefault(corridor.corridor_id, []).append(current_index)
                subgraph_indices.setdefault(intersection_corridor.corridor_id, []).append(current_index)
                current_index += 1

        return subgraphs, subgraph_indices
    
    def _get_corridor_by_id(self, corridor_id: str) -> Corridor:
        for corridor in self.corridors:
            if corridor.corridor_id == corridor_id:
                return corridor
        return None

    def _extract_doorway_intersection_nodes(self, 
                                             doorway: Doorway,
                                             corridor: Corridor) -> Tuple[List[TraversalNode], List[TraversalNode], List[TraversalNode]]:
        room_nodes = []
        door_nodes = []
        left_nodes = []
        right_nodes = []

        if corridor.direction == "horizontal":
            doorway_start_x = doorway.start.x
            doorway_end_x = doorway.end.x
            for i, corridor_lane in enumerate(corridor.lanes):
                lane_y = corridor_lane.start_point.y
                left_node_location = Coordinate(x=doorway_start_x, y=lane_y)
                right_node_location = Coordinate(x=doorway_end_x, y=lane_y)
                if i == 0:
                    orientation_vec = (-1.0, 0.0)  # Facing  left
                elif i == len(corridor.lanes) - 1:
                    orientation_vec = (1.0, 0.0)  # Facing right
                else:
                    orientation_vec = None
                if orientation_vec is None:
                    possible_orientations = [(-1.0, 0.0), (1.0, 0.0)]
                    for possible_orientation in possible_orientations:
                        left_node = TraversalNode(label=f"{doorway.corridor_id}_left_{len(left_nodes)}",
                                                    position=left_node_location,
                                                    orientation_vec=possible_orientation,
                                                    connections=[])
                        right_node = TraversalNode(label=f"{doorway.corridor_id}_right_{len(right_nodes)}",
                                                    position=right_node_location,
                                                    orientation_vec=possible_orientation,
                                                    connections=[])
                        left_nodes.append(left_node)
                        right_nodes.append(right_node)
                else:
                    left_node = TraversalNode(label=f"{doorway.corridor_id}_left_{len(left_nodes)}",
                                                position=left_node_location,
                                                orientation_vec=orientation_vec,
                                                connections=[])
                    right_node = TraversalNode(label=f"{doorway.corridor_id}_right_{len(right_nodes)}",
                                                position=right_node_location,
                                                orientation_vec=orientation_vec,
                                                connections=[])
                    left_nodes.append(left_node)
                    right_nodes.append(right_node)

            if doorway.start.y <= corridor.width_start.y:
                if len(doorway.lanes) == 1:
                    possible_orientations = [(0.0, 1.0), (0.0, -1.0)]
                    for possible_orientation in possible_orientations:
                        lane = doorway.lanes[0]
                        lane_x = lane.start_point.x
                        lane_y = lane.start_point.y
                        room_node_location = Coordinate(x=lane_x, y=(lane_y-self.doorway_node_offset))
                        doorway_node_location = Coordinate(x=lane_x, y=lane_y)

                        room_node = TraversalNode(label=f"{doorway.corridor_id}_room_0",
                                                    position=room_node_location,
                                                    orientation_vec=possible_orientation,
                                                    connections=[])
                        room_nodes.append(room_node)

                        door_node = TraversalNode(label=f"{doorway.corridor_id}_door_0",
                                                    position=doorway_node_location,
                                                    orientation_vec=possible_orientation,
                                                    connections=[])
                        door_nodes.append(door_node)
                else:

                    for i, lane in enumerate(doorway.lanes):
                        lane_x = lane.start_point.x
                        lane_y = lane.start_point.y
                        room_node_location = Coordinate(x=lane_x, y=(lane_y-self.doorway_node_offset))
                        doorway_node_location = Coordinate(x=lane_x, y=lane_y)

                        if i == 0:
                            orientation_vec = (0.0, 1.0)  # Facing down
                        else:
                            orientation_vec = (0.0, -1.0)  # Facing up

                        room_node = TraversalNode(label=f"{doorway.corridor_id}_room_{i}",
                                                    position=room_node_location,
                                                    orientation_vec=orientation_vec,
                                                    connections=[])
                        room_nodes.append(room_node)

                        door_node = TraversalNode(label=f"{doorway.corridor_id}_door_{i}",
                                                    position=doorway_node_location,
                                                    orientation_vec=orientation_vec,
                                                    connections=[])
                        door_nodes.append(door_node)
            else:
                if len(doorway.lanes) == 1:
                    possible_orientations = [(0.0, 1.0), (0.0, -1.0)]
                    for possible_orientation in possible_orientations:
                        lane = doorway.lanes[0]
                        lane_x = lane.start_point.x
                        lane_y = lane.start_point.y
                        room_node_location = Coordinate(x=lane_x, y=(lane_y+self.doorway_node_offset))
                        doorway_node_location = Coordinate(x=lane_x, y=lane_y)

                        room_node = TraversalNode(label=f"{doorway.corridor_id}_room_0",
                                                    position=room_node_location,
                                                    orientation_vec=possible_orientation,
                                                    connections=[])
                        room_nodes.append(room_node)

                        door_node = TraversalNode(label=f"{doorway.corridor_id}_door_0",
                                                    position=doorway_node_location,
                                                    orientation_vec=possible_orientation,
                                                    connections=[])
                        door_nodes.append(door_node)
                else:
                    for i, lane in enumerate(doorway.lanes):
                        lane_x = lane.start_point.x
                        lane_y = lane.start_point.y
                        room_node_location = Coordinate(x=lane_x, y=(lane_y+self.doorway_node_offset))
                        doorway_node_location = Coordinate(x=lane_x, y=lane_y)

                        if i == 0:
                            orientation_vec = (0.0, 1.0)  # Facing down
                        else:
                            orientation_vec = (0.0, -1.0)  # Facing up

                        room_node = TraversalNode(label=f"{doorway.corridor_id}_room_{i}",
                                                    position=room_node_location,
                                                    orientation_vec=orientation_vec,
                                                    connections=[])
                        room_nodes.append(room_node)

                        door_node = TraversalNode(label=f"{doorway.corridor_id}_door_{i}",
                                                    position=doorway_node_location,
                                                    orientation_vec=orientation_vec,
                                                    connections=[])
                        door_nodes.append(door_node)
        else:
            doorway_start_y = doorway.start.y
            doorway_end_y = doorway.end.y
            for i, corridor_lane in enumerate(corridor.lanes):
                lane_x = corridor_lane.start_point.x
                right_node_location = Coordinate(x=lane_x, y=doorway_start_y)
                left_node_location = Coordinate(x=lane_x, y=doorway_end_y)
                if i == 0:
                    orientation_vec = (0.0, 1.0)  # Facing down
                elif i == len(corridor.lanes) - 1:
                    orientation_vec = (0.0, -1.0)  # Facing up
                else:
                    orientation_vec = None
                if orientation_vec is None:
                    possible_orientations = [(0.0, 1.0), (0.0, -1.0)]
                    for possible_orientation in possible_orientations:
                        left_node = TraversalNode(label=f"{doorway.corridor_id}_left_{len(left_nodes)}",
                                                    position=left_node_location,
                                                    orientation_vec=possible_orientation,
                                                    connections=[])
                        right_node = TraversalNode(label=f"{doorway.corridor_id}_right_{len(right_nodes)}",
                                                    position=right_node_location,
                                                    orientation_vec=possible_orientation,
                                                    connections=[])
                        left_nodes.append(left_node)
                        right_nodes.append(right_node)
                else:
                    left_node = TraversalNode(label=f"{doorway.corridor_id}_left_{len(left_nodes)}",
                                                position=left_node_location,
                                                orientation_vec=orientation_vec,
                                                connections=[])
                    right_node = TraversalNode(label=f"{doorway.corridor_id}_right_{len(right_nodes)}",
                                                position=right_node_location,
                                                orientation_vec=orientation_vec,
                                                connections=[])
                    left_nodes.append(left_node)
                    right_nodes.append(right_node)

            if doorway.start.x <= corridor.width_start.x:
                if len(doorway.lanes) == 1:
                    possible_orientations = [(-1.0, 0.0), (1.0, 0.0)]
                    for possible_orientation in possible_orientations:
                        lane = doorway.lanes[0]
                        lane_x = lane.start_point.x
                        lane_y = lane.start_point.y
                        room_node_location = Coordinate(x=(lane_x-self.doorway_node_offset), y=lane_y)
                        doorway_node_location = Coordinate(x=lane_x, y=lane_y)

                        room_node = TraversalNode(label=f"{doorway.corridor_id}_room_0",
                                                    position=room_node_location,
                                                    orientation_vec=possible_orientation,
                                                    connections=[])
                        room_nodes.append(room_node)

                        door_node = TraversalNode(label=f"{doorway.corridor_id}_door_0",
                                                    position=doorway_node_location,
                                                    orientation_vec=possible_orientation,
                                                    connections=[])
                        door_nodes.append(door_node)
                else:

                    for i, lane in enumerate(doorway.lanes):
                        lane_x = lane.start_point.x
                        lane_y = lane.start_point.y
                        room_node_location = Coordinate(x=(lane_x-self.doorway_node_offset), y=lane_y)
                        doorway_node_location = Coordinate(x=lane_x, y=lane_y)

                        if i == 0:
                            orientation_vec = (-1.0, 0.0) 
                        else:
                            orientation_vec = (1.0, 0.0)

                        room_node = TraversalNode(label=f"{doorway.corridor_id}_room_{i}",
                                                    position=room_node_location,
                                                    orientation_vec=orientation_vec,
                                                    connections=[])
                        room_nodes.append(room_node)

                        door_node = TraversalNode(label=f"{doorway.corridor_id}_door_{i}",
                                                    position=doorway_node_location,
                                                    orientation_vec=orientation_vec,
                                                    connections=[])
                        door_nodes.append(door_node)
            else:
                if len(doorway.lanes) == 1:
                    possible_orientations = [(-1.0, 0.0), (1.0, 0.0)]
                    for possible_orientation in possible_orientations:
                        lane = doorway.lanes[0]
                        lane_x = lane.start_point.x
                        lane_y = lane.start_point.y
                        room_node_location = Coordinate(x=(lane_x+self.doorway_node_offset), y=lane_y)
                        doorway_node_location = Coordinate(x=lane_x, y=lane_y)

                        room_node = TraversalNode(label=f"{doorway.corridor_id}_room_0",
                                                    position=room_node_location,
                                                    orientation_vec=possible_orientation,
                                                    connections=[])
                        room_nodes.append(room_node)

                        door_node = TraversalNode(label=f"{doorway.corridor_id}_door_0",
                                                    position=doorway_node_location,
                                                    orientation_vec=possible_orientation,
                                                    connections=[])
                        door_nodes.append(door_node)
                else:
                    
                    for i, lane in enumerate(doorway.lanes):
                        lane_x = lane.start_point.x
                        lane_y = lane.start_point.y
                        room_node_location = Coordinate(x=(lane_x+self.doorway_node_offset), y=lane_y)
                        doorway_node_location = Coordinate(x=lane_x, y=lane_y)

                        if i == 0:
                            orientation_vec = (-1.0, 0.0)
                        else:
                            orientation_vec = (1.0, 0.0)

                        room_node = TraversalNode(label=f"{doorway.corridor_id}_room_{i}",
                                                    position=room_node_location,
                                                    orientation_vec=orientation_vec,
                                                    connections=[])
                        room_nodes.append(room_node)

                        door_node = TraversalNode(label=f"{doorway.corridor_id}_door_{i}",
                                                    position=doorway_node_location,
                                                    orientation_vec=orientation_vec,
                                                    connections=[])
                        door_nodes.append(door_node)

        return room_nodes, door_nodes, left_nodes, right_nodes
    
    def _create_door_room_connections_for_nodes(self, 
                                                doorway: Doorway,
                                                corridor: Corridor,
                                                room_nodes: List[TraversalNode], 
                                                door_nodes: List[TraversalNode]) -> List[TraversalEdge]:
        edges = []
        if corridor.direction == "horizontal":
            if doorway.start.y <= corridor.width_start.y:
                left_room_node = room_nodes[0]
                right_room_node = room_nodes[1]
                left_door_node = door_nodes[0]
                right_door_node = door_nodes[1]
            else:
                left_room_node = room_nodes[1]
                left_door_node = door_nodes[1]
                right_room_node = room_nodes[0]
                right_door_node = door_nodes[0]
        else:
            if doorway.start.x <= corridor.width_start.x:
                left_room_node = room_nodes[1]
                left_door_node = door_nodes[1]
                right_room_node = room_nodes[0]
                right_door_node = door_nodes[0]
            else:
                left_room_node = room_nodes[0]
                left_door_node = door_nodes[0]
                right_room_node = room_nodes[1]
                right_door_node = door_nodes[1]
            
        right_edge = self._create_edge_between_nodes(from_node=right_door_node,
                                                     to_node=right_room_node, 
                                                     action="go_straight")
        left_edge = self._create_edge_between_nodes(from_node=left_room_node,
                                                    to_node=left_door_node, 
                                                    action="go_straight")
        switch_edge = self._create_edge_between_nodes(from_node=right_room_node,
                                                    to_node=left_room_node, 
                                                    action="switch_directions")
        edges.append(right_edge)
        edges.append(left_edge)
        edges.append(switch_edge)
            
        
        return edges
    
    def _create_door_corridor_connections_for_nodes(self, 
                                                  doorway: Doorway,
                                                  corridor: Corridor,
                                                  door_nodes: List[TraversalNode], 
                                                  left_nodes: List[TraversalNode], 
                                                  right_nodes: List[TraversalNode]) -> List[TraversalEdge]:
        edges = []
        if corridor.direction == "horizontal":
            if doorway.start.y <= corridor.width_start.y:
                door_right_node = door_nodes[1]
                door_left_node = door_nodes[0]
                corridor_right_nodes = right_nodes
                corridor_left_nodes = left_nodes
                right_orientation_vec = (1.0, 0.0)
                left_orientation_vec = (-1.0, 0.0)
            else:
                door_right_node = door_nodes[0]
                door_left_node = door_nodes[1]
                corridor_right_nodes = left_nodes
                corridor_left_nodes = right_nodes
                right_orientation_vec = (-1.0, 0.0)
                left_orientation_vec = (1.0, 0.0)
        else:
            if doorway.start.x <= corridor.width_start.x:
                door_right_node = door_nodes[0]
                door_left_node = door_nodes[1]
                corridor_right_nodes = right_nodes
                corridor_left_nodes = left_nodes
                right_orientation_vec = (0.0, -1.0)
                left_orientation_vec = (0.0, 1.0)
            else:
                door_right_node = door_nodes[1]
                door_left_node = door_nodes[0]
                corridor_right_nodes = left_nodes
                corridor_left_nodes = right_nodes
                right_orientation_vec = (0.0, 1.0)
                left_orientation_vec = (0.0, -1.0)
            
        for right_node in corridor_right_nodes:
            if right_node.orientation_vec == left_orientation_vec:
                edge = self._create_edge_between_nodes(from_node=right_node,
                                                        to_node=door_right_node, 
                                                        action="turn_right")
                edges.append(edge)
            else:
                edge = self._create_edge_between_nodes(from_node=door_left_node,
                                                        to_node=right_node, 
                                                        action="turn_left")
                edges.append(edge)
        
        for left_node in corridor_left_nodes:
            if left_node.orientation_vec == right_orientation_vec:
                edge = self._create_edge_between_nodes(from_node=left_node,
                                                        to_node=door_right_node, 
                                                        action="turn_left")
                edges.append(edge)
            else:
                edge = self._create_edge_between_nodes(from_node=door_left_node,
                                                        to_node=left_node, 
                                                        action="turn_right")
                edges.append(edge)
                
        return edges


    def _create_doorway_connections_for_nodes(self, 
                                              doorway: Doorway,
                                              corridor: Corridor,
                                              room_nodes: List[TraversalNode], 
                                              door_nodes: List[TraversalNode], 
                                              left_nodes: List[TraversalNode], 
                                              right_nodes: List[TraversalNode]) -> List[TraversalEdge]:
        edges = []

        room_edges = self._create_door_room_connections_for_nodes(doorway=doorway,
                                                                  corridor=corridor,
                                                                  room_nodes=room_nodes,
                                                                  door_nodes=door_nodes)
        edges.extend(room_edges)

        corridor_edges = self._create_door_corridor_connections_for_nodes(doorway=doorway,
                                                                        corridor=corridor,
                                                                        door_nodes=door_nodes,
                                                                        left_nodes=left_nodes,
                                                                        right_nodes=right_nodes)
        edges.extend(corridor_edges)
        
        return edges

    def _generate_doorway_traversal_subgraph(self) -> Tuple[List[IntersectionSubgraph], dict[str, List[int]]]:
        subgraphs = []
        subgraph_indices = {}
        current_index = 0
        for doorway in self.doorways:
            edges = []
            corridor = self._get_corridor_by_id(doorway.corridor_id)
            if corridor is None:
                print(f"Warning: Doorway corridor ID '{doorway.corridor_id}' not found among corridors.")
                continue

            room_nodes, door_nodes, left_nodes, right_nodes = self._extract_doorway_intersection_nodes(doorway=doorway, 
                                                                                           corridor=corridor)

            doorway_edges = self._create_doorway_connections_for_nodes(doorway=doorway,
                                                                       corridor=corridor,
                                                                       room_nodes=room_nodes,
                                                                       door_nodes=door_nodes,
                                                                       left_nodes=left_nodes,
                                                                       right_nodes=right_nodes)
            edges.extend(doorway_edges)

            current_subgraph = DoorwaySubgraph(room_nodes=room_nodes, 
                                                doorway_nodes=door_nodes, 
                                                left_nodes=left_nodes, 
                                                right_nodes=right_nodes, 
                                                edges=edges)
            subgraphs.append(current_subgraph)
            subgraph_indices.setdefault(doorway.corridor_id, []).append(current_index)
            current_index += 1
        return subgraphs, subgraph_indices

    def plot_intersection_subgraph(self, subgraphs: List[IntersectionSubgraph], filename: str):
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
        for subgraph in subgraphs:
            for edge in subgraph.edges:
                samples_x = edge.edge_connector.connector_dict['X']
                samples_y = edge.edge_connector.connector_dict['Y']
                if edge.action == "go_straight":
                    ax.plot(samples_x, samples_y, color='green', linewidth=0.5)
                elif edge.action == "turn_left":
                    ax.plot(samples_x, samples_y, color='orange', linewidth=0.5)
                elif edge.action == "turn_right":
                    ax.plot(samples_x, samples_y, color='purple', linewidth=0.5)

            for node in subgraph.upper_nodes + subgraph.lower_nodes + subgraph.left_nodes + subgraph.right_nodes:
                if node.orientation_vec == (1.0, 0.0):
                    ax.scatter(node.position.x, node.position.y, color='blue', s=1, alpha=0.5)
                elif node.orientation_vec == (-1.0, 0.0):
                    ax.scatter(node.position.x, node.position.y, color='cyan', s=1, alpha=0.5)
                elif node.orientation_vec == (0.0, 1.0):
                    ax.scatter(node.position.x, node.position.y, color='magenta', s=1, alpha=0.5)
                elif node.orientation_vec == (0.0, -1.0):
                    ax.scatter(node.position.x, node.position.y, color='red', s=1, alpha=0.5)

        ax.set_title("Intersection Subgraph Overlay")
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        plt.savefig(f"results/{filename}")
        plt.close()
    
    def plot_doorway_subgraph(self, subgraphs: List[DoorwaySubgraph], filename: str):
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
        for subgraph in subgraphs:
            for edge in subgraph.edges:
                samples_x = edge.edge_connector.connector_dict['X']
                samples_y = edge.edge_connector.connector_dict['Y']
                if edge.action == "go_straight":
                    ax.plot(samples_x, samples_y, color='green', linewidth=0.5)
                elif edge.action == "turn_left":
                    ax.plot(samples_x, samples_y, color='orange', linewidth=0.5)
                elif edge.action == "turn_right":
                    ax.plot(samples_x, samples_y, color='purple', linewidth=0.5)
                elif edge.action == "switch_directions":
                    ax.plot(samples_x, samples_y, color='brown', linewidth=0.5)

            for node in subgraph.room_nodes + subgraph.doorway_nodes + subgraph.left_nodes + subgraph.right_nodes:
                if node.orientation_vec == (1.0, 0.0):
                    ax.scatter(node.position.x, node.position.y, color='blue', s=1, alpha=0.5)
                elif node.orientation_vec == (-1.0, 0.0):
                    ax.scatter(node.position.x, node.position.y, color='cyan', s=1, alpha=0.5)
                elif node.orientation_vec == (0.0, 1.0):
                    ax.scatter(node.position.x, node.position.y, color='magenta', s=1, alpha=0.5)
                elif node.orientation_vec == (0.0, -1.0):
                    ax.scatter(node.position.x, node.position.y, color='red', s=1, alpha=0.5)

        ax.set_title("Doorway Subgraph Overlay")
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        plt.savefig(f"results/{filename}")
        plt.close()

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
        plt.close()
    

    
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
    tg_generator.plot_intersection_subgraph(tg_generator.corridor_intersection_subgraphs, filename="intersection_subgraph_0.svg")
    tg_generator.plot_doorway_subgraph(tg_generator.doorway_subgraphs, filename="doorway_subgraph_0.svg")
    print(tg_generator.corridor_intersection_subgraph_indices)
    print(tg_generator.doorway_subgraph_indices)
