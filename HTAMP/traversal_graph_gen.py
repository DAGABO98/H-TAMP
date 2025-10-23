import copy
from typing import Dict, List, Tuple, Union
import yaml
import argparse
import numpy as np

from HTAMP.traversal_dataclasses import Lane, OrientationVector, Corridor, DriveThrough, Coordinate
from HTAMP.traversal_dataclasses import Doorway, TraversalNode, TraversalEdge, IntersectionSubgraph
from HTAMP.traversal_dataclasses import DoorwaySubgraph, DriveThroughSubgraph, SwitchingPointSubgraph
from HTAMP.traversal_dataclasses import EndPointSubgraph, TraversalGraph
from HTAMP.geometry_helpers import CurvedConnector
from HTAMP.plotting_helpers import TraversalGraphPlottingHelper

class TraversalGraphGenerator:
    def __init__(self, occupancy_map_path: str, config_path: str, meters_per_pixel: float = 0.036, factor: int = 1,
                 num_lanes_per_corridor: int = 3, num_lanes_per_drive_through: int = 1, num_lanes_per_doorway: int = 2,
                 doorway_lane_threshold: float = 20.0, tangent_scaling_factor: float = 1.2, num_samples: int = 10, threshold: float = 10.0,
                 switching_point_offset: float = 15.0):
        self.occupancy_map_path = occupancy_map_path
        self.config_path = config_path
        self.tangent_scaling_factor = tangent_scaling_factor
        self.num_samples = num_samples
        self.meters_per_cell = meters_per_pixel * factor
        self.threshold = threshold * self.meters_per_cell
        self.switching_point_offset = switching_point_offset * self.meters_per_cell
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
        self.corridor_intersection_subgraphs, self.corridor_intersection_subgraph_indices = self._generate_corridor_intersection_traversal_subgraphs()
        self.doorway_subgraphs, self.doorway_subgraph_indices = self._generate_doorway_traversal_subgraphs()
        self.drive_through_subgraphs, self.drive_through_subgraph_indices = self._generate_drive_through_traversal_subgraphs()
        self.switching_point_subgraphs = []
        self._merge_overlapping_doorway_subgraphs()
        self._merge_overlapping_drive_through_doorway_subgraphs()
        self._merge_overlapping_intersections_doorway_subgraphs()
        self.traversal_graph = self._assemble_traversal_graph()

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
                                             intersection_corridor: Corridor) -> Tuple[List[str], List[str], List[str], List[str], Dict[str, TraversalNode]]:
        upper_nodes = []
        lower_nodes = []
        left_nodes = []
        right_nodes = []
        nodes_dict = {}

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
                    orientation_vec = OrientationVector(0.0, 1.0)  # Facing down
                elif i == len(intersection_corridor.lanes) - 1:
                    orientation_vec = OrientationVector(0.0, -1.0)  # Facing up
                else:
                    orientation_vec = None
                
                if orientation_vec is not None:
                    if start_node_location is not None:
                        upper_node = TraversalNode(label=f"{start_node_location.x:.2f}_{start_node_location.y:.2f}_{orientation_vec}",
                                                    position=start_node_location,
                                                    orientation_vec=orientation_vec,
                                                    connections=[])
                        nodes_dict[upper_node.label] = upper_node
                        upper_nodes.append(upper_node.label)
                    if end_node_location is not None:
                        lower_node = TraversalNode(label=f"{end_node_location.x:.2f}_{end_node_location.y:.2f}_{orientation_vec}",
                                                    position=end_node_location,
                                                    orientation_vec=orientation_vec,
                                                    connections=[])
                        nodes_dict[lower_node.label] = lower_node
                        lower_nodes.append(lower_node.label)
                else:
                    possible_orientations = [OrientationVector(0.0, 1.0), OrientationVector(0.0, -1.0)]
                    for possible_orientation in possible_orientations:
                        if start_node_location is not None:
                            upper_node = TraversalNode(label=f"{start_node_location.x:.2f}_{start_node_location.y:.2f}_{possible_orientation}",
                                                        position=start_node_location,
                                                        orientation_vec=possible_orientation,
                                                        connections=[])
                            nodes_dict[upper_node.label] = upper_node
                            upper_nodes.append(upper_node.label)
                        if end_node_location is not None:
                            lower_node = TraversalNode(label=f"{end_node_location.x:.2f}_{end_node_location.y:.2f}_{possible_orientation}",
                                                    position=end_node_location,
                                                    orientation_vec=possible_orientation,
                                                    connections=[])
                            nodes_dict[lower_node.label] = lower_node
                            lower_nodes.append(lower_node.label)

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
                    orientation_vec = OrientationVector(-1.0, 0.0)  # Facing  left
                elif j == len(corridor.lanes) - 1:
                    orientation_vec = OrientationVector(1.0, 0.0)  # Facing right
                else:
                    orientation_vec = None
                
                if orientation_vec is not None:
                    if start_node_location is not None:
                        left_node = TraversalNode(label=f"{start_node_location.x:.2f}_{start_node_location.y:.2f}_{orientation_vec}",
                                                    position=start_node_location,
                                                    orientation_vec=orientation_vec,
                                                    connections=[])
                        nodes_dict[left_node.label] = left_node
                        left_nodes.append(left_node.label)
                    if end_node_location is not None:
                        right_node = TraversalNode(label=f"{end_node_location.x:.2f}_{end_node_location.y:.2f}_{orientation_vec}",
                                                    position=end_node_location,
                                                    orientation_vec=orientation_vec,
                                                    connections=[])
                        nodes_dict[right_node.label] = right_node
                        right_nodes.append(right_node.label)
                else:
                    possible_orientations = [OrientationVector(-1.0, 0.0), OrientationVector(1.0, 0.0)]
                    for possible_orientation in possible_orientations:
                        if start_node_location is not None:
                            left_node = TraversalNode(label=f"{start_node_location.x:.2f}_{start_node_location.y:.2f}_{possible_orientation}",
                                                        position=start_node_location,
                                                        orientation_vec=possible_orientation,
                                                        connections=[])
                            nodes_dict[left_node.label] = left_node
                            left_nodes.append(left_node.label)
                        if end_node_location is not None:
                            right_node = TraversalNode(label=f"{end_node_location.x:.2f}_{end_node_location.y:.2f}_{possible_orientation}",
                                                    position=end_node_location,
                                                    orientation_vec=possible_orientation,
                                                    connections=[])
                            nodes_dict[right_node.label] = right_node
                            right_nodes.append(right_node.label)

        return upper_nodes, lower_nodes, left_nodes, right_nodes, nodes_dict
    
    def _create_edge_between_nodes(self, 
                                   from_node: TraversalNode, 
                                   to_node: TraversalNode, 
                                   action: str) -> TraversalEdge:
        from_node.connections.append(to_node.label)
        edge_connector = CurvedConnector(origin=from_node.position, 
                                        destination=to_node.position, 
                                        vec_origin=(from_node.orientation_vec.x, from_node.orientation_vec.y),
                                        vec_destination=(to_node.orientation_vec.x, to_node.orientation_vec.y),
                                        tangent_scaling_factor=self.tangent_scaling_factor,
                                        num_samples=self.num_samples)
        edge = TraversalEdge(from_node=from_node.label, 
                            to_node=to_node.label, 
                            action=action, 
                            edge_connector=edge_connector)
        return edge
    
    def _get_corridor_by_id(self, corridor_id: str) -> Corridor:
        for corridor in self.corridors:
            if corridor.corridor_id == corridor_id:
                return corridor
        return None
    
    def _create_intersection_connections_for_nodes(self, 
                                                   reference_nodes_labels: List[str], 
                                                   ref_orientation: OrientationVector,
                                                   opposite_nodes_labels: List[str],
                                                   right_nodes_labels: List[str],
                                                   right_orientation: OrientationVector,
                                                   left_nodes_labels: List[str],
                                                   left_orientation: OrientationVector,
                                                   nodes_dict: Dict[str, TraversalNode],
                                                   horizontal: bool,
                                                   invert: bool) -> List[TraversalEdge]:
        edges = []
        if invert:
            ref_index = len(reference_nodes_labels) - 1
        else:
            ref_index = 0

        for i, ref_node_label in enumerate(reference_nodes_labels):
            ref_node = nodes_dict[ref_node_label]
            if ref_node.orientation_vec == ref_orientation:
                if i == ref_index:
                    if len(opposite_nodes_labels) > 0:
                        opp_node = nodes_dict[opposite_nodes_labels[i]]
                        assert opp_node.orientation_vec == ref_node.orientation_vec
                        edge = self._create_edge_between_nodes(from_node=ref_node, 
                                                               to_node=opp_node, 
                                                               action="go_straight")
                        edges.append(edge)
                    
                    if len(right_nodes_labels) > 0:
                        for right_node_label in right_nodes_labels:
                            right_node = nodes_dict[right_node_label]
                            if right_node.orientation_vec == right_orientation:
                                edge = self._create_edge_between_nodes(from_node=ref_node, 
                                                                       to_node=right_node, 
                                                                       action="turn_right")
                                edges.append(edge)
                else:
                    if len(opposite_nodes_labels) > 0:
                        for opp_node_label in opposite_nodes_labels:
                            opp_node = nodes_dict[opp_node_label]
                            if horizontal:
                                directionality_flag = opp_node.position.x == ref_node.position.x
                            else:
                                directionality_flag = opp_node.position.y == ref_node.position.y

                            if opp_node.orientation_vec == ref_orientation and directionality_flag:
                                edge = self._create_edge_between_nodes(from_node=ref_node, 
                                                                       to_node=opp_node, 
                                                                       action="go_straight")
                                edges.append(edge)

                    if len(left_nodes_labels) > 0:
                        for left_node_label in left_nodes_labels:
                            left_node = nodes_dict[left_node_label]
                            if left_node.orientation_vec == left_orientation:
                                edge = self._create_edge_between_nodes(from_node=ref_node,
                                                                       to_node=left_node,
                                                                       action="turn_left")
                                edges.append(edge)
        return edges

    def _generate_corridor_intersection_traversal_subgraphs(self) -> Tuple[List[IntersectionSubgraph], dict[str, List[int]]]:
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

                extraction_results = self._extract_corridor_intersection_nodes(corridor=corridor,
                                                                               intersection_corridor=intersection_corridor)
                
                upper_nodes_labels, lower_nodes_labels, left_nodes_labels, right_nodes_labels, nodes_dict = extraction_results
                
                seen_corridors.add(corridor.corridor_id)

                upper_edges = self._create_intersection_connections_for_nodes(reference_nodes_labels=upper_nodes_labels,
                                                                            ref_orientation=OrientationVector(0.0, 1.0),
                                                                            opposite_nodes_labels=lower_nodes_labels,
                                                                            right_nodes_labels=left_nodes_labels,
                                                                            right_orientation=OrientationVector(-1.0, 0.0),
                                                                            left_nodes_labels=right_nodes_labels,
                                                                            left_orientation=OrientationVector(1.0, 0.0),
                                                                            nodes_dict=nodes_dict,
                                                                            horizontal=True,
                                                                            invert=False)
                edges.extend(upper_edges)

                lower_edges = self._create_intersection_connections_for_nodes(reference_nodes_labels=lower_nodes_labels,
                                                                            ref_orientation=OrientationVector(0.0, -1.0),
                                                                            opposite_nodes_labels=upper_nodes_labels,
                                                                            right_nodes_labels=right_nodes_labels,
                                                                            right_orientation=OrientationVector(1.0, 0.0),
                                                                            left_nodes_labels=left_nodes_labels,
                                                                            left_orientation=OrientationVector(-1.0, 0.0),
                                                                            nodes_dict=nodes_dict,
                                                                            horizontal=True,
                                                                            invert=True)
                edges.extend(lower_edges)

                right_edges = self._create_intersection_connections_for_nodes(reference_nodes_labels=right_nodes_labels,
                                                                            ref_orientation=OrientationVector(-1.0, 0.0),
                                                                            opposite_nodes_labels=left_nodes_labels,
                                                                            right_nodes_labels=upper_nodes_labels,
                                                                            right_orientation=OrientationVector(0.0, -1.0),
                                                                            left_nodes_labels=lower_nodes_labels,
                                                                            left_orientation=OrientationVector(0.0, 1.0),
                                                                            nodes_dict=nodes_dict,
                                                                            horizontal=False,
                                                                            invert=False)
                edges.extend(right_edges)

                left_edges = self._create_intersection_connections_for_nodes(reference_nodes_labels=left_nodes_labels,
                                                                            ref_orientation=OrientationVector(1.0, 0.0),
                                                                            opposite_nodes_labels=right_nodes_labels,
                                                                            right_nodes_labels=lower_nodes_labels,
                                                                            right_orientation=OrientationVector(0.0, 1.0),
                                                                            left_nodes_labels=upper_nodes_labels,
                                                                            left_orientation=OrientationVector(0.0, -1.0),
                                                                            nodes_dict=nodes_dict,
                                                                            horizontal=False,
                                                                            invert=True)
                edges.extend(left_edges)
                current_subgraph = IntersectionSubgraph(upper_nodes=upper_nodes_labels, 
                                                        lower_nodes=lower_nodes_labels, 
                                                        left_nodes=left_nodes_labels, 
                                                        right_nodes=right_nodes_labels, 
                                                        nodes_dict=nodes_dict,
                                                        edges=edges)

                subgraphs.append(current_subgraph)
                subgraph_indices.setdefault(corridor.corridor_id, []).append(current_index)
                subgraph_indices.setdefault(intersection_corridor.corridor_id, []).append(current_index)
                current_index += 1

        return subgraphs, subgraph_indices

    def _extract_doorway_intersection_nodes(self, 
                                             doorway: Doorway,
                                             corridor: Corridor) -> Tuple[List[str], List[str], List[str], List[str], Dict[str, TraversalNode]]:
        room_nodes = []
        door_nodes = []
        left_nodes = []
        right_nodes = []
        nodes_dict = {}

        if corridor.direction == "horizontal":
            doorway_start_x = doorway.start.x
            doorway_end_x = doorway.end.x
            for i, corridor_lane in enumerate(corridor.lanes):
                lane_y = corridor_lane.start_point.y
                left_node_location = Coordinate(x=doorway_start_x, y=lane_y)
                right_node_location = Coordinate(x=doorway_end_x, y=lane_y)
                if i == 0:
                    orientation_vec = OrientationVector(-1.0, 0.0)  # Facing  left
                elif i == len(corridor.lanes) - 1:
                    orientation_vec = OrientationVector(1.0, 0.0)  # Facing right
                else:
                    orientation_vec = None
                if orientation_vec is None:
                    possible_orientations = [OrientationVector(-1.0, 0.0), OrientationVector(1.0, 0.0)]
                    for possible_orientation in possible_orientations:
                        left_node = TraversalNode(label=f"{left_node_location.x:.2f}_{left_node_location.y:.2f}_{possible_orientation}",
                                                    position=left_node_location,
                                                    orientation_vec=possible_orientation,
                                                    connections=[])
                        right_node = TraversalNode(label=f"{right_node_location.x:.2f}_{right_node_location.y:.2f}_{possible_orientation}",
                                                    position=right_node_location,
                                                    orientation_vec=possible_orientation,
                                                    connections=[])
                        nodes_dict[left_node.label] = left_node
                        nodes_dict[right_node.label] = right_node
                        left_nodes.append(left_node.label)
                        right_nodes.append(right_node.label)
                else:
                    left_node = TraversalNode(label=f"{left_node_location.x:.2f}_{left_node_location.y:.2f}_{orientation_vec}",
                                                position=left_node_location,
                                                orientation_vec=orientation_vec,
                                                connections=[])
                    right_node = TraversalNode(label=f"{right_node_location.x:.2f}_{right_node_location.y:.2f}_{orientation_vec}",
                                                position=right_node_location,
                                                orientation_vec=orientation_vec,
                                                connections=[])
                    nodes_dict[left_node.label] = left_node
                    nodes_dict[right_node.label] = right_node
                    left_nodes.append(left_node.label)
                    right_nodes.append(right_node.label)

            if doorway.start.y <= corridor.width_start.y:
                if len(doorway.lanes) == 1:
                    possible_orientations = [OrientationVector(0.0, 1.0), OrientationVector(0.0, -1.0)]
                    for possible_orientation in possible_orientations:
                        lane = doorway.lanes[0]
                        lane_x = lane.start_point.x
                        lane_y = lane.start_point.y
                        room_node_location = Coordinate(x=lane_x, y=(lane_y-self.doorway_node_offset))
                        doorway_node_location = Coordinate(x=lane_x, y=lane_y)

                        room_node = TraversalNode(label=f"{room_node_location.x:.2f}_{room_node_location.y:.2f}_{possible_orientation}",
                                                    position=room_node_location,
                                                    orientation_vec=possible_orientation,
                                                    connections=[])
                        nodes_dict[room_node.label] = room_node
                        room_nodes.append(room_node.label)

                        door_node = TraversalNode(label=f"{doorway_node_location.x:.2f}_{doorway_node_location.y:.2f}_{possible_orientation}",
                                                    position=doorway_node_location,
                                                    orientation_vec=possible_orientation,
                                                    connections=[])
                        nodes_dict[door_node.label] = door_node
                        door_nodes.append(door_node.label)
                else:

                    for i, lane in enumerate(doorway.lanes):
                        lane_x = lane.start_point.x
                        lane_y = lane.start_point.y
                        room_node_location = Coordinate(x=lane_x, y=(lane_y-self.doorway_node_offset))
                        doorway_node_location = Coordinate(x=lane_x, y=lane_y)

                        if i == 0:
                            orientation_vec = OrientationVector(0.0, 1.0)  # Facing down
                        else:
                            orientation_vec = OrientationVector(0.0, -1.0)  # Facing up

                        room_node = TraversalNode(label=f"{room_node_location.x:.2f}_{room_node_location.y:.2f}_{orientation_vec}",
                                                    position=room_node_location,
                                                    orientation_vec=orientation_vec,
                                                    connections=[])
                        nodes_dict[room_node.label] = room_node
                        room_nodes.append(room_node.label)

                        door_node = TraversalNode(label=f"{doorway_node_location.x:.2f}_{doorway_node_location.y:.2f}_{orientation_vec}",
                                                    position=doorway_node_location,
                                                    orientation_vec=orientation_vec,
                                                    connections=[])
                        nodes_dict[door_node.label] = door_node
                        door_nodes.append(door_node.label)
            else:
                if len(doorway.lanes) == 1:
                    possible_orientations = [OrientationVector(0.0, 1.0), OrientationVector(0.0, -1.0)]
                    for possible_orientation in possible_orientations:
                        lane = doorway.lanes[0]
                        lane_x = lane.start_point.x
                        lane_y = lane.start_point.y
                        room_node_location = Coordinate(x=lane_x, y=(lane_y+self.doorway_node_offset))
                        doorway_node_location = Coordinate(x=lane_x, y=lane_y)

                        room_node = TraversalNode(label=f"{room_node_location.x:.2f}_{room_node_location.y:.2f}_{possible_orientation}",
                                                    position=room_node_location,
                                                    orientation_vec=possible_orientation,
                                                    connections=[])
                        nodes_dict[room_node.label] = room_node
                        room_nodes.append(room_node.label)

                        door_node = TraversalNode(label=f"{doorway_node_location.x:.2f}_{doorway_node_location.y:.2f}_{possible_orientation}",
                                                    position=doorway_node_location,
                                                    orientation_vec=possible_orientation,
                                                    connections=[])
                        nodes_dict[door_node.label] = door_node
                        door_nodes.append(door_node.label)
                else:
                    for i, lane in enumerate(doorway.lanes):
                        lane_x = lane.start_point.x
                        lane_y = lane.start_point.y
                        room_node_location = Coordinate(x=lane_x, y=(lane_y+self.doorway_node_offset))
                        doorway_node_location = Coordinate(x=lane_x, y=lane_y)

                        if i == 0:
                            orientation_vec = OrientationVector(0.0, 1.0)  # Facing down
                        else:
                            orientation_vec = OrientationVector(0.0, -1.0)  # Facing up

                        room_node = TraversalNode(label=f"{room_node_location.x:.2f}_{room_node_location.y:.2f}_{orientation_vec}",
                                                    position=room_node_location,
                                                    orientation_vec=orientation_vec,
                                                    connections=[])
                        nodes_dict[room_node.label] = room_node
                        room_nodes.append(room_node.label)

                        door_node = TraversalNode(label=f"{doorway_node_location.x:.2f}_{doorway_node_location.y:.2f}_{orientation_vec}",
                                                    position=doorway_node_location,
                                                    orientation_vec=orientation_vec,
                                                    connections=[])
                        nodes_dict[door_node.label] = door_node
                        door_nodes.append(door_node.label)
        else:
            doorway_start_y = doorway.start.y
            doorway_end_y = doorway.end.y
            for i, corridor_lane in enumerate(corridor.lanes):
                lane_x = corridor_lane.start_point.x
                right_node_location = Coordinate(x=lane_x, y=doorway_start_y)
                left_node_location = Coordinate(x=lane_x, y=doorway_end_y)
                if i == 0:
                    orientation_vec = OrientationVector(0.0, 1.0)  # Facing down
                elif i == len(corridor.lanes) - 1:
                    orientation_vec = OrientationVector(0.0, -1.0)  # Facing up
                else:
                    orientation_vec = None
                if orientation_vec is None:
                    possible_orientations = [OrientationVector(0.0, 1.0), OrientationVector(0.0, -1.0)]
                    for possible_orientation in possible_orientations:
                        left_node = TraversalNode(label=f"{left_node_location.x:.2f}_{left_node_location.y:.2f}_{possible_orientation}",
                                                    position=left_node_location,
                                                    orientation_vec=possible_orientation,
                                                    connections=[])
                        right_node = TraversalNode(label=f"{right_node_location.x:.2f}_{right_node_location.y:.2f}_{possible_orientation}",
                                                    position=right_node_location,
                                                    orientation_vec=possible_orientation,
                                                    connections=[])
                        nodes_dict[left_node.label] = left_node
                        nodes_dict[right_node.label] = right_node
                        left_nodes.append(left_node.label)
                        right_nodes.append(right_node.label)
                else:
                    left_node = TraversalNode(label=f"{left_node_location.x:.2f}_{left_node_location.y:.2f}_{orientation_vec}",
                                                position=left_node_location,
                                                orientation_vec=orientation_vec,
                                                connections=[])
                    right_node = TraversalNode(label=f"{right_node_location.x:.2f}_{right_node_location.y:.2f}_{orientation_vec}",
                                                position=right_node_location,
                                                orientation_vec=orientation_vec,
                                                connections=[])
                    nodes_dict[left_node.label] = left_node
                    nodes_dict[right_node.label] = right_node
                    left_nodes.append(left_node.label)
                    right_nodes.append(right_node.label)

            if doorway.start.x <= corridor.width_start.x:
                if len(doorway.lanes) == 1:
                    possible_orientations = [(OrientationVector(-1.0, 0.0)), (OrientationVector(1.0, 0.0))]
                    for possible_orientation in possible_orientations:
                        lane = doorway.lanes[0]
                        lane_x = lane.start_point.x
                        lane_y = lane.start_point.y
                        room_node_location = Coordinate(x=(lane_x-self.doorway_node_offset), y=lane_y)
                        doorway_node_location = Coordinate(x=lane_x, y=lane_y)

                        room_node = TraversalNode(label=f"{room_node_location.x:.2f}_{room_node_location.y:.2f}_{possible_orientation}",
                                                    position=room_node_location,
                                                    orientation_vec=possible_orientation,
                                                    connections=[])
                        nodes_dict[room_node.label] = room_node
                        room_nodes.append(room_node.label)

                        door_node = TraversalNode(label=f"{doorway_node_location.x:.2f}_{doorway_node_location.y:.2f}_{possible_orientation}",
                                                    position=doorway_node_location,
                                                    orientation_vec=possible_orientation,
                                                    connections=[])
                        nodes_dict[door_node.label] = door_node
                        door_nodes.append(door_node.label)
                else:

                    for i, lane in enumerate(doorway.lanes):
                        lane_x = lane.start_point.x
                        lane_y = lane.start_point.y
                        room_node_location = Coordinate(x=(lane_x-self.doorway_node_offset), y=lane_y)
                        doorway_node_location = Coordinate(x=lane_x, y=lane_y)

                        if i == 0:
                            orientation_vec = OrientationVector(-1.0, 0.0)
                        else:
                            orientation_vec = OrientationVector(1.0, 0.0)

                        room_node = TraversalNode(label=f"{room_node_location.x:.2f}_{room_node_location.y:.2f}_{orientation_vec}",
                                                    position=room_node_location,
                                                    orientation_vec=orientation_vec,
                                                    connections=[])
                        nodes_dict[room_node.label] = room_node
                        room_nodes.append(room_node.label)

                        door_node = TraversalNode(label=f"{doorway_node_location.x:.2f}_{doorway_node_location.y:.2f}_{orientation_vec}",
                                                    position=doorway_node_location,
                                                    orientation_vec=orientation_vec,
                                                    connections=[])
                        nodes_dict[door_node.label] = door_node
                        door_nodes.append(door_node.label)
            else:
                if len(doorway.lanes) == 1:
                    possible_orientations = [OrientationVector(-1.0, 0.0), OrientationVector(1.0, 0.0)]
                    for possible_orientation in possible_orientations:
                        lane = doorway.lanes[0]
                        lane_x = lane.start_point.x
                        lane_y = lane.start_point.y
                        room_node_location = Coordinate(x=(lane_x+self.doorway_node_offset), y=lane_y)
                        doorway_node_location = Coordinate(x=lane_x, y=lane_y)

                        room_node = TraversalNode(label=f"{room_node_location.x:.2f}_{room_node_location.y:.2f}_{possible_orientation}",
                                                    position=room_node_location,
                                                    orientation_vec=possible_orientation,
                                                    connections=[])
                        nodes_dict[room_node.label] = room_node
                        room_nodes.append(room_node.label)

                        door_node = TraversalNode(label=f"{doorway_node_location.x:.2f}_{doorway_node_location.y:.2f}_{possible_orientation}",
                                                    position=doorway_node_location,
                                                    orientation_vec=possible_orientation,
                                                    connections=[])
                        nodes_dict[door_node.label] = door_node
                        door_nodes.append(door_node.label)
                else:
                    
                    for i, lane in enumerate(doorway.lanes):
                        lane_x = lane.start_point.x
                        lane_y = lane.start_point.y
                        room_node_location = Coordinate(x=(lane_x+self.doorway_node_offset), y=lane_y)
                        doorway_node_location = Coordinate(x=lane_x, y=lane_y)

                        if i == 0:
                            orientation_vec = OrientationVector(-1.0, 0.0)
                        else:
                            orientation_vec = OrientationVector(1.0, 0.0)

                        room_node = TraversalNode(label=f"{room_node_location.x:.2f}_{room_node_location.y:.2f}_{orientation_vec}",
                                                    position=room_node_location,
                                                    orientation_vec=orientation_vec,
                                                    connections=[])
                        nodes_dict[room_node.label] = room_node
                        room_nodes.append(room_node.label)

                        door_node = TraversalNode(label=f"{doorway_node_location.x:.2f}_{doorway_node_location.y:.2f}_{orientation_vec}",
                                                    position=doorway_node_location,
                                                    orientation_vec=orientation_vec,
                                                    connections=[])
                        nodes_dict[door_node.label] = door_node
                        door_nodes.append(door_node.label)

        return room_nodes, door_nodes, left_nodes, right_nodes, nodes_dict
    
    def _create_door_room_connections_for_nodes(self, 
                                                doorway: Doorway,
                                                corridor: Corridor,
                                                room_nodes: List[str], 
                                                door_nodes: List[str],
                                                nodes_dict: Dict[str, TraversalNode]) -> List[TraversalEdge]:
        edges = []
        if corridor.direction == "horizontal":
            if doorway.start.y <= corridor.width_start.y:
                left_room_node_label = room_nodes[0]
                right_room_node_label = room_nodes[1]
                left_door_node_label = door_nodes[0]
                right_door_node_label = door_nodes[1]
            else:
                left_room_node_label = room_nodes[1]
                left_door_node_label = door_nodes[1]
                right_room_node_label = room_nodes[0]
                right_door_node_label = door_nodes[0]
        else:
            if doorway.start.x <= corridor.width_start.x:
                left_room_node_label = room_nodes[1]
                left_door_node_label = door_nodes[1]
                right_room_node_label = room_nodes[0]
                right_door_node_label = door_nodes[0]
            else:
                left_room_node_label = room_nodes[0]
                left_door_node_label = door_nodes[0]
                right_room_node_label = room_nodes[1]
                right_door_node_label = door_nodes[1]
        
        right_room_node = nodes_dict[right_room_node_label]
        right_door_node = nodes_dict[right_door_node_label]
        left_room_node = nodes_dict[left_room_node_label]
        left_door_node = nodes_dict[left_door_node_label]
            
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
                                                  door_nodes: List[str], 
                                                  left_nodes: List[str], 
                                                  right_nodes: List[str],
                                                  nodes_dict: Dict[str, TraversalNode]) -> List[TraversalEdge]:
        edges = []
        if corridor.direction == "horizontal":
            if doorway.start.y <= corridor.width_start.y:
                door_right_node_label = door_nodes[1]
                door_left_node_label = door_nodes[0]
                corridor_right_nodes_labels = right_nodes
                corridor_left_nodes_labels = left_nodes
                right_orientation_vec = OrientationVector(1.0, 0.0)
                left_orientation_vec = OrientationVector(-1.0, 0.0)
            else:
                door_right_node_label = door_nodes[0]
                door_left_node_label = door_nodes[1]
                corridor_right_nodes_labels = left_nodes
                corridor_left_nodes_labels = right_nodes
                right_orientation_vec = OrientationVector(-1.0, 0.0)
                left_orientation_vec = OrientationVector(1.0, 0.0)
        else:
            if doorway.start.x <= corridor.width_start.x:
                door_right_node_label = door_nodes[0]
                door_left_node_label = door_nodes[1]
                corridor_right_nodes_labels = right_nodes
                corridor_left_nodes_labels = left_nodes
                right_orientation_vec = OrientationVector(0.0, -1.0)
                left_orientation_vec = OrientationVector(0.0, 1.0)
            else:
                door_right_node_label = door_nodes[1]
                door_left_node_label = door_nodes[0]
                corridor_right_nodes_labels = left_nodes
                corridor_left_nodes_labels = right_nodes
                right_orientation_vec = OrientationVector(0.0, 1.0)
                left_orientation_vec = OrientationVector(0.0, -1.0)

        door_right_node = nodes_dict[door_right_node_label]
        door_left_node = nodes_dict[door_left_node_label]

        for right_node_label in corridor_right_nodes_labels:
            right_node = nodes_dict[right_node_label]
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

        for left_node_label in corridor_left_nodes_labels:
            left_node = nodes_dict[left_node_label]
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

    def _create_straight_connections_for_nodes(self, 
                                               ref_nodes: List[str], 
                                               opp_nodes: List[str],
                                               ref_direction_vec: Tuple[float, float],
                                               nodes_dict: Dict[str, TraversalNode]) -> List[TraversalEdge]:
        edges = []
        for ref_node_label in ref_nodes:
            ref_node = nodes_dict[ref_node_label]
            if ref_node.orientation_vec == ref_direction_vec:
                for opp_node_label in opp_nodes:
                    opp_node = nodes_dict[opp_node_label]
                    if opp_node.orientation_vec == ref_direction_vec:
                        if opp_node.position.x == ref_node.position.x or opp_node.position.y == ref_node.position.y:
                            edge = self._create_edge_between_nodes(from_node=ref_node,
                                                                    to_node=opp_node, 
                                                                    action="go_straight")
                            edges.append(edge)
        return edges
    
    def _create_corridor_straight_connections_for_nodes(self,
                                                        corridor: Corridor,
                                                        left_nodes: List[str], 
                                                        right_nodes: List[str],
                                                        nodes_dict: dict[str, TraversalNode]) -> List[TraversalEdge]:
        edges = []
        if corridor.direction == "horizontal":
            original_ref_direction_vec = OrientationVector(1.0, 0.0)  # Facing right
            inverted_ref_direction_vec = OrientationVector(-1.0, 0.0)  # Facing left
        else:
            original_ref_direction_vec = OrientationVector(0.0, -1.0)  # Facing up
            inverted_ref_direction_vec = OrientationVector(0.0, 1.0)  # Facing down
        
        straight_edges = self._create_straight_connections_for_nodes(ref_nodes=left_nodes,
                                                                     opp_nodes=right_nodes,
                                                                     ref_direction_vec=original_ref_direction_vec,
                                                                     nodes_dict=nodes_dict)
        edges.extend(straight_edges)

        inverted_straight_edges = self._create_straight_connections_for_nodes(ref_nodes=right_nodes,
                                                                             opp_nodes=left_nodes,
                                                                             ref_direction_vec=inverted_ref_direction_vec,
                                                                             nodes_dict=nodes_dict)
        edges.extend(inverted_straight_edges)

        return edges


    def _create_doorway_connections_for_nodes(self, 
                                              doorway: Doorway,
                                              corridor: Corridor,
                                              room_nodes: List[str], 
                                              door_nodes: List[str], 
                                              left_nodes: List[str], 
                                              right_nodes: List[str],
                                              nodes_dict: dict[str, TraversalNode]) -> List[TraversalEdge]:
        edges = []

        room_edges = self._create_door_room_connections_for_nodes(doorway=doorway,
                                                                  corridor=corridor,
                                                                  room_nodes=room_nodes,
                                                                  door_nodes=door_nodes,
                                                                  nodes_dict=nodes_dict)
        edges.extend(room_edges)

        corridor_edges = self._create_door_corridor_connections_for_nodes(doorway=doorway,
                                                                        corridor=corridor,
                                                                        door_nodes=door_nodes,
                                                                        left_nodes=left_nodes,
                                                                        right_nodes=right_nodes,
                                                                        nodes_dict=nodes_dict)
        edges.extend(corridor_edges)

        straight_edges = self._create_corridor_straight_connections_for_nodes(corridor=corridor,
                                                                             left_nodes=left_nodes,
                                                                             right_nodes=right_nodes,
                                                                             nodes_dict=nodes_dict)
        edges.extend(straight_edges)

        return edges

    def _generate_doorway_traversal_subgraphs(self) -> Tuple[List[IntersectionSubgraph], dict[str, List[int]]]:
        subgraphs = []
        subgraph_indices = {}
        current_index = 0
        for doorway in self.doorways:
            edges = []
            corridor = self._get_corridor_by_id(doorway.corridor_id)
            if corridor is None:
                print(f"Warning: Doorway corridor ID '{doorway.corridor_id}' not found among corridors.")
                continue

            room_nodes, door_nodes, left_nodes, right_nodes, nodes_dict = self._extract_doorway_intersection_nodes(doorway=doorway, 
                                                                                                                   corridor=corridor)

            doorway_edges = self._create_doorway_connections_for_nodes(doorway=doorway,
                                                                       corridor=corridor,
                                                                       room_nodes=room_nodes,
                                                                       door_nodes=door_nodes,
                                                                       left_nodes=left_nodes,
                                                                       right_nodes=right_nodes,
                                                                       nodes_dict=nodes_dict)
            edges.extend(doorway_edges)

            current_subgraph = DoorwaySubgraph(room_nodes=room_nodes, 
                                                doorway_nodes=door_nodes, 
                                                left_nodes=left_nodes, 
                                                right_nodes=right_nodes, 
                                                nodes_dict=nodes_dict,
                                                edges=edges)
            subgraphs.append(current_subgraph)
            subgraph_indices.setdefault(doorway.corridor_id, []).append(current_index)
            current_index += 1
        return subgraphs, subgraph_indices
    
    def _extract_drive_through_intersection_nodes(self,
                                                  drive_through: DriveThrough) -> Tuple[List[TraversalNode], List[TraversalNode], 
                                                                                    List[TraversalNode], List[TraversalNode],
                                                                                    List[TraversalNode], List[TraversalNode],
                                                                                    Dict[str, TraversalNode]]:
        entry_nodes = []
        exit_nodes = []
        left_entry_nodes = []
        right_entry_nodes = []
        left_exit_nodes = []
        right_exit_nodes = []
        nodes_dict = {}

        for lane in drive_through.lanes:
            entry_corridor = self._get_corridor_by_id(drive_through.entry_corridor_id)
            entry_point = copy.deepcopy(lane.start_point)
            if entry_corridor.direction == "horizontal":
                possible_orientations = [OrientationVector(0.0, 1.0), OrientationVector(0.0, -1.0)]
                for possible_orientation in possible_orientations:
                    entry_node = TraversalNode(label=f"{entry_point.x:.2f}_{entry_point.y:.2f}_{possible_orientation}",
                                                position=entry_point,
                                                orientation_vec=possible_orientation,
                                                connections=[])
                    nodes_dict[entry_node.label] = entry_node
                    entry_nodes.append(entry_node.label)
                
                for i, corridor_lane in enumerate(entry_corridor.lanes):
                    lane_y = corridor_lane.start_point.y
                    left_lane_x = drive_through.entry_start.x
                    right_lane_x = drive_through.entry_end.x
                    left_node_location = Coordinate(x=left_lane_x, y=lane_y)
                    right_node_location = Coordinate(x=right_lane_x, y=lane_y)
                    if i == 0:
                        orientation_vec = OrientationVector(-1.0, 0.0)  # Facing left
                    elif i == len(entry_corridor.lanes) - 1:
                        orientation_vec = OrientationVector(1.0, 0.0)  # Facing right
                    else:
                        orientation_vec = None
                    if orientation_vec is None:
                        possible_orientations = [OrientationVector(-1.0, 0.0), OrientationVector(1.0, 0.0)]
                        for possible_orientation in possible_orientations:
                            left_node = TraversalNode(label=f"{left_node_location.x:.2f}_{left_node_location.y:.2f}_{possible_orientation}",
                                                        position=left_node_location,
                                                        orientation_vec=possible_orientation,
                                                        connections=[])
                            right_node = TraversalNode(label=f"{right_node_location.x:.2f}_{right_node_location.y:.2f}_{possible_orientation}",
                                                        position=right_node_location,
                                                        orientation_vec=possible_orientation,
                                                        connections=[])
                            nodes_dict[left_node.label] = left_node
                            nodes_dict[right_node.label] = right_node
                            left_entry_nodes.append(left_node.label)
                            right_entry_nodes.append(right_node.label)
                    else:
                        left_node = TraversalNode(label=f"{left_node_location.x:.2f}_{left_node_location.y:.2f}_{orientation_vec}",
                                                    position=left_node_location,
                                                    orientation_vec=orientation_vec,
                                                    connections=[])
                        right_node = TraversalNode(label=f"{right_node_location.x:.2f}_{right_node_location.y:.2f}_{orientation_vec}",
                                                    position=right_node_location,
                                                    orientation_vec=orientation_vec,
                                                    connections=[])
                        nodes_dict[left_node.label] = left_node
                        nodes_dict[right_node.label] = right_node
                        left_entry_nodes.append(left_node.label)
                        right_entry_nodes.append(right_node.label)
            else:
                possible_orientations = [OrientationVector(-1.0, 0.0), OrientationVector(1.0, 0.0)]
                for possible_orientation in possible_orientations:
                    entry_node = TraversalNode(label=f"{entry_point.x:.2f}_{entry_point.y:.2f}_{possible_orientation}",
                                                position=entry_point,
                                                orientation_vec=possible_orientation,
                                                connections=[])
                    nodes_dict[entry_node.label] = entry_node
                    entry_nodes.append(entry_node.label)

                for i, corridor_lane in enumerate(entry_corridor.lanes):
                    lane_x = corridor_lane.start_point.x
                    left_lane_y = drive_through.entry_start.y
                    right_lane_y = drive_through.entry_end.y
                    left_node_location = Coordinate(x=lane_x, y=left_lane_y)
                    right_node_location = Coordinate(x=lane_x, y=right_lane_y)
                    if i == 0:
                        orientation_vec = OrientationVector(0.0, 1.0)  # Facing down
                    elif i == len(entry_corridor.lanes) - 1:
                        orientation_vec = OrientationVector(0.0, -1.0)  # Facing up
                    else:
                        orientation_vec = None
                    if orientation_vec is None:
                        possible_orientations = [OrientationVector(0.0, 1.0), OrientationVector(0.0, -1.0)]
                        for possible_orientation in possible_orientations:
                            left_node = TraversalNode(label=f"{left_node_location.x:.2f}_{left_node_location.y:.2f}_{possible_orientation}",
                                                        position=left_node_location,
                                                        orientation_vec=possible_orientation,
                                                        connections=[])
                            right_node = TraversalNode(label=f"{right_node_location.x:.2f}_{right_node_location.y:.2f}_{possible_orientation}",
                                                        position=right_node_location,
                                                        orientation_vec=possible_orientation,
                                                        connections=[])
                            nodes_dict[left_node.label] = left_node
                            nodes_dict[right_node.label] = right_node
                            left_entry_nodes.append(left_node.label)
                            right_entry_nodes.append(right_node.label)
                    else:
                        left_node = TraversalNode(label=f"{left_node_location.x:.2f}_{left_node_location.y:.2f}_{orientation_vec}",
                                                    position=left_node_location,
                                                    orientation_vec=orientation_vec,
                                                    connections=[])
                        right_node = TraversalNode(label=f"{right_node_location.x:.2f}_{right_node_location.y:.2f}_{orientation_vec}",
                                                    position=right_node_location,
                                                    orientation_vec=orientation_vec,
                                                    connections=[])
                        nodes_dict[left_node.label] = left_node
                        nodes_dict[right_node.label] = right_node
                        left_entry_nodes.append(left_node.label)
                        right_entry_nodes.append(right_node.label)

            exit_corridor = self._get_corridor_by_id(drive_through.exit_corridor_id)
            exit_point = copy.deepcopy(lane.end_point)
            if exit_corridor.direction == "horizontal":
                possible_orientations = [OrientationVector(0.0, 1.0), OrientationVector(0.0, -1.0)]
                for possible_orientation in possible_orientations:
                    exit_node = TraversalNode(label=f"{exit_point.x:.2f}_{exit_point.y:.2f}_{possible_orientation}",
                                                position=exit_point,
                                                orientation_vec=possible_orientation,
                                                connections=[])
                    nodes_dict[exit_node.label] = exit_node
                    exit_nodes.append(exit_node.label)
                
                for i, corridor_lane in enumerate(exit_corridor.lanes):
                    lane_y = corridor_lane.start_point.y
                    left_lane_x = drive_through.exit_start.x
                    right_lane_x = drive_through.exit_end.x
                    left_node_location = Coordinate(x=left_lane_x, y=lane_y)
                    right_node_location = Coordinate(x=right_lane_x, y=lane_y)
                    if i == 0:
                        orientation_vec = OrientationVector(-1.0, 0.0)  # Facing left
                    elif i == len(exit_corridor.lanes) - 1:
                        orientation_vec = OrientationVector(1.0, 0.0)  # Facing right
                    else:
                        orientation_vec = None
                    if orientation_vec is None:
                        possible_orientations = [OrientationVector(-1.0, 0.0), OrientationVector(1.0, 0.0)]
                        for possible_orientation in possible_orientations:
                            left_node = TraversalNode(label=f"{left_node_location.x:.2f}_{left_node_location.y:.2f}_{possible_orientation}",
                                                        position=left_node_location,
                                                        orientation_vec=possible_orientation,
                                                        connections=[])
                            right_node = TraversalNode(label=f"{right_node_location.x:.2f}_{right_node_location.y:.2f}_{possible_orientation}",
                                                        position=right_node_location,
                                                        orientation_vec=possible_orientation,
                                                        connections=[])
                            nodes_dict[left_node.label] = left_node
                            nodes_dict[right_node.label] = right_node
                            left_exit_nodes.append(left_node.label)
                            right_exit_nodes.append(right_node.label)
                    else:
                        left_node = TraversalNode(label=f"{left_node_location.x:.2f}_{left_node_location.y:.2f}_{orientation_vec}",
                                                    position=left_node_location,
                                                    orientation_vec=orientation_vec,
                                                    connections=[])
                        right_node = TraversalNode(label=f"{right_node_location.x:.2f}_{right_node_location.y:.2f}_{orientation_vec}",
                                                    position=right_node_location,
                                                    orientation_vec=orientation_vec,
                                                    connections=[])
                        nodes_dict[left_node.label] = left_node
                        nodes_dict[right_node.label] = right_node
                        left_exit_nodes.append(left_node.label)
                        right_exit_nodes.append(right_node.label)
            else:
                possible_orientations = [OrientationVector(-1.0, 0.0), OrientationVector(1.0, 0.0)]
                for possible_orientation in possible_orientations:
                    exit_node = TraversalNode(label=f"{exit_point.x:.2f}_{exit_point.y:.2f}_{possible_orientation}",
                                                position=exit_point,
                                                orientation_vec=possible_orientation,
                                                connections=[])
                    nodes_dict[exit_node.label] = exit_node
                    exit_nodes.append(exit_node.label)

                for i, corridor_lane in enumerate(exit_corridor.lanes):
                    lane_x = corridor_lane.start_point.x
                    left_lane_y = drive_through.exit_start.y
                    right_lane_y = drive_through.exit_end.y
                    left_node_location = Coordinate(x=lane_x, y=left_lane_y)
                    right_node_location = Coordinate(x=lane_x, y=right_lane_y)
                    if i == 0:
                        orientation_vec = OrientationVector(0.0, 1.0)  # Facing down
                    elif i == len(exit_corridor.lanes) - 1:
                        orientation_vec = OrientationVector(0.0, -1.0)  # Facing up
                    else:
                        orientation_vec = None
                    if orientation_vec is None:
                        possible_orientations = [OrientationVector(0.0, 1.0), OrientationVector(0.0, -1.0)]
                        for possible_orientation in possible_orientations:
                            left_node = TraversalNode(label=f"{left_node_location.x:.2f}_{left_node_location.y:.2f}_{possible_orientation}",
                                                        position=left_node_location,
                                                        orientation_vec=possible_orientation,
                                                        connections=[])
                            right_node = TraversalNode(label=f"{right_node_location.x:.2f}_{right_node_location.y:.2f}_{possible_orientation}",
                                                        position=right_node_location,
                                                        orientation_vec=possible_orientation,
                                                        connections=[])
                            nodes_dict[left_node.label] = left_node
                            nodes_dict[right_node.label] = right_node
                            left_exit_nodes.append(left_node.label)
                            right_exit_nodes.append(right_node.label)
                    else:
                        left_node = TraversalNode(label=f"{left_node_location.x:.2f}_{left_node_location.y:.2f}_{orientation_vec}",
                                                    position=left_node_location,
                                                    orientation_vec=orientation_vec,
                                                    connections=[])
                        right_node = TraversalNode(label=f"{right_node_location.x:.2f}_{right_node_location.y:.2f}_{orientation_vec}",
                                                    position=right_node_location,
                                                    orientation_vec=orientation_vec,
                                                    connections=[])
                        nodes_dict[left_node.label] = left_node
                        nodes_dict[right_node.label] = right_node
                        left_exit_nodes.append(left_node.label)
                        right_exit_nodes.append(right_node.label)

        return entry_nodes, exit_nodes, left_entry_nodes, right_entry_nodes, left_exit_nodes, right_exit_nodes, nodes_dict

    def _create_drive_through_connections_for_nodes(self, 
                                                    drive_through: DriveThrough,
                                                    entry_nodes: List[str],
                                                    exit_nodes: List[str],
                                                    left_entry_nodes: List[str],
                                                    right_entry_nodes: List[str],
                                                    left_exit_nodes: List[str],
                                                    right_exit_nodes: List[str],
                                                    nodes_dict: Dict[str, TraversalNode]) -> List[TraversalEdge]:
        
        edges = []

        entry_corridor = self._get_corridor_by_id(drive_through.entry_corridor_id)
        exit_corridor = self._get_corridor_by_id(drive_through.exit_corridor_id)

        entry_straight_edges = self._create_corridor_straight_connections_for_nodes(corridor=entry_corridor,
                                                                             left_nodes=left_entry_nodes,
                                                                             right_nodes=right_entry_nodes,
                                                                             nodes_dict=nodes_dict)
        edges.extend(entry_straight_edges)

        exit_straight_edges = self._create_corridor_straight_connections_for_nodes(corridor=exit_corridor,
                                                                             left_nodes=left_exit_nodes,
                                                                             right_nodes=right_exit_nodes,
                                                                             nodes_dict=nodes_dict)
        edges.extend(exit_straight_edges)

        if entry_corridor.direction == "horizontal":
            left_orientation_vec = OrientationVector(-1.0, 0.0)
            right_orientation_vec = OrientationVector(1.0, 0.0)
            out_direction_vec = OrientationVector(0.0, -1.0)
            original_ref_direction_vec = OrientationVector(0.0, -1.0)
            inverted_ref_direction_vec = OrientationVector(0.0, 1.0)
        else:
            left_orientation_vec = OrientationVector(0.0, 1.0)
            right_orientation_vec = OrientationVector(0.0, -1.0)
            out_direction_vec = OrientationVector(-1.0, 0.0)
            original_ref_direction_vec = OrientationVector(1.0, 0.0)
            inverted_ref_direction_vec = OrientationVector(-1.0, 0.0)

        original_drive_through_edges = self._create_straight_connections_for_nodes(ref_nodes=entry_nodes,
                                                                         opp_nodes=exit_nodes,
                                                                         ref_direction_vec=original_ref_direction_vec,
                                                                         nodes_dict=nodes_dict)
        edges.extend(original_drive_through_edges)

        inverted_drive_through_edges = self._create_straight_connections_for_nodes(ref_nodes=exit_nodes,
                                                                         opp_nodes=entry_nodes,
                                                                         ref_direction_vec=inverted_ref_direction_vec,
                                                                         nodes_dict=nodes_dict)
        edges.extend(inverted_drive_through_edges)

        for i, entry_node_label in enumerate(entry_nodes):
            exit_node_label = exit_nodes[i]
            entry_node = nodes_dict[entry_node_label]
            exit_node = nodes_dict[exit_node_label]

            for left_entry_node_label in left_entry_nodes:
                left_entry_node = nodes_dict[left_entry_node_label]
                if entry_node.orientation_vec == out_direction_vec:
                    if left_entry_node.orientation_vec == right_orientation_vec:
                        edge = self._create_edge_between_nodes(from_node=left_entry_node,
                                                            to_node=entry_node, 
                                                            action="turn_left")
                        edges.append(edge)
                else:
                    if left_entry_node.orientation_vec == left_orientation_vec:
                        edge = self._create_edge_between_nodes(from_node=entry_node,
                                                                to_node=left_entry_node, 
                                                                action="turn_right")
                        edges.append(edge)

            for left_exit_node_label in left_exit_nodes:
                left_exit_node = nodes_dict[left_exit_node_label]
                if exit_node.orientation_vec == out_direction_vec:
                    if left_exit_node.orientation_vec == left_orientation_vec:
                        edge = self._create_edge_between_nodes(from_node=exit_node,
                                                            to_node=left_exit_node, 
                                                            action="turn_left")
                        edges.append(edge)
                else:
                    if left_exit_node.orientation_vec == right_orientation_vec:
                        edge = self._create_edge_between_nodes(from_node=left_exit_node,
                                                                to_node=exit_node, 
                                                                action="turn_right")
                        edges.append(edge)

            for right_entry_node_label in right_entry_nodes:
                right_entry_node = nodes_dict[right_entry_node_label]
                if entry_node.orientation_vec == out_direction_vec:
                    if right_entry_node.orientation_vec == left_orientation_vec:
                        edge = self._create_edge_between_nodes(from_node=right_entry_node,
                                                            to_node=entry_node, 
                                                            action="turn_right")
                        edges.append(edge)
                else:
                    if right_entry_node.orientation_vec == right_orientation_vec:
                        edge = self._create_edge_between_nodes(from_node=entry_node,
                                                                to_node=right_entry_node, 
                                                                action="turn_left")
                        edges.append(edge)

            for right_exit_node_label in right_exit_nodes:
                right_exit_node = nodes_dict[right_exit_node_label]
                if exit_node.orientation_vec == out_direction_vec:
                    if right_exit_node.orientation_vec == right_orientation_vec:
                        edge = self._create_edge_between_nodes(from_node=exit_node,
                                                            to_node=right_exit_node, 
                                                            action="turn_right")
                        edges.append(edge)
                else:
                    if right_exit_node.orientation_vec == left_orientation_vec:
                        edge = self._create_edge_between_nodes(from_node=right_exit_node,
                                                                to_node=exit_node, 
                                                                action="turn_left")
                        edges.append(edge)
        return edges

    def _generate_drive_through_traversal_subgraphs(self) -> Tuple[List[DriveThroughSubgraph], dict[str, List[int]]]:
        subgraphs = []
        subgraph_indices = {}
        current_index = 0
        for drive_through in self.drive_throughs:
            edges = []
            extraction_result = self._extract_drive_through_intersection_nodes(drive_through=drive_through)
            entry_nodes, exit_nodes, left_entry_nodes, right_entry_nodes, left_exit_nodes, right_exit_nodes, nodes_dict = extraction_result

            drive_through_edges = self._create_drive_through_connections_for_nodes(drive_through=drive_through,
                                                                                   entry_nodes=entry_nodes,
                                                                                   exit_nodes=exit_nodes,
                                                                                   left_entry_nodes=left_entry_nodes,
                                                                                   right_entry_nodes=right_entry_nodes,
                                                                                   left_exit_nodes=left_exit_nodes,
                                                                                   right_exit_nodes=right_exit_nodes,
                                                                                   nodes_dict=nodes_dict)
            edges.extend(drive_through_edges)

            current_subgraph = DriveThroughSubgraph(entry_nodes=entry_nodes,
                                                    entry_corridor_id=drive_through.entry_corridor_id,
                                                    left_entry_nodes=left_entry_nodes,
                                                    right_entry_nodes=right_entry_nodes,
                                                    left_exit_nodes=left_exit_nodes,
                                                    right_exit_nodes=right_exit_nodes,
                                                    exit_nodes=exit_nodes,
                                                    exit_corridor_id=drive_through.exit_corridor_id,
                                                    nodes_dict=nodes_dict,
                                                    edges=edges)
            subgraphs.append(current_subgraph)
            subgraph_indices.setdefault(drive_through.entry_corridor_id, []).append(current_index)
            subgraph_indices.setdefault(drive_through.exit_corridor_id, []).append(current_index)
            current_index += 1
        return subgraphs, subgraph_indices
    
    def _extract_switching_point_nodes(self, 
                                       corridor: Corridor,
                                       center: Coordinate,) -> Tuple[List[str], List[str], Dict[str, TraversalNode]]:
        left_nodes = []
        right_nodes = []
        nodes_dict = {}
        if corridor.direction == "horizontal":
            right_nodes_x = center.x + self.switching_point_offset
            left_nodes_x = center.x - self.switching_point_offset
            for i, lane in enumerate(corridor.lanes):
                lane_y = lane.start_point.y
                left_node_location = Coordinate(x=left_nodes_x, y=lane_y)
                right_node_location = Coordinate(x=right_nodes_x, y=lane_y)
                if i == 0:
                    orientation_vec = OrientationVector(-1.0, 0.0)  # Facing left
                elif i == len(corridor.lanes) - 1:
                    orientation_vec = OrientationVector(1.0, 0.0)  # Facing right
                else:
                    orientation_vec = None
                if orientation_vec is None:
                    possible_orientations = [OrientationVector(-1.0, 0.0), OrientationVector(1.0, 0.0)]
                    for possible_orientation in possible_orientations:
                        left_node = TraversalNode(label=f"{left_node_location.x:.2f}_{left_node_location.y:.2f}_{possible_orientation}",
                                                    position=left_node_location,
                                                    orientation_vec=possible_orientation,
                                                    connections=[])
                        right_node = TraversalNode(label=f"{right_node_location.x:.2f}_{right_node_location.y:.2f}_{possible_orientation}",
                                                    position=right_node_location,
                                                    orientation_vec=possible_orientation,
                                                    connections=[])
                        nodes_dict[left_node.label] = left_node
                        nodes_dict[right_node.label] = right_node
                        left_nodes.append(left_node.label)
                        right_nodes.append(right_node.label)
                else:
                    left_node = TraversalNode(label=f"{left_node_location.x:.2f}_{left_node_location.y:.2f}_{orientation_vec}",
                                                position=left_node_location,
                                                orientation_vec=orientation_vec,
                                                connections=[])
                    right_node = TraversalNode(label=f"{right_node_location.x:.2f}_{right_node_location.y:.2f}_{orientation_vec}",
                                                position=right_node_location,
                                                orientation_vec=orientation_vec,
                                                connections=[])
                    nodes_dict[left_node.label] = left_node
                    nodes_dict[right_node.label] = right_node
                    left_nodes.append(left_node.label)
                    right_nodes.append(right_node.label)
        else:
            right_nodes_y = center.y - self.switching_point_offset
            left_nodes_y = center.y + self.switching_point_offset
            for i, lane in enumerate(corridor.lanes):
                lane_x = lane.start_point.x
                right_node_location = Coordinate(x=lane_x, y=right_nodes_y)
                left_node_location = Coordinate(x=lane_x, y=left_nodes_y)
                if i == 0:
                    orientation_vec = OrientationVector(0.0, 1.0)  # Facing down
                elif i == len(corridor.lanes) - 1:
                    orientation_vec = OrientationVector(0.0, -1.0)  # Facing up
                else:
                    orientation_vec = None
                if orientation_vec is None:
                    possible_orientations = [OrientationVector(0.0, 1.0), OrientationVector(0.0, -1.0)]
                    for possible_orientation in possible_orientations:
                        left_node = TraversalNode(label=f"{left_node_location.x:.2f}_{left_node_location.y:.2f}_{possible_orientation}",
                                                    position=left_node_location,
                                                    orientation_vec=possible_orientation,
                                                    connections=[])
                        right_node = TraversalNode(label=f"{right_node_location.x:.2f}_{right_node_location.y:.2f}_{possible_orientation}",
                                                    position=right_node_location,
                                                    orientation_vec=possible_orientation,
                                                    connections=[])
                        nodes_dict[left_node.label] = left_node
                        nodes_dict[right_node.label] = right_node
                        left_nodes.append(left_node.label)
                        right_nodes.append(right_node.label)
                else:
                    left_node = TraversalNode(label=f"{left_node_location.x:.2f}_{left_node_location.y:.2f}_{orientation_vec}",
                                                position=left_node_location,
                                                orientation_vec=orientation_vec,
                                                connections=[])
                    right_node = TraversalNode(label=f"{right_node_location.x:.2f}_{right_node_location.y:.2f}_{orientation_vec}",
                                                position=right_node_location,
                                                orientation_vec=orientation_vec,
                                                connections=[])
                    nodes_dict[left_node.label] = left_node
                    nodes_dict[right_node.label] = right_node
                    left_nodes.append(left_node.label)
                    right_nodes.append(right_node.label)
        return left_nodes, right_nodes, nodes_dict
    
    def _create_turn_point_connections_for_nodes(self, 
                                                 ref_nodes: List[str],
                                                 ref_direction_vec: Tuple[float, float],
                                                 nodes_dict: Dict[str, TraversalNode]
                                                 ) -> List[TraversalEdge]:
        edges = []

        for ref_node_label in ref_nodes:
            ref_node = nodes_dict[ref_node_label]
            if ref_node.orientation_vec == ref_direction_vec:
                for second_node_label in ref_nodes:
                    second_node = nodes_dict[second_node_label]
                    if second_node.orientation_vec != ref_direction_vec and second_node.position != ref_node.position:
                        edge = self._create_edge_between_nodes(from_node=ref_node,
                                                        to_node=second_node, 
                                                        action="switch_directions")
                        edges.append(edge)

        return edges
    
    def _create_lane_switches_for_nodes(self, 
                                       ref_nodes: List[str], 
                                       opp_nodes: List[str],
                                       ref_direction_vec: Tuple[float, float],
                                       nodes_dict: Dict[str, TraversalNode]
                                       ) -> List[TraversalEdge]:
        edges = []
        for ref_node_label in ref_nodes:
            ref_node = nodes_dict[ref_node_label]
            if ref_node.orientation_vec == ref_direction_vec:
                for opp_node_label in opp_nodes:
                    opp_node = nodes_dict[opp_node_label]
                    if opp_node.orientation_vec == ref_direction_vec:
                        if opp_node.position.x == ref_node.position.x or opp_node.position.y == ref_node.position.y:
                            edge = self._create_edge_between_nodes(from_node=ref_node,
                                                                    to_node=opp_node, 
                                                                    action="go_straight")
                        else:
                            edge = self._create_edge_between_nodes(from_node=ref_node,
                                                            to_node=opp_node, 
                                                            action="switch_lanes")
                        edges.append(edge)
        return edges
    
    def _create_switching_point_connections_for_nodes(self, 
                                                     ref_nodes: List[TraversalNode], 
                                                     opp_nodes: List[TraversalNode],
                                                     ref_direction_vec: Tuple[float, float],
                                                     nodes_dict: Dict[str, TraversalNode]
                                                     ) -> List[TraversalEdge]:
        edges = []
        turn_edges = self._create_turn_point_connections_for_nodes(ref_nodes=ref_nodes,
                                                                   ref_direction_vec=ref_direction_vec,
                                                                   nodes_dict=nodes_dict)
        edges.extend(turn_edges)

        lane_switch_edges = self._create_lane_switches_for_nodes(ref_nodes=ref_nodes,
                                                                 opp_nodes=opp_nodes,
                                                                 ref_direction_vec=ref_direction_vec,
                                                                 nodes_dict=nodes_dict)
        edges.extend(lane_switch_edges)

        return edges

    def _generate_switching_point_subgraph(self, 
                                           center: Coordinate, 
                                           corridor: Corridor) -> SwitchingPointSubgraph:
        edges = []
        left_nodes, right_nodes, nodes_dict = self._extract_switching_point_nodes(corridor=corridor,
                                                                                  center=center)
        
        if corridor.direction == "horizontal":
            original_ref_direction_vec = OrientationVector(1.0, 0.0)  # Facing right
            inverted_ref_direction_vec = OrientationVector(-1.0, 0.0)  # Facing left
        else:
            original_ref_direction_vec = OrientationVector(0.0, -1.0)  # Facing up
            inverted_ref_direction_vec = OrientationVector(0.0, 1.0)  # Facing down

        original_switching_edges = self._create_switching_point_connections_for_nodes(ref_nodes=left_nodes,
                                                                             opp_nodes=right_nodes,
                                                                             ref_direction_vec=original_ref_direction_vec,
                                                                             nodes_dict=nodes_dict)
        edges.extend(original_switching_edges)

        inverted_switching_edges = self._create_switching_point_connections_for_nodes(ref_nodes=right_nodes,
                                                                             opp_nodes=left_nodes,
                                                                             ref_direction_vec=inverted_ref_direction_vec,
                                                                             nodes_dict=nodes_dict)
        edges.extend(inverted_switching_edges)

        switching_point_subgraph = SwitchingPointSubgraph(left_nodes=left_nodes,
                                                          right_nodes=right_nodes,
                                                          nodes_dict=nodes_dict,
                                                          edges=edges)
        return switching_point_subgraph
    
    def _are_doorway_subgraphs_overlapping(self,
                                           direction: str,
                                           subgraph_a: DoorwaySubgraph,
                                           subgraph_b: DoorwaySubgraph) -> bool:
        
        subgraph_a_left_node_label = subgraph_a.left_nodes[0]
        subgraph_a_left_node = subgraph_a.nodes_dict[subgraph_a_left_node_label]
        subgraph_a_right_node_label = subgraph_a.right_nodes[0]
        subgraph_a_right_node = subgraph_a.nodes_dict[subgraph_a_right_node_label]
        subgraph_b_left_node_label = subgraph_b.left_nodes[0]
        subgraph_b_left_node = subgraph_b.nodes_dict[subgraph_b_left_node_label]
        subgraph_b_right_node_label = subgraph_b.right_nodes[0]
        subgraph_b_right_node = subgraph_b.nodes_dict[subgraph_b_right_node_label]
        
        if direction == "horizontal":
            return (subgraph_a_left_node.position.x <= subgraph_b_right_node.position.x and
                    subgraph_a_right_node.position.x >= subgraph_b_left_node.position.x )
        else:
            return (subgraph_a_left_node.position.y >= subgraph_b_right_node.position.y and
                    subgraph_a_right_node.position.y <= subgraph_b_left_node.position.y )
    
    def _replace_node_in_edges(self,
                                subgraph: IntersectionSubgraph | DoorwaySubgraph | DriveThroughSubgraph,
                                old_node_label: str,
                                new_node: TraversalNode):

        for i, traversal_edge in enumerate(subgraph.edges):
            if traversal_edge.from_node == old_node_label:
                new_edge = self._create_edge_between_nodes(from_node=new_node,
                                                           to_node=subgraph.nodes_dict[traversal_edge.to_node],
                                                           action=traversal_edge.action)
                subgraph.edges[i] = new_edge

            elif traversal_edge.to_node == old_node_label:
                origin_node = subgraph.nodes_dict.get(traversal_edge.from_node)
                assert origin_node is not None
                origin_node.connections.remove(old_node_label)
                new_edge = self._create_edge_between_nodes(from_node=origin_node,
                                                           to_node=new_node,
                                                           action=traversal_edge.action)
                subgraph.edges[i] = new_edge

    def _replace_node_in_doorway_subgraph(self,
                                          subgraph: DoorwaySubgraph,
                                          old_node_label: str,
                                          new_node: TraversalNode):
        
        self._replace_node_in_edges(subgraph, old_node_label, new_node)

        for i in range(len(subgraph.left_nodes)):
            if subgraph.left_nodes[i] == old_node_label:
                subgraph.left_nodes[i] = new_node.label
            elif subgraph.right_nodes[i] == old_node_label:
                subgraph.right_nodes[i] = new_node.label
        
        subgraph.nodes_dict.pop(old_node_label)
        subgraph.nodes_dict[new_node.label] = new_node
        

    def _merge_doorway_subgraphs(self,
                                 direction: str,
                                 subgraph_a: DoorwaySubgraph,
                                 subgraph_b: DoorwaySubgraph):
        
        for i in range(len(subgraph_b.left_nodes)):
            subgraph_a_left_node_label = subgraph_a.left_nodes[i]
            subgraph_a_left_node = subgraph_a.nodes_dict[subgraph_a_left_node_label]
            subgraph_b_left_node_label = subgraph_b.left_nodes[i]
            subgraph_b_left_node = subgraph_b.nodes_dict[subgraph_b_left_node_label]
            subgraph_a_right_node_label = subgraph_a.right_nodes[i]
            subgraph_a_right_node = subgraph_a.nodes_dict[subgraph_a_right_node_label]
            subgraph_b_right_node_label = subgraph_b.right_nodes[i]
            subgraph_b_right_node = subgraph_b.nodes_dict[subgraph_b_right_node_label]

            if direction == "horizontal":
                new_left_x = min(subgraph_a_left_node.position.x, subgraph_b_left_node.position.x)
                y_position = subgraph_a_left_node.position.y

                new_left_node = TraversalNode(label=f"{new_left_x:.2f}_{y_position:.2f}_{subgraph_a_left_node.orientation_vec}",
                                                position=Coordinate(x=new_left_x, y=y_position),
                                                orientation_vec=subgraph_a_left_node.orientation_vec,
                                                connections=[])

                new_right_x = max(subgraph_a_right_node.position.x, subgraph_b_right_node.position.x)
                new_right_node = TraversalNode(label=f"{new_right_x:.2f}_{y_position:.2f}_{subgraph_a_right_node.orientation_vec}",
                                                position=Coordinate(x=new_right_x, y=y_position),
                                                orientation_vec=subgraph_a_right_node.orientation_vec,
                                                connections=[])
            else:
                new_left_y = max(subgraph_a_left_node.position.y, subgraph_b_left_node.position.y)
                x_position = subgraph_a_left_node.position.x

                new_left_node = TraversalNode(label=f"{x_position:.2f}_{new_left_y:.2f}_{subgraph_a_left_node.orientation_vec}",
                                                position=Coordinate(x=x_position, y=new_left_y),
                                                orientation_vec=subgraph_a_left_node.orientation_vec,
                                                connections=[])

                new_right_y = min(subgraph_a_right_node.position.y, subgraph_b_right_node.position.y)
                new_right_node = TraversalNode(label=f"{x_position:.2f}_{new_right_y:.2f}_{subgraph_a_right_node.orientation_vec}",
                                                position=Coordinate(x=x_position, y=new_right_y),
                                                orientation_vec=subgraph_a_right_node.orientation_vec,
                                                connections=[])
                
            self._replace_node_in_doorway_subgraph(subgraph=subgraph_a,
                                                   old_node_label=subgraph_a_left_node_label,
                                                   new_node=new_left_node)
            self._replace_node_in_doorway_subgraph(subgraph=subgraph_b,
                                                    old_node_label=subgraph_b_left_node_label,
                                                    new_node=new_left_node)
            self._replace_node_in_doorway_subgraph(subgraph=subgraph_a,
                                                    old_node_label=subgraph_a_right_node_label,
                                                    new_node=new_right_node)
            self._replace_node_in_doorway_subgraph(subgraph=subgraph_b,
                                                    old_node_label=subgraph_b_right_node_label,
                                                    new_node=new_right_node)
            

    def _merge_overlapping_doorway_subgraphs(self):

        # TODO: only works when only two structures are overlapping. Generalize this to more structures later by 
        # grouping overlapped subgraphs and merging them all together

        for corridor_id in self.doorway_subgraph_indices.keys():
            subgraph_indices = self.doorway_subgraph_indices[corridor_id]
            current_corridor = self._get_corridor_by_id(corridor_id)
            current_direction = current_corridor.direction

            if len(subgraph_indices) > 1:
                for i, subgraph_index in enumerate(subgraph_indices):
                    current_subgraph = self.doorway_subgraphs[subgraph_index]
                    for j in range(i + 1, len(subgraph_indices)):
                        compare_subgraph_index = subgraph_indices[j]
                        compare_subgraph = self.doorway_subgraphs[compare_subgraph_index]

                        if self._are_doorway_subgraphs_overlapping(direction=current_direction,
                                                                   subgraph_a=current_subgraph,
                                                                   subgraph_b=compare_subgraph):
                            self._merge_doorway_subgraphs(direction=current_direction,
                                                          subgraph_a=current_subgraph,
                                                          subgraph_b=compare_subgraph)
    def _are_drive_through_and_doorway_subgraphs_overlapping(self,
                                                           direction: str,
                                                           doorway_subgraph: DoorwaySubgraph,
                                                           drive_through_subgraph: DriveThroughSubgraph,
                                                           exit_flag: bool) -> bool:
        
        doorway_left_node_label = doorway_subgraph.left_nodes[0]
        doorway_left_node = doorway_subgraph.nodes_dict[doorway_left_node_label]
        doorway_right_node_label = doorway_subgraph.right_nodes[0]
        doorway_right_node = doorway_subgraph.nodes_dict[doorway_right_node_label]

        if exit_flag:
            drive_through_left_node_label = drive_through_subgraph.left_exit_nodes[0]
            drive_through_right_node_label = drive_through_subgraph.right_exit_nodes[0]
        else:
            drive_through_left_node_label = drive_through_subgraph.left_entry_nodes[0]
            drive_through_right_node_label = drive_through_subgraph.right_entry_nodes[0]
        
        drive_through_left_node = drive_through_subgraph.nodes_dict[drive_through_left_node_label]
        drive_through_right_node = drive_through_subgraph.nodes_dict[drive_through_right_node_label]

        if direction == "horizontal":
            return (doorway_left_node.position.x <= drive_through_right_node.position.x and
                    doorway_right_node.position.x >= drive_through_left_node.position.x )
        else:
            return (doorway_left_node.position.y >= drive_through_right_node.position.y and
                    doorway_right_node.position.y <= drive_through_left_node.position.y )
    
    def _replace_node_in_drive_through_subgraph(self,
                                                subgraph: DriveThroughSubgraph,
                                                old_node_label: str,
                                                new_node: TraversalNode,
                                                exit_flag: bool) -> None:
        
        self._replace_node_in_edges(subgraph, old_node_label, new_node)
        
        if exit_flag:
            for i in range(len(subgraph.left_exit_nodes)):
                if subgraph.left_exit_nodes[i] == old_node_label:
                    subgraph.left_exit_nodes[i] = new_node.label
                elif subgraph.right_exit_nodes[i] == old_node_label:
                    subgraph.right_exit_nodes[i] = new_node.label
        else:
            for i in range(len(subgraph.left_entry_nodes)):
                if subgraph.left_entry_nodes[i] == old_node_label:
                    subgraph.left_entry_nodes[i] = new_node.label
                elif subgraph.right_entry_nodes[i] == old_node_label:
                    subgraph.right_entry_nodes[i] = new_node.label
        
        subgraph.nodes_dict.pop(old_node_label)
        subgraph.nodes_dict[new_node.label] = new_node

    def _merge_drive_through_and_doorway_subgraphs(self,
                                                   direction: str,
                                                   doorway_subgraph: DoorwaySubgraph,
                                                   drive_through_subgraph: DriveThroughSubgraph,
                                                   exit_flag: bool):
        if exit_flag:
            for i in range(len(drive_through_subgraph.left_exit_nodes)):
                doorway_left_node_label = doorway_subgraph.left_nodes[i]
                doorway_left_node = doorway_subgraph.nodes_dict[doorway_left_node_label]
                drive_through_left_node_label = drive_through_subgraph.left_exit_nodes[i]
                drive_through_left_node = drive_through_subgraph.nodes_dict[drive_through_left_node_label]

                doorway_right_node_label = doorway_subgraph.right_nodes[i]
                doorway_right_node = doorway_subgraph.nodes_dict[doorway_right_node_label]
                drive_through_right_node_label = drive_through_subgraph.right_exit_nodes[i]
                drive_through_right_node = drive_through_subgraph.nodes_dict[drive_through_right_node_label]

                if direction == "horizontal":
                    new_left_x = min(doorway_left_node.position.x, drive_through_left_node.position.x)
                    y_position = doorway_left_node.position.y
                    new_left_node = TraversalNode(label=f"{new_left_x:.2f}_{y_position:.2f}_{doorway_left_node.orientation_vec}",
                                             position=Coordinate(x=new_left_x, y=y_position),
                                             orientation_vec=doorway_left_node.orientation_vec,
                                             connections=[])
                    new_right_x = max(doorway_right_node.position.x, drive_through_right_node.position.x)
                    new_right_node = TraversalNode(label=f"{new_right_x:.2f}_{y_position:.2f}_{doorway_right_node.orientation_vec}",
                                                  position=Coordinate(x=new_right_x, y=y_position),
                                                  orientation_vec=doorway_right_node.orientation_vec,
                                                  connections=[])
                else:
                    new_left_y = max(doorway_left_node.position.y, drive_through_left_node.position.y)
                    x_position = doorway_left_node.position.x
                    new_left_node = TraversalNode(label=f"{x_position:.2f}_{new_left_y:.2f}_{doorway_left_node.orientation_vec}",
                                             position=Coordinate(x=x_position, y=new_left_y),
                                             orientation_vec=doorway_left_node.orientation_vec,
                                             connections=[])
                    new_right_y = min(doorway_right_node.position.y, drive_through_right_node.position.y)
                    new_right_node = TraversalNode(label=f"{x_position:.2f}_{new_right_y:.2f}_{doorway_right_node.orientation_vec}",
                                                  position=Coordinate(x=x_position, y=new_right_y),
                                                  orientation_vec=doorway_right_node.orientation_vec,
                                                  connections=[])

                self._replace_node_in_doorway_subgraph(subgraph=doorway_subgraph,
                                                       old_node_label=doorway_left_node_label,
                                                       new_node=new_left_node)
                self._replace_node_in_doorway_subgraph(subgraph=doorway_subgraph,
                                                       old_node_label=doorway_right_node_label,
                                                       new_node=new_right_node)
                
                self._replace_node_in_drive_through_subgraph(subgraph=drive_through_subgraph,
                                                         old_node_label=drive_through_left_node_label,
                                                         new_node=new_left_node,
                                                         exit_flag=exit_flag)

                self._replace_node_in_drive_through_subgraph(subgraph=drive_through_subgraph,
                                                         old_node_label=drive_through_right_node_label,
                                                         new_node=new_right_node,
                                                         exit_flag=exit_flag)
        else:
            for i in range(len(drive_through_subgraph.left_entry_nodes)):
                doorway_left_node_label = doorway_subgraph.left_nodes[i]
                doorway_left_node = doorway_subgraph.nodes_dict[doorway_left_node_label]
                drive_through_left_node_label = drive_through_subgraph.left_entry_nodes[i]
                drive_through_left_node = drive_through_subgraph.nodes_dict[drive_through_left_node_label]

                doorway_right_node_label = doorway_subgraph.right_nodes[i]
                doorway_right_node = doorway_subgraph.nodes_dict[doorway_right_node_label]
                drive_through_right_node_label = drive_through_subgraph.right_entry_nodes[i]
                drive_through_right_node = drive_through_subgraph.nodes_dict[drive_through_right_node_label]

                if direction == "horizontal":
                    new_left_x = min(doorway_left_node.position.x, drive_through_left_node.position.x)
                    y_position = doorway_left_node.position.y
                    new_left_node = TraversalNode(label=f"{new_left_x:.2f}_{y_position:.2f}_{doorway_left_node.orientation_vec}",
                                             position=Coordinate(x=new_left_x, y=y_position),
                                             orientation_vec=doorway_left_node.orientation_vec,
                                             connections=[])
                    new_right_x = max(doorway_right_node.position.x, drive_through_right_node.position.x)
                    new_right_node = TraversalNode(label=f"{new_right_x:.2f}_{y_position:.2f}_{doorway_right_node.orientation_vec}",
                                                  position=Coordinate(x=new_right_x, y=y_position),
                                                  orientation_vec=doorway_right_node.orientation_vec,
                                                  connections=[])
                else:
                    new_left_y = max(doorway_left_node.position.y, drive_through_left_node.position.y)
                    x_position = doorway_left_node.position.x
                    new_left_node = TraversalNode(label=f"{x_position:.2f}_{new_left_y:.2f}_{doorway_left_node.orientation_vec}",
                                             position=Coordinate(x=x_position, y=new_left_y),
                                             orientation_vec=doorway_left_node.orientation_vec,
                                             connections=[])
                    new_right_y = min(doorway_right_node.position.y, drive_through_right_node.position.y)
                    new_right_node = TraversalNode(label=f"{x_position:.2f}_{new_right_y:.2f}_{doorway_right_node.orientation_vec}",
                                                  position=Coordinate(x=x_position, y=new_right_y),
                                                  orientation_vec=doorway_right_node.orientation_vec,
                                                  connections=[])
                
                self._replace_node_in_doorway_subgraph(subgraph=doorway_subgraph,
                                                       old_node_label=doorway_left_node_label,
                                                       new_node=new_left_node)
                self._replace_node_in_doorway_subgraph(subgraph=doorway_subgraph,
                                                       old_node_label=doorway_right_node_label,
                                                       new_node=new_right_node)
                
                self._replace_node_in_drive_through_subgraph(subgraph=drive_through_subgraph,
                                                         old_node_label=drive_through_left_node_label,
                                                         new_node=new_left_node,
                                                         exit_flag=exit_flag)

                self._replace_node_in_drive_through_subgraph(subgraph=drive_through_subgraph,
                                                         old_node_label=drive_through_right_node_label,
                                                         new_node=new_right_node,
                                                         exit_flag=exit_flag)

    def _merge_overlapping_drive_through_doorway_subgraphs(self):

        for corridor_id in self.doorway_subgraph_indices.keys():
            doorway_subgraph_indices = self.doorway_subgraph_indices[corridor_id]
            drive_through_subgraph_indices = self.drive_through_subgraph_indices.get(corridor_id, [])
            current_corridor = self._get_corridor_by_id(corridor_id)
            current_direction = current_corridor.direction

            if len(doorway_subgraph_indices) > 0 and len(drive_through_subgraph_indices) > 0:
                for j in range(len(drive_through_subgraph_indices)):
                    drive_through_subgraph_index = drive_through_subgraph_indices[j]
                    compare_drive_through_subgraph = self.drive_through_subgraphs[drive_through_subgraph_index]

                    if compare_drive_through_subgraph.entry_corridor_id == corridor_id:
                        exit_flag = False
                    else:
                        exit_flag = True
                    for doorway_subgraph_index in doorway_subgraph_indices:
                        current_doorway_subgraph = self.doorway_subgraphs[doorway_subgraph_index]

                        if self._are_drive_through_and_doorway_subgraphs_overlapping(direction=current_direction,
                                                                                     doorway_subgraph=current_doorway_subgraph,
                                                                                     drive_through_subgraph=compare_drive_through_subgraph,
                                                                                     exit_flag=exit_flag):
                            self._merge_drive_through_and_doorway_subgraphs(direction=current_direction,
                                                                            doorway_subgraph=current_doorway_subgraph,
                                                                            drive_through_subgraph=compare_drive_through_subgraph,
                                                                            exit_flag=exit_flag)
    def _are_intersection_and_doorway_subgraphs_overlapping(self,
                                                            direction: str,
                                                            doorway_subgraph: DoorwaySubgraph,
                                                            intersection_subgraph: IntersectionSubgraph) -> bool:
        doorway_left_node_label = doorway_subgraph.left_nodes[0]
        doorway_left_node = doorway_subgraph.nodes_dict[doorway_left_node_label]
        doorway_right_node_label = doorway_subgraph.right_nodes[0]
        doorway_right_node = doorway_subgraph.nodes_dict[doorway_right_node_label]

        if direction == "horizontal":
            if intersection_subgraph.left_nodes and intersection_subgraph.right_nodes:
                intersection_left_node_label = intersection_subgraph.left_nodes[0]
                intersection_left_node = intersection_subgraph.nodes_dict[intersection_left_node_label]
                intersection_right_node_label = intersection_subgraph.right_nodes[0]
                intersection_right_node = intersection_subgraph.nodes_dict[intersection_right_node_label]

                return (doorway_left_node.position.x <= intersection_right_node.position.x and
                        doorway_right_node.position.x >= intersection_left_node.position.x )
            else:
                return False
        else:
            if intersection_subgraph.upper_nodes and intersection_subgraph.lower_nodes:
                intersection_upper_node_label = intersection_subgraph.upper_nodes[0]
                intersection_upper_node = intersection_subgraph.nodes_dict[intersection_upper_node_label]
                intersection_lower_node_label = intersection_subgraph.lower_nodes[0]
                intersection_lower_node = intersection_subgraph.nodes_dict[intersection_lower_node_label]

                return (doorway_left_node.position.y <= intersection_upper_node.position.y and
                        doorway_right_node.position.y >= intersection_lower_node.position.y )
            else:
                return False

    def _replace_node_in_intersection_subgraph(self,
                                               direction: str,
                                               subgraph: IntersectionSubgraph,
                                               old_node_label: str,
                                               new_node: TraversalNode):
        
        self._replace_node_in_edges(subgraph, old_node_label, new_node)

        if direction == "horizontal":
            for i in range(len(subgraph.left_nodes)):
                if subgraph.left_nodes[i] == old_node_label:
                    subgraph.left_nodes[i] = new_node.label
                elif subgraph.right_nodes[i] == old_node_label:
                    subgraph.right_nodes[i] = new_node.label
        else:
            for i in range(len(subgraph.upper_nodes)):
                if subgraph.upper_nodes[i] == old_node_label:
                    subgraph.upper_nodes[i] = new_node.label
                elif subgraph.lower_nodes[i] == old_node_label:
                    subgraph.lower_nodes[i] = new_node.label
        
        subgraph.nodes_dict.pop(old_node_label)
        subgraph.nodes_dict[new_node.label] = new_node

                            

    def _merge_intersections_and_doorway_subgraphs(self,
                                                   direction: str,
                                                   doorway_subgraph: DoorwaySubgraph,
                                                   intersection_subgraph: IntersectionSubgraph):
        if direction == "horizontal":
            for i in range(len(intersection_subgraph.left_nodes)):
                doorway_left_node_label = doorway_subgraph.left_nodes[i]
                doorway_left_node = doorway_subgraph.nodes_dict[doorway_left_node_label]
                intersection_left_node_label = intersection_subgraph.left_nodes[i]
                intersection_left_node = intersection_subgraph.nodes_dict[intersection_left_node_label]

                doorway_right_node_label = doorway_subgraph.right_nodes[i]
                doorway_right_node = doorway_subgraph.nodes_dict[doorway_right_node_label]
                intersection_right_node_label = intersection_subgraph.right_nodes[i]
                intersection_right_node = intersection_subgraph.nodes_dict[intersection_right_node_label]

                new_left_x = min(doorway_left_node.position.x, intersection_left_node.position.x)
                y_position = doorway_left_node.position.y
                new_left_node = TraversalNode(label=f"{new_left_x:.2f}_{y_position:.2f}_{doorway_left_node.orientation_vec}",
                                             position=Coordinate(x=new_left_x, y=y_position),
                                             orientation_vec=doorway_left_node.orientation_vec,
                                             connections=[])

                new_right_x = max(doorway_right_node.position.x, intersection_right_node.position.x)
                new_right_node = TraversalNode(label=f"{new_right_x:.2f}_{y_position:.2f}_{doorway_right_node.orientation_vec}",
                                              position=Coordinate(x=new_right_x, y=y_position),
                                              orientation_vec=doorway_right_node.orientation_vec,
                                              connections=[])

                self._replace_node_in_doorway_subgraph(subgraph=doorway_subgraph,
                                                       old_node_label=doorway_left_node_label,
                                                       new_node=new_left_node)
                self._replace_node_in_intersection_subgraph(direction=direction,
                                                            subgraph=intersection_subgraph,
                                                            old_node_label=intersection_left_node_label,
                                                            new_node=new_left_node)

                self._replace_node_in_doorway_subgraph(subgraph=doorway_subgraph,
                                                       old_node_label=doorway_right_node_label,
                                                       new_node=new_right_node)
                self._replace_node_in_intersection_subgraph(direction=direction,
                                                            subgraph=intersection_subgraph,
                                                            old_node_label=intersection_right_node_label,
                                                            new_node=new_right_node)
        else:
            for i in range(len(intersection_subgraph.upper_nodes)):
                doorway_left_node_label = doorway_subgraph.left_nodes[i]
                doorway_left_node = doorway_subgraph.nodes_dict[doorway_left_node_label]
                intersection_upper_node_label = intersection_subgraph.upper_nodes[i]
                intersection_upper_node = intersection_subgraph.nodes_dict[intersection_upper_node_label]

                doorway_right_node_label = doorway_subgraph.right_nodes[i]
                doorway_right_node = doorway_subgraph.nodes_dict[doorway_right_node_label]
                intersection_lower_node_label = intersection_subgraph.lower_nodes[i]
                intersection_lower_node = intersection_subgraph.nodes_dict[intersection_lower_node_label]

                new_right_y = max(doorway_right_node.position.y, intersection_upper_node.position.y)
                x_position = doorway_right_node.position.x
                new_right_node = TraversalNode(label=f"{x_position:.2f}_{new_right_y:.2f}_{doorway_left_node.orientation_vec}",
                                             position=Coordinate(x=x_position, y=new_right_y),
                                             orientation_vec=doorway_right_node.orientation_vec,
                                             connections=[])

                new_left_y = min(doorway_left_node.position.y, intersection_lower_node.position.y)
                new_left_node = TraversalNode(label=f"{x_position:.2f}_{new_left_y:.2f}_{doorway_left_node.orientation_vec}",
                                              position=Coordinate(x=x_position, y=new_left_y),
                                              orientation_vec=doorway_left_node.orientation_vec,
                                              connections=[])

                self._replace_node_in_doorway_subgraph(subgraph=doorway_subgraph,
                                                       old_node_label=doorway_left_node_label,
                                                       new_node=new_left_node)
                self._replace_node_in_intersection_subgraph(direction=direction,
                                                            subgraph=intersection_subgraph,
                                                            old_node_label=intersection_upper_node_label,
                                                            new_node=new_right_node)

                self._replace_node_in_doorway_subgraph(subgraph=doorway_subgraph,
                                                       old_node_label=doorway_right_node_label,
                                                       new_node=new_right_node)
                self._replace_node_in_intersection_subgraph(direction=direction,
                                                            subgraph=intersection_subgraph,
                                                            old_node_label=intersection_lower_node_label,
                                                            new_node=new_left_node)
    
    def _merge_overlapping_intersections_doorway_subgraphs(self):

        for corridor_id in self.doorway_subgraph_indices.keys():
            doorway_subgraph_indices = self.doorway_subgraph_indices[corridor_id]
            intersection_subgraph_indices = self.corridor_intersection_subgraph_indices.get(corridor_id, [])
            current_corridor = self._get_corridor_by_id(corridor_id)
            current_direction = current_corridor.direction

            if len(doorway_subgraph_indices) > 0 and len(intersection_subgraph_indices) > 0:
                for j in range(len(intersection_subgraph_indices)):
                    intersection_subgraph_index = intersection_subgraph_indices[j]
                    compare_intersection_subgraph = self.corridor_intersection_subgraphs[intersection_subgraph_index]

                    for doorway_subgraph_index in doorway_subgraph_indices:
                        current_doorway_subgraph = self.doorway_subgraphs[doorway_subgraph_index]

                        if self._are_intersection_and_doorway_subgraphs_overlapping(direction=current_direction,
                                                                                   doorway_subgraph=current_doorway_subgraph,
                                                                                   intersection_subgraph=compare_intersection_subgraph):
                            self._merge_intersections_and_doorway_subgraphs(direction=current_direction,
                                                                            doorway_subgraph=current_doorway_subgraph,
                                                                            intersection_subgraph=compare_intersection_subgraph)
    
    def _get_subgraphs_for_corridor(self,
                                    corridor_id: str) -> List[Union[IntersectionSubgraph, DoorwaySubgraph, DriveThroughSubgraph]]:
        sub_graphs = []
        doorway_subgraph_indices = self.doorway_subgraph_indices[corridor_id]
        intersection_subgraph_indices = self.corridor_intersection_subgraph_indices.get(corridor_id, [])
        drive_through_subgraph_indices = self.drive_through_subgraph_indices.get(corridor_id, [])

        if len(doorway_subgraph_indices) > 0:
            for doorway_subgraph_index in doorway_subgraph_indices:
                current_doorway_subgraph = self.doorway_subgraphs[doorway_subgraph_index]
                sub_graphs.append(current_doorway_subgraph)

        if len(intersection_subgraph_indices) > 0:
            for intersection_subgraph_index in intersection_subgraph_indices:
                current_intersection_subgraph = self.corridor_intersection_subgraphs[intersection_subgraph_index]
                sub_graphs.append(current_intersection_subgraph)

        if len(drive_through_subgraph_indices) > 0:
            for drive_through_subgraph_index in drive_through_subgraph_indices:
                current_drive_through_subgraph = self.drive_through_subgraphs[drive_through_subgraph_index]
                sub_graphs.append(current_drive_through_subgraph)
        
        return sub_graphs
    
    def _subgraph_sort_key(self,
                           corridor_id: str,
                           direction: str,
                           subgraph: Union[IntersectionSubgraph, DoorwaySubgraph, DriveThroughSubgraph]) -> float:
        if direction == "horizontal":
            if isinstance(subgraph, DriveThroughSubgraph):
                if corridor_id in subgraph.entry_corridor_id:
                    if subgraph.right_entry_nodes:
                        return subgraph.nodes_dict[subgraph.right_entry_nodes[0]].position.x
                    elif subgraph.left_entry_nodes:
                        return subgraph.nodes_dict[subgraph.left_entry_nodes[0]].position.x
                else:
                    if subgraph.right_exit_nodes:
                        return subgraph.nodes_dict[subgraph.right_exit_nodes[0]].position.x
                    elif subgraph.left_exit_nodes:
                        return subgraph.nodes_dict[subgraph.left_exit_nodes[0]].position.x
            elif isinstance(subgraph, DoorwaySubgraph):
                if subgraph.right_nodes:
                    return subgraph.nodes_dict[subgraph.right_nodes[0]].position.x
                elif subgraph.left_nodes:
                    return subgraph.nodes_dict[subgraph.left_nodes[0]].position.x
            elif isinstance(subgraph, IntersectionSubgraph):
                if subgraph.right_nodes:
                    return subgraph.nodes_dict[subgraph.right_nodes[0]].position.x
                elif subgraph.left_nodes:
                    return subgraph.nodes_dict[subgraph.left_nodes[0]].position.x
            else:
                raise ValueError("Invalid subgraph type for sorting.")
        else:
            if isinstance(subgraph, DriveThroughSubgraph):
                if corridor_id in subgraph.entry_corridor_id:
                    if subgraph.left_entry_nodes:
                        return subgraph.nodes_dict[subgraph.left_entry_nodes[0]].position.y
                    elif subgraph.right_entry_nodes:
                        return subgraph.nodes_dict[subgraph.right_entry_nodes[0]].position.y
                else:
                    if subgraph.left_exit_nodes:
                        return subgraph.nodes_dict[subgraph.left_exit_nodes[0]].position.y
                    elif subgraph.right_exit_nodes:
                        return subgraph.nodes_dict[subgraph.right_exit_nodes[0]].position.y
            elif isinstance(subgraph, DoorwaySubgraph):
                if subgraph.left_nodes:
                    return subgraph.nodes_dict[subgraph.left_nodes[0]].position.y
                elif subgraph.right_nodes:
                    return subgraph.nodes_dict[subgraph.right_nodes[0]].position.y
            elif isinstance(subgraph, IntersectionSubgraph):
                if subgraph.lower_nodes:
                    return subgraph.nodes_dict[subgraph.lower_nodes[0]].position.y
                elif subgraph.upper_nodes:
                    return subgraph.nodes_dict[subgraph.upper_nodes[0]].position.y
            else:
                raise ValueError("Invalid subgraph type for sorting.")

    def _sort_subgraphs(self,
                        corridor_id: str,
                        subgraphs: List[Union[IntersectionSubgraph, DoorwaySubgraph, DriveThroughSubgraph]],
                        direction: str) -> List[Union[IntersectionSubgraph, DoorwaySubgraph, DriveThroughSubgraph]]:
        if direction == "horizontal":
            sorted_subgraphs = sorted(subgraphs,
                                      key=lambda sg: self._subgraph_sort_key(corridor_id, direction, sg))
        else:
            sorted_subgraphs = sorted(subgraphs,
                                      key=lambda sg: self._subgraph_sort_key(corridor_id, direction, sg))
        return sorted_subgraphs
    
    def _subgraph_repr_key(self,
                           corridor_id: str,
                           direction: str,
                           subgraph: Union[IntersectionSubgraph, DoorwaySubgraph, DriveThroughSubgraph]) -> float:
        if direction == "horizontal":
            if isinstance(subgraph, DriveThroughSubgraph):
                if corridor_id in subgraph.entry_corridor_id:
                    if subgraph.right_entry_nodes:
                        return subgraph.nodes_dict[subgraph.right_entry_nodes[0]].label
                    elif subgraph.left_entry_nodes:
                        return subgraph.nodes_dict[subgraph.left_entry_nodes[0]].label
                else:
                    if subgraph.right_exit_nodes:
                        return subgraph.nodes_dict[subgraph.right_exit_nodes[0]].label
                    elif subgraph.left_exit_nodes:
                        return subgraph.nodes_dict[subgraph.left_exit_nodes[0]].label
            elif isinstance(subgraph, DoorwaySubgraph):
                if subgraph.right_nodes:
                    return subgraph.nodes_dict[subgraph.right_nodes[0]].label
                elif subgraph.left_nodes:
                    return subgraph.nodes_dict[subgraph.left_nodes[0]].label
            elif isinstance(subgraph, IntersectionSubgraph):
                if subgraph.right_nodes:
                    return subgraph.nodes_dict[subgraph.right_nodes[0]].label
                elif subgraph.left_nodes:
                    return subgraph.nodes_dict[subgraph.left_nodes[0]].label
            else:
                raise ValueError("Invalid subgraph type for sorting.")
        else:
            if isinstance(subgraph, DriveThroughSubgraph):
                if corridor_id in subgraph.entry_corridor_id:
                    if subgraph.left_entry_nodes:
                        return subgraph.nodes_dict[subgraph.left_entry_nodes[0]].label
                    elif subgraph.right_entry_nodes:
                        return subgraph.nodes_dict[subgraph.right_entry_nodes[0]].label
                else:
                    if subgraph.left_exit_nodes:
                        return subgraph.nodes_dict[subgraph.left_exit_nodes[0]].label
                    elif subgraph.right_exit_nodes:
                        return subgraph.nodes_dict[subgraph.right_exit_nodes[0]].label
            elif isinstance(subgraph, DoorwaySubgraph):
                if subgraph.left_nodes:
                    return subgraph.nodes_dict[subgraph.left_nodes[0]].label
                elif subgraph.right_nodes:
                    return subgraph.nodes_dict[subgraph.right_nodes[0]].label
            elif isinstance(subgraph, IntersectionSubgraph):
                if subgraph.lower_nodes:
                    return subgraph.nodes_dict[subgraph.lower_nodes[0]].label
                elif subgraph.upper_nodes:
                    return subgraph.nodes_dict[subgraph.upper_nodes[0]].label
            else:
                raise ValueError("Invalid subgraph type for sorting.")
    
    def _group_subgraphs_with_shared_nodes(self,
                                           corridor_id: str,
                                           direction: str,
                                           subgraphs: List[Union[IntersectionSubgraph, 
                                                                  DoorwaySubgraph, 
                                                                  DriveThroughSubgraph]]) -> Dict[str, List[Union[IntersectionSubgraph, 
                                                                                                                  DoorwaySubgraph, 
                                                                                                                  DriveThroughSubgraph]]]:
        node_to_subgraphs: Dict[str, List[Union[IntersectionSubgraph, 
                                               DoorwaySubgraph, 
                                               DriveThroughSubgraph]]] = {}
        
        for subgraph in subgraphs:
            subgraph_key = self._subgraph_repr_key(corridor_id=corridor_id, 
                                                   direction=direction,
                                                   subgraph=subgraph)
            if subgraph_key not in node_to_subgraphs:
                node_to_subgraphs[subgraph_key] = []
            node_to_subgraphs[subgraph_key].append(subgraph)
        
        return node_to_subgraphs
    
    def _obtain_representative_nodes_from_subgraphs(self,
                                                    current_direction: str,
                                                    current_subgraph: Union[IntersectionSubgraph, 
                                                                            DoorwaySubgraph, 
                                                                            DriveThroughSubgraph],
                                                    next_subgraph: Union[IntersectionSubgraph, 
                                                                         DoorwaySubgraph, 
                                                                         DriveThroughSubgraph],
                                                    corridor_id: str) -> Tuple[List[str], List[str]]:
        
        current_nodes_labels = []
        next_nodes_labels = []

        if current_direction == "horizontal":
            if isinstance(current_subgraph, DriveThroughSubgraph):
                if corridor_id in current_subgraph.entry_corridor_id:
                    current_nodes_labels = current_subgraph.right_entry_nodes
                else:
                    current_nodes_labels = current_subgraph.right_exit_nodes
            elif isinstance(current_subgraph, DoorwaySubgraph):
                current_nodes_labels = current_subgraph.right_nodes
            elif isinstance(current_subgraph, IntersectionSubgraph):
                current_nodes_labels = current_subgraph.right_nodes

            if isinstance(next_subgraph, DriveThroughSubgraph):
                if corridor_id in next_subgraph.entry_corridor_id:
                    next_nodes_labels = next_subgraph.left_entry_nodes
                else:
                    next_nodes_labels = next_subgraph.left_exit_nodes
            elif isinstance(next_subgraph, DoorwaySubgraph):
                next_nodes_labels = next_subgraph.left_nodes
            elif isinstance(next_subgraph, IntersectionSubgraph):
                next_nodes_labels = next_subgraph.left_nodes

        else:
            if isinstance(current_subgraph, DriveThroughSubgraph):
                if corridor_id in current_subgraph.entry_corridor_id:
                    current_nodes_labels = current_subgraph.left_entry_nodes
                else:
                    current_nodes_labels = current_subgraph.left_exit_nodes
            elif isinstance(current_subgraph, DoorwaySubgraph):
                current_nodes_labels = current_subgraph.left_nodes
            elif isinstance(current_subgraph, IntersectionSubgraph):
                current_nodes_labels = current_subgraph.lower_nodes

            if isinstance(next_subgraph, DriveThroughSubgraph):
                if corridor_id in next_subgraph.entry_corridor_id:
                    next_nodes_labels = next_subgraph.right_entry_nodes
                else:
                    next_nodes_labels = next_subgraph.right_exit_nodes
            elif isinstance(next_subgraph, DoorwaySubgraph):
                next_nodes_labels = next_subgraph.right_nodes
            elif isinstance(next_subgraph, IntersectionSubgraph):
                next_nodes_labels = next_subgraph.upper_nodes

        return current_nodes_labels, next_nodes_labels
    
    def _replace_node_in_subgraph(self,
                                  current_direction: str,
                                  subgraph: Union[IntersectionSubgraph, DoorwaySubgraph, DriveThroughSubgraph],
                                  old_node_label: str,
                                  new_node: TraversalNode) -> None:
        
        if isinstance(subgraph, DoorwaySubgraph):
            self._replace_node_in_doorway_subgraph(subgraph=subgraph,
                                                    old_node_label=old_node_label,
                                                    new_node=new_node)
        elif isinstance(subgraph, IntersectionSubgraph):
            self._replace_node_in_intersection_subgraph(direction=current_direction,
                                                        subgraph=subgraph,
                                                        old_node_label=old_node_label,
                                                        new_node=new_node)
        elif isinstance(subgraph, DriveThroughSubgraph):
            self._replace_node_in_drive_through_subgraph(subgraph=subgraph,
                                                         old_node_label=old_node_label,
                                                         new_node=new_node,
                                                         exit_flag=(old_node_label in subgraph.right_exit_nodes or
                                                                    old_node_label in subgraph.left_exit_nodes))
        
        assert old_node_label not in subgraph.nodes_dict


    def _merge_nodes(self,
                     current_direction: str,
                     current_subgraph_group: List[Union[IntersectionSubgraph, DoorwaySubgraph, DriveThroughSubgraph]],
                     current_nodes_labels: list[str],
                     next_subgraph_group: List[Union[IntersectionSubgraph, DoorwaySubgraph, DriveThroughSubgraph]],
                     next_nodes_labels: list[str]) -> None:
        
        for i in range(len(current_nodes_labels)):
            current_node_label = current_nodes_labels[i]
            next_node_label = next_nodes_labels[i]

            if current_node_label != next_node_label:
                current_node = current_subgraph_group[0].nodes_dict[current_node_label]
                next_node = next_subgraph_group[0].nodes_dict[next_node_label]

                if current_direction == "horizontal":
                    merged_x = (current_node.position.x + next_node.position.x) / 2.0
                    merged_node = TraversalNode(label=f"{merged_x:.2f}_{current_node.position.y:.2f}_{current_node.orientation_vec}",
                                                position=Coordinate(x=merged_x, y=current_node.position.y),
                                                orientation_vec=current_node.orientation_vec,
                                                connections=[])
                else:
                    merged_y = (current_node.position.y + next_node.position.y) / 2.0
                    merged_node = TraversalNode(label=f"{current_node.position.x:.2f}_{merged_y:.2f}_{current_node.orientation_vec}",
                                                position=Coordinate(x=current_node.position.x, y=merged_y),
                                                orientation_vec=current_node.orientation_vec,
                                                connections=[])
                    
                for j in range(len(current_subgraph_group)):
                    current_subgraph = current_subgraph_group[j]
                    self._replace_node_in_subgraph(current_direction=current_direction,
                                                subgraph=current_subgraph,
                                                old_node_label=current_node_label,
                                                new_node=merged_node)
                
                for m in range(len(next_subgraph_group)):
                    next_subgraph = next_subgraph_group[m]
                    self._replace_node_in_subgraph(current_direction=current_direction,
                                                subgraph=next_subgraph,
                                                old_node_label=next_node_label,
                                                new_node=merged_node)
    
    def _create_straight_connections_for_adjacent_subgraph_nodes(self,
                                                        corridor: Corridor,
                                                        current_nodes: List[str], 
                                                        next_nodes: List[str],
                                                        nodes_dict: dict[str, TraversalNode]) -> Tuple[List[TraversalEdge], List[TraversalEdge]]:
        if corridor.direction == "horizontal":
            original_ref_direction_vec = OrientationVector(1.0, 0.0)  # Facing right
            inverted_ref_direction_vec = OrientationVector(-1.0, 0.0)  # Facing left
        else:
            original_ref_direction_vec = OrientationVector(0.0, 1.0)  # Facing down
            inverted_ref_direction_vec = OrientationVector(0.0, -1.0)  # Facing up

        straight_edges = self._create_straight_connections_for_nodes(ref_nodes=current_nodes,
                                                                     opp_nodes=next_nodes,
                                                                     ref_direction_vec=original_ref_direction_vec,
                                                                     nodes_dict=nodes_dict)

        inverted_straight_edges = self._create_straight_connections_for_nodes(ref_nodes=next_nodes,
                                                                             opp_nodes=current_nodes,
                                                                             ref_direction_vec=inverted_ref_direction_vec,
                                                                             nodes_dict=nodes_dict)

        return straight_edges, inverted_straight_edges
    
    def _connect_subgraphs(self,
                           current_corridor: Corridor,
                           current_subgraph_group: List[Union[IntersectionSubgraph, DoorwaySubgraph, 
                                                   DriveThroughSubgraph, SwitchingPointSubgraph]],
                           current_nodes_labels: list[str],
                           next_subgraph_group: List[Union[IntersectionSubgraph, DoorwaySubgraph, 
                                                DriveThroughSubgraph, SwitchingPointSubgraph]],
                           next_nodes_labels: list[str]) -> None:
        
        nodes_dict = next_subgraph_group[0].nodes_dict | current_subgraph_group[0].nodes_dict
        current_edges, inverted_edges = self._create_straight_connections_for_adjacent_subgraph_nodes(corridor=current_corridor,
                                                                                            current_nodes=current_nodes_labels,
                                                                                            next_nodes=next_nodes_labels,
                                                                                            nodes_dict=nodes_dict)
        for subgraph_index in range(len(current_subgraph_group)):
            current_subgraph = current_subgraph_group[subgraph_index]
            current_subgraph.edges.extend(current_edges)

        for subgraph_index in range(len(next_subgraph_group)):
            next_subgraph = next_subgraph_group[subgraph_index]
            next_subgraph.edges.extend(inverted_edges)
    
    def _create_connective_structure_between_subgraphs(self,
                                                       current_corridor: Corridor,
                                                       current_subgraph_group: List[Union[IntersectionSubgraph, DoorwaySubgraph, DriveThroughSubgraph]],
                                                       current_nodes_labels: list[str],
                                                       next_subgraph_group: List[Union[IntersectionSubgraph, DoorwaySubgraph, DriveThroughSubgraph]],
                                                       next_nodes_labels: list[str]) -> List[SwitchingPointSubgraph]:
        initial_node_label = current_nodes_labels[0]
        next_node_label = next_nodes_labels[0]
        current_direction = current_corridor.direction

        initial_node = current_subgraph_group[0].nodes_dict[initial_node_label]
        next_node = next_subgraph_group[0].nodes_dict[next_node_label]

        if current_direction == "horizontal":
            distance = abs(next_node.position.x - initial_node.position.x)
        else:
            distance = abs(next_node.position.y - initial_node.position.y)

        if distance <= 2*self.threshold:
            self._merge_nodes(current_direction=current_direction,
                              current_subgraph_group=current_subgraph_group,
                              current_nodes_labels=current_nodes_labels,
                              next_subgraph_group=next_subgraph_group,
                              next_nodes_labels=next_nodes_labels)
            return []
        elif distance > 3 * self.switching_point_offset:
            switching_point_subgraph = self._generate_switching_point_subgraph(center=Coordinate(
                                                                                (initial_node.position.x + next_node.position.x) / 2.0, 
                                                                                (initial_node.position.y + next_node.position.y) / 2.0
                                                                               ),
                                                                           corridor=current_corridor)
            if current_direction == "horizontal":
                switching_point_current_labels = switching_point_subgraph.right_nodes
                switching_point_next_labels = switching_point_subgraph.left_nodes
            else:
                switching_point_current_labels = switching_point_subgraph.left_nodes
                switching_point_next_labels = switching_point_subgraph.right_nodes

            self._connect_subgraphs(current_corridor=current_corridor,
                                    current_subgraph_group=current_subgraph_group,
                                    current_nodes_labels=current_nodes_labels,
                                    next_subgraph_group=[switching_point_subgraph],
                                    next_nodes_labels=switching_point_next_labels)

            self._connect_subgraphs(current_corridor=current_corridor,
                                    current_subgraph_group=[switching_point_subgraph],
                                    current_nodes_labels=switching_point_current_labels,
                                    next_subgraph_group=next_subgraph_group,
                                    next_nodes_labels=next_nodes_labels)

            return [switching_point_subgraph]
        else:
            self._connect_subgraphs(current_corridor=current_corridor,
                                    current_subgraph_group=current_subgraph_group,
                                    current_nodes_labels=current_nodes_labels,
                                    next_subgraph_group=next_subgraph_group,
                                    next_nodes_labels=next_nodes_labels)
            
            return []
    
    def _connect_corridor_subgraphs(self) -> None:

        for corridor_id in self.doorway_subgraph_indices.keys():
            current_corridor = self._get_corridor_by_id(corridor_id)
            current_direction = current_corridor.direction

            sub_graphs = self._get_subgraphs_for_corridor(corridor_id=corridor_id)
            
            sorted_subgraphs = self._sort_subgraphs(corridor_id=corridor_id,
                                                    subgraphs=sub_graphs,
                                                    direction=current_direction)

            grouped_subgraphs = self._group_subgraphs_with_shared_nodes(corridor_id=corridor_id,
                                                                        direction=current_direction,
                                                                        subgraphs=sorted_subgraphs)

            node_keys = list(grouped_subgraphs.keys())

            switching_point_subgraphs = []

            for i in range(len(node_keys)-1):
                current_subgraph_group = grouped_subgraphs[node_keys[i]]
                next_subgraph_group = grouped_subgraphs[node_keys[i+1]]

                current_nodes_labels, next_nodes_labels = self._obtain_representative_nodes_from_subgraphs(current_direction=current_direction,
                                                                                                       current_subgraph=current_subgraph_group[0],
                                                                                                       next_subgraph=next_subgraph_group[0],
                                                                                                       corridor_id=corridor_id)

                curr_switching_point_subgraphs = self._create_connective_structure_between_subgraphs(current_corridor=current_corridor,
                                                                    current_subgraph_group=current_subgraph_group,
                                                                    current_nodes_labels=current_nodes_labels,
                                                                    next_subgraph_group=next_subgraph_group,
                                                                    next_nodes_labels=next_nodes_labels)
                switching_point_subgraphs.extend(curr_switching_point_subgraphs)

            self.switching_point_subgraphs.extend(switching_point_subgraphs)

    def _populate_nodes_from_subgraph(self,
                                      subgraph: Union[IntersectionSubgraph, DoorwaySubgraph, DriveThroughSubgraph, SwitchingPointSubgraph],
                                      traversal_graph: TraversalGraph) -> None:

        for node_label, node in subgraph.nodes_dict.items():
            if node_label not in traversal_graph.nodes_dict:
                traversal_graph.nodes_dict[node_label] = node
            else:
                existing_node = traversal_graph.nodes_dict[node_label]
                assert node == existing_node , f"Node mismatch for label {node_label} during traversal graph assembly."

    def _populate_nodes_dict(self,
                             traversal_graph: TraversalGraph) -> None:
        
        for intersection_subgraph in self.corridor_intersection_subgraphs:
            self._populate_nodes_from_subgraph(subgraph=intersection_subgraph,
                                              traversal_graph=traversal_graph)
        
        for doorway_subgraph in self.doorway_subgraphs:
            self._populate_nodes_from_subgraph(subgraph=doorway_subgraph,
                                              traversal_graph=traversal_graph)
        
        for drive_through_subgraph in self.drive_through_subgraphs:
            self._populate_nodes_from_subgraph(subgraph=drive_through_subgraph,
                                              traversal_graph=traversal_graph)
        
        for switching_point_subgraph in self.switching_point_subgraphs:
            self._populate_nodes_from_subgraph(subgraph=switching_point_subgraph,
                                              traversal_graph=traversal_graph)
    
    def _check_edge_validity(self,
                             edge: TraversalEdge,
                             nodes_dict: dict[str, TraversalNode],
                             node_connections: dict[str, list[str]]) -> bool:

        # Check if the edge's start and end nodes exist in the nodes_dict

        assert edge.from_node in nodes_dict, f"Edge start node {edge.from_node} not found in nodes_dict."
        assert edge.to_node in nodes_dict, f"Edge end node {edge.to_node} not found in nodes_dict."

        origin_node = nodes_dict[edge.from_node]
        destination_node = nodes_dict[edge.to_node]
        edge_action = edge.action

        assert destination_node.label in origin_node.connections, f"Edge {edge} not found in origin node connections."

        if edge_action == "go_straight":
            assert origin_node.orientation_vec == destination_node.orientation_vec, f"Orientation mismatch for edge {edge}."
            if origin_node.orientation_vec == OrientationVector(1.0, 0.0):  # Facing right
                assert origin_node.position.y == destination_node.position.y, \
                    f"Invalid straight edge y locations from {edge.from_node} to {edge.to_node}."
                assert origin_node.position.x < destination_node.position.x, \
                    f"Invalid straight edge x locations from {edge.from_node} to {edge.to_node}."
            elif origin_node.orientation_vec == OrientationVector(-1.0, 0.0):  # Facing left
                assert origin_node.position.y == destination_node.position.y, \
                    f"Invalid straight edge y locations from {edge.from_node} to {edge.to_node}."
                assert origin_node.position.x > destination_node.position.x, \
                    f"Invalid straight edge x locations from {edge.from_node} to {edge.to_node}."
            elif origin_node.orientation_vec == OrientationVector(0.0, 1.0):  # Facing down
                assert origin_node.position.x == destination_node.position.x, \
                    f"Invalid straight edge x locations from {edge.from_node} to {edge.to_node}."
                assert origin_node.position.y < destination_node.position.y, \
                    f"Invalid straight edge y locations from {edge.from_node} to {edge.to_node}."
            elif origin_node.orientation_vec == OrientationVector(0.0, -1.0):  # Facing up
                assert origin_node.position.x == destination_node.position.x, \
                    f"Invalid straight edge x locations from {edge.from_node} to {edge.to_node}."
                assert origin_node.position.y > destination_node.position.y, \
                    f"Invalid straight edge y locations from {edge.from_node} to {edge.to_node}."
            else:
                raise ValueError(f"Unknown orientation vector {origin_node.orientation_vec} for edge {edge}.")
        elif edge_action == "turn_left":
            if origin_node.orientation_vec == OrientationVector(1.0, 0.0):  # Facing right
                assert destination_node.orientation_vec == OrientationVector(0.0, -1.0), \
                    f"Invalid left turn edge orientation for {edge}"
                assert origin_node.position.x < destination_node.position.x, \
                    f"Invalid left turn edge x locations from {edge.from_node} to {edge.to_node}."
                assert origin_node.position.y > destination_node.position.y, \
                    f"Invalid left turn edge y locations from {edge.from_node} to {edge.to_node}."
            elif origin_node.orientation_vec == OrientationVector(-1.0, 0.0):  # Facing left
                assert destination_node.orientation_vec == OrientationVector(0.0, 1.0), \
                    f"Invalid left turn edge orientation for {edge}"
                assert origin_node.position.x > destination_node.position.x, \
                    f"Invalid left turn edge x locations from {edge.from_node} to {edge.to_node}."
                assert origin_node.position.y < destination_node.position.y, \
                    f"Invalid left turn edge y locations from {edge.from_node} to {edge.to_node}."
            elif origin_node.orientation_vec == OrientationVector(0.0, 1.0):  # Facing down
                assert destination_node.orientation_vec == OrientationVector(1.0, 0.0), \
                    f"Invalid left turn edge orientation for {edge}"
                assert origin_node.position.y < destination_node.position.y, \
                    f"Invalid left turn edge y locations from {edge.from_node} to {edge.to_node}."
                assert origin_node.position.x < destination_node.position.x, \
                    f"Invalid left turn edge x locations from {edge.from_node} to {edge.to_node}."
            elif origin_node.orientation_vec == OrientationVector(0.0, -1.0):  # Facing up
                assert destination_node.orientation_vec == OrientationVector(-1.0, 0.0), \
                    f"Invalid left turn edge orientation for {edge}"
                assert origin_node.position.y > destination_node.position.y, \
                    f"Invalid left turn edge y locations from {edge.from_node} to {edge.to_node}."
                assert origin_node.position.x > destination_node.position.x, \
                    f"Invalid left turn edge x locations from {edge.from_node} to {edge.to_node}."
            else:
                raise ValueError(f"Unknown orientation vector {origin_node.orientation_vec} for edge {edge}.")
        elif edge_action == "turn_right":
            if origin_node.orientation_vec == OrientationVector(1.0, 0.0):  # Facing right
                assert destination_node.orientation_vec == OrientationVector(0.0, 1.0), \
                    f"Invalid right turn edge orientation for {edge}."
                assert origin_node.position.x < destination_node.position.x, \
                    f"Invalid right turn edge x locations from {edge.from_node} to {edge.to_node}."
                assert origin_node.position.y < destination_node.position.y, \
                    f"Invalid right turn edge y locations from {edge.from_node} to {edge.to_node}."
            elif origin_node.orientation_vec == OrientationVector(-1.0, 0.0):  # Facing left
                assert destination_node.orientation_vec == OrientationVector(0.0, -1.0), \
                    f"Invalid right turn edge orientation for {edge}."
                assert origin_node.position.x > destination_node.position.x, \
                    f"Invalid right turn edge x locations from {edge.from_node} to {edge.to_node}."
                assert origin_node.position.y > destination_node.position.y, \
                    f"Invalid right turn edge y locations from {edge.from_node} to {edge.to_node}."
            elif origin_node.orientation_vec == OrientationVector(0.0, 1.0):  # Facing down
                assert destination_node.orientation_vec == OrientationVector(-1.0, 0.0), \
                    f"Invalid right turn edge orientation for {edge}."
                assert origin_node.position.y < destination_node.position.y, \
                    f"Invalid right turn edge y locations from {edge.from_node} to {edge.to_node}."
                assert origin_node.position.x > destination_node.position.x, \
                    f"Invalid right turn edge x locations from {edge.from_node} to {edge.to_node}."
            elif origin_node.orientation_vec == OrientationVector(0.0, -1.0):  # Facing up
                assert destination_node.orientation_vec == OrientationVector(1.0, 0.0), \
                    f"Invalid right turn edge orientation for {edge}."
                assert origin_node.position.y > destination_node.position.y, \
                    f"Invalid right turn edge y locations from {edge.from_node} to {edge.to_node}."
                assert origin_node.position.x < destination_node.position.x, \
                    f"Invalid right turn edge x locations from {edge.from_node} to {edge.to_node}."
            else:
                raise ValueError(f"Unknown orientation vector {origin_node.orientation_vec} for edge {edge}.")
        elif edge_action == "switch_directions":
            if origin_node.orientation_vec == OrientationVector(1.0, 0.0):  # Facing right
                assert destination_node.orientation_vec == OrientationVector(-1.0, 0.0), \
                    f"Invalid switch direction edge {edge}."
                assert origin_node.position.x == destination_node.position.x, \
                    f"Invalid switch direction edge x locations from {edge.from_node} to {edge.to_node}."
                assert origin_node.position.y > destination_node.position.y, \
                    f"Invalid switch direction edge y locations from {edge.from_node} to {edge.to_node}."
            elif origin_node.orientation_vec == OrientationVector(-1.0, 0.0):  # Facing left
                assert destination_node.orientation_vec == OrientationVector(1.0, 0.0), \
                    f"Invalid switch direction edge {edge}."
                assert origin_node.position.x == destination_node.position.x, \
                    f"Invalid switch direction edge x locations from {edge.from_node} to {edge.to_node}."
                assert origin_node.position.y < destination_node.position.y, \
                    f"Invalid switch direction edge y locations from {edge.from_node} to {edge.to_node}."
            elif origin_node.orientation_vec == OrientationVector(0.0, 1.0):  # Facing down
                assert destination_node.orientation_vec == OrientationVector(0.0, -1.0), \
                    f"Invalid switch direction edge {edge}."
                assert origin_node.position.x < destination_node.position.x, \
                    f"Invalid switch direction edge x locations from {edge.from_node} to {edge.to_node}."
                assert origin_node.position.y == destination_node.position.y, \
                    f"Invalid switch direction edge y locations from {edge.from_node} to {edge.to_node}."
            elif origin_node.orientation_vec == OrientationVector(0.0, -1.0):  # Facing up
                assert destination_node.orientation_vec == OrientationVector(0.0, 1.0), \
                    f"Invalid switch direction edge {edge}."
                assert origin_node.position.x > destination_node.position.x, \
                    f"Invalid switch direction edge x locations from {edge.from_node} to {edge.to_node}."
                assert origin_node.position.y == destination_node.position.y, \
                    f"Invalid switch direction edge y locations from {edge.from_node} to {edge.to_node}."
            else:
                raise ValueError(f"Unknown orientation vector {origin_node.orientation_vec} for edge {edge}.")
        elif edge_action == "switch_lanes":
            # For switching lanes, the orientation should remain the same
            assert origin_node.orientation_vec == destination_node.orientation_vec, f"Orientation mismatch for edge {edge}."
            if origin_node.orientation_vec == OrientationVector(1.0, 0.0):
                assert origin_node.position.y != destination_node.position.y and origin_node.position.x < destination_node.position.x, \
                    f"Invalid lane switch edge from {edge.from_node} to {edge.to_node}."
            elif origin_node.orientation_vec == OrientationVector(-1.0, 0.0):
                assert origin_node.position.y != destination_node.position.y and origin_node.position.x > destination_node.position.x, \
                    f"Invalid lane switch edge from {edge.from_node} to {edge.to_node}."
            elif origin_node.orientation_vec == OrientationVector(0.0, 1.0):
                assert origin_node.position.x != destination_node.position.x and origin_node.position.y < destination_node.position.y, \
                    f"Invalid lane switch edge from {edge.from_node} to {edge.to_node}."
            elif origin_node.orientation_vec == OrientationVector(0.0, -1.0):
                assert origin_node.position.x != destination_node.position.x and origin_node.position.y > destination_node.position.y, \
                    f"Invalid lane switch edge from {edge.from_node} to {edge.to_node}."
            else: 
                raise ValueError(f"Unknown orientation vector {origin_node.orientation_vec} for edge {edge}.")
                
        else:
            raise ValueError(f"Unknown edge action {edge_action} for edge {edge}.")
        
        node_connections.setdefault(edge.from_node, []).append(edge.to_node)
    
    def _populate_edges_from_subgraph(self,
                                     subgraph: Union[IntersectionSubgraph, DoorwaySubgraph, DriveThroughSubgraph, SwitchingPointSubgraph],
                                     traversal_graph: TraversalGraph,
                                     node_connections: Dict[str, List[str]]) -> None:
        for edge in subgraph.edges:
            if edge not in traversal_graph.edges:
                self._check_edge_validity(edge=edge, 
                                          nodes_dict=traversal_graph.nodes_dict,
                                          node_connections=node_connections)
                traversal_graph.edges.append(edge)

    def _populate_edges(self, traversal_graph: TraversalGraph) -> None:
        node_connections = {}

        for intersection_subgraph in self.corridor_intersection_subgraphs:
            self._populate_edges_from_subgraph(subgraph=intersection_subgraph,
                                              traversal_graph=traversal_graph,
                                              node_connections=node_connections)

        for doorway_subgraph in self.doorway_subgraphs:
            self._populate_edges_from_subgraph(subgraph=doorway_subgraph,
                                              traversal_graph=traversal_graph,
                                              node_connections=node_connections)

        for drive_through_subgraph in self.drive_through_subgraphs:
            self._populate_edges_from_subgraph(subgraph=drive_through_subgraph,
                                              traversal_graph=traversal_graph,
                                              node_connections=node_connections)

        for switching_point_subgraph in self.switching_point_subgraphs:
            self._populate_edges_from_subgraph(subgraph=switching_point_subgraph,
                                              traversal_graph=traversal_graph,
                                              node_connections=node_connections)

        for node_label, connections in node_connections.items():
            current_node = traversal_graph.nodes_dict[node_label]
            assert set(connections) == set(current_node.connections), f"Node connections mismatch for node {node_label}."

    def _build_and_check_traversal_graph(self) -> TraversalGraph:
        traversal_graph = TraversalGraph(nodes_dict={}, edges=[])

        self._populate_nodes_dict(traversal_graph=traversal_graph)
        self._populate_edges(traversal_graph=traversal_graph)

        return traversal_graph
            

    def _assemble_traversal_graph(self) -> TraversalGraph:

        self._connect_corridor_subgraphs()

        traversal_graph = self._build_and_check_traversal_graph()
        
        return traversal_graph

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

    TraversalGraphPlottingHelper.plot_extracted_structs(occupancy_map=tg_generator.occupancy_map,
                                                        origin_x=tg_generator.origin_x,
                                                        origin_y=tg_generator.origin_y,
                                                        resolution=tg_generator.resolution,
                                                        corridors=tg_generator.corridors,
                                                        drive_throughs=tg_generator.drive_throughs,
                                                        doorways=tg_generator.doorways,
                                                        filename="results/environment/extracted_structs.svg")
    TraversalGraphPlottingHelper.plot_intersection_subgraphs(occupancy_map=tg_generator.occupancy_map,
                                                             origin_x=tg_generator.origin_x,
                                                             origin_y=tg_generator.origin_y,
                                                             resolution=tg_generator.resolution,
                                                             subgraphs=tg_generator.corridor_intersection_subgraphs,
                                                             filename="results/environment/intersection_subgraph_0.svg")
    
    TraversalGraphPlottingHelper.plot_doorway_subgraphs(occupancy_map=tg_generator.occupancy_map,
                                                        origin_x=tg_generator.origin_x,
                                                        origin_y=tg_generator.origin_y,
                                                        resolution=tg_generator.resolution,
                                                        subgraphs=tg_generator.doorway_subgraphs,
                                                        filename="results/environment/doorway_subgraph_1.svg")

    TraversalGraphPlottingHelper.plot_switching_point_subgraphs(occupancy_map=tg_generator.occupancy_map,
                                                               origin_x=tg_generator.origin_x,
                                                               origin_y=tg_generator.origin_y,
                                                               resolution=tg_generator.resolution,
                                                               subgraphs=tg_generator.switching_point_subgraphs,
                                                               filename="results/environment/switching_point_subgraph_0.svg")
    TraversalGraphPlottingHelper.plot_drive_through_subgraphs(occupancy_map=tg_generator.occupancy_map,
                                                               origin_x=tg_generator.origin_x,
                                                               origin_y=tg_generator.origin_y,
                                                               resolution=tg_generator.resolution,
                                                               subgraphs=tg_generator.drive_through_subgraphs,
                                                               filename="results/environment/drive_through_subgraph_0.svg")
    TraversalGraphPlottingHelper.plot_subgraphs_in_one_plot(occupancy_map=tg_generator.occupancy_map,
                                                             origin_x=tg_generator.origin_x,
                                                             origin_y=tg_generator.origin_y,
                                                             resolution=tg_generator.resolution,
                                                             intersection_subgraphs=tg_generator.corridor_intersection_subgraphs,
                                                             doorway_subgraphs=tg_generator.doorway_subgraphs,
                                                             drive_through_subgraphs=tg_generator.drive_through_subgraphs,
                                                             switching_point_subgraphs=tg_generator.switching_point_subgraphs,
                                                             filename="results/environment/all_subgraphs.svg")
    print(tg_generator.corridor_intersection_subgraph_indices)
    print(tg_generator.doorway_subgraph_indices)
    print(tg_generator.drive_through_subgraph_indices)
