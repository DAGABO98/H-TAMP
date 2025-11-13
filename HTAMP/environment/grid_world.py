import pickle
import argparse
import datetime
import numpy as np

from typing import Dict, List, Set, Tuple

from HTAMP.environment.geometry_helpers import CurvedConnector
from HTAMP.environment.loc_dataclasses import BoundingIndices, Cell, Coordinate, GridIndex 
from HTAMP.environment.loc_dataclasses import MotionReservation, RobotOccupancy, TimeInterval
from HTAMP.environment.robot_dataclasses import RobotProfile
from HTAMP.plotting.motion_planning_plotting import MotionPlanningPlotter
from HTAMP.environment.traversal_dataclasses import TraversalGraph, TraversalNode
from HTAMP.environment.traversal_graph_gen import TraversalGraphGenerator

OCCUPANCY_CACHE_VERSION = 1

class GridWorld:

    def __init__(self, 
                 cell_size: float,
                 fps: float, 
                 occupancy_map: np.ndarray, 
                 traversal_graph: TraversalGraph,
                 shortest_paths: Dict[str, Dict[str, Tuple[List[str], float]]],
                 robot_profiles: List[RobotProfile],
                 use_saved_data: bool,
                 occupancy_reservations_file: str) -> None:
        self.cell_size = cell_size  # meters per cell
        self.fps = fps
        self.occupancy_map = occupancy_map  # 2D numpy array with 1=occupied, 0=free
        self.traversal_graph = traversal_graph
        self.shortest_paths = shortest_paths
        self.robot_profiles = robot_profiles
        self.occupancy_reservations_file = occupancy_reservations_file
        if use_saved_data:
            self.occupancy_reservations = self._load_occupancy_reservations()
        else:
            self.occupancy_reservations = self._create_occupancy_reservations()
            self._save_occupancy_reservations()
    
    def _load_occupancy_reservations(self) -> Dict[str, Dict[str, Dict[str, List[MotionReservation]]]]:
        with open(self.occupancy_reservations_file, 'rb') as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict) or payload.get("_version") != OCCUPANCY_CACHE_VERSION:
            raise ValueError("Occupancy reservation cache version mismatch or corrupted file.")
        return payload["data"]

    def _save_occupancy_reservations(self) -> None:
        import os
        os.makedirs(os.path.dirname(self.occupancy_reservations_file) or ".", exist_ok=True)
        payload = {"_version": OCCUPANCY_CACHE_VERSION, "data": self.occupancy_reservations}
        with open(self.occupancy_reservations_file, 'wb') as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _cell_rect(self, 
                  cell_index: GridIndex) -> Cell:
        lower_x = cell_index.index_x * self.cell_size
        lower_y = cell_index.index_y * self.cell_size
        return Cell(lower_x, lower_y, lower_x + self.cell_size, lower_y + self.cell_size)
    
    # Add this helper in GridWorld
    def _clamp_index(self, ix: int, iy: int) -> GridIndex:
        h, w = self.occupancy_map.shape
        return GridIndex(max(0, min(ix, w - 1)), max(0, min(iy, h - 1)))

    
    def _get_cell_index(self, position: Coordinate) -> GridIndex:
        index_x = int(np.floor(position.x / self.cell_size))
        index_y = int(np.floor(position.y / self.cell_size))
        return self._clamp_index(index_x, index_y)
    
    def _generate_closest_cell_coordinate(self, 
                                        robot_center: Coordinate,
                                        cell_index: GridIndex) -> Coordinate:
        cell = self._cell_rect(cell_index)
        selected_x = min(max(robot_center.x, cell.lower_x), cell.upper_x)
        selected_y = min(max(robot_center.y, cell.lower_y), cell.upper_y)

        return Coordinate(selected_x, selected_y)

    def robot_intersects_cell(self, 
                              robot_center: Coordinate, 
                              robot_profile: RobotProfile,
                              cell_index: GridIndex) -> bool:
        selected_coordinate = self._generate_closest_cell_coordinate(robot_center,
                                                                    cell_index)
        dev_x = robot_center.x - selected_coordinate.x
        dev_y = robot_center.y - selected_coordinate.y
        return ((dev_x * dev_x) + (dev_y * dev_y)) <= (robot_profile.radius * robot_profile.radius)
    
    def _get_robot_bounding_indices(self, 
                                    robot_center: Coordinate, 
                                    robot_profile: RobotProfile) -> BoundingIndices:
        lower_index = self._get_cell_index(Coordinate(robot_center.x - robot_profile.radius, 
                                                     robot_center.y - robot_profile.radius))
        upper_index = self._get_cell_index(Coordinate(robot_center.x + robot_profile.radius, 
                                                     robot_center.y + robot_profile.radius))

        return BoundingIndices(lower_index.index_x, lower_index.index_y, 
                               upper_index.index_x, upper_index.index_y)

    def get_occupied_cells_for_robot(self, 
                                     robot_center: Coordinate, 
                                     robot_profile: RobotProfile) -> List[GridIndex]:
        bounding = self._get_robot_bounding_indices(robot_center, robot_profile)
        h, w = self.occupancy_map.shape
        xs = range(max(0, bounding.lower_x), min(w - 1, bounding.upper_x) + 1)
        ys = range(max(0, bounding.lower_y), min(h - 1, bounding.upper_y) + 1)

        occupied_cells = []
        for index_x in xs:
            for index_y in ys:
                cell_index = GridIndex(index_x, index_y)
                if self.robot_intersects_cell(robot_center, robot_profile, cell_index):
                    occupied_cells.append(cell_index)
        return occupied_cells
    
    def is_robot_in_free_space(self, 
                                robot_center: Coordinate, 
                                robot_profile: RobotProfile) -> bool:
        occupied_cells = self.get_occupied_cells_for_robot(robot_center, robot_profile)
        for cell_index in occupied_cells:
            if self.occupancy_map[cell_index.index_y, cell_index.index_x] == 1:
                return False
        return True
    
    def _get_occupied_cells_for_static_position(self, 
                                               robot_position: Coordinate,
                                               robot_profile: RobotProfile) -> Set[GridIndex]:
        occupied_cells = set(self.get_occupied_cells_for_robot(robot_position, robot_profile))
        return occupied_cells
    
    def _get_occupied_cells_for_partial_move(self, 
                                          robot_start_pos: Coordinate, 
                                          robot_end_pos: Coordinate,
                                          robot_profile: RobotProfile) -> Set[GridIndex]:
        initial_occupied_cells = set(self.get_occupied_cells_for_robot(robot_start_pos, robot_profile))
        final_occupied_cells = set(self.get_occupied_cells_for_robot(robot_end_pos, robot_profile))
        occupied_cells = initial_occupied_cells.union(final_occupied_cells)
        return occupied_cells
    
    def _generate_robot_occupancy_for_move(self,
                                           curved_connector: CurvedConnector,
                                           robot_profile: RobotProfile) -> List[RobotOccupancy]:
        connector_length = curved_connector.length()
        num_frames = int(np.ceil((connector_length / robot_profile.speed) * self.fps))
        robot_occupancies: List[RobotOccupancy] = []

        if num_frames > 1:
            segments, _ = curved_connector.split_connector_into_n(num_frames)
            for segment in segments:
                segment_start = Coordinate(segment['X'][0], segment['Y'][0])
                segment_end = Coordinate(segment['X'][-1], segment['Y'][-1])
                occupied_cells = self._get_occupied_cells_for_partial_move(segment_start,
                                                                        segment_end,
                                                                        robot_profile)
                occupancy = RobotOccupancy(occupied_cells=occupied_cells,
                                        start_location=segment_start,
                                        end_location=segment_end)
                robot_occupancies.append(occupancy)
        else:
            start_point = curved_connector.origin
            end_point = curved_connector.destination
            occupied_cells = self._get_occupied_cells_for_partial_move(start_point,
                                                                      end_point,
                                                                      robot_profile)
            occupancy = RobotOccupancy(occupied_cells=occupied_cells,
                                    start_location=start_point,
                                    end_location=end_point)
            robot_occupancies.append(occupancy)

        return robot_occupancies
    
    def _generate_robot_timing_for_move(self, curved_connector: CurvedConnector, robot_profile: RobotProfile) -> List[TimeInterval]:
        connector_length = curved_connector.length()
        if robot_profile.speed <= 0:
            raise ValueError("Robot speed must be > 0")
        total_time = connector_length / robot_profile.speed
        num_frames = int(np.ceil(total_time * self.fps))
        time_intervals: List[TimeInterval] = []

        if num_frames < 1:
            # Zero-length edge => single zero-time interval
            time_intervals.append(TimeInterval(start=0.0, end=0.0))
            return time_intervals

        time_per_frame = total_time / num_frames
        for i in range(num_frames):
            start_time = i * time_per_frame
            end_time = (i + 1) * time_per_frame
            time_intervals.append(TimeInterval(start=start_time, end=end_time))

        # Be tolerant to floating point
        assert np.isclose(time_intervals[-1].end, total_time, rtol=1e-6, atol=1e-9)
        return time_intervals
    
    def _check_for_occupancy_conflict(self,
                                    robot_occupancies: List[RobotOccupancy]) -> bool:
        for occupancy in robot_occupancies:
            for cell_index in occupancy.occupied_cells:
                if self.occupancy_map[cell_index.index_y, cell_index.index_x] == 1:
                    return True
        return False
    
    def get_node_footprint_cells(self, node: TraversalNode, robot_profile: RobotProfile) -> Set[GridIndex]:
        return self._get_occupied_cells_for_static_position(node.position, robot_profile)

    def _create_robot_occupancy_reservations(self, robot_profile: RobotProfile) -> Dict[str, Dict[str, List[MotionReservation]]]:
        occupancy_reservations: Dict[str, Dict[str, List[MotionReservation]]] = {}
        length_edges = len(self.traversal_graph.edge_dict)
        for i, traversal_edge in enumerate(self.traversal_graph.edge_dict.values()):
            print(f"Processing edge {i+1}/{length_edges} for robot {robot_profile.robot_id}")
            print(f"Creating occupancy reservations for robot {robot_profile.robot_id} "
                  f"on edge from {traversal_edge.from_node} to {traversal_edge.to_node}")
            occupancy_reservations.setdefault(traversal_edge.from_node, {})
            edges = occupancy_reservations[traversal_edge.from_node]
            edges.setdefault(traversal_edge.to_node, [])
            robot_occupancies = self._generate_robot_occupancy_for_move(traversal_edge.edge_connector,
                                                                       robot_profile)
            
            if self._check_for_occupancy_conflict(robot_occupancies):
                raise ValueError(f"Occupancy conflict detected for robot {robot_profile.robot_id} "
                                 f"on edge from {traversal_edge.from_node} to {traversal_edge.to_node}")
            
            time_intervals = self._generate_robot_timing_for_move(traversal_edge.edge_connector,
                                                                   robot_profile)
            assert len(robot_occupancies) == len(time_intervals)

            for i in range(len(robot_occupancies)):
                reservation = MotionReservation(time_interval=time_intervals[i],
                                                robot_occupancy=robot_occupancies[i])
                edges[traversal_edge.to_node].append(reservation)
        
        for node in self.traversal_graph.nodes_dict.values():
            edges = occupancy_reservations.setdefault(node.label, {})
            edges.setdefault(node.label, [])
            occupied_cells = self.get_node_footprint_cells(node, robot_profile)
            reservation = MotionReservation(
                time_interval=TimeInterval(start=0.0, end=0.0),
                robot_occupancy=RobotOccupancy(
                    occupied_cells=occupied_cells,
                    start_location=node.position,
                    end_location=node.position
                )
            )
            edges[node.label].append(reservation)

        return occupancy_reservations
    
    def _robot_key(self, robot_profile: RobotProfile) -> str:
        return f"robot_{robot_profile.speed}_{robot_profile.radius}"

    def _create_occupancy_reservations(self) -> Dict[str, Dict[str, Dict[str, List[MotionReservation]]]]:
        occupancy_reservations: Dict[str, Dict[str, Dict[str, List[MotionReservation]]]] = {}
        for robot_profile in self.robot_profiles:
            robot_id_str = self._robot_key(robot_profile)
            occupancy_reservations[robot_id_str] = self._create_robot_occupancy_reservations(robot_profile)

        return occupancy_reservations

    def _get_robot_reservation_bucket(self, robot_profile: RobotProfile):
        key = self._robot_key(robot_profile)
        if key not in self.occupancy_reservations:
            raise KeyError(
                f"No occupancy reservations for robot_id={robot_profile.robot_id}. "
                "Make sure this profile was included when building GridWorld."
            )
        return self.occupancy_reservations[key]
    
    def get_robot_occupancy_for_move(self, from_node, to_node, robot_profile) -> List[RobotOccupancy]:
        robot_motion_reservation = self._get_robot_reservation_bucket(robot_profile)
        motion_reservation = robot_motion_reservation[from_node.label][to_node.label]
        return [res.robot_occupancy for res in motion_reservation]

    def get_robot_timing_for_move(self, from_node, to_node, robot_profile) -> List[TimeInterval]:
        robot_motion_reservation = self._get_robot_reservation_bucket(robot_profile)
        motion_reservation = robot_motion_reservation[from_node.label][to_node.label]
        return [res.time_interval for res in motion_reservation]

    def get_robot_reservations_for_move(self, from_node, to_node, robot_profile, current_time) -> List[MotionReservation]:
        robot_occupancies = self.get_robot_occupancy_for_move(from_node, to_node, robot_profile)
        time_intervals = self.get_robot_timing_for_move(from_node, to_node, robot_profile)
        reservations: List[MotionReservation] = []
        for occ, ti in zip(robot_occupancies, time_intervals):
            new_time_interval = TimeInterval(start=ti.start + current_time, end=ti.end + current_time)
            reservations.append(MotionReservation(time_interval=new_time_interval, robot_occupancy=occ))
        return reservations
    
    def get_shortest_path(self,
                          start: TraversalNode,
                          goal: TraversalNode) -> Tuple[List[str], float]:
        if start.label not in self.shortest_paths or goal.label not in self.shortest_paths[start.label]:
            return ([], float('inf'))
        else:
            return self.shortest_paths[start.label][goal.label]
    

if __name__ == "__main__":
    pStart = datetime.datetime.now()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="maps/FA3/FA3_lanes.yaml", help="Path to the configuration file")
    parser.add_argument("--occupancy_map_path", type=str, default="maps/FA3/occupancy_map.npy", help="Path to the input occupancy map")
    parser.add_argument("--factor", type=int, default=1, help="Downsampling factor")
    parser.add_argument("--meters_per_pixel", type=float, default=0.036, help="Meters per pixel in the original image")
    parser.add_argument("--fps", type=float, default=2.0, help="Frames per second for the grid world")
    parser.add_argument("--occupancy_reservations_file", type=str, default="data/occupancy_reservations.pkl", help="Path to the occupancy reservations file")
    parser.add_argument("--use_saved_data", action='store_true', help="Whether to use saved occupancy reservations data")
    args = parser.parse_args()

    print("Generating Traversal Graph...")

    tg_generator = TraversalGraphGenerator(occupancy_map_path=args.occupancy_map_path,
                                           config_path=args.config_path,
                                           meters_per_pixel=args.meters_per_pixel,
                                           factor=args.factor)

    print("Traversal Graph generated.")

    print("Creating Grid World...")

    robot_profile = RobotProfile(radius=0.1, speed=0.2, robot_id=0)

    world = GridWorld(cell_size=tg_generator.meters_per_cell,
                      fps=args.fps,
                      occupancy_map=tg_generator.occupancy_map,
                      traversal_graph=tg_generator.traversal_graph,
                      shortest_paths=tg_generator.shortest_paths,
                      robot_profiles=[robot_profile],
                      use_saved_data=args.use_saved_data,
                      occupancy_reservations_file=args.occupancy_reservations_file)

    print("Grid World created.")

    traversal_edges = list(world.traversal_graph.edge_dict.values())

    occupancy_reservations = world.get_robot_reservations_for_move(
        from_node=tg_generator.traversal_graph.nodes_dict[traversal_edges[2].from_node],
        to_node=tg_generator.traversal_graph.nodes_dict[traversal_edges[2].to_node],
        robot_profile=robot_profile,
        current_time=0.0
    )

    print(len(occupancy_reservations))

    MotionPlanningPlotter.plot_motion_reservations(occupancy_map=world.occupancy_map,
                                                          origin_x=tg_generator.origin_x,
                                                          origin_y=tg_generator.origin_y,
                                                          resolution=tg_generator.resolution,
                                                          motion_reservations=occupancy_reservations,
                                                          filename="results/motion_planning/robot_motion_reservations.svg")
    pEnd = datetime.datetime.now()
    print(f"Total generation time: {pEnd - pStart}")

    
        
