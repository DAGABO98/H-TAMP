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

@dataclass
class DriveThrough:
    entry_lane: Lane
    exit_lane: Lane
    entry_start: Coordinate
    entry_end: Coordinate
    exit_start: Coordinate
    exit_end: Coordinate
    entry_corridor_id: str
    exit_corridor_id: str

@dataclass
class Doorway:
    start: Coordinate
    end: Coordinate
    entry_lane: Lane
    exit_lane: Lane
    corridor_id: str

@dataclass
class TraversalNode:
    label: str
    position: Coordinate
    connections: list["TraversalNode"]

@dataclass
class TraversalEdge:
    from_node: TraversalNode
    to_node: TraversalNode
    action: str

@dataclass
class TraversalGraph:
    nodes: list[TraversalNode]
    edges: list[TraversalEdge]

class CurvedConnector:
    def __init__(self,
                 origin: Coordinate, 
                 destination: Coordinate, 
                 vec_origin: Tuple[float, float], 
                 vec_destination: Tuple[float, float]):
        self.origin = origin
        self.destination = destination
        self.vec_origin = vec_origin
        self.vec_destination = vec_destination
        self.connector_points = self._generate_connector_points()

    def _unit_vector(self, vec: Tuple[float, float]) -> Tuple[float, float]:
        vec = np.asarray(vec, dtype=float)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return (0.0, 0.0)
        return (vec[0]/norm, vec[1]/norm)

    def _cubic_hermite(self, T0, T1, n=200):
        """
        Evaluate a cubic Hermite curve with endpoints A,B and tangents T0,T1.
        Returns (X, Y) arrays of length n.
        """
        A = np.asarray([self.origin.x, self.origin.y], dtype=float)
        B = np.asarray([self.destination.x, self.destination.y], dtype=float)
        T0 = np.asarray(T0, dtype=float)
        T1 = np.asarray(T1, dtype=float)

        t = np.linspace(0.0, 1.0, n)
        h00 =  2*t**3 - 3*t**2 + 1
        h10 =      t**3 - 2*t**2 + t
        h01 = -2*t**3 + 3*t**2
        h11 =      t**3 -     t**2

        C = (h00[:,None]*A + h10[:,None]*T0 + h01[:,None]*B + h11[:,None]*T1)
        return C[:,0], C[:,1]

    def _generate_connector_points(self, tangent_scaling_factor: float) -> List[Coordinate]:
        # Generate points along the curved connector
        unit_vec_origin = self._unit_vector(self.vec_origin)
        unit_vec_destination = self._unit_vector(self.vec_destination)

        # Scale tangents by distance between points
        distance = np.linalg.norm(np.array([self.destination.x - self.origin.x, 
                                            self.destination.y - self.origin.y]))
        tangent_scale = distance * tangent_scaling_factor  # Arbitrary scaling factor for tangents

        T0 = (unit_vec_origin[0] * tangent_scale, unit_vec_origin[1] * tangent_scale)
        T1 = (unit_vec_destination[0] * tangent_scale, unit_vec_destination[1] * tangent_scale)

        x_points, y_points = self._cubic_hermite(T0, T1, n=200)
        return [Coordinate(x=x, y=y) for x, y in zip(x_points, y_points)] 

class TraversalGraphGenerator:
    def __init__(self, occupancy_map_path: str, config_path: str, meters_per_pixel: float = 0.036, factor: int = 1):
        self.occupancy_map_path = occupancy_map_path
        self.config_path = config_path
        self.meters_per_cell = meters_per_pixel * factor
        self.occupancy_map = self._load_map(occupancy_map_path)
        self.config = self._load_config(config_path)
        self.corridors = self._extract_corridors_from_config()
        self.drive_throughs = self._extract_drive_throughs_from_config()
        self.doorways = self._extract_doorways_from_config()

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
                if 'length_start' in corridor and 'length_end' in corridor and 'width_start' in corridor and 'width_end' in corridor:
                    
                    width_start_x = (corridor['width_start'][0])
                    width_start_y = (corridor['width_start'][1])
                    width_end_x = (corridor['width_end'][0])
                    width_end_y = (corridor['width_end'][1])

                    corridor_struct = Corridor(corridor_id=corridor.get('id', 'unknown'),
                        direction=corridor.get('direction', 'unknown'),
                        width_start=Coordinate(x=(width_start_x*self.meters_per_cell), y=(width_start_y*self.meters_per_cell)),
                        width_end=Coordinate(x=(width_end_x*self.meters_per_cell), y=(width_end_y*self.meters_per_cell)),
                        lanes=[])

                    corridors.append(corridor_struct)
        return corridors
    
    def _extract_drive_throughs_from_config(self) -> List[DriveThrough]:
        drive_throughs = []
        if 'drive_throughs' in self.config:
            for dt in self.config['drive_throughs']:
                if 'entry_start_point' in dt and 'entry_end_point' in dt and 'exit_start_point' in dt and 'exit_end_point' in dt:
                    entry_start = Coordinate(x=(dt['entry_start_point'][0]*self.meters_per_cell), 
                                             y=(dt['entry_start_point'][1]*self.meters_per_cell))
                    entry_end = Coordinate(x=(dt['entry_end_point'][0]*self.meters_per_cell), 
                                           y=(dt['entry_end_point'][1]*self.meters_per_cell))
                    exit_start = Coordinate(x=(dt['exit_start_point'][0]*self.meters_per_cell), 
                                            y=(dt['exit_start_point'][1]*self.meters_per_cell))
                    exit_end = Coordinate(x=(dt['exit_end_point'][0]*self.meters_per_cell), 
                                          y=(dt['exit_end_point'][1]*self.meters_per_cell))
                    drive_throughs.append(DriveThrough(entry_start, entry_end, exit_start, exit_end))
        return drive_throughs

    def _extract_doorways_from_config(self) -> List[Doorway]:
        doorways = []
        if 'doorways' in self.config:
            for dw in self.config['doorways']:
                if 'start_point' in dw and 'end_point' in dw:
                    start = Coordinate(x=(dw['start_point'][0]*self.meters_per_cell), 
                                       y=(dw['start_point'][1]*self.meters_per_cell))
                    end = Coordinate(x=(dw['end_point'][0]*self.meters_per_cell), 
                                     y=(dw['end_point'][1]*self.meters_per_cell))
                    doorways.append(Doorway(start, end))
        return doorways
    

    
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
