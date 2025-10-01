from typing import List
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

from HTAMP.grid_world import Coordinate

@dataclass
class Corridor:
    length_start: Coordinate
    length_end: Coordinate
    width_start: Coordinate
    width_end: Coordinate

@dataclass
class DriveThrough:
    entry_start_point: Coordinate
    entry_end_point: Coordinate
    exit_start_point: Coordinate
    exit_end_point: Coordinate

@dataclass
class Doorway:
    start_point: Coordinate
    end_point: Coordinate

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
                    length_start = Coordinate(x=(corridor['length_start'][0]*self.meters_per_cell), 
                                              y=(corridor['length_start'][1]*self.meters_per_cell))
                    length_end = Coordinate(x=(corridor['length_end'][0]*self.meters_per_cell), 
                                            y=(corridor['length_end'][1]*self.meters_per_cell))
                    width_start = Coordinate(x=(corridor['width_start'][0]*self.meters_per_cell), 
                                             y=(corridor['width_start'][1]*self.meters_per_cell))
                    width_end = Coordinate(x=(corridor['width_end'][0]*self.meters_per_cell), 
                                           y=(corridor['width_end'][1]*self.meters_per_cell))
                    corridors.append(Corridor(length_start, length_end, width_start, width_end))
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
    parser.add_argument("--config_path", type=str, default="maps/FA3/occupFA3_lanes.yaml", help="Path to the configuration file")
    parser.add_argument("--occupancy_map_path", type=str, default="maps/FA3/occupancy_map.npy", help="Path to the input occupancy map")
    parser.add_argument("--factor", type=int, default=1, help="Downsampling factor")
    parser.add_argument("--meters_per_pixel", type=float, default=0.036, help="Meters per pixel in the original image")
    args = parser.parse_args()

    tg_generator = TraversalGraphGenerator(occupancy_map_path=args.occupancy_map_path,
                                           config_path=args.config_path,
                                           meters_per_pixel=args.meters_per_pixel,
                                           factor=args.factor)
