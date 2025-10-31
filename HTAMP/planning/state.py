import copy

import numpy as np
from HTAMP.environment.loc_dataclasses import TimeInterval
from HTAMP.environment.traversal_dataclasses import TraversalNode
from HTAMP.planning.planning_dataclasses import SimulatorConfig

class PlanningHelpers:
    @staticmethod
    def compute_cumulative_path_length(path_points):
        deltas = np.diff(path_points, axis=0)
        segment_lengths = np.linalg.norm(deltas, axis=1)
        cumulative_path_length = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        return cumulative_path_length
    
    @staticmethod
    def position_at_point(path_points, cumulative_path_length, traversed_distance):
        s = np.clip(traversed_distance, 0.0, cumulative_path_length[-1])
        i = np.searchsorted(cumulative_path_length, s, side='right') - 1
        i = np.clip(i, 0, len(cumulative_path_length) - 2)
        t = (s - cumulative_path_length[i]) / (cumulative_path_length[i+1] - cumulative_path_length[i] + 1e-12)
        return path_points[i] * (1 - t) + path_points[i+1] * t


class PlanningState:
    def __init__(self, simulator_config: SimulatorConfig):
        self.simulator_config = simulator_config
        self.robots_positions = copy.deepcopy(simulator_config.initial_robot_positions)
        self.robot_paths: dict[int, list[tuple[TraversalNode, TimeInterval]]] = {profile.robot_id: [] for profile in simulator_config.robot_profiles}
        self.cumulative_path_lengths: dict[int, np.ndarray] = {profile.robot_id: np.array([]) for profile in simulator_config.robot_profiles}
        self.previous_traversed_distances: dict[int, float] = {profile.robot_id: 0.0 for profile in simulator_config.robot_profiles}

    def set_robot_path(self, robot_id, path_points):
        self.robot_paths[robot_id] = path_points

    def get_robot_path(self, robot_id):
        return self.robot_paths.get(robot_id, [])
    
    def get_robot_position(self, robot_id):
        return self.robots_positions.get(robot_id, None)

    def set_robot_position(self, robot_id, position):
        self.robots_positions[robot_id] = position
    
    def update_robot_path(self, robot_id, path: list[tuple[TraversalNode, TimeInterval]]):
        self.robot_paths[robot_id] = path

    def calculate_traversed_distance(self, robot_id):
        path = self.robot_paths.get(robot_id, [])
        if not path:
            return 0.0
        cumulative_path_length = PlanningHelpers.compute_cumulative_path_length(path)
        current_position = self.robots_positions.get(robot_id, None)
        if current_position is None:
            return 0.0
        distances = np.linalg.norm(path - current_position, axis=1)
        closest_index = np.argmin(distances)
        return cumulative_path_length[closest_index]

    def _update_robot_location(self, robot_id):
        # TODO: update
        if robot_id in self.robots_positions:
            path = self.robot_paths[robot_id]
            if path:
                traversed_distance = self.previous_traversed_distances[robot_id] + self.simulator_config.robot_speed * self.simulator_config.time_step
                next_position = PlanningHelpers.position_at_point(path, PlanningHelpers.compute_cumulative_path_length(path), traversed_distance)
                self.robots_positions[robot_id] = next_position

    def step(self):
        for robot_id in self.robots_positions:
            traversed_distance = self.calculate_traversed_distance(robot_id)
            self._update_robot_location(robot_id, traversed_distance)