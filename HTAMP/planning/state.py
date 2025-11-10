import copy

import numpy as np
from HTAMP.environment.loc_dataclasses import TimeInterval
from HTAMP.environment.traversal_dataclasses import TraversalGraph, TraversalNode
from HTAMP.planning.planning_dataclasses import SimulatorConfig

class PlanningHelpers:
    @staticmethod
    def compute_cumulative_path_length(path_points: np.ndarray) -> np.ndarray:
        deltas = np.diff(path_points, axis=0)
        segment_lengths = np.linalg.norm(deltas, axis=1)
        cumulative_path_length = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        return cumulative_path_length
    
    @staticmethod
    def position_at_point(path_points: np.ndarray, cumulative_path_length: np.ndarray, traversed_distance: float) -> np.ndarray:
        s = np.clip(traversed_distance, 0.0, cumulative_path_length[-1])
        i = np.searchsorted(cumulative_path_length, s, side='right') - 1
        i = np.clip(i, 0, len(cumulative_path_length) - 2)
        t = (s - cumulative_path_length[i]) / (cumulative_path_length[i+1] - cumulative_path_length[i] + 1e-12)
        return path_points[i] * (1 - t) + path_points[i+1] * t


class PlanningState:
    def __init__(self, simulator_config: SimulatorConfig):
        self.simulator_config = simulator_config
        self.robots_positions = copy.deepcopy(simulator_config.initial_robot_positions)
        self.robots_current_node_index: dict[int, int] = {robot_id: 0 for robot_id in simulator_config.initial_robot_positions.keys()}
        self.robots_current_nodes: dict[int, TraversalNode | None] = {robot_id: None for robot_id in simulator_config.initial_robot_positions.keys()}
        self.robots_next_nodes: dict[int, TraversalNode | None] = {robot_id: None for robot_id in simulator_config.initial_robot_positions.keys()}
        self.edge_samples: dict[int, np.ndarray] = {robot_id: np.array([]) for robot_id in simulator_config.initial_robot_positions.keys()}
        self.edge_lengths: dict[int, float] = {robot_id: 0.0 for robot_id in simulator_config.initial_robot_positions.keys()}
        self.cumulative_path_lengths: dict[int, np.ndarray] = {robot_id: np.array([]) for robot_id in simulator_config.initial_robot_positions.keys()}
        self.previous_traversed_distances: dict[int, float] = {robot_id: 0.0 for robot_id in simulator_config.initial_robot_positions.keys()}

        self.current_wait_times: dict[int, float] = {robot_id: 0.0 for robot_id in simulator_config.initial_robot_positions.keys()}

        self.robot_paths: dict[int, list[tuple[TraversalNode, TimeInterval]]] = {profile.robot_id: [] for profile in simulator_config.robot_profiles}

    def _extract_edge_samples_and_cumulative_lengths(self, 
                                                start_node: TraversalNode, 
                                                end_node: TraversalNode, 
                                                traversal_graph: TraversalGraph) -> tuple[np.ndarray, np.ndarray, float]:

        edge = traversal_graph.edge_dict.get((start_node.label, end_node.label))
        samples_x = edge.edge_connector.connector_dict['X']
        samples_y = edge.edge_connector.connector_dict['Y']
        edge_length = edge.edge_connector.length()
        zipped_samples = np.array(list(zip(samples_x, samples_y)))
        cumulative_lengths = PlanningHelpers.compute_cumulative_path_length(zipped_samples)

        return zipped_samples, cumulative_lengths, edge_length

    
    def assign_robot_path(self, 
                          robot_id: int, 
                          path: list[tuple[TraversalNode, TimeInterval]], 
                          traversal_graph: TraversalGraph) -> None:
        self.robot_paths[robot_id] = path
        if len(path) >= 2:
            start_node, start_time_interval = path[0]
            end_node, _ = path[1]
            self.robots_current_nodes[robot_id] = start_node
            self.robots_next_nodes[robot_id] = end_node

            extraction_results = self._extract_edge_samples_and_cumulative_lengths(start_node, 
                                                                                   end_node, 
                                                                                   traversal_graph)
            zipped_samples, cumulative_lengths, edge_length = extraction_results
            self.edge_samples[robot_id] = zipped_samples
            self.cumulative_path_lengths[robot_id] = cumulative_lengths
            self.edge_lengths[robot_id] = edge_length
            self.current_wait_times[robot_id] = start_time_interval.end - start_time_interval.start
            self.robots_current_node_index[robot_id] = 0
        else:
            self.robots_current_nodes[robot_id] = None
            self.robots_next_nodes[robot_id] = None
            self.edge_samples[robot_id] = np.array([])
            self.cumulative_path_lengths[robot_id] = np.array([])
            self.edge_lengths[robot_id] = 0.0
            self.current_wait_times[robot_id] = 0.0
            self.robots_current_node_index[robot_id] = 0
    
    def _move_to_next_node(self, robot_id: int, traversal_graph: TraversalGraph) -> None:
        current_index = self.robots_current_node_index[robot_id] + 1
        path = self.robot_paths[robot_id]
        if current_index < len(path):
            next_index = current_index + 1
            start_node, start_time_interval = path[current_index]
            self.robots_current_nodes[robot_id] = start_node
            if next_index >= len(path):
                self.robots_next_nodes[robot_id] = None
                self.edge_samples[robot_id] = np.array([])
                self.cumulative_path_lengths[robot_id] = np.array([])
                self.edge_lengths[robot_id] = 0.0
                self.previous_traversed_distances[robot_id] = 0.0
                self.current_wait_times[robot_id] = 0.0
                self.robots_current_node_index[robot_id] = current_index
            else:
                end_node, _ = path[next_index]
                self.robots_next_nodes[robot_id] = end_node

                extraction_results = self._extract_edge_samples_and_cumulative_lengths(start_node, 
                                                                                    end_node, 
                                                                                    traversal_graph=traversal_graph)
                zipped_samples, cumulative_lengths, edge_length = extraction_results
                self.edge_samples[robot_id] = zipped_samples
                self.cumulative_path_lengths[robot_id] = cumulative_lengths
                self.edge_lengths[robot_id] = edge_length
                self.previous_traversed_distances[robot_id] = 0.0
                self.current_wait_times[robot_id] = start_time_interval.end - start_time_interval.start
                self.robots_current_node_index[robot_id] = current_index
        else:
            self.robots_current_nodes[robot_id] = None
            self.robots_next_nodes[robot_id] = None
            self.edge_samples[robot_id] = np.array([])
            self.cumulative_path_lengths[robot_id] = np.array([])
            self.edge_lengths[robot_id] = 0.0
            self.previous_traversed_distances[robot_id] = 0.0
    
    def _calculate_traversed_distance(self, robot_id: int, time_step: float) -> float:
        traversed_distance = self.previous_traversed_distances[robot_id] \
            + self.simulator_config.robot_profiles[robot_id].speed * time_step
        return traversed_distance

    def _update_robot_location(self, robot_id: int, traversal_graph: TraversalGraph, time_step: float) -> None:
        if self.robots_next_nodes[robot_id] is None:
            if self.robots_current_nodes[robot_id] is not None:
                self.robots_positions[robot_id] = self.robots_current_nodes[robot_id].position
            return

        if self.current_wait_times[robot_id] > time_step:
            self.current_wait_times[robot_id] -= time_step
            return
        elif self.current_wait_times[robot_id] > 0.0:
            time_remaining = time_step - self.current_wait_times[robot_id]
            self.current_wait_times[robot_id] = 0.0
            traversed_distance = self._calculate_traversed_distance(robot_id, time_remaining)
        else:
            traversed_distance = self._calculate_traversed_distance(robot_id, time_step)

        edge_length = self.edge_lengths[robot_id]

        if traversed_distance < edge_length:
            position = PlanningHelpers.position_at_point(self.edge_samples[robot_id], 
                                                     self.cumulative_path_lengths[robot_id], 
                                                     traversed_distance)
            self.robots_positions[robot_id] = position
            self.previous_traversed_distances[robot_id] = traversed_distance

        else:
            remaining_distance = traversed_distance - edge_length
            self._move_to_next_node(robot_id, traversal_graph)
            remaining_time = remaining_distance / self.simulator_config.robot_profiles[robot_id].speed
            self._update_robot_location(robot_id, traversal_graph, remaining_time)
        

    def step(self, traversal_graphs: dict[int, TraversalGraph]) -> None:
        for robot_id in self.robots_positions:
            self._update_robot_location(robot_id, traversal_graphs[robot_id], self.simulator_config.time_step)