import argparse
import copy
from datetime import datetime
import math
import random
import traceback

import numpy as np
import pandas as pd
from HTAMP.environment.grid_world import GridWorld
from HTAMP.environment.loc_dataclasses import Coordinate, TimeInterval
from HTAMP.environment.robot_dataclasses import RobotProfile
from HTAMP.environment.traversal_dataclasses import TraversalGraph, TraversalNode
from HTAMP.environment.traversal_graph_gen import TraversalGraphGenerator
from HTAMP.planning.motion_planner import MotionPlanner
from HTAMP.planning.planning_dataclasses import SimulatorConfig, TaskRequest
from HTAMP.plotting.motion_planning_plotting import MotionPlanningPlotter

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
        return path_points[i] * (1 - t) + path_points[i+1] * t, i+1


class PlanningState:
    def __init__(self, simulator_config: SimulatorConfig):
        self.simulator_config = simulator_config
        self.simulator_time = 0.0
        self.robot_depots = copy.deepcopy(simulator_config.initial_nodes)
        self.robots_positions = copy.deepcopy(simulator_config.initial_robot_positions)
        self.robots_current_time: dict[int, float] = {robot_id: 0.0 for robot_id in simulator_config.initial_robot_positions.keys()}
        self.robots_current_node_index: dict[int, int] = {robot_id: 0 for robot_id in simulator_config.initial_robot_positions.keys()}
        self.robots_current_nodes: dict[int, TraversalNode] = copy.deepcopy(simulator_config.initial_nodes)
        self.robots_next_nodes: dict[int, TraversalNode | None] = {robot_id: None for robot_id in simulator_config.initial_robot_positions.keys()}
        self.edge_samples: dict[int, np.ndarray] = {robot_id: np.array([]) for robot_id in simulator_config.initial_robot_positions.keys()}
        self.edge_lengths: dict[int, float] = {robot_id: 0.0 for robot_id in simulator_config.initial_robot_positions.keys()}
        self.cumulative_path_lengths: dict[int, np.ndarray] = {robot_id: np.array([]) for robot_id in simulator_config.initial_robot_positions.keys()}
        self.previous_traversed_distances: dict[int, float] = {robot_id: 0.0 for robot_id in simulator_config.initial_robot_positions.keys()}
        self.point_indices_on_edge: dict[int, int] = {robot_id: 0 for robot_id in simulator_config.initial_robot_positions.keys()}

        self.current_wait_times: dict[int, float] = {robot_id: 0.0 for robot_id in simulator_config.initial_robot_positions.keys()}

        self.robot_paths: dict[int, list[tuple[TraversalNode, TimeInterval]]] = {key: [] for key in simulator_config.robot_profiles.keys()}

        self.requests: dict[int, TaskRequest] = {}

        self.assigned_requests: dict[int, list[int]] = {key: [] for key in simulator_config.robot_profiles.keys()}

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
    
    def add_request(self, request: TaskRequest) -> None:
        self.requests[request.request_id] = request

    def add_new_requests(self, requests: list[TaskRequest]) -> None:
        for request in requests:
            self.add_request(request=request)
    
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
            self.point_indices_on_edge[robot_id] = 0
        else:
            self.robots_next_nodes[robot_id] = None
            self.edge_samples[robot_id] = np.array([])
            self.cumulative_path_lengths[robot_id] = np.array([])
            self.edge_lengths[robot_id] = 0.0
            self.current_wait_times[robot_id] = 0.0
            self.robots_current_node_index[robot_id] = 0
            self.point_indices_on_edge[robot_id] = 0
    
    def assign_request_to_robot(self, 
                                robot_id: int, 
                                request_id: int, 
                                path: list[tuple[TraversalNode, TimeInterval]],
                                traversal_graph: TraversalGraph) -> None:
        self.assign_robot_path(robot_id=robot_id, path=path, traversal_graph=traversal_graph)
        self.assigned_requests[robot_id].append(request_id)
    
    def reassign_requests_to_robot(self, 
                                 robot_id: int, 
                                 request_ids: list[int],
                                 path: list[tuple[TraversalNode, TimeInterval]],
                                 traversal_graph: TraversalGraph) -> None:
        self.assigned_requests[robot_id] = request_ids
        self.assign_robot_path(robot_id=robot_id, path=path, traversal_graph=traversal_graph)

    def _check_if_next_node_is_task_start(self, robot_id: int, traversal_node: TraversalNode) -> None:
        assigned_requests = self.assigned_requests[robot_id]
        if assigned_requests:
            current_request_id = assigned_requests[0]
            current_request = self.requests[current_request_id]
            if not current_request.started:
                if traversal_node.label == current_request.goal_nodes[0]:
                    current_request.mark_started()
    
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
                self.point_indices_on_edge[robot_id] = 0
                self.robots_current_time[robot_id] = self.simulator_time
            else:
                end_node, end_time_interval = path[next_index]
                self.robots_next_nodes[robot_id] = end_node

                self._check_if_next_node_is_task_start(robot_id=robot_id,
                                                       traversal_node=end_node)

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
                self.point_indices_on_edge[robot_id] = 0
                if next_index == len(path) - 1:
                    self.robots_current_time[robot_id] = end_time_interval.start
                else:
                    self.robots_current_time[robot_id] = end_time_interval.end
        else:
            self.robots_next_nodes[robot_id] = None
            self.edge_samples[robot_id] = np.array([])
            self.cumulative_path_lengths[robot_id] = np.array([])
            self.edge_lengths[robot_id] = 0.0
            self.previous_traversed_distances[robot_id] = 0.0
            self.current_wait_times[robot_id] = 0.0
            self.robots_current_node_index[robot_id] = 0
            self.point_indices_on_edge[robot_id] = 0
            self.robots_current_time[robot_id] = self.simulator_time

    def _calculate_traversed_distance(self, robot_id: int, time_step: float) -> float:
        traversed_distance = self.previous_traversed_distances[robot_id] \
            + self.simulator_config.robot_profiles[robot_id].speed * time_step
        return traversed_distance
    
    def _check_if_final_objective_is_reached(self, robot_id: int) -> None:
        path = self.robot_paths[robot_id]
        current_index = self.robots_current_node_index[robot_id]
        if current_index < len(path):
            current_node, time_interval = path[current_index]
            assigned_requests = self.assigned_requests[robot_id]
            if assigned_requests:
                current_request_id = assigned_requests[0]
                current_request = self.requests[current_request_id]
                if current_request.completed_goals < len(current_request.goal_nodes):
                    goal_node_label = current_request.goal_nodes[current_request.completed_goals]
                    if current_node.label == goal_node_label:
                        current_request.completed_goals += 1
                        if current_request.completed_goals >= len(current_request.goal_nodes):
                            assert math.isclose(time_interval.end, self.simulator_time + self.current_wait_times[robot_id]), \
                                f"Time mismatch at goal for robot {robot_id}: expected {time_interval.end}, got {self.simulator_time + self.current_wait_times[robot_id]}"
                            completed_request_id = self.assigned_requests[robot_id].pop(0)
                            completed_request = self.requests[completed_request_id]
                            completed_request.mark_completed(completion_time=self.simulator_time + self.current_wait_times[robot_id])

    def _update_robot_location(self, robot_id: int, traversal_graph: TraversalGraph, time_step: float) -> None:
        if self.robots_next_nodes[robot_id] is None:
            self.robots_positions[robot_id] = self.robots_current_nodes[robot_id].position
            self.robots_current_time[robot_id] = self.simulator_time + time_step
            return

        if self.current_wait_times[robot_id] > time_step:
            self.current_wait_times[robot_id] -= time_step
            return
        elif self.current_wait_times[robot_id] > 0.0:
            time_remaining = time_step - self.current_wait_times[robot_id]
            self._check_if_final_objective_is_reached(robot_id)
            self.current_wait_times[robot_id] = 0.0
            traversed_distance = self._calculate_traversed_distance(robot_id, time_remaining)
        else:
            traversed_distance = self._calculate_traversed_distance(robot_id, time_step)

        edge_length = self.edge_lengths[robot_id]

        if traversed_distance < edge_length:
            position, pos_index = PlanningHelpers.position_at_point(self.edge_samples[robot_id], 
                                                     self.cumulative_path_lengths[robot_id], 
                                                     traversed_distance)
            self.point_indices_on_edge[robot_id] = pos_index
            robot_position = Coordinate(x=position[0], y=position[1])
            self.robots_positions[robot_id] = robot_position
            self.previous_traversed_distances[robot_id] = traversed_distance

        else:
            remaining_distance = traversed_distance - edge_length
            self._move_to_next_node(robot_id, traversal_graph)
            remaining_time = remaining_distance / self.simulator_config.robot_profiles[robot_id].speed
            self._update_robot_location(robot_id, traversal_graph, remaining_time)
    
    def get_available_robots(self, robot_type: str) -> list[int]:
        available_robots = []
        for robot_id in self.robots_positions:
            if not self.assigned_requests[robot_id]:
                profile = self.simulator_config.robot_profiles[robot_id]
                if profile.robot_type == robot_type:
                    available_robots.append(robot_id)
        return available_robots

    def step(self, traversal_graph: TraversalGraph) -> None:
        for robot_id in self.robots_positions:
            self._update_robot_location(robot_id, traversal_graph, self.simulator_config.time_step)
        self.simulator_time += self.simulator_config.time_step
    
    def get_completed_requests(self) -> dict[str, list[TaskRequest]]:
        completed_requests = {}
        for request in self.requests.values():
            if request.completed:
                if request.request_type not in completed_requests:
                    completed_requests[request.request_type] = []
                completed_requests[request.request_type].append(request)
        return completed_requests
    
    def get_rejected_requests(self) -> dict[str, list[TaskRequest]]:
        rejected_requests = {}
        for request in self.requests.values():
            if request.rejected:
                if request.request_type not in rejected_requests:
                    rejected_requests[request.request_type] = []
                rejected_requests[request.request_type].append(request)
        return rejected_requests
    
    def compute_total_costs_for_completed_requests(self) -> dict[str, float]:
        total_costs = {}
        for request_type, request_list in self.get_completed_requests().items():
            total_cost = 0.0
            for request in request_list:
                total_cost += request.total_cost
            total_costs[request_type] = total_cost
        return total_costs
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="maps/hospital_floor/floor_config.yaml", help="Path to the configuration file")
    parser.add_argument("--occupancy_map_path", type=str, default="maps/hospital_floor/occupancy_map.npy", help="Path to the input occupancy map")
    parser.add_argument("--factor", type=int, default=1, help="Downsampling factor")
    parser.add_argument("--meters_per_pixel", type=float, default=0.036, help="Meters per pixel in the original image")
    parser.add_argument("--fps", type=float, default=2.0, help="Frames per second for the grid world")
    parser.add_argument("--occupancy_reservations_file", type=str, default="data/occupancy_reservations.pkl", help="Path to the occupancy reservations file")
    parser.add_argument("--use_saved_data", action='store_true', help="Whether to use saved occupancy reservations data")
    parser.add_argument("--num_robots", type=int, default=1, help="Number of robots to plan for")
    args = parser.parse_args()

    print("Generating Traversal Graph...")

    tg_generator = TraversalGraphGenerator(occupancy_map_path=args.occupancy_map_path,
                                           config_path=args.config_path,
                                           meters_per_pixel=args.meters_per_pixel,
                                           factor=args.factor)

    print("Traversal Graph generated.")

    potential_target_nodes = []

    for doorway in tg_generator.doorway_subgraphs:
        room_nodes = doorway.room_nodes
        for room_node_label in room_nodes:
            room_node = tg_generator.traversal_graph.nodes_dict[room_node_label]
            potential_target_nodes.append(room_node)
    
    potential_start_nodes = []
    for parking_space in tg_generator.parking_spaces_subgraphs:
        entry_nodes = parking_space.up_parking_nodes_exit + parking_space.down_parking_nodes_exit
        for entry_node_label in entry_nodes:
            entry_node = tg_generator.traversal_graph.nodes_dict[entry_node_label]
            potential_start_nodes.append(entry_node)
            
    robot_profiles = {}

    # randomly select start and goal nodes for each robot
    random.seed(11)
    selected_start_nodes = random.sample(potential_start_nodes, args.num_robots)
    selected_goal_nodes = random.sample(potential_target_nodes, 2*args.num_robots)

    selected_start_nodes.reverse()
    selected_goal_nodes.reverse()  # to avoid selecting the same node as start and goal

    for i in range(args.num_robots):
        robot_profile = RobotProfile(radius=0.10, speed=0.20, robot_id=i)
        robot_profiles[i] = robot_profile

    print("Creating Grid World...")

    world = GridWorld(cell_size=tg_generator.meters_per_cell,
                      fps=args.fps,
                      occupancy_map=tg_generator.occupancy_map,
                      traversal_graph=tg_generator.traversal_graph,
                      shortest_paths=tg_generator.shortest_paths,
                      robot_profiles=robot_profiles,
                      use_saved_data=args.use_saved_data,
                      occupancy_reservations_file=args.occupancy_reservations_file)

    print("Grid World created.")

    planner = MotionPlanner(grid=world, weight_factor=1.0)

    paths = []

    initial_time = pd.Timestamp(2024, 1, 1, 0, 0, 0)

    simulator_config = SimulatorConfig(fps=args.fps,
                                       robot_profiles=robot_profiles,
                                       rejection_penalty=100.0,
                                       initial_time=initial_time,
                                       initial_robot_positions={i: selected_start_nodes[i].position for i in range(args.num_robots)},
                                       horizon=5000.0)
    
    state = PlanningState(simulator_config=simulator_config)

    pStart = datetime.now()

    requests: list[TaskRequest] = []

    for i in range(args.num_robots):
        planner._initialize_robot_reservations(initial_node=selected_start_nodes[i],
                                               robot_profile=robot_profiles[i],
                                               current_time=0.0,
                                               horizon=simulator_config.horizon)
        ordered_timestamp = pd.Timestamp(2024, 1, 1, 0, 0, 0)
        scheduled_timestamp = ordered_timestamp + pd.Timedelta(minutes=5 * i)

        ordered_time = (ordered_timestamp - initial_time).total_seconds()
        scheduled_time = (scheduled_timestamp - initial_time).total_seconds()

        print(f"Robot {i} - Ordered Time: {ordered_time}, Scheduled Time: {scheduled_time}")
        
        current_request = TaskRequest(request_id=i, 
                                      request_type="move", 
                                      goal_nodes=[selected_goal_nodes[i].label, selected_goal_nodes[i + args.num_robots].label], 
                                      wait_times_at_goals_seconds=[10.0, 10.0],
                                      time_for_rejection_minutes=30.0,
                                      ordered_time=ordered_time,
                                      scheduled_time=scheduled_time)
        
        state.add_request(current_request)
        
        requests.append(current_request)
        

    for i in range(args.num_robots):
        current_request = requests[i]
        start_node = selected_start_nodes[i]
        sub_paths: list[list[tuple[TraversalNode, TimeInterval]]] = []
        planned_goal_indices: list[int] = []
        for j, goal_node_label in enumerate(current_request.goal_nodes):
            goal_node = tg_generator.traversal_graph.nodes_dict[goal_node_label]
            current_time = 0.0 if not sub_paths else sub_paths[-1][-1][1].end
            sub_path = planner.obtain_path_for_agent(start_traversal_node=start_node,
                                                    goal_traversal_node=goal_node,
                                                    robot_profile=robot_profiles[i],
                                                    current_time=current_time,
                                                    wait_time_at_goal=current_request.wait_times_at_goals_seconds[j],
                                                    horizon=simulator_config.horizon)
            if not sub_path:
                sub_paths = []
                break
            sub_paths.append(sub_path)
            if planned_goal_indices:
                planned_goal_indices.append(planned_goal_indices[-1] + len(sub_path) - 1)
            else:
                planned_goal_indices.append(len(sub_path) - 1)
            start_node = goal_node
        
        if sub_paths:
            return_path = planner.obtain_path_for_agent(start_traversal_node=sub_paths[-1][-1][0],
                                                       goal_traversal_node=selected_start_nodes[i],
                                                       robot_profile=robot_profiles[i],
                                                       current_time=sub_paths[-1][-1][1].end,
                                                       wait_time_at_goal=simulator_config.horizon,
                                                       horizon=simulator_config.horizon)
            if return_path:
                sub_paths.append(return_path)
                current_request.schedule_task(planned_time=sub_paths[-2][-1][1].end,
                                             planned_goal_indices=planned_goal_indices)
                planner.clear_reservations_for_agent(robot_profile=robot_profiles[i])
                final_path = planner.combine_paths(sub_paths)
                planner.reserve_path_for_agent(path=final_path, 
                                           robot_profile=robot_profiles[i], 
                                           wait_time_at_goal=simulator_config.horizon)
                paths.append(final_path)
                state.assign_request_to_robot(robot_id=i, 
                                              request_id=current_request.request_id, 
                                              path=final_path, 
                                              traversal_graph=tg_generator.traversal_graph)
                
                for k, planned_goal_index in enumerate(planned_goal_indices):
                    node_at_goal, time_interval_at_goal = final_path[planned_goal_index]
                    request_goal_node_label = current_request.goal_nodes[k]
                    assert node_at_goal.label == request_goal_node_label, \
                        f"Mismatch in planned goal node: expected {request_goal_node_label}, got {node_at_goal.label}"
                    
                print(f"Planned Path for Robot {i}:")
                for traversal_node, time_interval in final_path:
                    print(f"Node: ({traversal_node.label}), Time: [{time_interval.start:.2f}, {time_interval.end:.2f}]")
            else:
                print(f"No return path found for Robot {i}")
        else:
            print(f"No path found for Robot {i}")

    pEnd = datetime.now()
    
    MotionPlanningPlotter.plot_paths(occupancy_map=tg_generator.occupancy_map,
                                origin_x=tg_generator.origin_x,
                                origin_y=tg_generator.origin_y,
                                resolution=tg_generator.meters_per_cell,
                                paths=paths,
                                traversal_graph=tg_generator.traversal_graph,
                                robot_profiles=robot_profiles)
    
    robot_positions_seq: list[dict[int, Coordinate]] = []
    robots_current_node_index_seq: list[dict[int, int]] =[]
    point_indices_on_edge_seq: list[dict[int, int]] =[]
    robot_paths_seq: list[dict[int, list[tuple[TraversalNode, TimeInterval]]]] = []
    planned_goal_indices_seq: list[dict[int, list[int]]] = []
    completed_goals_seq: list[dict[int, int]] = []
    
    for i in range(1000):
        print(f"Step {i}:")
        state.step(traversal_graph=tg_generator.traversal_graph)
        robot_positions_seq.append(copy.deepcopy(state.robots_positions))
        robots_current_node_index_seq.append(copy.deepcopy(state.robots_current_node_index))
        point_indices_on_edge_seq.append(copy.deepcopy(state.point_indices_on_edge))
        robot_paths_seq.append(copy.deepcopy(state.robot_paths))
        planned_goal_indices_dict: dict[int, list[int]] = {}
        completed_goals_dict: dict[int, int] = {}
        for robot_id, requests in state.assigned_requests.items():
            if requests:
                current_request_id = requests[0]
                current_request = state.requests[current_request_id]
                planned_goal_indices_dict[robot_id] = current_request.planned_goal_indices
                completed_goals_dict[robot_id] = current_request.completed_goals
        planned_goal_indices_seq.append(copy.deepcopy(planned_goal_indices_dict))
        completed_goals_seq.append(copy.deepcopy(completed_goals_dict))
    
    print(state.get_completed_requests())
    
    MotionPlanningPlotter.generate_state_animation(occupancy_map=tg_generator.occupancy_map,
                                            origin_x=tg_generator.origin_x,
                                            origin_y=tg_generator.origin_y,
                                            resolution=tg_generator.meters_per_cell,
                                            robot_positions_seq=robot_positions_seq,
                                            robots_current_node_index_seq=robots_current_node_index_seq,
                                            point_indices_on_edge_seq=point_indices_on_edge_seq,
                                            robot_paths_seq=robot_paths_seq,
                                            planned_goal_indices_seq=planned_goal_indices_seq,
                                            completed_goals_seq=completed_goals_seq,
                                            traversal_graph=tg_generator.traversal_graph,
                                            robot_profiles=robot_profiles,
                                            fps_sim=args.fps,
                                            num_sim_frames=1000)
    print(f"Total planning time: {pEnd - pStart}")


if __name__ == "__main__":
    pStart = datetime.now()
    try:
        main()
    except Exception as errorMainContext:
        print("Fail End Process: ", errorMainContext)
        traceback.print_exc()
    pEnd = datetime.now()
    print(f"Total Execution Time: {pEnd - pStart}")