from typing import Optional
from HTAMP.environment.loc_dataclasses import TimeInterval
from HTAMP.environment.robot_dataclasses import RobotProfile
from HTAMP.environment.traversal_dataclasses import TraversalNode
from HTAMP.environment.traversal_graph_gen import TraversalGraphGenerator
from HTAMP.planning.motion_planner import MotionPlanner
from HTAMP.planning.planning_dataclasses import RequestsLists, TaskRequest
from HTAMP.planning.state import PlanningState
from HTAMP.plotting.motion_planning_plotting import MotionPlanningPlotter


class TokenPassingWithDeadlines:
    def __init__(self):
        self.alpha: float = 0.1  # Weighting factor between urgency and travel time
        self.monitoring_requests_dict: dict[str, float] = {}
        self.delivery_requests_dict: dict[str, float] = {}
        self.dummy_delivery_robot_profile = RobotProfile(radius=0.10, speed=0.20, robot_id=-1, robot_type="delivery")
    
    def _add_request_to_dict(self, 
                             request: TaskRequest, 
                             task_dict: dict[str, float],
                             state: PlanningState,
                             motion_planner: MotionPlanner,
                             traversal_graph_generator: TraversalGraphGenerator):
        if request.request_type != "medication":
            pickup_deadline = request.time_for_service - request.wait_times_at_goals_seconds[0]
        else:
            start_time = request.time_for_service - (10 * 60) + request.wait_times_at_goals_seconds[0]  # Assume a fixed 10-minute delivery time for medications
            start_node_label = request.goal_nodes[0]
            start_node = traversal_graph_generator.traversal_graph.nodes_dict[start_node_label]
            end_node_label = request.goal_nodes[1]
            end_node = traversal_graph_generator.traversal_graph.nodes_dict[end_node_label]
            sub_path = motion_planner.obtain_path_for_agent(start_traversal_node=start_node,
                                                    goal_traversal_node=end_node,
                                                    robot_profile=self.dummy_delivery_robot_profile,
                                                    current_time=start_time,
                                                    wait_time_at_goal=request.wait_times_at_goals_seconds[1],
                                                    horizon=state.simulator_config.horizon)
            planned_time = sub_path[-1][1].end if sub_path else request.time_for_service
            trip_time = planned_time - start_time
            pickup_deadline = request.time_for_service - trip_time - request.wait_times_at_goals_seconds[0]

        task_dict[request.request_id] = pickup_deadline
    
    def _remove_expired_requests_from_dict(self, 
                                           task_dict: dict[str, float], 
                                           state: PlanningState,
                                           motion_planner: MotionPlanner,
                                           traversal_graph_generator: TraversalGraphGenerator):
        for request_id, deadline in task_dict.items():
            request = state.requests[request_id]
            if not request.is_expired(state.simulator_time):
                self._add_request_to_dict(request, 
                                            task_dict=task_dict, 
                                            state=state, 
                                            motion_planner=motion_planner,
                                            traversal_graph_generator=traversal_graph_generator)
            else:
                print(f"Request {request_id} has expired and is being removed from the list.")
                task_dict.pop(request_id)
                request.mark_rejected(rejection_penalty=state.simulator_config.rejection_penalty)

    def _check_if_requests_in_dicts_expired(self, 
                                            state: PlanningState, 
                                            motion_planner: MotionPlanner, 
                                            traversal_graph_generator: TraversalGraphGenerator):
        self._remove_expired_requests_from_dict(task_dict=self.monitoring_requests_dict, 
                                                state=state,
                                                motion_planner=motion_planner,
                                                traversal_graph_generator=traversal_graph_generator)
        self._remove_expired_requests_from_dict(task_dict=self.delivery_requests_dict, 
                                                state=state,
                                                motion_planner=motion_planner,
                                                traversal_graph_generator=traversal_graph_generator)
    
    def _add_all_requests_to_dicts(self, 
                                   requests_lists: Optional[RequestsLists], 
                                   state: PlanningState, 
                                   motion_planner: MotionPlanner,
                                   traversal_graph_generator: TraversalGraphGenerator):
        if requests_lists is None:
            return
        for request in requests_lists.blood_pressure_requests + requests_lists.heart_rate_requests + \
                       requests_lists.respiratory_rate_requests + requests_lists.temperature_requests + \
                       requests_lists.oxygen_saturation_requests:
            self._add_request_to_dict(request, self.monitoring_requests_dict, state, motion_planner, traversal_graph_generator)
        for request in requests_lists.medications_requests:
            self._add_request_to_dict(request, self.delivery_requests_dict, state, motion_planner, traversal_graph_generator)
    
    def _determine_robot_locations(self, robot_id: int, state: PlanningState) -> TraversalNode:
        if state.robots_next_nodes[robot_id] is None:
            robot_location = state.robots_current_nodes[robot_id]
        else:
            robot_location =  state.robots_next_nodes[robot_id]
        return robot_location
    
    def _determine_path_for_robot(self, 
                                  request_id: str, 
                                  robot_id: int, 
                                  state: PlanningState,
                                  motion_planner: MotionPlanner,
                                  traversal_graph_generator: TraversalGraphGenerator) \
                                    -> tuple[list[tuple[TraversalNode, TimeInterval]], list[int], float]:
        current_request = state.requests[request_id]
        start_node = self._determine_robot_locations(robot_id, state)
        sub_paths: list[list[tuple[TraversalNode, TimeInterval]]] = []
        planned_goal_indices: list[int] = []
        planned_time_to_service_request: float = float('inf')
        for j, goal_node_label in enumerate(current_request.goal_nodes):
            goal_node = traversal_graph_generator.traversal_graph.nodes_dict[goal_node_label]
            current_time = state.robots_current_time[robot_id] if not sub_paths else sub_paths[-1][-1][1].end
            sub_path = motion_planner.obtain_path_for_agent(start_traversal_node=start_node,
                                                    goal_traversal_node=goal_node,
                                                    robot_profile=state.simulator_config.robot_profiles[robot_id],
                                                    current_time=current_time,
                                                    wait_time_at_goal=current_request.wait_times_at_goals_seconds[j],
                                                    horizon=state.simulator_config.horizon)
            if sub_path is None:
                sub_paths = []
                planned_goal_indices = []
                planned_time_to_service_request = float('inf')
                break
            sub_paths.append(sub_path)
            if planned_goal_indices:
                planned_goal_indices.append(planned_goal_indices[-1] + len(sub_path) - 1)
            else:
                planned_goal_indices.append(len(sub_path) - 1)
            start_node = goal_node
        
        if sub_paths:
            return_path = motion_planner.obtain_path_for_agent(start_traversal_node=sub_paths[-1][-1][0],
                                                       goal_traversal_node=state.robot_depots[robot_id],
                                                       robot_profile=state.simulator_config.robot_profiles[robot_id],
                                                       current_time=sub_paths[-1][-1][1].end,
                                                       wait_time_at_goal=120.0,
                                                       horizon=state.simulator_config.horizon)
            if return_path is not None:
                sub_paths.append(return_path)
                planned_time_to_service_request = sub_paths[-2][-1][1].end
                if planned_time_to_service_request > current_request.time_for_service:
                    final_path = []
                    planned_goal_indices = []
                    planned_time_to_service_request = float('inf')
                else:
                    final_path = motion_planner.combine_paths(sub_paths)

            else:
                final_path = []
                planned_goal_indices = []
                planned_time_to_service_request = float('inf')
        else:
            final_path = []
            planned_goal_indices = []
            planned_time_to_service_request = float('inf')
        return final_path, planned_goal_indices, planned_time_to_service_request
    
    def _heuristic_cost_for_robot(self, 
                                  request_id: str, 
                                  robot_id: int, 
                                  state: PlanningState,
                                  motion_planner: MotionPlanner,
                                  traversal_graph_generator: TraversalGraphGenerator) \
                                    -> float:
        current_request = state.requests[request_id]
        start_node = self._determine_robot_locations(robot_id, state)
        trip_time: float = 0.0

        for j, goal_node_label in enumerate(current_request.goal_nodes):
            goal_node = traversal_graph_generator.traversal_graph.nodes_dict[goal_node_label]
            subpath_length = motion_planner.planner.heuristic(start_traversal_node=start_node,
                                                              goal_traversal_node=goal_node,
                                                              robot_profile=state.simulator_config.robot_profiles[robot_id])
            trip_time += subpath_length + current_request.wait_times_at_goals_seconds[j]
            start_node = goal_node

        return trip_time
    
    def _determine_closest_request(self,
                                   robot_id: int,
                                   request_dict: dict[str, float],
                                   state: PlanningState,
                                   motion_planner: MotionPlanner,
                                   traversal_graph_generator: TraversalGraphGenerator) -> Optional[str]:
        closest_request_id = None
        smallest_cost = float('inf')
        for request_id in request_dict.keys():
            request_pickup_deadline = request_dict[request_id]
            trip_time = self._heuristic_cost_for_robot(request_id=request_id,
                                                         robot_id=robot_id,
                                                         state=state,
                                                         motion_planner=motion_planner,
                                                         traversal_graph_generator=traversal_graph_generator)
            
            current_cost = (self.alpha * (request_pickup_deadline - state.robots_current_time[robot_id])) + \
                            (1 - self.alpha) * (trip_time)
            if current_cost < smallest_cost:
                smallest_cost = current_cost
                closest_request_id = request_id

        return closest_request_id
    
    def _assign_request_to_robot(self,
                                 state: PlanningState,
                                 request_id: str, 
                                 robot_id: int, 
                                 planned_path: list[list[tuple[TraversalNode, TimeInterval]]], 
                                 planned_time: float,
                                 planned_goal_indices: list[int],
                                 motion_planner: MotionPlanner,
                                 traversal_graph_generator: TraversalGraphGenerator,
                                 debug: bool):
        state.requests[request_id].schedule_task(planned_time=planned_time,
                                                planned_goal_indices=planned_goal_indices,
                                                assigned_robot_id=robot_id)

        if debug:
            MotionPlanningPlotter.plot_assigned_path(
                occupancy_map=traversal_graph_generator.occupancy_map,
                origin_x=traversal_graph_generator.origin_x,
                origin_y=traversal_graph_generator.origin_y,
                resolution=traversal_graph_generator.resolution,
                request_id=request_id,
                results_folder="results/motion_planning/debug",
                planned_path=planned_path,
                traversal_graph=traversal_graph_generator.traversal_graph,
                robot_profile=state.simulator_config.robot_profiles[robot_id]
            )
            
        motion_planner.clear_reservations_for_agent(robot_profile=state.simulator_config.robot_profiles[robot_id])
        motion_planner.reserve_path_for_agent(path=planned_path,
                                                robot_profile=state.simulator_config.robot_profiles[robot_id],
                                                wait_time_at_goal=state.simulator_config.horizon)
        
        state.assign_request_to_robot(request_id=request_id, 
                                    robot_id=robot_id, 
                                    path=planned_path, 
                                    traversal_graph=traversal_graph_generator.traversal_graph)
    
    def _assign_requests_to_available_robots(self,
                                            state: PlanningState,
                                            motion_planner: MotionPlanner,
                                            traversal_graph_generator: TraversalGraphGenerator,
                                            robot_type: str,
                                            requests_dict: dict[str, float],
                                            debug: bool):
        available_robots = state.get_available_robots(robot_type=robot_type)
        print(f"Available robots for {robot_type}: {available_robots}")
        for robot_id in available_robots:
            closest_request_id = self._determine_closest_request(robot_id=robot_id,
                                                                 request_dict=requests_dict,
                                                                 state=state,
                                                                 motion_planner=motion_planner,
                                                                 traversal_graph_generator=traversal_graph_generator)
            
            print(f"Robot {robot_id} closest request: {closest_request_id}")
            if closest_request_id is not None:
                planning_results = self._determine_path_for_robot(request_id=closest_request_id,
                                                                  robot_id=robot_id,
                                                                  state=state,
                                                                  motion_planner=motion_planner,
                                                                  traversal_graph_generator=traversal_graph_generator)
                
                closest_path, closest_planned_goal_indices, shortest_time = planning_results
                self._assign_request_to_robot(state=state,
                                              request_id=closest_request_id,
                                              robot_id=robot_id,
                                              planned_path=closest_path,
                                              planned_time=shortest_time,
                                              planned_goal_indices=closest_planned_goal_indices,
                                              motion_planner=motion_planner,
                                              traversal_graph_generator=traversal_graph_generator,
                                              debug=debug)
                
                requests_dict.pop(closest_request_id)
            else:
                print(f"No feasible request found for robot {robot_id}")
    
    def _assign_requests_for_monitoring_robots(self, 
                                               state: PlanningState, 
                                               motion_planner: MotionPlanner, 
                                               traversal_graph_generator: TraversalGraphGenerator,
                                               debug: bool):
        self._assign_requests_to_available_robots(state=state,
                                                 motion_planner=motion_planner,
                                                 traversal_graph_generator=traversal_graph_generator,
                                                 robot_type="monitoring",
                                                 requests_dict=self.monitoring_requests_dict,
                                                 debug=debug)
    
    def _assign_requests_for_delivery_robots(self, 
                                             state: PlanningState, 
                                             motion_planner: MotionPlanner, 
                                             traversal_graph_generator: TraversalGraphGenerator,
                                             debug: bool):
        self._assign_requests_to_available_robots(state=state,
                                                 motion_planner=motion_planner,
                                                 traversal_graph_generator=traversal_graph_generator,
                                                 robot_type="delivery",
                                                 requests_dict=self.delivery_requests_dict,
                                                 debug=debug)

    def assign_requests_to_robots(self, 
                                  state: PlanningState,
                                  requests_lists: Optional[RequestsLists], 
                                  motion_planner: MotionPlanner,
                                  traversal_graph_generator: TraversalGraphGenerator,
                                  debug: bool):
        # Add new requests to the appropriate dicts
        self._check_if_requests_in_dicts_expired(state=state, 
                                                 motion_planner=motion_planner,
                                                 traversal_graph_generator=traversal_graph_generator)
        self._add_all_requests_to_dicts(requests_lists=requests_lists, 
                                        state=state, 
                                        motion_planner=motion_planner,
                                        traversal_graph_generator=traversal_graph_generator)

        # Assignment logic for monitoring robots
        self._assign_requests_for_monitoring_robots(state=state, 
                                                   motion_planner=motion_planner, 
                                                   traversal_graph_generator=traversal_graph_generator,
                                                   debug=debug)
        
        # Assignment logic for delivery robots
        self._assign_requests_for_delivery_robots(state=state, 
                                                 motion_planner=motion_planner, 
                                                 traversal_graph_generator=traversal_graph_generator,
                                                 debug=debug)