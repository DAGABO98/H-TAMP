from typing import Optional
from HTAMP.environment.loc_dataclasses import TimeInterval
from HTAMP.environment.robot_dataclasses import RobotProfile
from HTAMP.environment.traversal_dataclasses import TraversalNode
from HTAMP.environment.traversal_graph_gen import TraversalGraphGenerator
from HTAMP.planning.motion_planner import MotionPlanner
from HTAMP.planning.planning_dataclasses import RequestsLists, TaskRequest
from HTAMP.planning.state import PlanningState
from HTAMP.plotting.motion_planning_plotting import MotionPlanningPlotter

class DeadlineAwareTokenPassingwithTaskSwaps():
    
    def __init__(self):
        self.alpha: float = 0.1  # Weighting factor between urgency and travel time
        self.monitoring_requests_dict: dict[str, float] = {}
        self.assigned_monitoring_requests_dict: dict[str, tuple[int, float]] = {}
        self.delivery_requests_dict: dict[str, float] = {}
        self.assigned_delivery_requests_dict: dict[str, float] = {}
        self.dummy_delivery_robot_profile = RobotProfile(radius=0.10, speed=0.20, robot_id=-1, robot_type="delivery")
    
    def _calculate_pickup_deadline(self, 
                                   request: TaskRequest,
                                   state: PlanningState,
                                   motion_planner: MotionPlanner,
                                   traversal_graph_generator: TraversalGraphGenerator) -> float:
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
        return pickup_deadline
    
    def _add_request_to_dict(self, 
                             request: TaskRequest, 
                             task_dict: dict[str, float],
                             state: PlanningState,
                             motion_planner: MotionPlanner,
                             traversal_graph_generator: TraversalGraphGenerator,
                             assigned: bool = False):
        pickup_deadline = self._calculate_pickup_deadline(request=request,
                                                          state=state,
                                                          motion_planner=motion_planner,
                                                          traversal_graph_generator=traversal_graph_generator)

        if assigned:
            task_dict[request.request_id] = (request.assigned_robot_id, pickup_deadline)
        else:
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
                                            traversal_graph_generator=traversal_graph_generator,
                                            assigned=False)
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
    
    def _remove_started_requests_from_dict(self,
                                           task_dict: dict[str, tuple[int, float]],
                                           state: PlanningState,
                                           motion_planner: MotionPlanner,
                                           traversal_graph_generator: TraversalGraphGenerator):
        for request_id in list(task_dict.keys()):
            request = state.requests[request_id]
            if not request.is_started():
                self._add_request_to_dict(request, 
                                          task_dict=task_dict, 
                                          state=state, 
                                          motion_planner=motion_planner,
                                          traversal_graph_generator=traversal_graph_generator,
                                          assigned=True)
            else:
                print(f"Request {request_id} has started and is being removed from the list.")
                task_dict.pop(request_id)
    
    def _check_if_requests_in_dicts_started(self,
                                         state: PlanningState,
                                         motion_planner: MotionPlanner,
                                         traversal_graph_generator: TraversalGraphGenerator):
        self._remove_started_requests_from_dict(task_dict=self.assigned_monitoring_requests_dict,
                                                state=state,
                                                motion_planner=motion_planner,
                                                traversal_graph_generator=traversal_graph_generator)
        self._remove_started_requests_from_dict(task_dict=self.assigned_delivery_requests_dict,
                                                state=state,
                                                motion_planner=motion_planner,
                                                traversal_graph_generator=traversal_graph_generator)
        
    def _check_if_new_request_triggers_reassignment(self,
                                                    request: TaskRequest,
                                                    state: PlanningState,
                                                    unassigned_requests_dict: dict[str, float],
                                                    assigned_requests_dict: dict[str, tuple[int, float]],
                                                    motion_planner: MotionPlanner,
                                                    traversal_graph_generator: TraversalGraphGenerator):
        self._add_request_to_dict(request=request,
                                  task_dict=unassigned_requests_dict,
                                  state=state,
                                  motion_planner=motion_planner,
                                  traversal_graph_generator=traversal_graph_generator,
                                  assigned=False)
        request_pickup_deadline = unassigned_requests_dict[request.request_id]
        for assigned_request_id, (assigned_robot_id, assigned_deadline) in assigned_requests_dict.items():
            if request_pickup_deadline < assigned_deadline:
                # Check if the newly arrived request can be assigned to the robot
                new_trip_time = self._heuristic_cost_for_robot(request_id=request.request_id,
                                                                robot_id=assigned_robot_id,
                                                                state=state,
                                                                motion_planner=motion_planner,
                                                                traversal_graph_generator=traversal_graph_generator)
                
                prev_trip_time = self._heuristic_cost_for_robot(request_id=assigned_request_id,
                                                                 robot_id=assigned_robot_id,
                                                                 state=state,
                                                                 motion_planner=motion_planner,
                                                                 traversal_graph_generator=traversal_graph_generator)
                
                if new_trip_time < prev_trip_time:
                    assigned_request = state.requests[assigned_request_id]
                    print(f"Reassigning request {assigned_request_id} from robot {assigned_robot_id} to accommodate new request {request.request_id}.")
                    self._add_request_to_dict(request=assigned_request,
                                                task_dict=unassigned_requests_dict,
                                                state=state,
                                                motion_planner=motion_planner,
                                                traversal_graph_generator=traversal_graph_generator,
                                                assigned=False)
                    assigned_requests_dict.pop(assigned_request_id)
                    assigned_request.reset_assignment()
                    state.remove_request_from_robot(robot_id=assigned_robot_id,
                                                    request_id=assigned_request_id)
                    motion_planner.clear_reservations_for_agent(robot_profile=state.simulator_config.robot_profiles[assigned_robot_id])
    
    def _add_new_requests_to_dicts(self,
                                   requests_lists: Optional[RequestsLists], 
                                   state: PlanningState, 
                                   motion_planner: MotionPlanner,
                                   traversal_graph_generator: TraversalGraphGenerator):
        if requests_lists is None:
            return
        for request in requests_lists.blood_pressure_requests + requests_lists.heart_rate_requests + \
                       requests_lists.respiratory_rate_requests + requests_lists.temperature_requests + \
                       requests_lists.oxygen_saturation_requests:
            self._check_if_new_request_triggers_reassignment(request=request,
                                                             state=state,
                                                             unassigned_requests_dict=self.monitoring_requests_dict,
                                                             assigned_requests_dict=self.assigned_monitoring_requests_dict,
                                                             motion_planner=motion_planner,
                                                             traversal_graph_generator=traversal_graph_generator)
        for request in requests_lists.medications_requests:
            self._check_if_new_request_triggers_reassignment(request=request,
                                                             state=state,
                                                             unassigned_requests_dict=self.delivery_requests_dict,
                                                             assigned_requests_dict=self.assigned_delivery_requests_dict,
                                                             motion_planner=motion_planner,
                                                             traversal_graph_generator=traversal_graph_generator)
    
    def _determine_robot_locations(self, robot_id: int, state: PlanningState) -> TraversalNode:
        if state.robots_next_nodes[robot_id] is None:
            robot_location = state.robots_current_nodes[robot_id]
        else:
            robot_location =  state.robots_next_nodes[robot_id]
        return robot_location
    
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
    
    def _generate_sorted_allocation_costs_for_requests(self,
                                   robot_id: int,
                                   unassigned_requests_dict: dict[str, float],
                                   assigned_requests_dict: dict[str, tuple[int, float]],
                                   state: PlanningState,
                                   motion_planner: MotionPlanner,
                                   traversal_graph_generator: TraversalGraphGenerator):
        allocations = []
        for request_id in list(unassigned_requests_dict.keys()) + list(assigned_requests_dict.keys()):
            if request_id in assigned_requests_dict:
                request_pickup_deadline = assigned_requests_dict[request_id][1]
            else:
                request_pickup_deadline = unassigned_requests_dict[request_id]
            trip_time = self._heuristic_cost_for_robot(request_id=request_id,
                                                       robot_id=robot_id,
                                                       state=state,
                                                       motion_planner=motion_planner,
                                                       traversal_graph_generator=traversal_graph_generator)
            
            current_cost = (self.alpha * (request_pickup_deadline - state.robots_current_time[robot_id]) + \
                            ((1 - self.alpha) * (trip_time)))
            allocations.append((request_id, current_cost))
        
        allocations.sort(key=lambda x: x[1])  # Sort by cost
        return allocations
    
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
                                            unassigned_requests_dict: dict[str, float],
                                            assigned_requests_dict: dict[str, tuple[int, float]],
                                            debug: bool):
        available_robots = state.get_available_robots(robot_type=robot_type)
        print(f"Available robots for {robot_type}: {available_robots}")
        for robot_id in available_robots:
            allocations = self._generate_sorted_allocation_costs_for_requests(robot_id=robot_id,
                                                                            unassigned_requests_dict=unassigned_requests_dict,
                                                                            assigned_requests_dict=assigned_requests_dict,
                                                                            state=state,
                                                                            motion_planner=motion_planner,
                                                                            traversal_graph_generator=traversal_graph_generator)
            request_allocated = False
            for request_id, _ in allocations:
                planned_path, planned_goal_indices, planned_time = self._determine_path_for_robot(request_id=request_id,
                                                                                                    robot_id=robot_id,
                                                                                                    state=state,
                                                                                                    motion_planner=motion_planner,
                                                                                                    traversal_graph_generator=traversal_graph_generator)
                if planned_path:
                    if request_id in unassigned_requests_dict:
                        unassigned_requests_dict.pop(request_id)
                        self._assign_request_to_robot(state=state,
                                                request_id=request_id,
                                                robot_id=robot_id,
                                                planned_path=planned_path,
                                                planned_time=planned_time,
                                                planned_goal_indices=planned_goal_indices,
                                                motion_planner=motion_planner,
                                                traversal_graph_generator=traversal_graph_generator,
                                                debug=debug)
                        self._add_request_to_dict(request=state.requests[request_id],
                                                  task_dict=assigned_requests_dict,
                                                  state=state,
                                                  motion_planner=motion_planner,
                                                  traversal_graph_generator=traversal_graph_generator,
                                                  assigned=True)
                        request_allocated = True
                        break
                    elif request_id in assigned_requests_dict:
                        prev_planned_time = state.requests[request_id].planned_time
                        if planned_time < prev_planned_time:
                            print(f"Reassigning request {request_id} to robot {robot_id} from robot {assigned_requests_dict[request_id][0]}.")
                            assigned_robot_id, _ = assigned_requests_dict.pop(request_id)
                            assigned_request = state.requests[request_id]
                            assigned_request.reset_assignment()
                            state.remove_request_from_robot(robot_id=assigned_robot_id,
                                                            request_id=request_id)
                            motion_planner.clear_reservations_for_agent(robot_profile=state.simulator_config.robot_profiles[assigned_robot_id])
                            self._assign_request_to_robot(state=state,
                                                    request_id=request_id,
                                                    robot_id=robot_id,
                                                    planned_path=planned_path,
                                                    planned_time=planned_time,
                                                    planned_goal_indices=planned_goal_indices,
                                                    motion_planner=motion_planner,
                                                    traversal_graph_generator=traversal_graph_generator,
                                                    debug=debug)
                            self._add_request_to_dict(request=state.requests[request_id],
                                                      task_dict=assigned_requests_dict,
                                                      state=state,
                                                      motion_planner=motion_planner,
                                                      traversal_graph_generator=traversal_graph_generator,
                                                      assigned=True)
                            request_allocated = True
                            break
            if not request_allocated:
                print(f"No suitable requests found for robot {robot_id}. Planning return to depot.")
                return_path = motion_planner.obtain_path_for_agent(start_traversal_node=self._determine_robot_locations(robot_id, state),
                                                       goal_traversal_node=state.robot_depots[robot_id],
                                                       robot_profile=state.simulator_config.robot_profiles[robot_id],
                                                       current_time=state.robots_current_time[robot_id],
                                                       wait_time_at_goal=120.0,
                                                       horizon=state.simulator_config.horizon)
                assert return_path is not None, "Return path to depot could not be found."
                motion_planner.clear_reservations_for_agent(robot_profile=state.simulator_config.robot_profiles[robot_id])
                motion_planner.reserve_path_for_agent(path=return_path,
                                                        robot_profile=state.simulator_config.robot_profiles[robot_id],
                                                        wait_time_at_goal=state.simulator_config.horizon)
                state.assign_robot_path(robot_id=robot_id, 
                                        path=return_path, 
                                        traversal_graph=traversal_graph_generator.traversal_graph)
    
    def _assign_requests_for_monitoring_robots(self, 
                                               state: PlanningState, 
                                               motion_planner: MotionPlanner, 
                                               traversal_graph_generator: TraversalGraphGenerator,
                                               debug: bool):
        self._assign_requests_to_available_robots(state=state,
                                                 motion_planner=motion_planner,
                                                 traversal_graph_generator=traversal_graph_generator,
                                                 robot_type="monitoring",
                                                 unassigned_requests_dict=self.monitoring_requests_dict,
                                                 assigned_requests_dict=self.assigned_monitoring_requests_dict,
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
                                                 unassigned_requests_dict=self.delivery_requests_dict,
                                                 assigned_requests_dict=self.assigned_delivery_requests_dict,
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
        
        self._check_if_requests_in_dicts_started(state=state, 
                                         motion_planner=motion_planner,
                                         traversal_graph_generator=traversal_graph_generator)

        self._add_new_requests_to_dicts(requests_lists=requests_lists, 
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