from typing import Optional
from HTAMP.assignment.assignment_helpers import AssignmentHelpers
from HTAMP.environment.robot_dataclasses import RobotProfile
from HTAMP.environment.traversal_graph_gen import TraversalGraphGenerator
from HTAMP.planning.motion_planner import MotionPlanner
from HTAMP.planning.planning_dataclasses import RequestsLists, TaskRequest
from HTAMP.planning.state import PlanningState

class DeadlineAwareTokenPassingwithTaskSwaps():
    
    def __init__(self):
        self.alpha: float = 0.1  # Weighting factor between urgency and travel time
        self.monitoring_requests_dict: dict[str, float] = {}
        self.assigned_monitoring_requests_dict: dict[str, tuple[int, float]] = {}
        self.delivery_requests_dict: dict[str, float] = {}
        self.assigned_delivery_requests_dict: dict[str, tuple[int, float]] = {}
        self.robots_to_be_sent_to_depot: list[int] = list()
        self.dummy_delivery_robot_profile = RobotProfile(radius=0.10, speed=0.20, robot_id=-1, robot_type="delivery")
        self.previous_available_monitoring_robots: list[int] = list()
        self.previous_available_delivery_robots: list[int] = list()
    
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
                robot_node, _ = AssignmentHelpers.determine_robot_nodes_and_times(robot_id=assigned_robot_id, 
                                                                                  state=state)
                new_request = state.requests[request.request_id]
                new_trip_time = AssignmentHelpers.heuristic_cost_from_robot_to_request(current_request=new_request,
                                                                                       robot_node=robot_node,
                                                                                       robot_id=assigned_robot_id,
                                                                                       state=state,
                                                                                       motion_planner=motion_planner,
                                                                                       traversal_graph_generator=traversal_graph_generator)
                
                prev_request = state.requests[assigned_request_id]
                prev_trip_time = AssignmentHelpers.heuristic_cost_from_robot_to_request(current_request=prev_request,
                                                                                        robot_node=robot_node,
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
                    self.robots_to_be_sent_to_depot.append(assigned_robot_id)
    
    def _add_new_requests_to_dicts(self,
                                   requests_lists: Optional[RequestsLists], 
                                   state: PlanningState, 
                                   motion_planner: MotionPlanner,
                                   traversal_graph_generator: TraversalGraphGenerator):
        if requests_lists is None:
            return False
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
        return True
    
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
            current_request = state.requests[request_id]
            start_node, _ = AssignmentHelpers.determine_robot_nodes_and_times(robot_id=robot_id, state=state)
            trip_time = AssignmentHelpers.heuristic_cost_from_robot_to_request(current_request=current_request,
                                                                               robot_node=start_node,
                                                                               robot_id=robot_id,
                                                                               state=state,
                                                                               motion_planner=motion_planner,
                                                                               traversal_graph_generator=traversal_graph_generator)
            
            current_cost = (self.alpha * (request_pickup_deadline - state.simulator_time) + ((1 - self.alpha) * (trip_time)))
            allocations.append((request_id, current_cost))
        
        allocations.sort(key=lambda x: x[1])  # Sort by cost
        return allocations
    
    def _assign_requests_to_available_robots(self,
                                            state: PlanningState,
                                            motion_planner: MotionPlanner,
                                            traversal_graph_generator: TraversalGraphGenerator,
                                            robot_type: str,
                                            unassigned_requests_dict: dict[str, float],
                                            assigned_requests_dict: dict[str, tuple[int, float]],
                                            new_requests_flag: bool,
                                            debug: bool):
        available_robots = state.get_available_robots(robot_type=robot_type)
        if debug:
            print(f"Available robots for {robot_type}: {available_robots}")
        
        if robot_type == "monitoring":
            if available_robots == self.previous_available_monitoring_robots and not new_requests_flag:
                if debug:
                    print("No new monitoring requests and no change in available monitoring robots. Skipping reassignment.")
                return
            self.previous_available_monitoring_robots = available_robots.copy()
        else:
            if available_robots == self.previous_available_delivery_robots and not new_requests_flag:
                if debug:
                    print("No new delivery requests and no change in available delivery robots. Skipping reassignment.")
                return
            self.previous_available_delivery_robots = available_robots.copy()
            
        while available_robots:
            robot_id = available_robots.pop(0)
            allocations = self._generate_sorted_allocation_costs_for_requests(robot_id=robot_id,
                                                                            unassigned_requests_dict=unassigned_requests_dict,
                                                                            assigned_requests_dict=assigned_requests_dict,
                                                                            state=state,
                                                                            motion_planner=motion_planner,
                                                                            traversal_graph_generator=traversal_graph_generator)
            for request_id, _ in allocations:
                planned_path, planned_goal_indices, planned_time = AssignmentHelpers.determine_path_from_robot_location_to_request(request_id=request_id,
                                                                                                    robot_id=robot_id,
                                                                                                    state=state,
                                                                                                    motion_planner=motion_planner,
                                                                                                    traversal_graph_generator=traversal_graph_generator,
                                                                                                    debug=debug)
                if planned_path:
                    if request_id in unassigned_requests_dict:
                        unassigned_requests_dict.pop(request_id)
                        AssignmentHelpers.assign_request_to_robot(state=state,
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
                        if robot_id in self.robots_to_be_sent_to_depot:
                            self.robots_to_be_sent_to_depot.remove(robot_id)
                        print(f"Assigned request {request_id} to robot {robot_id}.")
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
                            AssignmentHelpers.assign_request_to_robot(state=state,
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
                            if robot_id in self.robots_to_be_sent_to_depot:
                                self.robots_to_be_sent_to_depot.remove(robot_id)
                            self.robots_to_be_sent_to_depot.append(assigned_robot_id)
                            available_robots.append(assigned_robot_id)
                            break

        while self.robots_to_be_sent_to_depot:
            robot_to_depot_id = self.robots_to_be_sent_to_depot.pop(0)
            print(f"No suitable requests found for robot {robot_to_depot_id}. Planning return to depot.")
            start_node, initial_time = AssignmentHelpers.determine_robot_nodes_and_times(robot_to_depot_id, 
                                                                                         state)
            return_path = motion_planner.obtain_path_for_agent(start_traversal_node=start_node,
                                                    goal_traversal_node=state.robot_depots[robot_to_depot_id],
                                                    robot_profile=state.simulator_config.robot_profiles[robot_to_depot_id],
                                                    current_time=initial_time,
                                                    wait_time_at_goal=state.simulator_config.horizon,
                                                    horizon=2*state.simulator_config.horizon)
            assert return_path is not None, "Return path to depot could not be found."
            motion_planner.clear_reservations_for_agent(robot_profile=state.simulator_config.robot_profiles[robot_to_depot_id])
            motion_planner.reserve_path_for_agent(path=return_path,
                                                  robot_profile=state.simulator_config.robot_profiles[robot_to_depot_id])
            state.assign_robot_path(robot_id=robot_to_depot_id, 
                                    path=return_path, 
                                    traversal_graph=traversal_graph_generator.traversal_graph)
                
    def _assign_requests_for_monitoring_robots(self, 
                                               state: PlanningState, 
                                               motion_planner: MotionPlanner, 
                                               traversal_graph_generator: TraversalGraphGenerator,
                                               new_requests_flag: bool,
                                               debug: bool):
        self._assign_requests_to_available_robots(state=state,
                                                 motion_planner=motion_planner,
                                                 traversal_graph_generator=traversal_graph_generator,
                                                 robot_type="monitoring",
                                                 unassigned_requests_dict=self.monitoring_requests_dict,
                                                 assigned_requests_dict=self.assigned_monitoring_requests_dict,
                                                 new_requests_flag=new_requests_flag,
                                                 debug=debug)
    
    def _assign_requests_for_delivery_robots(self, 
                                             state: PlanningState, 
                                             motion_planner: MotionPlanner, 
                                             traversal_graph_generator: TraversalGraphGenerator,
                                             new_requests_flag: bool,
                                             debug: bool):
        self._assign_requests_to_available_robots(state=state,
                                                 motion_planner=motion_planner,
                                                 traversal_graph_generator=traversal_graph_generator,
                                                 robot_type="delivery",
                                                 unassigned_requests_dict=self.delivery_requests_dict,
                                                 assigned_requests_dict=self.assigned_delivery_requests_dict,
                                                 new_requests_flag=new_requests_flag,
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

        new_requests_flag = self._add_new_requests_to_dicts(requests_lists=requests_lists, 
                                                            state=state, 
                                                            motion_planner=motion_planner,
                                                            traversal_graph_generator=traversal_graph_generator)

        # Assignment logic for monitoring robots
        self._assign_requests_for_monitoring_robots(state=state, 
                                                   motion_planner=motion_planner, 
                                                   traversal_graph_generator=traversal_graph_generator,
                                                   new_requests_flag=new_requests_flag,
                                                   debug=debug)
        
        # Assignment logic for delivery robots
        self._assign_requests_for_delivery_robots(state=state, 
                                                 motion_planner=motion_planner, 
                                                 traversal_graph_generator=traversal_graph_generator,
                                                 new_requests_flag=new_requests_flag,
                                                 debug=debug)