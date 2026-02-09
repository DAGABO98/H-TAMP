import copy
from typing import Optional
from HTAMP.assignment.assignment_helpers import AssignmentHelpers, TaskQueue
from HTAMP.environment.loc_dataclasses import TimeInterval
from HTAMP.environment.robot_dataclasses import RobotProfile
from HTAMP.environment.traversal_dataclasses import TraversalNode
from HTAMP.environment.traversal_graph_gen import TraversalGraphGenerator
from HTAMP.planning.motion_planner import MotionPlanner
from HTAMP.planning.planning_dataclasses import RequestsLists, TaskRequest, NodeReservationTable, TimeReservation
from HTAMP.planning.state import PlanningState

class SequentialGreedy:
    def __init__(self):
        self.requests_queue = TaskQueue()
        self.dummy_delivery_robot_profile = RobotProfile(radius=0.10, speed=0.20, robot_id=-1, robot_type="delivery")
        self.assigned_requests:  dict[int, list[str]]  = {}
        self.node_reservation_table = NodeReservationTable(reservations={},
                                                          robot_node_dict={})

    def _extract_assigned_requests_from_state(self, 
                                              state: PlanningState):
        self.assigned_requests = copy.deepcopy(state.assigned_requests)
    
    def _calculate_pickup_deadline(self, 
                                   request: TaskRequest,
                                   motion_planner: MotionPlanner,
                                   traversal_graph_generator: TraversalGraphGenerator) -> float:
        if request.request_type != "medication":
            pickup_deadline = request.time_for_service - request.wait_times_at_goals_seconds[0]
        else:
            start_node = traversal_graph_generator.traversal_graph.nodes_dict[request.goal_nodes[0]]
            goal_node = traversal_graph_generator.traversal_graph.nodes_dict[request.goal_nodes[1]]
            travel_time_to_pickup = motion_planner.planner.heuristic(start_traversal_node=start_node,
                                                                     goal_traversal_node=goal_node,
                                                                     robot_profile=self.dummy_delivery_robot_profile)
            pickup_deadline = request.time_for_service - travel_time_to_pickup - request.wait_times_at_goals_seconds[0] - request.wait_times_at_goals_seconds[1]
            
        return pickup_deadline
    
    def _add_request_to_queue(self, 
                             request: TaskRequest, 
                             task_queue: TaskQueue,
                             motion_planner: MotionPlanner,
                             traversal_graph_generator: TraversalGraphGenerator):
        pickup_deadline = self._calculate_pickup_deadline(request=request,
                                                          motion_planner=motion_planner,
                                                          traversal_graph_generator=traversal_graph_generator)

        task_queue.add_task(priority=pickup_deadline, 
                            task_id=request.request_id)
        
        return pickup_deadline
    
    def _add_all_requests_to_queues(self, 
                                    requests_lists: Optional[RequestsLists], 
                                    motion_planner: MotionPlanner, 
                                    traversal_graph_generator: TraversalGraphGenerator):
        if requests_lists is None:
            return None
        pickup_deadlines = []
        for data_field in requests_lists.__dataclass_fields__.keys():
            requests_list = getattr(requests_lists, data_field)
            for request in requests_list:
                pickup_deadline = self._add_request_to_queue(request=request,
                                           task_queue=self.requests_queue,
                                           motion_planner=motion_planner,
                                           traversal_graph_generator=traversal_graph_generator)
                pickup_deadlines.append(pickup_deadline)
        
        smallest_pickup_deadline = min(pickup_deadlines) if pickup_deadlines else None
        return smallest_pickup_deadline
    
    def _deallocate_requests_with_larger_pickup_deadlines(self,
                                                        smallest_pickup_deadline: float,
                                                        state: PlanningState,
                                                        motion_planner: MotionPlanner,
                                                        traversal_graph_generator: TraversalGraphGenerator):
        robots_with_deallocated_requests = []
        for robot_id in self.assigned_requests.keys():
            if not self.assigned_requests[robot_id]:
                continue
            for request_id in self.assigned_requests[robot_id]:
                request_struct = state.requests[request_id]
                if request_struct.started:
                    continue
                pickup_deadline = self._calculate_pickup_deadline(request=request_struct,
                                                                  motion_planner=motion_planner,
                                                                  traversal_graph_generator=traversal_graph_generator)
                if pickup_deadline > smallest_pickup_deadline:
                    robots_with_deallocated_requests.append(robot_id)
                    self.requests_queue.add_task(priority=pickup_deadline, 
                                                task_id=request_id)
                    self.assigned_requests[robot_id].remove(request_id)
                    request_struct.reset_assignment()
                    state.remove_request_from_robot(request_id=request_id, 
                                                    robot_id=robot_id)
        for robot_id in robots_with_deallocated_requests:
            motion_planner.clear_reservations_for_agent(robot_id=robot_id)
            current_node, current_time = AssignmentHelpers.determine_robot_nodes_and_times(robot_id=robot_id,
                                                                                          state=state)
            if not self.assigned_requests[robot_id]:
                if current_node == state.robot_depots[robot_id]:
                    motion_planner._initialize_robot_reservations(initial_node=current_node,
                                                                  robot_profile=state.simulator_config.robot_profiles[robot_id],
                                                                  current_time=current_time,
                                                                  horizon=state.simulator_config.horizon)
                    state.robot_paths[robot_id] = [(current_node, TimeInterval(start=current_time, end=current_time + state.simulator_config.horizon))]
                else:
                    planned_path = state.robot_paths[robot_id][:state.robots_current_node_index[robot_id]+2]
                    motion_planner.reserve_path_for_agent(path=planned_path,
                                                          robot_profile=state.simulator_config.robot_profiles[robot_id])
                    state.robot_paths[robot_id] = planned_path
                    
            else:
                last_assigned_request_id = self.assigned_requests[robot_id][-1]
                last_assigned_request_struct = state.requests[last_assigned_request_id]
                planned_goal_index = last_assigned_request_struct.planned_goal_indices[-1]
                path_to_keep = state.robot_paths[robot_id][:planned_goal_index+1]
                motion_planner.reserve_path_for_agent(path=path_to_keep,
                                                      robot_profile=state.simulator_config.robot_profiles[robot_id])
                state.robot_paths[robot_id] = path_to_keep
    
    def _extract_reservations_from_state(self, 
                                         state: PlanningState):
        self.node_reservation_table.reset()
        for robot_id in self.assigned_requests.keys():
            if not self.assigned_requests[robot_id]:
                continue
            for request_id in self.assigned_requests[robot_id]:
                request_struct = state.requests[request_id]
                for goal_index in range(request_struct.completed_goals, len(request_struct.goal_nodes)):
                    goal_node_label = request_struct.goal_nodes[goal_index]
                    planned_goal_index = request_struct.planned_goal_indices[goal_index]
                    planned_goal_label = state.robot_paths[robot_id][planned_goal_index][0].label
                    assert goal_node_label == planned_goal_label, \
                        f"Mismatch in planned goal node labels: {goal_node_label} vs {planned_goal_label}"
                    start_time = state.robot_paths[robot_id][planned_goal_index][1].start
                    wait_time = request_struct.wait_times_at_goals_seconds[goal_index]
                    planned_time = state.robot_paths[robot_id][planned_goal_index][1].end
                    assert abs(planned_time - (start_time + wait_time)) < 1e-3, \
                        f"Mismatch in planned time and calculated time: {planned_time} vs {start_time + wait_time}"
                    reservation_interval = TimeInterval(start=start_time,
                                                        end=planned_time)
                    reservation = TimeReservation(robot_id=robot_id,
                                                  interval=reservation_interval)
                    self.node_reservation_table.add_reservation(node=goal_node_label,
                                                                reservation=reservation)
                
    
    def _obtain_time_to_service_node(self,
                                     robot_id: int,
                                     node_reservation_table: NodeReservationTable,
                                     node_label: str,
                                     arrival_time: float,
                                     wait_time: float,
                                     movement_time: float = 10.0) -> TimeInterval:
        
        reservations = node_reservation_table.get_reservations(node=node_label)
        requested_interval = TimeInterval(start=arrival_time, end=arrival_time + wait_time)
        if not reservations:
            return requested_interval
        else:
            interval_start = requested_interval.start
            interval_end = requested_interval.end
            reservations.sort(key=lambda x: x.interval.start)
            reservations = [res for res in reservations if res.robot_id != robot_id]
            for reservation in reservations:
                if interval_end <= reservation.interval.start:
                    break
                elif interval_start >= reservation.interval.end:
                    continue
                else:
                    interval_start = reservation.interval.end + movement_time
                    interval_end = interval_start + wait_time
            return TimeInterval(start=interval_start, end=interval_end)
    
    def _find_path_for_goal_nodes(self,
                                 robot_id: int,
                                 start_node: TraversalNode,
                                 start_time: float,
                                 goal_nodes: list[str],
                                 wait_times_at_goals_seconds: list[float],
                                 initial_planned_goal_index: int,
                                 state: PlanningState,
                                 motion_planner: MotionPlanner,
                                 traversal_graph_generator: TraversalGraphGenerator) \
                                    -> tuple[list[tuple[TraversalNode, TimeInterval]], list[int], float]:
        sub_paths: list[list[tuple[TraversalNode, TimeInterval]]] = []
        planned_goal_indices: list[int] = []
        planned_time_to_reach_last_goal: float = float('inf')
        for j, goal_node_label in enumerate(goal_nodes):
            goal_node = traversal_graph_generator.traversal_graph.nodes_dict[goal_node_label]
            current_time = start_time if not sub_paths else sub_paths[-1][-1][1].end
            sub_path = motion_planner.obtain_path_for_agent(start_traversal_node=start_node,
                                                    goal_traversal_node=goal_node,
                                                    robot_profile=state.simulator_config.robot_profiles[robot_id],
                                                    current_time=current_time,
                                                    wait_time_at_goal=wait_times_at_goals_seconds[j],
                                                    horizon=state.simulator_config.horizon)
            if sub_path is None:
                sub_paths = []
                planned_goal_indices = []
                planned_time_to_reach_last_goal = float('inf')
                break
            sub_paths.append(sub_path)
            if initial_planned_goal_index == 0:
                new_goal_index = len(sub_path) - 1
            else:
                new_goal_index = initial_planned_goal_index + len(sub_path)
            planned_goal_indices.append(new_goal_index)
            initial_planned_goal_index = new_goal_index
            start_node = goal_node
        if sub_paths:
            final_path = motion_planner.combine_paths(sub_paths)
            planned_time_to_reach_last_goal = sub_paths[-1][-1][1].end
        else:
            final_path = []
            planned_goal_indices = []
            planned_time_to_reach_last_goal = float('inf')

        return final_path, planned_goal_indices, planned_time_to_reach_last_goal
    
    def _generate_motion_plan_to_depot(self,
                                      robot_id: int,
                                      state: PlanningState,
                                      motion_planner: MotionPlanner,
                                      traversal_graph_generator: TraversalGraphGenerator):
        start_node, current_time = AssignmentHelpers.determine_robot_nodes_and_times(robot_id=robot_id,
                                                                                        state=state)
        depot_node = state.robot_depots[robot_id]
        planned_path = motion_planner.obtain_path_for_agent(start_traversal_node=start_node,
                                                        goal_traversal_node=depot_node,
                                                        robot_profile=state.simulator_config.robot_profiles[robot_id],
                                                        current_time=current_time,
                                                        wait_time_at_goal=state.simulator_config.horizon,
                                                        horizon=2*state.simulator_config.horizon)
    
    def _determine_potential_assignments_for_request(self,
                                                     request_id: str,
                                                     state: PlanningState,
                                                     motion_planner: MotionPlanner,
                                                     traversal_graph_generator: TraversalGraphGenerator) -> list[tuple[int, float, list[tuple[TraversalNode, TimeInterval]], list[int]]]:
        request_struct = state.requests[request_id]
        potential_assignments = []
        if request_struct.request_type == "medication":
            robot_type = "delivery"
        else:
            robot_type = "monitoring"
        robot_ids = state.get_robots_of_type(robot_type=robot_type)

        for robot_id in robot_ids:
            heuristic_cost = AssignmentHelpers._estimate_heuristic_cost_to_fulfill_request(robot_id=robot_id,
                                                                                            request_id=request_id,
                                                                                            state=state,
                                                                                            motion_planner=motion_planner,
                                                                                            traversal_graph_generator=traversal_graph_generator)
            if heuristic_cost == float('inf'):
                continue
            potential_assignments.append((robot_id, heuristic_cost))
        
        potential_assignments.sort(key=lambda x: x[1])

        return potential_assignments
        
    def _assign_requests_to_robots(self,
                                  state: PlanningState,
                                  motion_planner: MotionPlanner,
                                  traversal_graph_generator: TraversalGraphGenerator,
                                  debug: bool):
        while self.requests_queue.heap:
            next_request_id = self.requests_queue.pop_task()
            if debug:
                print(f"Attempting to assign request {next_request_id} with pickup deadline {self.requests_queue.priorities[next_request_id]}")
            potential_assignments = self._determine_potential_assignments_for_request(request_id=next_request_id,
                                                                                      state=state,
                                                                                      motion_planner=motion_planner,
                                                                                      traversal_graph_generator=traversal_graph_generator)
            if len(potential_assignments) == 0:
                print(f"No potential assignments found for request {next_request_id}. Request is rejected.")
                request = state.requests[next_request_id]
                request.mark_rejected()
                continue
            elif len(potential_assignments) == 1:
                robot_id, _ = potential_assignments[0]
                # TODO: Fix potential bug where determine_robot_nodes_and_times does not consider last assigned request
                start_node, current_time = AssignmentHelpers.determine_robot_nodes_and_times(robot_id=robot_id,
                                                                                            state=state)
                request_struct = state.requests[next_request_id]
                planned_path, planned_goal_indices, planned_time_to_reach_last_goal = self._find_path_for_goal_nodes(robot_id=robot_id,
                                                                                                                    start_node=start_node,
                                                                                                                    start_time=current_time,
                                                                                                                    goal_nodes=request_struct.goal_nodes,
                                                                                                                    wait_times_at_goals_seconds=request_struct.wait_times_at_goals_seconds,
                                                                                                                    initial_planned_goal_index=0,
                                                                                                                    state=state,
                                                                                                                    motion_planner=motion_planner,
                                                                                                                    traversal_graph_generator=traversal_graph_generator)
                if planned_path:
                    AssignmentHelpers.assign_request_to_robot(state=state,
                                                              request_id=next_request_id,
                                                              robot_id=robot_id,
                                                              planned_path=planned_path,
                                                              planned_time=planned_time_to_reach_last_goal,
                                                              planned_goal_indices=planned_goal_indices,
                                                              motion_planner=motion_planner,
                                                              traversal_graph_generator=traversal_graph_generator,
                                                              debug=debug)
                    if robot_id not in self.assigned_requests:
                        self.assigned_requests[robot_id] = []
                    self.assigned_requests[robot_id].append(next_request_id)
                    if debug:
                        print(f"Assigned request {next_request_id} to robot {robot_id}")
                else:
                    print(f"Failed to find a valid path for the only potential assignment of request {next_request_id} to robot {robot_id}. Request is rejected.")
                    request = state.requests[next_request_id]
                    request.mark_rejected()
                continue
            for i in range(len(potential_assignments)):
                robot_id, _ = potential_assignments[i]
                start_node, current_time = AssignmentHelpers.determine_robot_nodes_and_times(robot_id=robot_id,
                                                                                            state=state)
                request_struct = state.requests[next_request_id]
                initial_planned_goal_index = 0
                if robot_id in self.assigned_requests and self.assigned_requests[robot_id]:
                    last_assigned_request_id = self.assigned_requests[robot_id][-1]
                    last_assigned_request_struct = state.requests[last_assigned_request_id]
                    initial_planned_goal_index = last_assigned_request_struct.planned_goal_indices[-1]
                planned_path, planned_goal_indices, planned_time_to_reach_last_goal = self._find_path_for_goal_nodes(robot_id=robot_id,
                                                                                                                    start_node=start_node,
                                                                                                                    start_time=current_time,
                                                                                                                    goal_nodes=request_struct.goal_nodes,
                                                                                                                    wait_times_at_goals_seconds=request_struct.wait_times_at_goals_seconds,
                                                                                                                    initial_planned_goal_index=initial_planned_goal_index,
                                                                                                                    state=state,
                                                                                                                    motion_planner=motion_planner,
                                                                                                                    traversal_graph_generator=traversal_graph_generator)
                if planned_path:
                    AssignmentHelpers.assign_request_to_robot(state=state,
                                                              request_id=next_request_id,
                                                              robot_id=robot_id,
                                                              planned_path=planned_path,
                                                              planned_time=planned_time_to_reach_last_goal,
                                                              planned_goal_indices=planned_goal_indices,
                                                              motion_planner=motion_planner,
                                                              traversal_graph_generator=traversal_graph_generator,
                                                              debug=debug)
                    if robot_id not in self.assigned_requests:
                        self.assigned_requests[robot_id] = []
                    self.assigned_requests[robot_id].append(next_request_id)
                    if debug:
                        print(f"Assigned request {next_request_id} to robot {robot_id}")
                    break


    def assign_requests_to_robots(self, 
                                  state: PlanningState,
                                  requests_lists: Optional[RequestsLists], 
                                  motion_planner: MotionPlanner,
                                  traversal_graph_generator: TraversalGraphGenerator,
                                  debug: bool):
        # Extract assigned requests from state
        self._extract_assigned_requests_from_state(state=state)

        # Add new requests to the appropriate queues
        smallest_pickup_deadline = self._add_all_requests_to_queues(requests_lists=requests_lists,
                                                                    motion_planner=motion_planner,
                                                                    traversal_graph_generator=traversal_graph_generator)
        
        if smallest_pickup_deadline:
            self._deallocate_requests_with_larger_pickup_deadlines(smallest_pickup_deadline=smallest_pickup_deadline,
                                                                   state=state,
                                                                   motion_planner=motion_planner,
                                                                   traversal_graph_generator=traversal_graph_generator)
            self._extract_reservations_from_state(state=state)

        # Assignment logic for robots
        self._assign_requests_to_robots(state=state,
                                        motion_planner=motion_planner,
                                        traversal_graph_generator=traversal_graph_generator,
                                        debug=debug)