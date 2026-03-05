import copy
import math
from typing import Optional
from HTAMP.assignment.assignment_helpers import AssignmentHelpers, TaskQueue
from HTAMP.assignment.policies.helpers import PolicyHelpers
from HTAMP.environment.loc_dataclasses import TimeInterval
from HTAMP.environment.robot_dataclasses import RobotProfile
from HTAMP.environment.traversal_dataclasses import TraversalNode
from HTAMP.environment.traversal_graph_gen import TraversalGraphGenerator
from HTAMP.planning.motion_planner import MotionPlanner
from HTAMP.planning.planning_dataclasses import RequestsLists, TaskRequest, NodeReservationTable, TimeReservation
from HTAMP.planning.state import PlanningState

class SequentialGreedy:
    def __init__(self, allow_deallocation: bool = False):
        self.requests_queue = TaskQueue()
        self.dummy_delivery_robot_profile = RobotProfile(radius=0.10, speed=0.20, robot_id=-1, robot_type="delivery")
        self.assigned_requests:  dict[int, list[str]]  = {}
        self.node_reservation_table = NodeReservationTable(reservations={},
                                                          robot_node_dict={})
        self.allow_deallocation = allow_deallocation

    def _extract_assigned_requests_from_state(self, 
                                              state: PlanningState):
        self.assigned_requests = copy.deepcopy(state.assigned_requests)
    
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
                pickup_deadline = PolicyHelpers._add_request_to_queue(request=request,
                                           task_queue=self.requests_queue,
                                           delivery_robot_profile=self.dummy_delivery_robot_profile,
                                           motion_planner=motion_planner,
                                           traversal_graph_generator=traversal_graph_generator)
                pickup_deadlines.append(pickup_deadline)
        
        smallest_pickup_deadline = min(pickup_deadlines) if pickup_deadlines else None
        return smallest_pickup_deadline
    
    def _deallocate_requests_with_larger_pickup_deadlines(self,
                                                        smallest_pickup_deadline: float,
                                                        state: PlanningState,
                                                        motion_planner: MotionPlanner,
                                                        traversal_graph_generator: TraversalGraphGenerator,
                                                        debug: bool):
        robots_with_deallocated_requests = []
        for robot_id in self.assigned_requests.keys():
            if not self.assigned_requests[robot_id]:
                continue

            deallocation_index = None
            for i, request_id in enumerate(self.assigned_requests[robot_id]):
                request_struct = state.requests[request_id]
                if request_struct.started:
                    if debug:
                        print(f"Skipping deallocation of request {request_id} from robot {robot_id} because the request has already started.")
                    continue
                pickup_deadline = PolicyHelpers._calculate_pickup_deadline(delivery_robot_profile=self.dummy_delivery_robot_profile,
                                                                          request=request_struct,
                                                                          motion_planner=motion_planner,
                                                                          traversal_graph_generator=traversal_graph_generator)
                if pickup_deadline > smallest_pickup_deadline:
                    robots_with_deallocated_requests.append(robot_id)
                    deallocation_index = i
                    break

            if deallocation_index is not None:
                for j in range(deallocation_index, len(self.assigned_requests[robot_id])):
                    if debug:
                        print(f"Deallocating request {self.assigned_requests[robot_id][j]} from robot {robot_id} due to pickup deadline larger than smallest pickup deadline of {smallest_pickup_deadline}.")
                    pickup_deadline = PolicyHelpers._calculate_pickup_deadline(delivery_robot_profile=self.dummy_delivery_robot_profile,
                                                                              request=request_struct,
                                                                              motion_planner=motion_planner,
                                                                              traversal_graph_generator=traversal_graph_generator)
                    request_id = self.assigned_requests[robot_id][j]
                    request_struct = state.requests[request_id]
                    self.requests_queue.add_task(priority=pickup_deadline, 
                                                task_id=request_id)
                    request_struct.reset_assignment()
                    state.remove_request_from_robot(request_id=request_id, 
                                                    robot_id=robot_id)
                    
                self.assigned_requests[robot_id] = self.assigned_requests[robot_id][:deallocation_index]
                    
        for robot_id in robots_with_deallocated_requests:
            PolicyHelpers._generate_motion_plan_to_depot(robot_id=robot_id,
                                                         currently_assigned_request_ids=self.assigned_requests[robot_id],
                                                         state=state,
                                                         motion_planner=motion_planner,
                                                         traversal_graph_generator=traversal_graph_generator,
                                                         debug=debug)
    
    def _extract_node_reservations_from_state(self, 
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
    
    def _estimate_heuristic_cost_to_fulfill_request(self,
                                                     robot_id: int,
                                                     request_id: str,
                                                     state: PlanningState,
                                                     motion_planner: MotionPlanner,
                                                     traversal_graph_generator: TraversalGraphGenerator) -> float:
        request_struct = state.requests[request_id]
        if self.assigned_requests[robot_id]:
            last_assigned_request_id = self.assigned_requests[robot_id][-1]
            last_assigned_request_struct = state.requests[last_assigned_request_id]
            planned_goal_node_label = last_assigned_request_struct.goal_nodes[-1]
            last_planned_goal_index = last_assigned_request_struct.planned_goal_indices[-1]
            last_path_step = state.robot_paths[robot_id][last_planned_goal_index]
            start_node = last_path_step[0]
            start_time = last_path_step[1].end
            assert planned_goal_node_label == start_node.label, \
                f"Mismatch in planned goal node label and start node label: {planned_goal_node_label} vs {start_node.label}"
        else:
            start_node, start_time = AssignmentHelpers.determine_robot_nodes_and_times(robot_id=robot_id,
                                                                                    state=state)
        heuristic_cost = 0.0
        for j, goal_node_label in enumerate(request_struct.goal_nodes):
            goal_node = traversal_graph_generator.traversal_graph.nodes_dict[goal_node_label]
            travel_time_to_goal = motion_planner.planner.heuristic(start_traversal_node=start_node,
                                                                    goal_traversal_node=goal_node,
                                                                    robot_profile=state.simulator_config.robot_profiles[robot_id])
            if travel_time_to_goal == float('inf'):
                return float('inf')
            arrival_time = start_time + travel_time_to_goal
            wait_time = request_struct.wait_times_at_goals_seconds[j]
            service_interval = PolicyHelpers._obtain_time_to_service_node(robot_id=robot_id,
                                                                node_reservation_table=self.node_reservation_table,
                                                                node_label=goal_node_label,
                                                                arrival_time=arrival_time,
                                                                wait_time=wait_time)
            if service_interval.end > request_struct.time_for_service:
                return float('inf')
            else:
                heuristic_cost = service_interval.end - request_struct.scheduled_time
                start_node = goal_node
                start_time = service_interval.end
        
        return heuristic_cost
    
    def _determine_potential_assignments_for_request(self,
                                                     request_id: str,
                                                     state: PlanningState,
                                                     motion_planner: MotionPlanner,
                                                     traversal_graph_generator: TraversalGraphGenerator) -> list[tuple[int, float]]:
        request_struct = state.requests[request_id]
        potential_assignments = []
        if request_struct.request_type == "medication":
            robot_type = "delivery"
        else:
            robot_type = "monitoring"
        robot_ids = state.get_robots_of_type(robot_type=robot_type)

        for robot_id in robot_ids:
            heuristic_cost = self._estimate_heuristic_cost_to_fulfill_request(robot_id=robot_id,
                                                                              request_id=request_id,
                                                                              state=state,
                                                                              motion_planner=motion_planner,
                                                                              traversal_graph_generator=traversal_graph_generator)
            if heuristic_cost == float('inf'):
                continue
            potential_assignments.append((robot_id, heuristic_cost))
        
        potential_assignments.sort(key=lambda x: x[1])

        return potential_assignments
    
    def _get_path_with_minimum_cost_for_request_assignment(self,
                                                          potential_assignments: list[tuple[int, float]],
                                                          request_id: str,
                                                          state: PlanningState,
                                                          motion_planner: MotionPlanner,
                                                          traversal_graph_generator: TraversalGraphGenerator,
                                                          debug: bool) -> tuple[int, list[tuple[TraversalNode, TimeInterval]], list[int], float]:
        min_path_cost = float('inf')
        min_planned_path = None
        min_planned_goal_indices = None
        min_planned_time_to_reach_last_goal = float('inf')
        min_robot_id = None
        for i in range(len(potential_assignments)-1):
            robot_id, _ = potential_assignments[i]
            _, next_heuristic_cost = potential_assignments[i+1]
            request_struct = state.requests[request_id]
            path_results = PolicyHelpers._get_planned_path_for_request_assignment(robot_id=robot_id,
                                                                        request_id=request_id,
                                                                        currently_assigned_request_ids=self.assigned_requests[robot_id],
                                                                        state=state,
                                                                        motion_planner=motion_planner,
                                                                        traversal_graph_generator=traversal_graph_generator,
                                                                        debug=debug)
            planned_path, planned_goal_indices, planned_time_to_reach_last_goal = path_results
            if planned_path:
                real_cost = planned_time_to_reach_last_goal - request_struct.scheduled_time
                if real_cost < min_path_cost:
                    min_path_cost = real_cost
                    min_planned_path = planned_path
                    min_planned_goal_indices = planned_goal_indices
                    min_planned_time_to_reach_last_goal = planned_time_to_reach_last_goal
                    min_robot_id = robot_id
                if min_path_cost < next_heuristic_cost:
                    break
            else:
                if min_path_cost < next_heuristic_cost:
                    break
                else:
                    continue
        
        return min_robot_id, min_planned_path, min_planned_goal_indices, min_planned_time_to_reach_last_goal
        
    def _assign_requests_to_robots(self,
                                  state: PlanningState,
                                  motion_planner: MotionPlanner,
                                  traversal_graph_generator: TraversalGraphGenerator,
                                  debug: bool):
        while self.requests_queue.heap:
            next_request_id = self.requests_queue.pop_task()
            if debug:
                print(f"Attempting to assign request {next_request_id} at simulator time {state.simulator_time}")
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
                path_results = PolicyHelpers._get_planned_path_for_request_assignment(robot_id=robot_id,
                                                                            request_id=next_request_id,
                                                                            currently_assigned_request_ids=self.assigned_requests[robot_id],
                                                                            state=state,
                                                                            motion_planner=motion_planner,
                                                                            traversal_graph_generator=traversal_graph_generator,
                                                                            debug=debug)
                planned_path, planned_goal_indices, planned_time_to_reach_last_goal = path_results
                
                if planned_path:
                    if debug:
                        print(f"1) assigned requests for robot {robot_id}: {self.assigned_requests[robot_id]}")
                        print(f"1) State path for robot {robot_id}: {state.robot_paths[robot_id]}")
                        print(f"1) Planned path for assignment of request {next_request_id} to robot {robot_id}: {planned_path}")
                    PolicyHelpers._schedule_request(robot_id=robot_id,
                                            request_id=next_request_id,
                                            currently_assigned_request_ids=self.assigned_requests[robot_id],
                                            node_reservation_table=self.node_reservation_table,
                                            planned_path=planned_path,
                                            planned_goal_indices=planned_goal_indices,
                                            planned_time_to_reach_last_goal=planned_time_to_reach_last_goal,
                                            state=state,
                                            motion_planner=motion_planner,
                                            traversal_graph_generator=traversal_graph_generator)
                    
                    print(f"Assigned request {next_request_id} to robot {robot_id}")
                    
                else:
                    print(f"Failed to find a valid path for the only potential assignment of request {next_request_id} to robot {robot_id}. Request is rejected.")
                    request_struct = state.requests[next_request_id]
                    request_struct.mark_rejected()
                continue
            else:
                path_results = self._get_path_with_minimum_cost_for_request_assignment(potential_assignments=potential_assignments,
                                                                                      request_id=next_request_id,
                                                                                      state=state,
                                                                                      motion_planner=motion_planner,
                                                                                      traversal_graph_generator=traversal_graph_generator,
                                                                                      debug=debug)
                min_robot_id, min_planned_path, min_planned_goal_indices, min_planned_time_to_reach_last_goal = path_results
                
                if min_planned_path:
                    request_struct = state.requests[next_request_id]
                    if debug:
                        print(f"2) assigned requests for robot {min_robot_id}: {self.assigned_requests[min_robot_id]} at time {state.simulator_time}")
                        print(f"2) State path: {state.robot_paths[min_robot_id]}")
                        print(f"2) Planned path for assignment of request {next_request_id} to robot {min_robot_id}: {min_planned_path}")
                    PolicyHelpers._schedule_request(robot_id=min_robot_id,
                                            request_id=next_request_id,
                                            currently_assigned_request_ids=self.assigned_requests[min_robot_id],
                                            node_reservation_table=self.node_reservation_table,
                                            planned_path=min_planned_path,
                                            planned_goal_indices=min_planned_goal_indices,
                                            planned_time_to_reach_last_goal=min_planned_time_to_reach_last_goal,
                                            state=state,
                                            motion_planner=motion_planner,
                                            traversal_graph_generator=traversal_graph_generator)
                    
                    print(f"Assigned request {next_request_id} to robot {min_robot_id}")
                else:
                    print(f"Failed to find a valid path for any of the potential assignments of request {next_request_id}. Request is rejected.")
                    request = state.requests[next_request_id]
                    request.mark_rejected()

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
            if self.allow_deallocation:
                self._deallocate_requests_with_larger_pickup_deadlines(smallest_pickup_deadline=smallest_pickup_deadline,
                                                                       state=state,
                                                                       motion_planner=motion_planner,
                                                                       traversal_graph_generator=traversal_graph_generator,
                                                                       debug=debug)
            
            self._extract_node_reservations_from_state(state=state)

            # Assignment logic for robots
            self._assign_requests_to_robots(state=state,
                                            motion_planner=motion_planner,
                                            traversal_graph_generator=traversal_graph_generator,
                                            debug=debug)