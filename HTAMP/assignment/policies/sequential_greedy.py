import copy
import math
from typing import Optional
from HTAMP.assignment.assignment_helpers import AssignmentHelpers, TaskQueue
from HTAMP.environment.loc_dataclasses import TimeInterval
from HTAMP.environment.robot_dataclasses import RobotProfile
from HTAMP.environment.traversal_dataclasses import TraversalNode
from HTAMP.environment.traversal_graph_gen import TraversalGraphGenerator
from HTAMP.planning.motion_planner import MotionPlanner
from HTAMP.planning.planning_dataclasses import RequestsLists, TaskRequest, NodeReservationTable, TimeReservation
from HTAMP.planning.state import PlanningState
from HTAMP.plotting.motion_planning_plotting import MotionPlanningPlotter

class SequentialGreedy:
    def __init__(self):
        self.requests_queue = TaskQueue()
        self.dummy_delivery_robot_profile = RobotProfile(radius=0.10, speed=0.20, robot_id=-1, robot_type="delivery")
        self.assigned_requests:  dict[int, list[str]]  = {}
        self.node_reservation_table = NodeReservationTable(reservations={},
                                                          robot_node_dict={})
        self.start_times: dict[int, float]  = {}
        self.start_nodes: dict[int, TraversalNode]  = {}
        self.first_request_started: dict[int, bool]  = {}
        self.current_trip_times: dict[int, float]  = {}
        self.requests_costs: dict[str, tuple[float, float]] = {}
        self.need_assignment: bool = True
    
    def _extract_start_times_and_nodes(self, 
                                       state: PlanningState,
                                       traversal_graph_generator: TraversalGraphGenerator):
        for robot_id in self.assigned_requests.keys():
            if len(self.assigned_requests[robot_id]) == 0:
                start_node, current_time = AssignmentHelpers.determine_robot_nodes_and_times(robot_id=robot_id,
                                                                                             state=state)
                self.start_times[robot_id] = current_time
                self.start_nodes[robot_id] = start_node
                self.first_request_started[robot_id] = False
                continue
            else:
                first_request_id = self.assigned_requests[robot_id][0]
                first_task_started = state.requests[first_request_id].is_started()
                if first_task_started:
                    start_node_label = state.requests[first_request_id].goal_nodes[-1]
                    start_node = traversal_graph_generator.traversal_graph.nodes_dict[start_node_label]
                    current_time = state.requests[first_request_id].planned_time
                else:
                    start_node, current_time = AssignmentHelpers.determine_robot_nodes_and_times(robot_id=robot_id,
                                                                                                 state=state)
                self.start_times[robot_id] = current_time
                self.start_nodes[robot_id] = start_node
                self.first_request_started[robot_id] = first_task_started
    
    def _extract_trip_times(self):
        for robot_id in self.assigned_requests.keys():
            self.current_trip_times[robot_id] = 0.0

    def _extract_assigned_requests_from_state(self, 
                                              state: PlanningState, 
                                              traversal_graph_generator: TraversalGraphGenerator):
        self.assigned_requests = copy.deepcopy(state.assigned_requests)
        self.node_reservation_table.reset()
        self.need_assignment = True
        self._extract_trip_times()
        self._extract_start_times_and_nodes(state=state,
                                            traversal_graph_generator=traversal_graph_generator)
    
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
    
    def _add_all_requests_to_queues(self, 
                                    requests_lists: Optional[RequestsLists], 
                                    motion_planner: MotionPlanner, 
                                    traversal_graph_generator: TraversalGraphGenerator):
        if requests_lists is None:
            return
        for request in requests_lists.blood_pressure_requests + requests_lists.heart_rate_requests + \
                       requests_lists.respiratory_rate_requests + requests_lists.temperature_requests + \
                       requests_lists.oxygen_saturation_requests + requests_lists.medications_requests:
            self._add_request_to_queue(request=request, 
                                       task_queue=self.requests_queue,
                                       motion_planner=motion_planner,
                                       traversal_graph_generator=traversal_graph_generator)
    
    def _generate_robot_order(self,
                              round_index: int, 
                              assigned_requests: dict[int, list[str]],
                              state: PlanningState,
                              motion_planner: MotionPlanner,
                              traversal_graph_generator: TraversalGraphGenerator) -> list[int]:
        robot_priorities = []
        for robot_id in assigned_requests.keys():
            if round_index >= len(assigned_requests[robot_id]):
                robot_priorities.append((robot_id, state.simulator_config.horizon))
                continue
            request_id = assigned_requests[robot_id][round_index]
            pickup_deadline = self._calculate_pickup_deadline(request=state.requests[request_id],
                                                              motion_planner=motion_planner,
                                                              traversal_graph_generator=traversal_graph_generator)
            
            robot_priorities.append((robot_id, pickup_deadline))
        robot_priorities.sort(key=lambda x: x[1])
        sorted_robot_ids = [robot_priority[0] for robot_priority in robot_priorities]
        return sorted_robot_ids
    
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
    
    def _add_reservations_for_goal_intervals(self,
                                             node_reservation_table: NodeReservationTable,
                                             robot_id: int,
                                             goal_intervals: list[tuple[TraversalNode, TimeInterval]]):
        for goal_node, time_interval in goal_intervals:
            reservation = TimeReservation(robot_id=robot_id,
                                          interval=time_interval)
            node_reservation_table.add_reservation(node=goal_node.label,
                                                   reservation=reservation)
    
    def _calculate_service_time_for_request(self,
                                            robot_id: int,
                                            request: TaskRequest,
                                            current_trip_times: dict[int, float],
                                            node_reservation_table: NodeReservationTable,
                                            state: PlanningState,
                                            motion_planner: MotionPlanner,
                                            traversal_graph_generator: TraversalGraphGenerator) -> float:
        start_node = self.start_nodes[robot_id]
        current_time = self.start_times[robot_id] + current_trip_times[robot_id]

        total_trip_time = 0.0
        can_service_request = True
        requested_intervals: list[tuple[TraversalNode, TimeInterval]] = []
        for j, goal_node_label in enumerate(request.goal_nodes):
            goal_node = traversal_graph_generator.traversal_graph.nodes_dict[goal_node_label]
            subpath_time = motion_planner.planner.heuristic(start_traversal_node=start_node,
                                                            goal_traversal_node=goal_node,
                                                            robot_profile=state.simulator_config.robot_profiles[robot_id])
            total_trip_time += subpath_time
            arrival_time = current_time + total_trip_time
            wait_time = request.wait_times_at_goals_seconds[j]
            requested_interval = self._obtain_time_to_service_node(robot_id=robot_id,
                                                                   node_reservation_table=node_reservation_table,
                                                                   node_label=goal_node_label,
                                                                   arrival_time=arrival_time,
                                                                   wait_time=wait_time)
            requested_intervals.append((goal_node, requested_interval))
            new_wait_time = requested_interval.end - arrival_time
            total_trip_time += new_wait_time
            start_node = goal_node
        time_to_complete_request = current_time + total_trip_time
        if time_to_complete_request > request.time_for_service:
            can_service_request = False

        return time_to_complete_request, total_trip_time, requested_intervals, can_service_request
    
    def _remove_requests_after_violation(self,
                                       robot_id: int,
                                       request_id: str,
                                       state: PlanningState,
                                       motion_planner: MotionPlanner,
                                       traversal_graph_generator: TraversalGraphGenerator):
        
        request_id_index = self.assigned_requests[robot_id].index(request_id)
        requests_to_be_added_to_queues = self.assigned_requests[robot_id][request_id_index:]
        for req_id in requests_to_be_added_to_queues:
            req_struct = state.requests[req_id]
            self._add_request_to_queue(request=req_struct,
                                        task_queue=self.requests_queue,
                                        motion_planner=motion_planner,
                                        traversal_graph_generator=traversal_graph_generator)
        self.assigned_requests[robot_id] = self.assigned_requests[robot_id][:request_id_index]
        self.current_trip_times[robot_id] = 0.0
    
    def _remove_all_requests_except_started_requests(self,
                             failed_request_id: str,
                             state: PlanningState,
                             motion_planner: MotionPlanner,
                             traversal_graph_generator: TraversalGraphGenerator):
        for robot_id in self.assigned_requests.keys():
            requests_to_be_added_to_queues = self.assigned_requests[robot_id]
            if self.first_request_started[robot_id]:
                requests_to_be_added_to_queues = self.assigned_requests[robot_id][1:]
            for req_id in requests_to_be_added_to_queues:
                if req_id == failed_request_id:
                    continue
                req_struct = state.requests[req_id]
                self._add_request_to_queue(request=req_struct,
                                            task_queue=self.requests_queue,
                                            motion_planner=motion_planner,
                                            traversal_graph_generator=traversal_graph_generator)
            if self.first_request_started[robot_id]:
                self.assigned_requests[robot_id] = self.assigned_requests[robot_id][:1]
            else:
                self.assigned_requests[robot_id] = []
            self.current_trip_times[robot_id] = 0.0
    
    def _check_validity_of_requests_in_current_round(self,
                                                     round_index: int,
                                                     state: PlanningState,
                                                     motion_planner: MotionPlanner,
                                                     traversal_graph_generator: TraversalGraphGenerator):
        robot_order = self._generate_robot_order(round_index=round_index,
                                                 assigned_requests=self.assigned_requests,
                                                 state=state,
                                                 motion_planner=motion_planner,
                                                 traversal_graph_generator=traversal_graph_generator)
        
        for robot_id in robot_order:
            if round_index >= len(self.assigned_requests[robot_id]):
                continue

            if self.first_request_started[robot_id] and round_index == 0:
                continue

            request_id = self.assigned_requests[robot_id][round_index]
            request_struct = state.requests[request_id]

            service_time_results = self._calculate_service_time_for_request(robot_id=robot_id,
                                                                           request=request_struct,
                                                                           current_trip_times=self.current_trip_times,
                                                                            node_reservation_table=self.node_reservation_table,
                                                                           state=state,
                                                                           motion_planner=motion_planner,
                                                                           traversal_graph_generator=traversal_graph_generator)
            time_to_complete_request, total_trip_time, requested_intervals, can_service_request = service_time_results

            if not can_service_request:
                self._remove_requests_after_violation(robot_id=robot_id,
                                                      request_id=request_id,
                                                      state=state,
                                                      motion_planner=motion_planner,
                                                      traversal_graph_generator=traversal_graph_generator)
            else:
                self.current_trip_times[robot_id] += total_trip_time
                self._add_reservations_for_goal_intervals(node_reservation_table=self.node_reservation_table,
                                                          robot_id=robot_id,
                                                          goal_intervals=requested_intervals)
                self.requests_costs[request_id] = (time_to_complete_request - request_struct.scheduled_time,
                                                   max(time_to_complete_request - request_struct.scheduled_time, 0.0))
                
    def _generate_reservations_for_started_requests(self,
                                                    state: PlanningState,
                                                    assigned_requests: dict[int, list[str]],
                                                    node_reservation_table: NodeReservationTable):
        for robot_id in assigned_requests.keys():
            if not assigned_requests[robot_id]:
                continue
            first_task_started = self.first_request_started[robot_id]
            if first_task_started:
                request_id = assigned_requests[robot_id][0]
                request_struct = state.requests[request_id]
                for goal_index in range(request_struct.completed_goals, len(request_struct.goal_nodes)):
                    goal_node_label = request_struct.goal_nodes[goal_index]
                    planned_goal_index = request_struct.planned_goal_indices[goal_index]
                    planned_goal_label = state.robot_paths[robot_id][planned_goal_index][0].label
                    assert goal_node_label == planned_goal_label, \
                        f"Mismatch in planned goal node labels: {goal_node_label} vs {planned_goal_label}"
                    planned_time = state.robot_paths[robot_id][planned_goal_index][1].end
                    wait_time = request_struct.wait_times_at_goals_seconds[goal_index]
                    reservation_interval = TimeInterval(start=planned_time - wait_time,
                                                        end=planned_time)
                    reservation = TimeReservation(robot_id=robot_id,
                                                  interval=reservation_interval)
                    node_reservation_table.add_reservation(node=goal_node_label,
                                                           reservation=reservation)
                    
    def _check_assigned_requests_validity_and_generate_reservations(self,
                                                                   state: PlanningState,
                                                                   motion_planner: MotionPlanner,
                                                                   traversal_graph_generator: TraversalGraphGenerator):
        self._generate_reservations_for_started_requests(state=state,
                                                         assigned_requests=self.assigned_requests,
                                                        node_reservation_table=self.node_reservation_table)
        
        max_rounds = max(len(requests) for requests in self.assigned_requests.values())
        for round_index in range(max_rounds):
            self._check_validity_of_requests_in_current_round(round_index=round_index,
                                                              state=state,
                                                              motion_planner=motion_planner,
                                                              traversal_graph_generator=traversal_graph_generator)
            
    def _calculate_costs_of_assignment_for_current_round(self,
                                                        round_index: int,
                                                        assigned_requests: dict[int, list[str]],
                                                        node_reservation_table: NodeReservationTable,
                                                        trial_trip_times: dict[int, float],
                                                        trial_requests_costs: dict[str, tuple[float, float]],
                                                        state: PlanningState,
                                                        motion_planner: MotionPlanner,
                                                        traversal_graph_generator: TraversalGraphGenerator):
        robot_order = self._generate_robot_order(round_index=round_index,
                                                 assigned_requests=assigned_requests,
                                                 state=state,
                                                 motion_planner=motion_planner,
                                                 traversal_graph_generator=traversal_graph_generator)
        
        for robot_id in robot_order:
            if round_index >= len(assigned_requests[robot_id]):
                continue

            if self.first_request_started[robot_id] and round_index == 0:
                continue

            request_id = assigned_requests[robot_id][round_index]
            request_struct = state.requests[request_id]

            service_time_results = self._calculate_service_time_for_request(robot_id=robot_id,
                                                                           request=request_struct,
                                                                           current_trip_times=trial_trip_times,
                                                                            node_reservation_table=node_reservation_table,
                                                                           state=state,
                                                                           motion_planner=motion_planner,
                                                                           traversal_graph_generator=traversal_graph_generator)
            time_to_complete_request, total_trip_time, requested_intervals, can_service_request = service_time_results

            if not can_service_request:
                return False
                
            else:
                trial_trip_times[robot_id] += total_trip_time
                self._add_reservations_for_goal_intervals(node_reservation_table=node_reservation_table,
                                                          robot_id=robot_id,
                                                          goal_intervals=requested_intervals)
                trial_requests_costs[request_id] = (time_to_complete_request - request_struct.scheduled_time,
                                                   max(time_to_complete_request - request_struct.scheduled_time, 0.0))
        return True
    
    def _calculate_costs_for_request_assignment(self,
                                                state: PlanningState,
                                                assigned_requests: dict[int, list[str]],
                                                motion_planner: MotionPlanner,
                                                traversal_graph_generator: TraversalGraphGenerator):
        trial_trip_times: dict[int, float]  = {robot_id: 0.0 for robot_id in assigned_requests.keys()}
        trial_requests_costs: dict[str, tuple[float, float]] = {}
        node_reservation_table = NodeReservationTable(reservations={},
                                                      robot_node_dict={})
        self._generate_reservations_for_started_requests(state=state,
                                                         assigned_requests=assigned_requests,
                                                         node_reservation_table=node_reservation_table)
        
        max_rounds = max(len(requests) for requests in assigned_requests.values())
        for round_index in range(max_rounds):
            valid_assignment = self._calculate_costs_of_assignment_for_current_round(round_index=round_index,
                                                                                    assigned_requests=assigned_requests,
                                                                                    node_reservation_table=node_reservation_table,
                                                                                    trial_trip_times=trial_trip_times,
                                                                                    trial_requests_costs=trial_requests_costs,
                                                                                    state=state,
                                                                                    motion_planner=motion_planner,
                                                                                    traversal_graph_generator=traversal_graph_generator)
            if not valid_assignment:
                return float('inf'), float('inf')
        total_unmodified_cost = sum(costs[0] for costs in trial_requests_costs.values())
        total_cost = sum(costs[1] for costs in trial_requests_costs.values())

        return total_unmodified_cost, total_cost
    
    def _determine_lowest_cost_insertion_in_fleet(self, 
                                 request_id: str, 
                                 robots_list: list[int], 
                                 state: PlanningState, 
                                 motion_planner: MotionPlanner, 
                                 traversal_graph_generator: TraversalGraphGenerator):
        best_request_assignment: Optional[dict[int, list[str]]] = None
        lowest_total_cost: float = float('inf')
        lowest_unmodified_cost: float = float('inf')
        for robot_id in robots_list:
            trial_request_assignment = copy.deepcopy(self.assigned_requests)
            trial_request_assignment[robot_id].append(request_id)
            total_costs = self._calculate_costs_for_request_assignment(state=state,
                                                                      assigned_requests=trial_request_assignment,
                                                                      motion_planner=motion_planner,
                                                                      traversal_graph_generator=traversal_graph_generator)
            trial_unmodified_cost, trial_total_cost = total_costs
            if trial_total_cost < lowest_total_cost:
                lowest_total_cost = trial_total_cost
                lowest_unmodified_cost = trial_unmodified_cost
                best_request_assignment = trial_request_assignment
            elif trial_total_cost == lowest_total_cost:
                if trial_unmodified_cost < lowest_unmodified_cost:
                    lowest_unmodified_cost = trial_unmodified_cost
                    best_request_assignment = trial_request_assignment
        return best_request_assignment
    
    def _assign_requests_to_robots(self,
                                   state: PlanningState,
                                   motion_planner: MotionPlanner,
                                   traversal_graph_generator: TraversalGraphGenerator,
                                   debug: bool):

        while self.requests_queue.heap:
            next_request_id = self.requests_queue.pop_task()
            next_request = state.requests[next_request_id]
            if next_request.request_type == "medication":
                robots_list = state.get_robots_of_type(robot_type="delivery")
            else:
                robots_list = state.get_robots_of_type(robot_type="monitoring")
                
            best_request_assignment = self._determine_lowest_cost_insertion_in_fleet(request_id=next_request_id,
                                                                                     robots_list=robots_list,
                                                                                     state=state,
                                                                                     motion_planner=motion_planner,
                                                                                     traversal_graph_generator=traversal_graph_generator)
            if best_request_assignment is None:
                request_struct = state.requests[next_request_id]
                request_struct.mark_rejected()
                continue
            else:
                self.assigned_requests = best_request_assignment
    
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
            new_goal_index = initial_planned_goal_index + len(sub_path) - 1
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
    
    def _reset_motion_planning_reservations(self,
                                            state: PlanningState,
                                            motion_planner: MotionPlanner,
                                            traversal_graph_generator: TraversalGraphGenerator):
        robot_order = self._generate_robot_order(round_index=0,
                                                 assigned_requests=self.assigned_requests,
                                                 state=state,
                                                 motion_planner=motion_planner,
                                                 traversal_graph_generator=traversal_graph_generator)
        for robot_id in robot_order:
            if not state.assigned_requests[robot_id] and not self.assigned_requests[robot_id]:
                # No assigned requests, no plan needed
                continue
            elif not state.assigned_requests[robot_id] and self.assigned_requests[robot_id]:
                # New assignment for idle robot
                motion_planner.clear_reservations_for_agent(robot_profile=state.simulator_config.robot_profiles[robot_id])
            elif state.assigned_requests[robot_id] and not self.assigned_requests[robot_id]:
                # Robot has been unassigned all requests, plan to return to depot
                motion_planner.clear_reservations_for_agent(robot_profile=state.simulator_config.robot_profiles[robot_id])
            elif state.assigned_requests[robot_id] and self.assigned_requests[robot_id] and \
                 state.assigned_requests[robot_id][0] == self.assigned_requests[robot_id][0]:
                first_request_id = state.assigned_requests[robot_id][0]
                request_struct = state.requests[first_request_id]
                if request_struct.planned_time > state.simulator_time + 60.0:
                    # Continue with existing plan
                    continue
                else:
                    motion_planner.clear_reservations_for_agent(robot_profile=state.simulator_config.robot_profiles[robot_id])
            else:
                motion_planner.clear_reservations_for_agent(robot_profile=state.simulator_config.robot_profiles[robot_id])
    
    def _handle_failed_request_at_planning(self,
                                          failed_request_id: str,
                                          state: PlanningState,
                                          motion_planner: MotionPlanner,
                                          traversal_graph_generator: TraversalGraphGenerator):
        self._remove_all_requests_except_started_requests(failed_request_id=failed_request_id,
                                                                      state=state,
                                                                      motion_planner=motion_planner,
                                                                      traversal_graph_generator=traversal_graph_generator)
        failed_request = state.requests[failed_request_id]
        failed_request.mark_rejected()
        
    
    def _generate_motion_plans_for_assigned_requests(self,
                                                    robot_id: int,
                                                    state: PlanningState,
                                                    motion_planner: MotionPlanner,
                                                    traversal_graph_generator: TraversalGraphGenerator):
        start_node, current_time = AssignmentHelpers.determine_robot_nodes_and_times(robot_id=robot_id,
                                                                                     state=state)
        failed_request_id = None
        print(f"Generating motion plan for robot {robot_id} starting at node {start_node.label} and time {current_time}.")
        planned_goal_indices_list: list[list[int]] = []
        planned_paths: list[list[tuple[TraversalNode, TimeInterval]]] = []
        planned_request_ids: list[str] = []
        planned_times_to_reach_last_goals: list[float] = []
        initial_planned_goal_index = 0
        for request_id in self.assigned_requests[robot_id]:
            request_struct = state.requests[request_id]
            goal_nodes = request_struct.goal_nodes[request_struct.completed_goals:]
            wait_times = request_struct.wait_times_at_goals_seconds[request_struct.completed_goals:]
            path_results = self._find_path_for_goal_nodes(robot_id=robot_id,
                                                        start_node=start_node,
                                                        start_time=current_time,
                                                        goal_nodes=goal_nodes,
                                                        wait_times_at_goals_seconds=wait_times,
                                                        initial_planned_goal_index=initial_planned_goal_index,
                                                        state=state,
                                                        motion_planner=motion_planner,
                                                        traversal_graph_generator=traversal_graph_generator)
            planned_path, planned_goal_indices, planned_time_to_reach_last_goal = path_results
            if not planned_path:
                print(f"Warning: Unable to generate plan for robot {robot_id} with assigned requests.")
                print(f"Failed at request {request_id} with goals {goal_nodes}.")
                failed_request_id = request_id
                break
            planned_paths.append(planned_path)
            planned_goal_indices_list.append(planned_goal_indices)
            planned_request_ids.append(request_id)
            planned_times_to_reach_last_goals.append(planned_time_to_reach_last_goal)
            start_node = planned_path[-1][0]
            current_time = planned_time_to_reach_last_goal
            initial_planned_goal_index = planned_goal_indices[-1]
            if current_time > state.simulator_time + 60.0:
                break
        return planned_paths, planned_goal_indices_list, planned_request_ids, planned_times_to_reach_last_goals, failed_request_id
    
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
    def _reserve_motion_plan_for_robot(self, 
                                        robot_id: int,
                                        planned_paths: list[list[tuple[TraversalNode, TimeInterval]]],
                                        planned_goal_indices_list: list[list[int]],
                                        planned_request_ids: list[str],
                                        planned_times_to_reach_last_goals: list[float],
                                        state: PlanningState,
                                        motion_planner: MotionPlanner,
                                        traversal_graph_generator: TraversalGraphGenerator):
        
        final_path = motion_planner.combine_paths(planned_paths)
        print(f"Reserving motion plan for robot {robot_id}: {final_path}")
        
        for i in range(len(planned_request_ids)):
            request_id = planned_request_ids[i]
            request_struct = state.requests[request_id]
            print(f"  Request {request_id} has goal nodes: {request_struct.goal_nodes}")
            if request_struct.started:
                partial_goal_indices = planned_goal_indices_list[i]
                request_struct.planned_goal_indices[request_struct.completed_goals:] = partial_goal_indices
                request_struct.planned_time = planned_times_to_reach_last_goals[i]
            else:
                planned_goal_indices = planned_goal_indices_list[i]
                planned_time_to_reach_last_goal = planned_times_to_reach_last_goals[i]
                request_struct.schedule_task(planned_time=planned_time_to_reach_last_goal,
                                            planned_goal_indices=planned_goal_indices,
                                            assigned_robot_id=robot_id)

        motion_planner.reserve_path_for_agent(path=final_path,
                                              robot_profile=state.simulator_config.robot_profiles[robot_id])
        
        state.assign_robot_path(robot_id=robot_id,
                                path=final_path,
                                traversal_graph=traversal_graph_generator.traversal_graph)
    
    def _create_plan_and_reservation_for_robot(self, 
                                               robot_id: int, 
                                               state: PlanningState, 
                                               motion_planner: MotionPlanner, 
                                               traversal_graph_generator: TraversalGraphGenerator):
        plan_results = self._generate_motion_plans_for_assigned_requests(robot_id=robot_id,
                                                                        state=state,
                                                                        motion_planner=motion_planner,
                                                                        traversal_graph_generator=traversal_graph_generator)
        planned_paths, planned_goal_indices_list, planned_request_ids, planned_times_to_reach_last_goals, failed_request_id = plan_results

        print(f"Robot {robot_id} planned paths for requests: {planned_request_ids}")
        
        if failed_request_id is not None:
            print(f"Failed to generate motion plan for robot {robot_id} at request {failed_request_id}.")
            self._handle_failed_request_at_planning(failed_request_id=failed_request_id,
                                                    state=state,
                                                    motion_planner=motion_planner,
                                                    traversal_graph_generator=traversal_graph_generator)
        else:
            self._reserve_motion_plan_for_robot(robot_id=robot_id,
                                                planned_paths=planned_paths,
                                                planned_goal_indices_list=planned_goal_indices_list,
                                                planned_request_ids=planned_request_ids,
                                                planned_times_to_reach_last_goals=planned_times_to_reach_last_goals,
                                                state=state,
                                                motion_planner=motion_planner,
                                                traversal_graph_generator=traversal_graph_generator)
        
        return failed_request_id
    
    def _create_depot_plan_and_reservation_for_robot(self,
                                                     robot_id: int,
                                                     state: PlanningState,
                                                     motion_planner: MotionPlanner,
                                                     traversal_graph_generator: TraversalGraphGenerator):
        planned_path = self._generate_motion_plan_to_depot(robot_id=robot_id,
                                                            state=state,
                                                            motion_planner=motion_planner,
                                                            traversal_graph_generator=traversal_graph_generator)
        assert planned_path, f"Failed to generate path to depot for robot {robot_id}"
        print(f"Robot {robot_id} returning to depot.")
        self._reserve_motion_plan_for_robot(robot_id=robot_id,
                                            planned_paths=[planned_path],
                                            planned_goal_indices_list=[],
                                            planned_request_ids=[],
                                            planned_times_to_reach_last_goals=[],
                                            state=state,
                                            motion_planner=motion_planner,
                                            traversal_graph_generator=traversal_graph_generator)
    
    def _generate_motion_plans_for_fleet(self,
                                          state: PlanningState,
                                          motion_planner: MotionPlanner,
                                          traversal_graph_generator: TraversalGraphGenerator):
        robot_order = self._generate_robot_order(round_index=0,
                                                 assigned_requests=self.assigned_requests,
                                                 state=state,
                                                 motion_planner=motion_planner,
                                                 traversal_graph_generator=traversal_graph_generator)
        failed_request_id = None
        for robot_id in robot_order:
            if not state.assigned_requests[robot_id] and not self.assigned_requests[robot_id]:
                # No assigned requests, no plan needed
                continue
            elif not state.assigned_requests[robot_id] and self.assigned_requests[robot_id]:
                # New assignment for idle robot
                current_failed_request_id = self._create_plan_and_reservation_for_robot(robot_id=robot_id,
                                                                                        state=state,
                                                                                        motion_planner=motion_planner,
                                                                                        traversal_graph_generator=traversal_graph_generator)
                if current_failed_request_id is not None:
                    failed_request_id = current_failed_request_id
                    break
            elif state.assigned_requests[robot_id] and not self.assigned_requests[robot_id]:
                # Robot has been unassigned all requests, plan to return to depot
                self._create_depot_plan_and_reservation_for_robot(robot_id=robot_id,
                                                                  state=state,
                                                                  motion_planner=motion_planner,
                                                                  traversal_graph_generator=traversal_graph_generator)
        
            elif state.assigned_requests[robot_id] and self.assigned_requests[robot_id] and \
                 state.assigned_requests[robot_id][0] == self.assigned_requests[robot_id][0]:
                # Continuing with the first assigned request
                first_request_id = state.assigned_requests[robot_id][0]
                request_struct = state.requests[first_request_id]
                if request_struct.planned_time > state.simulator_time + 60.0:
                    # Continue with existing plan
                    continue
                else:
                    # Replan from current location
                    current_failed_request_id = self._create_plan_and_reservation_for_robot(robot_id=robot_id,
                                                                                        state=state,
                                                                                        motion_planner=motion_planner,
                                                                                        traversal_graph_generator=traversal_graph_generator)
                    if current_failed_request_id is not None:
                        failed_request_id = current_failed_request_id
                        break
            else:
                # New assignment different from current one
                current_failed_request_id = self._create_plan_and_reservation_for_robot(robot_id=robot_id,
                                                                                        state=state,
                                                                                        motion_planner=motion_planner,
                                                                                        traversal_graph_generator=traversal_graph_generator)
                if current_failed_request_id is not None:
                    failed_request_id = current_failed_request_id
                    break
        
        return failed_request_id
    
    def _generate_motion_plans_and_update_state(self,
                                                state: PlanningState,
                                                motion_planner: MotionPlanner,
                                                traversal_graph_generator: TraversalGraphGenerator):
        failed_request_id = self._generate_motion_plans_for_fleet(state=state,
                                                                  motion_planner=motion_planner,
                                                                  traversal_graph_generator=traversal_graph_generator)
        if failed_request_id is not None:
            print(f"Failed to generate motion plan for request {failed_request_id}, reassigning all available requests...")
            self.need_assignment = True
        else:
            state.assigned_requests = copy.deepcopy(self.assigned_requests)
            self.need_assignment = False


    def assign_requests_to_robots(self, 
                                  state: PlanningState,
                                  requests_lists: Optional[RequestsLists], 
                                  motion_planner: MotionPlanner,
                                  traversal_graph_generator: TraversalGraphGenerator,
                                  debug: bool):
        # Extract assigned requests from state
        self._extract_assigned_requests_from_state(state=state,
                                                   traversal_graph_generator=traversal_graph_generator)

        # Add new requests to the appropriate queues
        self._add_all_requests_to_queues(requests_lists=requests_lists,
                                         motion_planner=motion_planner,
                                         traversal_graph_generator=traversal_graph_generator)

        # Check if assigned requests can be reached
        self._check_assigned_requests_validity_and_generate_reservations(state=state,
                                                                         motion_planner=motion_planner,
                                                                         traversal_graph_generator=traversal_graph_generator)
        
        while self.need_assignment:
            print("Starting new assignment iteration...")
            # Assignment logic for monitoring robots
            self._assign_requests_to_robots(state=state,
                                           motion_planner=motion_planner,
                                           traversal_graph_generator=traversal_graph_generator,
                                           debug=debug)
            
            # Reset motion planning reservations
            self._reset_motion_planning_reservations(state=state,
                                                     motion_planner=motion_planner,
                                                     traversal_graph_generator=traversal_graph_generator)
            
            # Generate motion plans for assigned requests and update state
            self._generate_motion_plans_and_update_state(state=state,
                                                        motion_planner=motion_planner,
                                                        traversal_graph_generator=traversal_graph_generator)
        
        print("Final assigned requests:", self.assigned_requests)