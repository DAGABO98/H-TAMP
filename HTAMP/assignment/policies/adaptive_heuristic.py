import copy
import math
from typing import Optional
from HTAMP.assignment.assignment_helpers import TaskQueue
from HTAMP.environment.loc_dataclasses import TimeInterval
from HTAMP.environment.robot_dataclasses import RobotProfile
from HTAMP.environment.traversal_dataclasses import TraversalNode
from HTAMP.environment.traversal_graph_gen import TraversalGraphGenerator
from HTAMP.planning.motion_planner import MotionPlanner
from HTAMP.planning.planning_dataclasses import RequestsLists, TaskRequest, NodeReservationTable, TimeReservation
from HTAMP.planning.state import PlanningState
from HTAMP.plotting.motion_planning_plotting import MotionPlanningPlotter

class AdaptiveHeuristicPolicy:
    def __init__(self):
        self.monitoring_requests_queue = TaskQueue()
        self.delivery_requests_queue = TaskQueue()
        self.dummy_delivery_robot_profile = RobotProfile(radius=0.10, speed=0.20, robot_id=-1, robot_type="delivery")
        self.assigned_requests:  dict[int, list[str]]  = {}
        self.trial_assigned_requests:  dict[int, list[str]]  = {}
        self.node_reservation_table = NodeReservationTable()
        self.trial_node_reservation_table = NodeReservationTable()
        self.start_times: dict[int, float]  = {}
        self.start_nodes: dict[int, TraversalNode]  = {}
        self.first_request_started: dict[int, bool]  = {}
        self.current_trip_times: dict[int, float]  = {}
        self.trial_current_trip_times: dict[int, float]  = {}
        self.requests_costs: dict[str, tuple[float, float]] = {}
        self.trial_requests_costs: dict[str, tuple[float, float]] = {}
    
    def _determine_robot_locations(self, robot_id: int, state: PlanningState) -> TraversalNode:
        if state.robots_next_nodes[robot_id] is None:
            robot_location = state.robots_current_nodes[robot_id]
        else:
            robot_location =  state.robots_next_nodes[robot_id]
        return robot_location
    
    def _extract_start_times_and_nodes(self, 
                                       state: PlanningState,
                                       traversal_graph_generator: TraversalGraphGenerator):
        for robot_id in self.assigned_requests.keys():
            first_request_id = self.assigned_requests[robot_id][0]
            first_task_started = state.requests[first_request_id].is_started()
            if first_task_started:
                start_node_label = state.requests[first_request_id].goal_nodes[-1]
                start_node = traversal_graph_generator.traversal_graph.nodes_dict[start_node_label]
                current_time = state.requests[first_request_id].planned_time
            else:
                start_node = self._determine_robot_locations(robot_id, state)
                current_time: float = state.robots_current_time[robot_id]
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
                       requests_lists.oxygen_saturation_requests:
            self._add_request_to_queue(request=request, 
                                       task_queue=self.monitoring_requests_queue,
                                       motion_planner=motion_planner,
                                       traversal_graph_generator=traversal_graph_generator)
        for request in requests_lists.medications_requests:
            self._add_request_to_queue(request=request, 
                                       task_queue=self.delivery_requests_queue,
                                       motion_planner=motion_planner,
                                       traversal_graph_generator=traversal_graph_generator)
    
    def _generate_robot_order(self,
                              round_index: int, 
                              state: PlanningState,
                              motion_planner: MotionPlanner,
                              traversal_graph_generator: TraversalGraphGenerator) -> list[int]:
        robot_priorities = []
        for robot_id in self.assigned_requests.keys():
            if round_index >= len(self.assigned_requests[robot_id]):
                robot_priorities.append((robot_id, state.simulator_config.horizon))
                continue
            request_id = self.assigned_requests[robot_id][round_index]
            pickup_deadline = self._calculate_pickup_deadline(request=state.requests[request_id],
                                                              motion_planner=motion_planner,
                                                              traversal_graph_generator=traversal_graph_generator)
            
            robot_priorities.append((robot_id, pickup_deadline))
        robot_priorities.sort(key=lambda x: x[1])
        sorted_robot_ids = [robot_priority[0] for robot_priority in robot_priorities]
        return sorted_robot_ids
    
    def _obtain_time_to_service_node(self,
                                    node_label: str,
                                    arrival_time: float,
                                    wait_time: float,
                                    movement_time: float = 10.0) -> TimeInterval:
        
        reservations = self.node_reservation_table.get_reservations(node=node_label)
        requested_interval = TimeInterval(start=arrival_time, end=arrival_time + wait_time)
        if not reservations:
            return requested_interval
        else:
            reservations.sort(key=lambda x: x.interval.start)
            for reservation in reservations:
                if requested_interval.end <= reservation.interval.start:
                    return requested_interval
                elif requested_interval.start >= reservation.interval.end:
                    continue
                else:
                    requested_interval.start = reservation.interval.end + movement_time
                    requested_interval.end = requested_interval.start + wait_time
            return requested_interval
    
    def _add_reservations_for_goal_intervals(self,
                                             robot_id: int,
                                             goal_intervals: list[tuple[TraversalNode, TimeInterval]]):
        for goal_node, time_interval in goal_intervals:
            reservation = TimeReservation(robot_id=robot_id,
                                          interval=time_interval)
            self.node_reservation_table.add_reservation(node=goal_node.label,
                                                        reservation=reservation)
    
    def _check_validity_of_requests_in_current_round(self,
                                                     round_index: int,
                                                     state: PlanningState,
                                                     motion_planner: MotionPlanner,
                                                     traversal_graph_generator: TraversalGraphGenerator):
        robot_order = self._generate_robot_order(round_index=round_index,
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
            start_node = self.start_nodes[robot_id]
            current_time = self.start_times[robot_id] + self.current_trip_times[robot_id]
            total_trip_time = 0.0
            can_service_request = True
            requested_intervals: list[tuple[TraversalNode, TimeInterval]] = []
            for j, goal_node_label in enumerate(request_struct.goal_nodes):
                goal_node = traversal_graph_generator.traversal_graph.nodes_dict[goal_node_label]
                subpath_time = motion_planner.planner.heuristic(start_traversal_node=start_node,
                                                                goal_traversal_node=goal_node,
                                                                robot_profile=state.simulator_config.robot_profiles[robot_id])
                total_trip_time += subpath_time
                arrival_time = current_time + total_trip_time
                wait_time = request_struct.wait_times_at_goals_seconds[j]
                requested_interval = self._obtain_time_to_service_node(node_label=goal_node_label,
                                                                       arrival_time=arrival_time,
                                                                       wait_time=wait_time)
                requested_intervals.append((goal_node, requested_interval))
                new_wait_time = requested_interval.end - arrival_time
                total_trip_time += new_wait_time
                start_node = goal_node
            time_to_complete_request = current_time + total_trip_time
            if time_to_complete_request > request_struct.time_for_service:
                can_service_request = False

            if not can_service_request:
                request_id_index = self.assigned_requests[robot_id].index(request_id)
                requests_to_be_added_to_queues = self.assigned_requests[robot_id][request_id_index:]
                for req_id in requests_to_be_added_to_queues:
                    req_struct = state.requests[req_id]
                    if req_struct.request_type in ["blood_pressure", "heart_rate", "respiratory_rate", "temperature", "oxygen_saturation"]:
                        self._add_request_to_queue(request=req_struct,
                                                   task_queue=self.monitoring_requests_queue,
                                                   motion_planner=motion_planner,
                                                   traversal_graph_generator=traversal_graph_generator)
                    else:
                        self._add_request_to_queue(request=req_struct,
                                                   task_queue=self.delivery_requests_queue,
                                                   motion_planner=motion_planner,
                                                   traversal_graph_generator=traversal_graph_generator)
                self.assigned_requests[robot_id] = self.assigned_requests[robot_id][:request_id_index]
            else:
                self.current_trip_times[robot_id] += total_trip_time
                self._add_reservations_for_goal_intervals(robot_id=robot_id,
                                                         goal_intervals=requested_intervals)
                
    def _generate_reservations_for_started_requests(self,
                                                  state: PlanningState):
        for robot_id in self.assigned_requests.keys():
            if not self.assigned_requests[robot_id]:
                continue
            first_task_started = self.first_request_started[robot_id]
            if first_task_started:
                request_id = self.assigned_requests[robot_id][0]
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
                    self.node_reservation_table.add_reservation(node=goal_node_label,
                                                                reservation=reservation)
                    
    def _check_assigned_requests_validity_and_generate_reservations(self,
                                                                   state: PlanningState,
                                                                   motion_planner: MotionPlanner,
                                                                   traversal_graph_generator: TraversalGraphGenerator):
        self._generate_reservations_for_started_requests(state=state)
        
        max_rounds = max(len(requests) for requests in self.assigned_requests.values())
        for round_index in range(max_rounds):
            self._check_validity_of_requests_in_current_round(round_index=round_index,
                                                              state=state,
                                                              motion_planner=motion_planner,
                                                              traversal_graph_generator=traversal_graph_generator)
    
    def _calculate_costs_of_request_order(self, 
                                          request_order: list[str], 
                                          first_task_started: bool,
                                          robot_id: int, 
                                          state: PlanningState, 
                                          motion_planner: MotionPlanner, 
                                          traversal_graph_generator: TraversalGraphGenerator) -> tuple[float, float]:
        if first_task_started:
            start_node_label = state.requests[request_order[0]].goal_nodes[-1]
            start_node = traversal_graph_generator.traversal_graph.nodes_dict[start_node_label]
            current_time = state.requests[request_order[0]].planned_time
        else:
            start_node = self._determine_robot_locations(robot_id, state)
            current_time: float = state.robots_current_time[robot_id]

        if first_task_started:
            new_request_order = request_order[1:]
        else:
            new_request_order = request_order

        total_unmodified_cost = 0.0
        total_cost = 0.0
        total_trip_time = 0.0
        for request_id in new_request_order:
            current_request = state.requests[request_id]
            for j, goal_node_label in enumerate(current_request.goal_nodes):
                goal_node = traversal_graph_generator.traversal_graph.nodes_dict[goal_node_label]
                subpath_time = motion_planner.planner.heuristic(start_traversal_node=start_node,
                                                                  goal_traversal_node=goal_node,
                                                                  robot_profile=state.simulator_config.robot_profiles[robot_id])
                # TODO Check if goal_node is free at the estimated arrival time and for the time period of wait_time_at_goal
                # If not, determine next available time slot and adjust subpath_time accordingly

                total_trip_time += subpath_time + current_request.wait_times_at_goals_seconds[j]
                start_node = goal_node
            time_to_complete_request = current_time + total_trip_time
            if time_to_complete_request > current_request.time_for_service:
                return float('inf'), float('inf')
            total_unmodified_cost += (time_to_complete_request - current_request.scheduled_time)
            total_cost += max(time_to_complete_request - current_request.scheduled_time, 0.0)
        return total_unmodified_cost, total_cost
    
    def _determine_lowest_cost_insertion_for_robot(self, 
                                                   request_id: str, 
                                                   robot_id: int, 
                                                   state: PlanningState, 
                                                   motion_planner: MotionPlanner, 
                                                   traversal_graph_generator: TraversalGraphGenerator) -> tuple[list[str], float, float]:
        new_request_order = []
        total_cost = float('inf')
        total_unmodified_cost = float('inf')

        if not self.assigned_requests[robot_id]:
            new_request_order = [request_id]
            total_unmodified_cost, total_cost = self._calculate_costs_of_request_order(request_order=new_request_order,
                                                                                        first_task_started=False,
                                                                                        robot_id=robot_id,
                                                                                        state=state,
                                                                                        motion_planner=motion_planner,
                                                                                        traversal_graph_generator=traversal_graph_generator)
        else:
            start_index = 0
            first_request_id = self.assigned_requests[robot_id][0]
            first_task_started = state.requests[first_request_id].is_started()

            if first_task_started:
                start_index = 1
            
            original_unmodified_cost, original_total_cost = self._calculate_costs_of_request_order(request_order=self.assigned_requests[robot_id],
                                                                                               first_task_started=first_task_started,
                                                                                               robot_id=robot_id,
                                                                                               state=state,
                                                                                               motion_planner=motion_planner,
                                                                                               traversal_graph_generator=traversal_graph_generator)

            for i in range(start_index, len(self.assigned_requests[robot_id]) + 1):
                trial_request_order = self.assigned_requests[robot_id][:i] + [request_id] + self.assigned_requests[robot_id][i:]
                trial_unmodified_cost, trial_total_cost = self._calculate_costs_of_request_order(request_order=trial_request_order,
                                                                                                 first_task_started=first_task_started,
                                                                                                 robot_id=robot_id,
                                                                                                 state=state,
                                                                                                 motion_planner=motion_planner,
                                                                                                 traversal_graph_generator=traversal_graph_generator)
                current_total_cost = trial_total_cost - original_total_cost
                current_unmodified_cost = trial_unmodified_cost - original_unmodified_cost
                if current_total_cost < total_cost:
                    total_cost = current_total_cost
                    total_unmodified_cost = current_unmodified_cost
                    new_request_order = trial_request_order
                elif current_total_cost == total_cost:
                    if current_unmodified_cost < total_unmodified_cost:
                        total_unmodified_cost = current_unmodified_cost
                        new_request_order = trial_request_order

        return new_request_order, total_unmodified_cost, total_cost
    
    def _determine_lowest_cost_insertion_in_fleet(self, 
                                 request_id: str, 
                                 robots_list: list[int], 
                                 state: PlanningState, 
                                 motion_planner: MotionPlanner, 
                                 traversal_graph_generator: TraversalGraphGenerator):
        best_robot_id: Optional[int] = None
        best_request_order: list[str] = []
        lowest_total_cost: float = float('inf')
        lowest_unmodified_cost: float = float('inf')
        for robot_id in robots_list:
            insertion_results = self._determine_lowest_cost_insertion_for_robot(request_id=request_id,
                                                                                robot_id=robot_id,
                                                                                state=state,
                                                                                motion_planner=motion_planner,
                                                                                traversal_graph_generator=traversal_graph_generator)
            trial_request_order, trial_unmodified_cost, trial_total_cost = insertion_results
            if trial_total_cost < lowest_total_cost:
                lowest_total_cost = trial_total_cost
                lowest_unmodified_cost = trial_unmodified_cost
                best_request_order = trial_request_order
                best_robot_id = robot_id
            elif trial_total_cost == lowest_total_cost:
                if trial_unmodified_cost < lowest_unmodified_cost:
                    lowest_unmodified_cost = trial_unmodified_cost
                    best_request_order = trial_request_order
                    best_robot_id = robot_id
        return best_robot_id, best_request_order
    
    def _reserve_request_order_for_robot(self,
                                        robot_id: int,
                                        state: PlanningState,
                                        motion_planner: MotionPlanner,
                                        traversal_graph_generator: TraversalGraphGenerator):
        self.node_reservation_table.remove_reservations_for_robot(robot_id=robot_id)
        self._create_reservation_for_robot(robot_id=robot_id,
                                          state=state,
                                          motion_planner=motion_planner,
                                          traversal_graph_generator=traversal_graph_generator)
    
    def _assign_requests_to_robots(self,
                                   state: PlanningState,
                                   motion_planner: MotionPlanner,
                                   traversal_graph_generator: TraversalGraphGenerator,
                                   robot_type: str,
                                   requests_queue: TaskQueue,
                                   debug: bool):
        robots_list = state.get_robots_of_type(robot_type=robot_type)

        while requests_queue.heap:
            next_request_id = requests_queue.pop_task()
            fleet_insertion_results = self._determine_lowest_cost_insertion_in_fleet(request_id=next_request_id,
                                                                                     robots_list=robots_list,
                                                                                     state=state,
                                                                                     motion_planner=motion_planner,
                                                                                     traversal_graph_generator=traversal_graph_generator)
            best_robot_id, best_request_order = fleet_insertion_results
            if best_robot_id is not None:
                self.assigned_requests[best_robot_id] = best_request_order
                self._reserve_request_order_for_robot(robot_id=best_robot_id,
                                                     state=state,
                                                     motion_planner=motion_planner,
                                                     traversal_graph_generator=traversal_graph_generator)
            else:
                request_struct = state.requests[next_request_id]
                request_struct.mark_rejected()

    def _assign_requests_for_monitoring_robots(self, 
                                               state: PlanningState, 
                                               motion_planner: MotionPlanner, 
                                               traversal_graph_generator: TraversalGraphGenerator,
                                               debug: bool):
        self._assign_requests_to_robots(state=state,
                                        motion_planner=motion_planner,
                                        traversal_graph_generator=traversal_graph_generator,
                                        robot_type="monitoring",
                                        requests_queue=self.monitoring_requests_queue,         
                                        debug=debug)
    
    def _assign_requests_for_delivery_robots(self, 
                                             state: PlanningState, 
                                             motion_planner: MotionPlanner, 
                                             traversal_graph_generator: TraversalGraphGenerator,
                                             debug: bool):
        self._assign_requests_to_robots(state=state,
                                        motion_planner=motion_planner,
                                        traversal_graph_generator=traversal_graph_generator,
                                        robot_type="delivery",
                                        requests_queue=self.delivery_requests_queue,         
                                        debug=debug)
    
    def _find_path_for_goal_nodes(self,
                                 robot_id: int,
                                 start_node: TraversalNode,
                                 start_time: float,
                                 goal_nodes: list[str],
                                 wait_times_at_goals_seconds: list[float],
                                 initial_planned_goal_indices: Optional[list[int]],
                                 state: PlanningState,
                                 motion_planner: MotionPlanner,
                                 traversal_graph_generator: TraversalGraphGenerator) \
                                    -> tuple[list[tuple[TraversalNode, TimeInterval]], list[int], float]:
        sub_paths: list[list[tuple[TraversalNode, TimeInterval]]] = []
        if initial_planned_goal_indices is not None:
            planned_goal_indices = copy.deepcopy(initial_planned_goal_indices)
        else:
            planned_goal_indices = []
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
            if planned_goal_indices:
                planned_goal_indices.append(planned_goal_indices[-1] + len(sub_path) - 1)
            else:
                planned_goal_indices.append(len(sub_path) - 1)
            start_node = goal_node
        if sub_paths:
            final_path = motion_planner.combine_paths(sub_paths)
            planned_time_to_reach_last_goal = sub_paths[-1][-1][1].end
        else:
            final_path = []
            planned_goal_indices = []
            planned_time_to_reach_last_goal = float('inf')

        return final_path, planned_goal_indices, planned_time_to_reach_last_goal
    
    def _generate_robot_priorities(self,
                                   state: PlanningState,
                                   motion_planner: MotionPlanner,
                                   traversal_graph_generator: TraversalGraphGenerator) -> list[int]:
        robot_priorities = []
        for robot_id in self.assigned_requests.keys():
            if not self.assigned_requests[robot_id]:
                robot_priorities.append((robot_id, state.simulator_config.horizon))
                continue
            first_request_id = self.assigned_requests[robot_id][0]
            pickup_deadline = self._calculate_pickup_deadline(request=state.requests[first_request_id],
                                                              motion_planner=motion_planner,
                                                              traversal_graph_generator=traversal_graph_generator)
            
            robot_priorities.append((robot_id, pickup_deadline))
        robot_priorities.sort(key=lambda x: x[1])
        sorted_robot_ids = [robot_priority[0] for robot_priority in robot_priorities]
        return sorted_robot_ids
    
    def _update_state_with_assigned_requests_and_generate_plans(self, 
                                                                sorted_robot_ids: list[int],
                                                                state: PlanningState,
                                                                motion_planner: MotionPlanner,
                                                                traversal_graph_generator: TraversalGraphGenerator):
        for robot_id in sorted_robot_ids:
            if not self.assigned_requests[robot_id]:
                continue
            first_request_id = self.assigned_requests[robot_id][0]
            first_request = state.requests[first_request_id]
            first_task_started = state.requests[first_request_id].is_started()
            if first_task_started:
                if first_request.planned_time > state.simulator_time + 60.0:
                    continue
                else:
                    start_node = self._determine_robot_locations(robot_id=robot_id, 
                                                                state=state)
                    current_time = state.robots_current_time[robot_id]
                    remaining_goal_nodes = first_request.goal_nodes[first_request.completed_goals:]
                    remaining_wait_times = first_request.wait_times_at_goals_seconds[first_request.completed_goals:]
                    path_results = self._find_path_for_goal_nodes(robot_id=robot_id,
                                                                 start_node=start_node,
                                                                 start_time=current_time,
                                                                 goal_nodes=remaining_goal_nodes,
                                                                 wait_times_at_goals_seconds=remaining_wait_times,
                                                                 initial_planned_goal_indices=None,
                                                                 state=state,
                                                                 motion_planner=motion_planner,
                                                                 traversal_graph_generator=traversal_graph_generator)
                    planned_path, planned_goal_indices, planned_time_to_reach_last_goal = path_results

            else:
                start_node = self._determine_robot_locations(robot_id=robot_id, 
                                                             state=state)
                current_time = state.robots_current_time[robot_id]


    def assign_requests_to_robots(self, 
                                  state: PlanningState,
                                  requests_lists: Optional[RequestsLists], 
                                  motion_planner: MotionPlanner,
                                  traversal_graph_generator: TraversalGraphGenerator,
                                  debug: bool):
        # Extract assigned requests from state
        self._extract_assigned_requests_from_state(state=state)

        # Add new requests to the appropriate queues
        self._add_all_requests_to_queues(requests_lists=requests_lists,
                                         motion_planner=motion_planner,
                                         traversal_graph_generator=traversal_graph_generator)

        # Check if assigned requests can be reached
        self._check_assigned_requests_validity_and_generate_reservations(state=state,
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
        
        # Determine robot priorities based on earliest deadlines
        sorted_robot_ids = self._generate_robot_priorities(state=state,
                                                           motion_planner=motion_planner,
                                                           traversal_graph_generator=traversal_graph_generator)
        
        # Update the state with the new assignments and generate motion plans
        self._update_state_with_assigned_requests_and_generate_plans(sorted_robot_ids=sorted_robot_ids,
                                                                    state=state,
                                                                    motion_planner=motion_planner,
                                                                    traversal_graph_generator=traversal_graph_generator)