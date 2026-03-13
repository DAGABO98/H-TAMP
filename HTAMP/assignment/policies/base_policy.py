from typing import Tuple
import copy
from typing import Optional
from HTAMP.assignment.assignment_helpers import AssignmentHelpers, TaskQueue
from HTAMP.assignment.policies.basic_helpers import PolicyHelpers
from HTAMP.environment.loc_dataclasses import TimeInterval
from HTAMP.environment.robot_dataclasses import RobotProfile
from HTAMP.environment.traversal_dataclasses import TraversalNode
from HTAMP.environment.traversal_graph_gen import TraversalGraphGenerator
from HTAMP.planning.motion_planner import MotionPlanner
from HTAMP.planning.planning_dataclasses import RequestsLists, NodeReservationTable, TimeReservation
from HTAMP.planning.state import PlanningState, SimulatedState

class Helpers:
    @staticmethod
    def generate_simulated_state_from_current_state(state: PlanningState) -> SimulatedState:
        simulated_state = SimulatedState(planning_state=state)
        return simulated_state
    
    @staticmethod
    def add_requests_to_simulated_state(requests_lists: Optional[RequestsLists],
                                        simulated_state: SimulatedState):
        if requests_lists is not None:
            simulated_state.add_requests_to_state(requests_lists=requests_lists)
    
    @staticmethod
    def extract_node_reservations_from_state(state: PlanningState,
                                              assigned_requests: dict[int, list[str]],
                                              node_reservation_table: NodeReservationTable):
        node_reservation_table.reset()
        for robot_id in assigned_requests.keys():
            if not assigned_requests[robot_id]:
                continue
            for request_id in assigned_requests[robot_id]:
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
                    node_reservation_table.add_reservation(node=goal_node_label,
                                                           reservation=reservation)
    
    @staticmethod
    def estimate_simulated_cost_to_fulfill_request(robot_id: int,
                                                     request_id: str,
                                                     currently_assigned_request_ids: list[str],
                                                     node_reservation_table: NodeReservationTable,
                                                     simulated_state: SimulatedState,
                                                     motion_planner: MotionPlanner,
                                                     traversal_graph_generator: TraversalGraphGenerator) -> Tuple[float, list[TimeInterval]]:
        request_struct = simulated_state.requests[request_id]
        if currently_assigned_request_ids:
            last_assigned_request_id = currently_assigned_request_ids[-1]
            last_assigned_request_struct = simulated_state.requests[last_assigned_request_id]
            planned_goal_node_label = last_assigned_request_struct.goal_nodes[-1]
            start_node = traversal_graph_generator.traversal_graph.nodes_dict[planned_goal_node_label]
            start_time = last_assigned_request_struct.planned_time
        else:
            start_node, start_time = AssignmentHelpers.determine_robot_nodes_and_times(robot_id=robot_id,
                                                                                    state=simulated_state)
        heuristic_cost = 0.0
        service_intervals = []
        for j, goal_node_label in enumerate(request_struct.goal_nodes):
            goal_node = traversal_graph_generator.traversal_graph.nodes_dict[goal_node_label]
            travel_time_to_goal = motion_planner.planner.heuristic(start_traversal_node=start_node,
                                                                    goal_traversal_node=goal_node,
                                                                    robot_profile=simulated_state.robot_profiles[robot_id])
            if travel_time_to_goal == float('inf'):
                return float('inf'), []
            arrival_time = start_time + travel_time_to_goal
            wait_time = request_struct.wait_times_at_goals_seconds[j]
            service_interval = PolicyHelpers._obtain_time_to_service_node(robot_id=robot_id,
                                                                node_reservation_table=node_reservation_table,
                                                                node_label=goal_node_label,
                                                                arrival_time=arrival_time,
                                                                wait_time=wait_time)
            
            if j == 0 and service_interval.start < request_struct.ordered_time:
                arrival_time = request_struct.ordered_time
                service_interval = PolicyHelpers._obtain_time_to_service_node(robot_id=robot_id,
                                                                node_reservation_table=node_reservation_table,
                                                                node_label=goal_node_label,
                                                                arrival_time=arrival_time,
                                                                wait_time=wait_time)
                
            if service_interval.end > request_struct.time_for_service:
                return float('inf'), []
            else:
                heuristic_cost = service_interval.end - request_struct.scheduled_time
                service_intervals.append(service_interval)
                start_node = goal_node
                start_time = service_interval.end
        
        return heuristic_cost, service_intervals
    
    @staticmethod
    def schedule_request_for_robot(robot_id: int,
                                   request_id: str,
                                   currently_assigned_request_ids: list[str],
                                   node_reservation_table: NodeReservationTable,
                                   service_intervals: list[TimeInterval],
                                   simulated_state: SimulatedState):
        currently_assigned_request_ids.append(request_id)
        request_struct = simulated_state.requests[request_id]
        request_struct.schedule_task(planned_time=service_intervals[-1].end,
                                     planned_goal_indices=list(range(request_struct.completed_goals, len(request_struct.goal_nodes))),
                                     assigned_robot_id=robot_id)
        simulated_state.assigned_requests[robot_id].append(request_id)
        for i, goal_label in enumerate(request_struct.goal_nodes):
            reservation_interval = service_intervals[i]
            reservation = TimeReservation(robot_id=robot_id,
                                        interval=reservation_interval)
            node_reservation_table.add_reservation(node=goal_label,
                                                    reservation=reservation)

class BasePolicy:
    def __init__(self, base_policy_use: bool = False):
        self.requests_queue = TaskQueue()
        self.dummy_delivery_robot_profile = RobotProfile(radius=0.10, speed=0.20, robot_id=-1, robot_type="delivery")
        self.assigned_requests:  dict[int, list[str]]  = {}
        self.node_reservation_table = NodeReservationTable(reservations={},
                                                          robot_node_dict={})
        self.base_policy_use = base_policy_use

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
                pickup_deadline = PolicyHelpers._add_request_to_queue_using_pickup_deadline(request=request,
                                           task_queue=self.requests_queue,
                                           delivery_robot_profile=self.dummy_delivery_robot_profile,
                                           motion_planner=motion_planner,
                                           traversal_graph_generator=traversal_graph_generator)
                pickup_deadlines.append(pickup_deadline)
        
        smallest_pickup_deadline = min(pickup_deadlines) if pickup_deadlines else None
        return smallest_pickup_deadline
    
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

        for i in range(len(potential_assignments)):
            robot_id, _ = potential_assignments[i]
            path_results = PolicyHelpers._get_planned_path_for_request_assignment(robot_id=robot_id,
                                                                        request_id=request_id,
                                                                        currently_assigned_request_ids=self.assigned_requests[robot_id],
                                                                        state=state,
                                                                        motion_planner=motion_planner,
                                                                        traversal_graph_generator=traversal_graph_generator,
                                                                        debug=debug)
            planned_path, planned_goal_indices, planned_time_to_reach_last_goal = path_results
            if planned_path:
                return robot_id, planned_path, planned_goal_indices, planned_time_to_reach_last_goal
        
        return None, None, None, None
        
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
                if not self.base_policy_use:
                    print(f"No potential assignments found for request {next_request_id}. Request is rejected.")
                request = state.requests[next_request_id]
                request.mark_rejected()
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
                    
                    if not self.base_policy_use:
                        print(f"Assigned request {next_request_id} to robot {min_robot_id}")
                else:
                    if not self.base_policy_use:
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
            Helpers.extract_node_reservations_from_state(state=state,
                                                         assigned_requests=self.assigned_requests,
                                                        node_reservation_table=self.node_reservation_table)

            # Assignment logic for robots
            self._assign_requests_to_robots(state=state,
                                            motion_planner=motion_planner,
                                            traversal_graph_generator=traversal_graph_generator,
                                            debug=debug)

class FutureCostEstimation:
    def __init__(self):
        self.requests_queue = TaskQueue()
        self.dummy_delivery_robot_profile = RobotProfile(radius=0.10, speed=0.20, robot_id=-1, robot_type="delivery")
        self.assigned_requests:  dict[int, list[str]]  = {}
        self.node_reservation_table = NodeReservationTable(reservations={},
                                                          robot_node_dict={})
    
    def reset(self):
        self.requests_queue = TaskQueue()
        self.assigned_requests = {}
        self.node_reservation_table.reset()

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
                pickup_deadline = PolicyHelpers._add_request_to_queue_using_pickup_deadline(request=request,
                                           task_queue=self.requests_queue,
                                           delivery_robot_profile=self.dummy_delivery_robot_profile,
                                           motion_planner=motion_planner,
                                           traversal_graph_generator=traversal_graph_generator)
                pickup_deadlines.append(pickup_deadline)
        
        smallest_pickup_deadline = min(pickup_deadlines) if pickup_deadlines else None
        return smallest_pickup_deadline
    
    def _determine_best_assignment_for_request(self,
                                              request_id: str,
                                              simulated_state: SimulatedState,
                                              motion_planner: MotionPlanner,
                                              traversal_graph_generator: TraversalGraphGenerator) -> Tuple[Optional[int], list[TimeInterval]]:
        best_robot_id = None
        best_heuristic_cost = float('inf')
        best_service_intervals = []

        for robot_id in simulated_state.robots_current_nodes.keys():
            heuristic_cost, service_intervals = Helpers.estimate_simulated_cost_to_fulfill_request(robot_id=robot_id,
                                                                                                 request_id=request_id,
                                                                                                 currently_assigned_request_ids=self.assigned_requests[robot_id],
                                                                                                 node_reservation_table=self.node_reservation_table,
                                                                                                 simulated_state=simulated_state,
                                                                                                 motion_planner=motion_planner,
                                                                                                 traversal_graph_generator=traversal_graph_generator)
            if heuristic_cost < best_heuristic_cost:
                best_heuristic_cost = heuristic_cost
                best_robot_id = robot_id
                best_service_intervals = service_intervals

        return best_robot_id, best_service_intervals
        
    def _assign_requests_to_robots(self,
                                  simulated_state: SimulatedState,
                                  motion_planner: MotionPlanner,
                                  traversal_graph_generator: TraversalGraphGenerator,
                                  debug: bool):
        while self.requests_queue.heap:
            next_request_id = self.requests_queue.pop_task()

            best_robot_id, best_service_intervals = self._determine_best_assignment_for_request(request_id=next_request_id,
                                                                                                 simulated_state=simulated_state,
                                                                                                 motion_planner=motion_planner,
                                                                                                 traversal_graph_generator=traversal_graph_generator)
            if best_robot_id is not None:
                Helpers.schedule_request_for_robot(robot_id=best_robot_id,
                                                request_id=next_request_id,
                                                currently_assigned_request_ids=self.assigned_requests[best_robot_id],
                                                node_reservation_table=self.node_reservation_table,
                                                service_intervals=best_service_intervals,
                                                simulated_state=simulated_state)
            else:
                request_struct = simulated_state.requests[next_request_id]
                request_struct.mark_rejected()
                

    def assign_requests_to_robots(self, 
                                  state: PlanningState,
                                  simulated_state: Optional[SimulatedState],
                                  node_reservation_table: Optional[NodeReservationTable],
                                  requests_lists: Optional[RequestsLists], 
                                  motion_planner: MotionPlanner,
                                  traversal_graph_generator: TraversalGraphGenerator,
                                  add_requests_in_request_lists: bool,
                                  debug: bool):

        if simulated_state is None:
            # Extract assigned requests from state
            self._extract_assigned_requests_from_state(state=state)
            simulated_state = Helpers.generate_simulated_state_from_current_state(state=state)
        else:
            self.assigned_requests = copy.deepcopy(simulated_state.assigned_requests)
        
        if add_requests_in_request_lists:
            Helpers.add_requests_to_simulated_state(requests_lists=requests_lists,
                                                    simulated_state=simulated_state)

        # Add new requests to the appropriate queues
        smallest_pickup_deadline = self._add_all_requests_to_queues(requests_lists=requests_lists,
                                                                    motion_planner=motion_planner,
                                                                    traversal_graph_generator=traversal_graph_generator)
        
        if smallest_pickup_deadline:
            if node_reservation_table is None:
                self.node_reservation_table.reset()
                Helpers.extract_node_reservations_from_state(state=state,
                                                             assigned_requests=self.assigned_requests,
                                                            node_reservation_table=self.node_reservation_table)
            else:
                self.node_reservation_table = copy.deepcopy(node_reservation_table)

            # Assignment logic for robots
            self._assign_requests_to_robots(simulated_state=simulated_state,
                                            motion_planner=motion_planner,
                                            traversal_graph_generator=traversal_graph_generator,
                                            debug=debug)
        
        return simulated_state