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

class HeuristicBasePolicy:
    def __init__(self, allow_deallocation: bool = False, base_policy_use: bool = False):
        self.requests_queue = TaskQueue()
        self.dummy_delivery_robot_profile = RobotProfile(radius=0.10, speed=0.20, robot_id=-1, robot_type="delivery")
        self.assigned_requests:  dict[int, list[str]]  = {}
        self.node_reservation_table = NodeReservationTable(reservations={},
                                                          robot_node_dict={})
        self.allow_deallocation = allow_deallocation
        self.base_policy_use = base_policy_use

    def _extract_assigned_requests_from_state(self, 
                                              state: PlanningState):
        self.assigned_requests = copy.deepcopy(state.assigned_requests)
    
    def _generate_simulated_state_from_current_state(self,
                                                     requests_lists: Optional[RequestsLists],
                                                     state: PlanningState):
        simulated_state = SimulatedState(planning_state=state)
        if requests_lists is not None:
            simulated_state.add_requests_to_state(requests_lists=requests_lists)
        return simulated_state
    
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
                                                     simulated_state: SimulatedState,
                                                     motion_planner: MotionPlanner,
                                                     traversal_graph_generator: TraversalGraphGenerator) -> float:
        request_struct = simulated_state.requests[request_id]
        if self.assigned_requests[robot_id]:
            last_assigned_request_id = self.assigned_requests[robot_id][-1]
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
                return float('inf')
            arrival_time = start_time + travel_time_to_goal
            if j == 0 and service_interval.start < request_struct.ordered_time:
                arrival_time = request_struct.ordered_time
            wait_time = request_struct.wait_times_at_goals_seconds[j]
            service_interval = PolicyHelpers._obtain_time_to_service_node(robot_id=robot_id,
                                                                node_reservation_table=self.node_reservation_table,
                                                                node_label=goal_node_label,
                                                                arrival_time=arrival_time,
                                                                wait_time=wait_time)
            if service_interval.end > request_struct.time_for_service:
                return float('inf'), None
            else:
                heuristic_cost = service_interval.end - request_struct.scheduled_time
                service_intervals.append(service_interval)
                start_node = goal_node
                start_time = service_interval.end
        
        return heuristic_cost, service_intervals
        
    def _assign_requests_to_robots(self,
                                  simulated_state: SimulatedState,
                                  motion_planner: MotionPlanner,
                                  traversal_graph_generator: TraversalGraphGenerator,
                                  debug: bool):
        while self.requests_queue.heap:
            next_request_id = self.requests_queue.pop_task()

            

    def assign_requests_to_robots(self, 
                                  state: PlanningState,
                                  simulated_state: Optional[SimulatedState],
                                  requests_lists: Optional[RequestsLists], 
                                  motion_planner: MotionPlanner,
                                  traversal_graph_generator: TraversalGraphGenerator,
                                  debug: bool):
        # Extract assigned requests from state
        self._extract_assigned_requests_from_state(state=state)

        if simulated_state is None:
            simulated_state = self._generate_simulated_state_from_current_state(state=state)

        # Add new requests to the appropriate queues
        smallest_pickup_deadline = self._add_all_requests_to_queues(requests_lists=requests_lists,
                                                                    motion_planner=motion_planner,
                                                                    traversal_graph_generator=traversal_graph_generator)
        
        if smallest_pickup_deadline:
            self._extract_node_reservations_from_state(state=state)

            # Assignment logic for robots
            self._assign_requests_to_robots(simulated_state=simulated_state,
                                            motion_planner=motion_planner,
                                            traversal_graph_generator=traversal_graph_generator,
                                            debug=debug)
        
        return simulated_state