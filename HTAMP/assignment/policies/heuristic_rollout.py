import copy
from typing import Optional

import pandas as pd
from HTAMP.assignment.assignment_helpers import AssignmentHelpers, TaskQueue
from HTAMP.assignment.policies.basic_helpers import PolicyHelpers
from HTAMP.assignment.policies.base_policy import FutureCostEstimation, Helpers
from HTAMP.assignment.policies.rollout_helpers import RolloutHelpers
from HTAMP.data_processing.processing_dataclasses import AnnotatedDataFiles
from HTAMP.environment.robot_dataclasses import RobotProfile
from HTAMP.environment.traversal_graph_gen import TraversalGraphGenerator
from HTAMP.planning.motion_planner import MotionPlanner
from HTAMP.planning.planning_dataclasses import AllTaskProperties, NodeReservationTable, RequestsLists, TimeReservation
from HTAMP.planning.state import PlanningState
from HTAMP.planning.request_handler import PlanningRequestHandler

class HeuristicRollout:
    def __init__(self,
                 start_date: str,
                 end_date: str,
                 date_stamp: pd.Timestamp,
                 end_hour: int,
                 floor_number: int,
                 annotated_data_files: AnnotatedDataFiles,
                 request_dir: str,
                 use_saved_request_data: bool,
                 initial_time: pd.Timestamp,
                 all_task_properties:AllTaskProperties,
                 allow_deallocation: bool,
                 allow_reweighting: bool):
        self.requests_queue = TaskQueue()
        self.predicted_requests_queue = TaskQueue()
        self.dummy_delivery_robot_profile = RobotProfile(radius=0.10, speed=0.20, robot_id=-1, robot_type="delivery")
        self.assigned_requests:  dict[int, list[str]]  = {}
        self.date_stamp = date_stamp
        self.end_hour = end_hour
        self.initial_time = initial_time
        self.all_task_properties = all_task_properties
        self.planning_request_handler = PlanningRequestHandler(start_date=start_date,
                                              end_date=end_date,
                                              date_stamp=date_stamp,
                                              floor_number=floor_number,
                                              annotated_data_files=annotated_data_files,
                                              request_dir=request_dir,
                                              use_saved_data=use_saved_request_data)
        self.node_reservation_table = NodeReservationTable(reservations={},
                                                          robot_node_dict={})
        self.cost_estimator = FutureCostEstimation()
        self.allow_deallocation = allow_deallocation
        self.allow_reweighting = allow_reweighting
        
    def _extract_assigned_requests_from_state(self, 
                                              state: PlanningState):
        self.assigned_requests = copy.deepcopy(state.assigned_requests)
    
    def _add_all_real_requests_to_queues(self, 
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
    
    def _deallocate_requests_from_robot(self,
                                       robot_id: int,
                                       state: PlanningState,
                                       motion_planner: MotionPlanner,
                                       traversal_graph_generator: TraversalGraphGenerator,
                                       debug: bool) -> dict[int, list[str]]:
        
        if not self.assigned_requests[robot_id]:
            return None
        
        largest_assigned_request_index = None
        for i, request_id in enumerate(self.assigned_requests[robot_id]):
            request_struct = state.requests[request_id]
            if request_struct.started:
                largest_assigned_request_index = i
                if debug:
                    print(f"Skipping deallocation of request {request_id} from robot {robot_id} because the request has already started.")
                continue
            pickup_deadline = PolicyHelpers._calculate_pickup_deadline(delivery_robot_profile=self.dummy_delivery_robot_profile,
                                                                      request=request_struct,
                                                                      motion_planner=motion_planner,
                                                                      traversal_graph_generator=traversal_graph_generator)
            if debug:
                print(f"Deallocating request {request_id} from robot {robot_id}.")
            self.requests_queue.add_task(priority=pickup_deadline, 
                                        task_id=request_id)
            request_struct.reset_assignment()
            state.remove_request_from_robot(request_id=request_id, 
                                            robot_id=robot_id)
        
        return largest_assigned_request_index
    
    def _determine_if_new_request_triggers_reassignment(self,
                                        smallest_pickup_deadline: float,
                                        state: PlanningState,
                                        motion_planner: MotionPlanner,
                                        traversal_graph_generator: TraversalGraphGenerator) -> bool:
        for robot_id in self.assigned_requests.keys():
            if not self.assigned_requests[robot_id]:
                continue

            for i, request_id in enumerate(self.assigned_requests[robot_id]):
                request_struct = state.requests[request_id]
                if request_struct.started:
                    continue
                pickup_deadline = PolicyHelpers._calculate_pickup_deadline(delivery_robot_profile=self.dummy_delivery_robot_profile,
                                                                          request=request_struct,
                                                                          motion_planner=motion_planner,
                                                                          traversal_graph_generator=traversal_graph_generator)
                if pickup_deadline > smallest_pickup_deadline:
                    return True
        return False
    
    def _determine_if_reweighting_triggers_reassignment(self,
                                        state: PlanningState,
                                        debug: bool) -> bool:
        # TODO: to be implemented once the prediction errors is calculated
        return False

    def _deallocate_requests(self,
                             smallest_pickup_deadline: float,
                             state: PlanningState,
                             motion_planner: MotionPlanner,
                             traversal_graph_generator: TraversalGraphGenerator,
                             debug: bool):
        new_req_trigger_reassignment = self._determine_if_new_request_triggers_reassignment(smallest_pickup_deadline=smallest_pickup_deadline,
                                                                                   state=state,
                                                                                   motion_planner=motion_planner,
                                                                                   traversal_graph_generator=traversal_graph_generator)
        
        if self.allow_reweighting:
            weighting_trigger_reassignment = self._determine_if_reweighting_triggers_reassignment(state=state,
                                                                                                  debug=debug)
            trigger_reassignment = new_req_trigger_reassignment or weighting_trigger_reassignment
        else:
            trigger_reassignment = new_req_trigger_reassignment

        if trigger_reassignment:
            print("New request triggers reassignment. Deallocating requests from robots...")
            for robot_id in self.assigned_requests.keys():
                largest_assigned_request_index = self._deallocate_requests_from_robot(robot_id=robot_id,
                                                                                     state=state,
                                                                                     motion_planner=motion_planner,
                                                                                     traversal_graph_generator=traversal_graph_generator,
                                                                                     debug=debug)
                if largest_assigned_request_index is not None:
                    if largest_assigned_request_index < len(self.assigned_requests[robot_id]) - 1:
                        self.assigned_requests[robot_id] = self.assigned_requests[robot_id][:largest_assigned_request_index+1]
                        PolicyHelpers._generate_motion_plan_to_depot(robot_id=robot_id,
                                                         currently_assigned_request_ids=self.assigned_requests[robot_id],
                                                         state=state,
                                                         motion_planner=motion_planner,
                                                         traversal_graph_generator=traversal_graph_generator,
                                                         debug=debug)
                    else:
                        continue

                else:
                    if self.assigned_requests[robot_id]:
                        self.assigned_requests[robot_id] = []
                        PolicyHelpers._generate_motion_plan_to_depot(robot_id=robot_id,
                                                            currently_assigned_request_ids=self.assigned_requests[robot_id],
                                                            state=state,
                                                            motion_planner=motion_planner,
                                                            traversal_graph_generator=traversal_graph_generator,
                                                            debug=debug)
                    else:
                        continue
            else:
                print("New request does not trigger reassignment. No requests are deallocated.")
    
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
    
    def _get_assignment_with_minimum_future_costs(self,
                                            request_id: str,
                                            requests_lists: RequestsLists,
                                            potential_assignments: list[int],
                                            state: PlanningState,
                                            motion_planner: MotionPlanner,
                                            traversal_graph_generator: TraversalGraphGenerator,
                                            hour: int,
                                            minute: int,
                                            look_ahead_minutes: int,
                                            debug: bool):
        future_scheduled_requests_lists = RolloutHelpers._extract_scheduled_requests(date_stamp=self.date_stamp,
                                                                                    hour=hour,
                                                                                    minute=minute,
                                                                                    look_ahead_minutes=look_ahead_minutes,
                                                                                    end_hour=self.end_hour,
                                                                                    planning_request_handler=self.planning_request_handler,
                                                                                    initial_time=self.initial_time,
                                                                                    all_task_properties=self.all_task_properties,
                                                                                    traversal_graph_generator=traversal_graph_generator)
        
        predicted_requests_dict = RolloutHelpers._extract_predicted_requests(state=state, 
                                                                   hour=hour,
                                                                   minute=minute)
        
        current_predicted_requests_dict, future_predicted_requests_dict = RolloutHelpers._split_predicted_requests_dict(predicted_requests_dict=predicted_requests_dict,
                                                                                                                        look_ahead_minutes=look_ahead_minutes)
        
        
        min_robot_id = None
        min_planned_path = None
        min_planned_goal_indices = None
        min_planned_time_to_reach_last_goal = float('inf')
        min_path_cost = float('inf')
        min_path_raw_cost = float('inf')
        
        
        for i in range(len(potential_assignments)):
            robot_id = potential_assignments[i][0]
            current_state = state.fork()
            current_motion_planner = motion_planner.fork_with_reservations()
            currently_assigned_request_ids = copy.deepcopy(self.assigned_requests[robot_id])
            path_results = PolicyHelpers._get_planned_path_for_request_assignment(robot_id=robot_id,
                                                                        request_id=request_id,
                                                                        currently_assigned_request_ids=currently_assigned_request_ids,
                                                                        state=current_state,
                                                                        motion_planner=current_motion_planner,
                                                                        traversal_graph_generator=traversal_graph_generator,
                                                                        debug=debug)
            planned_path, planned_goal_indices, planned_time_to_reach_last_goal = path_results

            if planned_path:
                if debug:
                    print(f"Found a valid path for potential assignment of request {request_id} to robot {robot_id}. Simulating future assignments...")
                    print(f"Planned goal indices for assignment of request {request_id} to robot {robot_id}: {planned_goal_indices}")
                current_node_reservation_table = copy.deepcopy(self.node_reservation_table)
                PolicyHelpers._schedule_request(robot_id=robot_id,
                                                request_id=request_id,
                                                currently_assigned_request_ids=currently_assigned_request_ids,
                                                node_reservation_table=current_node_reservation_table,
                                                planned_path=planned_path,
                                                planned_goal_indices=planned_goal_indices,
                                                planned_time_to_reach_last_goal=planned_time_to_reach_last_goal,
                                                state=current_state,
                                                motion_planner=current_motion_planner,
                                                traversal_graph_generator=traversal_graph_generator)
                
                self.cost_estimator.reset()
                unmodified_cost, truncated_cost = RolloutHelpers._estimate_future_costs_for_scheduled_and_predicted_assignments(
                                                                                        cost_estimator=self.cost_estimator,
                                                                                        current_state=current_state,
                                                                                        requests_lists=requests_lists,
                                                                                        current_node_reservation_table=current_node_reservation_table,
                                                                                        current_predicted_requests_dict=current_predicted_requests_dict,
                                                                                        future_scheduled_requests_lists=future_scheduled_requests_lists,
                                                                                        future_predicted_requests_dict=future_predicted_requests_dict,
                                                                                        motion_planner=current_motion_planner,
                                                                                        traversal_graph_generator=traversal_graph_generator)

                if truncated_cost < min_path_cost or (truncated_cost == min_path_cost and unmodified_cost < min_path_raw_cost):
                    min_path_cost = truncated_cost
                    min_path_raw_cost = unmodified_cost
                    min_planned_path = planned_path
                    min_planned_goal_indices = planned_goal_indices
                    min_planned_time_to_reach_last_goal = planned_time_to_reach_last_goal
                    min_robot_id = robot_id

        return min_robot_id, min_planned_path, min_planned_goal_indices, min_planned_time_to_reach_last_goal
    
    def _assign_single_request_to_robot_team(self,
                                            request_id: str,
                                            requests_lists: RequestsLists,
                                            state: PlanningState,
                                            motion_planner: MotionPlanner,
                                            traversal_graph_generator: TraversalGraphGenerator,
                                            hour: int,
                                            minute: int,
                                            look_ahead_minutes: int,
                                            debug: bool):
        
        potential_assignments = self._determine_potential_assignments_for_request(request_id=request_id,
                                                                                 state=state,
                                                                                 motion_planner=motion_planner,
                                                                                 traversal_graph_generator=traversal_graph_generator)
        if len(potential_assignments) == 0:
            if debug:
                print(f"No valid paths found for any potential assignment of request {request_id}.")
            return None, None, None, None
        elif len(potential_assignments) == 1:
            robot_id = potential_assignments[0][0]
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
            else:
                if debug:
                    print(f"No valid path found for the only potential assignment of request {request_id} to robot {robot_id}.")
                return None, None, None, None
        else:
            return self._get_assignment_with_minimum_future_costs(request_id=request_id,
                                                                requests_lists=requests_lists,
                                                                potential_assignments=potential_assignments,
                                                                state=state,
                                                                motion_planner=motion_planner,
                                                                traversal_graph_generator=traversal_graph_generator,
                                                                hour=hour,
                                                                minute=minute,
                                                                look_ahead_minutes=look_ahead_minutes,
                                                                debug=debug)
        

    def _assign_requests_to_robots(self,
                                   requests_lists: Optional[RequestsLists],
                                  state: PlanningState,
                                  motion_planner: MotionPlanner,
                                  traversal_graph_generator: TraversalGraphGenerator,
                                  hour: int,
                                  minute: int,
                                  look_ahead_minutes: int,
                                  debug: bool):
        if requests_lists is None:
            return
        else:
            current_requests_lists = copy.deepcopy(requests_lists)

        while self.requests_queue.heap:
            request_id = self.requests_queue.pop_task()
            if debug:
                print(f"Attempting to assign request {request_id} at simulator time {state.simulator_time}")
            RolloutHelpers._remove_request_from_requests_lists(request_id=request_id, 
                                                              requests_lists=current_requests_lists)
            
            path_results = self._assign_single_request_to_robot_team(request_id=request_id,
                                                                     requests_lists=current_requests_lists,
                                                                     state=state,
                                                                     motion_planner=motion_planner,
                                                                     traversal_graph_generator=traversal_graph_generator,
                                                                     hour=hour,
                                                                     minute=minute,
                                                                     look_ahead_minutes=look_ahead_minutes,
                                                                     debug=debug)

            robot_id, planned_path, planned_goal_indices, planned_time_to_reach_last_goal = path_results
            if planned_path:
                if debug:
                    print(f"1) assigned requests for robot {robot_id}: {self.assigned_requests[robot_id]}")
                    print(f"1) State path for robot {robot_id}: {state.robot_paths[robot_id]}")
                    print(f"1) Planned path for assignment of request {request_id} to robot {robot_id}: {planned_path}")
                    print(f"1) Planned goal indices for assignment of request {request_id} to robot {robot_id}: {planned_goal_indices}")
                PolicyHelpers._schedule_request(robot_id=robot_id,
                                                request_id=request_id,
                                                currently_assigned_request_ids=self.assigned_requests[robot_id],
                                                node_reservation_table=self.node_reservation_table,
                                                planned_path=planned_path,
                                                planned_goal_indices=planned_goal_indices,
                                                planned_time_to_reach_last_goal=planned_time_to_reach_last_goal,
                                                state=state,
                                                motion_planner=motion_planner,
                                                traversal_graph_generator=traversal_graph_generator,
                                                debug=debug)
                print(f"Assigned request {request_id} to robot {robot_id}")
            else:
                print(f"Failed to find a valid path for any of the potential assignments of request {request_id}. Request is rejected.")
                request = state.requests[request_id]
                request.mark_rejected()

    def assign_requests_to_robots(self, 
                                  state: PlanningState,
                                  requests_lists: Optional[RequestsLists], 
                                  motion_planner: MotionPlanner,
                                  traversal_graph_generator: TraversalGraphGenerator,
                                  hour: int,
                                  minute: int,
                                  look_ahead_minutes: int,
                                  debug: bool):
        # Extract assigned requests from state
        self._extract_assigned_requests_from_state(state=state)

        # Add new requests to the appropriate queues
        smallest_pickup_deadline = self._add_all_real_requests_to_queues(requests_lists=requests_lists,
                                                                    motion_planner=motion_planner,
                                                                    traversal_graph_generator=traversal_graph_generator)
        
        if smallest_pickup_deadline:
            if self.allow_deallocation:
                self._deallocate_requests(smallest_pickup_deadline=smallest_pickup_deadline,
                                          state=state,
                                          motion_planner=motion_planner,
                                          traversal_graph_generator=traversal_graph_generator,
                                          debug=debug)
            
            Helpers.extract_node_reservations_from_state(state=state,
                                                         assigned_requests=self.assigned_requests,
                                                        node_reservation_table=self.node_reservation_table)

            # Assignment logic for robots
            self._assign_requests_to_robots(state=state,
                                            requests_lists=requests_lists,
                                            motion_planner=motion_planner,
                                            traversal_graph_generator=traversal_graph_generator,
                                            hour=hour,
                                            minute=minute,
                                            look_ahead_minutes=look_ahead_minutes,
                                            debug=debug)