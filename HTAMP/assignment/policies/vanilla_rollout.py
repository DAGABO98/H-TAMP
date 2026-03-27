import copy
from typing import Optional

import pandas as pd
from HTAMP.assignment.assignment_helpers import AssignmentHelpers, RequestsDict
from HTAMP.assignment.policies.base_policy import FutureCostEstimation, Helpers
from HTAMP.assignment.policies.basic_helpers import PolicyHelpers
from HTAMP.assignment.policies.rollout_helpers import RolloutHelpers
from HTAMP.data_processing.processing_dataclasses import AnnotatedDataFiles
from HTAMP.environment.robot_dataclasses import RobotProfile
from HTAMP.environment.traversal_graph_gen import TraversalGraphGenerator
from HTAMP.planning.motion_planner import MotionPlanner
from HTAMP.planning.planning_dataclasses import AllTaskProperties, NodeReservationTable, RequestsLists
from HTAMP.planning.request_handler import PlanningRequestHandler
from HTAMP.planning.state import PlanningState

class VanillaRollout:
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
        self.unassigned_requests_dict = RequestsDict(monitoring=[], medication=[])
        self.dummy_delivery_robot_profile = RobotProfile(radius=0.10, speed=0.20, robot_id=-1, robot_type="delivery")
        self.assigned_requests:  dict[int, list[str]]  = {}
        self.blocked_robots = set()
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
    
    def _add_all_real_requests_to_requests_dict(self, 
                                    requests_lists: Optional[RequestsLists]):
        if requests_lists is None:
            return False
        for data_field in requests_lists.__dataclass_fields__.keys():
            requests_list = getattr(requests_lists, data_field)
            for request in requests_list:
                self.unassigned_requests_dict.add_request(request)
        
        return True
    
    def _get_available_robots(self,
                              state: PlanningState,
                              new_requests_added: bool):
        if new_requests_added:
            self.blocked_robots = set()

        available_robots = set()
        for robot_id in state.assigned_requests.keys():
            if robot_id in self.blocked_robots:
                continue
            if len(state.assigned_requests[robot_id]) == 0:
                available_robots.add(robot_id)
            else:
                last_assigned_request_id = state.assigned_requests[robot_id][-1]
                last_assigned_request = state.requests[last_assigned_request_id]
                planned_time_to_reach_request = last_assigned_request.planned_time
                if state.simulator_time >= planned_time_to_reach_request - 60.0: 
                    available_robots.add(robot_id)

        return available_robots

    
    def _convert_requests_dict_to_requests_lists(self, state: PlanningState):
        requests_lists = RequestsLists(blood_pressure_requests=[], 
                                       heart_rate_requests=[],
                                       respiratory_rate_requests=[],
                                        oxygen_saturation_requests=[],
                                        temperature_requests=[],
                                        medications_requests=[])
        for request_id in self.unassigned_requests_dict.monitoring:
            request = copy.deepcopy(state.requests[request_id])
            if request.request_type == "blood_pressure":
                requests_lists.blood_pressure_requests.append(request)
            elif request.request_type == "heart_rate":
                requests_lists.heart_rate_requests.append(request)
            elif request.request_type == "respiratory_rate":
                requests_lists.respiratory_rate_requests.append(request)
            elif request.request_type == "oxygen_saturation":
                requests_lists.oxygen_saturation_requests.append(request)
            elif request.request_type == "temperature":
                requests_lists.temperature_requests.append(request)
        for request_id in self.unassigned_requests_dict.medication:
            request = copy.deepcopy(state.requests[request_id])
            if request.request_type == "medication":
                requests_lists.medications_requests.append(request)
    
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
            current_node_reservation_table = copy.deepcopy(self.node_reservation_table)
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
        new_requests_added = self._add_all_real_requests_to_requests_dict(requests_lists=requests_lists)
        available_robots = self._get_available_robots(state=state,
                                                     new_requests_added=new_requests_added)
        
        if available_robots:
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