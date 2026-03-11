import copy
from typing import Optional

import pandas as pd
from HTAMP.assignment.policies.basic_helpers import PolicyHelpers
from HTAMP.assignment.policies.rollout_helpers import RolloutHelpers
from HTAMP.assignment.policies.sequential_greedy import SequentialGreedy
from HTAMP.data_processing.processing_dataclasses import AnnotatedDataFiles
from HTAMP.environment.robot_dataclasses import RobotProfile
from HTAMP.environment.traversal_graph_gen import TraversalGraphGenerator
from HTAMP.planning.motion_planner import MotionPlanner
from HTAMP.planning.planning_dataclasses import AllTaskProperties, RequestsLists
from HTAMP.planning.request_handler import PlanningRequestHandler
from HTAMP.planning.state import PlanningState

class VanillaRollout:
    def __init__(self,
                 start_date: str,
                 end_date: str,
                 date_stamp: pd.Timestamp,
                 floor_number: int,
                 annotated_data_files: AnnotatedDataFiles,
                 request_dir: str,
                 use_saved_request_data: bool,
                 initial_time: pd.Timestamp,
                 all_task_properties:AllTaskProperties,
                 fps: int,
                 allow_premptive_moves: bool):
        self.dummy_delivery_robot_profile = RobotProfile(radius=0.10, speed=0.20, robot_id=-1, robot_type="delivery")
        self.assigned_requests:  dict[int, list[str]]  = {}
        self.date_stamp = date_stamp
        self.initial_time = initial_time
        self.all_task_properties = all_task_properties
        self.fps = fps
        self.base_policy = SequentialGreedy(base_policy_use=True)
        self.planning_request_handler = PlanningRequestHandler(start_date=start_date,
                                              end_date=end_date,
                                              date_stamp=date_stamp,
                                              floor_number=floor_number,
                                              annotated_data_files=annotated_data_files,
                                              request_dir=request_dir,
                                              use_saved_data=use_saved_request_data)
        self.allow_premptive_moves = allow_premptive_moves
        
    def _extract_assigned_requests_from_state(self, 
                                              state: PlanningState):
        self.assigned_requests = copy.deepcopy(state.assigned_requests)
    
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
        min_path_cost = float('inf')
        min_path_raw_cost = float('inf')
        min_planned_path = None
        min_planned_goal_indices = None
        min_planned_time_to_reach_last_goal = float('inf')
        min_robot_id = None

        orig_unmodified_cost, orig_truncated_cost = RolloutHelpers._extract_cost_for_assigned_requests(state=state)
        request_struct = state.requests[request_id]
        if request_struct.request_type == "medication":
            robot_type = "delivery"
        else:
            robot_type = "monitoring"
        robot_ids = state.get_robots_of_type(robot_type=robot_type)

        for robot_id in robot_ids:
            current_state = copy.deepcopy(state)
            path_results = PolicyHelpers._get_planned_path_for_request_assignment(robot_id=robot_id,
                                                                        request_id=request_id,
                                                                        currently_assigned_request_ids=self.assigned_requests[robot_id],
                                                                        state=current_state,
                                                                        motion_planner=motion_planner,
                                                                        traversal_graph_generator=traversal_graph_generator,
                                                                        debug=debug)
            planned_path, planned_goal_indices, planned_time_to_reach_last_goal = path_results

            if planned_path:
                if debug:
                    print(f"Found a valid path for potential assignment of request {request_id} to robot {robot_id}. Simulating future assignments...")
                PolicyHelpers._schedule_request(robot_id=robot_id,
                                                request_id=request_id,
                                                currently_assigned_request_ids=self.assigned_requests[robot_id],
                                                node_reservation_table=None,
                                                planned_path=planned_path,
                                                planned_goal_indices=planned_goal_indices,
                                                planned_time_to_reach_last_goal=planned_time_to_reach_last_goal,
                                                state=current_state,
                                                motion_planner=motion_planner,
                                                traversal_graph_generator=traversal_graph_generator)
                
                RolloutHelpers._simulate_future_assignments(base_policy=self.base_policy,
                                                           current_state=current_state,
                                                           requests_lists=requests_lists,
                                                           motion_planner=motion_planner,
                                                           traversal_graph_generator=traversal_graph_generator,
                                                           date_stamp=self.date_stamp,
                                                           hour=hour,
                                                           minute=minute,
                                                           look_ahead_minutes=look_ahead_minutes,
                                                           planning_request_handler=self.planning_request_handler,
                                                           initial_time=self.initial_time,
                                                           all_task_properties=self.all_task_properties,
                                                           fps=self.fps)
                
                new_unmodified_cost, new_truncated_cost = RolloutHelpers._extract_cost_for_assigned_requests(state=current_state)
                path_cost = new_truncated_cost - orig_truncated_cost
                path_raw_cost = new_unmodified_cost - orig_unmodified_cost

                if path_cost < min_path_cost or (path_cost == min_path_cost and path_raw_cost < min_path_raw_cost):
                    min_path_cost = path_cost
                    min_path_raw_cost = path_raw_cost
                    min_planned_path = planned_path
                    min_planned_goal_indices = planned_goal_indices
                    min_planned_time_to_reach_last_goal = planned_time_to_reach_last_goal
                    min_robot_id = robot_id

        return min_robot_id, min_planned_path, min_planned_goal_indices, min_planned_time_to_reach_last_goal

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
                PolicyHelpers._schedule_request(robot_id=robot_id,
                                                request_id=request_id,
                                                currently_assigned_request_ids=self.assigned_requests[robot_id],
                                                node_reservation_table=None,
                                                planned_path=planned_path,
                                                planned_goal_indices=planned_goal_indices,
                                                planned_time_to_reach_last_goal=planned_time_to_reach_last_goal,
                                                state=state,
                                                motion_planner=motion_planner,
                                                traversal_graph_generator=traversal_graph_generator)
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
        
        # TODO: determine if robots are unassigned
        
        if smallest_pickup_deadline:

            # Assignment logic for robots
            self._assign_requests_to_robots(state=state,
                                            requests_lists=requests_lists,
                                            motion_planner=motion_planner,
                                            traversal_graph_generator=traversal_graph_generator,
                                            hour=hour,
                                            minute=minute,
                                            look_ahead_minutes=look_ahead_minutes,
                                            debug=debug)