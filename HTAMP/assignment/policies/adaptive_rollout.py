import copy
from typing import Optional

import pandas as pd
from HTAMP.assignment.assignment_helpers import TaskQueue
from HTAMP.assignment.policies.basic_helpers import PolicyHelpers
from HTAMP.assignment.policies.rollout_helpers import RolloutHelpers
from HTAMP.assignment.policies.sequential_greedy import SequentialGreedy
from HTAMP.data_processing.processing_dataclasses import AnnotatedDataFiles
from HTAMP.environment.robot_dataclasses import RobotProfile
from HTAMP.environment.traversal_graph_gen import TraversalGraphGenerator
from HTAMP.planning.motion_planner import MotionPlanner
from HTAMP.planning.planning_dataclasses import AllTaskProperties, RequestsLists
from HTAMP.planning.state import PlanningState
from HTAMP.planning.request_handler import PlanningRequestHandler

class AdaptiveRollout:
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
                 allow_deallocation: bool,
                 allow_reweighting: bool):
        self.requests_queue = TaskQueue()
        self.predicted_requests_queue = TaskQueue()
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

        future_scheduled_requests_lists = RolloutHelpers._extract_scheduled_requests(date_stamp=self.date_stamp,
                                                                                    hour=hour,
                                                                                    minute=minute,
                                                                                    look_ahead_minutes=look_ahead_minutes,
                                                                                    planning_request_handler=self.planning_request_handler,
                                                                                    initial_time=self.initial_time,
                                                                                    all_task_properties=self.all_task_properties,
                                                                                    traversal_graph_generator=traversal_graph_generator)
        
        predicted_requests_dict = RolloutHelpers._extract_predicted_requests(state=current_state, 
                                                                   hour=hour,
                                                                   minute=minute)

        for robot_id in robot_ids:
            current_state = copy.deepcopy(state)
            RolloutHelpers._add_requests_to_state(requests_lists=future_scheduled_requests_lists,
                                             state=current_state)
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
                                                           future_scheduled_requests_lists=future_scheduled_requests_lists,
                                                           predicted_requests_dict=predicted_requests_dict,
                                                           motion_planner=motion_planner,
                                                           traversal_graph_generator=traversal_graph_generator,
                                                           look_ahead_minutes=look_ahead_minutes,
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
        
        if smallest_pickup_deadline:
            if self.allow_deallocation:
                self._deallocate_requests(smallest_pickup_deadline=smallest_pickup_deadline,
                                          state=state,
                                          motion_planner=motion_planner,
                                          traversal_graph_generator=traversal_graph_generator,
                                          debug=debug)

            # Assignment logic for robots
            self._assign_requests_to_robots(state=state,
                                            requests_lists=requests_lists,
                                            motion_planner=motion_planner,
                                            traversal_graph_generator=traversal_graph_generator,
                                            hour=hour,
                                            minute=minute,
                                            look_ahead_minutes=look_ahead_minutes,
                                            debug=debug)