import copy
from typing import Optional

import pandas as pd
from HTAMP.assignment.assignment_helpers import AssignmentHelpers, TaskQueue
from HTAMP.assignment.policies.base_policy import SequentialGreedyBasePolicy
from HTAMP.assignment.policies.helpers import PolicyHelpers
from HTAMP.data_processing.processing_dataclasses import AnnotatedDataFiles
from HTAMP.environment.loc_dataclasses import TimeInterval
from HTAMP.environment.robot_dataclasses import RobotProfile
from HTAMP.environment.traversal_dataclasses import TraversalNode
from HTAMP.environment.traversal_graph_gen import TraversalGraphGenerator
from HTAMP.planning.motion_planner import MotionPlanner
from HTAMP.planning.planning_dataclasses import AllTaskProperties, NodeReservationTable, RequestsLists, TaskRequest, TimeReservation, TimeSignal
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
                 allow_deallocation: bool,
                 allow_reweighting: bool):
        self.requests_queue = TaskQueue()
        self.dummy_delivery_robot_profile = RobotProfile(radius=0.10, speed=0.20, robot_id=-1, robot_type="delivery")
        self.assigned_requests:  dict[int, list[str]]  = {}
        self.node_reservation_table = NodeReservationTable(reservations={},
                                                          robot_node_dict={})
        self.date_stamp = date_stamp
        self.initial_time = initial_time
        self.all_task_properties = all_task_properties
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
                                        state: PlanningState) -> bool:
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
    
    def _extract_scheduled_requests(self,
                                    hour: int,
                                    minute: int,
                                    look_ahead_minutes: int,
                                    traversal_graph_generator: TraversalGraphGenerator) -> RequestsLists:
        if hour <= 23:
            original_time_signal = TimeSignal(year=self.date_stamp.year,
                                    month=self.date_stamp.month,
                                    day=self.date_stamp.day,
                                    hour=hour,
                                    minute=minute)
            
            shifted_time_stamp = original_time_signal.time_stamp + pd.Timedelta(minutes=look_ahead_minutes)
            if shifted_time_stamp.day != self.date_stamp.day:
                return None
            else:
                if shifted_time_stamp.hour == 23 and 60 - shifted_time_stamp.minute < look_ahead_minutes:
                    return None
                else:
                    shifted_time_signal = TimeSignal(year=shifted_time_stamp.year,
                                                 month=shifted_time_stamp.month,
                                                 day=shifted_time_stamp.day,
                                                 hour=shifted_time_stamp.hour,
                                                 minute=shifted_time_stamp.minute)
                    scheduled_requests_lists: RequestsLists = self.planning_request_handler.get_all_requests_for_time_signal(time_signal=shifted_time_signal,
                                                                                                    initial_time=self.initial_time,
                                                                                                    all_task_properties=self.all_task_properties,
                                                                                                    look_ahead_minutes=look_ahead_minutes,
                                                                                                    traversal_graph_generator=traversal_graph_generator)

                return scheduled_requests_lists
        else:
            return None
    
    def _add_requests_to_state(self, 
                               requests_lists: RequestsLists, 
                               state: PlanningState):
        requests: list[TaskRequest] = []
        for request_list in [requests_lists.blood_pressure_requests,
                             requests_lists.heart_rate_requests,
                             requests_lists.respiratory_rate_requests,
                             requests_lists.temperature_requests,
                             requests_lists.oxygen_saturation_requests,
                             requests_lists.medications_requests]:
            requests.extend(request_list)
        state.add_new_requests(requests=requests)

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
        smallest_pickup_deadline = self._add_all_requests_to_queues(requests_lists=requests_lists,
                                                                    motion_planner=motion_planner,
                                                                    traversal_graph_generator=traversal_graph_generator)
        
        if smallest_pickup_deadline:
            if self.allow_deallocation:
                self._deallocate_requests(smallest_pickup_deadline=smallest_pickup_deadline,
                                          state=state,
                                          motion_planner=motion_planner,
                                          traversal_graph_generator=traversal_graph_generator,
                                          debug=debug)
                
            # TODO: Only add requests to state after copying state for rollout
            scheduled_requests = self._extract_scheduled_requests(hour=hour,
                                                                minute=minute,
                                                                look_ahead_minutes=look_ahead_minutes,
                                                                traversal_graph_generator=traversal_graph_generator)
            
            self._extract_node_reservations_from_state(state=state)

            # Assignment logic for robots
            self._assign_requests_to_robots(state=state,
                                            motion_planner=motion_planner,
                                            traversal_graph_generator=traversal_graph_generator,
                                            debug=debug)