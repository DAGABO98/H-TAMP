import copy
from typing import Optional

import pandas as pd
from HTAMP.assignment.assignment_helpers import AssignmentHelpers, TaskQueue
from HTAMP.assignment.policies.base_policy import SequentialGreedyBasePolicy
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
                 all_task_properties:AllTaskProperties):
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
    
    def _extract_scheduled_requests(self,
                                    hour: int,
                                    minute: int,
                                    look_ahead_minutes: int,
                                    traversal_graph_generator: TraversalGraphGenerator) -> RequestsLists:
        original_time_signal = TimeSignal(year=self.date_stamp.year,
                                 month=self.date_stamp.month,
                                 day=self.date_stamp.day,
                                 hour=hour,
                                 minute=minute)
        
        shifted_time_stamp = original_time_signal.time_stamp + pd.Timedelta(minutes=look_ahead_minutes)
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
            scheduled_requests = self._extract_scheduled_requests(hour=hour,
                                                                minute=minute,
                                                                look_ahead_minutes=look_ahead_minutes,
                                                                traversal_graph_generator=traversal_graph_generator)

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