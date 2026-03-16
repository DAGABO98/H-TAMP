from typing import Optional, Tuple

import pandas as pd

from HTAMP.assignment.policies.base_policy import FutureCostEstimation
from HTAMP.assignment.policies.sequential_greedy import SequentialGreedy
from HTAMP.environment.traversal_graph_gen import TraversalGraphGenerator
from HTAMP.planning.motion_planner import MotionPlanner
from HTAMP.planning.planning_dataclasses import AllTaskProperties, NodeReservationTable, RequestsLists, TaskRequest, TimeSignal
from HTAMP.planning.request_handler import PlanningRequestHandler
from HTAMP.planning.state import PlanningState, SimulatedState


class RolloutHelpers:

    @staticmethod
    def _remove_request_from_requests_lists(request_id: str, 
                                            requests_lists: Optional[RequestsLists]):
        if requests_lists is not None:
            for data_field in requests_lists.__dataclass_fields__.keys():
                requests_list = getattr(requests_lists, data_field)
                for request in requests_list:
                    if request.request_id == request_id:
                        requests_list.remove(request)
                        break

    @staticmethod
    def _extract_cost_for_assigned_requests(state: PlanningState | SimulatedState, rejection_penalty: float) -> list[float]:
        unmodified_costs = []
        truncated_costs = []
        for request_id in state.requests.keys():
            request_struct = state.requests[request_id]
            if request_struct.planned_time == -1.0:
                if request_struct.rejected:
                    cost = rejection_penalty
                    truncated_cost = rejection_penalty
                else:
                    cost = 0
                    truncated_cost = 0
            else:
                cost = request_struct.planned_time - request_struct.scheduled_time
                truncated_cost = max(cost, 0)

            unmodified_costs.append(cost)
            truncated_costs.append(truncated_cost)
        unmodified_cost = sum(unmodified_costs)
        truncated_cost = sum(truncated_costs)
        return unmodified_cost, truncated_cost
    
    @staticmethod
    def _add_requests_to_state(requests_lists: RequestsLists, 
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
    
    @staticmethod
    def _extract_predicted_requests(state: PlanningState,
                                    hour: int,
                                    minute: int) -> dict[float, RequestsLists]:
        current_time = state.simulator_time
        current_predicted_requests = RequestsLists(blood_pressure_requests=[],
                                                  heart_rate_requests=[],
                                                  respiratory_rate_requests=[],
                                                  temperature_requests=[],
                                                  oxygen_saturation_requests=[],
                                                  medications_requests=[]) # TODO: to be implemented once the prediction model is implemented
        pred_dict = {current_time: current_predicted_requests}
        
        return {}
    
    @staticmethod
    def _split_predicted_requests_dict(predicted_requests_dict: dict[float, RequestsLists],
                                       look_ahead_minutes: int) -> Tuple[dict[float, RequestsLists], dict[float, RequestsLists]]:
        # TODO: Fix this method to do proper division of predicted requests
        current_predicted_requests_dict = {}
        future_predicted_requests_dict = {}
        for time_signal, predicted_requests_lists in predicted_requests_dict.items():
            if time_signal <= look_ahead_minutes:
                current_predicted_requests_dict[time_signal] = predicted_requests_lists
            else:
                future_predicted_requests_dict[time_signal] = predicted_requests_lists
        return current_predicted_requests_dict, future_predicted_requests_dict
    
    @staticmethod
    def _extract_scheduled_requests(date_stamp: pd.Timestamp,
                                    hour: int,
                                    minute: int,
                                    look_ahead_minutes: int,
                                    end_hour: int,
                                    planning_request_handler: PlanningRequestHandler,
                                    initial_time: pd.Timestamp,
                                    all_task_properties:AllTaskProperties,
                                    traversal_graph_generator: TraversalGraphGenerator) -> RequestsLists:
        if hour <= end_hour:
            original_time_signal = TimeSignal(year=date_stamp.year,
                                    month=date_stamp.month,
                                    day=date_stamp.day,
                                    hour=hour,
                                    minute=minute)
            
            shifted_time_stamp = original_time_signal.time_stamp + pd.Timedelta(minutes=look_ahead_minutes)
            if shifted_time_stamp.day != date_stamp.day:
                return None
            else:
                if shifted_time_stamp.hour == end_hour and 60 - shifted_time_stamp.minute < look_ahead_minutes:
                    return None
                else:
                    shifted_time_signal = TimeSignal(year=shifted_time_stamp.year,
                                                 month=shifted_time_stamp.month,
                                                 day=shifted_time_stamp.day,
                                                 hour=shifted_time_stamp.hour,
                                                 minute=shifted_time_stamp.minute)
                    scheduled_requests_lists: RequestsLists = planning_request_handler.get_all_requests_for_time_signal(time_signal=shifted_time_signal,
                                                                                                    initial_time=initial_time,
                                                                                                    all_task_properties=all_task_properties,
                                                                                                    look_ahead_minutes=look_ahead_minutes,
                                                                                                    traversal_graph_generator=traversal_graph_generator)

                return scheduled_requests_lists
        else:
            return None
    
    @staticmethod
    def _simulate_scheduled_and_predicted_assignments(base_policy: SequentialGreedy,
                                                      current_state: PlanningState,
                                                      requests_lists: Optional[RequestsLists],
                                                      predicted_requests_dict: dict[float, RequestsLists],
                                                      motion_planner: MotionPlanner,
                                                      traversal_graph_generator: TraversalGraphGenerator,
                                                      look_ahead_minutes: int,
                                                      fps: int):
        base_policy.assign_requests_to_robots(state=current_state,
                                                requests_lists=requests_lists,
                                                motion_planner=motion_planner,
                                                traversal_graph_generator=traversal_graph_generator,
                                                debug=False)
        
        for i in range(look_ahead_minutes):
            for second in range(60):
                current_time = current_state.simulator_time
                predicted_requests_lists = predicted_requests_dict.get(current_time, None)
                if predicted_requests_lists:
                    RolloutHelpers._add_requests_to_state(requests_lists=predicted_requests_lists, 
                                                         state=current_state)
                    base_policy.assign_requests_to_robots(state=current_state,
                                                            requests_lists=predicted_requests_lists,
                                                            motion_planner=motion_planner,
                                                            traversal_graph_generator=traversal_graph_generator,
                                                            debug=False)
                for frame in range(fps):
                    current_state.step(traversal_graph=traversal_graph_generator.traversal_graph,
                                       planning_flag=True)
    
    @staticmethod
    def _simulate_future_assignments(base_policy: SequentialGreedy,
                                     current_state: PlanningState,
                                     requests_lists: Optional[RequestsLists],
                                     future_scheduled_requests_lists: Optional[RequestsLists],
                                     predicted_requests_dict: dict[float, RequestsLists],
                                     motion_planner: MotionPlanner,
                                     traversal_graph_generator: TraversalGraphGenerator,
                                     look_ahead_minutes: int,
                                     fps: int):
        
        RolloutHelpers._simulate_scheduled_and_predicted_assignments(base_policy=base_policy,
                                                                    current_state=current_state,
                                                                    requests_lists=requests_lists,
                                                                    predicted_requests_dict=predicted_requests_dict,
                                                                    motion_planner=motion_planner,
                                                                    traversal_graph_generator=traversal_graph_generator,
                                                                    look_ahead_minutes=look_ahead_minutes,
                                                                    fps=fps)
        
        RolloutHelpers._simulate_scheduled_and_predicted_assignments(base_policy=base_policy,
                                                                    current_state=current_state,
                                                                    requests_lists=future_scheduled_requests_lists,
                                                                    predicted_requests_dict=predicted_requests_dict,
                                                                    motion_planner=motion_planner,
                                                                    traversal_graph_generator=traversal_graph_generator,
                                                                    look_ahead_minutes=10,
                                                                    fps=fps)
    
    @staticmethod
    def _convert_predicted_requests_dict_into_combined_requests_lists(predicted_requests_dict: dict[float, RequestsLists]) -> RequestsLists:
        combined_requests_lists = RequestsLists(blood_pressure_requests=[],
                                              heart_rate_requests=[],
                                              respiratory_rate_requests=[],
                                              temperature_requests=[],
                                              oxygen_saturation_requests=[],
                                              medications_requests=[])
        for predicted_requests_lists in predicted_requests_dict.values():
            for data_field in combined_requests_lists.__dataclass_fields__.keys():
                combined_list = getattr(combined_requests_lists, data_field)
                predicted_list = getattr(predicted_requests_lists, data_field)
                combined_list.extend(predicted_list)
        return combined_requests_lists

    @staticmethod
    def _estimate_future_costs_for_scheduled_and_predicted_assignments(cost_estimator: FutureCostEstimation,
                                                                       current_state: PlanningState,
                                                                       requests_lists: Optional[RequestsLists],
                                                                       current_node_reservation_table: Optional[NodeReservationTable],
                                                                       current_predicted_requests_dict: dict[float, RequestsLists],
                                                                       future_scheduled_requests_lists: Optional[RequestsLists],
                                                                       future_predicted_requests_dict: dict[float, RequestsLists],
                                                                       motion_planner: MotionPlanner,
                                                                       traversal_graph_generator: TraversalGraphGenerator,
                                                                       debug: bool=False) -> Tuple[float, float]:
        
        if debug:
            print(f"Estimating costs for current and future scheduled and predicted requests...")
            print(f"Current node reservation table: {current_node_reservation_table} at the beginning of cost estimation")
        # Cost Estimation for current requests
        simulated_state = cost_estimator.assign_requests_to_robots(state=current_state,
                                                                   simulated_state=None,
                                                                   node_reservation_table=current_node_reservation_table,
                                                                   requests_lists=requests_lists,
                                                                   motion_planner=motion_planner,
                                                                   traversal_graph_generator=traversal_graph_generator,
                                                                   add_requests_in_request_lists=True,
                                                                   debug=debug)
        
        if debug:
            print(f"Node reservation table after assigning current scheduled requests: {current_node_reservation_table}")
        
        current_combined_requests_lists = RolloutHelpers._convert_predicted_requests_dict_into_combined_requests_lists(predicted_requests_dict=current_predicted_requests_dict)
        cost_estimator.assign_requests_to_robots(state=current_state,
                                                 simulated_state=simulated_state,
                                                 node_reservation_table=current_node_reservation_table,
                                                 requests_lists=current_combined_requests_lists,
                                                 motion_planner=motion_planner,
                                                 traversal_graph_generator=traversal_graph_generator,
                                                 add_requests_in_request_lists=True,
                                                 debug=debug)
        
        if debug:
            print(f"Node reservation table after assigning current predicted requests: {current_node_reservation_table}")
        
        # Cost estimation for future requests
        cost_estimator.assign_requests_to_robots(state=current_state,
                                                 simulated_state=simulated_state,
                                                 node_reservation_table=current_node_reservation_table,
                                                 requests_lists=future_scheduled_requests_lists,
                                                 motion_planner=motion_planner,
                                                 traversal_graph_generator=traversal_graph_generator,
                                                 add_requests_in_request_lists=True,
                                                 debug=debug)
        
        if debug:
            print(f"Node reservation table after assigning future scheduled requests: {current_node_reservation_table}")
    
        future_combined_requests_lists = RolloutHelpers._convert_predicted_requests_dict_into_combined_requests_lists(predicted_requests_dict=future_predicted_requests_dict)
        cost_estimator.assign_requests_to_robots(state=current_state,
                                                 simulated_state=simulated_state,
                                                 node_reservation_table=current_node_reservation_table,
                                                 requests_lists=future_combined_requests_lists,
                                                 motion_planner=motion_planner,
                                                 traversal_graph_generator=traversal_graph_generator,
                                                 add_requests_in_request_lists=True,
                                                 debug=debug)
        if debug:
            print(f"Node reservation table after assigning future predicted requests: {current_node_reservation_table}")
        
        unmodified_cost, truncated_cost = RolloutHelpers._extract_cost_for_assigned_requests(state=simulated_state,
                                                                                             rejection_penalty=current_state.simulator_config.rejection_penalty)
        
        return unmodified_cost, truncated_cost
        
