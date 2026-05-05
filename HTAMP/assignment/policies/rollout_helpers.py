import copy
from typing import Optional, Tuple

import pandas as pd

from HTAMP.assignment.assignment_helpers import AssignmentHelpers
from HTAMP.assignment.policies.base_policy import FutureCostEstimation
from HTAMP.assignment.policies.basic_helpers import PolicyHelpers
from HTAMP.assignment.policies.prediction_cache import OfflinePredictionCache
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
                                       look_ahead_minutes: int,
                                       current_time: float = 0.0) -> Tuple[dict[float, RequestsLists], dict[float, RequestsLists]]:
        current_predicted_requests_dict = {}
        future_predicted_requests_dict = {}
        cutoff_time = float(current_time) + float(look_ahead_minutes) * 60.0
        for request_time, predicted_requests_lists in predicted_requests_dict.items():
            if float(request_time) <= cutoff_time:
                current_predicted_requests_dict[request_time] = predicted_requests_lists
            else:
                future_predicted_requests_dict[request_time] = predicted_requests_lists
        return current_predicted_requests_dict, future_predicted_requests_dict

    @staticmethod
    def _extract_prediction_sample_sets(
        prediction_cache: Optional[OfflinePredictionCache],
        state: PlanningState,
        real_requests_lists: Optional[RequestsLists],
        initial_time: pd.Timestamp,
        all_task_properties: AllTaskProperties,
        traversal_graph_generator: TraversalGraphGenerator,
        prediction_lookahead_minutes: float,
        prediction_match_tolerance_minutes: float,
        planning_horizon_end_timestamp: pd.Timestamp | None = None,
        debug: bool = False,
    ) -> list[dict[float, RequestsLists]]:
        if prediction_cache is None or not prediction_cache.enabled:
            return []
        sample_sets = prediction_cache.prediction_sample_sets(
            state=state,
            real_requests_lists=real_requests_lists,
            initial_time=initial_time,
            all_task_properties=all_task_properties,
            traversal_graph_generator=traversal_graph_generator,
            lookahead_minutes=prediction_lookahead_minutes,
            match_tolerance_minutes=prediction_match_tolerance_minutes,
            planning_horizon_end_timestamp=planning_horizon_end_timestamp,
        )
        if debug:
            print(f"Extracted {len(sample_sets)} prediction sample request set(s).")
        return sample_sets
    
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
    def _estimate_heuristic_cost_to_fulfill_request(assigned_requests: dict[int, list[str]],
                                                    node_reservation_table: NodeReservationTable,
                                                     robot_id: int,
                                                     request_id: str,
                                                     state: PlanningState,
                                                     motion_planner: MotionPlanner,
                                                     traversal_graph_generator: TraversalGraphGenerator) -> float:
        request_struct = state.requests[request_id]
        if assigned_requests[robot_id]:
            last_assigned_request_id = assigned_requests[robot_id][-1]
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
                                                                node_reservation_table=node_reservation_table,
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
                                                                       future_predicted_requests_dict: dict[float, RequestsLists],
                                                                       motion_planner: MotionPlanner,
                                                                       traversal_graph_generator: TraversalGraphGenerator,
                                                                       blocked_robots: Optional[set[int]] = None,
                                                                       debug: bool=False) -> Tuple[float, float]:
        
        if debug:
            print(f"2) Estimating costs for current and future scheduled and predicted requests...")
            print(f"2) Current node reservation table: {current_node_reservation_table} at the beginning of cost estimation")
        # Cost Estimation for current requests
        simulated_state = cost_estimator.assign_requests_to_robots(state=current_state,
                                                                   simulated_state=None,
                                                                   node_reservation_table=current_node_reservation_table,
                                                                   requests_lists=requests_lists,
                                                                   motion_planner=motion_planner,
                                                                   traversal_graph_generator=traversal_graph_generator,
                                                                   add_requests_in_request_lists=True,
                                                                   debug=debug,
                                                                   blocked_robots=blocked_robots)
        
        if debug:
            print(f"2) Node reservation table after assigning current scheduled requests: {current_node_reservation_table}")
        
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
            print(f"2) Node reservation table after assigning current predicted requests: {current_node_reservation_table}")
    
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
            print(f"2) Node reservation table after assigning future predicted requests: {current_node_reservation_table}")
        
        unmodified_cost, truncated_cost = RolloutHelpers._extract_cost_for_assigned_requests(state=simulated_state,
                                                                                             rejection_penalty=current_state.simulator_config.rejection_penalty)
        
        return unmodified_cost, truncated_cost

    @staticmethod
    def _estimate_average_future_costs_for_prediction_sample_sets(
        cost_estimator: FutureCostEstimation,
        current_state: PlanningState,
        requests_lists: Optional[RequestsLists],
        current_node_reservation_table: Optional[NodeReservationTable],
        prediction_sample_sets: list[dict[float, RequestsLists]],
        motion_planner: MotionPlanner,
        traversal_graph_generator: TraversalGraphGenerator,
        look_ahead_minutes: int,
        blocked_robots: Optional[set[int]] = None,
        debug: bool = False,
    ) -> Tuple[float, float]:
        if not prediction_sample_sets:
            cost_estimator.reset()
            return RolloutHelpers._estimate_future_costs_for_scheduled_and_predicted_assignments(
                cost_estimator=cost_estimator,
                current_state=current_state,
                requests_lists=requests_lists,
                current_node_reservation_table=current_node_reservation_table,
                current_predicted_requests_dict={},
                future_predicted_requests_dict={},
                motion_planner=motion_planner,
                traversal_graph_generator=traversal_graph_generator,
                blocked_robots=blocked_robots,
                debug=debug,
            )

        raw_cost_sum = 0.0
        truncated_cost_sum = 0.0
        for prediction_sample_set in prediction_sample_sets:
            sample_state = current_state.fork()
            sample_motion_planner = motion_planner.fork_with_reservations()
            sample_node_reservation_table = copy.deepcopy(current_node_reservation_table)
            current_predicted_requests_dict, future_predicted_requests_dict = (
                RolloutHelpers._split_predicted_requests_dict(
                    predicted_requests_dict=copy.deepcopy(prediction_sample_set),
                    look_ahead_minutes=look_ahead_minutes,
                    current_time=current_state.simulator_time,
                )
            )
            cost_estimator.reset()
            unmodified_cost, truncated_cost = RolloutHelpers._estimate_future_costs_for_scheduled_and_predicted_assignments(
                cost_estimator=cost_estimator,
                current_state=sample_state,
                requests_lists=copy.deepcopy(requests_lists),
                current_node_reservation_table=sample_node_reservation_table,
                current_predicted_requests_dict=current_predicted_requests_dict,
                future_predicted_requests_dict=future_predicted_requests_dict,
                motion_planner=sample_motion_planner,
                traversal_graph_generator=traversal_graph_generator,
                blocked_robots=copy.deepcopy(blocked_robots),
                debug=debug,
            )
            raw_cost_sum += unmodified_cost
            truncated_cost_sum += truncated_cost

        sample_count = float(len(prediction_sample_sets))
        return raw_cost_sum / sample_count, truncated_cost_sum / sample_count
        
