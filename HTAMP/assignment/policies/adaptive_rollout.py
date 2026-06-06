import copy
from typing import Optional

import pandas as pd
from HTAMP.assignment.assignment_helpers import RequestsDict
from HTAMP.assignment.policies.base_policy import FutureCostEstimation, Helpers
from HTAMP.assignment.policies.basic_helpers import PolicyHelpers
from HTAMP.assignment.policies.prediction_cache import ActivePatientFloorFilter, OfflinePredictionCache
from HTAMP.assignment.policies.rollout_helpers import RolloutHelpers
from HTAMP.data_processing.processing_dataclasses import AnnotatedDataFiles
from HTAMP.environment.robot_dataclasses import RobotProfile
from HTAMP.environment.traversal_graph_gen import TraversalGraphGenerator
from HTAMP.planning.motion_planner import MotionPlanner
from HTAMP.planning.planning_dataclasses import AllTaskProperties, NodeReservationTable, RequestsLists
from HTAMP.planning.request_handler import PlanningRequestHandler
from HTAMP.planning.state import PlanningState
from HTAMP.assignment.policies.weighing_function import PredictionWeightTracker, WeightConfig

class AdaptiveRollout:
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
                 allow_reweighting: bool,
                 prediction_cache_path: Optional[str] = None,
                 prediction_cache_run_names: Optional[str] = None,
                 prediction_match_tolerance_minutes: float = 10.0,
                 prediction_lookahead_minutes: float = 60.0,
                 prediction_cache_patient_filter_mode: str = "active_floor",
                 prediction_weight_bin_minutes: float = 5.0,
                 prediction_weight_alpha_over: Optional[float] = None,
                 prediction_weight_alpha_time: Optional[float] = None,
                 prediction_weight_beta_over: Optional[float] = None,
                 prediction_weight_beta_time: Optional[float] = None,
                 prediction_weight_lambda_min: Optional[float] = None,
                 prediction_weight_cold_start_weight: Optional[float] = None,
                 prediction_weight_fallback_min_positive_prediction_windows: Optional[int] = None,
                 prediction_weight_patient_credibility_prior: Optional[float] = None,
                 include_future_scheduled_requests: bool = False,
                 prediction_rollout_sample_limit: Optional[int] = 20,
                 adaptive_rollout_candidate_limit: Optional[int] = 25,
                 adaptive_rollout_decision_interval_seconds: float = 10.0):
        self.unassigned_requests_dict = RequestsDict()
        self.dummy_delivery_robot_profile = RobotProfile(radius=0.10, speed=0.20, robot_id=-1, robot_type="delivery")
        self.assigned_requests:  dict[int, list[str]]  = {}
        self.blocked_robots = set()
        self.robots_to_be_sent_to_depot = set()
        self.robots_servicing_pred_requests = set()
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
        self.prediction_cache = (
            OfflinePredictionCache(
                csv_path=prediction_cache_path,
                date_stamp=date_stamp,
                floor_number=floor_number,
                selected_run_names=prediction_cache_run_names,
            )
            if prediction_cache_path
            else None
        )
        self.prediction_match_tolerance_minutes = prediction_match_tolerance_minutes
        self.prediction_lookahead_minutes = prediction_lookahead_minutes
        self.prediction_cache_patient_filter_mode = prediction_cache_patient_filter_mode
        self.include_future_scheduled_requests = include_future_scheduled_requests
        self.prediction_rollout_sample_limit = (
            int(prediction_rollout_sample_limit)
            if prediction_rollout_sample_limit is not None and int(prediction_rollout_sample_limit) > 0
            else None
        )
        self.adaptive_rollout_candidate_limit = (
            int(adaptive_rollout_candidate_limit)
            if adaptive_rollout_candidate_limit is not None and int(adaptive_rollout_candidate_limit) > 0
            else None
        )
        self.adaptive_rollout_decision_interval_seconds = max(
            float(adaptive_rollout_decision_interval_seconds),
            0.0,
        )
        self._last_rollout_decision_time: Optional[float] = None
        self._last_available_robots: set[int] = set()
        self._last_pending_request_ids: set[str] = set()
        prediction_weight_config_kwargs = {
            "alpha_over": prediction_weight_alpha_over,
            "alpha_time": prediction_weight_alpha_time,
            "beta_over": prediction_weight_beta_over,
            "beta_time": prediction_weight_beta_time,
            "lambda_min": prediction_weight_lambda_min,
            "cold_start_weight": prediction_weight_cold_start_weight,
            "fallback_min_positive_prediction_windows": prediction_weight_fallback_min_positive_prediction_windows,
            "patient_credibility_prior": prediction_weight_patient_credibility_prior,
        }
        prediction_weight_config = WeightConfig(
            **{
                key: value
                for key, value in prediction_weight_config_kwargs.items()
                if value is not None
            }
        )
        self.prediction_weight_tracker = PredictionWeightTracker(
            bin_minutes=prediction_weight_bin_minutes,
            config=prediction_weight_config,
            default_floor=floor_number,
            default_day=pd.Timestamp(date_stamp).date().isoformat(),
        )
        self.active_patient_filter = (
            ActivePatientFloorFilter(
                floor_number=floor_number,
                annotated_visits_path=annotated_data_files.annotated_visits,
                annotated_admissions_discharges_path=annotated_data_files.annotated_admissions_discharges,
            )
            if prediction_cache_path
            else None
        )

    def _planning_horizon_end_timestamp(self) -> pd.Timestamp:
        return pd.Timestamp(self.date_stamp).normalize() + pd.Timedelta(hours=int(self.end_hour))

    def _extract_assigned_requests_from_state(self, 
                                              state: PlanningState):
        self.assigned_requests = copy.deepcopy(state.assigned_requests)
    
    def _check_for_expired_requests_in_request_dict(self, state: PlanningState, debug: bool=False):
        expired_request_ids = []
        for request_id in self.unassigned_requests_dict.monitoring:
            request = state.requests[request_id]
            if request.time_for_service < state.simulator_time:
                request.mark_rejected()
                expired_request_ids.append(request_id)
        for request_id in self.unassigned_requests_dict.medication:
            request = state.requests[request_id]
            if request.time_for_service < state.simulator_time:
                request.mark_rejected()
                expired_request_ids.append(request_id)

        for request_id in expired_request_ids:
            self.unassigned_requests_dict.remove_request(request_id)
            if debug:
                print(f"Removed expired request {request_id} from unassigned requests.")
    
    def _add_all_real_requests_to_requests_dict(self, 
                                    requests_lists: Optional[RequestsLists],
                                    motion_planner: MotionPlanner,
                                    traversal_graph_generator: TraversalGraphGenerator
                                    ) -> float:
        if requests_lists is None:
            return float('inf')
        
        min_pickup_deadline = float('inf')
        for data_field in requests_lists.__dataclass_fields__.keys():
            requests_list = getattr(requests_lists, data_field)
            for request in requests_list:
                current_pickup_deadline = PolicyHelpers._calculate_pickup_deadline(delivery_robot_profile=self.dummy_delivery_robot_profile,
                                                                                   request=request,
                                                                                   motion_planner=motion_planner,
                                                                                   traversal_graph_generator=traversal_graph_generator)
                if current_pickup_deadline < min_pickup_deadline:
                    min_pickup_deadline = current_pickup_deadline
                self.unassigned_requests_dict.add_request(request)
        
        return min_pickup_deadline

    def _extract_prediction_sample_sets(
        self,
        *,
        state: PlanningState,
        real_requests_lists: Optional[RequestsLists],
        traversal_graph_generator: TraversalGraphGenerator,
        remove_real_matches: bool,
        debug: bool,
    ) -> list[dict[float, RequestsLists]]:
        return RolloutHelpers._extract_prediction_sample_sets(
            prediction_cache=self.prediction_cache,
            state=state,
            real_requests_lists=real_requests_lists,
            initial_time=self.initial_time,
            all_task_properties=self.all_task_properties,
            traversal_graph_generator=traversal_graph_generator,
            prediction_lookahead_minutes=self.prediction_lookahead_minutes,
            prediction_match_tolerance_minutes=self.prediction_match_tolerance_minutes,
            planning_horizon_end_timestamp=self._planning_horizon_end_timestamp(),
            prediction_cache_patient_filter_mode=self.prediction_cache_patient_filter_mode,
            active_patient_filter=self.active_patient_filter,
            remove_real_matches=remove_real_matches,
            prediction_rollout_sample_limit=self.prediction_rollout_sample_limit,
            debug=debug,
        )

    def _update_prediction_weights(
        self,
        *,
        state: PlanningState,
        requests_lists: Optional[RequestsLists],
        traversal_graph_generator: TraversalGraphGenerator,
        debug: bool,
    ) -> None:
        if not self.allow_reweighting:
            return
        if self.prediction_cache is None or not self.prediction_cache.enabled:
            return

        self.prediction_weight_tracker.record_observed_requests(requests_lists)

        if self.prediction_weight_tracker.should_record_prediction_snapshot(state.simulator_time):
            raw_prediction_sample_sets = self._extract_prediction_sample_sets(
                state=state,
                real_requests_lists=requests_lists,
                traversal_graph_generator=traversal_graph_generator,
                remove_real_matches=False,
                debug=debug,
            )
            self.prediction_weight_tracker.record_prediction_snapshot(
                current_time=state.simulator_time,
                prediction_sample_sets=raw_prediction_sample_sets,
            )

        changed_keys = self.prediction_weight_tracker.update_due_weights(state.simulator_time)
        if changed_keys:
            if debug:
                fallback_changed_count = sum(
                    key.patient_id == PredictionWeightTracker.FALLBACK_PATIENT_ID
                    for key in changed_keys
                )
                patient_changed_count = len(changed_keys) - fallback_changed_count
                weights = list(self.prediction_weight_tracker.weights.values())
                if weights:
                    mean_weight = sum(weights) / len(weights)
                    weight_summary = (
                        f"min={min(weights):.3f}, mean={mean_weight:.3f}, "
                        f"max={max(weights):.3f}"
                    )
                else:
                    weight_summary = "no stored weights"
                print(
                    "Updated prediction weights for "
                    f"{patient_changed_count} patient group(s) and "
                    f"{fallback_changed_count} floor/family fallback group(s); "
                    f"{weight_summary}."
                )

    def _prediction_request_weight(self, request) -> float:
        if not self.allow_reweighting:
            return 1.0
        return self.prediction_weight_tracker.weight_for_request(request)

    @staticmethod
    def _count_requests_in_requests_lists(requests_lists: Optional[RequestsLists]) -> int:
        if requests_lists is None:
            return 0
        return sum(
            len(getattr(requests_lists, data_field))
            for data_field in requests_lists.__dataclass_fields__.keys()
        )

    @classmethod
    def _count_requests_in_prediction_dict(cls, prediction_dict: dict[float, RequestsLists]) -> int:
        return sum(
            cls._count_requests_in_requests_lists(requests_lists)
            for requests_lists in prediction_dict.values()
        )

    @classmethod
    def _prediction_sample_summary(
        cls,
        prediction_sample_sets: list[dict[float, RequestsLists]],
        current_time: float,
        look_ahead_minutes: int,
    ) -> tuple[int, float, float]:
        sample_count = len(prediction_sample_sets)
        if sample_count <= 0:
            return 0, 0.0, 0.0

        current_predicted_count = 0
        future_predicted_count = 0
        for prediction_sample_set in prediction_sample_sets:
            current_predictions, future_predictions = RolloutHelpers._split_predicted_requests_dict(
                predicted_requests_dict=prediction_sample_set,
                look_ahead_minutes=look_ahead_minutes,
                current_time=current_time,
            )
            current_predicted_count += cls._count_requests_in_prediction_dict(current_predictions)
            future_predicted_count += cls._count_requests_in_prediction_dict(future_predictions)

        return (
            sample_count,
            current_predicted_count / float(sample_count),
            future_predicted_count / float(sample_count),
        )

    @classmethod
    def _iter_requests_from_requests_lists(cls, requests_lists: Optional[RequestsLists]):
        if requests_lists is None:
            return
        for data_field in requests_lists.__dataclass_fields__.keys():
            yield from getattr(requests_lists, data_field)

    @classmethod
    def _iter_requests_from_prediction_sample_sets(cls, prediction_sample_sets: list[dict[float, RequestsLists]]):
        for prediction_sample_set in prediction_sample_sets:
            for requests_lists in prediction_sample_set.values():
                yield from cls._iter_requests_from_requests_lists(requests_lists)

    def _prediction_weight_summary(self, prediction_sample_sets: list[dict[float, RequestsLists]]) -> str:
        if not self.allow_reweighting:
            return "disabled"

        weights = [
            float(self._prediction_request_weight(request))
            for request in self._iter_requests_from_prediction_sample_sets(prediction_sample_sets)
        ]
        if not weights:
            return "none"

        mean_weight = sum(weights) / float(len(weights))
        return (
            f"n={len(weights)}, min={min(weights):.3f}, "
            f"mean={mean_weight:.3f}, max={max(weights):.3f}"
        )

    def _estimate_candidate_future_costs(
        self,
        *,
        current_state: PlanningState,
        requests_lists: Optional[RequestsLists],
        current_node_reservation_table: Optional[NodeReservationTable],
        prediction_sample_sets: list[dict[float, RequestsLists]],
        split_prediction_sample_sets: list[tuple[dict[float, RequestsLists], dict[float, RequestsLists]]],
        motion_planner: MotionPlanner,
        traversal_graph_generator: TraversalGraphGenerator,
        look_ahead_minutes: int,
        blocked_robots: Optional[set[int]],
        future_scheduled_requests_lists: Optional[RequestsLists],
        best_truncated_cost: float,
        debug: bool,
    ) -> tuple[float, float, Optional[tuple[float, float]]]:
        prediction_request_weight_fn = self._prediction_request_weight if self.allow_reweighting else None
        weighted_raw_cost, weighted_truncated_cost = RolloutHelpers._estimate_average_future_costs_for_prediction_sample_sets(
            cost_estimator=self.cost_estimator,
            current_state=current_state.fork(),
            requests_lists=requests_lists,
            current_node_reservation_table=copy.deepcopy(current_node_reservation_table),
            prediction_sample_sets=prediction_sample_sets,
            motion_planner=motion_planner.fork_with_reservations(),
            traversal_graph_generator=traversal_graph_generator,
            look_ahead_minutes=look_ahead_minutes,
            split_prediction_sample_sets=split_prediction_sample_sets,
            blocked_robots=blocked_robots,
            future_scheduled_requests_lists=future_scheduled_requests_lists,
            prediction_request_weight_fn=prediction_request_weight_fn,
            early_stop_truncated_cost=None if debug else best_truncated_cost,
            debug=False,
        )

        unweighted_costs = None
        if debug and self.allow_reweighting:
            unweighted_costs = RolloutHelpers._estimate_average_future_costs_for_prediction_sample_sets(
                cost_estimator=self.cost_estimator,
                current_state=current_state.fork(),
                requests_lists=requests_lists,
                current_node_reservation_table=copy.deepcopy(current_node_reservation_table),
                prediction_sample_sets=prediction_sample_sets,
                motion_planner=motion_planner.fork_with_reservations(),
                traversal_graph_generator=traversal_graph_generator,
                look_ahead_minutes=look_ahead_minutes,
                split_prediction_sample_sets=split_prediction_sample_sets,
                blocked_robots=blocked_robots,
                future_scheduled_requests_lists=future_scheduled_requests_lists,
                prediction_request_weight_fn=None,
                debug=False,
            )

        return weighted_raw_cost, weighted_truncated_cost, unweighted_costs

    def _print_candidate_decision_diagnostics(
        self,
        *,
        robot_id: int,
        request_id: int | str,
        heuristic_cost: float,
        weighted_raw_cost: float,
        weighted_truncated_cost: float,
        unweighted_costs: Optional[tuple[float, float]],
        prediction_sample_sets: list[dict[float, RequestsLists]],
        current_time: float,
        look_ahead_minutes: int,
        blocked_robots: Optional[set[int]],
    ) -> None:
        action = "wait" if request_id == -1 else f"assign {request_id}"
        sample_count, avg_current_predictions, avg_future_predictions = self._prediction_sample_summary(
            prediction_sample_sets=prediction_sample_sets,
            current_time=current_time,
            look_ahead_minutes=look_ahead_minutes,
        )
        unweighted_summary = ""
        if unweighted_costs is not None:
            unweighted_raw_cost, unweighted_truncated_cost = unweighted_costs
            unweighted_summary = (
                f", unweighted_raw={unweighted_raw_cost:.3f}, "
                f"unweighted_truncated={unweighted_truncated_cost:.3f}, "
                f"weight_delta_truncated={unweighted_truncated_cost - weighted_truncated_cost:.3f}"
            )

        print(
            "Rollout candidate diagnostics: "
            f"robot={robot_id}, action={action}, "
            f"heuristic_cost={heuristic_cost:.3f}, "
            f"weighted_raw={weighted_raw_cost:.3f}, "
            f"weighted_truncated={weighted_truncated_cost:.3f}"
            f"{unweighted_summary}, "
            f"samples={sample_count}, "
            f"avg_current_pred={avg_current_predictions:.2f}, "
            f"avg_future_pred={avg_future_predictions:.2f}, "
            f"prediction_weights={self._prediction_weight_summary(prediction_sample_sets)}, "
            f"blocked_robots={sorted(blocked_robots) if blocked_robots else []}."
        )
    
    def _determine_if_deallocation_should_occur(self, 
                                                min_pickup_deadline: float,
                                                state: PlanningState,
                                                motion_planner: MotionPlanner,
                                                traversal_graph_generator: TraversalGraphGenerator,
                                                debug: bool = False,
                                                ) -> bool:
        if not self.allow_deallocation:
            return False
        
        if min_pickup_deadline == float('inf'):
            return False
        
        for robot_id, assigned_request_ids in self.assigned_requests.items():
            for request_id in assigned_request_ids:
                request = state.requests[request_id]
                if request.started:
                    continue
                current_pickup_deadline = PolicyHelpers._calculate_pickup_deadline(delivery_robot_profile=self.dummy_delivery_robot_profile,
                                                                                   request=request,
                                                                                   motion_planner=motion_planner,
                                                                                   traversal_graph_generator=traversal_graph_generator)
                if current_pickup_deadline > min_pickup_deadline:
                    if debug:
                        print(
                            "New request triggers reassignment: assigned "
                            f"request {request_id} on robot {robot_id} has "
                            f"pickup deadline {current_pickup_deadline:.3f}, "
                            f"which is later than new request deadline "
                            f"{min_pickup_deadline:.3f}."
                        )
                    return True
        
        return False
    
    def _deallocate_requests(self,
                             state: PlanningState,
                             motion_planner: MotionPlanner,
                             traversal_graph_generator: TraversalGraphGenerator,
                             debug: bool):
        for robot_id in state.assigned_requests.keys():
            requests_to_be_removed = []
            for request_id in state.assigned_requests[robot_id]:
                request = state.requests[request_id]
                if request.started:
                    if debug:
                        print(f"Not deallocating request {request_id} from robot {robot_id} because it has already been started.")
                    continue
                request.reset_assignment()
                requests_to_be_removed.append(request_id)
                self.unassigned_requests_dict.add_request(request)
            for request_id in requests_to_be_removed:
                if debug:                    
                    print(f"Deallocating request {request_id} from robot {robot_id} and adding it back to the pool of unassigned requests.")
                self.assigned_requests[robot_id].remove(request_id)
                state.remove_request_from_robot(robot_id=robot_id, request_id=request_id)
            if len(requests_to_be_removed) > 0:
                PolicyHelpers._generate_motion_plan_to_depot(robot_id=robot_id,
                                                             currently_assigned_request_ids=self.assigned_requests[robot_id],
                                                             state=state,
                                                             motion_planner=motion_planner,
                                                             traversal_graph_generator=traversal_graph_generator,
                                                             debug=debug)
    
    def _get_available_robots(self,
                              state: PlanningState,
                              min_pickup_deadline: float,
                              motion_planner: MotionPlanner,
                              traversal_graph_generator: TraversalGraphGenerator,
                              debug: bool) -> set[int]:
        
        deallocate_flag = self._determine_if_deallocation_should_occur(min_pickup_deadline=min_pickup_deadline,
                                                                       state=state,
                                                                       motion_planner=motion_planner,
                                                                       traversal_graph_generator=traversal_graph_generator,
                                                                       debug=debug)
        trigger_reassignment = deallocate_flag

        if min_pickup_deadline != float('inf'):
                self.blocked_robots = set()

        if trigger_reassignment and self.allow_deallocation:
            self._deallocate_requests(state=state,
                                      motion_planner=motion_planner,
                                      traversal_graph_generator=traversal_graph_generator,
                                      debug=debug)
        
        available_robots = set()
        for robot_id in state.assigned_requests.keys():
            if robot_id in self.blocked_robots:
                continue
            if len(state.assigned_requests[robot_id]) == 0:
                available_robots.add(robot_id)
            else:
                last_assigned_request_id = state.assigned_requests[robot_id][-1]
                last_assigned_request = state.requests[last_assigned_request_id]
                planned_time_to_service_request = last_assigned_request.planned_time
                if state.simulator_time >= planned_time_to_service_request - 30.0: 
                    available_robots.add(robot_id)

        return available_robots

    def _pending_request_ids(self) -> set[str]:
        return set(self.unassigned_requests_dict.monitoring) | set(self.unassigned_requests_dict.medication)

    def _reconsider_blocked_robots_if_due(self, state: PlanningState, debug: bool) -> None:
        if not self.blocked_robots or not self._pending_request_ids():
            return

        current_time = float(state.simulator_time)
        retry_due = (
            self._last_rollout_decision_time is None
            or current_time - self._last_rollout_decision_time >= self.adaptive_rollout_decision_interval_seconds
        )
        if not retry_due:
            return

        if debug:
            print(
                "Reconsidering blocked adaptive-rollout robot(s) after "
                f"{self.adaptive_rollout_decision_interval_seconds:.1f}s retry interval."
            )
        self.blocked_robots = set()

    def _should_run_rollout_decision(
        self,
        *,
        state: PlanningState,
        requests_lists: Optional[RequestsLists],
        available_robots: set[int],
        debug: bool,
    ) -> bool:
        pending_request_ids = self._pending_request_ids()
        if not pending_request_ids:
            self._last_available_robots = set(available_robots)
            self._last_pending_request_ids = set()
            return False

        current_time = float(state.simulator_time)
        new_real_requests = RolloutHelpers._requests_lists_has_requests(requests_lists)
        newly_available_robots = set(available_robots) - self._last_available_robots
        pending_requests_changed = pending_request_ids != self._last_pending_request_ids
        retry_due = (
            self._last_rollout_decision_time is None
            or current_time - self._last_rollout_decision_time >= self.adaptive_rollout_decision_interval_seconds
        )
        should_run = bool(
            new_real_requests
            or newly_available_robots
            or pending_requests_changed
            or retry_due
        )

        if debug and not should_run:
            print(
                "Skipping adaptive rollout decision: no new request, "
                "newly available robot, pending-request change, or retry interval."
            )

        self._last_available_robots = set(available_robots)
        self._last_pending_request_ids = set(pending_request_ids)
        if should_run:
            self._last_rollout_decision_time = current_time
        return should_run

    
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
        
        return requests_lists
    
    def _determine_potential_assignments_for_robot(self,
                                                   robot_id: int,
                                                   state: PlanningState,
                                                   motion_planner: MotionPlanner,
                                                   traversal_graph_generator: TraversalGraphGenerator):
        robot_type = state.simulator_config.robot_profiles[robot_id].robot_type
        if robot_type == "delivery":
            potential_request_ids = self.unassigned_requests_dict.medication
        else:
            potential_request_ids = self.unassigned_requests_dict.monitoring

        potential_assignments = []
        for request_id in potential_request_ids:
            request = state.requests[request_id]
            heuristic_cost = RolloutHelpers._estimate_heuristic_cost_to_fulfill_request(assigned_requests=self.assigned_requests,
                                                                                        node_reservation_table=self.node_reservation_table,
                                                                                        robot_id=robot_id,
                                                                                        request_id=request_id,
                                                                                        state=state,
                                                                                        motion_planner=motion_planner,
                                                                                        traversal_graph_generator=traversal_graph_generator)
            if heuristic_cost == float('inf'):
                continue
            estimated_service_end = heuristic_cost + request.scheduled_time
            pickup_deadline = PolicyHelpers._calculate_pickup_deadline(
                delivery_robot_profile=self.dummy_delivery_robot_profile,
                request=request,
                motion_planner=motion_planner,
                traversal_graph_generator=traversal_graph_generator,
            )
            potential_assignments.append((request_id, heuristic_cost, pickup_deadline, estimated_service_end))
        
        potential_assignments.sort(key=lambda x: (x[2], x[3]))
        if self.adaptive_rollout_candidate_limit is not None:
            potential_assignments = potential_assignments[:self.adaptive_rollout_candidate_limit]
        potential_assignments = [(request_id, heuristic_cost) for request_id, heuristic_cost, _, _ in potential_assignments]
        if len(potential_assignments) > 0:
            potential_assignments.append((-1, 0))

        return potential_assignments, state
        

    def _get_assignment_with_minimum_future_costs(self,
                                            robot_id: str,
                                            requests_lists: RequestsLists,
                                            potential_assignments: list[int],
                                            prediction_sample_sets: list[dict[float, RequestsLists]],
                                            split_prediction_sample_sets: list[tuple[dict[float, RequestsLists], dict[float, RequestsLists]]],
                                            future_scheduled_requests_lists: Optional[RequestsLists],
                                            state: PlanningState,
                                            motion_planner: MotionPlanner,
                                            traversal_graph_generator: TraversalGraphGenerator,
                                            hour: int,
                                            minute: int,
                                            look_ahead_minutes: int,
                                            debug: bool):
        min_request_id = None
        min_planned_path = None
        min_planned_goal_indices = None
        min_planned_time_to_reach_last_goal = float('inf')
        min_path_cost = float('inf')
        min_path_raw_cost = float('inf')
        selected_diagnostics = None
        
        
        for i in range(len(potential_assignments)):
            request_id = potential_assignments[i][0]
            heuristic_cost = potential_assignments[i][1]
            current_state = state.fork()
            current_motion_planner = motion_planner.fork_with_reservations()
            currently_assigned_request_ids = copy.deepcopy(self.assigned_requests[robot_id])
            current_node_reservation_table = copy.deepcopy(self.node_reservation_table)
            
            if robot_id in self.robots_servicing_pred_requests:
                current_node_reservation_table.remove_reservations_for_robot(robot_id=robot_id)

            if request_id == -1:
                if debug:
                    print(f"Evaluating the option of not assigning any request to robot {robot_id} at this time step and instead waiting for future requests. Simulating future assignments...")
                current_requests_lists = copy.deepcopy(requests_lists)
                current_blocked_robots = copy.deepcopy(self.blocked_robots)
                current_blocked_robots.add(robot_id)
                unmodified_cost, truncated_cost, unweighted_costs = self._estimate_candidate_future_costs(
                    current_state=current_state,
                    requests_lists=current_requests_lists,
                    current_node_reservation_table=current_node_reservation_table,
                    prediction_sample_sets=prediction_sample_sets,
                    split_prediction_sample_sets=split_prediction_sample_sets,
                    motion_planner=current_motion_planner,
                    traversal_graph_generator=traversal_graph_generator,
                    look_ahead_minutes=look_ahead_minutes,
                    blocked_robots=current_blocked_robots,
                    future_scheduled_requests_lists=future_scheduled_requests_lists,
                    best_truncated_cost=min_path_cost,
                    debug=debug,
                )
                if debug:
                    self._print_candidate_decision_diagnostics(
                        robot_id=robot_id,
                        request_id=request_id,
                        heuristic_cost=heuristic_cost,
                        weighted_raw_cost=unmodified_cost,
                        weighted_truncated_cost=truncated_cost,
                        unweighted_costs=unweighted_costs,
                        prediction_sample_sets=prediction_sample_sets,
                        current_time=state.simulator_time,
                        look_ahead_minutes=look_ahead_minutes,
                        blocked_robots=current_blocked_robots,
                    )

                if truncated_cost < min_path_cost or (truncated_cost == min_path_cost and unmodified_cost < min_path_raw_cost):
                    min_path_cost = truncated_cost
                    min_path_raw_cost = unmodified_cost
                    min_planned_path = None
                    min_planned_goal_indices = None
                    min_planned_time_to_reach_last_goal = None
                    min_request_id = -1
                    selected_diagnostics = (request_id, unmodified_cost, truncated_cost)
            else:
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
                    if planned_time_to_reach_last_goal < state.requests[request_id].ordered_time:
                        if debug:
                            print(f"However, the planned time to reach the last goal {planned_time_to_reach_last_goal} is not greater than the request's ordered time {state.requests[request_id].ordered_time}, so this assignment is not valid.")
                        continue
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
                    current_requests_lists = copy.deepcopy(requests_lists)
                    RolloutHelpers._remove_request_from_requests_lists(request_id=request_id,
                                                                       requests_lists=current_requests_lists)
                    current_blocked_robots = copy.deepcopy(self.blocked_robots)
                    unmodified_cost, truncated_cost, unweighted_costs = self._estimate_candidate_future_costs(
                        current_state=current_state,
                        requests_lists=current_requests_lists,
                        current_node_reservation_table=current_node_reservation_table,
                        prediction_sample_sets=prediction_sample_sets,
                        split_prediction_sample_sets=split_prediction_sample_sets,
                        motion_planner=current_motion_planner,
                        traversal_graph_generator=traversal_graph_generator,
                        look_ahead_minutes=look_ahead_minutes,
                        blocked_robots=current_blocked_robots,
                        future_scheduled_requests_lists=future_scheduled_requests_lists,
                        best_truncated_cost=min_path_cost,
                        debug=debug,
                    )
                    if debug:
                        self._print_candidate_decision_diagnostics(
                            robot_id=robot_id,
                            request_id=request_id,
                            heuristic_cost=heuristic_cost,
                            weighted_raw_cost=unmodified_cost,
                            weighted_truncated_cost=truncated_cost,
                            unweighted_costs=unweighted_costs,
                            prediction_sample_sets=prediction_sample_sets,
                            current_time=state.simulator_time,
                            look_ahead_minutes=look_ahead_minutes,
                            blocked_robots=current_blocked_robots,
                        )

                    if truncated_cost < min_path_cost or (truncated_cost == min_path_cost and unmodified_cost < min_path_raw_cost):
                        min_path_cost = truncated_cost
                        min_path_raw_cost = unmodified_cost
                        min_planned_path = planned_path
                        min_planned_goal_indices = planned_goal_indices
                        min_planned_time_to_reach_last_goal = planned_time_to_reach_last_goal
                        min_request_id = request_id
                        selected_diagnostics = (request_id, unmodified_cost, truncated_cost)

        if debug and selected_diagnostics is not None:
            selected_request_id, selected_raw_cost, selected_truncated_cost = selected_diagnostics
            selected_action = "wait" if selected_request_id == -1 else f"assign {selected_request_id}"
            print(
                "Rollout selected candidate: "
                f"robot={robot_id}, action={selected_action}, "
                f"weighted_raw={selected_raw_cost:.3f}, "
                f"weighted_truncated={selected_truncated_cost:.3f}."
            )

        return min_request_id, min_planned_path, min_planned_goal_indices, min_planned_time_to_reach_last_goal
    
    def _generate_robot_assignment(self,
                                   robot_id: int,
                                   state: PlanningState,
                                   motion_planner: MotionPlanner,
                                   traversal_graph_generator: TraversalGraphGenerator,
                                   prediction_sample_sets: list[dict[float, RequestsLists]],
                                   split_prediction_sample_sets: list[tuple[dict[float, RequestsLists], dict[float, RequestsLists]]],
                                   future_scheduled_requests_lists: Optional[RequestsLists],
                                   hour: int,
                                   minute: int,
                                   look_ahead_minutes: int,
                                   debug: bool,
                                   potential_assignments: Optional[list[tuple[object, float]]] = None):
        
        if potential_assignments is None:
            potential_assignments, new_state = self._determine_potential_assignments_for_robot(robot_id=robot_id,
                                                                                               state=state,
                                                                                               motion_planner=motion_planner,
                                                                                               traversal_graph_generator=traversal_graph_generator)
        else:
            new_state = state
        if len(potential_assignments) == 0:
            if state.simulator_config.robot_profiles[robot_id].robot_type == "delivery":
                if self.unassigned_requests_dict.medication:
                    if debug:
                        print(f"Robot {robot_id} has no valid potential assignments at this time step, but there are unassigned medication requests.")
            else:
                if self.unassigned_requests_dict.monitoring:
                    if debug:
                        print(f"Robot {robot_id} has no valid potential assignments at this time step, but there are unassigned monitoring requests.")
                        
            return None, None, None, None, new_state
        else:
            requests_lists = self._convert_requests_dict_to_requests_lists(state=new_state)
            path_results = self._get_assignment_with_minimum_future_costs(robot_id=robot_id,
                                                                requests_lists=requests_lists,
                                                                potential_assignments=potential_assignments,
                                                                prediction_sample_sets=prediction_sample_sets,
                                                                split_prediction_sample_sets=split_prediction_sample_sets,
                                                                future_scheduled_requests_lists=future_scheduled_requests_lists,
                                                                state=new_state,
                                                                motion_planner=motion_planner,
                                                                traversal_graph_generator=traversal_graph_generator,
                                                                hour=hour,
                                                                minute=minute,
                                                                look_ahead_minutes=look_ahead_minutes,
                                                                debug=debug)
            assigned_request_id, planned_path, planned_goal_indices, planned_time_to_reach_last_goal = path_results
            
            return assigned_request_id, planned_path, planned_goal_indices, planned_time_to_reach_last_goal, new_state

    def _extract_rollout_prediction_context(
        self,
        *,
        state: PlanningState,
        requests_lists: Optional[RequestsLists],
        traversal_graph_generator: TraversalGraphGenerator,
        hour: int,
        minute: int,
        look_ahead_minutes: int,
        debug: bool,
    ) -> tuple[
        list[dict[float, RequestsLists]],
        list[tuple[dict[float, RequestsLists], dict[float, RequestsLists]]],
        Optional[RequestsLists],
    ]:
        future_scheduled_requests_lists = None
        prediction_match_requests_lists = requests_lists
        if self.include_future_scheduled_requests:
            future_scheduled_requests_lists = RolloutHelpers._extract_scheduled_requests(
                date_stamp=self.date_stamp,
                hour=hour,
                minute=minute,
                look_ahead_minutes=look_ahead_minutes,
                end_hour=self.end_hour,
                planning_request_handler=self.planning_request_handler,
                initial_time=self.initial_time,
                all_task_properties=self.all_task_properties,
                traversal_graph_generator=traversal_graph_generator,
            )
            prediction_match_requests_lists = RolloutHelpers._combine_requests_lists(
                requests_lists,
                future_scheduled_requests_lists,
            )

        prediction_sample_sets = self._extract_prediction_sample_sets(
            state=state,
            real_requests_lists=prediction_match_requests_lists,
            traversal_graph_generator=traversal_graph_generator,
            remove_real_matches=True,
            debug=debug,
        )
        split_prediction_sample_sets = RolloutHelpers._split_prediction_sample_sets(
            prediction_sample_sets=prediction_sample_sets,
            look_ahead_minutes=look_ahead_minutes,
            current_time=state.simulator_time,
        )
        return prediction_sample_sets, split_prediction_sample_sets, future_scheduled_requests_lists

    def _assign_requests_to_robots(self,
                                  state: PlanningState,
                                  available_robots: set[int],
                                  motion_planner: MotionPlanner,
                                  traversal_graph_generator: TraversalGraphGenerator,
                                  requests_lists: Optional[RequestsLists],
                                  hour: int,
                                  minute: int,
                                  look_ahead_minutes: int,
                                  debug: bool):
        
        prediction_context = None

        for robot_id in available_robots:
            potential_assignments, _ = self._determine_potential_assignments_for_robot(robot_id=robot_id,
                                                                                       state=state,
                                                                                       motion_planner=motion_planner,
                                                                                       traversal_graph_generator=traversal_graph_generator)
            if potential_assignments:
                if prediction_context is None:
                    prediction_context = self._extract_rollout_prediction_context(
                        state=state,
                        requests_lists=requests_lists,
                        traversal_graph_generator=traversal_graph_generator,
                        hour=hour,
                        minute=minute,
                        look_ahead_minutes=look_ahead_minutes,
                        debug=debug,
                    )
                prediction_sample_sets, split_prediction_sample_sets, future_scheduled_requests_lists = prediction_context
            else:
                prediction_sample_sets = []
                split_prediction_sample_sets = []
                future_scheduled_requests_lists = None

            path_results = self._generate_robot_assignment(robot_id=robot_id,
                                                            state=state,
                                                            motion_planner=motion_planner,
                                                            traversal_graph_generator=traversal_graph_generator,
                                                            prediction_sample_sets=prediction_sample_sets,
                                                            split_prediction_sample_sets=split_prediction_sample_sets,
                                                            future_scheduled_requests_lists=future_scheduled_requests_lists,
                                                            hour=hour,
                                                            minute=minute,
                                                            look_ahead_minutes=look_ahead_minutes,
                                                            debug=debug,
                                                            potential_assignments=potential_assignments)
            assigned_request_id, planned_path, planned_goal_indices, planned_time_to_reach_last_goal, new_state = path_results
            if assigned_request_id is not None:
                if assigned_request_id == -1:
                    self.blocked_robots.add(robot_id)
                    if debug:
                        print(f"Decided not to assign any request to robot {robot_id} at this time step and instead wait for future requests.")
                else:
                    if debug:
                        print(f"1) assigned requests for robot {robot_id}: {self.assigned_requests[robot_id]}")
                        print(f"1) State path for robot {robot_id}: {state.robot_paths[robot_id]}")
                        print(f"1) Planned path for assignment of request {assigned_request_id} to robot {robot_id}: {planned_path}")
                        print(f"1) Planned goal indices for assignment of request {assigned_request_id} to robot {robot_id}: {planned_goal_indices}")
                    
                    if robot_id in self.robots_servicing_pred_requests:
                        self.node_reservation_table.remove_reservations_for_robot(robot_id=robot_id)
                        self.robots_servicing_pred_requests.remove(robot_id)
                    
                    if assigned_request_id in self.unassigned_requests_dict.monitoring or assigned_request_id in self.unassigned_requests_dict.medication:
                        PolicyHelpers._schedule_request(robot_id=robot_id,
                                                        request_id=assigned_request_id,
                                                        currently_assigned_request_ids=self.assigned_requests[robot_id],
                                                        node_reservation_table=self.node_reservation_table,
                                                        planned_path=planned_path,
                                                        planned_goal_indices=planned_goal_indices,
                                                        planned_time_to_reach_last_goal=planned_time_to_reach_last_goal,
                                                        state=state,
                                                        motion_planner=motion_planner,
                                                        traversal_graph_generator=traversal_graph_generator)
                        self.unassigned_requests_dict.remove_request(assigned_request_id)
                        print(f"Assigned request {assigned_request_id} to robot {robot_id}")
                    else:
                        if state.assigned_requests[robot_id]:
                            last_assigned_request_id = state.assigned_requests[robot_id][-1]
                            last_goal_index = state.requests[last_assigned_request_id].planned_goal_indices[-1]
                            combined_path = motion_planner.combine_paths([state.robot_paths[robot_id][:last_goal_index+1], planned_path])
                        else:
                            combined_path = planned_path
                        PolicyHelpers._update_path_and_requests_indices(robot_id=robot_id,
                                                                        planned_path=combined_path,
                                                                        currently_assigned_request_ids=self.assigned_requests[robot_id],
                                                                        state=state,
                                                                        traversal_graph_generator=traversal_graph_generator,
                                                                        debug=debug)
                        motion_planner.clear_reservations_for_agent(robot_profile=state.simulator_config.robot_profiles[robot_id])
                        motion_planner.reserve_path_for_agent(path=state.robot_paths[robot_id],
                                                            robot_profile=state.simulator_config.robot_profiles[robot_id])
                        
                        PolicyHelpers._reserve_nodes_for_request(robot_id=robot_id,
                                        request_id=assigned_request_id,
                                        node_reservation_table=self.node_reservation_table,
                                        state=new_state)
                        
                        self.robots_servicing_pred_requests.add(robot_id)



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
        self._check_for_expired_requests_in_request_dict(state=state)
        min_pickup_deadline = self._add_all_real_requests_to_requests_dict(requests_lists=requests_lists,
                                                                           motion_planner=motion_planner,
                                                                           traversal_graph_generator=traversal_graph_generator)
        self._update_prediction_weights(
            state=state,
            requests_lists=requests_lists,
            traversal_graph_generator=traversal_graph_generator,
            debug=debug,
        )
        self._reconsider_blocked_robots_if_due(state=state, debug=debug)

        available_robots = self._get_available_robots(state=state,
                                                      min_pickup_deadline=min_pickup_deadline,
                                                      motion_planner=motion_planner,
                                                      traversal_graph_generator=traversal_graph_generator,
                                                      debug=debug)
        if not available_robots:
            self._last_available_robots = set()
            return

        if not self._should_run_rollout_decision(
            state=state,
            requests_lists=requests_lists,
            available_robots=available_robots,
            debug=debug,
        ):
            return

        Helpers.extract_node_reservations_from_state(state=state,
                                                     assigned_requests=self.assigned_requests,
                                                     node_reservation_table=self.node_reservation_table)

        # Assignment logic for robots
        self._assign_requests_to_robots(state=state,
                                        available_robots=available_robots,
                                        motion_planner=motion_planner,
                                        traversal_graph_generator=traversal_graph_generator,
                                        requests_lists=requests_lists,
                                        hour=hour,
                                        minute=minute,
                                        look_ahead_minutes=look_ahead_minutes,
                                        debug=debug)
