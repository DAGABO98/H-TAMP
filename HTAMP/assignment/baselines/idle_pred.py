from typing import Optional
from HTAMP.assignment.assignment_helpers import AssignmentHelpers, TaskQueue
from HTAMP.assignment.policies.prediction_cache import ActivePatientFloorFilter, OfflinePredictionCache
from HTAMP.data_processing.processing_dataclasses import AnnotatedDataFiles
from HTAMP.environment.loc_dataclasses import TimeInterval
from HTAMP.environment.traversal_dataclasses import TraversalNode
from HTAMP.environment.traversal_graph_gen import TraversalGraphGenerator
from HTAMP.planning.motion_planner import MotionPlanner
from HTAMP.planning.planning_dataclasses import AllTaskProperties, RequestsLists, TaskRequest
from HTAMP.planning.state import PlanningState
import pandas as pd


class IdleTaskPrediction:
    
    def __init__(self,
                 date_stamp: Optional[pd.Timestamp] = None,
                 end_hour: Optional[int] = None,
                 floor_number: Optional[int] = None,
                 initial_time: Optional[pd.Timestamp] = None,
                 all_task_properties: Optional[AllTaskProperties] = None,
                 annotated_data_files: Optional[AnnotatedDataFiles] = None,
                 prediction_cache_path: Optional[str] = None,
                 prediction_cache_run_names: Optional[str] = None,
                 prediction_match_tolerance_minutes: float = 10.0,
                 prediction_lookahead_minutes: float = 60.0):
        self.monitoring_requests_queue = TaskQueue()
        self.delivery_requests_queue = TaskQueue()
        self.prediction_tasks: list[TaskRequest] = []
        self.consumed_prediction_request_ids: set[str] = set()
        self.robots_servicing_pred_requests: dict[int, TaskRequest] = {}
        self.prediction_file_path: Optional[str] = None
        self.date_stamp = date_stamp
        self.end_hour = end_hour
        self.floor_number = floor_number
        self.initial_time = initial_time
        self.all_task_properties = all_task_properties
        self.prediction_match_tolerance_minutes = prediction_match_tolerance_minutes
        self.prediction_lookahead_minutes = prediction_lookahead_minutes
        self.prediction_cache = (
            OfflinePredictionCache(
                csv_path=prediction_cache_path,
                date_stamp=date_stamp,
                floor_number=int(floor_number),
                selected_run_names=prediction_cache_run_names,
            )
            if prediction_cache_path and date_stamp is not None and floor_number is not None
            else None
        )
        self.active_patient_filter = (
            ActivePatientFloorFilter(
                floor_number=int(floor_number),
                annotated_visits_path=getattr(annotated_data_files, "annotated_visits", None),
                annotated_admissions_discharges_path=getattr(
                    annotated_data_files,
                    "annotated_admissions_discharges",
                    None,
                ),
            )
            if floor_number is not None
            else None
        )

    def _planning_horizon_end_timestamp(self) -> pd.Timestamp | None:
        if self.date_stamp is None or self.end_hour is None:
            return None
        return pd.Timestamp(self.date_stamp).normalize() + pd.Timedelta(hours=int(self.end_hour))

    def _current_timestamp(self, state: PlanningState) -> pd.Timestamp:
        return pd.Timestamp(self.initial_time) + pd.Timedelta(seconds=float(state.simulator_time))

    def _prediction_cache_ready(self) -> bool:
        return (
            self.prediction_cache is not None
            and self.prediction_cache.enabled
            and self.initial_time is not None
            and self.all_task_properties is not None
        )

    def _unavailable_prediction_request_ids(self) -> set[str]:
        active_prediction_ids = {
            prediction_task.request_id
            for prediction_task in self.robots_servicing_pred_requests.values()
        }
        return self.consumed_prediction_request_ids | active_prediction_ids

    def _prediction_route_finished(self, state: PlanningState, robot_id: int) -> bool:
        path = state.robot_paths.get(robot_id, [])
        if not path:
            return True
        return (
            state.robots_next_nodes.get(robot_id) is None
            and float(state.current_wait_times.get(robot_id, 0.0)) <= 0.0
            and int(state.robots_current_node_index.get(robot_id, 0)) >= len(path) - 1
        )

    def _prediction_service_complete(self, state: PlanningState, prediction_task: TaskRequest) -> bool:
        planned_time = float(getattr(prediction_task, "planned_time", -1.0))
        return planned_time >= 0.0 and planned_time <= float(state.simulator_time)

    def _refresh_prediction_assignments(
        self,
        *,
        state: PlanningState,
        motion_planner: MotionPlanner,
        debug: bool,
    ) -> None:
        for robot_id, prediction_task in list(self.robots_servicing_pred_requests.items()):
            if state.assigned_requests.get(robot_id):
                self.robots_servicing_pred_requests.pop(robot_id, None)
                continue
            if self._prediction_service_complete(
                state=state,
                prediction_task=prediction_task,
            ) or self._prediction_route_finished(state=state, robot_id=robot_id):
                self._release_prediction_assignment(
                    robot_id=robot_id,
                    state=state,
                    motion_planner=motion_planner,
                    consumed=True,
                    clear_reservations=True,
                    debug=debug,
                )

    def _release_prediction_assignment(
        self,
        *,
        robot_id: int,
        state: PlanningState,
        motion_planner: MotionPlanner,
        consumed: bool,
        clear_reservations: bool,
        debug: bool,
    ) -> None:
        prediction_task = self.robots_servicing_pred_requests.pop(robot_id, None)
        if prediction_task is None:
            return
        if consumed:
            self.consumed_prediction_request_ids.add(prediction_task.request_id)
        if clear_reservations:
            motion_planner.clear_reservations_for_agent(
                robot_profile=state.simulator_config.robot_profiles[robot_id]
            )
        if debug:
            status = "completed" if consumed else "preempted"
            print(
                f"Robot {robot_id} {status} predicted request "
                f"{prediction_task.request_id}."
            )

    def _release_all_prediction_assignments(
        self,
        *,
        state: PlanningState,
        motion_planner: MotionPlanner,
        consumed: bool,
        clear_reservations: bool,
        debug: bool,
    ) -> None:
        for robot_id in list(self.robots_servicing_pred_requests):
            self._release_prediction_assignment(
                robot_id=robot_id,
                state=state,
                motion_planner=motion_planner,
                consumed=consumed,
                clear_reservations=clear_reservations,
                debug=debug,
            )

    def _motion_planner_without_prediction_reservations(
        self,
        *,
        state: PlanningState,
        motion_planner: MotionPlanner,
    ) -> MotionPlanner:
        if not self.robots_servicing_pred_requests:
            return motion_planner
        candidate_motion_planner = motion_planner.fork_with_reservations()
        for robot_id in self.robots_servicing_pred_requests:
            candidate_motion_planner.clear_reservations_for_agent(
                robot_profile=state.simulator_config.robot_profiles[robot_id]
            )
        return candidate_motion_planner

    def _flatten_prediction_sample(
        self,
        sample_bucket: dict[float, RequestsLists],
    ) -> list[TaskRequest]:
        prediction_tasks: list[TaskRequest] = []
        for request_time in sorted(sample_bucket):
            requests_lists = sample_bucket[request_time]
            unavailable_prediction_request_ids = self._unavailable_prediction_request_ids()
            prediction_tasks.extend(
                request
                for request in (
                    requests_lists.blood_pressure_requests
                    + requests_lists.heart_rate_requests
                    + requests_lists.respiratory_rate_requests
                    + requests_lists.temperature_requests
                    + requests_lists.oxygen_saturation_requests
                    + requests_lists.medications_requests
                )
                if request.request_id not in unavailable_prediction_request_ids
            )
        prediction_tasks.sort(key=lambda request: float(request.scheduled_time))
        return prediction_tasks
    
    def _generate_prediction_tasks(self,
                                   state: PlanningState,
                                   requests_lists: Optional[RequestsLists],
                                   motion_planner: MotionPlanner,
                                   traversal_graph_generator: TraversalGraphGenerator):
        self.prediction_tasks = []
        if not self._prediction_cache_ready():
            return
        active_patient_keys = None
        if self.active_patient_filter is not None:
            active_patient_keys = self.active_patient_filter.active_patient_keys(
                timestamp=self._current_timestamp(state),
            )
            if active_patient_keys is not None and not active_patient_keys:
                return
        sample_sets = self.prediction_cache.prediction_sample_sets(
            state=state,
            real_requests_lists=requests_lists,
            initial_time=pd.Timestamp(self.initial_time),
            all_task_properties=self.all_task_properties,
            traversal_graph_generator=traversal_graph_generator,
            lookahead_minutes=self.prediction_lookahead_minutes,
            match_tolerance_minutes=self.prediction_match_tolerance_minutes,
            planning_horizon_end_timestamp=self._planning_horizon_end_timestamp(),
            filter_to_observed_patients=False,
            patient_key_filter=active_patient_keys,
        )
        if not sample_sets:
            return
        self.prediction_tasks = self._flatten_prediction_sample(sample_sets[0])

    def _check_if_requests_in_queues_expired(self, state: PlanningState):
        AssignmentHelpers.remove_expired_requests_from_queue(queue=self.monitoring_requests_queue, 
                                                 state=state)
        AssignmentHelpers.remove_expired_requests_from_queue(queue=self.delivery_requests_queue, 
                                                 state=state)
    
    def _add_all_requests_to_queues(self, requests_lists: Optional[RequestsLists]):
        if requests_lists is None:
            return
        for request in requests_lists.blood_pressure_requests + requests_lists.heart_rate_requests + \
                    requests_lists.respiratory_rate_requests + requests_lists.temperature_requests + \
                    requests_lists.oxygen_saturation_requests:
            AssignmentHelpers.add_request_to_queue(request, self.monitoring_requests_queue)
        for request in requests_lists.medications_requests:
            AssignmentHelpers.add_request_to_queue(request, self.delivery_requests_queue)
    
    def _prediction_task_matches_robot_type(self, prediction_task: TaskRequest, robot_type: str) -> bool:
        if robot_type == "delivery":
            return prediction_task.request_type == "medication"
        return prediction_task.request_type != "medication"
    
    def _determine_path_for_predicted_task(self,
                                           current_request: TaskRequest,
                                           robot_id: int,
                                           state: PlanningState,
                                           motion_planner: MotionPlanner,
                                           traversal_graph_generator: TraversalGraphGenerator) \
                                             -> tuple[list[tuple[TraversalNode, TimeInterval]], list[int], float]:
        start_node, initial_time = AssignmentHelpers.determine_robot_nodes_and_times(robot_id=robot_id, 
                                                                                     state=state)
        sub_paths: list[list[tuple[TraversalNode, TimeInterval]]] = []
        planned_goal_indices: list[int] = []
        planned_time_to_service_request: float = float('inf')
        current_time = initial_time
        for j, goal_node_label in enumerate(current_request.goal_nodes):
            goal_node = traversal_graph_generator.traversal_graph.nodes_dict[goal_node_label]
            sub_path = motion_planner.obtain_path_for_agent(start_traversal_node=start_node,
                                                    goal_traversal_node=goal_node,
                                                    robot_profile=state.simulator_config.robot_profiles[robot_id],
                                                    current_time=current_time,
                                                    wait_time_at_goal=current_request.wait_times_at_goals_seconds[j],
                                                    horizon=state.simulator_config.horizon)
            if sub_path is None:
                sub_paths = []
                planned_goal_indices = []
                planned_time_to_service_request = float('inf')
                break
            sub_paths.append(sub_path)
            if planned_goal_indices:
                planned_goal_indices.append(planned_goal_indices[-1] + len(sub_path))
            else:
                planned_goal_indices.append(len(sub_path) - 1)
            start_node = goal_node
            current_time = sub_path[-1][1].end
        if sub_paths:
            wait_time_at_goal = state.simulator_config.horizon - sub_paths[-1][-1][1].end
            return_path = motion_planner.obtain_path_for_agent(start_traversal_node=sub_paths[-1][-1][0],
                                                       goal_traversal_node=state.robot_depots[robot_id],
                                                       robot_profile=state.simulator_config.robot_profiles[robot_id],
                                                       current_time=sub_paths[-1][-1][1].end,
                                                       wait_time_at_goal=wait_time_at_goal,
                                                       horizon=2*state.simulator_config.horizon)
            if return_path is not None:
                sub_paths.append(return_path)
                planned_time_to_service_request = sub_paths[-2][-1][1].end
                if planned_time_to_service_request > current_request.time_for_service:
                    final_path = []
                    planned_goal_indices = []
                    planned_time_to_service_request = float('inf')
                else:
                    final_path = motion_planner.combine_paths(sub_paths)

            else:
                final_path = []
                planned_goal_indices = []
                planned_time_to_service_request = float('inf')
        else:
            final_path = []
            planned_goal_indices = []
            planned_time_to_service_request = float('inf')
        return final_path, planned_goal_indices, planned_time_to_service_request
    
    def _handle_idle_robot(self,
                           robot_id: int,
                           state: PlanningState,
                           motion_planner: MotionPlanner,
                           traversal_graph_generator: TraversalGraphGenerator,
                           debug: bool):
        robot_type = state.simulator_config.robot_profiles[robot_id].robot_type
        for selected_task in list(self.prediction_tasks):
            if not self._prediction_task_matches_robot_type(
                prediction_task=selected_task,
                robot_type=robot_type,
            ):
                continue
            planned_path, planned_goal_indices, planned_time = self._determine_path_for_predicted_task(current_request=selected_task,
                                                                                             robot_id=robot_id,
                                                                                             state=state,
                                                                                             motion_planner=motion_planner,
                                                                                             traversal_graph_generator=traversal_graph_generator)
            if len(planned_path) > 0:
                selected_task.schedule_task(
                    assigned_robot_id=robot_id,
                    planned_goal_indices=planned_goal_indices,
                    planned_time=planned_time,
                )
                motion_planner.clear_reservations_for_agent(robot_profile=state.simulator_config.robot_profiles[robot_id])
                motion_planner.reserve_path_for_agent(path=planned_path,
                                                        robot_profile=state.simulator_config.robot_profiles[robot_id])
                
                state.assign_robot_path(robot_id=robot_id,
                                        path=planned_path,
                                        traversal_graph=traversal_graph_generator.traversal_graph)
                self.robots_servicing_pred_requests[robot_id] = selected_task
                self.prediction_tasks.remove(selected_task)
                if debug:
                    print(
                        f"Assigned idle {robot_type} robot {robot_id} to predicted "
                        f"request {selected_task.request_id} scheduled at "
                        f"{selected_task.scheduled_time}."
                    )
                return

    def _determine_closest_robot_to_real_request(
        self,
        *,
        request_id: str,
        available_robots: list[int],
        state: PlanningState,
        motion_planner: MotionPlanner,
        traversal_graph_generator: TraversalGraphGenerator,
        debug: bool,
    ) -> tuple[Optional[int], list[tuple[TraversalNode, TimeInterval]], list[int], float]:
        closest_robot = None
        closest_path = []
        closest_planned_goal_indices = []
        shortest_time = float("inf")
        candidate_motion_planner = self._motion_planner_without_prediction_reservations(
            state=state,
            motion_planner=motion_planner,
        )

        for robot_id in available_robots:
            path, planned_goal_indices, planned_time = (
                AssignmentHelpers.determine_path_from_robot_location_to_request(
                    request_id=request_id,
                    robot_id=robot_id,
                    state=state,
                    motion_planner=candidate_motion_planner,
                    traversal_graph_generator=traversal_graph_generator,
                    debug=debug,
                )
            )
            if len(path) > 0 and planned_time < shortest_time:
                shortest_time = planned_time
                closest_robot = robot_id
                closest_path = path
                closest_planned_goal_indices = planned_goal_indices

        return closest_robot, closest_path, closest_planned_goal_indices, shortest_time
            
    
    def _assign_requests_to_available_robots(self,
                                            state: PlanningState,
                                            motion_planner: MotionPlanner,
                                            traversal_graph_generator: TraversalGraphGenerator,
                                            robot_type: str,
                                            requests_queue: TaskQueue,
                                            requests_lists: Optional[RequestsLists],
                                            debug: bool):
        available_robots = state.get_available_robots(robot_type=robot_type)
        requests_to_add_back = []
        print(f"Available robots for {robot_type}: {available_robots}")
        while available_robots:
            if not requests_queue.heap:
                break  # No more requests to assign

            # Get the next request from the highest priority queue
            next_request_id = requests_queue.pop_task()
            print(f"Assigning request {next_request_id} to {robot_type} robot")
            closest_robot_results = self._determine_closest_robot_to_real_request(
                request_id=next_request_id,
                available_robots=available_robots,
                state=state,
                motion_planner=motion_planner,
                traversal_graph_generator=traversal_graph_generator,
                debug=debug,
            )
            
            closest_robot, closest_path, closest_planned_goal_indices, shortest_time = closest_robot_results

            print(f"Closest robot: {closest_robot}, Shortest time: {shortest_time}")
            
            if closest_robot is not None:
                self._release_all_prediction_assignments(
                    state=state,
                    motion_planner=motion_planner,
                    consumed=False,
                    clear_reservations=True,
                    debug=debug,
                )
                AssignmentHelpers.assign_request_to_robot(state=state,
                                              request_id=next_request_id,
                                              robot_id=closest_robot,
                                              planned_path=closest_path,
                                              planned_time=shortest_time,
                                              planned_goal_indices=closest_planned_goal_indices,
                                              motion_planner=motion_planner,
                                              traversal_graph_generator=traversal_graph_generator,
                                              debug=debug,
                                              adjust_goal_indices_for_idle_motion=True)
                available_robots.remove(closest_robot)
            else:
                # No feasible robot found for this request, re-add it to the queue
                request = state.requests[next_request_id]
                requests_to_add_back.append(request)

        for request in requests_to_add_back:
            AssignmentHelpers.add_request_to_queue(request, requests_queue)

        self._assign_predictions_to_idle_robots(
            state=state,
            motion_planner=motion_planner,
            traversal_graph_generator=traversal_graph_generator,
            robot_type=robot_type,
            requests_lists=requests_lists,
            debug=debug,
        )

    def _assign_predictions_to_idle_robots(
        self,
        *,
        state: PlanningState,
        motion_planner: MotionPlanner,
        traversal_graph_generator: TraversalGraphGenerator,
        robot_type: str,
        requests_lists: Optional[RequestsLists],
        debug: bool,
    ) -> None:
        available_robots = state.get_available_robots(robot_type=robot_type)
        prediction_available_robots = [
            robot_id
            for robot_id in available_robots
            if robot_id not in self.robots_servicing_pred_requests
        ]
        if prediction_available_robots:
            print(f"Idle robots for {robot_type}: {prediction_available_robots}")
            self._generate_prediction_tasks(state=state,
                                            requests_lists=requests_lists,
                                            motion_planner=motion_planner,
                                            traversal_graph_generator=traversal_graph_generator)
        
        while prediction_available_robots:
            # handle idle robots if needed
            idle_robot_id = prediction_available_robots.pop()
            self._handle_idle_robot(robot_id=idle_robot_id,
                                    state=state,
                                    motion_planner=motion_planner,
                                    traversal_graph_generator=traversal_graph_generator,
                                    debug=debug)
    
    def _assign_requests_for_monitoring_robots(self, 
                                          state: PlanningState, 
                                          motion_planner: MotionPlanner, 
                                          traversal_graph_generator: TraversalGraphGenerator,
                                          requests_lists: Optional[RequestsLists],
                                          debug: bool):
        self._assign_requests_to_available_robots(state=state,
                                                 motion_planner=motion_planner,
                                                 traversal_graph_generator=traversal_graph_generator,
                                                 robot_type="monitoring",
                                                 requests_queue=self.monitoring_requests_queue,
                                                 requests_lists=requests_lists,
                                                 debug=debug)
    
    def _assign_requests_for_delivery_robots(self, 
                                          state: PlanningState, 
                                          motion_planner: MotionPlanner, 
                                          traversal_graph_generator: TraversalGraphGenerator,
                                          requests_lists: Optional[RequestsLists],
                                          debug: bool):
        self._assign_requests_to_available_robots(state=state,
                                                 motion_planner=motion_planner,
                                                 traversal_graph_generator=traversal_graph_generator,
                                                 robot_type="delivery",
                                                 requests_queue=self.delivery_requests_queue,
                                                 requests_lists=requests_lists,
                                                 debug=debug)

    def assign_requests_to_robots(self, 
                                  state: PlanningState, 
                                  requests_lists: Optional[RequestsLists], 
                                  motion_planner: MotionPlanner,
                                  traversal_graph_generator: TraversalGraphGenerator,
                                  debug: bool):
        # Add new requests to the appropriate queues
        self._refresh_prediction_assignments(
            state=state,
            motion_planner=motion_planner,
            debug=debug,
        )
        self._check_if_requests_in_queues_expired(state)
        self._add_all_requests_to_queues(requests_lists)

        # Assignment logic for monitoring robots
        self._assign_requests_for_monitoring_robots(state, 
                                                   motion_planner, 
                                                   traversal_graph_generator,
                                                   requests_lists=requests_lists,
                                                   debug=debug)
        
        # Assignment logic for delivery robots
        self._assign_requests_for_delivery_robots(state, 
                                                 motion_planner, 
                                                 traversal_graph_generator,
                                                 requests_lists=requests_lists,
                                                 debug=debug)

        self._assign_predictions_to_idle_robots(
            state=state,
            motion_planner=motion_planner,
            traversal_graph_generator=traversal_graph_generator,
            robot_type="monitoring",
            requests_lists=requests_lists,
            debug=debug,
        )
        self._assign_predictions_to_idle_robots(
            state=state,
            motion_planner=motion_planner,
            traversal_graph_generator=traversal_graph_generator,
            robot_type="delivery",
            requests_lists=requests_lists,
            debug=debug,
        )
