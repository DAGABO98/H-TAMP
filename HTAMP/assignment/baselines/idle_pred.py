import heapq
from typing import Optional
from HTAMP.assignment.assignment_helpers import AssignmentHelpers, TaskQueue
from HTAMP.environment.loc_dataclasses import TimeInterval
from HTAMP.environment.traversal_dataclasses import TraversalNode
from HTAMP.environment.traversal_graph_gen import TraversalGraphGenerator
from HTAMP.planning.motion_planner import MotionPlanner
from HTAMP.planning.planning_dataclasses import RequestsLists, TaskRequest
from HTAMP.planning.state import PlanningState
from HTAMP.plotting.motion_planning_plotting import MotionPlanningPlotter


class IdleTaskPrediction:
    
    def __init__(self):
        self.monitoring_requests_queue = TaskQueue()
        self.delivery_requests_queue = TaskQueue()
        self.prediction_tasks: list[TaskRequest] = []
        self.prediction_file_path: Optional[str] = None
    
    def _generate_prediction_tasks(self,
                                   state: PlanningState,
                                   motion_planner: MotionPlanner,
                                   traversal_graph_generator: TraversalGraphGenerator):
        self.prediction_tasks = []
        # TODO extract tasks from requests in stored prediction file

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
    
    def _select_task_for_idle_robot(self,
                                   robot_id: int,
                                   state: PlanningState,
                                   motion_planner: MotionPlanner,
                                   traversal_graph_generator: TraversalGraphGenerator,
                                   debug: bool) -> Optional[TaskRequest]:
        shortest_time = float('inf')
        selected_task = None
        for prediction_task in self.prediction_tasks:
            trip_time = AssignmentHelpers.heuristic_cost_from_robot_to_request(current_request=prediction_task,
                                                       robot_id=robot_id,
                                                       state=state,
                                                       motion_planner=motion_planner,
                                                       traversal_graph_generator=traversal_graph_generator)
            if trip_time < shortest_time:
                shortest_time = trip_time
                selected_task = prediction_task
        return selected_task
    
    def _determine_path_for_predicted_task(self,
                                           current_request: TaskRequest,
                                           robot_id: int,
                                           state: PlanningState,
                                           motion_planner: MotionPlanner,
                                           traversal_graph_generator: TraversalGraphGenerator) \
                                             -> tuple[list[tuple[TraversalNode, TimeInterval]], list[int], float]:
        start_node = AssignmentHelpers.determine_robot_locations(robot_id, state)
        sub_paths: list[list[tuple[TraversalNode, TimeInterval]]] = []
        planned_goal_indices: list[int] = []
        planned_time_to_service_request: float = float('inf')
        for j, goal_node_label in enumerate(current_request.goal_nodes):
            goal_node = traversal_graph_generator.traversal_graph.nodes_dict[goal_node_label]
            current_time = state.robots_current_time[robot_id] if not sub_paths else sub_paths[-1][-1][1].end
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
                planned_goal_indices.append(planned_goal_indices[-1] + len(sub_path) - 1)
            else:
                planned_goal_indices.append(len(sub_path) - 1)
            start_node = goal_node
        
        if sub_paths:
            return_path = motion_planner.obtain_path_for_agent(start_traversal_node=sub_paths[-1][-1][0],
                                                       goal_traversal_node=state.robot_depots[robot_id],
                                                       robot_profile=state.simulator_config.robot_profiles[robot_id],
                                                       current_time=sub_paths[-1][-1][1].end,
                                                       wait_time_at_goal=state.simulator_config.horizon,
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
        selected_task = self._select_task_for_idle_robot(robot_id=robot_id,
                                                         state=state,
                                                         motion_planner=motion_planner,
                                                         traversal_graph_generator=traversal_graph_generator,
                                                         debug=debug)
        if selected_task is not None:
            planned_path, planned_goal_indices, planned_time = self._determine_path_for_predicted_task(current_request=selected_task,
                                                                                             robot_id=robot_id,
                                                                                             state=state,
                                                                                             motion_planner=motion_planner,
                                                                                             traversal_graph_generator=traversal_graph_generator)
            if len(planned_path) > 0:
                motion_planner.clear_reservations_for_agent(robot_profile=state.simulator_config.robot_profiles[robot_id])
                motion_planner.reserve_path_for_agent(path=planned_path,
                                                        robot_profile=state.simulator_config.robot_profiles[robot_id])
                
                state.assign_robot_path(robot_id=robot_id,
                                        path=planned_path,
                                        traversal_graph=traversal_graph_generator.traversal_graph)
                self.prediction_tasks.remove(selected_task)
            
    
    def _assign_requests_to_available_robots(self,
                                            state: PlanningState,
                                            motion_planner: MotionPlanner,
                                            traversal_graph_generator: TraversalGraphGenerator,
                                            robot_type: str,
                                            requests_queue: TaskQueue,
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
            closest_robot_results = AssignmentHelpers.determine_closest_robot_to_request(request_id=next_request_id, 
                                                                    available_robots=available_robots, 
                                                                  state=state, 
                                                                  motion_planner=motion_planner,
                                                                  traversal_graph_generator=traversal_graph_generator)
            
            closest_robot, closest_path, closest_planned_goal_indices, shortest_time = closest_robot_results

            print(f"Closest robot: {closest_robot}, Shortest time: {shortest_time}")
            
            if closest_robot is not None:
                AssignmentHelpers.assign_request_to_robot(state=state,
                                              request_id=next_request_id,
                                              robot_id=closest_robot,
                                              planned_path=closest_path,
                                              planned_time=shortest_time,
                                              planned_goal_indices=closest_planned_goal_indices,
                                              motion_planner=motion_planner,
                                              traversal_graph_generator=traversal_graph_generator,
                                              debug=debug)
                available_robots.remove(closest_robot)
            else:
                # No feasible robot found for this request, re-add it to the queue
                request = state.requests[next_request_id]
                requests_to_add_back.append(request)

        for request in requests_to_add_back:
            AssignmentHelpers.add_request_to_queue(request, requests_queue)
        
        if available_robots:
            print(f"Idle robots for {robot_type}: {available_robots}")
            self._generate_prediction_tasks(state=state,
                                            motion_planner=motion_planner,
                                            traversal_graph_generator=traversal_graph_generator)
        
        while available_robots:
            # handle idle robots if needed
            idle_robot_id = available_robots.pop()
            self._handle_idle_robot(robot_id=idle_robot_id,
                                    state=state,
                                    motion_planner=motion_planner,
                                    traversal_graph=traversal_graph_generator.traversal_graph,
                                    debug=debug)
    
    def _assign_requests_for_monitoring_robots(self, 
                                          state: PlanningState, 
                                          motion_planner: MotionPlanner, 
                                          traversal_graph_generator: TraversalGraphGenerator,
                                          debug: bool):
        self._assign_requests_to_available_robots(state=state,
                                                 motion_planner=motion_planner,
                                                 traversal_graph_generator=traversal_graph_generator,
                                                 robot_type="monitoring",
                                                 requests_queue=self.monitoring_requests_queue,
                                                 debug=debug)
    
    def _assign_requests_for_delivery_robots(self, 
                                          state: PlanningState, 
                                          motion_planner: MotionPlanner, 
                                          traversal_graph_generator: TraversalGraphGenerator,
                                          debug: bool):
        self._assign_requests_to_available_robots(state=state,
                                                 motion_planner=motion_planner,
                                                 traversal_graph_generator=traversal_graph_generator,
                                                 robot_type="delivery",
                                                 requests_queue=self.delivery_requests_queue,
                                                 debug=debug)

    def assign_requests_to_robots(self, 
                                  state: PlanningState, 
                                  requests_lists: Optional[RequestsLists], 
                                  motion_planner: MotionPlanner,
                                  traversal_graph_generator: TraversalGraphGenerator,
                                  debug: bool):
        # Add new requests to the appropriate queues
        self._check_if_requests_in_queues_expired(state)
        self._add_all_requests_to_queues(requests_lists)

        # Assignment logic for monitoring robots
        self._assign_requests_for_monitoring_robots(state, 
                                                   motion_planner, 
                                                   traversal_graph_generator,
                                                   debug=debug)
        
        # Assignment logic for delivery robots
        self._assign_requests_for_delivery_robots(state, 
                                                 motion_planner, 
                                                 traversal_graph_generator,
                                                 debug=debug)