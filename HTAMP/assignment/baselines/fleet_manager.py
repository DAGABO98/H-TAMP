# Greedy assignment heuristic
# Request is assigned to closest robot that is available. 
# Requests that enter the system are placed in a priority queue, where requests with earlier released times have higher priority.
# Requests are assigned when they reach the front of the queue and there is at least one available robot.

import heapq
from HTAMP.assignment.assignment_helpers import TaskQueue
from HTAMP.environment.loc_dataclasses import TimeInterval
from HTAMP.environment.traversal_dataclasses import TraversalNode
from HTAMP.environment.traversal_graph_gen import TraversalGraphGenerator
from HTAMP.planning.motion_planner import MotionPlanner
from HTAMP.planning.planning_dataclasses import RequestsLists, TaskRequest
from HTAMP.planning.state import PlanningState
from HTAMP.plotting.motion_planning_plotting import MotionPlanningPlotter


class FleetManager:
    def __init__(self):
        self.monitoring_requests_queue = TaskQueue()
        self.delivery_requests_queue = TaskQueue()
    
    def _remove_expired_requests_from_queue(self, queue: TaskQueue, state: PlanningState):
            temp_heap = []
            while queue.heap:
                priority, request_id = heapq.heappop(queue.heap)
                if request_id in state.requests:
                    request = state.requests[request_id]
                    if not request.is_expired(state.simulator_time):
                        heapq.heappush(temp_heap, (priority, request_id))
                    else:
                        request.mark_rejected()
            queue.heap = temp_heap

    def _check_if_requests_in_queues_expired(self, state: PlanningState):
        self._remove_expired_requests_from_queue(queue=self.monitoring_requests_queue, 
                                                 state=state)
        self._remove_expired_requests_from_queue(queue=self.delivery_requests_queue, 
                                                 state=state)
    
    def _add_request_to_queue(self, request: TaskRequest, queue: TaskQueue):
        queue.add_task(request.scheduled_time, request.request_id)
    
    def _add_all_requests_to_queues(self, requests_lists: RequestsLists):
        for request in requests_lists.blood_pressure_requests + requests_lists.heart_rate_requests + \
                       requests_lists.respiratory_rate_requests + requests_lists.temperature_requests + \
                       requests_lists.oxygen_saturation_requests:
            self._add_request_to_queue(request, self.monitoring_requests_queue)
        for request in requests_lists.medications_requests:
            self._add_request_to_queue(request, self.delivery_requests_queue)
    
    def _determine_robot_locations(self, robot_id: int, state: PlanningState) -> TraversalNode:
        if state.robots_next_nodes[robot_id] is None:
            robot_location = state.robots_current_nodes[robot_id]
        else:
            robot_location =  state.robots_next_nodes[robot_id]
        return robot_location
    
    def _determine_path_for_robot(self, 
                                  request_id: str, 
                                  robot_id: int, 
                                  state: PlanningState,
                                  motion_planner: MotionPlanner,
                                  traversal_graph_generator: TraversalGraphGenerator) \
                                    -> tuple[list[tuple[TraversalNode, TimeInterval]], list[int], float]:
        current_request = state.requests[request_id]
        start_node = self._determine_robot_locations(robot_id, state)
        sub_paths: list[list[tuple[TraversalNode, TimeInterval]]] = []
        planned_goal_indices: list[int] = []
        planned_time_to_service_request: float = float('inf')
        for j, room_node_label in enumerate(current_request.goal_nodes):
            goal_node_id = traversal_graph_generator.doorway_to_node_dict[room_node_label]
            goal_node = traversal_graph_generator.traversal_graph.nodes_dict[goal_node_id]
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
                                                       wait_time_at_goal=120.0,
                                                       horizon=state.simulator_config.horizon)
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
    
    def _determine_closest_robot(self, 
                                 request_id: str, 
                                 available_robots: list[int], 
                                 state: PlanningState, 
                                 motion_planner: MotionPlanner, 
                                 traversal_graph_generator: TraversalGraphGenerator):
        closest_robot = None
        closest_path = []
        closest_planned_goal_indices = []
        shortest_time = float('inf')
        for robot_id in available_robots:
            path, planned_goal_indices, planned_time = self._determine_path_for_robot(request_id=request_id, 
                                                                                      robot_id=robot_id,
                                                                                      state=state, 
                                                                                      motion_planner=motion_planner, 
                                                                                      traversal_graph_generator=traversal_graph_generator)
            if len(path) > 0 and planned_time < shortest_time:
                shortest_time = planned_time
                closest_robot = robot_id
                closest_path = path
                closest_planned_goal_indices = planned_goal_indices
    
        return closest_robot, closest_path, closest_planned_goal_indices, shortest_time
    
    def _assign_request_to_robot(self,
                                 state: PlanningState,
                                 request_id: str, 
                                 robot_id: int, 
                                 planned_path: list[list[tuple[TraversalNode, TimeInterval]]], 
                                 planned_time: float,
                                 planned_goal_indices: list[int],
                                 motion_planner: MotionPlanner,
                                 traversal_graph_generator: TraversalGraphGenerator,
                                 debug: bool):
            state.requests[request_id].schedule_task(planned_time=planned_time,
                                                    planned_goal_indices=planned_goal_indices)

            if debug:
                MotionPlanningPlotter.plot_assigned_path(
                    occupancy_map=traversal_graph_generator.occupancy_map,
                    origin_x=traversal_graph_generator.origin_x,
                    origin_y=traversal_graph_generator.origin_y,
                    resolution=traversal_graph_generator.resolution,
                    request_id=request_id,
                    results_folder="results/motion_planning/debug",
                    planned_path=planned_path,
                    traversal_graph=traversal_graph_generator.traversal_graph,
                    robot_profile=state.simulator_config.robot_profiles[robot_id]
                )
                
            motion_planner.clear_reservations_for_agent(robot_profile=state.simulator_config.robot_profiles[robot_id])
            motion_planner.reserve_path_for_agent(path=planned_path,
                                                 robot_profile=state.simulator_config.robot_profiles[robot_id],
                                                 wait_time_at_goal=state.simulator_config.horizon)
            
            state.assign_request_to_robot(request_id=request_id, 
                                        robot_id=robot_id, 
                                        path=planned_path, 
                                        traversal_graph=traversal_graph_generator.traversal_graph)
    
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
            closest_robot_results = self._determine_closest_robot(request_id=next_request_id, 
                                                                  available_robots=available_robots, 
                                                                  state=state, 
                                                                  motion_planner=motion_planner,
                                                                  traversal_graph_generator=traversal_graph_generator)
            
            closest_robot, closest_path, closest_planned_goal_indices, shortest_time = closest_robot_results

            print(f"Closest robot: {closest_robot}, Shortest time: {shortest_time}")
            
            if closest_robot is not None:
                self._assign_request_to_robot(state=state,
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
            self._add_request_to_queue(request, requests_queue)
    
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
                                  requests_lists: RequestsLists, 
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