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


class FleetManager:
    def __init__(self):
        self.blood_pressure_requests_queue = TaskQueue()
        self.heart_rate_requests_queue = TaskQueue()
        self.respiratory_rate_requests_queue = TaskQueue()
        self.temperature_requests_queue = TaskQueue()
        self.oxygen_saturation_requests_queue = TaskQueue()
        self.medications_requests_queue = TaskQueue()
    
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
        self._remove_expired_requests_from_queue(queue=self.blood_pressure_requests_queue, 
                                                 state=state)
        self._remove_expired_requests_from_queue(queue=self.heart_rate_requests_queue, 
                                                 state=state)
        self._remove_expired_requests_from_queue(queue=self.respiratory_rate_requests_queue, 
                                                 state=state)
        self._remove_expired_requests_from_queue(queue=self.temperature_requests_queue, 
                                                 state=state)
        self._remove_expired_requests_from_queue(queue=self.oxygen_saturation_requests_queue, 
                                                 state=state)
        self._remove_expired_requests_from_queue(queue=self.medications_requests_queue, 
                                                 state=state)
    
    def _add_request_to_queue(self, request: TaskRequest, queue: TaskQueue):
        queue.add_task(request.scheduled_time, request.request_id)
    
    def _add_all_requests_to_queues(self, requests_lists: RequestsLists):
        for request in requests_lists.blood_pressure_requests:
            self._add_request_to_queue(request, self.blood_pressure_requests_queue)
        for request in requests_lists.heart_rate_requests:
            self._add_request_to_queue(request, self.heart_rate_requests_queue)
        for request in requests_lists.respiratory_rate_requests:
            self._add_request_to_queue(request, self.respiratory_rate_requests_queue)
        for request in requests_lists.temperature_requests:
            self._add_request_to_queue(request, self.temperature_requests_queue)
        for request in requests_lists.oxygen_saturation_requests:
            self._add_request_to_queue(request, self.oxygen_saturation_requests_queue)
        for request in requests_lists.medications_requests:
            self._add_request_to_queue(request, self.medications_requests_queue)
    
    def _determine_robot_locations(self, robot_id: int, state: PlanningState):
        if state.robots_next_nodes[robot_id] is None:
            robot_location = state.robots_positions[robot_id]
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
        for j, goal_node_label in enumerate(current_request.goal_nodes):
            print(traversal_graph_generator.traversal_graph.nodes_dict)
            goal_node = traversal_graph_generator.traversal_graph.nodes_dict[goal_node_label]
            current_time = 0.0 if not sub_paths else sub_paths[-1][-1][1].end
            sub_path = motion_planner.obtain_path_for_agent(start_traversal_node=start_node,
                                                    goal_traversal_node=goal_node,
                                                    robot_profile=state.simulator_config.robot_profiles[robot_id],
                                                    current_time=current_time,
                                                    wait_time_at_goal=current_request.wait_times_at_goals_seconds[j],
                                                    horizon=state.simulator_config.horizon)
            if not sub_path:
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
                                                       horizon=state.simulator_config.horizon)
            if return_path:
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
            if path and planned_time < shortest_time:
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
                                 traversal_graph_generator: TraversalGraphGenerator):
            state.requests[request_id].schedule_task(planned_time=planned_time,
                                                    planned_goal_indices=planned_goal_indices)
            motion_planner.clear_reservations_for_agent(robot_profile=state.simulator_config.robot_profiles[robot_id])
            motion_planner.reserve_path_for_agent(path=planned_path,
                                                 robot_profile=state.simulator_config.robot_profiles[robot_id],
                                                 wait_time_at_goal=state.simulator_config.horizon)
            state.assign_request_to_robot(request_id=request_id, 
                                        robot_id=robot_id, 
                                        path=planned_path, 
                                        traversal_graph=traversal_graph_generator.traversal_graph)
            
    
    def _assign_requests_for_robot_type_1(self, 
                                          state: PlanningState, 
                                          motion_planner: MotionPlanner, 
                                          traversal_graph_generator: TraversalGraphGenerator):
        available_robots = state.get_available_robots(robot_type="type_1")
        requests_to_add_back_heart_rate = []
        requests_to_add_back_oxygen_saturation = []
        while available_robots:
            if not self.heart_rate_requests_queue.heap and not self.oxygen_saturation_requests_queue.heap:
                break  # No more requests to assign

            # Get the next request from the highest priority queue
            next_request_id = None
            if (self.heart_rate_requests_queue.heap and 
                (not self.oxygen_saturation_requests_queue.heap or 
                 self.heart_rate_requests_queue.heap[0] < self.oxygen_saturation_requests_queue.heap[0])):
                next_request_id = self.heart_rate_requests_queue.pop_task()
            else:
                next_request_id = self.oxygen_saturation_requests_queue.pop_task()

            if next_request_id is not None:
                closest_robot_results = self._determine_closest_robot(request_id=next_request_id, 
                                                                      available_robots=available_robots, 
                                                                      state=state, 
                                                                      motion_planner=motion_planner,
                                                                      traversal_graph_generator=traversal_graph_generator)
                
                closest_robot, closest_path, closest_planned_goal_indices, shortest_time = closest_robot_results
                
                if closest_robot is not None:
                    self._assign_request_to_robot(state=state,
                                                 request_id=next_request_id,
                                                 robot_id=closest_robot,
                                                 planned_path=closest_path,
                                                 planned_time=shortest_time,
                                                 planned_goal_indices=closest_planned_goal_indices,
                                                 motion_planner=motion_planner,
                                                 traversal_graph_generator=traversal_graph_generator)
                    available_robots.remove(closest_robot)
                else:
                    # No feasible robot found for this request, re-add it to the queue
                    if next_request_id in state.requests:
                        request = state.requests[next_request_id]
                        if request.request_type == "heart_rate":
                            requests_to_add_back_heart_rate.append(request)
                        elif request.request_type == "oxygen_saturation":
                            requests_to_add_back_oxygen_saturation.append(request)
                            
        for request in requests_to_add_back_heart_rate:
            self._add_request_to_queue(request, self.heart_rate_requests_queue)
        for request in requests_to_add_back_oxygen_saturation:
            self._add_request_to_queue(request, self.oxygen_saturation_requests_queue)
    
    def _assign_requests_for_robot_type_2(self, 
                                          state: PlanningState, 
                                          motion_planner: MotionPlanner, 
                                          traversal_graph_generator: TraversalGraphGenerator):
        available_robots = state.get_available_robots(robot_type="type_2")
        requests_to_add_back_blood_pressure = []
        requests_to_add_back_heart_rate = []
        while available_robots:
            if not self.blood_pressure_requests_queue.heap and not self.heart_rate_requests_queue.heap:
                break  # No more requests to assign

            # Get the next request from the highest priority queue
            next_request_id = None
            if (self.blood_pressure_requests_queue.heap and 
                (not self.heart_rate_requests_queue.heap or 
                 self.blood_pressure_requests_queue.heap[0] < self.heart_rate_requests_queue.heap[0])):
                next_request_id = self.blood_pressure_requests_queue.pop_task()
            else:
                next_request_id = self.heart_rate_requests_queue.pop_task()
            if next_request_id is not None:
                closest_robot_results = self._determine_closest_robot(request_id=next_request_id, 
                                                                      available_robots=available_robots, 
                                                                      state=state, 
                                                                      motion_planner=motion_planner,
                                                                      traversal_graph_generator=traversal_graph_generator)
                
                closest_robot, closest_path, closest_planned_goal_indices, shortest_time = closest_robot_results
                
                if closest_robot is not None:
                    self._assign_request_to_robot(state=state,
                                                 request_id=next_request_id,
                                                 robot_id=closest_robot,
                                                 planned_path=closest_path,
                                                 planned_time=shortest_time,
                                                 planned_goal_indices=closest_planned_goal_indices,
                                                 motion_planner=motion_planner,
                                                 traversal_graph_generator=traversal_graph_generator)
                    available_robots.remove(closest_robot)
                else:
                    # No feasible robot found for this request, re-add it to the queue
                    if next_request_id in state.requests:
                        request = state.requests[next_request_id]
                        if request.request_type == "blood_pressure":
                            requests_to_add_back_blood_pressure.append(request)
                        elif request.request_type == "heart_rate":
                            requests_to_add_back_heart_rate.append(request)
                            
        for request in requests_to_add_back_blood_pressure:
            self._add_request_to_queue(request, self.blood_pressure_requests_queue)
        for request in requests_to_add_back_heart_rate:
            self._add_request_to_queue(request, self.heart_rate_requests_queue)
    
    def _assign_requests_for_robot_type_3(self, 
                                          state: PlanningState, 
                                          motion_planner: MotionPlanner, 
                                          traversal_graph_generator: TraversalGraphGenerator):
        available_robots = state.get_available_robots(robot_type="type_3")
        requests_to_add_back_temperature = []
        requests_to_add_back_respiratory_rate = []
        while available_robots:
            if not self.temperature_requests_queue.heap and not self.respiratory_rate_requests_queue.heap:
                break  # No more requests to assign

            # Get the next request from the highest priority queue
            next_request_id = None
            if (self.temperature_requests_queue.heap and 
                (not self.respiratory_rate_requests_queue.heap or 
                 self.temperature_requests_queue.heap[0] < self.respiratory_rate_requests_queue.heap[0])):
                next_request_id = self.temperature_requests_queue.pop_task()
            else:
                next_request_id = self.respiratory_rate_requests_queue.pop_task()
            if next_request_id is not None:
                closest_robot_results = self._determine_closest_robot(request_id=next_request_id, 
                                                                      available_robots=available_robots, 
                                                                      state=state, 
                                                                      motion_planner=motion_planner,
                                                                      traversal_graph_generator=traversal_graph_generator)
                
                closest_robot, closest_path, closest_planned_goal_indices, shortest_time = closest_robot_results
                
                if closest_robot is not None:
                    self._assign_request_to_robot(state=state,
                                                 request_id=next_request_id,
                                                 robot_id=closest_robot,
                                                 planned_path=closest_path,
                                                 planned_time=shortest_time,
                                                 planned_goal_indices=closest_planned_goal_indices,
                                                 motion_planner=motion_planner,
                                                 traversal_graph_generator=traversal_graph_generator)
                    available_robots.remove(closest_robot)
                else:
                    # No feasible robot found for this request, re-add it to the queue
                    if next_request_id in state.requests:
                        request = state.requests[next_request_id]
                        if request.request_type == "temperature":
                            requests_to_add_back_temperature.append(request)
                        elif request.request_type == "respiratory_rate":
                            requests_to_add_back_respiratory_rate.append(request)
                            
        for request in requests_to_add_back_temperature:
            self._add_request_to_queue(request, self.temperature_requests_queue)
        for request in requests_to_add_back_respiratory_rate:
            self._add_request_to_queue(request, self.respiratory_rate_requests_queue)
    
    def _assign_requests_for_robot_type_4(self, 
                                          state: PlanningState, 
                                          motion_planner: MotionPlanner, 
                                          traversal_graph_generator: TraversalGraphGenerator):
        available_robots = state.get_available_robots(robot_type="type_4")
        requests_to_add_back_medications = []
        while available_robots:
            if not self.medications_requests_queue.heap:
                break  # No more requests to assign

            # Get the next request from the highest priority queue
            next_request_id = self.medications_requests_queue.pop_task()
            if next_request_id is not None:
                closest_robot_results = self._determine_closest_robot(request_id=next_request_id, 
                                                                      available_robots=available_robots, 
                                                                      state=state, 
                                                                      motion_planner=motion_planner,
                                                                      traversal_graph_generator=traversal_graph_generator)
                
                closest_robot, closest_path, closest_planned_goal_indices, shortest_time = closest_robot_results
                
                if closest_robot is not None:
                    self._assign_request_to_robot(state=state,
                                                 request_id=next_request_id,
                                                 robot_id=closest_robot,
                                                 planned_path=closest_path,
                                                 planned_time=shortest_time,
                                                 planned_goal_indices=closest_planned_goal_indices,
                                                 motion_planner=motion_planner,
                                                 traversal_graph_generator=traversal_graph_generator)
                    available_robots.remove(closest_robot)
                else:
                    # No feasible robot found for this request, re-add it to the queue
                    if next_request_id in state.requests:
                        request = state.requests[next_request_id]
                        if request.request_type == "medications":
                            requests_to_add_back_medications.append(request)
                            
        for request in requests_to_add_back_medications:
            self._add_request_to_queue(request, self.medications_requests_queue)

    def assign_requests_to_robots(self, 
                                  state: PlanningState, 
                                  requests_lists: RequestsLists, 
                                  motion_planner: MotionPlanner,
                                  traversal_graph_generator: TraversalGraphGenerator):
        # Add new requests to the appropriate queues
        self._check_if_requests_in_queues_expired(state)
        self._add_all_requests_to_queues(requests_lists)

        # Assignment logic for robot type 1
        self._assign_requests_for_robot_type_1(state, 
                                               motion_planner, 
                                               traversal_graph_generator)
        
        # Assignment logic for robot type 2
        self._assign_requests_for_robot_type_2(state, 
                                               motion_planner, 
                                               traversal_graph_generator)
        
        # Assignment logic for robot type 3
        self._assign_requests_for_robot_type_3(state, 
                                               motion_planner, 
                                               traversal_graph_generator)
        
        # Assignment logic for robot type 4
        self._assign_requests_for_robot_type_4(state, 
                                               motion_planner, 
                                               traversal_graph_generator)