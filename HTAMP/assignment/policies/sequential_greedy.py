import copy
from typing import Optional
from HTAMP.assignment.assignment_helpers import TaskQueue
from HTAMP.environment.loc_dataclasses import TimeInterval
from HTAMP.environment.robot_dataclasses import RobotProfile
from HTAMP.environment.traversal_dataclasses import TraversalNode
from HTAMP.environment.traversal_graph_gen import TraversalGraphGenerator
from HTAMP.planning.motion_planner import MotionPlanner
from HTAMP.planning.planning_dataclasses import RequestsLists, TaskRequest
from HTAMP.planning.state import PlanningState
from HTAMP.plotting.motion_planning_plotting import MotionPlanningPlotter

class SequentialGreedy:
    def __init__(self):
        self.monitoring_requests_queue = TaskQueue()
        self.delivery_requests_queue = TaskQueue()
        self.dummy_delivery_robot_profile = RobotProfile(radius=0.10, speed=0.20, robot_id=-1, robot_type="delivery")
    
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
    
    def _add_all_requests_to_queues(self, 
                                    requests_lists: Optional[RequestsLists], 
                                    motion_planner: MotionPlanner, 
                                    traversal_graph_generator: TraversalGraphGenerator):
        if requests_lists is None:
            return
        for request in requests_lists.blood_pressure_requests + requests_lists.heart_rate_requests + \
                       requests_lists.respiratory_rate_requests + requests_lists.temperature_requests + \
                       requests_lists.oxygen_saturation_requests:
            self._add_request_to_queue(request=request, 
                                       task_queue=self.monitoring_requests_queue,
                                       motion_planner=motion_planner,
                                       traversal_graph_generator=traversal_graph_generator)
        for request in requests_lists.medications_requests:
            self._add_request_to_queue(request=request, 
                                       task_queue=self.delivery_requests_queue,
                                       motion_planner=motion_planner,
                                       traversal_graph_generator=traversal_graph_generator)
    
    def _determine_robot_locations(self, robot_id: int, state: PlanningState) -> TraversalNode:
        if state.robots_next_nodes[robot_id] is None:
            robot_location = state.robots_current_nodes[robot_id]
        else:
            robot_location =  state.robots_next_nodes[robot_id]
        return robot_location
    
    def _determine_path_for_robot(self, 
                                  request_id: str, 
                                  robot_id: int, 
                                  end_node: TraversalNode,
                                  state: PlanningState,
                                  motion_planner: MotionPlanner,
                                  traversal_graph_generator: TraversalGraphGenerator) \
                                    -> tuple[list[tuple[TraversalNode, TimeInterval]], list[int], float]:
        # TODO: fix planning to plan for multiple requests in a single path
        current_request = state.requests[request_id]
        start_node = self._determine_robot_locations(robot_id, state)
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
                                                       goal_traversal_node=end_node,
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
    
    def _calculate_costs_of_request_order(self, 
                                          request_order: list[str], 
                                          robot_id: int, 
                                          state: PlanningState, 
                                          motion_planner: MotionPlanner, 
                                          traversal_graph_generator: TraversalGraphGenerator) -> tuple[float, float]:
        start_node = self._determine_robot_locations(robot_id, state)
        total_unmodified_cost = 0.0
        total_cost = 0.0
        current_time: float = state.robots_current_time[robot_id]
        for request_id in request_order:
            current_request = state.requests[request_id]
            for j, goal_node_label in enumerate(current_request.goal_nodes):
                goal_node = traversal_graph_generator.traversal_graph.nodes_dict[goal_node_label]
                subpath_time = motion_planner.planner.heuristic(start_traversal_node=start_node,
                                                                  goal_traversal_node=goal_node,
                                                                  robot_profile=state.simulator_config.robot_profiles[robot_id])
                total_trip_time += subpath_time + current_request.wait_times_at_goals_seconds[j]
                start_node = goal_node
            time_to_complete_request = current_time + total_trip_time
            if time_to_complete_request > current_request.time_for_service:
                return float('inf'), float('inf')
            total_unmodified_cost += (time_to_complete_request - current_request.scheduled_time)
            total_cost += max(time_to_complete_request - current_request.scheduled_time, 0.0)
        return total_unmodified_cost, total_cost
        
    
    def _determine_lowest_cost_insertion_for_robot(self, 
                                                   request_id: str, 
                                                   robot_id: int, 
                                                   state: PlanningState, 
                                                   motion_planner: MotionPlanner, 
                                                   traversal_graph_generator: TraversalGraphGenerator):
        currently_assigned_requests = copy.deepcopy(state.assigned_requests[robot_id])
        new_request_order = []
        total_cost = float('inf')
        total_unmodified_cost = float('inf')
        if not currently_assigned_requests:
            new_request_order = [request_id]
            total_unmodified_cost, total_cost = self._calculate_costs_of_request_order(request_order=new_request_order,
                                                                                      robot_id=robot_id,
                                                                                      state=state,
                                                                                      motion_planner=motion_planner,
                                                                                      traversal_graph_generator=traversal_graph_generator)
        else:
            for i in range(len(currently_assigned_requests) + 1):
                trial_request_order = currently_assigned_requests[:i] + [request_id] + currently_assigned_requests[i:]
                trial_unmodified_cost, trial_total_cost = self._calculate_costs_of_request_order(request_order=trial_request_order,
                                                                                               robot_id=robot_id,
                                                                                               state=state,
                                                                                               motion_planner=motion_planner,
                                                                                               traversal_graph_generator=traversal_graph_generator)
                if trial_total_cost < total_cost:
                    total_cost = trial_total_cost
                    total_unmodified_cost = trial_unmodified_cost
                    new_request_order = trial_request_order
                elif trial_total_cost == total_cost:
                    if trial_unmodified_cost < total_unmodified_cost:
                        total_unmodified_cost = trial_unmodified_cost
                        new_request_order = trial_request_order

        return new_request_order, total_unmodified_cost, total_cost
    
    def _determine_lowest_cost_insertion_in_fleet(self, 
                                 request_id: str, 
                                 robots_list: list[int], 
                                 state: PlanningState, 
                                 motion_planner: MotionPlanner, 
                                 traversal_graph_generator: TraversalGraphGenerator):
        # TODO: find lowest cost insertion point based on heuristic costs for requests already in the system
        pass
    
    def _assign_requests_to_robots(self,
                                   state: PlanningState,
                                   motion_planner: MotionPlanner,
                                   traversal_graph_generator: TraversalGraphGenerator,
                                   robot_type: str,
                                   requests_queue: TaskQueue,
                                   debug: bool):
        # TODO: implement assignment logic
        robots_list = state.get_robots_of_type(robot_type=robot_type)

        while requests_queue.heap:
            next_request_id = requests_queue.pop_task()
    

    def _assign_requests_for_monitoring_robots(self, 
                                               state: PlanningState, 
                                               motion_planner: MotionPlanner, 
                                               traversal_graph_generator: TraversalGraphGenerator,
                                               debug: bool):
        self._assign_requests_to_robots(state=state,
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
        self._assign_requests_to_robots(state=state,
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
        self._add_all_requests_to_queues(requests_lists=requests_lists,
                                         motion_planner=motion_planner,
                                         traversal_graph_generator=traversal_graph_generator)

        # Assignment logic for monitoring robots
        self._assign_requests_for_monitoring_robots(state=state, 
                                                   motion_planner=motion_planner, 
                                                   traversal_graph_generator=traversal_graph_generator,
                                                   debug=debug)
        
        # Assignment logic for delivery robots
        self._assign_requests_for_delivery_robots(state=state, 
                                                 motion_planner=motion_planner, 
                                                 traversal_graph_generator=traversal_graph_generator,
                                                 debug=debug)