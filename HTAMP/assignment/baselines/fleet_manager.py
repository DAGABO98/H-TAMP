# Greedy assignment heuristic
# Request is assigned to closest robot that is available. 
# Requests that enter the system are placed in a priority queue, where requests with earlier released times have higher priority.
# Requests are assigned when they reach the front of the queue and there is at least one available robot.

from typing import Optional
from HTAMP.assignment.assignment_helpers import AssignmentHelpers, TaskQueue
from HTAMP.environment.traversal_graph_gen import TraversalGraphGenerator
from HTAMP.planning.motion_planner import MotionPlanner
from HTAMP.planning.planning_dataclasses import RequestsLists
from HTAMP.planning.state import PlanningState

class FleetManager:
    def __init__(self):
        self.monitoring_requests_queue = TaskQueue()
        self.delivery_requests_queue = TaskQueue()

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
    
    def _assign_requests_to_available_robots(self,
                                            state: PlanningState,
                                            motion_planner: MotionPlanner,
                                            traversal_graph_generator: TraversalGraphGenerator,
                                            robot_type: str,
                                            requests_queue: TaskQueue,
                                            debug: bool):
        available_robots = state.get_available_robots(robot_type=robot_type)
        requests_to_add_back = []
        if debug:
            print(f"Available robots for {robot_type}: {available_robots}")
        while available_robots:
            if not requests_queue.heap:
                break  # No more requests to assign

            # Get the next request from the highest priority queue
            next_request_id = requests_queue.pop_task()
            if debug:
                print(f"Attempting to assign request {next_request_id} of type {robot_type}")
            closest_robot_results = AssignmentHelpers.determine_closest_robot_to_request(request_id=next_request_id, 
                                                                  available_robots=available_robots, 
                                                                  state=state, 
                                                                  motion_planner=motion_planner,
                                                                  traversal_graph_generator=traversal_graph_generator,
                                                                  debug=debug)
            
            closest_robot, closest_path, closest_planned_goal_indices, shortest_time = closest_robot_results

            print(f"Closest robot for request {next_request_id}: {closest_robot} with path {closest_path} and time {shortest_time}")
            
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
                print(f"No available robot could be assigned to request {next_request_id} at this time. It will be re-added to the queue.")

        for request in requests_to_add_back:
            AssignmentHelpers.add_request_to_queue(request, requests_queue)
    
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