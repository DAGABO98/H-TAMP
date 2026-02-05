import heapq

from HTAMP.environment.loc_dataclasses import TimeInterval
from HTAMP.environment.traversal_dataclasses import TraversalNode
from HTAMP.environment.traversal_graph_gen import TraversalGraphGenerator
from HTAMP.planning.motion_planner import MotionPlanner
from HTAMP.planning.planning_dataclasses import TaskRequest
from HTAMP.planning.state import PlanningState
from HTAMP.plotting.motion_planning_plotting import MotionPlanningPlotter

class TaskQueue:
    def __init__(self):
        self.heap = []

    def add_task(self, priority: float, task_id: str):
        heapq.heappush(self.heap, (priority, task_id))

    def pop_task(self):
        return heapq.heappop(self.heap)[1] if self.heap else None
    
class AssignmentHelpers:
    
    @staticmethod
    def determine_robot_locations(robot_id: int, state: PlanningState) -> TraversalNode:
        if state.robots_next_nodes[robot_id] is None:
            robot_location = state.robots_current_nodes[robot_id]
        else:
            robot_location =  state.robots_next_nodes[robot_id]
        return robot_location
    
    @staticmethod
    def remove_expired_requests_from_queue(queue: TaskQueue, state: PlanningState):
        temp_heap = []
        while queue.heap:
            priority, request_id = heapq.heappop(queue.heap)
            if request_id in state.requests:
                request = state.requests[request_id]
                if not request.is_expired(state.simulator_time):
                    heapq.heappush(temp_heap, (priority, request_id))
                else:
                    print(f"Request {request_id} has expired and is being removed from the queue.")
                    request.mark_rejected(rejection_penalty=state.simulator_config.rejection_penalty)
        queue.heap = temp_heap
    
    @staticmethod
    def add_request_to_queue(request: TaskRequest, queue: TaskQueue):
        queue.add_task(request.scheduled_time, request.request_id)
    
    @staticmethod
    def determine_path_from_robot_location_to_request(request_id: str, 
                                                      robot_id: int, 
                                                      state: PlanningState,
                                                      motion_planner: MotionPlanner,
                                                      traversal_graph_generator: TraversalGraphGenerator) \
                                    -> tuple[list[tuple[TraversalNode, TimeInterval]], list[int], float]:
        current_request = state.requests[request_id]
        start_node = AssignmentHelpers.determine_robot_locations(robot_id, state)
        sub_paths: list[list[tuple[TraversalNode, TimeInterval]]] = []
        planned_goal_indices: list[int] = []
        planned_time_to_service_request: float = float('inf')
        current_time = state.robots_current_time[robot_id]
        print(f"Generating motion plan for robot {robot_id} starting at node {start_node.label} and time {state.robots_current_time[robot_id]}.")
        for j, goal_node_label in enumerate(current_request.goal_nodes):
            goal_node = traversal_graph_generator.traversal_graph.nodes_dict[goal_node_label]
            print(f"  Planning path to goal node {goal_node.label} from start node {start_node.label} at time {current_time}.")
            sub_path = motion_planner.obtain_path_for_agent(start_traversal_node=start_node,
                                                    goal_traversal_node=goal_node,
                                                    robot_profile=state.simulator_config.robot_profiles[robot_id],
                                                    current_time=current_time,
                                                    wait_time_at_goal=current_request.wait_times_at_goals_seconds[j],
                                                    horizon=current_request.time_for_service)
            print(f"    Obtained sub-path: {sub_path}")
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
            current_time = sub_paths[-1][-1][1].end
        
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
                    print(f"  Final planned path for robot {robot_id} to service request {request_id}: {final_path}")

            else:
                final_path = []
                planned_goal_indices = []
                planned_time_to_service_request = float('inf')
        else:
            final_path = []
            planned_goal_indices = []
            planned_time_to_service_request = float('inf')
        return final_path, planned_goal_indices, planned_time_to_service_request
    
    @staticmethod
    def determine_closest_robot_to_request(request_id: str, 
                                           available_robots: list[int], 
                                           state: PlanningState, 
                                           motion_planner: MotionPlanner, 
                                           traversal_graph_generator: TraversalGraphGenerator):
        closest_robot = None
        closest_path = []
        closest_planned_goal_indices = []
        shortest_time = float('inf')
        for robot_id in available_robots:
            path, planned_goal_indices, planned_time = AssignmentHelpers.determine_path_from_robot_location_to_request(
                request_id=request_id,
                robot_id=robot_id,
                state=state,
                motion_planner=motion_planner,
                traversal_graph_generator=traversal_graph_generator
            )
            if len(path) > 0 and planned_time < shortest_time:
                shortest_time = planned_time
                closest_robot = robot_id
                closest_path = path
                closest_planned_goal_indices = planned_goal_indices
    
        return closest_robot, closest_path, closest_planned_goal_indices, shortest_time
    
    @staticmethod
    def assign_request_to_robot(state: PlanningState,
                                request_id: str, 
                                robot_id: int, 
                                planned_path: list[tuple[TraversalNode, TimeInterval]], 
                                planned_time: float,
                                planned_goal_indices: list[int],
                                motion_planner: MotionPlanner,
                                traversal_graph_generator: TraversalGraphGenerator,
                                debug: bool):
        state.requests[request_id].schedule_task(planned_time=planned_time,
                                                planned_goal_indices=planned_goal_indices,
                                                assigned_robot_id=robot_id)

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

        print(f"Reserving motion plan for robot {robot_id}: {planned_path}")
            
        motion_planner.clear_reservations_for_agent(robot_profile=state.simulator_config.robot_profiles[robot_id])
        motion_planner.reserve_path_for_agent(path=planned_path,
                                              robot_profile=state.simulator_config.robot_profiles[robot_id])
        
        state.assign_request_to_robot(request_id=request_id, 
                                    robot_id=robot_id, 
                                    path=planned_path, 
                                    traversal_graph=traversal_graph_generator.traversal_graph)
    
    @staticmethod
    def heuristic_cost_from_robot_to_request(current_request: TaskRequest,
                                            robot_id: int, 
                                            state: PlanningState,
                                            motion_planner: MotionPlanner,
                                            traversal_graph_generator: TraversalGraphGenerator) -> float:
        start_node = AssignmentHelpers.determine_robot_locations(robot_id, state)
        trip_time: float = 0.0

        for j, goal_node_label in enumerate(current_request.goal_nodes):
            goal_node = traversal_graph_generator.traversal_graph.nodes_dict[goal_node_label]
            subpath_length = motion_planner.planner.heuristic(start_traversal_node=start_node,
                                                              goal_traversal_node=goal_node,
                                                              robot_profile=state.simulator_config.robot_profiles[robot_id])
            trip_time += subpath_length + current_request.wait_times_at_goals_seconds[j]
            start_node = goal_node

        return trip_time