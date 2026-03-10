from HTAMP.assignment.assignment_helpers import AssignmentHelpers, TaskQueue
from HTAMP.environment.loc_dataclasses import TimeInterval
from HTAMP.environment.robot_dataclasses import RobotProfile
from HTAMP.environment.traversal_dataclasses import TraversalNode
from HTAMP.environment.traversal_graph_gen import TraversalGraphGenerator
from HTAMP.planning.motion_planner import MotionPlanner
from HTAMP.planning.planning_dataclasses import NodeReservationTable, TaskRequest, TimeReservation
from HTAMP.planning.state import PlanningState


class PolicyHelpers:

    @staticmethod
    def _add_request_to_queue_using_pickup_deadline(request: TaskRequest, 
                             task_queue: TaskQueue,
                             delivery_robot_profile: RobotProfile,
                             motion_planner: MotionPlanner,
                             traversal_graph_generator: TraversalGraphGenerator):
        pickup_deadline = PolicyHelpers._calculate_pickup_deadline(delivery_robot_profile=delivery_robot_profile,
                                                                  request=request,
                                                                  motion_planner=motion_planner,
                                                                  traversal_graph_generator=traversal_graph_generator)

        task_queue.add_task(priority=pickup_deadline, 
                            task_id=request.request_id)
        
        return pickup_deadline
    
    @staticmethod
    def _add_request_to_queue_using_ordered_time(request: TaskRequest, 
                             task_queue: TaskQueue):
        ordered_time = request.ordered_time

        task_queue.add_task(priority=ordered_time, 
                            task_id=request.request_id)
        
        return ordered_time

    @staticmethod
    def _calculate_pickup_deadline(delivery_robot_profile: RobotProfile,
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
                                                                     robot_profile=delivery_robot_profile)
            pickup_deadline = request.time_for_service - travel_time_to_pickup - request.wait_times_at_goals_seconds[0] - request.wait_times_at_goals_seconds[1]
            
        return pickup_deadline
    
    @staticmethod
    def _reserve_nodes_for_request(robot_id: int,
                                  request_id: str,
                                  node_reservation_table: NodeReservationTable,
                                  state: PlanningState):
        request_struct = state.requests[request_id]
        for goal_index in range(request_struct.completed_goals, len(request_struct.goal_nodes)):
            goal_node_label = request_struct.goal_nodes[goal_index]
            planned_goal_index = request_struct.planned_goal_indices[goal_index]
            planned_goal_label = state.robot_paths[robot_id][planned_goal_index][0].label
            assert goal_node_label == planned_goal_label, \
                f"Mismatch in planned goal node labels: {goal_node_label} vs {planned_goal_label}"
            start_time = state.robot_paths[robot_id][planned_goal_index][1].start
            wait_time = request_struct.wait_times_at_goals_seconds[goal_index]
            planned_time = state.robot_paths[robot_id][planned_goal_index][1].end
            assert abs(planned_time - (start_time + wait_time)) < 1e-3, \
                f"Mismatch in planned time and calculated time: {planned_time} vs {start_time + wait_time}"
            reservation_interval = TimeInterval(start=start_time,
                                                end=planned_time)
            reservation = TimeReservation(robot_id=robot_id,
                                          interval=reservation_interval)
            node_reservation_table.add_reservation(node=goal_node_label,
                                                   reservation=reservation)

    @staticmethod
    def _obtain_time_to_service_node(robot_id: int,
                                     node_reservation_table: NodeReservationTable,
                                     node_label: str,
                                     arrival_time: float,
                                     wait_time: float,
                                     movement_time: float = 10.0) -> TimeInterval:
        
        reservations = node_reservation_table.get_reservations(node=node_label)
        requested_interval = TimeInterval(start=arrival_time, end=arrival_time + wait_time)
        if not reservations:
            return requested_interval
        else:
            interval_start = requested_interval.start
            interval_end = requested_interval.end
            reservations.sort(key=lambda x: x.interval.start)
            reservations = [res for res in reservations if res.robot_id != robot_id]
            for reservation in reservations:
                if interval_end <= reservation.interval.start:
                    break
                elif interval_start >= reservation.interval.end:
                    continue
                else:
                    interval_start = reservation.interval.end + movement_time
                    interval_end = interval_start + wait_time
            return TimeInterval(start=interval_start, end=interval_end)
    
    @staticmethod
    def _update_path_and_requests_indices(robot_id: int,
                                          planned_path: list[tuple[TraversalNode, TimeInterval]],
                                          currently_assigned_request_ids: list[int],
                                          state: PlanningState,
                                          traversal_graph_generator: TraversalGraphGenerator):
        if state.robots_next_nodes[robot_id] is not None:
            if state.assigned_requests[robot_id]:
                if state.current_wait_times[robot_id] > 0.0:
                    current_node_index = state.robots_current_node_index[robot_id]
                    next_node_index = state.robots_current_node_index[robot_id]
                    final_path = planned_path[next_node_index:]
                else:
                    current_node_index = state.robots_current_node_index[robot_id]
                    next_node_index = state.robots_current_node_index[robot_id] + 1
                    final_path = planned_path[next_node_index:]
            else:
                current_node_index = -1
                next_node_index = 0
                final_path = planned_path
        else:
            if state.assigned_requests[robot_id]:
                current_node_index = state.robots_current_node_index[robot_id]
                next_node_index = state.robots_current_node_index[robot_id]
                final_path = planned_path[next_node_index:]
            else:
                current_node_index = 0
                next_node_index = 0
                final_path = planned_path

        for new_request_id in currently_assigned_request_ids:
            request_struct = state.requests[new_request_id]
            for i in range(request_struct.completed_goals, len(request_struct.goal_nodes)):
                request_struct.planned_goal_indices[i] = request_struct.planned_goal_indices[i] - current_node_index
                checking_index = request_struct.planned_goal_indices[i] + (current_node_index - next_node_index)
                current_step = final_path[checking_index]
                current_node = current_step[0]
                assert current_node.label == request_struct.goal_nodes[i], \
                    f"Mismatch in planned goal node label and current node label after path update with index {current_node_index}: {request_struct.goal_nodes[i]} vs {current_node.label}"
        
        state.assign_robot_path(robot_id=robot_id, 
                                path=final_path, 
                                traversal_graph=traversal_graph_generator.traversal_graph)
        
    @staticmethod
    def _generate_motion_plan_to_depot(robot_id: int,
                                       currently_assigned_request_ids: list[int],
                                      state: PlanningState,
                                      motion_planner: MotionPlanner,
                                      traversal_graph_generator: TraversalGraphGenerator,
                                      debug: bool):
        depot_node = state.robot_depots[robot_id]
        if currently_assigned_request_ids:
            if debug:
                print(f"Generating motion plan to depot for robot {robot_id} after deallocation from requests. Robot has assigned requests: {currently_assigned_request_ids}")
            last_assigned_request_id = currently_assigned_request_ids[-1]
            last_assigned_request_struct = state.requests[last_assigned_request_id]
            planned_goal_index = last_assigned_request_struct.planned_goal_indices[-1]
            planned_goal_node_label = last_assigned_request_struct.goal_nodes[-1]
            last_path_step = state.robot_paths[robot_id][planned_goal_index]
            start_node = last_path_step[0]
            current_time = last_path_step[1].end
            assert planned_goal_node_label == start_node.label, \
                f"Mismatch in planned goal node label and start node label: {planned_goal_node_label} vs {start_node.label}"
            wait_time_at_goal = state.simulator_config.horizon - current_time
            planned_path = motion_planner.obtain_path_for_agent(start_traversal_node=start_node,
                                                        goal_traversal_node=depot_node,
                                                        robot_profile=state.simulator_config.robot_profiles[robot_id],
                                                        current_time=current_time,
                                                        wait_time_at_goal=wait_time_at_goal,
                                                        horizon=2*state.simulator_config.horizon)
            assert planned_path is not None, f"Failed to find a path to the depot for robot {robot_id} after deallocation. Robot will remain idle."

            current_path = state.robot_paths[robot_id][:planned_goal_index+1]
            planned_path = motion_planner.combine_paths([current_path, planned_path])
            if debug:
                print(f"Planned path to depot for robot {robot_id} after deallocation: {planned_path}")
                print(f"State path for robot {robot_id} before path update: {state.robot_paths[robot_id]}")
                print(f"Current wait time for robot {robot_id} before path update: {state.current_wait_times[robot_id]}")
            PolicyHelpers._update_path_and_requests_indices(robot_id=robot_id,
                                                  planned_path=planned_path,
                                                  currently_assigned_request_ids=currently_assigned_request_ids,
                                                  state=state,
                                                  traversal_graph_generator=traversal_graph_generator)
        else:
            start_node, current_time = AssignmentHelpers.determine_robot_nodes_and_times(robot_id=robot_id,
                                                                                        state=state)
            wait_time_at_goal = state.simulator_config.horizon - current_time
            planned_path = motion_planner.obtain_path_for_agent(start_traversal_node=start_node,
                                                            goal_traversal_node=depot_node,
                                                            robot_profile=state.simulator_config.robot_profiles[robot_id],
                                                            current_time=current_time,
                                                            wait_time_at_goal=wait_time_at_goal,
                                                            horizon=2*state.simulator_config.horizon)
            assert planned_path is not None, f"Failed to find a path to the depot for robot {robot_id} after deallocation. Robot will remain idle."
            state.assign_robot_path(robot_id=robot_id, 
                                    path=planned_path, 
                                    traversal_graph=traversal_graph_generator.traversal_graph)
        motion_planner.clear_reservations_for_agent(robot_profile=state.simulator_config.robot_profiles[robot_id])
        motion_planner.reserve_path_for_agent(path=state.robot_paths[robot_id],
                                              robot_profile=state.simulator_config.robot_profiles[robot_id])
    
    @staticmethod
    def _find_path_for_goal_nodes(robot_id: int,
                                 start_node: TraversalNode,
                                 start_time: float,
                                 current_request: TaskRequest,
                                 initial_planned_goal_index: int,
                                 state: PlanningState,
                                 motion_planner: MotionPlanner,
                                 traversal_graph_generator: TraversalGraphGenerator,
                                 debug: bool) \
                                    -> tuple[list[tuple[TraversalNode, TimeInterval]], list[int], float]:
        sub_paths: list[list[tuple[TraversalNode, TimeInterval]]] = []
        planned_goal_indices: list[int] = []
        planned_time_to_reach_last_goal: float = float('inf')
        for j, goal_node_label in enumerate(current_request.goal_nodes):
            goal_node = traversal_graph_generator.traversal_graph.nodes_dict[goal_node_label]
            current_time = start_time if not sub_paths else sub_paths[-1][-1][1].end
            sub_path = motion_planner.obtain_path_for_agent(start_traversal_node=start_node,
                                                    goal_traversal_node=goal_node,
                                                    robot_profile=state.simulator_config.robot_profiles[robot_id],
                                                    current_time=current_time,
                                                    wait_time_at_goal=current_request.wait_times_at_goals_seconds[j],
                                                    horizon=state.simulator_config.horizon)
            if sub_path is None:
                sub_paths = []
                planned_goal_indices = []
                planned_time_to_reach_last_goal = float('inf')
                break
            sub_paths.append(sub_path)
            if initial_planned_goal_index == -1:
                new_goal_index = len(sub_path) - 1
            else:
                new_goal_index = initial_planned_goal_index + len(sub_path)
            planned_goal_indices.append(new_goal_index)
            initial_planned_goal_index = new_goal_index
            start_node = goal_node
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
                planned_time_to_reach_last_goal = sub_paths[-2][-1][1].end
                if planned_time_to_reach_last_goal > current_request.time_for_service:
                    if debug:
                        print(f"Planned time to reach last goal for robot {robot_id} and request {current_request.request_id} exceeds time for service. Planned time: {planned_time_to_reach_last_goal}, time for service: {current_request.time_for_service}.")
                    final_path = []
                    planned_goal_indices = []
                    planned_time_to_reach_last_goal = float('inf')
                else:
                    final_path = motion_planner.combine_paths(sub_paths)
            else:
                if debug:
                    print(f"Failed to find a return path to the depot for robot {robot_id} after reaching last goal node {goal_node_label} for request {current_request.request_id}.")
                final_path = []
                planned_goal_indices = []
                planned_time_to_reach_last_goal = float('inf')
        else:
            if debug:
                print(f"Failed to find a path for robot {robot_id} to at least one of the goal nodes for request {current_request.request_id}.")
            final_path = []
            planned_goal_indices = []
            planned_time_to_reach_last_goal = float('inf') 

        return final_path, planned_goal_indices, planned_time_to_reach_last_goal
    
    @staticmethod
    def _get_planned_path_for_request_assignment(robot_id: int,
                                                 request_id: str,
                                                 currently_assigned_request_ids: list[int],
                                                 state: PlanningState,
                                                 motion_planner: MotionPlanner,
                                                 traversal_graph_generator: TraversalGraphGenerator,
                                                 debug: bool) -> tuple[list[tuple[TraversalNode, TimeInterval]], list[int], float]:
        request_struct = state.requests[request_id]
        if currently_assigned_request_ids:
            last_assigned_request_id = currently_assigned_request_ids[-1]
            last_assigned_request_struct = state.requests[last_assigned_request_id]
            last_planned_goal_index = last_assigned_request_struct.planned_goal_indices[-1]
            last_path_step = state.robot_paths[robot_id][last_planned_goal_index]
            start_node = last_path_step[0]
            start_time = last_path_step[1].end
            assert last_assigned_request_struct.goal_nodes[-1] == start_node.label, \
                f"Mismatch in planned goal node label and start node label: {last_assigned_request_struct.goal_nodes[-1]} vs {start_node.label}"
            initial_planned_goal_index = last_planned_goal_index
        else:
            start_node, start_time = AssignmentHelpers.determine_robot_nodes_and_times(robot_id=robot_id,
                                                                                    state=state)
            initial_planned_goal_index = -1
        planned_path, planned_goal_indices, planned_time_to_reach_last_goal = PolicyHelpers._find_path_for_goal_nodes(robot_id=robot_id,
                                                                                                            start_node=start_node,
                                                                                                            start_time=start_time,
                                                                                                            current_request=request_struct,
                                                                                                            initial_planned_goal_index=initial_planned_goal_index,
                                                                                                            state=state,
                                                                                                            motion_planner=motion_planner,
                                                                                                            traversal_graph_generator=traversal_graph_generator,
                                                                                                            debug=debug)
        
        return planned_path, planned_goal_indices, planned_time_to_reach_last_goal
    
    @staticmethod
    def _schedule_request(robot_id: int,
                          request_id: str,
                          currently_assigned_request_ids: list[int],
                          node_reservation_table: NodeReservationTable,
                          planned_path: list[tuple[TraversalNode, TimeInterval]],
                          planned_goal_indices: list[int],
                          planned_time_to_reach_last_goal: float,
                          state: PlanningState,
                          motion_planner: MotionPlanner,
                          traversal_graph_generator: TraversalGraphGenerator):
        if state.assigned_requests[robot_id]:
            last_assigned_request_id = state.assigned_requests[robot_id][-1]
            last_goal_index = state.requests[last_assigned_request_id].planned_goal_indices[-1]
            combined_path = motion_planner.combine_paths([state.robot_paths[robot_id][:last_goal_index+1], planned_path])
        else:
            combined_path = planned_path
        request_struct = state.requests[request_id]
        request_struct.schedule_task(assigned_robot_id=robot_id,
                                    planned_goal_indices=planned_goal_indices,
                                    planned_time=planned_time_to_reach_last_goal)
        currently_assigned_request_ids.append(request_id)
        PolicyHelpers._update_path_and_requests_indices(robot_id=robot_id,
                                              planned_path=combined_path,
                                              currently_assigned_request_ids=currently_assigned_request_ids,
                                              state=state,
                                              traversal_graph_generator=traversal_graph_generator)
        state.assigned_requests[robot_id].append(request_id)
        motion_planner.clear_reservations_for_agent(robot_profile=state.simulator_config.robot_profiles[robot_id])
        motion_planner.reserve_path_for_agent(path=state.robot_paths[robot_id],
                                            robot_profile=state.simulator_config.robot_profiles[robot_id])
    
        PolicyHelpers._reserve_nodes_for_request(robot_id=robot_id,
                                       request_id=request_id,
                                       node_reservation_table=node_reservation_table,
                                       state=state)