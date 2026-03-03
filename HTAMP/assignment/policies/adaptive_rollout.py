import copy
from typing import Optional
from HTAMP.assignment.assignment_helpers import AssignmentHelpers, TaskQueue
from HTAMP.assignment.policies.base_policy import SequentialGreedyBasePolicy
from HTAMP.environment.loc_dataclasses import TimeInterval
from HTAMP.environment.robot_dataclasses import RobotProfile
from HTAMP.environment.traversal_dataclasses import TraversalNode
from HTAMP.environment.traversal_graph_gen import TraversalGraphGenerator
from HTAMP.planning.motion_planner import MotionPlanner
from HTAMP.planning.planning_dataclasses import NodeReservationTable, RequestsLists, TaskRequest, TimeReservation
from HTAMP.planning.state import PlanningState

class AdaptiveRollout:
    def __init__(self):
        self.requests_queue = TaskQueue()
        self.dummy_delivery_robot_profile = RobotProfile(radius=0.10, speed=0.20, robot_id=-1, robot_type="delivery")
        self.assigned_requests:  dict[int, list[str]]  = {}
        self.node_reservation_table = NodeReservationTable(reservations={},
                                                          robot_node_dict={})