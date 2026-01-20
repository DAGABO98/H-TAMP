from HTAMP.planning.motion_planner import MotionPlanner
from HTAMP.planning.planning_dataclasses import RequestsLists
from HTAMP.planning.state import PlanningState


class GreedyPolicyWithReoptimization:
    def __init__(self):
        pass

    def assign_requests_to_robots(self, state: PlanningState, requests_lists: RequestsLists, motion_planner: MotionPlanner):
        pass