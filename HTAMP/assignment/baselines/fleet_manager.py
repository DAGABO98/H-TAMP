# Greedy assignment heuristic
# Request is assigned to closest robot that is available. 
# Requests that enter the system are placed in a priority queue, where requests with earlier released times have higher priority.
# Requests are assigned when they reach the front of the queue and there is at least one available robot.

from HTAMP.planning.motion_planner import MotionPlanner
from HTAMP.planning.planning_dataclasses import RequestsLists
from HTAMP.planning.state import PlanningState


class FleetManager:
    def __init__(self):
        pass

    def assign_requests_to_robots(self, state: PlanningState, requests_lists: RequestsLists, motion_planner: MotionPlanner):
        pass