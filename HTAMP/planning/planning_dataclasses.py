

from HTAMP.environment.robot_dataclasses import RobotProfile
from HTAMP.environment.traversal_dataclasses import TraversalNode

class DateOperationalRange:
    def __init__(self, year: int, month: int, day: int, start_hour: int, end_hour: int, month_lengths=None):
        self.start_date = (year, month, day, start_hour)
        self.end_date = (year, month, day, end_hour)
        if month_lengths is None:
            month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        self.month_lengths = month_lengths

    def __repr__(self):
        return f"DateOperationalRange(start_date={self.start_date}, end_date={self.end_date})"


class Request:
    def __init__(self, user_id: int, request_type: str, payload: dict):
        self.user_id = user_id
        self.request_type = request_type
        self.payload = payload
    def __repr__(self):
        return f"Request(user_id={self.user_id}, request_type='{self.request_type}', payload={self.payload})"


class SimulatorConfig:
    def __init__(self, 
                 robot_profiles: list[RobotProfile], 
                 rejection_penalty: float, 
                 date_range: DateOperationalRange,
                 initial_robot_positions: dict[int, str]):
        self.robot_profiles = robot_profiles
        self.rejection_penalty = rejection_penalty
        self.date_range = date_range
        self.initial_robot_positions = initial_robot_positions
