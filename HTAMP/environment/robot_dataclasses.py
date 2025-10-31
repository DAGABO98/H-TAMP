from dataclasses import dataclass

@dataclass
class RobotProfile:
    radius: float
    robot_id: int
    speed: float  # meters per second
    robot_type: str = "default"