import argparse
import copy
import traceback
import random
from datetime import datetime
from typing import Optional

from HTAMP.environment.grid_world import GridWorld
from HTAMP.environment.robot_dataclasses import RobotProfile
from HTAMP.environment.traversal_dataclasses import TraversalNode
from HTAMP.environment.traversal_graph_gen import TraversalGraphGenerator
from HTAMP.planning.motion_planner import MotionPlanner
from HTAMP.planning.planning_dataclasses import DateStamp, FrameData, SimulatorConfig
from HTAMP.planning.state import PlanningState
from HTAMP.plotting.motion_planning_plotting import MotionPlanningPlotter

class AssignmentEvaluator:
    def __init__(self, args, random_seed=None, robot_profiles: Optional[list[RobotProfile]] = None):
        self.args = args
        self.random_seed = random_seed
        print("Generating Traversal Graph...")
        self._initialize_traversal_graph_generator()
        self._initialize_robots(robot_profiles=robot_profiles)
        self._initialize_grid_world()
        self._initialize_motion_planner()
        self._initialize_simulator_config()
        self.state = PlanningState(simulator_config=self.simulator_config)
    
    def _initialize_traversal_graph_generator(self):
        print("Generating Traversal Graph...")
        self.tg_generator = TraversalGraphGenerator(occupancy_map_path=self.args.occupancy_map_path,
                                            config_path=self.args.config_path,
                                            meters_per_pixel=self.args.meters_per_pixel,
                                            factor=self.args.factor)
    
    def _initialize_robots(self, robot_profiles: Optional[list[RobotProfile]] = None):
        if robot_profiles is not None:
            self.robot_profiles = robot_profiles
        else:
            self.robot_profiles = []
            for i in range(self.args.num_robots):
                robot_profile = RobotProfile(radius=0.10, speed=0.20, robot_id=i)
                self.robot_profiles.append(robot_profile)
    
    def _initialize_grid_world(self):
        print("Creating Grid World...")
        self.world = GridWorld(cell_size=self.tg_generator.meters_per_cell,
                      fps=self.args.fps,
                      occupancy_map=self.tg_generator.occupancy_map,
                      traversal_graph=self.tg_generator.traversal_graph,
                      shortest_paths=self.tg_generator.shortest_paths,
                      robot_profiles=self.robot_profiles,
                      use_saved_data=self.args.use_saved_data,
                      occupancy_reservations_file=self.args.occupancy_reservations_file)
        print("Grid World created.")
    
    def _initialize_motion_planner(self):
        print("Initializing Motion Planner...")
        self.motion_planner = MotionPlanner(grid=self.world, weight_factor=1.0)
        for i in range(self.args.num_robots):
            self.motion_planner._initialize_robot_reservations(initial_node=self.selected_start_nodes[i],
                                               robot_profile=self.robot_profiles[i],
                                               current_time=0.0,
                                               horizon=self.simulator_config.horizon)
        print("Motion Planner initialized.")
    
    def _generate_parking_positions(self):
        potential_start_nodes = []
        for parking_space in self.tg_generator.parking_spaces_subgraphs:
            entry_nodes = parking_space.up_parking_nodes_exit + parking_space.down_parking_nodes_exit
            for entry_node_label in entry_nodes:
                entry_node = self.tg_generator.traversal_graph.nodes_dict[entry_node_label]
                potential_start_nodes.append(entry_node)
        
        if self.random_seed is not None:
            random.seed(self.random_seed)
        
        self.selected_start_nodes: list[TraversalNode] = random.sample(potential_start_nodes, self.args.num_robots)
    
    def _initialize_simulator_config(self):
        self._generate_parking_positions()
        self.simulator_config = SimulatorConfig(fps=self.args.fps,
                                       robot_profiles=self.robot_profiles,
                                       rejection_penalty=100.0,
                                       date_range=None,
                                       initial_robot_positions={i: self.selected_start_nodes[i].position for i in range(self.args.num_robots)},
                                       horizon=5000.0)
    
    def evaluate_assignment(self, date_stamp: DateStamp, 
                            hour_range: Optional[tuple[int, int]] = None,
                              save_frame_data: bool = False) -> tuple[FrameData, float]:
        if save_frame_data:
            frame_data = FrameData()
        else:
            frame_data = None
        
        if hour_range is not None:
            start_hour, end_hour = hour_range
        else:
            start_hour, end_hour = 0, 24
        
        # extract initial requests for the day
        # requests must be ordered before the current time signal and must be scheduled to be serviced within 30 minutes of the current time

        for hour in range(start_hour, end_hour):
            for minute in range(60):
                time_signal = (date_stamp.year, date_stamp.month, date_stamp.day, hour, minute)

                # extract requests for the current time signal
                # requests must be ordered before the current time signal and must be scheduled to be serviced within 30 minutes of the current time
                requests = self.request_handler.extract_requests_for_time_signal(time_signal)

                for second in range(60):
                    for frames in range(self.args.fps):
                        self.state.step(self.tg_generator.traversal_graph)
                        if save_frame_data and frame_data is not None:
                            frame_data.robot_positions_seq.append(copy.deepcopy(self.state.robots_positions))
                            frame_data.robots_current_node_index_seq.append(copy.deepcopy(self.state.robots_current_node_index))
                            frame_data.point_indices_on_edge_seq.append(copy.deepcopy(self.state.point_indices_on_edge))
                            frame_data.robot_paths_seq.append(copy.deepcopy(self.state.robot_paths))
                            planned_goal_indices_dict: dict[int, list[int]] = {}
                            completed_goals_dict: dict[int, int] = {}
                            for robot_id, requests in self.state.assigned_requests.items():
                                if requests:
                                    planned_goal_indices_dict[robot_id] = self.state.assigned_requests[robot_id][0].planned_goal_indices
                                    completed_goals_dict[robot_id] = self.state.assigned_requests[robot_id][0].completed_goals
                            frame_data.planned_goal_indices_seq.append(copy.deepcopy(planned_goal_indices_dict))
                            frame_data.completed_goals_seq.append(copy.deepcopy(completed_goals_dict))

        return frame_data, 0.0  # Placeholder for actual evaluation metric


def experiment(args, random_seed=None):
    robot_profiles = []
    for i in range(args.num_robots):
        robot_profile = RobotProfile(radius=0.10, speed=0.20, robot_id=i)
        robot_profiles.append(robot_profile)

    evaluator = AssignmentEvaluator(args, random_seed=random_seed, robot_profiles=robot_profiles)
    frame_data, total_cost = evaluator.evaluate_assignment(date_stamp=DateStamp(year=args.year, month=args.month, day=args.day), 
                                                           hour_range=(13,14),
                                                           save_frame_data=False)

    if frame_data is not None:
        MotionPlanningPlotter.generate_state_animation(occupancy_map=evaluator.tg_generator.occupancy_map,
                                            origin_x=evaluator.tg_generator.origin_x,
                                            origin_y=evaluator.tg_generator.origin_y,
                                            resolution=evaluator.tg_generator.meters_per_cell,
                                            robot_positions_seq=frame_data.robot_positions_seq,
                                            robots_current_node_index_seq=frame_data.robots_current_node_index_seq,
                                            point_indices_on_edge_seq=frame_data.point_indices_on_edge_seq,
                                            robot_paths_seq=frame_data.robot_paths_seq,
                                            planned_goal_indices_seq=frame_data.planned_goal_indices_seq,
                                            completed_goals_seq=frame_data.completed_goals_seq,
                                            traversal_graph=evaluator.tg_generator.traversal_graph,
                                            robot_profiles=robot_profiles,
                                            fps_sim=args.fps,
                                            num_sim_frames=1000)

def main():
    parser = argparse.ArgumentParser(prog='evaluate_assignment.py',
                                     description='Evaluate assignment algorithms in a hospital floor environment.')
    # date_operational_range parameters
    parser.add_argument("--year", type=int, dest='year', default=2022, help='Select year of interest.')
    parser.add_argument("--month", type=int, dest='month', default=10, help='Select month of interest.')
    parser.add_argument("--day", type=int, dest='day', default=17, help='Select day of interest.')

    # environment parameters
    parser.add_argument("--config_path", type=str, default="maps/hospital_floor/floor_config.yaml", help="Path to the configuration file")
    parser.add_argument("--occupancy_map_path", type=str, default="maps/hospital_floor/occupancy_map.npy", help="Path to the input occupancy map")
    parser.add_argument("--factor", type=int, default=1, help="Downsampling factor")
    parser.add_argument("--meters_per_pixel", type=float, default=0.036, help="Meters per pixel in the original image")
    parser.add_argument("--fps", type=float, default=2.0, help="Frames per second for the grid world")
    parser.add_argument("--occupancy_reservations_file", type=str, default="data/occupancy_reservations.pkl", help="Path to the occupancy reservations file")
    parser.add_argument("--use_saved_data", action='store_true', help="Whether to use saved occupancy reservations data")

    # simulation parameters
    parser.add_argument("--mode", type=int, dest='mode', default=0, help='Select mode of operation.')
    parser.add_argument("--num_robots", type=int, default=1, help="Number of robots used in the team")
    parser.add_argument("--rejection_penalty", type=int, dest='rejection_penalty', default=28800, help='Penalty for rejecting a request. Default value set to the number of seconds in 8 hours.')

    args = parser.parse_args()

    random_seed = 42

    experiment(args, random_seed)
    

if __name__ == "__main__":
    pStart = datetime.now()
    try:
        main()
    except Exception as errorMainContext:
        print("Fail End Process: ", errorMainContext)
        traceback.print_exc()
    pEnd = datetime.now()
    print(f"Total Execution Time: {pEnd - pStart}")