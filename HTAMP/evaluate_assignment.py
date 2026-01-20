import argparse
import copy
import traceback
import random
import pandas as pd
from datetime import datetime
from typing import Optional

from HTAMP.assignment.baselines.D_TPTS import DeadlineAwareTokenPassingwithTaskSwaps
from HTAMP.assignment.baselines.TP_D import TokenPassingWithDeadlines
from HTAMP.assignment.baselines.idle_pred import IdleTaskPrediction
from HTAMP.assignment.policies.adaptive_rollout import AdaptiveRollout
from HTAMP.assignment.policies.greedy_reopt import GreedyPolicyWithReoptimization
from HTAMP.assignment.policies.vanilla_rollout import VanillaRollout
from HTAMP.assignment.policies.sequential_greedy import SequentialGreedy
from HTAMP.data_processing.processing_dataclasses import AnnotatedDataFiles
from HTAMP.environment.grid_world import GridWorld
from HTAMP.environment.robot_dataclasses import RobotProfile
from HTAMP.environment.traversal_dataclasses import TraversalNode
from HTAMP.environment.traversal_graph_gen import TraversalGraphGenerator
from HTAMP.planning.motion_planner import MotionPlanner
from HTAMP.planning.planning_dataclasses import AllTaskProperties, DateStamp, FrameData, RequestsLists, SimulatorConfig, TaskProperties, TaskRequest, TimeSignal
from HTAMP.planning.request_handler import DailyRequestHandler
from HTAMP.planning.state import PlanningState
from HTAMP.plotting.motion_planning_plotting import MotionPlanningPlotter
from HTAMP.assignment.baselines.fleet_manager import FleetManager

class AssignmentEvaluator:
    def __init__(self, 
                 args, 
                 robot_profiles: list[RobotProfile], 
                 annotated_data_files: AnnotatedDataFiles, 
                 all_task_properties: AllTaskProperties, 
                 random_seed=None):
        self.args = args
        self.random_seed = random_seed
        self.date_stamp = DateStamp(year=args.year, month=args.month, day=args.day)
        self.floor_number = args.floor_number
        self.robot_profiles = robot_profiles
        self.annotated_data_files = annotated_data_files
        self.all_task_properties = all_task_properties
        print("Generating Traversal Graph...")
        self._initialize_traversal_graph_generator()
        self._initialize_grid_world()
        self._initialize_motion_planner()
        self._initialize_simulator_config()
        self.state = PlanningState(simulator_config=self.simulator_config)
        self.policy: FleetManager | TokenPassingWithDeadlines | \
                     DeadlineAwareTokenPassingwithTaskSwaps | IdleTaskPrediction | \
                     SequentialGreedy | GreedyPolicyWithReoptimization | \
                     VanillaRollout | AdaptiveRollout = self._initialize_policy(args.mode)
    
    def _initialize_traversal_graph_generator(self):
        print("Generating Traversal Graph...")
        self.tg_generator = TraversalGraphGenerator(occupancy_map_path=self.args.occupancy_map_path,
                                            config_path=self.args.config_path,
                                            meters_per_pixel=self.args.meters_per_pixel,
                                            factor=self.args.factor)
    
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
                                       initial_time=pd.Timestamp(year=self.date_stamp.year, month=self.date_stamp.month, day=self.date_stamp.day, hour=self.args.hour_start, minute=0),
                                       initial_robot_positions={i: self.selected_start_nodes[i].position for i in range(self.args.num_robots)},
                                       horizon=5000.0)
    
    def _get_request_handler(self,
                             start_date: str,
                             end_date: str,
                             annotated_data_files: AnnotatedDataFiles,
                             request_dir: Optional[str] = None,
                             use_saved_request_data: bool = False) -> DailyRequestHandler:
        request_handler = DailyRequestHandler(start_date=start_date,
                                              end_date=end_date,
                                              date_stamp=self.date_stamp,
                                              floor_number=self.floor_number,
                                              annotated_data_files=annotated_data_files,
                                              request_dir=request_dir,
                                              use_saved_data=use_saved_request_data)
        return request_handler
    
    def _initialize_policy(self, mode: int) -> FleetManager | TokenPassingWithDeadlines | \
                                                DeadlineAwareTokenPassingwithTaskSwaps | IdleTaskPrediction | \
                                                    SequentialGreedy | GreedyPolicyWithReoptimization | \
                                                        VanillaRollout | AdaptiveRollout:
        if mode == 0:
            policy = FleetManager()
        elif mode == 1:
            policy = TokenPassingWithDeadlines()
        elif mode == 2:
            policy = DeadlineAwareTokenPassingwithTaskSwaps()
        elif mode == 3:
            policy = IdleTaskPrediction()
        elif mode == 4:
            policy = SequentialGreedy()
        elif mode == 5:
            policy = GreedyPolicyWithReoptimization()
        elif mode == 6:
            policy = VanillaRollout()
        elif mode == 7:
            policy = AdaptiveRollout()
        else:
            raise ValueError(f"Invalid mode {mode} selected for policy initialization.")
        return policy
    
    def _add_requests_to_state(self, requests_lists: RequestsLists):
        requests: list[TaskRequest] = []
        for request_list in [requests_lists.blood_pressure_requests,
                             requests_lists.heart_rate_requests,
                             requests_lists.respiratory_rate_requests,
                             requests_lists.temperature_requests,
                             requests_lists.oxygen_saturation_requests,
                             requests_lists.medications_requests]:
            requests.extend(request_list)
        self.state.add_new_requests(requests=requests)
    
    def evaluate_assignment(self, 
                            start_date: str,
                            end_date: str,
                            hour_range: Optional[tuple[int, int]],
                            request_dir: Optional[str] = None,
                            use_saved_request_data: bool = False,
                            save_frame_data: bool = False,
                            look_ahead_minutes: int = 60) -> tuple[FrameData, float, int, int, int]:
        request_handler = self._get_request_handler(start_date=start_date,
                                                    end_date=end_date,
                                                    annotated_data_files=self.annotated_data_files,
                                                    request_dir=request_dir,
                                                    use_saved_request_data=use_saved_request_data)
        if save_frame_data:
            frame_data = FrameData()
        else:
            frame_data = None
        
        if hour_range is not None:
            start_hour, end_hour = hour_range
        else:
            start_hour, end_hour = 0, 24

        for hour in range(start_hour, end_hour):
            for minute in range(60):
                time_signal = TimeSignal(year=self.date_stamp.year,
                                         month=self.date_stamp.month,
                                         day=self.date_stamp.day,
                                         hour=hour,
                                         minute=minute)

                requests_lists: RequestsLists = request_handler.get_all_requests_for_time_signal(time_signal=time_signal,
                                                                           all_task_properties=self.all_task_properties,
                                                                           look_ahead_minutes=look_ahead_minutes)
                
                self._add_requests_to_state(requests_lists=requests_lists)

                self.policy.assign_requests_to_robots(state=self.state,
                                                      requests_lists=requests_lists,
                                                      motion_planner=self.motion_planner)

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
        
        total_cost = self.state.compute_total_costs_for_completed_requests()
        rejected_requests = self.state.get_rejected_requests()
        number_of_rejections = len(rejected_requests)
        completed_requests = self.state.get_completed_requests()
        number_of_completed_requests = len(completed_requests)
        total_number_of_requests = len(list(self.state.requests.keys()))
        return frame_data, total_cost, number_of_completed_requests, number_of_rejections, total_number_of_requests

class Experiment():

    def __init__(self,
                 args,
                 start_date: str,
                 end_date: str,
                 random_seed: Optional[int] = None):
        self.start_date = start_date
        self.end_date = end_date
        robot_profiles = self._generate_robot_profiles(num_type_1_robots=args.num_type_1_robots,
                                                       num_type_2_robots=args.num_type_2_robots,
                                                       num_type_3_robots=args.num_type_3_robots,
                                                       num_type_4_robots=args.num_type_4_robots)
        self.all_task_properties = self._generate_task_properties()
        self.evaluator = AssignmentEvaluator(args, random_seed=random_seed, robot_profiles=robot_profiles)
        annotated_data_files = AnnotatedDataFiles(
            annotated_visits=None,
            annotated_admissions_discharges=None,
            annotated_medications=args.medications_orders_file,
            annotated_blood_pressure=args.blood_pressure_orders_file,
            annotated_heart_rate=args.heart_rate_orders_file,
            annotated_respiratory_rate=args.respiratory_rate_orders_file,
            annotated_temperature=args.temperature_orders_file,
            annotated_oxygen_saturation=args.oxygen_saturation_orders_file,
        )
    
    def _generate_robot_profiles(self,
                                 num_type_1_robots: int,
                                 num_type_2_robots: int,
                                 num_type_3_robots: int,
                                 num_type_4_robots: int) -> list[RobotProfile]:
        # type 1 robots: heart rate + SPO2
        # type 2 robots: blood pressure + heart rate
        # type 3 robots: respiratory rate + temperature
        # type 4 robots: medications

        robot_profiles = []
        for i in range(num_type_1_robots):
            robot_profile = RobotProfile(radius=0.10, speed=0.20, robot_id=i, robot_type="type_1")
            robot_profiles.append(robot_profile)
        for i in range(num_type_2_robots):
            robot_profile = RobotProfile(radius=0.10, speed=0.20, robot_id=num_type_1_robots + i, robot_type="type_2")
            robot_profiles.append(robot_profile)
        for i in range(num_type_3_robots):
            robot_profile = RobotProfile(radius=0.10, speed=0.20, robot_id=num_type_1_robots + num_type_2_robots + i, robot_type="type_3")
            robot_profiles.append(robot_profile)
        for i in range(num_type_4_robots):
            robot_profile = RobotProfile(radius=0.10, speed=0.20, robot_id=num_type_1_robots + num_type_2_robots + num_type_3_robots + i, robot_type="type_4")
            robot_profiles.append(robot_profile)
        return robot_profiles
    
    def _generate_task_properties(self) -> AllTaskProperties:
        blood_pressure_properties = TaskProperties(
            task_type="blood_pressure",
            wait_time_seconds=30.0,
            time_for_rejection_minutes=30.0
        )

        heart_rate_properties = TaskProperties(
            task_type="heart_rate",
            wait_time_seconds=30.0,
            time_for_rejection_minutes=30.0
        )

        respiratory_rate_properties = TaskProperties(
            task_type="respiratory_rate",
            wait_time_seconds=30.0,
            time_for_rejection_minutes=30.0
        )

        temperature_properties = TaskProperties(
            task_type="temperature",
            wait_time_seconds=30.0,
            time_for_rejection_minutes=30.0
        )

        oxygen_saturation_properties = TaskProperties(
            task_type="oxygen_saturation",
            wait_time_seconds=30.0,
            time_for_rejection_minutes=30.0
        )

        medications_properties = TaskProperties(
            task_type="medication",
            wait_time_seconds=60.0,
            time_for_rejection_minutes=60.0
        )

        all_task_properties = AllTaskProperties(
            blood_pressure=blood_pressure_properties,
            heart_rate=heart_rate_properties,
            respiratory_rate=respiratory_rate_properties,
            temperature=temperature_properties,
            oxygen_saturation=oxygen_saturation_properties,
            medications=medications_properties
        )

        return all_task_properties


def run_experiment(args):
    start_date="2024-06-24"
    end_date="2025-06-29"

    experiment = Experiment(args,
                            start_date=start_date,
                            end_date=end_date)

    evaluate_results = experiment.evaluator.evaluate_assignment(start_date=start_date,
                                                     end_date=end_date,
                                                     hour_range=(args.hour_start,args.hour_end),
                                                     request_dir=args.request_dir,
                                                     use_saved_request_data=args.use_saved_request_data,
                                                     save_frame_data=False)
    
    frame_data, total_cost, number_of_completed_requests, number_of_rejections, total_number_of_requests = evaluate_results

    if frame_data is not None:
        MotionPlanningPlotter.generate_state_animation(occupancy_map=experiment.evaluator.tg_generator.occupancy_map,
                                            origin_x=experiment.evaluator.tg_generator.origin_x,
                                            origin_y=experiment.evaluator.tg_generator.origin_y,
                                            resolution=experiment.evaluator.tg_generator.meters_per_cell,
                                            robot_positions_seq=frame_data.robot_positions_seq,
                                            robots_current_node_index_seq=frame_data.robots_current_node_index_seq,
                                            point_indices_on_edge_seq=frame_data.point_indices_on_edge_seq,
                                            robot_paths_seq=frame_data.robot_paths_seq,
                                            planned_goal_indices_seq=frame_data.planned_goal_indices_seq,
                                            completed_goals_seq=frame_data.completed_goals_seq,
                                            traversal_graph=experiment.evaluator.tg_generator.traversal_graph,
                                            robot_profiles=experiment.evaluator.robot_profiles,
                                            fps_sim=args.fps,
                                            num_sim_frames=1000)

def main():
    parser = argparse.ArgumentParser(prog='evaluate_assignment.py',
                                     description='Evaluate assignment algorithms in a hospital floor environment.')
    # date_operational_range parameters
    parser.add_argument("--year", type=int, dest='year', default=2022, help='Select year of interest.')
    parser.add_argument("--month", type=int, dest='month', default=10, help='Select month of interest.')
    parser.add_argument("--day", type=int, dest='day', default=17, help='Select day of interest.')
    parser.add_argument("--hour_start", type=int, dest='hour_start', default=0, help='Select starting hour of operational range.')
    parser.add_argument("--hour_end", type=int, dest='hour_end', default=24, help='Select ending hour of operational range.')
    parser.add_argument("--floor_number", type=int, dest='floor_number', default=9, help='Select floor number of interest.')

    # file parameters
    parser.add_argument("--request_dir", type=str, default="data/requests", help="Directory to save global requests data.")
    parser.add_argument("--use_saved_request_data", action='store_true', help="Flag to use previously saved request data.")
    parser.add_argument("--medications_orders_file", type=str, default="data/processed/medication_orders_annotated.csv", help="Path to the medications orders CSV file.")
    parser.add_argument("--blood_pressure_orders_file", type=str, default="data/processed/blood_pressure_orders_annotated.csv", help="Path to the blood pressure orders CSV file.")
    parser.add_argument("--heart_rate_orders_file", type=str, default="data/processed/heart_rate_orders_annotated.csv", help="Path to the heart rate orders CSV file.")
    parser.add_argument("--respiratory_rate_orders_file", type=str, default="data/processed/respiratory_rate_orders_annotated.csv", help="Path to the respiratory rate orders CSV file.")
    parser.add_argument("--temperature_orders_file", type=str, default="data/processed/temperature_orders_annotated.csv", help="Path to the temperature orders CSV file.")
    parser.add_argument("--oxygen_saturation_orders_file", type=str, default="data/processed/oxygen_saturation_orders_annotated.csv", help="Path to the oxygen saturation orders CSV file.")

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
    parser.add_argument("--num_type_1_robots", type=int, default=1, help="Number of robots of type 1 to be used in the team")
    parser.add_argument("--num_type_2_robots", type=int, default=1, help="Number of robots of type 2 to be used in the team")
    parser.add_argument("--num_type_3_robots", type=int, default=1, help="Number of robots of type 3 to be used in the team")
    parser.add_argument("--num_type_4_robots", type=int, default=1, help="Number of robots of type 4 to be used in the team")
    parser.add_argument("--rejection_penalty", type=int, dest='rejection_penalty', default=28800, help='Penalty for rejecting a request. Default value set to the number of seconds in 8 hours.')

    args = parser.parse_args()

    run_experiment(args)
    

if __name__ == "__main__":
    pStart = datetime.now()
    try:
        main()
    except Exception as errorMainContext:
        print("Fail End Process: ", errorMainContext)
        traceback.print_exc()
    pEnd = datetime.now()
    print(f"Total Execution Time: {pEnd - pStart}")