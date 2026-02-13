import argparse
import copy
import os
import traceback
import random
import pandas as pd
from datetime import datetime
from typing import Optional

from HTAMP.assignment.policies.sequential_greedy import SequentialGreedy
from HTAMP.data_processing.processing_dataclasses import AnnotatedDataFiles
from HTAMP.environment.grid_world import GridWorld
from HTAMP.environment.robot_dataclasses import RobotProfile
from HTAMP.environment.traversal_dataclasses import TraversalNode
from HTAMP.environment.traversal_graph_gen import TraversalGraphGenerator
from HTAMP.planning.motion_planner import MotionPlanner
from HTAMP.planning.planning_dataclasses import AllTaskProperties, DateStamp, RequestsLists, SimulatorConfig, TaskProperties, TaskRequest, TimeSignal
from HTAMP.planning.request_handler import DailyRequestHandler
from HTAMP.planning.state import PlanningState
from HTAMP.plotting.motion_planning_plotting import MotionPlanningPlotter

class StabilityEvaluator:
    def __init__(self, 
                 args, 
                 team_size: int,
                 robot_profiles: dict[int, RobotProfile], 
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
        self.team_size = team_size
        self._initialize_traversal_graph_generator()
        self._initialize_grid_world()
        self._initialize_simulator_config()
        self._initialize_motion_planner()
        self.state = PlanningState(simulator_config=self.simulator_config)
        self.policy: SequentialGreedy = self._initialize_policy()
    
    def _initialize_traversal_graph_generator(self):
        print("Generating Traversal Graph...")
        self.tg_generator = TraversalGraphGenerator(occupancy_map_path=self.args.occupancy_map_path,
                                            config_path=self.args.config_path,
                                            meters_per_pixel=self.args.meters_per_pixel,
                                            factor=self.args.factor)
        print("Traversal Graph generated.")
    
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

    def _initialize_simulator_config(self):
        print("Initializing Simulator Config...")
        self._generate_parking_positions()
        self.simulator_config = SimulatorConfig(fps=int(self.args.fps),
                                       robot_profiles=self.robot_profiles,
                                       rejection_penalty=100.0,
                                       initial_time=pd.Timestamp(year=self.date_stamp.year, month=self.date_stamp.month, day=self.date_stamp.day, hour=self.args.hour_start, minute=0),
                                       initial_robot_positions={i: copy.deepcopy(self.selected_start_nodes[i].position) for i in range(self.team_size)},
                                       initial_nodes={i: copy.deepcopy(self.selected_start_nodes[i]) for i in range(self.team_size)},
                                       horizon=((self.args.hour_end+1) * 3600.0) - (self.args.hour_start * 3600.0))
        print("Simulator Config initialized.")
    
    def _initialize_motion_planner(self):
        print("Initializing Motion Planner...")
        self.motion_planner = MotionPlanner(grid=self.world, weight_factor=1.0)
        for i in range(self.team_size):
            self.motion_planner._initialize_robot_reservations(initial_node=self.simulator_config.initial_nodes[i],
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
        
        self.selected_start_nodes: list[TraversalNode] = random.sample(potential_start_nodes, self.team_size)
    
    def _get_request_handler(self,
                             start_date: str,
                             end_date: str,
                             annotated_data_files: AnnotatedDataFiles,
                             request_dir: Optional[str] = None,
                             use_saved_request_data: bool = False) -> DailyRequestHandler:
        request_handler = DailyRequestHandler(start_date=start_date,
                                              end_date=end_date,
                                              date_stamp=self.date_stamp.time_stamp,
                                              floor_number=self.floor_number,
                                              annotated_data_files=annotated_data_files,
                                              request_dir=request_dir,
                                              use_saved_data=use_saved_request_data)
        return request_handler
    
    def _initialize_policy(self) -> SequentialGreedy:
        print("Initializing Sequential Greedy policy...")
        policy = SequentialGreedy(allow_deallocation=True)
        print("Policy initialized.")
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
    
    def _extract_requests_for_hour_minute(self,
                                         hour: int,
                                         minute: int,
                                         request_handler: DailyRequestHandler,
                                         look_ahead_minutes: int = 60) -> RequestsLists:
        
        time_signal = TimeSignal(year=self.date_stamp.year,
                                 month=self.date_stamp.month,
                                 day=self.date_stamp.day,
                                 hour=hour,
                                 minute=minute)

        requests_lists: RequestsLists = request_handler.get_all_requests_for_time_signal(time_signal=time_signal,
                                                                                         initial_time=self.simulator_config.initial_time,
                                                                                         all_task_properties=self.all_task_properties,
                                                                                         look_ahead_minutes=look_ahead_minutes,
                                                                                         traversal_graph_generator=self.tg_generator)
        
        self._add_requests_to_state(requests_lists=requests_lists)

        print("Extracted Requests:")
        print(f"Blood Pressure Requests: {len(requests_lists.blood_pressure_requests)}")
        print(f"Heart Rate Requests: {len(requests_lists.heart_rate_requests)}")
        print(f"Respiratory Rate Requests: {len(requests_lists.respiratory_rate_requests)}")
        print(f"Temperature Requests: {len(requests_lists.temperature_requests)}")
        print(f"Oxygen Saturation Requests: {len(requests_lists.oxygen_saturation_requests)}")
        print(f"Medications Requests: {len(requests_lists.medications_requests)}")

        return requests_lists
    
    def _check_if_requests_are_rejected(self) -> Optional[str]:
        for request in self.state.requests.values():
            if request.rejected:
                return request.request_type
        return None

    
    def _assign_requests_and_step_simulator(self,
                                            requests_lists: Optional[RequestsLists] = None,
                                            debug: bool = False) -> Optional[str]:
        for second in range(60):
            if second == 0:
                self.policy.assign_requests_to_robots(state=self.state,
                                              requests_lists=requests_lists,
                                              motion_planner=self.motion_planner,
                                              traversal_graph_generator=self.tg_generator,
                                              debug=debug)
                rejected_type = self._check_if_requests_are_rejected()
                if rejected_type is not None:
                    return rejected_type
            else:
                self.policy.assign_requests_to_robots(state=self.state,
                                              requests_lists=None,
                                              motion_planner=self.motion_planner,
                                              traversal_graph_generator=self.tg_generator,
                                              debug=debug)
            for _ in range(int(self.args.fps)):
                self.state.step(self.tg_generator.traversal_graph)
        
        return None
    
    def _generate_assignment_for_minute(self, 
                                        hour, 
                                        minute, 
                                        request_handler: DailyRequestHandler,
                                        look_ahead_minutes: int = 60,
                                        debug = False) -> Optional[str]:
        
        requests_lists = self._extract_requests_for_hour_minute(hour=hour,
                                                              minute=minute,
                                                              request_handler=request_handler,
                                                              look_ahead_minutes=look_ahead_minutes)
        
        rejected_type = self._assign_requests_and_step_simulator(requests_lists=requests_lists,
                                                                debug=debug)
        
        return rejected_type
    
    def evaluate_assignment(self, 
                            start_date: str,
                            end_date: str,
                            hour_range: Optional[tuple[int, int]],
                            request_dir: Optional[str] = None,
                            use_saved_request_data: bool = False,
                            save_frame_data: bool = False,
                            look_ahead_minutes: int = 30,
                            debug = False) -> Optional[str]:
        request_handler = self._get_request_handler(start_date=start_date,
                                                    end_date=end_date,
                                                    annotated_data_files=self.annotated_data_files,
                                                    request_dir=request_dir,
                                                    use_saved_request_data=use_saved_request_data)
        start_hour, end_hour = hour_range
        for hour in range(start_hour, end_hour):
            for minute in range(60):
                reject_type = self._generate_assignment_for_minute(hour=hour,
                                                     minute=minute,
                                                     request_handler=request_handler,
                                                     look_ahead_minutes=look_ahead_minutes, 
                                                     debug=debug)
                if reject_type is not None:
                    return reject_type
                
        for minute in range(60):
            reject_type = self._assign_requests_and_step_simulator(requests_lists=None,
                                                     save_frame_data=save_frame_data,
                                                     debug=debug)
            if reject_type is not None:
                return reject_type
        
        return None
    
class StabilityExperiment():

    def __init__(self,
                 args,
                 start_date: str,
                 end_date: str):
        self.start_date = start_date
        self.end_date = end_date
        self.all_task_properties = self._generate_task_properties()
        self.annotated_data_files = AnnotatedDataFiles(
            annotated_visits=None,
            annotated_admissions_discharges=None,
            annotated_medications=args.medications_orders_file,
            annotated_blood_pressure=args.blood_pressure_orders_file,
            annotated_heart_rate=args.heart_rate_orders_file,
            annotated_respiratory_rate=args.respiratory_rate_orders_file,
            annotated_temperature=args.temperature_orders_file,
            annotated_oxygen_saturation=args.oxygen_saturation_orders_file,
        )
    
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
            time_for_rejection_minutes=30.0
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
    
    def _generate_robot_profiles(self,
                                 num_monitoring_robots: int,
                                 num_delivery_robots: int) -> dict[int, RobotProfile]:
        # monitoring robots: heart rate + SPO2 + blood pressure + respiratory rate + temperature
        # delivery robots: medications

        robot_profiles = {}
        for i in range(num_monitoring_robots):
            robot_profile = RobotProfile(radius=0.10, speed=0.20, robot_id=i, robot_type="monitoring")
            robot_profiles[i] = robot_profile
        for j in range(num_delivery_robots):
            robot_profile = RobotProfile(radius=0.10, speed=0.20, robot_id=num_monitoring_robots + j, robot_type="delivery")
            robot_profiles[num_monitoring_robots + j] = robot_profile
        return robot_profiles


def run_experiment(args):
    start_date="2024-06-24"
    end_date="2025-06-29"

    random_seed = 42

    experiment = StabilityExperiment(args,
                            start_date=start_date,
                            end_date=end_date)

def main():
    parser = argparse.ArgumentParser(prog='evaluate_assignment.py',
                                     description='Evaluate assignment algorithms in a hospital floor environment.')
    # date_operational_range parameters
    parser.add_argument("--year", type=int, dest='year', default=2024, help='Select year of interest.')
    parser.add_argument("--month", type=int, dest='month', default=6, help='Select month of interest.')
    parser.add_argument("--day", type=int, dest='day', default=24, help='Select day of interest.')
    parser.add_argument("--hour_start", type=int, dest='hour_start', default=8, help='Select starting hour of operational range.')
    parser.add_argument("--hour_end", type=int, dest='hour_end', default=9, help='Select ending hour of operational range.')
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
    parser.add_argument("--policy_name", type=str, dest='policy_name', default='fleet_manager', help='Name of the policy for saving results.')
    parser.add_argument("--num_monitoring_robots", type=int, default=6, help="Number of monitoring robots to be used in the team")
    parser.add_argument("--num_delivery_robots", type=int, default=3, help="Number of delivery robots to be used in the team")
    parser.add_argument("--rejection_penalty", type=int, dest='rejection_penalty', default=28800, help='Penalty for rejecting a request. Default value set to the number of seconds in 8 hours.')

    # Baseline parameters
    parser.add_argument("--alpha", type=float, dest='alpha', default=0.1, help='Alpha parameter for token passing policies.')

    # Policies parameters
    parser.add_argument("--allow_deallocation", action='store_true', help="Whether to allow deallocation of previously assigned requests in the Sequential Greedy policy.")

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