import argparse
import copy
import os
import traceback
import random
import pandas as pd
from datetime import datetime
from typing import Optional

from HTAMP.assignment.baselines.D_TPTS import DeadlineAwareTokenPassingwithTaskSwaps
from HTAMP.assignment.baselines.TP_D import TokenPassingWithDeadlines
from HTAMP.assignment.baselines.idle_pred import IdleTaskPrediction
from HTAMP.assignment.policies.adaptive_rollout import AdaptiveRollout
from HTAMP.assignment.policies.base_policy import BasePolicy
from HTAMP.data_processing.processing_dataclasses import AnnotatedDataFiles
from HTAMP.environment.grid_world import GridWorld
from HTAMP.environment.robot_dataclasses import RobotProfile
from HTAMP.environment.traversal_dataclasses import TraversalNode
from HTAMP.environment.traversal_graph_gen import TraversalGraphGenerator
from HTAMP.planning.motion_planner import MotionPlanner
from HTAMP.planning.planning_dataclasses import AllTaskProperties, DateStamp, FrameData, RequestsLists, SimulatorConfig, TaskProperties, TaskRequest, TimeSignal
from HTAMP.planning.request_handler import DailyRequestHandler
from HTAMP.planning.state import PlanningState
from HTAMP.plotting.planning.motion_planning_plotting import MotionPlanningPlotter
from HTAMP.assignment.baselines.fleet_manager import FleetManager

class AssignmentEvaluator:
    def __init__(self, 
                 args, 
                 start_date: str,
                 end_date: str,
                 robot_profiles: dict[int, RobotProfile], 
                 annotated_data_files: AnnotatedDataFiles, 
                 all_task_properties: AllTaskProperties, 
                 random_seed=None):
        self.args = args
        self.start_date = start_date
        self.end_date = end_date
        self.team_size = args.num_monitoring_robots + args.num_delivery_robots
        self.random_seed = random_seed
        self.date_stamp = DateStamp(year=args.year, month=args.month, day=args.day)
        self.floor_number = args.floor_number
        self.robot_profiles = robot_profiles
        self.annotated_data_files = annotated_data_files
        self.all_task_properties = all_task_properties
        self._initialize_traversal_graph_generator()
        self._initialize_grid_world()
        self._initialize_simulator_config()
        self._initialize_motion_planner()
        self.state = PlanningState(simulator_config=self.simulator_config)
        self.policy: FleetManager | TokenPassingWithDeadlines | \
                     DeadlineAwareTokenPassingwithTaskSwaps | IdleTaskPrediction | \
                     BasePolicy | AdaptiveRollout = self._initialize_policy(mode=args.mode, 
                                                                            alpha=args.alpha)
    
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
                                       rejection_penalty=self.args.rejection_penalty,
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
                             annotated_data_files: AnnotatedDataFiles,
                             request_dir: Optional[str] = None,
                             use_saved_request_data: bool = False) -> DailyRequestHandler:
        request_handler = DailyRequestHandler(start_date=self.start_date,
                                              end_date=self.end_date,
                                              date_stamp=self.date_stamp.time_stamp,
                                              floor_number=self.floor_number,
                                              annotated_data_files=annotated_data_files,
                                              request_dir=request_dir,
                                              use_saved_data=use_saved_request_data)
        return request_handler
    
    def _initialize_policy(self, mode: int, alpha: float) -> FleetManager | TokenPassingWithDeadlines | \
                                                DeadlineAwareTokenPassingwithTaskSwaps | IdleTaskPrediction | \
                                                    BasePolicy | AdaptiveRollout:
        print("Initializing Policy...")
        print(f"Selected mode: {mode}")
        policy_dict = {
            0: FleetManager,
            1: TokenPassingWithDeadlines,
            2: DeadlineAwareTokenPassingwithTaskSwaps,
            3: IdleTaskPrediction,
            4: BasePolicy,
            5: AdaptiveRollout,
            6: AdaptiveRollout,
            7: AdaptiveRollout,
            8: AdaptiveRollout,
        }
        print(f"Selected policy: {str(policy_dict[mode].__name__)}")
        
        if mode not in policy_dict:
            raise ValueError(f"Invalid mode {mode} selected for policy initialization.")
        
        if mode in [1, 2]:  # If the selected policy is one of the token passing variants that use alpha
            policy = policy_dict[mode](alpha=alpha)
            print(f"Initialized policy with alpha: {alpha}")
        elif mode in [5, 6, 7, 8]:
            if mode == 5:
                allow_deallocation = True
                allow_reweighting = True
            elif mode == 6:
                allow_deallocation = True
                allow_reweighting = False
            elif mode == 7:
                allow_deallocation = False
                allow_reweighting = True
            else:  # mode == 8
                allow_deallocation = False
                allow_reweighting = False
            policy = policy_dict[mode](start_date=self.start_date, 
                                       end_date=self.end_date,
                                       date_stamp=self.date_stamp.time_stamp,
                                       end_hour=self.args.hour_end,
                                       floor_number=self.floor_number,
                                       annotated_data_files=self.annotated_data_files,
                                       request_dir=self.args.request_dir,
                                       use_saved_request_data=self.args.use_saved_request_data,
                                       initial_time=self.simulator_config.initial_time,
                                       all_task_properties=self.all_task_properties,
                                       allow_deallocation=allow_deallocation,
                                       allow_reweighting=allow_reweighting)
            print(f"Initialized {str(policy_dict[mode].__name__)} policy with contextual information.")
        else:
            policy = policy_dict[mode]()
            print(f"Initialized {str(policy_dict[mode].__name__)} policy without additional parameters.")
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
    
    def _store_frame_data(self, frame_data: FrameData):
        frame_data.robot_positions_seq.append(copy.deepcopy(self.state.robots_positions))
        frame_data.robots_current_node_index_seq.append(copy.deepcopy(self.state.robots_current_node_index))
        frame_data.point_indices_on_edge_seq.append(copy.deepcopy(self.state.point_indices_on_edge))
        frame_data.robot_paths_seq.append(copy.deepcopy(self.state.robot_paths))
        planned_goal_indices_dict: dict[int, list[int]] = {}
        completed_goals_dict: dict[int, int] = {}
        for robot_id, requests in self.state.assigned_requests.items():
            if requests:
                current_request_id = requests[0]
                current_request = self.state.requests[current_request_id]
                planned_goal_indices_dict[robot_id] = current_request.planned_goal_indices
                completed_goals_dict[robot_id] = current_request.completed_goals
        frame_data.planned_goal_indices_seq.append(copy.deepcopy(planned_goal_indices_dict))
        frame_data.completed_goals_seq.append(copy.deepcopy(completed_goals_dict))

    def _plot_state_debug(self, before_assignment: bool = False):
        planned_goal_indices_dict: dict[int, list[int]] = {}
        for robot_id, requests in self.state.assigned_requests.items():
            if requests:
                current_request_id = requests[0]
                current_request = self.state.requests[current_request_id]
                planned_goal_indices_dict[robot_id] = current_request.planned_goal_indices
        MotionPlanningPlotter.plot_state_debug(
            occupancy_map=self.tg_generator.occupancy_map,
            origin_x=self.tg_generator.origin_x,
            origin_y=self.tg_generator.origin_y,
            resolution=self.tg_generator.meters_per_cell,
            robot_positions=self.state.robots_positions,
            robots_current_node_index=self.state.robots_current_node_index,
            point_indices_on_edge=self.state.point_indices_on_edge,
            robot_paths=self.state.robot_paths,
            planned_goal_indices=planned_goal_indices_dict,
            traversal_graph=self.tg_generator.traversal_graph,
            robot_profiles=self.simulator_config.robot_profiles,
            step_number=self.state.simulator_time,
            debug_folder="results/motion_planning/steps",
            before_assignment=before_assignment
        )
    
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
    
    def _assign_requests_to_robots(self, 
                                   hour: int,
                                   minute: int,
                                   look_ahead_minutes: int,
                                   requests_lists: Optional[RequestsLists] = None, 
                                   debug: bool = False):
        if self.args.mode in [5, 6, 7, 8]:  # If the selected policy is one of the rollout variants that use contextual information
            self.policy.assign_requests_to_robots(state=self.state,
                                              requests_lists=requests_lists,
                                              motion_planner=self.motion_planner,
                                              traversal_graph_generator=self.tg_generator,
                                              hour=hour,
                                              minute=minute,
                                              look_ahead_minutes=look_ahead_minutes,
                                              debug=debug)
        else:
            self.policy.assign_requests_to_robots(state=self.state,
                                              requests_lists=requests_lists,
                                              motion_planner=self.motion_planner,
                                              traversal_graph_generator=self.tg_generator,
                                              debug=debug)
    
    def _assign_requests_and_step_simulator(self,
                                            hour: int,
                                            minute: int,
                                            look_ahead_minutes: int,
                                            requests_lists: Optional[RequestsLists] = None,
                                            frame_data: Optional[FrameData] = None,
                                            save_frame_data: bool = False,
                                            debug: bool = False):
        for second in range(60):
            if second == 0:
                self._assign_requests_to_robots(requests_lists=requests_lists, 
                                                hour=hour,
                                                minute=minute,
                                                look_ahead_minutes=look_ahead_minutes,
                                                debug=debug)
            else:
                self._assign_requests_to_robots(requests_lists=None, 
                                                hour=hour,
                                                minute=minute,
                                                look_ahead_minutes=look_ahead_minutes,
                                                debug=debug)
            for frames in range(int(self.args.fps)):
                self.state.step(self.tg_generator.traversal_graph)
                if save_frame_data and frame_data is not None:
                    self._store_frame_data(frame_data=frame_data)
    
    def _generate_assignment_for_minute(self, 
                                        hour: int,
                                        minute: int,
                                        request_handler: DailyRequestHandler,
                                        frame_data: Optional[FrameData] = None,
                                        save_frame_data: bool = False,
                                        look_ahead_minutes: int = 60,
                                        debug: bool = False):
        
        requests_lists = self._extract_requests_for_hour_minute(hour=hour,
                                                              minute=minute,
                                                              request_handler=request_handler,
                                                              look_ahead_minutes=look_ahead_minutes)
        
        self._assign_requests_and_step_simulator(hour=hour,
                                                 minute=minute,
                                                 look_ahead_minutes=look_ahead_minutes,
                                                 requests_lists=requests_lists,
                                                 frame_data=frame_data,
                                                 save_frame_data=save_frame_data,
                                                 debug=debug)
        
        

    
    def _generate_results_summary(self) -> pd.DataFrame:
        results_df = pd.DataFrame([r.to_dict() for r in self.state.requests.values()])

        total_cost = self.state.compute_total_costs_for_completed_requests()
        rejected_requests = self.state.get_rejected_requests()
        number_of_rejected_requests = 0
        number_rejected_requests_dict = {}
        for request_type, request_list in rejected_requests.items():
            number_of_rejected_requests += len(request_list)
            number_rejected_requests_dict[request_type] = len(request_list)
        print(f"Rejected Requests: {rejected_requests}")
        completed_requests = self.state.get_completed_requests()
        number_of_completed_requests = 0
        number_completed_requests_dict = {}
        for request_type, request_list in completed_requests.items():
            number_of_completed_requests += len(request_list)
            number_completed_requests_dict[request_type] = len(request_list)
        total_number_of_requests = len(list(self.state.requests.keys()))

        print(f"Number of Completed Requests: {number_of_completed_requests}")
        print(f"Number of Rejected Requests: {number_of_rejected_requests}")
        print(f"Total Number of Requests: {total_number_of_requests}")
        print(f"Number of Rejected Requests by Type: {number_rejected_requests_dict}")
        print(f"Number of Completed Requests by Type: {number_completed_requests_dict}")
        print(f"Total Cost: {total_cost}")

        return results_df
    
    def evaluate_assignment(self,
                            hour_range: Optional[tuple[int, int]],
                            request_dir: Optional[str] = None,
                            use_saved_request_data: bool = False,
                            save_frame_data: bool = False,
                            look_ahead_minutes: int = 30,
                            debug = False) -> tuple[FrameData, pd.DataFrame]:
        request_handler = self._get_request_handler(annotated_data_files=self.annotated_data_files,
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
        
        pStart = datetime.now()

        for hour in range(start_hour, end_hour):
            for minute in range(60):
                self._generate_assignment_for_minute(hour=hour,
                                                     minute=minute,
                                                     request_handler=request_handler,
                                                     frame_data=frame_data,
                                                     save_frame_data=save_frame_data,
                                                     look_ahead_minutes=look_ahead_minutes, 
                                                     debug=debug)
                
        for minute in range(60):
            self._assign_requests_and_step_simulator(hour=end_hour,
                                                     minute=minute,
                                                     look_ahead_minutes=look_ahead_minutes,
                                                     requests_lists=None,
                                                     frame_data=frame_data,
                                                     save_frame_data=save_frame_data,
                                                     debug=debug)
        
        pEnd = datetime.now()
        print(f"Total Planning Time: {pEnd - pStart}")

        results_df = self._generate_results_summary()
        
        return frame_data, results_df
    
class Experiment():

    def __init__(self,
                 args,
                 start_date: str,
                 end_date: str,
                 random_seed: Optional[int] = None):
        robot_profiles = self._generate_robot_profiles(num_monitoring_robots=args.num_monitoring_robots,
                                                       num_delivery_robots=args.num_delivery_robots)
        print(f"Generated Robot Profiles: {robot_profiles}")
        all_task_properties = self._generate_task_properties()
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
        self.evaluator = AssignmentEvaluator(args=args, 
                                             start_date=start_date,
                                             end_date=end_date,
                                             robot_profiles=robot_profiles,
                                             annotated_data_files=annotated_data_files,
                                             all_task_properties=all_task_properties,
                                             random_seed=random_seed)
    
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


def run_experiment(args):
    start_date="2024-06-24"
    end_date="2025-06-29"

    random_seed = 42

    experiment = Experiment(args=args,
                            start_date=start_date,
                            end_date=end_date,
                            random_seed=random_seed)
    
    os.makedirs("results/motion_planning/debug", exist_ok=True)
    os.makedirs("results/motion_planning/steps", exist_ok=True)

    frame_data, results_df = experiment.evaluator.evaluate_assignment(hour_range=(args.hour_start,args.hour_end),
                                                                      request_dir=args.request_dir,
                                                                      use_saved_request_data=args.use_saved_request_data,
                                                                      save_frame_data=False,
                                                                      debug=args.debug)
    
    if args.save_results_csv:
        os.makedirs("results/policies", exist_ok=True)
        if args.policy_name in ["tp_d", "d_tpts"]:
            os.makedirs(f"results/policies/{args.policy_name}_alpha{args.alpha}", exist_ok=True)
            results_df.to_csv(f"results/policies/{args.policy_name}_alpha{args.alpha}/{args.year}-{args.month}-{args.day}_floor{args.floor_number}.csv", index=False)
        elif args.policy_name in ["adaptive_rollout"] and args.mode in [5, 6, 7, 8]:
            if args.mode == 5:
                deallocation_str = "ropt"
                reweighting_str = "rwt"
            elif args.mode == 6:
                deallocation_str = "ropt"
                reweighting_str = "norwt"
            elif args.mode == 7:
                deallocation_str = "nopt"
                reweighting_str = "rwt"
            else:  # mode == 8
                deallocation_str = "nopt"
                reweighting_str = "norwt"
            os.makedirs(f"results/policies/{args.policy_name}_{deallocation_str}_{reweighting_str}", exist_ok=True)
            results_df.to_csv(f"results/policies/{args.policy_name}_{deallocation_str}_{reweighting_str}/{args.year}-{args.month}-{args.day}_floor{args.floor_number}.csv", index=False)
        else:
            os.makedirs(f"results/policies/{args.policy_name}", exist_ok=True)
            results_df.to_csv(f"results/policies/{args.policy_name}/{args.year}-{args.month}-{args.day}_floor{args.floor_number}.csv", index=False)

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
                                            fps_sim=int(args.fps),
                                            num_sim_frames=1000)

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
    parser.add_argument("--debug", action='store_true', help="Whether to save debug plots during execution.")
    parser.add_argument("--save_results_csv", action='store_true', help="Whether to save the results summary as a CSV file.")

    # simulation parameters
    parser.add_argument("--mode", type=int, dest='mode', default=0, help='Select mode of operation.')
    parser.add_argument("--policy_name", type=str, dest='policy_name', default='fleet_manager', help='Name of the policy for saving results.')
    parser.add_argument("--num_monitoring_robots", type=int, default=6, help="Number of monitoring robots to be used in the team")
    parser.add_argument("--num_delivery_robots", type=int, default=3, help="Number of delivery robots to be used in the team")
    parser.add_argument("--rejection_penalty", type=int, dest='rejection_penalty', default=28800, help='Penalty for rejecting a request. Default value set to the number of seconds in 8 hours.')

    # Baseline parameters
    parser.add_argument("--alpha", type=float, dest='alpha', default=0.1, help='Alpha parameter for token passing policies.')

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