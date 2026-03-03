import argparse
from datetime import datetime
from pathlib import Path
import traceback

import pandas as pd

from HTAMP.data_processing.data_helpers import DataHelpers
from HTAMP.data_processing.processing_dataclasses import AnnotatedDataFiles, DailyRequestsDataFrames, PreprocessedDataFrames
from HTAMP.environment.traversal_graph_gen import TraversalGraphGenerator
from HTAMP.planning.planning_dataclasses import AllTaskProperties, RequestsLists, TaskProperties, TaskRequest, TimeSignal

class GlobalRequestHandler:

    def __init__(self, annotated_data_files: AnnotatedDataFiles, 
                 request_dir: str, start_date: str, end_date: str, use_saved_data: bool = False):
        self.annotated_data_files = annotated_data_files
        self.use_saved_data = use_saved_data
        self.request_dir = Path(request_dir)
        if not self.use_saved_data:
            self._make_dirs()
            preprocessed_dfs = self._load_preprocessed_data()
            self._extend_dataframes(preprocessed_dfs, start_date=start_date, end_date=end_date)
            self._save_dataframes()
        else:
            self._load_saved_data()

    def _make_dirs(self) -> None:
        self.request_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_preprocessed_data(self) -> PreprocessedDataFrames:
        bp_df = self._load_blood_pressure_data()
        hr_df = self._load_heart_rate_data()
        rr_df = self._load_respiratory_rate_data()
        temp_df = self._load_temperature_data()
        os_df = self._load_oxygen_saturation_data()
        med_df = self._load_medications_data()

        preprocessed_dfs = PreprocessedDataFrames(
            blood_pressure_df=bp_df,
            heart_rate_df=hr_df,
            respiratory_rate_df=rr_df,
            temperature_df=temp_df,
            oxygen_saturation_df=os_df,
            medications_df=med_df
        )

        return preprocessed_dfs

    def _load_blood_pressure_data(self):
        df = pd.read_csv(self.annotated_data_files.annotated_blood_pressure)
        return df
    
    def _load_heart_rate_data(self):
        df = pd.read_csv(self.annotated_data_files.annotated_heart_rate)
        return df
    
    def _load_respiratory_rate_data(self):
        df = pd.read_csv(self.annotated_data_files.annotated_respiratory_rate)
        return df
    
    def _load_temperature_data(self):
        df = pd.read_csv(self.annotated_data_files.annotated_temperature)
        return df
    
    def _load_oxygen_saturation_data(self):
        df = pd.read_csv(self.annotated_data_files.annotated_oxygen_saturation)
        return df
    
    def _load_medications_data(self):
        df = pd.read_csv(self.annotated_data_files.annotated_medications)
        return df
    
    def _extend_dataframes(self, 
                           preprocessed_dfs: PreprocessedDataFrames, 
                           start_date: str, 
                           end_date: str) -> None:

        self.bp_df = self._prepare_floor_data(original_df=preprocessed_dfs.blood_pressure_df, 
                                 time_col="Scheduled DTTM",
                                 room_col="scheduled_room", 
                                 start_date=start_date, 
                                 end_date=end_date)
        
        self.bp_df["Ordered DTTM"] = pd.to_datetime(self.bp_df["Ordered DTTM"])
        self.bp_df["Administered DTTM"] = pd.to_datetime(self.bp_df["Administered DTTM"])
        
        self.hr_df = self._prepare_floor_data(original_df=preprocessed_dfs.heart_rate_df, 
                                 time_col="Scheduled DTTM",
                                 room_col="scheduled_room", 
                                 start_date=start_date, 
                                 end_date=end_date)
        
        self.hr_df["Ordered DTTM"] = pd.to_datetime(self.hr_df["Ordered DTTM"])
        self.hr_df["Administered DTTM"] = pd.to_datetime(self.hr_df["Administered DTTM"])
        
        self.rr_df = self._prepare_floor_data(original_df=preprocessed_dfs.respiratory_rate_df, 
                                 time_col="Scheduled DTTM",
                                 room_col="scheduled_room", 
                                 start_date=start_date, 
                                 end_date=end_date)
        
        self.rr_df["Ordered DTTM"] = pd.to_datetime(self.rr_df["Ordered DTTM"])
        self.rr_df["Administered DTTM"] = pd.to_datetime(self.rr_df["Administered DTTM"])
        
        self.temp_df = self._prepare_floor_data(original_df=preprocessed_dfs.temperature_df, 
                                 time_col="Scheduled DTTM",
                                 room_col="scheduled_room", 
                                 start_date=start_date, 
                                 end_date=end_date)
        
        self.temp_df["Ordered DTTM"] = pd.to_datetime(self.temp_df["Ordered DTTM"])
        self.temp_df["Administered DTTM"] = pd.to_datetime(self.temp_df["Administered DTTM"])
        
        self.os_df = self._prepare_floor_data(original_df=preprocessed_dfs.oxygen_saturation_df, 
                                 time_col="Scheduled DTTM",
                                 room_col="scheduled_room", 
                                 start_date=start_date, 
                                 end_date=end_date)
        
        self.os_df["Ordered DTTM"] = pd.to_datetime(self.os_df["Ordered DTTM"])
        self.os_df["Administered DTTM"] = pd.to_datetime(self.os_df["Administered DTTM"])
        
        self.med_df = self._prepare_floor_data(original_df=preprocessed_dfs.medications_df, 
                                 time_col="Medication Scheduled DTTM",
                                 room_col="scheduled_room",
                                 start_date=start_date, 
                                 end_date=end_date)
        
        self.med_df["Medication Order DTTM"] = pd.to_datetime(self.med_df["Medication Order DTTM"])
        self.med_df["Administered DTTM"] = pd.to_datetime(self.med_df["Administered DTTM"])
    
    def _prepare_floor_data(self, 
                            original_df: pd.DataFrame, 
                            time_col: str, 
                            room_col: str,
                            start_date: str, 
                            end_date: str) -> pd.DataFrame:
        
        df_prep = DataHelpers.prepare_df(
            original_df,
            time_col=time_col,
            room_col=room_col,
        )

        df_filtered = DataHelpers.apply_date_filters(
            df_prep,
            start_date=start_date,
            end_date=end_date)

        return df_filtered
    
    def _save_dataframes(self) -> None:
        self.bp_df.to_csv(self.request_dir / "blood_pressure_extended.csv", index=False)
        self.hr_df.to_csv(self.request_dir / "heart_rate_extended.csv", index=False)
        self.rr_df.to_csv(self.request_dir / "respiratory_rate_extended.csv", index=False)
        self.temp_df.to_csv(self.request_dir / "temperature_extended.csv", index=False)
        self.os_df.to_csv(self.request_dir / "oxygen_saturation_extended.csv", index=False)
        self.med_df.to_csv(self.request_dir / "medications_extended.csv", index=False)

    def _load_saved_data(self) -> None:
        self.bp_df = pd.read_csv(self.request_dir / "blood_pressure_extended.csv")
        self.bp_df["Scheduled DTTM"] = pd.to_datetime(self.bp_df["Scheduled DTTM"])
        self.bp_df["Ordered DTTM"] = pd.to_datetime(self.bp_df["Ordered DTTM"])
        self.bp_df["Administered DTTM"] = pd.to_datetime(self.bp_df["Administered DTTM"])
        self.bp_df["__day__"] = pd.to_datetime(self.bp_df["__day__"]).dt.date

        self.hr_df = pd.read_csv(self.request_dir / "heart_rate_extended.csv")
        self.hr_df["Scheduled DTTM"] = pd.to_datetime(self.hr_df["Scheduled DTTM"])
        self.hr_df["Ordered DTTM"] = pd.to_datetime(self.hr_df["Ordered DTTM"])
        self.hr_df["Administered DTTM"] = pd.to_datetime(self.hr_df["Administered DTTM"])
        self.hr_df["__day__"] = pd.to_datetime(self.hr_df["__day__"]).dt.date

        self.rr_df = pd.read_csv(self.request_dir / "respiratory_rate_extended.csv")
        self.rr_df["Scheduled DTTM"] = pd.to_datetime(self.rr_df["Scheduled DTTM"])
        self.rr_df["Ordered DTTM"] = pd.to_datetime(self.rr_df["Ordered DTTM"])
        self.rr_df["Administered DTTM"] = pd.to_datetime(self.rr_df["Administered DTTM"])
        self.rr_df["__day__"] = pd.to_datetime(self.rr_df["__day__"]).dt.date

        self.temp_df = pd.read_csv(self.request_dir / "temperature_extended.csv")
        self.temp_df["Scheduled DTTM"] = pd.to_datetime(self.temp_df["Scheduled DTTM"])
        self.temp_df["Ordered DTTM"] = pd.to_datetime(self.temp_df["Ordered DTTM"])
        self.temp_df["Administered DTTM"] = pd.to_datetime(self.temp_df["Administered DTTM"])
        self.temp_df["__day__"] = pd.to_datetime(self.temp_df["__day__"]).dt.date

        self.os_df = pd.read_csv(self.request_dir / "oxygen_saturation_extended.csv")
        self.os_df["Scheduled DTTM"] = pd.to_datetime(self.os_df["Scheduled DTTM"])
        self.os_df["Ordered DTTM"] = pd.to_datetime(self.os_df["Ordered DTTM"])
        self.os_df["Administered DTTM"] = pd.to_datetime(self.os_df["Administered DTTM"])
        self.os_df["__day__"] = pd.to_datetime(self.os_df["__day__"]).dt.date

        self.med_df = pd.read_csv(self.request_dir / "medications_extended.csv")
        self.med_df["Medication Scheduled DTTM"] = pd.to_datetime(self.med_df["Medication Scheduled DTTM"])
        self.med_df["Medication Order DTTM"] = pd.to_datetime(self.med_df["Medication Order DTTM"])
        self.med_df["Administered DTTM"] = pd.to_datetime(self.med_df["Administered DTTM"])
        self.med_df["__day__"] = pd.to_datetime(self.med_df["__day__"]).dt.date

class DailyRequestHandler(GlobalRequestHandler):
    
    def __init__(self, 
                 start_date: str, 
                 end_date: str,
                 floor_number: int,
                 date_stamp: pd.Timestamp,
                 annotated_data_files: AnnotatedDataFiles, 
                 request_dir: str, 
                 use_saved_data: bool = False):
        super().__init__(annotated_data_files=annotated_data_files, 
                         request_dir=request_dir, 
                         start_date=start_date, 
                         end_date=end_date, 
                         use_saved_data=use_saved_data)
        self.daily_requests_dfs = self._process_daily_requests(date_stamp=date_stamp, floor_number=floor_number)
    
    def _process_daily_requests(self, date_stamp: pd.Timestamp, floor_number: int) -> DailyRequestsDataFrames:
        print(f"Processing daily requests for date: {date_stamp.date()} and floor: {floor_number}")
        daily_bp_requests = DataHelpers.get_daily_requests_for_floor(self.bp_df, date_stamp=date_stamp, floor_number=floor_number)
        daily_hr_requests = DataHelpers.get_daily_requests_for_floor(self.hr_df, date_stamp=date_stamp, floor_number=floor_number)
        daily_rr_requests = DataHelpers.get_daily_requests_for_floor(self.rr_df, date_stamp=date_stamp, floor_number=floor_number)
        daily_temp_requests = DataHelpers.get_daily_requests_for_floor(self.temp_df, date_stamp=date_stamp, floor_number=floor_number)
        daily_os_requests = DataHelpers.get_daily_requests_for_floor(self.os_df, date_stamp=date_stamp, floor_number=floor_number)
        daily_med_requests = DataHelpers.get_daily_requests_for_floor(self.med_df, date_stamp=date_stamp, floor_number=floor_number)
        daily_requests_dfs = DailyRequestsDataFrames(
            blood_pressure_requests_df=daily_bp_requests,
            heart_rate_requests_df=daily_hr_requests,
            respiratory_rate_requests_df=daily_rr_requests,
            temperature_requests_df=daily_temp_requests,
            oxygen_saturation_requests_df=daily_os_requests,
            medications_requests_df=daily_med_requests
        )

        return daily_requests_dfs
    
    def _extract_requests_df_for_time_signal(self, 
                                         df: pd.DataFrame, 
                                         time_signal: TimeSignal, 
                                         scheduled_time_col: str, 
                                         ordered_time_col: str,
                                         lookahead_minutes: int) -> pd.DataFrame:
        next_hour_time_stamp = time_signal.time_stamp + pd.Timedelta(minutes=lookahead_minutes)
        mask = (df[scheduled_time_col] >= time_signal.time_stamp) & \
               (df[scheduled_time_col] < next_hour_time_stamp) & \
               (df[ordered_time_col] <= time_signal.time_stamp)
        extracted_requests = df[mask].copy()
        df.drop(extracted_requests.index, inplace=True)
        return extracted_requests
    
    def _convert_df_into_requests_list(self, 
                                       df: pd.DataFrame, 
                                       initial_time: pd.Timestamp,
                                       request_type: str, 
                                       wait_time_seconds: float, 
                                       time_for_rejection_minutes: float,
                                       traversal_graph_generator: TraversalGraphGenerator) -> list[TaskRequest]:
        task_requets_list = []
        for req_index, row in df.iterrows():
            if request_type == "medication":
                supplies_node_label = traversal_graph_generator.doorway_to_node_dict[str(row["scheduled_space_supplies"])]
                room_node_label = traversal_graph_generator.doorway_to_node_dict[str(row["scheduled_space_id"])]
                goal_nodes = [supplies_node_label, room_node_label]
                wait_times_at_goals_seconds = [wait_time_seconds, wait_time_seconds]
                ordered_time = (pd.Timestamp(row["Medication Order DTTM"]) - initial_time).total_seconds()
                scheduled_time = (pd.Timestamp(row["Medication Scheduled DTTM"]) - initial_time).total_seconds()
                administered_time = (pd.Timestamp(row["Administered DTTM"]) - initial_time).total_seconds()
            else:
                room_node_label = traversal_graph_generator.doorway_to_node_dict[str(row["scheduled_space_id"])]
                goal_nodes = [room_node_label]
                wait_times_at_goals_seconds = [wait_time_seconds]
                ordered_time = (pd.Timestamp(row["Ordered DTTM"]) - initial_time).total_seconds()
                scheduled_time = (pd.Timestamp(row["Scheduled DTTM"]) - initial_time).total_seconds()
                administered_time = (pd.Timestamp(row["Administered DTTM"]) - initial_time).total_seconds()
            
            task_request = TaskRequest(
                request_id=request_type+"."+str(req_index),
                request_type=request_type,
                goal_nodes=goal_nodes,
                wait_times_at_goals_seconds=wait_times_at_goals_seconds,
                time_for_rejection_minutes=time_for_rejection_minutes,
                ordered_time=ordered_time,
                scheduled_time=scheduled_time,
                administered_time=administered_time
                )

            task_requets_list.append(task_request)
        return task_requets_list
            
    
    def _get_requests_for_time_signal(self, 
                                     df: pd.DataFrame, 
                                     initial_time: pd.Timestamp,
                                     time_signal: TimeSignal, 
                                     scheduled_time_col: str, 
                                     ordered_time_col: str,
                                     lookahead_minutes: int,
                                     request_type: str,
                                     wait_time_seconds: float,
                                     time_for_rejection_minutes: float,
                                     traversal_graph_generator: TraversalGraphGenerator) -> list[TaskRequest]:
        extracted_requests_df = self._extract_requests_df_for_time_signal(
                                                                    df=df,
                                                                    time_signal=time_signal,
                                                                    scheduled_time_col=scheduled_time_col,
                                                                    ordered_time_col=ordered_time_col,
                                                                    lookahead_minutes=lookahead_minutes
                                                                    )
        requests_list = self._convert_df_into_requests_list(
            df=extracted_requests_df,
            initial_time=initial_time,
            request_type=request_type,
            wait_time_seconds=wait_time_seconds,
            time_for_rejection_minutes=time_for_rejection_minutes,
            traversal_graph_generator=traversal_graph_generator
            )

        return requests_list

        
    
    def get_all_requests_for_time_signal(self, 
                                         time_signal: TimeSignal,
                                         initial_time: pd.Timestamp,
                                         look_ahead_minutes: int,
                                         all_task_properties: AllTaskProperties,
                                         traversal_graph_generator: TraversalGraphGenerator
                                         ) -> RequestsLists:
        
        extracted_bp_requests = self._get_requests_for_time_signal(
            df=self.daily_requests_dfs.blood_pressure_requests_df,
            initial_time=initial_time,
            time_signal=time_signal,
            scheduled_time_col="Scheduled DTTM",
            ordered_time_col="Ordered DTTM",
            lookahead_minutes=look_ahead_minutes,
            request_type=all_task_properties.blood_pressure.task_type,
            wait_time_seconds=all_task_properties.blood_pressure.wait_time_seconds,
            time_for_rejection_minutes=all_task_properties.blood_pressure.time_for_rejection_minutes,
            traversal_graph_generator=traversal_graph_generator)
        
        extracted_hr_requests = self._get_requests_for_time_signal(
            df=self.daily_requests_dfs.heart_rate_requests_df,
            initial_time=initial_time,
            time_signal=time_signal,
            scheduled_time_col="Scheduled DTTM",
            ordered_time_col="Ordered DTTM",
            lookahead_minutes=look_ahead_minutes,
            request_type=all_task_properties.heart_rate.task_type,
            wait_time_seconds=all_task_properties.heart_rate.wait_time_seconds,
            time_for_rejection_minutes=all_task_properties.heart_rate.time_for_rejection_minutes,
            traversal_graph_generator=traversal_graph_generator
            )
        
        extracted_rr_requests = self._get_requests_for_time_signal(
            df=self.daily_requests_dfs.respiratory_rate_requests_df,
            initial_time=initial_time,
            time_signal=time_signal,
            scheduled_time_col="Scheduled DTTM",
            ordered_time_col="Ordered DTTM",
            lookahead_minutes=look_ahead_minutes,
            request_type=all_task_properties.respiratory_rate.task_type,
            wait_time_seconds=all_task_properties.respiratory_rate.wait_time_seconds,
            time_for_rejection_minutes=all_task_properties.respiratory_rate.time_for_rejection_minutes,
            traversal_graph_generator=traversal_graph_generator
            )
        
        extracted_temp_requests = self._get_requests_for_time_signal(
            df=self.daily_requests_dfs.temperature_requests_df,
            initial_time=initial_time,
            time_signal=time_signal,
            scheduled_time_col="Scheduled DTTM",
            ordered_time_col="Ordered DTTM",
            lookahead_minutes=look_ahead_minutes,
            request_type=all_task_properties.temperature.task_type,
            wait_time_seconds=all_task_properties.temperature.wait_time_seconds,
            time_for_rejection_minutes=all_task_properties.temperature.time_for_rejection_minutes,
            traversal_graph_generator=traversal_graph_generator
            )
        
        extracted_os_requests = self._get_requests_for_time_signal(
            df=self.daily_requests_dfs.oxygen_saturation_requests_df,
            initial_time=initial_time,
            time_signal=time_signal,
            scheduled_time_col="Scheduled DTTM",
            ordered_time_col="Ordered DTTM",
            lookahead_minutes=look_ahead_minutes,
            request_type=all_task_properties.oxygen_saturation.task_type,
            wait_time_seconds=all_task_properties.oxygen_saturation.wait_time_seconds,
            time_for_rejection_minutes=all_task_properties.oxygen_saturation.time_for_rejection_minutes,
            traversal_graph_generator=traversal_graph_generator
            )
        
        extracted_med_requests = self._get_requests_for_time_signal(
            df=self.daily_requests_dfs.medications_requests_df,
            initial_time=initial_time,
            time_signal=time_signal,
            scheduled_time_col="Medication Scheduled DTTM",
            ordered_time_col="Medication Order DTTM",
            lookahead_minutes=look_ahead_minutes,
            request_type=all_task_properties.medications.task_type,
            wait_time_seconds=all_task_properties.medications.wait_time_seconds,
            time_for_rejection_minutes=all_task_properties.medications.time_for_rejection_minutes,
            traversal_graph_generator=traversal_graph_generator
            )
        
        extracted_requests_lists = RequestsLists(
            blood_pressure_requests=extracted_bp_requests,
            heart_rate_requests=extracted_hr_requests,
            respiratory_rate_requests=extracted_rr_requests,
            temperature_requests=extracted_temp_requests,
            oxygen_saturation_requests=extracted_os_requests,
            medications_requests=extracted_med_requests
            )

        return extracted_requests_lists

class PlanningRequestHandler(GlobalRequestHandler):
    
    def __init__(self, 
                 start_date: str, 
                 end_date: str,
                 floor_number: int,
                 date_stamp: pd.Timestamp,
                 annotated_data_files: AnnotatedDataFiles, 
                 request_dir: str, 
                 use_saved_data: bool = False):
        super().__init__(annotated_data_files=annotated_data_files, 
                         request_dir=request_dir, 
                         start_date=start_date, 
                         end_date=end_date, 
                         use_saved_data=use_saved_data)
        self.daily_requests_dfs = self._process_daily_requests(date_stamp=date_stamp, floor_number=floor_number)
    
    def _process_daily_requests(self, date_stamp: pd.Timestamp, floor_number: int) -> DailyRequestsDataFrames:
        print(f"Processing daily requests for date: {date_stamp.date()} and floor: {floor_number}")
        daily_bp_requests = DataHelpers.get_daily_requests_for_floor(self.bp_df, date_stamp=date_stamp, floor_number=floor_number)
        daily_hr_requests = DataHelpers.get_daily_requests_for_floor(self.hr_df, date_stamp=date_stamp, floor_number=floor_number)
        daily_rr_requests = DataHelpers.get_daily_requests_for_floor(self.rr_df, date_stamp=date_stamp, floor_number=floor_number)
        daily_temp_requests = DataHelpers.get_daily_requests_for_floor(self.temp_df, date_stamp=date_stamp, floor_number=floor_number)
        daily_os_requests = DataHelpers.get_daily_requests_for_floor(self.os_df, date_stamp=date_stamp, floor_number=floor_number)
        daily_med_requests = DataHelpers.get_daily_requests_for_floor(self.med_df, date_stamp=date_stamp, floor_number=floor_number)
        daily_requests_dfs = DailyRequestsDataFrames(
            blood_pressure_requests_df=daily_bp_requests,
            heart_rate_requests_df=daily_hr_requests,
            respiratory_rate_requests_df=daily_rr_requests,
            temperature_requests_df=daily_temp_requests,
            oxygen_saturation_requests_df=daily_os_requests,
            medications_requests_df=daily_med_requests
        )

        return daily_requests_dfs
    
    def _extract_requests_df_for_time_signal(self, 
                                         df: pd.DataFrame, 
                                         time_signal: TimeSignal, 
                                         scheduled_time_col: str, 
                                         ordered_time_col: str,
                                         lookahead_minutes: int) -> pd.DataFrame:
        next_hour_time_stamp = time_signal.time_stamp + pd.Timedelta(minutes=lookahead_minutes)
        mask = (df[scheduled_time_col] >= time_signal.time_stamp) & \
               (df[scheduled_time_col] < next_hour_time_stamp) & \
               (df[ordered_time_col] <= time_signal.time_stamp)
        extracted_requests = df[mask].copy()
        return extracted_requests
    
    def _convert_df_into_requests_list(self, 
                                       df: pd.DataFrame, 
                                       initial_time: pd.Timestamp,
                                       request_type: str, 
                                       wait_time_seconds: float, 
                                       time_for_rejection_minutes: float,
                                       traversal_graph_generator: TraversalGraphGenerator) -> list[TaskRequest]:
        task_requests_list = []
        for req_index, row in df.iterrows():
            if request_type == "medication":
                supplies_node_label = traversal_graph_generator.doorway_to_node_dict[str(row["scheduled_space_supplies"])]
                room_node_label = traversal_graph_generator.doorway_to_node_dict[str(row["scheduled_space_id"])]
                goal_nodes = [supplies_node_label, room_node_label]
                wait_times_at_goals_seconds = [wait_time_seconds, wait_time_seconds]
                ordered_time = (pd.Timestamp(row["Medication Order DTTM"]) - initial_time).total_seconds()
                scheduled_time = (pd.Timestamp(row["Medication Scheduled DTTM"]) - initial_time).total_seconds()
                administered_time = (pd.Timestamp(row["Administered DTTM"]) - initial_time).total_seconds()
            else:
                room_node_label = traversal_graph_generator.doorway_to_node_dict[str(row["scheduled_space_id"])]
                goal_nodes = [room_node_label]
                wait_times_at_goals_seconds = [wait_time_seconds]
                ordered_time = (pd.Timestamp(row["Ordered DTTM"]) - initial_time).total_seconds()
                scheduled_time = (pd.Timestamp(row["Scheduled DTTM"]) - initial_time).total_seconds()
                administered_time = (pd.Timestamp(row["Administered DTTM"]) - initial_time).total_seconds()
            
            task_request = TaskRequest(
                request_id=request_type+"."+str(req_index),
                request_type=request_type,
                goal_nodes=goal_nodes,
                wait_times_at_goals_seconds=wait_times_at_goals_seconds,
                time_for_rejection_minutes=time_for_rejection_minutes,
                ordered_time=ordered_time,
                scheduled_time=scheduled_time,
                administered_time=administered_time
                )

            task_requests_list.append(task_request)
        return task_requests_list
            
    
    def _get_requests_for_time_signal(self, 
                                     df: pd.DataFrame, 
                                     initial_time: pd.Timestamp,
                                     time_signal: TimeSignal, 
                                     scheduled_time_col: str, 
                                     ordered_time_col: str,
                                     lookahead_minutes: int,
                                     request_type: str,
                                     wait_time_seconds: float,
                                     time_for_rejection_minutes: float,
                                     traversal_graph_generator: TraversalGraphGenerator) -> list[TaskRequest]:
        extracted_requests_df = self._extract_requests_df_for_time_signal(
                                                                    df=df,
                                                                    time_signal=time_signal,
                                                                    scheduled_time_col=scheduled_time_col,
                                                                    ordered_time_col=ordered_time_col,
                                                                    lookahead_minutes=lookahead_minutes
                                                                    )
        requests_list = self._convert_df_into_requests_list(
            df=extracted_requests_df,
            initial_time=initial_time,
            request_type=request_type,
            wait_time_seconds=wait_time_seconds,
            time_for_rejection_minutes=time_for_rejection_minutes,
            traversal_graph_generator=traversal_graph_generator
            )

        return requests_list

        
    
    def get_all_requests_for_time_signal(self, 
                                         time_signal: TimeSignal,
                                         initial_time: pd.Timestamp,
                                         look_ahead_minutes: int,
                                         all_task_properties: AllTaskProperties,
                                         traversal_graph_generator: TraversalGraphGenerator
                                         ) -> RequestsLists:
        
        extracted_bp_requests = self._get_requests_for_time_signal(
            df=self.daily_requests_dfs.blood_pressure_requests_df,
            initial_time=initial_time,
            time_signal=time_signal,
            scheduled_time_col="Scheduled DTTM",
            ordered_time_col="Ordered DTTM",
            lookahead_minutes=look_ahead_minutes,
            request_type=all_task_properties.blood_pressure.task_type,
            wait_time_seconds=all_task_properties.blood_pressure.wait_time_seconds,
            time_for_rejection_minutes=all_task_properties.blood_pressure.time_for_rejection_minutes,
            traversal_graph_generator=traversal_graph_generator)
        
        extracted_hr_requests = self._get_requests_for_time_signal(
            df=self.daily_requests_dfs.heart_rate_requests_df,
            initial_time=initial_time,
            time_signal=time_signal,
            scheduled_time_col="Scheduled DTTM",
            ordered_time_col="Ordered DTTM",
            lookahead_minutes=look_ahead_minutes,
            request_type=all_task_properties.heart_rate.task_type,
            wait_time_seconds=all_task_properties.heart_rate.wait_time_seconds,
            time_for_rejection_minutes=all_task_properties.heart_rate.time_for_rejection_minutes,
            traversal_graph_generator=traversal_graph_generator
            )
        
        extracted_rr_requests = self._get_requests_for_time_signal(
            df=self.daily_requests_dfs.respiratory_rate_requests_df,
            initial_time=initial_time,
            time_signal=time_signal,
            scheduled_time_col="Scheduled DTTM",
            ordered_time_col="Ordered DTTM",
            lookahead_minutes=look_ahead_minutes,
            request_type=all_task_properties.respiratory_rate.task_type,
            wait_time_seconds=all_task_properties.respiratory_rate.wait_time_seconds,
            time_for_rejection_minutes=all_task_properties.respiratory_rate.time_for_rejection_minutes,
            traversal_graph_generator=traversal_graph_generator
            )
        
        extracted_temp_requests = self._get_requests_for_time_signal(
            df=self.daily_requests_dfs.temperature_requests_df,
            initial_time=initial_time,
            time_signal=time_signal,
            scheduled_time_col="Scheduled DTTM",
            ordered_time_col="Ordered DTTM",
            lookahead_minutes=look_ahead_minutes,
            request_type=all_task_properties.temperature.task_type,
            wait_time_seconds=all_task_properties.temperature.wait_time_seconds,
            time_for_rejection_minutes=all_task_properties.temperature.time_for_rejection_minutes,
            traversal_graph_generator=traversal_graph_generator
            )
        
        extracted_os_requests = self._get_requests_for_time_signal(
            df=self.daily_requests_dfs.oxygen_saturation_requests_df,
            initial_time=initial_time,
            time_signal=time_signal,
            scheduled_time_col="Scheduled DTTM",
            ordered_time_col="Ordered DTTM",
            lookahead_minutes=look_ahead_minutes,
            request_type=all_task_properties.oxygen_saturation.task_type,
            wait_time_seconds=all_task_properties.oxygen_saturation.wait_time_seconds,
            time_for_rejection_minutes=all_task_properties.oxygen_saturation.time_for_rejection_minutes,
            traversal_graph_generator=traversal_graph_generator
            )
        
        extracted_med_requests = self._get_requests_for_time_signal(
            df=self.daily_requests_dfs.medications_requests_df,
            initial_time=initial_time,
            time_signal=time_signal,
            scheduled_time_col="Medication Scheduled DTTM",
            ordered_time_col="Medication Order DTTM",
            lookahead_minutes=look_ahead_minutes,
            request_type=all_task_properties.medications.task_type,
            wait_time_seconds=all_task_properties.medications.wait_time_seconds,
            time_for_rejection_minutes=all_task_properties.medications.time_for_rejection_minutes,
            traversal_graph_generator=traversal_graph_generator
            )
        
        extracted_requests_lists = RequestsLists(
            blood_pressure_requests=extracted_bp_requests,
            heart_rate_requests=extracted_hr_requests,
            respiratory_rate_requests=extracted_rr_requests,
            temperature_requests=extracted_temp_requests,
            oxygen_saturation_requests=extracted_os_requests,
            medications_requests=extracted_med_requests
            )

        return extracted_requests_lists
        


def main():
    parser = argparse.ArgumentParser(description="Process hospital data and generate daily requests.")

    # date_operational_range parameters
    parser.add_argument("--year", type=int, dest='year', default=2024, help='Select year of interest.')
    parser.add_argument("--month", type=int, dest='month', default=6, help='Select month of interest.')
    parser.add_argument("--day", type=int, dest='day', default=24, help='Select day of interest.')
    parser.add_argument("--hour", type=int, dest='hour', default=8, help='Select hour of interest.')
    parser.add_argument("--minute", type=int, dest='minute', default=0, help='Select minute of interest.')
    parser.add_argument("--floor_number", type=int, dest='floor_number', default=9, help='Select floor number of interest.')

    # file paths
    parser.add_argument("--request_dir", type=str, default="data/requests", help="Directory to save global requests data.")
    parser.add_argument("--use_saved_request_data", action='store_true', help="Flag to use previously saved request data.")
    parser.add_argument("--medications_orders_file", type=str, default="data/processed/medication_orders_annotated.csv", help="Path to the medications orders CSV file.")
    parser.add_argument("--blood_pressure_orders_file", type=str, default="data/processed/blood_pressure_orders_annotated.csv", help="Path to the blood pressure orders CSV file.")
    parser.add_argument("--heart_rate_orders_file", type=str, default="data/processed/heart_rate_orders_annotated.csv", help="Path to the heart rate orders CSV file.")
    parser.add_argument("--respiratory_rate_orders_file", type=str, default="data/processed/respiratory_rate_orders_annotated.csv", help="Path to the respiratory rate orders CSV file.")
    parser.add_argument("--temperature_orders_file", type=str, default="data/processed/temperature_orders_annotated.csv", help="Path to the temperature orders CSV file.")
    parser.add_argument("--oxygen_saturation_orders_file", type=str, default="data/processed/oxygen_saturation_orders_annotated.csv", help="Path to the oxygen saturation orders CSV file.")

    # traversal graph parameters
    parser.add_argument("--config_path", type=str, default="maps/hospital_floor/floor_config.yaml", help="Path to the configuration file")
    parser.add_argument("--occupancy_map_path", type=str, default="maps/hospital_floor/occupancy_map.npy", help="Path to the input occupancy map")
    parser.add_argument("--factor", type=int, default=1, help="Downsampling factor")
    parser.add_argument("--meters_per_pixel", type=float, default=0.036, help="Meters per pixel in the original image")
    args = parser.parse_args()

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

    date_stamp = pd.Timestamp(year=args.year, month=args.month, day=args.day)
    start_date="2024-06-24"
    end_date="2025-06-29"

    request_handler = DailyRequestHandler(start_date=start_date,
                                          end_date=end_date,
                                          floor_number=args.floor_number,
                                          date_stamp=date_stamp,
                                          annotated_data_files=annotated_data_files,
                                          request_dir=args.request_dir,
                                          use_saved_data=args.use_saved_request_data)
    
    initial_time = pd.Timestamp(year=args.year, month=args.month, day=args.day, hour=0, minute=0)
    time_signal = TimeSignal(year=args.year, month=args.month, day=args.day, hour=args.hour, minute=args.minute)

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

    tg_generator = TraversalGraphGenerator(occupancy_map_path=args.occupancy_map_path,
                                           config_path=args.config_path,
                                           meters_per_pixel=args.meters_per_pixel,
                                           factor=args.factor)

    requests_lists = request_handler.get_all_requests_for_time_signal(
        initial_time=initial_time,
        time_signal=time_signal,
        look_ahead_minutes=60,
        all_task_properties=all_task_properties,
        traversal_graph_generator=tg_generator
    )

    print("Extracted Requests:")
    print(f"Blood Pressure Requests: {len(requests_lists.blood_pressure_requests)}")
    print(f"Heart Rate Requests: {len(requests_lists.heart_rate_requests)}")
    print(f"Respiratory Rate Requests: {len(requests_lists.respiratory_rate_requests)}")
    print(f"Temperature Requests: {len(requests_lists.temperature_requests)}")
    print(f"Oxygen Saturation Requests: {len(requests_lists.oxygen_saturation_requests)}")
    print(f"Medications Requests: {len(requests_lists.medications_requests)}")

    print("Request processing completed successfully.")

    print(f"requests_lists.blood_pressure_requests: {requests_lists.blood_pressure_requests}")
    
    
if __name__ == "__main__":
    pStart = datetime.now()
    try:
        main()
    except Exception as errorMainContext:
        print("Fail End Process: ", errorMainContext)
        traceback.print_exc()
    pEnd = datetime.now()
    print(f"Total Execution Time: {pEnd - pStart}")