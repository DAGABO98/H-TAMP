import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from HTAMP.data_processing.data_helpers import DataHelpers
from HTAMP.data_processing.processing_dataclasses import AnnotatedDataFiles, PreprocessedDataFrames

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
            self._load_saved_data(request_dir)

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
    
    def _extend_dataframes(self, preprocessed_dfs: PreprocessedDataFrames, start_date: str, end_date: str) -> pd.DataFrame:

        self.bp_df = self._prepare_floor_data(original_df=preprocessed_dfs.blood_pressure_df, 
                                 time_col="Scheduled DTTM",
                                 room_col="scheduled_room", 
                                 start_date=start_date, 
                                 end_date=end_date)
        
        self.hr_df = self._prepare_floor_data(original_df=preprocessed_dfs.heart_rate_df, 
                                 time_col="Scheduled DTTM",
                                 room_col="scheduled_room", 
                                 start_date=start_date, 
                                 end_date=end_date)
        
        self.rr_df = self._prepare_floor_data(original_df=preprocessed_dfs.respiratory_rate_df, 
                                 time_col="Scheduled DTTM",
                                 room_col="scheduled_room", 
                                 start_date=start_date, 
                                 end_date=end_date)
        
        self.temp_df = self._prepare_floor_data(original_df=preprocessed_dfs.temperature_df, 
                                 time_col="Scheduled DTTM",
                                 room_col="scheduled_room", 
                                 start_date=start_date, 
                                 end_date=end_date)
        
        self.os_df = self._prepare_floor_data(original_df=preprocessed_dfs.oxygen_saturation_df, 
                                 time_col="Scheduled DTTM",
                                 room_col="scheduled_room", 
                                 start_date=start_date, 
                                 end_date=end_date)
        
        self.med_df = self._prepare_floor_data(original_df=preprocessed_dfs.medications_df, 
                                 time_col="Medication Scheduled DTTM",
                                 room_col="scheduled_room",
                                 start_date=start_date, 
                                 end_date=end_date)
    
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
        self.hr_df = pd.read_csv(self.request_dir / "heart_rate_extended.csv")
        self.rr_df = pd.read_csv(self.request_dir / "respiratory_rate_extended.csv")
        self.temp_df = pd.read_csv(self.request_dir / "temperature_extended.csv")
        self.os_df = pd.read_csv(self.request_dir / "oxygen_saturation_extended.csv")
        self.med_df = pd.read_csv(self.request_dir / "medications_extended.csv")

class DailyRequestHandler(GlobalRequestHandler):
    
    def __init__(self, annotated_data_files: AnnotatedDataFiles, request_dir: str, use_saved_data: bool = False):
        super().__init__(annotated_data_files, request_dir, use_saved_data)

def main():
    parser = argparse.ArgumentParser(description="Weekly histograms of scheduled requests per floor")
    parser.add_argument("--visits_data_file", type=str, default="data/processed/patient_room_stays.csv", help="Path to the visits data CSV file.")
    parser.add_argument("--admissions_discharges_file", type=str, default="data/processed/admissions_discharges.csv", help="Path to the annotated admissions and discharges CSV file.")
    parser.add_argument("--medications_orders_file", type=str, default="data/processed/medication_orders_annotated.csv", help="Path to the medications orders CSV file.")
    parser.add_argument("--blood_pressure_orders_file", type=str, default="data/processed/blood_pressure_orders_annotated.csv", help="Path to the blood pressure orders CSV file.")
    parser.add_argument("--heart_rate_orders_file", type=str, default="data/processed/heart_rate_orders_annotated.csv", help="Path to the heart rate orders CSV file.")
    parser.add_argument("--respiratory_rate_orders_file", type=str, default="data/processed/respiratory_rate_orders_annotated.csv", help="Path to the respiratory rate orders CSV file.")
    parser.add_argument("--temperature_orders_file", type=str, default="data/processed/temperature_orders_annotated.csv", help="Path to the temperature orders CSV file.")
    parser.add_argument("--oxygen_saturation_orders_file", type=str, default="data/processed/oxygen_saturation_orders_annotated.csv", help="Path to the oxygen saturation orders CSV file.")
    parser.add_argument("--week-start", default="MON", help="Week anchor day: MON (default), SUN, TUE, ...")
    parser.add_argument("--outdir", default="results", help="Directory to write outputs (default: ./results)")
    parser.add_argument("--dist-data-dir", default="data/distributions", help="Directory to write distribution data outputs (default: ./data/distributions/)")
    args = parser.parse_args()

    annotated_data_files = AnnotatedDataFiles(
        annotated_visits=args.visits_data_file,
        annotated_admissions_discharges=args.admissions_discharges_file,
        annotated_medications=args.medications_orders_file,
        annotated_blood_pressure=args.blood_pressure_orders_file,
        annotated_heart_rate=args.heart_rate_orders_file,
        annotated_respiratory_rate=args.respiratory_rate_orders_file,
        annotated_temperature=args.temperature_orders_file,
        annotated_oxygen_saturation=args.oxygen_saturation_orders_file,
    )