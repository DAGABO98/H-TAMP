import argparse
from datetime import datetime
import re
import traceback

from pathlib import Path
from typing import Optional

import pandas as pd

from HTAMP.data_processing.processing_dataclasses import AnnotatedDataFiles
from HTAMP.plotting.data_statistics_plotting import DataStatisticsPlottingHelper

class DataStatisticsHelpers:

    @staticmethod
    def extract_floor(room: Optional[str]) -> Optional[int]:
        if room is None:
            return None
        s = str(room)

        m = re.search(r"FA?(\d+)", s, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))

        return None
    
    @staticmethod
    def build_weekly_counts(df: pd.DataFrame, date_col: str, room_col: str, week_start: str) -> pd.DataFrame:
        # Ensure datetime
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")


        # Parse floor from the chosen room column
        df["floor"] = df[room_col].apply(DataStatisticsHelpers.extract_floor)


        # Keep rows with valid date and floor
        df = df.dropna(subset=[date_col, "floor"]) # floor may be float from NA; coerce to Int64
        df["floor"] = df["floor"].astype("Int64")


        # Derive week bucket (start date of the week)
        week_alias = f"W-{week_start.upper()}" # e.g., W-MON
        df["week_start"] = df[date_col].dt.to_period(week_alias).dt.start_time


        # Aggregate
        counts = (
        df.groupby(["floor", "week_start"], dropna=False)
            .size()
            .reset_index(name="num_requests")
            .sort_values(["floor", "week_start"])
        )

        return counts


class DataStatistics:
    def __init__(self, outdir: str, week_start: str, annotated_data_files: AnnotatedDataFiles):
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        
        self.annotated_data_files = annotated_data_files
        self.week_start = week_start

        self.bp_df = self._load_bp_data()
        self.medications_df = self._load_medications_data()
        self.hr_df = self._load_hr_data()
        self.rr_df = self._load_rr_data()
        self.temp_df = self._load_temp_data()
        self.oximetry_df = self._load_oximetry_data()
        self.admissions_discharges_df = self._load_admissions_discharges_data()

        self.weekly_bp_counts = self._generate_weekly_bp_counts()
        self.weekly_medications_counts = self._generate_weekly_medications_counts()
        self.weekly_hr_counts = self._generate_weekly_hr_counts()
        self.weekly_rr_counts = self._generate_weekly_rr_counts()
        self.weekly_temp_counts = self._generate_weekly_temp_counts()
        self.weekly_oximetry_counts = self._generate_weekly_oximetry_counts()
        self.weekly_admissions_counts = self._generate_weekly_admissions_counts()
        self.weekly_discharges_counts = self._generate_weekly_discharges_counts()
    
    def _load_bp_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.annotated_data_files.annotated_blood_pressure)
        return df
    
    def _load_medications_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.annotated_data_files.annotated_medications)
        return df
    
    def _load_hr_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.annotated_data_files.annotated_heart_rate)
        return df
    
    def _load_rr_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.annotated_data_files.annotated_respiratory_rate)
        return df
    
    def _load_temp_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.annotated_data_files.annotated_temperature)
        return df
    
    def _load_oximetry_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.annotated_data_files.annotated_oxygen_saturation)
        return df
    
    def _load_visits_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.annotated_data_files.annotated_visits)
        return df
    
    def _load_admissions_discharges_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.annotated_data_files.annotated_admissions_discharges)
        return df
    
    def _generate_weekly_bp_counts(self) -> pd.DataFrame:
        counts = DataStatisticsHelpers.build_weekly_counts(
            self.bp_df,
            date_col="Scheduled DTTM",
            room_col="scheduled_room",
            week_start=self.week_start,
        )
        return counts
    
    def _generate_weekly_medications_counts(self) -> pd.DataFrame:
        counts = DataStatisticsHelpers.build_weekly_counts(
            self.medications_df,
            date_col="Medication Scheduled DTTM",
            room_col="scheduled_room",
            week_start=self.week_start,
        )
        return counts
    
    def _generate_weekly_hr_counts(self) -> pd.DataFrame:
        counts = DataStatisticsHelpers.build_weekly_counts(
            self.hr_df,
            date_col="Scheduled DTTM",
            room_col="scheduled_room",
            week_start=self.week_start,
        )
        return counts
    
    def _generate_weekly_rr_counts(self) -> pd.DataFrame:
        counts = DataStatisticsHelpers.build_weekly_counts(
            self.rr_df,
            date_col="Scheduled DTTM",
            room_col="scheduled_room",
            week_start=self.week_start,
        )
        return counts
    
    def _generate_weekly_temp_counts(self) -> pd.DataFrame:
        counts = DataStatisticsHelpers.build_weekly_counts(
            self.temp_df,
            date_col="Scheduled DTTM",
            room_col="scheduled_room",
            week_start=self.week_start,
        )
        return counts
    
    def _generate_weekly_oximetry_counts(self) -> pd.DataFrame:
        counts = DataStatisticsHelpers.build_weekly_counts(
            self.oximetry_df,
            date_col="Scheduled DTTM",
            room_col="scheduled_room",
            week_start=self.week_start,
        )
        return counts
    
    def _generate_weekly_admissions_counts(self) -> pd.DataFrame:
        counts = DataStatisticsHelpers.build_weekly_counts(
            self.admissions_discharges_df,
            date_col="HOSPITAL_ADMISSION",
            room_col="IN_DEP",
            week_start=self.week_start,
        )
        return counts
    
    def _generate_weekly_discharges_counts(self) -> pd.DataFrame:
        counts = DataStatisticsHelpers.build_weekly_counts(
            self.admissions_discharges_df,
            date_col="HOSPITAL_DISCHARGE",
            room_col="OUT_DEP",
            week_start=self.week_start,
        )
        return counts
    
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
    parser.add_argument("--outdir", default="results/distributions", help="Directory to write outputs (default: ./results/distributions)")
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

    data_stats = DataStatistics(
        outdir=args.outdir,
        week_start=args.week_start,
        annotated_data_files=annotated_data_files,
    )

    # Generate plots
    bp_fig_paths = DataStatisticsPlottingHelper.plot_per_floor(data_stats.weekly_bp_counts, Path(args.outdir))
    if bp_fig_paths:
        print("Saved figures:")
        for pth in bp_fig_paths:
            print(f" - {pth}")
    else:
        print("No floors found to plot.")



    
if __name__ == "__main__":
    pStart = datetime.now()
    try:
        main()
    except Exception as errorMainContext:
        print("Fail End Process: ", errorMainContext)
        traceback.print_exc()
    pEnd = datetime.now()
    print(f"Total Execution Time: {pEnd - pStart}")