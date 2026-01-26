import argparse
import traceback

from pathlib import Path
from datetime import datetime
from typing import Sequence, Tuple

import pandas as pd

from HTAMP.data_processing.processing_dataclasses import AnnotatedDataFiles
from HTAMP.plotting.data_statistics_plotting import DataStatisticsPlottingHelper
from HTAMP.data_processing.data_helpers import DataHelpers


class DataStatistics:
    def __init__(self, 
                 outdir: str, 
                 dist_data_dir: str,
                 annotated_data_files: AnnotatedDataFiles):
        
        self._make_dirs(outdir, dist_data_dir)
        self.annotated_data_files = annotated_data_files
        self._load_data()
    
    def _make_dirs(self, outdir: str, dist_data_dir: str) -> None:
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

        self.dist_outdir = self.outdir / "distributions"
        self.dist_outdir.mkdir(parents=True, exist_ok=True)

        self.indiv_dist_outdir = self.dist_outdir / "individual_distributions"
        self.indiv_dist_outdir.mkdir(parents=True, exist_ok=True)

        self.u_chart_outdir = self.outdir / "u_charts"
        self.u_chart_outdir.mkdir(parents=True, exist_ok=True)

        self.heatmap_outdir = self.outdir / "heatmaps"
        self.heatmap_outdir.mkdir(parents=True, exist_ok=True)

        self.wass_outdir = self.outdir / "wasserstein_distances"
        self.wass_outdir.mkdir(parents=True, exist_ok=True)

        self.dist_data_dir = Path(dist_data_dir)
        self.dist_data_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_data(self):
        self.bp_df = self._load_bp_data()
        self.medications_df = self._load_medications_data()
        self.hr_df = self._load_hr_data()
        self.rr_df = self._load_rr_data()
        self.temp_df = self._load_temp_data()
        self.oximetry_df = self._load_oximetry_data()
        self.admissions_discharges_df = self._load_admissions_discharges_data()
    
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
    
    def _load_admissions_discharges_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.annotated_data_files.annotated_admissions_discharges)
        return df
    
    def generate_count_distribution(self, 
                                    original_df: pd.DataFrame, 
                                    time_col: str, 
                                    room_col: str,
                                    start_date: str, 
                                    end_date: str, 
                                    exclude_iso_weeks: Sequence[int]) -> pd.DataFrame:
        df_prep = DataHelpers.prepare_df(
            original_df,
            time_col=time_col,
            room_col=room_col,
        )

        df_filtered = DataHelpers.apply_date_filters(
            df_prep,
            start_date=start_date,
            end_date=end_date,
            exclude_iso_weeks=exclude_iso_weeks,
        )

        per_day = DataHelpers.compute_per_day_counts(dff=df_filtered)

        dist = DataHelpers.distribution_of_counts(per_day=per_day)

        return dist, per_day
    
    def save_distribution_files(self, 
                                 dist: pd.DataFrame, 
                                 per_day: pd.DataFrame, 
                                 start_date: str, 
                                 end_date: str,
                                 label: str) -> None:
        dist_dir = self.dist_data_dir / label
        dist_dir.mkdir(parents=True, exist_ok=True)

        dist_csv = dist_dir / f"floor_requests_distribution_{start_date}_{end_date}.csv"
        per_day_csv = dist_dir / f"counts_per_floor_day_{start_date}_{end_date}.csv"

        dist.to_csv(dist_csv, index=False)
        per_day.to_csv(per_day_csv, index=False)
    
    def plot_distribution(self, dist: pd.DataFrame, out_png: str) -> None:
        png = self.dist_outdir / out_png
        DataStatisticsPlottingHelper.plot_distribution(
            dist=dist,
            out_png=png,
        )
    
    def generate_and_plot_distribution(self,
                                       original_df: pd.DataFrame,
                                       time_col: str,
                                       room_col: str,
                                       start_date: str,
                                       end_date: str,
                                       exclude_iso_weeks: Sequence[Tuple[int, int]],
                                       label: str) -> None:
        dist, per_day = self.generate_count_distribution(
            original_df=original_df,
            time_col=time_col,
            room_col=room_col,
            start_date=start_date,
            end_date=end_date,
            exclude_iso_weeks=exclude_iso_weeks
        )

        self.save_distribution_files(
            dist=dist,
            per_day=per_day,
            start_date=start_date,
            end_date=end_date,
            label=label
        )

        out_png = f"{label}_requests_distribution_{start_date}_{end_date}.png"
        self.plot_distribution(
            dist=dist,
            out_png=out_png
        )
    
    def generate_and_plot_all_distributions(self,
                                            start_date: str,
                                            end_date: str,
                                            exclude_iso_weeks: Sequence[Tuple[int, int]]) -> None:
        self.generate_and_plot_distribution(
            original_df=self.bp_df,
            time_col="Scheduled DTTM",
            room_col="scheduled_room",
            start_date=start_date,
            end_date=end_date,
            exclude_iso_weeks=exclude_iso_weeks,
            label="blood_pressure"
        )
        self.generate_and_plot_distribution(
            original_df=self.medications_df,
            time_col="Medication Scheduled DTTM",
            room_col="scheduled_room",
            start_date=start_date,
            end_date=end_date,
            exclude_iso_weeks=exclude_iso_weeks,
            label="medications"
        )
        self.generate_and_plot_distribution(
            original_df=self.hr_df,
            time_col="Scheduled DTTM",
            room_col="scheduled_room",
            start_date=start_date,
            end_date=end_date,
            exclude_iso_weeks=exclude_iso_weeks,
            label="heart_rate"
        )
        self.generate_and_plot_distribution(
            original_df=self.rr_df,
            time_col="Scheduled DTTM",
            room_col="scheduled_room",
            start_date=start_date,
            end_date=end_date,
            exclude_iso_weeks=exclude_iso_weeks,
            label="respiratory_rate"
        )
        self.generate_and_plot_distribution(
            original_df=self.temp_df,
            time_col="Scheduled DTTM",
            room_col="scheduled_room",
            start_date=start_date,
            end_date=end_date,
            exclude_iso_weeks=exclude_iso_weeks,
            label="temperature"
        )
        self.generate_and_plot_distribution(
            original_df=self.oximetry_df,
            time_col="Scheduled DTTM",
            room_col="scheduled_room",
            start_date=start_date,
            end_date=end_date,
            exclude_iso_weeks=exclude_iso_weeks,
            label="oxygen_saturation"
        )
        self.generate_and_plot_distribution(
            original_df=self.admissions_discharges_df,
            time_col="HOSPITAL_ADMISSION",
            room_col="IN_DEP",
            start_date=start_date,
            end_date=end_date,
            exclude_iso_weeks=exclude_iso_weeks,
            label="admissions"
        )
        self.generate_and_plot_distribution(
            original_df=self.admissions_discharges_df,
            time_col="HOSPITAL_DISCHARGE",
            room_col="OUT_DEP",
            start_date=start_date,
            end_date=end_date,
            exclude_iso_weeks=exclude_iso_weeks,
            label="discharges"
        )
    
    def generate_and_plot_weekly_u_chart(self,
                                         original_df: pd.DataFrame,
                                         time_col: str,
                                         room_col: str,
                                         start_date: str,
                                         end_date: str,
                                         label: str) -> None:
        df_prep = DataHelpers.prepare_df(
            original_df,
            time_col=time_col,
            room_col=room_col,
        )

        df_filtered = DataHelpers.apply_date_filters(
            df_prep,
            start_date=start_date,
            end_date=end_date,
        )

        per_day = DataHelpers.compute_per_day_counts(dff=df_filtered)

        weekly = DataHelpers.weekly_u_chart(per_day)

        out_png = self.u_chart_outdir / f"{label}_weekly_u_chart_{start_date}_{end_date}.png"
        DataStatisticsPlottingHelper.plot_weekly_u_chart(
            weekly=weekly,
            out_png=out_png
        )
    
    def generate_and_plot_all_weekly_u_charts(self,
                                            start_date: str,
                                            end_date: str) -> None:
        self.generate_and_plot_weekly_u_chart(
            original_df=self.bp_df,
            time_col="Scheduled DTTM",
            room_col="scheduled_room",
            start_date=start_date,
            end_date=end_date,
            label="blood_pressure"
        )
        self.generate_and_plot_weekly_u_chart(
            original_df=self.medications_df,
            time_col="Medication Scheduled DTTM",
            room_col="scheduled_room",
            start_date=start_date,
            end_date=end_date,
            label="medications"
        )
        self.generate_and_plot_weekly_u_chart(
            original_df=self.hr_df,
            time_col="Scheduled DTTM",
            room_col="scheduled_room",
            start_date=start_date,
            end_date=end_date,
            label="heart_rate"
        )
        self.generate_and_plot_weekly_u_chart(
            original_df=self.rr_df,
            time_col="Scheduled DTTM",
            room_col="scheduled_room",
            start_date=start_date,
            end_date=end_date,
            label="respiratory_rate"
        )
        self.generate_and_plot_weekly_u_chart(
            original_df=self.temp_df,
            time_col="Scheduled DTTM",
            room_col="scheduled_room",
            start_date=start_date,
            end_date=end_date,
            label="temperature"
        )
        self.generate_and_plot_weekly_u_chart(
            original_df=self.oximetry_df,
            time_col="Scheduled DTTM",
            room_col="scheduled_room",
            start_date=start_date,
            end_date=end_date,
            label="oxygen_saturation"
        )
        self.generate_and_plot_weekly_u_chart(
            original_df=self.admissions_discharges_df,
            time_col="HOSPITAL_ADMISSION",
            room_col="IN_DEP",
            start_date=start_date,
            end_date=end_date,
            label="admissions"
        )
        self.generate_and_plot_weekly_u_chart(
            original_df=self.admissions_discharges_df,
            time_col="HOSPITAL_DISCHARGE",
            room_col="OUT_DEP",
            start_date=start_date,
            end_date=end_date,
            label="discharges"
        )
    
    def generate_and_plot_heatmap(self,
                                        original_df: pd.DataFrame,
                                        time_col: str,
                                        room_col: str,
                                        start_date: str,
                                        end_date: str,
                                        label: str) -> None:
            df_prep = DataHelpers.prepare_df(
                original_df,
                time_col=time_col,
                room_col=room_col,
            )
    
            df_filtered = DataHelpers.apply_date_filters(
                df_prep,
                start_date=start_date,
                end_date=end_date,
            )
    
            weekly_dow = DataHelpers.compute_week_by_dow(df_filtered)
    
            out_png = self.heatmap_outdir / f"{label}_floor_week_heatmap_{start_date}_{end_date}.png"
            DataStatisticsPlottingHelper.plot_heatmap(
                weekly_dow=weekly_dow,
                out_png=out_png
            )
    
    def generate_and_plot_all_heatmaps(self,
                                     start_date: str,
                                     end_date: str) -> None:
        self.generate_and_plot_heatmap(
            original_df=self.bp_df,
            time_col="Scheduled DTTM",
            room_col="scheduled_room",
            start_date=start_date,
            end_date=end_date,
            label="blood_pressure"
        )
        self.generate_and_plot_heatmap(
            original_df=self.medications_df,
            time_col="Medication Scheduled DTTM",
            room_col="scheduled_room",
            start_date=start_date,
            end_date=end_date,
            label="medications"
        )
        self.generate_and_plot_heatmap(
            original_df=self.hr_df,
            time_col="Scheduled DTTM",
            room_col="scheduled_room",
            start_date=start_date,
            end_date=end_date,
            label="heart_rate"
        )
        self.generate_and_plot_heatmap(
            original_df=self.rr_df,
            time_col="Scheduled DTTM",
            room_col="scheduled_room",
            start_date=start_date,
            end_date=end_date,
            label="respiratory_rate"
        )
        self.generate_and_plot_heatmap(
            original_df=self.temp_df,
            time_col="Scheduled DTTM",
            room_col="scheduled_room",
            start_date=start_date,
            end_date=end_date,
            label="temperature"
        )
        self.generate_and_plot_heatmap(
            original_df=self.oximetry_df,
            time_col="Scheduled DTTM",
            room_col="scheduled_room",
            start_date=start_date,
            end_date=end_date,
            label="oxygen_saturation"
        )
        self.generate_and_plot_heatmap(
            original_df=self.admissions_discharges_df,
            time_col="HOSPITAL_ADMISSION",
            room_col="IN_DEP",
            start_date=start_date,
            end_date=end_date,
            label="admissions"
        )
        self.generate_and_plot_heatmap(
            original_df=self.admissions_discharges_df,
            time_col="HOSPITAL_DISCHARGE",
            room_col="OUT_DEP",
            start_date=start_date,
            end_date=end_date,
            label="discharges"
        )
    
    def generate_weekly_distributions(self,
                                      original_df: pd.DataFrame,
                                      time_col: str,
                                      room_col: str,
                                      start_date: str,
                                      end_date: str,
                                      label: str,
                                      top_k: int) -> pd.DataFrame:
        df_prep = DataHelpers.prepare_df(
            original_df,
            time_col=time_col,
            room_col=room_col,
        )

        df_filtered = DataHelpers.apply_date_filters(
            df_prep,
            start_date=start_date,
            end_date=end_date,
        )

        per_day = DataHelpers.compute_per_day_counts(dff=df_filtered)

        weekly_u_chart = DataHelpers.weekly_u_chart(per_day)

        # List ISO weeks
        weeks = (per_day[["iso_year","iso_week"]].drop_duplicates()
                .sort_values(["iso_year","iso_week"])
                .to_records(index=False))

        results = []

        for (yy, ww) in weeks:
            week_mask = (per_day["iso_year"] == int(yy)) & (per_day["iso_week"] == int(ww))
            dist_week = DataHelpers.distribution_of_counts(per_day[week_mask])

            u_chart_data = weekly_u_chart[(weekly_u_chart["iso_year"] == int(yy)) & (weekly_u_chart["iso_week"] == int(ww))]
            if not u_chart_data.empty:
                u_t = float(u_chart_data["u"].values[0])
                lcl = float(u_chart_data["lcl"].values[0])
                if u_t < lcl:
                    continue
            else:
                continue

            others_dist = DataHelpers.distribution_of_counts(per_day[~week_mask])
            pmf_base = DataHelpers.pmf_from_dist(others_dist)

            wdist = DataHelpers.wasserstein_1_intbins(DataHelpers.pmf_from_dist(dist_week), pmf_base)
            results.append({"iso_year": int(yy), "iso_week": int(ww), "wasserstein": wdist})

        results_df = pd.DataFrame(results).sort_values("wasserstein", ascending=False)

          # Take top-K weeks
        top = results_df.head(top_k)
        # Optional: bar chart of top-K distances
        out_png = self.wass_outdir / f"{label}_wasserstein_distances.png"
        if not top.empty:
            DataStatisticsPlottingHelper.plot_wasserstein_distance(top, out_png)

        for _, row in top.iterrows(): 
            yy, ww = int(row["iso_year"]), int(row["iso_week"])
            focal = per_day[(per_day["iso_year"] == yy) & (per_day["iso_week"] == ww)]
            others = per_day[~((per_day["iso_year"] == yy) & (per_day["iso_week"] == ww))]

            # Compute distributions
            dist_focal = DataHelpers.distribution_of_counts(focal)
            dist_others = DataHelpers.distribution_of_counts(others)

            # Align categories (bins) for plotting comparability
            if dist_focal.empty and dist_others.empty:
                # Nothing to plot; create empty charts for consistency
                cats = []
                aligned_focal = pd.DataFrame(columns=["requests_per_day","relative_frequency"])
                aligned_others = aligned_focal.copy()
            else:
                aligned_focal, aligned_others, cats = DataHelpers.align_distributions(dist_focal, dist_others)
            
            new_label = f"{label}_{yy}W{ww:02d}"
            focal_out_png = self.indiv_dist_outdir / f"{new_label}_focal.png"
            others_out_png = self.indiv_dist_outdir / f"{new_label}_others.png"
            title_focal = f"Week {new_label}: relative frequency of requests per floor-day"
            title_others = f"All other weeks (excl. {new_label}): relative frequency of requests per floor-day"

            if len(cats) == 0:
                # No data to plot
                print(f"No data for weekly distribution plot: {new_label}")
            else:
                DataStatisticsPlottingHelper.plot_rel_freq_bar(categories=cats,
                                                               rel_freq=aligned_focal["relative_frequency"],
                                                               title=title_focal,
                                                               out_png=focal_out_png)
                DataStatisticsPlottingHelper.plot_rel_freq_bar(categories=cats,
                                                               rel_freq=aligned_others["relative_frequency"],
                                                               title=title_others,
                                                               out_png=others_out_png)
    
    def generate_and_plot_all_weekly_distributions(self,
                                                 start_date: str,
                                                 end_date: str,
                                                 top_k: int) -> None:
        self.generate_weekly_distributions(
            original_df=self.bp_df,
            time_col="Scheduled DTTM",
            room_col="scheduled_room",
            start_date=start_date,
            end_date=end_date,
            label="blood_pressure",
            top_k=top_k
        )
        self.generate_weekly_distributions(
            original_df=self.medications_df,
            time_col="Medication Scheduled DTTM",
            room_col="scheduled_room",
            start_date=start_date,
            end_date=end_date,
            label="medications",
            top_k=top_k
        )
        self.generate_weekly_distributions(
            original_df=self.hr_df,
            time_col="Scheduled DTTM",
            room_col="scheduled_room",
            start_date=start_date,
            end_date=end_date,
            label="heart_rate",
            top_k=top_k
        )
        self.generate_weekly_distributions(
            original_df=self.rr_df,
            time_col="Scheduled DTTM",
            room_col="scheduled_room",
            start_date=start_date,
            end_date=end_date,
            label="respiratory_rate",
            top_k=top_k
        )
        self.generate_weekly_distributions(
            original_df=self.temp_df,
            time_col="Scheduled DTTM",
            room_col="scheduled_room",
            start_date=start_date,
            end_date=end_date,
            label="temperature",
            top_k=top_k
        )
        self.generate_weekly_distributions(
            original_df=self.oximetry_df,
            time_col="Scheduled DTTM",
            room_col="scheduled_room",
            start_date=start_date,
            end_date=end_date,
            label="oxygen_saturation",
            top_k=top_k
        )
        self.generate_weekly_distributions(
            original_df=self.admissions_discharges_df,
            time_col="HOSPITAL_ADMISSION",
            room_col="IN_DEP",
            start_date=start_date,
            end_date=end_date,
            label="admissions",
            top_k=top_k
        )
        self.generate_weekly_distributions(
            original_df=self.admissions_discharges_df,
            time_col="HOSPITAL_DISCHARGE",
            room_col="OUT_DEP",
            start_date=start_date,
            end_date=end_date,
            label="discharges",
            top_k=top_k
        )
    
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

    data_stats = DataStatistics(
        outdir=args.outdir,
        dist_data_dir=args.dist_data_dir,
        annotated_data_files=annotated_data_files,
    )

    data_stats.generate_and_plot_all_distributions(
        start_date="2024-06-24",
        end_date="2025-06-29",
        exclude_iso_weeks=[],
    )

    data_stats.generate_and_plot_all_weekly_u_charts(
        start_date="2024-06-24",
        end_date="2025-06-29"
    )

    data_stats.generate_and_plot_all_heatmaps(
        start_date="2024-06-24",
        end_date="2025-06-29"
    )

    data_stats.generate_and_plot_all_weekly_distributions(
        start_date="2024-06-24",
        end_date="2025-06-29",
        top_k=5
    )

    
if __name__ == "__main__":
    pStart = datetime.now()
    try:
        main()
    except Exception as errorMainContext:
        print("Fail End Process: ", errorMainContext)
        traceback.print_exc()
    pEnd = datetime.now()
    print(f"Total Execution Time: {pEnd - pStart}")