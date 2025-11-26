import argparse
from cProfile import label
from datetime import datetime
import re
import traceback

from pathlib import Path
from typing import Dict, Optional, Sequence, List, Tuple

import numpy as np
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
    def parse_iso_week_args(vals: Sequence[str]) -> List[Tuple[int, int]]:
        """
        Parse values like '2024W26', '2024-W26', '2024,26', or '2024-26' into (year, week).
        """
        out: List[Tuple[int, int]] = []
        for v in vals:
            v = v.strip()
            if not v:
                continue
            cleaned = (v.replace("W", "-")
                        .replace(",", "-")
                        .replace("_", "-"))
            parts = cleaned.split("-")
            parts = [p for p in parts if p]
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                out.append((int(parts[0]), int(parts[1])))
            else:
                raise ValueError(f"Could not parse ISO week value: '{v}'. Try formats like 2024W26 or 2024-W26.")
        return out
    
    @staticmethod
    def prepare_df(df: pd.DataFrame, time_col: str, room_col: str) -> pd.DataFrame:
        # Parse datetime columns if present
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

        if time_col not in df.columns:
            raise KeyError(f"TIME_COL '{time_col}' not in CSV columns: {list(df.columns)}")
        if room_col not in df.columns:
            raise KeyError(f"ROOM_COL '{room_col}' not in CSV columns: {list(df.columns)}")

        dff = df.copy()
        dff["__day__"] = dff[time_col].dt.date
        dff["__floor__"] = dff[room_col].apply(lambda x: DataStatisticsHelpers.extract_floor(x))
        dff = dff.dropna(subset=["__day__", "__floor__"])

        return dff
    
    @staticmethod
    def apply_date_filters(df: pd.DataFrame, 
                           start_date: Optional[str] = None, 
                           end_date: Optional[str] = None,
                           exclude_iso_weeks: Optional[Sequence[Tuple[int, int]]] = None) -> pd.DataFrame:
        dff = df.copy()

        # Inclusive date range
        if start_date:
            sd = pd.to_datetime(start_date).date()
            dff = dff[dff["__day__"] >= sd]
        if end_date:
            ed = pd.to_datetime(end_date).date()
            dff = dff[dff["__day__"] <= ed]

        # Exclude ISO weeks
        if exclude_iso_weeks:
            iso_year = dff["__day__"].apply(lambda d: d.isocalendar().year)
            iso_week = dff["__day__"].apply(lambda d: d.isocalendar().week)
            mask = pd.Series(True, index=dff.index)
            for (y, w) in exclude_iso_weeks:
                mask &= ~((iso_year == y) & (iso_week == w))
            dff = dff[mask]

        # Attach ISO year & week for later grouping
        dff["iso_year"] = dff["__day__"].apply(lambda d: d.isocalendar().year)
        dff["iso_week"] = dff["__day__"].apply(lambda d: d.isocalendar().week)

        return dff

    @staticmethod
    def compute_per_day_counts(dff: pd.DataFrame) -> pd.DataFrame:
        """
        Returns one row per (floor, day) with n_requests and ISO week/year attached.
        """
        if dff.empty:
            return pd.DataFrame(columns=["__floor__", "__day__", "n_requests", "iso_year", "iso_week"])
        # group by floor-day
        g = dff.groupby(["__floor__", "__day__"], as_index=False)
        per_day = g.size().rename(columns={"size": "n_requests"})
        # Re-attach ISO fields by merging from unique day -> (year, week)
        unique_days = (dff[["__day__", "iso_year", "iso_week"]].drop_duplicates())
        per_day = per_day.merge(unique_days, on="__day__", how="left")
        return per_day
    
    @staticmethod  
    def distribution_of_counts(per_day: pd.DataFrame) -> pd.DataFrame:
        """
        Compute distribution of n_requests across floor-days.
        Returns columns: requests_per_day, absolute_count, relative_frequency
        """
        if per_day.empty:
            return pd.DataFrame(columns=["requests_per_day", "absolute_count", "relative_frequency"])
        dist = (per_day["n_requests"]
                .value_counts()
                .sort_index()
                .rename_axis("requests_per_day")
                .reset_index(name="absolute_count"))
        total = dist["absolute_count"].sum()
        dist["relative_frequency"] = 0.0 if total == 0 else dist["absolute_count"] / total
        dist["requests_per_day"] = dist["requests_per_day"].astype(int)
        return dist
    
    @staticmethod
    def pmf_from_dist(dist: pd.DataFrame) -> dict[int, float]:
        return {int(r): float(p) for r, p in zip(dist["requests_per_day"], dist["relative_frequency"])}
    
    @staticmethod
    def wasserstein_1_intbins(pmf1: dict[int, float], pmf2: dict[int, float]) -> float:
        """
        1D Wasserstein distance for discrete integer supports (unit spacing).
        W1 = sum_i |F1(x_i) - F2(x_i)| * (x_{i+1} - x_i), with unit steps -> sum |ΔCDF|.
        """
        if not pmf1 and not pmf2:
            return 0.0
        keys = sorted(set(pmf1.keys()) | set(pmf2.keys()))
        # enforce full integer range (monotone CDF step at each integer)
        lo, hi = int(min(keys)), int(max(keys))
        xs = list(range(lo, hi + 1))
        p1 = np.array([pmf1.get(x, 0.0) for x in xs], dtype=float)
        p2 = np.array([pmf2.get(x, 0.0) for x in xs], dtype=float)
        F1 = np.cumsum(p1)
        F2 = np.cumsum(p2)
        # step size = 1 between consecutive integers
        return float(np.sum(np.abs(F1 - F2)))
    
    @staticmethod
    def align_distributions(dist_a: pd.DataFrame, dist_b: pd.DataFrame):
        """
        Align two distributions to a common set of integer categories (union of requests_per_day).
        Returns (dist_a_aligned, dist_b_aligned, categories)
        """
        cats = sorted(set(dist_a["requests_per_day"].tolist()) | set(dist_b["requests_per_day"].tolist()))
        def reindex(dist):
            out = pd.DataFrame({"requests_per_day": cats})
            out = out.merge(dist[["requests_per_day", "relative_frequency"]], on="requests_per_day", how="left")
            out["relative_frequency"] = out["relative_frequency"].fillna(0.0)
            return out
        return reindex(dist_a), reindex(dist_b), cats
    
    @staticmethod
    def align_for_plot(dist_a: pd.DataFrame, dist_b: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (bins, relA, relB) aligned on the union of integer bins for plotting.
        """
        keys = sorted(set(dist_a["requests_per_day"].astype(int).tolist()) |
                    set(dist_b["requests_per_day"].astype(int).tolist()))
        if not keys:
            return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=float)
        lo, hi = min(keys), max(keys)
        bins = np.arange(lo, hi + 1, dtype=int)
        pmfa = DataStatisticsHelpers.pmf_from_dist(dist_a)
        pmfb = DataStatisticsHelpers.pmf_from_dist(dist_b)
        relA = np.array([pmfa.get(int(k), 0.0) for k in bins], dtype=float)
        relB = np.array([pmfb.get(int(k), 0.0) for k in bins], dtype=float)
        return bins, relA, relB
    
    @staticmethod
    def weekly_u_chart(per_day: pd.DataFrame) -> pd.DataFrame:
        """
        Build weekly metrics for u-chart.
        Returns a DataFrame with columns:
        - iso_year, iso_week, n_units, total_requests, u, ucl, lcl, flagged
        - week_start (Monday) for plotting/sorting
        """
        if per_day.empty:
            return pd.DataFrame(columns=[
                "iso_year","iso_week","week_start","n_units","total_requests","u","ucl","lcl","flagged"
            ])

        # Add ISO year/week
        tmp = per_day.copy()
        tmp["iso_year"] = tmp["__day__"].apply(lambda d: d.isocalendar().year)
        tmp["iso_week"] = tmp["__day__"].apply(lambda d: d.isocalendar().week)

        # Compute weekly totals
        g = tmp.groupby(["iso_year","iso_week"], as_index=False)
        weekly = g.agg(
            n_units=("n_requests","size"),               # number of floor-days
            total_requests=("n_requests","sum"),         # total requests across floor-days
        )
        # u_t per week
        weekly["u"] = weekly["total_requests"] / weekly["n_units"]

        # Weighted center line u_bar
        total_requests_all = weekly["total_requests"].sum()
        total_units_all = weekly["n_units"].sum()
        u_bar = 0.0 if total_units_all == 0 else total_requests_all / total_units_all

        # Control limits depend on n_units_t
        weekly["ucl"] = u_bar + 3.0 * np.sqrt(np.where(weekly["n_units"]>0, u_bar/weekly["n_units"], 0.0))
        weekly["lcl"] = np.maximum(0.0, u_bar - 3.0 * np.sqrt(np.where(weekly["n_units"]>0, u_bar/weekly["n_units"], 0.0)))
        weekly["flagged"] = weekly["u"] > weekly["ucl"]

        # Add week_start (Monday) for plotting
        tmp_dates = tmp.groupby(["iso_year","iso_week"])["__day__"].min().reset_index()
        tmp_dates["__day__"] = pd.to_datetime(tmp_dates["__day__"])
        tmp_dates["week_start"] = tmp_dates["__day__"] - pd.to_timedelta(tmp_dates["__day__"].dt.weekday, unit="D")
        weekly = weekly.merge(tmp_dates[["iso_year","iso_week","week_start"]], on=["iso_year","iso_week"], how="left")

        weekly = weekly.sort_values(["week_start","iso_year","iso_week"], ignore_index=True)
        weekly.attrs["u_bar"] = u_bar
        return weekly
    
    @staticmethod
    def compute_week_by_dow(df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a long table of total_requests per (iso_year, iso_week, dow)
        where totals are summed across all floors.
        """
        if df.empty:
            return pd.DataFrame(columns=[
                "iso_year", "iso_week", "week_start", "dow", "dow_name", "date", "total_requests"
            ])

        # per (floor, day) => n_requests
        per_day = (
            df.groupby(["__floor__", "__day__"], as_index=False)
            .size()
            .rename(columns={"size": "n_requests"})
        )

        # collapse across floors: total per calendar day
        daily_totals = per_day.groupby("__day__", as_index=False)["n_requests"].sum().rename(
            columns={"n_requests": "total_requests"}
        )

        # ISO fields & week_start & day-of-week
        daily_totals["date"] = pd.to_datetime(daily_totals["__day__"])
        daily_totals["iso_year"] = daily_totals["date"].dt.isocalendar().year.astype(int)
        daily_totals["iso_week"] = daily_totals["date"].dt.isocalendar().week.astype(int)
        daily_totals["dow"] = daily_totals["date"].dt.weekday.astype(int)   # Monday=0
        daily_totals["dow_name"] = daily_totals["dow"].map(lambda d: {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}[d])

        # compute week_start (Monday) for ordering
        daily_totals["week_start"] = daily_totals["date"] - pd.to_timedelta(daily_totals["date"].dt.weekday, unit="D")

        # aggregate in case there are multiple records per (week,dow)
        weekly_dow = (daily_totals
                    .groupby(["iso_year","iso_week","week_start","dow","dow_name"], as_index=False)["total_requests"]
                    .sum())

        return weekly_dow


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
    
    def _load_visits_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.annotated_data_files.annotated_visits)
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
        df_prep = DataStatisticsHelpers.prepare_df(
            original_df,
            time_col=time_col,
            room_col=room_col,
        )

        df_filtered = DataStatisticsHelpers.apply_date_filters(
            df_prep,
            start_date=start_date,
            end_date=end_date,
            exclude_iso_weeks=exclude_iso_weeks,
        )

        per_day = DataStatisticsHelpers.compute_per_day_counts(dff=df_filtered)

        dist = DataStatisticsHelpers.distribution_of_counts(per_day=per_day)

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
        df_prep = DataStatisticsHelpers.prepare_df(
            original_df,
            time_col=time_col,
            room_col=room_col,
        )

        df_filtered = DataStatisticsHelpers.apply_date_filters(
            df_prep,
            start_date=start_date,
            end_date=end_date,
        )

        per_day = DataStatisticsHelpers.compute_per_day_counts(dff=df_filtered)

        weekly = DataStatisticsHelpers.weekly_u_chart(per_day)

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
            df_prep = DataStatisticsHelpers.prepare_df(
                original_df,
                time_col=time_col,
                room_col=room_col,
            )
    
            df_filtered = DataStatisticsHelpers.apply_date_filters(
                df_prep,
                start_date=start_date,
                end_date=end_date,
            )
    
            weekly_dow = DataStatisticsHelpers.compute_week_by_dow(df_filtered)
    
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
        df_prep = DataStatisticsHelpers.prepare_df(
            original_df,
            time_col=time_col,
            room_col=room_col,
        )

        df_filtered = DataStatisticsHelpers.apply_date_filters(
            df_prep,
            start_date=start_date,
            end_date=end_date,
        )

        per_day = DataStatisticsHelpers.compute_per_day_counts(dff=df_filtered)

        weekly_u_chart = DataStatisticsHelpers.weekly_u_chart(per_day)

        # List ISO weeks
        weeks = (per_day[["iso_year","iso_week"]].drop_duplicates()
                .sort_values(["iso_year","iso_week"])
                .to_records(index=False))

        results = []

        for (yy, ww) in weeks:
            week_mask = (per_day["iso_year"] == int(yy)) & (per_day["iso_week"] == int(ww))
            dist_week = DataStatisticsHelpers.distribution_of_counts(per_day[week_mask])

            u_chart_data = weekly_u_chart[(weekly_u_chart["iso_year"] == int(yy)) & (weekly_u_chart["iso_week"] == int(ww))]
            if not u_chart_data.empty:
                u_t = float(u_chart_data["u"].values[0])
                lcl = float(u_chart_data["lcl"].values[0])
                if u_t < lcl:
                    continue
            else:
                continue

            others_dist = DataStatisticsHelpers.distribution_of_counts(per_day[~week_mask])
            pmf_base = DataStatisticsHelpers.pmf_from_dist(others_dist)

            wdist = DataStatisticsHelpers.wasserstein_1_intbins(DataStatisticsHelpers.pmf_from_dist(dist_week), pmf_base)
            results.append({"iso_year": int(yy), "iso_week": int(ww), "wasserstein": wdist})

        results_df = pd.DataFrame(results).sort_values("wasserstein", ascending=False)

          # Take top-K weeks
        top = results_df.head(top_k)
        # Optional: bar chart of top-K distances
        out_png = self.outdir / f"{label}_wasserstein_distances.png"
        if not top.empty:
            DataStatisticsPlottingHelper.plot_wasserstein_distance(top, out_png)

        for _, row in top.iterrows(): 
            yy, ww = int(row["iso_year"]), int(row["iso_week"])
            focal = per_day[(per_day["iso_year"] == yy) & (per_day["iso_week"] == ww)]
            others = per_day[~((per_day["iso_year"] == yy) & (per_day["iso_week"] == ww))]

            # Compute distributions
            dist_focal = DataStatisticsHelpers.distribution_of_counts(focal)
            dist_others = DataStatisticsHelpers.distribution_of_counts(others)

            # Align categories (bins) for plotting comparability
            if dist_focal.empty and dist_others.empty:
                # Nothing to plot; create empty charts for consistency
                cats = []
                aligned_focal = pd.DataFrame(columns=["requests_per_day","relative_frequency"])
                aligned_others = aligned_focal.copy()
            else:
                aligned_focal, aligned_others, cats = DataStatisticsHelpers.align_distributions(dist_focal, dist_others)
            
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