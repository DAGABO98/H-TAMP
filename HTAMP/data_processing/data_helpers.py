import re
from typing import Optional, Sequence, List, Tuple

import numpy as np
import pandas as pd

class DataHelpers:

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
        dff["__floor__"] = dff[room_col].apply(lambda x: DataHelpers.extract_floor(x))
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
    def get_daily_requests_for_floor(df: pd.DataFrame, date_stamp: pd.Timestamp, floor_number: int) -> pd.DataFrame:
        """
        Extract all requests for a specific date_stamp (date) and floor number.
        """
        target_date = date_stamp.date()
        daily_df = df[(df["__day__"] == target_date) & (df["__floor__"] == floor_number)].copy()
        return daily_df

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
        pmfa = DataHelpers.pmf_from_dist(dist_a)
        pmfb = DataHelpers.pmf_from_dist(dist_b)
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