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
    def apply_online_requests_filters(df: pd.DataFrame, sched_time_label: str, ordered_time_label: str) -> pd.DataFrame:
        df[sched_time_label] = pd.to_datetime(df[sched_time_label], errors="coerce")
        df[ordered_time_label]   = pd.to_datetime(df[ordered_time_label], errors="coerce")

        # Compute delta and filter
        delta = df[sched_time_label] - df[ordered_time_label]
        mask = delta < pd.Timedelta(minutes=30)
        filtered = df.loc[mask].copy()

        return filtered
    
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
    def build_weekly_per_task(per_day_all: pd.DataFrame) -> pd.DataFrame:
        """
        per_day_all expected columns:
        __task__, __floor__, __day__, n_requests, iso_year, iso_week

        Returns wk with:
        iso_year, iso_week, __task__, O_it (weekly total), n_it (exposure = # floor-days for that task-week)
        """
        if per_day_all.empty:
            return pd.DataFrame(columns=["iso_year","iso_week","__task__","O_it","n_it"])

        wk = (per_day_all.groupby(["iso_year","iso_week","__task__"], as_index=False)
                        .agg(
                            O_it=("n_requests","sum"),
                            n_it=("n_requests","size"),
                        ))
        wk["iso_year"] = wk["iso_year"].astype(int)
        wk["iso_week"] = wk["iso_week"].astype(int)
        return wk
    
    @staticmethod
    def equal_task_influence_oe_chart(wk: pd.DataFrame,
                                      min_expected: float = 0.2,
                                      min_tasks_per_week: int = 1) -> pd.DataFrame:
        """
        Equal-influence combined chart:
        - compute per task-week expected E_it from task baseline
        - compute z_it = (O_it - E_it) / sqrt(E_it)
        - per week, Z = mean(z_it) (each task counts equally)
        - limits: ± 3 / sqrt(k) 

        min_expected:
        drop task-weeks with E_it < min_expected (too noisy)
        min_tasks_per_week:
        drop weeks with fewer than this many tasks contributing
        """
        if wk.empty:
            return pd.DataFrame(columns=["iso_year","iso_week","week_start","Z","k_tasks","ucl","lcl","flag"])

        w = wk.copy()

        # Baseline rate per task
        base = (w.groupby("__task__", as_index=False)
                .agg(O_tot=("O_it","sum"), n_tot=("n_it","sum")))
        base["lambda_hat"] = base["O_tot"] / base["n_tot"]

        w = w.merge(base[["__task__", "lambda_hat"]], on="__task__", how="left")

        # Expected + Pearson residual per task-week
        w["E_it"] = w["n_it"] * w["lambda_hat"]
        w = w[w["E_it"] >= float(min_expected)].copy()

        w["z_it"] = (w["O_it"] - w["E_it"]) / np.sqrt(w["E_it"])

        # Equal-weight combine across tasks per week
        weekly = (w.groupby(["iso_year","iso_week"], as_index=False)
                    .agg(
                        Z=("z_it", "mean"),
                        k_tasks=("z_it", "size"),
                    ))

        # Filter weeks with too few tasks
        weekly = weekly[weekly["k_tasks"] >= int(min_tasks_per_week)].copy()

        # Week start date for plotting
        weekly["week_start"] = pd.to_datetime(
            weekly["iso_year"].astype(str) + "-W" + weekly["iso_week"].astype(str).str.zfill(2) + "-1",
            format="%G-W%V-%u",
            errors="coerce",
        )
        weekly = weekly.sort_values("week_start").reset_index(drop=True)

        # Standard limits for mean of ~N(0,1) residuals
        weekly["ucl"] = 3.0 / np.sqrt(weekly["k_tasks"].astype(float))
        weekly["lcl"] = -weekly["ucl"]
        weekly["flag"] = (weekly["Z"] > weekly["ucl"]) | (weekly["Z"] < weekly["lcl"])

        return weekly
    
    @staticmethod
    def weekly_laney_u_chart(per_day: pd.DataFrame) -> pd.DataFrame:
        """
        Build weekly metrics for a Laney u' chart.
        Returns a DataFrame with columns:
        - iso_year, iso_week, week_start, n_units, total_requests, u,
          u_bar, sigma_z, ucl, lcl, flagged_high, flagged_low, flagged
        Stores u_bar and sigma_z in weekly.attrs.
        """
        if per_day.empty:
            return pd.DataFrame(columns=[
                "iso_year","iso_week","week_start","n_units","total_requests","u",
                "ucl","lcl","flagged_high","flagged_low","flagged"
            ])

        tmp = per_day.copy()

        # ISO year/week
        tmp["iso_year"] = tmp["__day__"].apply(lambda d: d.isocalendar().year)
        tmp["iso_week"] = tmp["__day__"].apply(lambda d: d.isocalendar().week)

        # Weekly totals
        g = tmp.groupby(["iso_year","iso_week"], as_index=False)
        weekly = g.agg(
            n_units=("n_requests", "size"),      # number of day-rows (your "units")
            total_requests=("n_requests", "sum")
        )

        weekly["u"] = np.where(weekly["n_units"] > 0,
                               weekly["total_requests"] / weekly["n_units"],
                               np.nan)

        # Center line (weighted)
        total_requests_all = weekly["total_requests"].sum()
        total_units_all = weekly["n_units"].sum()
        u_bar = 0.0 if total_units_all == 0 else total_requests_all / total_units_all

        # If u_bar == 0, Poisson variance is 0 => limits collapse; Laney can't help.
        if u_bar <= 0:
            weekly["ucl"] = 0.0
            weekly["lcl"] = 0.0
            weekly["flagged_high"] = weekly["u"] > 0.0
            weekly["flagged_low"] = False
            weekly["flagged"] = weekly["flagged_high"]
            weekly.attrs["u_bar"] = u_bar
            weekly.attrs["sigma_z"] = np.nan
        else:
            # Std error under Poisson for each week
            se = np.sqrt(np.where(weekly["n_units"] > 0, u_bar / weekly["n_units"], np.nan))

            # Standardized residuals
            z = (weekly["u"] - u_bar) / se
            weekly["z"] = z

            # ---- Laney sigma_z estimate (moving range of successive z's) ----
            # Use weeks in chronological order for MR.
            # Create week_start (Monday) first, then sort.
            tmp_dates = tmp.groupby(["iso_year","iso_week"])["__day__"].min().reset_index()
            tmp_dates["__day__"] = pd.to_datetime(tmp_dates["__day__"])
            tmp_dates["week_start"] = tmp_dates["__day__"] - pd.to_timedelta(tmp_dates["__day__"].dt.weekday, unit="D")
            weekly = weekly.merge(tmp_dates[["iso_year","iso_week","week_start"]],
                                  on=["iso_year","iso_week"], how="left")

            weekly = weekly.sort_values(["week_start","iso_year","iso_week"], ignore_index=True)

            # Recompute z in sorted order (same values, but aligned for diff)
            se_sorted = np.sqrt(np.where(weekly["n_units"] > 0, u_bar / weekly["n_units"], np.nan))
            z_sorted = (weekly["u"] - u_bar) / se_sorted

            mr = np.abs(np.diff(z_sorted.to_numpy()))
            mr = mr[~np.isnan(mr)]
            mr_bar = np.nan if mr.size == 0 else mr.mean()

            d2 = 1.128  # for moving range of 2
            sigma_z = np.nan if (mr_bar is np.nan) else (mr_bar / d2)

            # Guardrails: sigma_z should be positive; if weird, fall back to 1.0
            if not np.isfinite(sigma_z) or sigma_z <= 0:
                sigma_z = 1.0

            weekly["sigma_z"] = sigma_z

            # Laney limits
            weekly["ucl"] = u_bar + 3.0 * sigma_z * se_sorted
            weekly["lcl"] = np.maximum(0.0, u_bar - 3.0 * sigma_z * se_sorted)

            weekly["flagged_high"] = weekly["u"] > weekly["ucl"]
            weekly["flagged_low"] = weekly["u"] < weekly["lcl"]
            weekly["flagged"] = weekly["flagged_high"] | weekly["flagged_low"]

            weekly.attrs["u_bar"] = u_bar
            weekly.attrs["sigma_z"] = sigma_z

        # Keep your final sorting columns consistent
        weekly = weekly.sort_values(["week_start","iso_year","iso_week"], ignore_index=True)
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