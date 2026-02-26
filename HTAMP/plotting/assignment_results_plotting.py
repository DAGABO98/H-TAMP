import argparse
import os
import re
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt


class AssignmentResultsPlotter:
    """
    Reads per-policy folders of CSVs.
    Each CSV filename encodes day and floor, e.g. '2024-9-30_floor2.csv'.

    Computes:
      - counts per (policy, request_type): rejected/serviced/total
      - serviced wait time = max(0, planned_time - scheduled_time)
      - daily-per-floor summary statistic (mean or p95) per (policy, request_type, day, floor)

    Plots:
      - for each request_type: boxplot across policies using daily-per-floor stats
    """

    def __init__(
        self,
        root_dir: str | Path,
        out_dir: str | Path = "results/policies/plots",
        type_col: str = "request_type",
        daily_stat: str = "p95",
        logs_root_dir: str | Path = "results/policies/logs",
        file_glob: str = "*.out"
    ):
        self.policy_order = ["fleet_manager",
                             "tp_d_alpha0.0",
                             "d_tpts_alpha0.0",
                             "tp_d_alpha0.1",
                             "d_tpts_alpha0.1",
                             "tp_d_alpha0.2",
                             "d_tpts_alpha0.2",
                             "sequential_greedy_nopt",
                             "sequential_greedy_ropt"]

        self.baseline_colors = {"human_team": "#7f0000",
                                "fleet_manager": "#b30000",
                                "tp_d_alpha0.0": "#d7301f",
                                "d_tpts_alpha0.0": "#ef6548",
                                "tp_d_alpha0.1": "#fc8d59",
                                "d_tpts_alpha0.1": "#fdbb84",
                                "tp_d_alpha0.2": "#fdd49e",
                                "d_tpts_alpha0.2": "#fee8c8",
                                "idle_prediction": "#fdd49e",
                                "vanilla_rollout": "#fee8c8"}
        
        self.our_methods_colors = {"sequential_greedy_nopt": "#d9f0a3",
                                   "sequential_greedy_ropt": "#006837",
                                    "adaptive_rollout": "#006837"}
        
        self.ablation_study_colors = {"sequential_greedy_nopt": "#d9f0a3",
                                     "weighting_nopt": "#addd8e",
                                     "no_weighting_opt": "#78c679",
                                     "no_weighting_nopt": "#31a354",
                                     "adaptive_rollout": "#006837"}
        
        self.week_buckets: dict[str, set[tuple[int, int]]] = {
            "highest": {(2024, 40), (2025, 5)},
            "medium":  {(2024, 44), (2025, 14)},
            "lowest":  {(2024, 27), (2024, 36)},
        }
        self.root_dir = Path(root_dir)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.logs_root_dir = Path(logs_root_dir)
        self.file_glob = file_glob

        self.type_col = type_col
        self.daily_stat = daily_stat.lower()
        if self.daily_stat not in ("mean", "p95"):
            raise ValueError("daily_stat must be 'mean' or 'p95'")

        (
            self.summary_by_type,
            self.dailyfloor_wait_stats_by_policy_type,
            self.dailyfloor_stats_df,
        ) = self._generate_results_summary_from_root_dir()
        iso = self.dailyfloor_stats_df["_day"].dt.isocalendar()
        self.dailyfloor_stats_df["iso_year"] = iso["year"].astype(int)
        self.dailyfloor_stats_df["iso_week"] = iso["week"].astype(int)

        self.logs_df = self._parse_all_logs()
        iso = self.logs_df["day"].dt.isocalendar()
        self.logs_df["iso_year"] = iso["year"].astype(int)
        self.logs_df["iso_week"] = iso["week"].astype(int)

        self.raw_requests_df = self._load_raw_requests_df()  

    @staticmethod
    def _safe_filename(x: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x))

    @staticmethod
    def _parse_day_floor_from_filename(f: Path) -> tuple[pd.Timestamp, int]:
        """
        Parse filename like '2024-9-30_floor2.csv' or '2024-09-30_floor2.csv'
        -> (day, floor)
        """
        m = re.search(r"(\d{4}-\d{1,2}-\d{1,2})_floor(\d+)", f.stem)
        if not m:
            raise ValueError(
                f"Could not parse day/floor from filename: {f.name}\n"
                "Expected something like 'YYYY-M-D_floorN.csv' (e.g. 2024-9-30_floor2.csv)"
            )
        day = pd.to_datetime(m.group(1), format="%Y-%m-%d", errors="raise").floor("D")
        floor = int(m.group(2))
        return day, floor
    
    @staticmethod
    def _extract_value(pattern: str, text: str) -> str | None:
        m = re.search(pattern, text, flags=re.MULTILINE)
        return m.group(1).strip() if m else None

    @staticmethod
    def _parse_timedelta_to_seconds(s: str) -> float | None:
        """
        Accepts strings like:
        '0:01:45.747067'
        '1 day, 0:01:45.747067' (just in case)
        """
        try:
            td = pd.to_timedelta(s)
            return float(td.total_seconds())
        except Exception:
            return None
    
    def _get_policy_colors(self, policies: list[str]) -> list[str]:
        colors = []
        for p in policies:
            if p not in self.policy_order:
                print(f"[WARN] Policy '{p}' not in predefined policy order. Assigning gray color.")
                colors.append("lightgray")
                continue
            if p in self.baseline_colors:
                colors.append(self.baseline_colors[p])
            elif p in self.our_methods_colors:
                colors.append(self.our_methods_colors[p])
            else:
                print(f"[WARN] Policy '{p}' not categorized as baseline or our method. Assigning gray color.")
                colors.append("lightgray")

        return colors
        
    
    def filter_to_weeks(self, df: pd.DataFrame, weeks: set[tuple[int, int]]) -> pd.DataFrame:
        week_set = set(weeks)
        mask = list(zip(df["iso_year"], df["iso_week"]))
        return df[pd.Series(mask, index=df.index).isin(week_set)].copy()

    def _ordered_policies(self, policies: list[str]) -> list[str]:
        rank = {p: i for i, p in enumerate(self.policy_order)}
        return sorted(policies, key=lambda p: (rank.get(p, 10**9), p))
    
    def _parse_all_logs(self) -> pd.DataFrame:
        if not self.logs_root_dir.exists():
            raise FileNotFoundError(f"logs_root_dir not found: {self.logs_root_dir}")

        rows = []
        for policy_dir in sorted([p for p in self.logs_root_dir.iterdir() if p.is_dir()]):
            policy = policy_dir.name

            if policy not in self.policy_order:
                print(f"[WARN] Found logs for policy '{policy}' which is not in the predefined policy order list.")
                continue

            log_files = sorted(policy_dir.glob(self.file_glob))
            if not log_files:
                continue

            for f in log_files:
                try:
                    text = f.read_text(errors="ignore")
                except Exception as e:
                    print(f"[WARN] Failed reading {f}: {e}")
                    continue

                day, floor = self._parse_day_floor_from_filename(f)

                planning_str = self._extract_value(r"Total Planning Time:\s*([^\n\r]+)", text)
                req_str = self._extract_value(r"Total Number of Requests:\s*(\d+)", text)

                if req_str is None:
                    req_str = self._extract_value(r"Number of Requests:\s*(\d+)", text)

                planning_sec = self._parse_timedelta_to_seconds(planning_str) if planning_str else None
                total_requests = int(req_str) if req_str is not None else None

                if planning_sec is None or total_requests is None:
                    print(f"[WARN] Missing planning/requests in {f.name} (policy={policy})")
                    continue

                if total_requests <= 0:
                    continue

                rows.append(
                    {
                        "policy": policy,
                        "file": f.name,
                        "day": day,
                        "floor": floor,
                        "planning_time_sec": planning_sec,
                        "total_requests": total_requests,
                        "planning_time_per_request_sec": planning_sec / total_requests,
                    }
                )

        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError(
                f"No parsable log entries found under {self.logs_root_dir} "
                f"with glob '{self.file_glob}'."
            )

        df = df.dropna(subset=["day", "floor"]).copy()
        if df.empty:
            raise ValueError(
                "Parsed planning/request values, but couldn't parse (day,floor) from filenames. "
                "Adjust the filename regex in _parse_day_floor_from_filename()."
            )

        df["policy"] = df["policy"].astype(str)
        df = df.sort_values(["policy", "day", "floor"]).reset_index(drop=True)
        return df
    
    def _load_raw_requests_df(self) -> pd.DataFrame:
        rows = []

        if not self.root_dir.exists():
            raise FileNotFoundError(f"root_dir not found: {self.root_dir}")

        # Expect: root_dir/<policy_folder>/*.csv
        for policy_dir in sorted([p for p in self.root_dir.iterdir() if p.is_dir()]):
            policy = policy_dir.name
            csv_files = sorted(policy_dir.glob("*.csv"))
            if not csv_files:
                continue

            for f in csv_files:
                try:
                    df_one = pd.read_csv(f)
                except Exception as e:
                    print(f"[WARN] Failed reading {f}: {e}")
                    continue

                # parse day/floor from filename
                try:
                    day, floor = self._parse_day_floor_from_filename(f)
                except Exception as e:
                    print(f"[WARN] {e}")
                    continue

                required = {"completed", "rejected", self.type_col}
                missing = required - set(df_one.columns)
                if missing:
                    raise ValueError(f"{f} missing required columns: {sorted(missing)}")

                # normalize & keep only what we need for counting
                tmp = pd.DataFrame({
                    "policy": policy,
                    "request_type": df_one[self.type_col].astype(str).fillna("UNKNOWN"),
                    "completed": df_one["completed"].fillna(False).astype(bool),
                    "rejected": df_one["rejected"].fillna(False).astype(bool),
                    "_day": day,
                    "_floor": floor,
                })
                rows.append(tmp)

        if not rows:
            raise ValueError(f"No readable CSVs found under {self.root_dir}")

        df = pd.concat(rows, ignore_index=True)
        iso = df["_day"].dt.isocalendar()
        df["iso_year"] = iso["year"].astype(int)
        df["iso_week"] = iso["week"].astype(int)
        return df
    
    def print_policy_mean_pm_std(self) -> None:
        """
        Prints mean ± std of planning_time_per_request_sec across (day,floor) units,
        either overall or per week-bucket.
        """

        for label, weeks in self.week_buckets.items():
            sub = self.filter_to_weeks(self.logs_df, weeks) if label != "ALL_WEEKS" else self.logs_df
            if sub.empty:
                print(f"\n[WARN] No rows for bucket '{label}'.")
                continue

            g = (
                sub.groupby("policy")["planning_time_per_request_sec"]
                .agg(["count", "mean", "std"])
                .reset_index()
            )

            policies = self._ordered_policies(g["policy"].tolist())
            g = g.set_index("policy").reindex(policies).reset_index()

            print("\n" + "=" * 80)
            print(f"Planning time per request (day-floor): mean ± 1 std — {label}")
            if self.week_buckets:
                print(f"Weeks: {sorted(weeks)}")
            print("=" * 80)
            print(f"{'Policy':30s}  {'N(day-floor)':>12s}  {'Mean (s/req)':>14s}  {'Std (s/req)':>14s}  {'Mean ± Std':>24s}")
            print("-" * 80)

            for _, r in g.iterrows():
                n = int(r["count"])
                mean = float(r["mean"]) if np.isfinite(r["mean"]) else float("nan")
                std = float(r["std"]) if np.isfinite(r["std"]) else 0.0  # NaN if n==1
                print(
                    f"{str(r['policy'])[:30]:30s}  {n:12d}  {mean:14.4f}  {std:14.4f}  {mean:10.4f} ± {std:10.4f}"
                )
    
    def plot_box_per_policy_by_week_bucket(self) -> None:
        for label, weeks in self.week_buckets.items():
            out_subdir = self.out_dir / f"{label}"
            out_subdir.mkdir(parents=True, exist_ok=True)

            sub = self.filter_to_weeks(self.logs_df, weeks)
            if sub.empty:
                print(f"[WARN] No rows for bucket '{label}'. Skipping.")
                continue

            # same plotting logic, but using sub instead of self.df
            policies = self._ordered_policies(sub["policy"].unique().tolist())
            data, labels = [], []
            for p in policies:
                vals = sub.loc[sub["policy"] == p, "planning_time_per_request_sec"].to_numpy()
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                labels.append(p)
                data.append(vals)

            if not data:
                print(f"[WARN] Nothing to plot for bucket '{label}'.")
                continue

            plt.figure(figsize=(max(10, len(labels) * 0.9), 5))
            colors = self._get_policy_colors(policies=labels)
            bp = plt.boxplot(data, tick_labels=labels, showfliers=True, patch_artist=True)
            for box, c in zip(bp["boxes"], colors):
                box.set_facecolor(c)
                box.set_alpha(0.65)
                box.set_edgecolor("black")
            for med in bp["medians"]:
                med.set_color("black")
            plt.xticks(rotation=35, ha="right")
            plt.ylabel("Planning time per request (seconds) [per day-floor]")
            plt.title(f"Planning time per request by policy — {label} demand weeks")
            plt.tight_layout()
            out_png = out_subdir / f"planning_time_per_request.png"
            plt.savefig(out_png, dpi=200)
            plt.close()
            print(f"[INFO] Wrote plot: {out_png} in {out_subdir}")

    def _daily_agg(self, values: pd.Series) -> float:
        x = pd.to_numeric(values, errors="coerce").to_numpy()
        x = x[np.isfinite(x)]
        if x.size == 0:
            return np.nan
        if self.daily_stat == "mean":
            return float(np.mean(x))
        else:
            return float(np.percentile(x, 95))

    def _generate_results_summary_from_root_dir(self):
        rows = []
        dailyfloor_wait_stats_by_policy_type: dict[tuple[str, str], np.ndarray] = {}

        dailyfloor_stats_frames = []
        for policy_dir in sorted([p for p in self.root_dir.iterdir() if p.is_dir()]):
            policy_name = policy_dir.name
            if policy_name not in self.policy_order:
                print(f"[WARN] Found logs for policy '{policy_name}' which is not in the predefined policy order list.")
                continue

            csv_files = sorted(policy_dir.glob("*.csv"))
            if not csv_files:
                continue

            frames = []
            for f in csv_files:
                try:
                    df_one = pd.read_csv(f)
                    day, floor = self._parse_day_floor_from_filename(f)
                    df_one["_day"] = day
                    df_one["_floor"] = floor
                    df_one["_source_file"] = f.name
                    frames.append(df_one)
                except Exception as e:
                    print(f"[WARN] Failed reading {f}: {e}")

            if not frames:
                continue

            df = pd.concat(frames, ignore_index=True)

            required = ["completed", "rejected", "planned_time", "scheduled_time", self.type_col, "_day", "_floor"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f"Policy '{policy_name}' is missing columns: {missing}")

            df[self.type_col] = df[self.type_col].astype(str).fillna("UNKNOWN")

            # --------- Counts per (policy, request_type) ----------
            for req_type, g in df.groupby(self.type_col, dropna=False):
                total = int(len(g))
                serviced = int(g["completed"].fillna(False).astype(bool).sum())
                rejected = int(g["rejected"].fillna(False).astype(bool).sum())

                rows.append(
                    {
                        "policy": policy_name,
                        "request_type": req_type,
                        "rejected": rejected,
                        "serviced": serviced,
                        "total": total,
                    }
                )

            # --------- Daily-per-floor wait stat (serviced only) ----------
            serviced_df = df[df["completed"].fillna(False).astype(bool)].copy()
            if serviced_df.empty:
                continue

            planned = pd.to_numeric(serviced_df["planned_time"], errors="coerce")
            scheduled = pd.to_numeric(serviced_df["scheduled_time"], errors="coerce")
            serviced_df["_wait"] = (planned - scheduled).clip(lower=0)

            dailyfloor = (
                serviced_df.dropna(subset=["_day", "_floor"])
                .groupby([self.type_col, "_day", "_floor"], as_index=False)["_wait"]
                .apply(self._daily_agg)
                .rename(columns={"_wait": "dailyfloor_wait_stat"})
            )
            dailyfloor["policy"] = policy_name
            dailyfloor["daily_stat"] = self.daily_stat

            dailyfloor_stats_frames.append(dailyfloor)

            for req_type, g2 in dailyfloor.groupby(self.type_col):
                vals = pd.to_numeric(g2["dailyfloor_wait_stat"], errors="coerce").to_numpy()
                vals = vals[np.isfinite(vals)]
                dailyfloor_wait_stats_by_policy_type[(policy_name, str(req_type))] = vals

            dailyfloor_all = (
                serviced_df.dropna(subset=["_day", "_floor"])
                .groupby(["_day", "_floor"], as_index=False)["_wait"]
                .apply(self._daily_agg)
                .rename(columns={"_wait": "dailyfloor_wait_stat"})
            )
            dailyfloor_all[self.type_col] = "ALL"
            dailyfloor_all["policy"] = policy_name
            dailyfloor_all["daily_stat"] = self.daily_stat

            dailyfloor_stats_frames.append(dailyfloor_all)

            vals_all = pd.to_numeric(dailyfloor_all["dailyfloor_wait_stat"], errors="coerce").to_numpy()
            vals_all = vals_all[np.isfinite(vals_all)]
            dailyfloor_wait_stats_by_policy_type[(policy_name, "ALL")] = vals_all

        summary_by_type = pd.DataFrame(rows)
        if summary_by_type.empty:
            raise ValueError(f"No policy folders with readable CSVs found under: {self.root_dir}")

        summary_by_type = summary_by_type.sort_values(["policy", "request_type"]).reset_index(drop=True)

        dailyfloor_stats_df = None
        if dailyfloor_stats_frames:
            dailyfloor_stats_df = pd.concat(dailyfloor_stats_frames, ignore_index=True)
            dailyfloor_stats_df = dailyfloor_stats_df.sort_values(["policy", self.type_col, "_day", "_floor"]).reset_index(
                drop=True
            )

        return summary_by_type, dailyfloor_wait_stats_by_policy_type, dailyfloor_stats_df

    def print_counts_per_policy_per_request_type(self) -> None:
        """
        Prints per-policy rejected/serviced/total counts by request_type
        for each demand bucket (highest/medium/lowest).
        """
        df0 = self.raw_requests_df

        for label, weeks in self.week_buckets.items():
            df = self.filter_to_weeks(df0, weeks)
            if df.empty:
                print(f"\n[WARN] No requests for bucket '{label}'.")
                continue

            print("\n" + "#" * 90)
            print(f"Counts per policy — {label.upper()} demand weeks")
            print(f"Weeks: {sorted(weeks)}")
            print("#" * 90)

            pol_tbl = (
                df.groupby("policy", as_index=False)
                .agg(
                    entered=("request_type", "size"),
                    serviced=("completed", "sum"),
                    rejected=("rejected", "sum"),
                )
            )

            policies = self._ordered_policies(pol_tbl["policy"].tolist())
            pol_tbl = pol_tbl.set_index("policy").reindex(policies, fill_value=0).reset_index()

            print("\n" + "-" * 90)
            print(f"TOTALS PER POLICY — {label.upper()} demand weeks")
            print("-" * 90)
            print(f"{'Policy':30s}  {'Entered':>10s}  {'Serviced':>10s}  {'Rejected':>10s}")
            print("-" * 90)
            for _, r in pol_tbl.iterrows():
                print(
                    f"{str(r['policy'])[:30]:30s}  "
                    f"{int(r['entered']):10d}  "
                    f"{int(r['serviced']):10d}  "
                    f"{int(r['rejected']):10d}"
                )
            print("-" * 90)
            print(
                f"{'ALL POLICIES':30s}  "
                f"{int(pol_tbl['entered'].sum()):10d}  "
                f"{int(pol_tbl['serviced'].sum()):10d}  "
                f"{int(pol_tbl['rejected'].sum()):10d}"
            )

            # Build bucket-specific summary like your original summary_by_type
            summary = (
                df.groupby(["policy", "request_type"], as_index=False)
                .agg(
                    total=("request_type", "size"),
                    serviced=("completed", "sum"),
                    rejected=("rejected", "sum"),
                )
            )

            all_policies = self._ordered_policies(summary["policy"].unique().tolist())
            type_order = (
                summary.groupby("request_type")["total"].sum()
                .sort_values(ascending=False)
                .index.tolist()
            )

            for req_type in type_order:
                g = summary[summary["request_type"] == req_type]
                tbl = (
                    g.groupby("policy", as_index=False)[["rejected", "serviced", "total"]]
                    .sum()
                    .set_index("policy")
                    .reindex(all_policies, fill_value=0)
                )

                print("\n" + "=" * 72)
                print(f"Request type: {req_type}")
                print("=" * 72)
                print(f"{'Policy':30s}  {'Rejected':>10s}  {'Serviced':>10s}  {'Total':>10s}")
                print("-" * 72)

                for policy, row in tbl.iterrows():
                    r = int(row["rejected"])
                    s = int(row["serviced"])
                    t = int(row["total"])
                    print(f"{policy:30s}  {r:10d}  {s:10d}  {t:10d}")

                R = int(tbl["rejected"].sum())
                S = int(tbl["serviced"].sum())
                T = int(tbl["total"].sum())
                print("-" * 72)
                print(f"{'ALL POLICIES':30s}  {R:10d}  {S:10d}  {T:10d}")

    def plot_wait_time_by_week_bucket(self, week_buckets: dict[str, set[tuple[int,int]]]) -> None:
        df = self.dailyfloor_stats_df.copy()
        iso = df["_day"].dt.isocalendar()
        df["iso_year"] = iso["year"].astype(int)
        df["iso_week"] = iso["week"].astype(int)

        for label, weeks in week_buckets.items():
            out_subdir = self.out_dir / f"{label}"
            out_subdir.mkdir(parents=True, exist_ok=True)

            sub = self.filter_to_weeks(df.rename(columns={"_day":"day"}), weeks)  # reuse helper
            if sub.empty:
                print(f"[WARN] No rows for bucket '{label}'. Skipping.")
                continue

            # for each request_type: boxplot policies over dailyfloor_wait_stat
            for req_type, g in sub.groupby(self.type_col):
                # enforce policy order
                policies = self._ordered_policies(g["policy"].unique().tolist())

                data, labels = [], []
                for p in policies:
                    vals = g.loc[g["policy"] == p, "dailyfloor_wait_stat"].to_numpy()
                    vals = vals[np.isfinite(vals)]
                    if vals.size == 0:
                        continue
                    labels.append(p)
                    data.append(vals)

                if not data:
                    continue

                plt.figure(figsize=(max(10, len(labels) * 0.9), 5))
                colors = self._get_policy_colors(policies=labels)
                bp = plt.boxplot(data, tick_labels=labels, showfliers=True, patch_artist=True)
                for box, c in zip(bp["boxes"], colors):
                    box.set_facecolor(c)
                    box.set_alpha(0.65)
                    box.set_edgecolor("black")
                for med in bp["medians"]:
                    med.set_color("black")
                plt.xticks(rotation=35, ha="right")
                plt.xticks(rotation=35, ha="right")
                plt.ylabel(f"Daily-per-floor {self.daily_stat} wait time (seconds), serviced only")
                plt.title(f"Wait time by policy — request_type={req_type} — {label} demand weeks")
                plt.tight_layout()

                safe = self._safe_filename(req_type)
                plt.savefig(out_subdir / f"wait_times_{self.daily_stat}_{safe}.png", dpi=200)
                plt.close()
                print(f"[INFO] Wrote plot: wait_times_{self.daily_stat}_{safe}.png in {out_subdir}")


def main():
    parser = argparse.ArgumentParser(description="Plot assignment results from per-policy CSVs and logs.")
    parser.add_argument("--root_dir", type=str, default="results/policies", help="Root directory containing per-policy folders with CSVs.")
    parser.add_argument("--out_dir", type=str, default="results/policies/plots", help="Output directory for plots.")
    parser.add_argument("--logs_root_dir", type=str, default="results/policies/logs", help="Root directory containing per-policy log folders.")
    parser.add_argument("--file_glob", type=str, default="*.out", help="Glob pattern to find log files within each policy's log folder.")
    parser.add_argument("--type_col", type=str, default="request_type", help="Column name in CSVs that indicates request type.")
    parser.add_argument("--daily_stat", type=str, default="p95", choices=["mean", "p95"], help="Whether to compute daily mean or p95 wait times for boxplots.")
    args = parser.parse_args()

    plotter = AssignmentResultsPlotter(
        root_dir=args.root_dir,
        out_dir=args.out_dir,
        type_col=args.type_col,
        daily_stat=args.daily_stat,
        logs_root_dir=args.logs_root_dir,
        file_glob=args.file_glob
    )

    plotter.print_counts_per_policy_per_request_type()
    plotter.plot_wait_time_by_week_bucket(plotter.week_buckets)
    plotter.plot_box_per_policy_by_week_bucket()
    plotter.print_policy_mean_pm_std()

if __name__ == "__main__":
    pStart = datetime.now()
    try:
        main()
    except Exception as errorMainContext:
        print("Fail End Process: ", errorMainContext)
        traceback.print_exc()
    pEnd = datetime.now()
    print(f"Total Execution Time: {pEnd - pStart}")