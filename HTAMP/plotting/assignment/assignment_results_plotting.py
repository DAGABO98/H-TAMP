import argparse
import re
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

IsoWeek = tuple[int, int]
WeekBucketMap = dict[str, set[IsoWeek]]
FloorWeekBucketMap = dict[str, dict[int, set[IsoWeek]]]

DEFAULT_WEEK_BUCKETS: WeekBucketMap = {
    "highest": {(2024, 40), (2025, 5)},
    "medium": {(2024, 44), (2025, 14)},
    "lowest": {(2024, 27), (2024, 36)},
}

DEFAULT_WEEK_BUCKETS_BY_FLOOR: FloorWeekBucketMap = {
    "highest": {
        2: {(2025, 6), (2025, 8)},
        3: {(2025, 5), (2025, 10)},
        7: {(2024, 39), (2024, 40)},
        9: {(2024, 43), (2025, 6)},
    },
    "medium": {
        2: {(2024, 44), (2025, 14)},
        3: {(2024, 44), (2025, 14)},
        7: {(2024, 44), (2025, 14)},
        9: {(2024, 44), (2025, 14)},
    },
    "lowest": {
        2: {(2024, 27), (2024, 28)},
        3: {(2024, 46), (2025, 2)},
        7: {(2024, 46), (2025, 26)},
        9: {(2025, 19), (2025, 21)},
    }
}

BASELINE_POLICIES = [
    "fleet_manager",
    "tp_d_alpha0.0",
    "d_tpts_alpha0.0",
    "tp_d_alpha0.2",
    "d_tpts_alpha0.2",
    "idle_pred",
    "idle_prediction",
]

BASE_POLICY = "base_policy"

ROLLOUT_VARIANT_SUFFIXES = [
    "nopt_norwt",
    "ropt_norwt",
    "nopt_rwt",
    "ropt_rwt",
]

ROLLOUT_POLICIES = [
    f"adaptive_rollout_{suffix}"
    for suffix in ROLLOUT_VARIANT_SUFFIXES
]

FUTURE_SCHEDULED_ROLLOUT_POLICIES = [
    f"adaptive_rollout_future_scheduled_{suffix}"
    for suffix in ROLLOUT_VARIANT_SUFFIXES
]

UNCAPPED_ROLLOUT_POLICIES = [
    f"adaptive_rollout_uncapped_{suffix}"
    for suffix in ROLLOUT_VARIANT_SUFFIXES
]

FUTURE_SCHEDULED_UNCAPPED_ROLLOUT_POLICIES = [
    f"adaptive_rollout_future_scheduled_uncapped_{suffix}"
    for suffix in ROLLOUT_VARIANT_SUFFIXES
]


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
        file_glob: str = "*.out",
        week_buckets: WeekBucketMap | None = None,
        week_buckets_by_floor: FloorWeekBucketMap | None = None,
        include_future_scheduled_requests: bool = False,
    ):
        self.include_future_scheduled_requests = include_future_scheduled_requests
        self.rollout_policies = list(ROLLOUT_POLICIES)
        self.future_scheduled_rollout_policies = list(FUTURE_SCHEDULED_ROLLOUT_POLICIES)
        self.uncapped_rollout_policies = list(UNCAPPED_ROLLOUT_POLICIES)
        self.future_scheduled_uncapped_rollout_policies = list(FUTURE_SCHEDULED_UNCAPPED_ROLLOUT_POLICIES)
        self.selected_rollout_policies = (
            self.future_scheduled_rollout_policies
            if self.include_future_scheduled_requests
            else self.rollout_policies
        )
        self.selected_uncapped_rollout_policies = (
            self.future_scheduled_uncapped_rollout_policies
            if self.include_future_scheduled_requests
            else self.uncapped_rollout_policies
        )
        selected_ropt_rwt = (
            "adaptive_rollout_future_scheduled_ropt_rwt"
            if self.include_future_scheduled_requests
            else "adaptive_rollout_ropt_rwt"
        )
        selected_uncapped_ropt_rwt = (
            "adaptive_rollout_future_scheduled_uncapped_ropt_rwt"
            if self.include_future_scheduled_requests
            else "adaptive_rollout_uncapped_ropt_rwt"
        )
        self.main_comparison_policies = BASELINE_POLICIES + [BASE_POLICY, selected_ropt_rwt, selected_uncapped_ropt_rwt]
        self.all_rollout_ablation_policies = (
            [BASE_POLICY]
            + self.rollout_policies
            + self.future_scheduled_rollout_policies
            + self.uncapped_rollout_policies
            + self.future_scheduled_uncapped_rollout_policies
        )
        self.selected_rollout_ablation_policies = list(self.selected_rollout_policies) + list(self.selected_uncapped_rollout_policies)

        self.policy_order = BASELINE_POLICIES + [
                             BASE_POLICY,
                             "heuristic_rollout_ropt_rwt",
                             *self.rollout_policies,
                             *self.future_scheduled_rollout_policies,
                             *self.uncapped_rollout_policies,
                             *self.future_scheduled_uncapped_rollout_policies,]

        self.baseline_colors = {"human_team": "#7f0000",
                                "fleet_manager": "#b30000",
                                "tp_d_alpha0.0": "#d7301f",
                                "d_tpts_alpha0.0": "#ef6548",
                                "tp_d_alpha0.2": "#fc8d59",
                                "d_tpts_alpha0.2": "#fdbb84",
                                "idle_pred": "#fdd49e",
                                "idle_prediction": "#fdd49e",
                                "vanilla_rollout_prempt": "#fee8c8"}
        
        self.our_methods_colors = {"base_policy": "#d9f0a3",
                                    "heuristic_rollout_ropt_rwt": "#addd8e",
                                    "adaptive_rollout_nopt_norwt": "#78c679",
                                    "adaptive_rollout_ropt_norwt": "#31a354",
                                    "adaptive_rollout_nopt_rwt": "#238b45",
                                    "adaptive_rollout_ropt_rwt": "#006837",
                                    "adaptive_rollout_future_scheduled_nopt_norwt": "#bcbddc",
                                    "adaptive_rollout_future_scheduled_ropt_norwt": "#9e9ac8",
                                    "adaptive_rollout_future_scheduled_nopt_rwt": "#756bb1",
                                    "adaptive_rollout_future_scheduled_ropt_rwt": "#54278f",
                                    "adaptive_rollout_uncapped_nopt_norwt": "#9ecae1",
                                    "adaptive_rollout_uncapped_ropt_norwt": "#6baed6",
                                    "adaptive_rollout_uncapped_nopt_rwt": "#3182bd",
                                    "adaptive_rollout_uncapped_ropt_rwt": "#08519c",
                                    "adaptive_rollout_future_scheduled_uncapped_nopt_norwt": "#fdd0a2",
                                    "adaptive_rollout_future_scheduled_uncapped_ropt_norwt": "#fdae6b",
                                    "adaptive_rollout_future_scheduled_uncapped_nopt_rwt": "#e6550d",
                                    "adaptive_rollout_future_scheduled_uncapped_ropt_rwt": "#a63603"}
        
        self.ablation_study_colors = {"base_policy": "#d9f0a3",
                                     "adaptive_rollout_nopt_norwt": "#addd8e",
                                     "adaptive_rollout_ropt_norwt": "#78c679",
                                     "adaptive_rollout_nopt_rwt": "#31a354",
                                     "adaptive_rollout_ropt_rwt": "#006837",
                                     "adaptive_rollout_future_scheduled_nopt_norwt": "#bcbddc",
                                     "adaptive_rollout_future_scheduled_ropt_norwt": "#9e9ac8",
                                     "adaptive_rollout_future_scheduled_nopt_rwt": "#756bb1",
                                     "adaptive_rollout_future_scheduled_ropt_rwt": "#54278f",
                                     "adaptive_rollout_uncapped_nopt_norwt": "#9ecae1",
                                     "adaptive_rollout_uncapped_ropt_norwt": "#6baed6",
                                     "adaptive_rollout_uncapped_nopt_rwt": "#3182bd",
                                     "adaptive_rollout_uncapped_ropt_rwt": "#08519c",
                                     "adaptive_rollout_future_scheduled_uncapped_nopt_norwt": "#fdd0a2",
                                     "adaptive_rollout_future_scheduled_uncapped_ropt_norwt": "#fdae6b",
                                     "adaptive_rollout_future_scheduled_uncapped_nopt_rwt": "#e6550d",
                                     "adaptive_rollout_future_scheduled_uncapped_ropt_rwt": "#a63603"}
        
        if week_buckets is None:
            week_buckets = DEFAULT_WEEK_BUCKETS
        if week_buckets_by_floor is None:
            week_buckets_by_floor = DEFAULT_WEEK_BUCKETS_BY_FLOOR

        self.week_buckets: WeekBucketMap = {
            str(label): set(weeks)
            for label, weeks in week_buckets.items()
        }
        self.week_buckets_by_floor: FloorWeekBucketMap = {
            str(label): {int(floor): set(weeks) for floor, weeks in floor_map.items()}
            for label, floor_map in week_buckets_by_floor.items()
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
            elif p in self.ablation_study_colors:
                colors.append(self.ablation_study_colors[p])
            else:
                print(f"[WARN] Policy '{p}' not categorized as baseline or our method. Assigning gray color.")
                colors.append("lightgray")

        return colors
        
    
    def _iter_week_bucket_specs(self) -> list[tuple[str, set[IsoWeek], dict[int, set[IsoWeek]]]]:
        labels = list(dict.fromkeys(list(self.week_buckets) + list(self.week_buckets_by_floor)))
        return [
            (
                label,
                self.week_buckets.get(label, set()),
                self.week_buckets_by_floor.get(label, {}),
            )
            for label in labels
        ]

    @staticmethod
    def _format_weeks_by_floor(weeks_by_floor: dict[int, set[IsoWeek]]) -> dict[int, list[IsoWeek]]:
        return {
            int(floor): sorted(weeks)
            for floor, weeks in sorted(weeks_by_floor.items())
        }

    def filter_to_weeks(self,
                        df: pd.DataFrame,
                        weeks: set[IsoWeek],
                        weeks_by_floor: dict[int, set[IsoWeek]] | None = None,
                        floor_col: str = "floor") -> pd.DataFrame:
        week_set = set(weeks)
        if weeks_by_floor is None:
            weeks_by_floor = {}
        normalized_by_floor = {int(floor): set(values) for floor, values in weeks_by_floor.items()}

        if not week_set and not normalized_by_floor:
            return df.copy()

        if floor_col not in df.columns:
            mask = list(zip(df["iso_year"], df["iso_week"]))
            return df[pd.Series(mask, index=df.index).isin(week_set)].copy()

        def is_selected(row: pd.Series) -> bool:
            week_key = (int(row["iso_year"]), int(row["iso_week"]))
            floor_value = row.get(floor_col)
            if pd.notna(floor_value):
                selected_weeks = normalized_by_floor.get(int(floor_value), week_set)
            else:
                selected_weeks = week_set
            return week_key in selected_weeks

        return df[df.apply(is_selected, axis=1)].copy()

    def _ordered_policies(self, policies: list[str]) -> list[str]:
        rank = {p: i for i, p in enumerate(self.policy_order)}
        return sorted(policies, key=lambda p: (rank.get(p, 10**9), p))

    def _filter_to_policy_subset(self,
                                 df: pd.DataFrame,
                                 policy_subset: list[str] | None,
                                 policy_col: str = "policy") -> pd.DataFrame:
        if policy_subset is None:
            return df.copy()
        return df[df[policy_col].isin(policy_subset)].copy()

    @staticmethod
    def _warn_missing_policies(policy_subset: list[str] | None,
                               available_policies: list[str],
                               context: str) -> None:
        if policy_subset is None:
            return
        available_policy_set = set(available_policies)
        missing = [policy for policy in policy_subset if policy not in available_policy_set]
        if missing:
            print(f"[WARN] Missing policies for {context}: {missing}")
    
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

                if {"planned_time", "scheduled_time"}.issubset(df_one.columns):
                    planned = pd.to_numeric(df_one["planned_time"], errors="coerce")
                    scheduled = pd.to_numeric(df_one["scheduled_time"], errors="coerce")
                    wait_time = (planned - scheduled).clip(lower=0)
                else:
                    planned = pd.Series(np.nan, index=df_one.index)
                    scheduled = pd.Series(np.nan, index=df_one.index)
                    wait_time = pd.Series(np.nan, index=df_one.index)

                # normalize & keep request-level data for counts and tables
                tmp = pd.DataFrame({
                    "policy": policy,
                    "request_type": df_one[self.type_col].astype(str).fillna("UNKNOWN"),
                    "completed": df_one["completed"].fillna(False).astype(bool),
                    "rejected": df_one["rejected"].fillna(False).astype(bool),
                    "planned_time": planned,
                    "scheduled_time": scheduled,
                    "wait_time": wait_time,
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

        for label, weeks, weeks_by_floor in self._iter_week_bucket_specs():
            sub = (
                self.filter_to_weeks(self.logs_df, weeks, weeks_by_floor, floor_col="floor")
                if label != "ALL_WEEKS"
                else self.logs_df
            )
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
            if weeks_by_floor:
                print(f"Global weeks: {sorted(weeks)}")
                print(f"Per-floor weeks: {self._format_weeks_by_floor(weeks_by_floor)}")
            elif self.week_buckets:
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
    
    @staticmethod
    def _format_mean_pm_std(mean: float, std: float, plus_minus: str = " +/- ") -> str:
        if not np.isfinite(mean):
            return ""
        if not np.isfinite(std):
            std = 0.0
        return f"{mean:.2f}{plus_minus}{std:.2f}"

    def _wait_time_stats_per_serviced_request_by_load(
        self,
        policy_subset: list[str] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        load_display_names = {
            "highest": "High",
            "medium": "Medium",
            "lowest": "Low",
        }
        mean_tables: dict[str, pd.Series] = {}
        std_tables: dict[str, pd.Series] = {}
        count_tables: dict[str, pd.Series] = {}
        policies_seen: set[str] = set()

        for label, weeks, weeks_by_floor in self._iter_week_bucket_specs():
            display_label = load_display_names.get(label, str(label).title())
            sub = self.filter_to_weeks(self.raw_requests_df, weeks, weeks_by_floor, floor_col="_floor")
            self._warn_missing_policies(policy_subset, sub["policy"].unique().tolist(), f"average wait table ({label})")
            sub = self._filter_to_policy_subset(sub, policy_subset)
            serviced = sub[sub["completed"].fillna(False).astype(bool)].copy()
            serviced["wait_time"] = pd.to_numeric(serviced["wait_time"], errors="coerce")
            serviced = serviced.dropna(subset=["wait_time"])

            if serviced.empty:
                print(f"[WARN] No serviced requests with wait times for bucket '{label}'.")
                mean_tables[display_label] = pd.Series(dtype=float)
                std_tables[display_label] = pd.Series(dtype=float)
                count_tables[display_label] = pd.Series(dtype=int)
                continue

            stats = serviced.groupby("policy")["wait_time"].agg(["mean", "std", "count"])
            mean_tables[display_label] = stats["mean"]
            std_tables[display_label] = stats["std"].fillna(0.0)
            count_tables[display_label] = stats["count"].astype(int)
            policies_seen.update(stats.index.astype(str).tolist())

        if policy_subset is not None:
            policies = self._ordered_policies(list(policy_subset))
        else:
            policies = self._ordered_policies(list(policies_seen))

        mean_table = pd.DataFrame(mean_tables).reindex(policies)
        std_table = pd.DataFrame(std_tables).reindex(policies)
        count_table = pd.DataFrame(count_tables).reindex(policies)
        mean_table.index.name = "policy"
        std_table.index.name = "policy"
        count_table.index.name = "policy"
        return mean_table, std_table, count_table

    def write_average_wait_time_ablation_table_by_load(
        self,
        policy_subset: list[str] | None,
        output_subdir_name: str,
        filename_prefix: str,
        title: str,
    ) -> None:
        """
        Writes request-level average wait time tables for each load bucket.

        Unlike the boxplots, this averages directly over all serviced requests in
        the bucket and does not pre-aggregate by day or floor.
        """
        out_subdir = self.out_dir / output_subdir_name
        out_subdir.mkdir(parents=True, exist_ok=True)

        mean_table, std_table, count_table = self._wait_time_stats_per_serviced_request_by_load(
            policy_subset=policy_subset,
        )
        if mean_table.empty:
            print(f"[WARN] No rows for {title}. Skipping table.")
            return

        mean_table = mean_table.round(2)
        std_table = std_table.round(2)
        csv_table = mean_table.copy().astype(object)
        tex_table = mean_table.copy().astype(object)
        for row in csv_table.index:
            for col in csv_table.columns:
                csv_table.loc[row, col] = self._format_mean_pm_std(
                    mean_table.loc[row, col],
                    std_table.loc[row, col],
                )
                tex_table.loc[row, col] = self._format_mean_pm_std(
                    mean_table.loc[row, col],
                    std_table.loc[row, col],
                    plus_minus=r" $\pm$ ",
                )

        csv_path = out_subdir / f"{filename_prefix}.csv"
        tex_path = out_subdir / f"{filename_prefix}.tex"
        mean_csv_path = out_subdir / f"{filename_prefix}_mean_seconds.csv"
        std_csv_path = out_subdir / f"{filename_prefix}_std_seconds.csv"
        counts_csv_path = out_subdir / f"{filename_prefix}_serviced_counts.csv"

        csv_table.to_csv(csv_path, na_rep="")
        mean_table.to_csv(mean_csv_path, na_rep="")
        std_table.to_csv(std_csv_path, na_rep="")
        count_table.astype("Int64").to_csv(counts_csv_path, na_rep="")
        tex_table.to_latex(
            tex_path,
            escape=False,
            na_rep="--",
            caption=title,
            label=f"tab:{self._safe_filename(filename_prefix)}",
        )

        print("\n" + "=" * 100)
        print(title)
        print("Average +/- standard deviation wait time per serviced request (seconds); computed over requests, not day-floor aggregates.")
        print("=" * 100)
        print(csv_table.to_string(na_rep="--"))
        print(f"[INFO] Wrote table: {csv_path}")
        print(f"[INFO] Wrote LaTeX table: {tex_path}")
        print(f"[INFO] Wrote numeric means: {mean_csv_path}")
        print(f"[INFO] Wrote numeric standard deviations: {std_csv_path}")
        print(f"[INFO] Wrote serviced-request counts: {counts_csv_path}")

    def plot_box_per_policy_by_week_bucket(self,
                                           policy_subset: list[str] | None = None,
                                           output_subdir_name: str | None = None,
                                           filename: str = "planning_time_per_request.png",
                                           title_prefix: str = "Planning time per request by policy") -> None:
        for label, weeks, weeks_by_floor in self._iter_week_bucket_specs():
            out_subdir = self.out_dir / f"{label}"
            if output_subdir_name is not None:
                out_subdir = out_subdir / output_subdir_name
            out_subdir.mkdir(parents=True, exist_ok=True)

            sub = self.filter_to_weeks(self.logs_df, weeks, weeks_by_floor, floor_col="floor")
            self._warn_missing_policies(policy_subset, sub["policy"].unique().tolist(), f"{title_prefix} ({label})")
            sub = self._filter_to_policy_subset(sub, policy_subset)
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
            plt.title(f"{title_prefix} - {label} demand weeks")
            plt.tight_layout()
            out_png = out_subdir / filename
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

            if policy_name == "sequential_greedy_nopt" and "human_team" in self.policy_order:
                administered = pd.to_numeric(serviced_df["administered_time"], errors="coerce")
                serviced_df["_administered"] = (administered - scheduled).clip(lower=0)

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
            
            if policy_name == "sequential_greedy_nopt" and "human_team" in self.policy_order:
                dailyfloor_administered = (
                    serviced_df.dropna(subset=["_day", "_floor"])
                    .groupby([self.type_col, "_day", "_floor"], as_index=False)["_administered"]
                    .apply(self._daily_agg)
                    .rename(columns={"_administered": "dailyfloor_wait_stat"})
                )
                dailyfloor_administered["policy"] = "human_team"
                dailyfloor_administered["daily_stat"] = self.daily_stat

                dailyfloor_stats_frames.append(dailyfloor_administered)

                for req_type, g2 in dailyfloor_administered.groupby(self.type_col):
                    vals = pd.to_numeric(g2["dailyfloor_wait_stat"], errors="coerce").to_numpy()
                    vals = vals[np.isfinite(vals)]
                    dailyfloor_wait_stats_by_policy_type[("human_team", str(req_type))] = vals

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

            if policy_name == "sequential_greedy_nopt" and "human_team" in self.policy_order:
                dailyfloor_administered_all = (
                    serviced_df.dropna(subset=["_day", "_floor"])
                    .groupby(["_day", "_floor"], as_index=False)["_administered"]
                    .apply(self._daily_agg)
                    .rename(columns={"_administered": "dailyfloor_wait_stat"})
                )
                dailyfloor_administered_all[self.type_col] = "ALL"
                dailyfloor_administered_all["policy"] = "human_team"
                dailyfloor_administered_all["daily_stat"] = self.daily_stat

                dailyfloor_stats_frames.append(dailyfloor_administered_all)

                vals_administered_all = pd.to_numeric(dailyfloor_administered_all["dailyfloor_wait_stat"], errors="coerce").to_numpy()
                vals_administered_all = vals_administered_all[np.isfinite(vals_administered_all)]
                dailyfloor_wait_stats_by_policy_type[("human_team", "ALL")] = vals_administered_all

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

        for label, weeks, weeks_by_floor in self._iter_week_bucket_specs():
            df = self.filter_to_weeks(df0, weeks, weeks_by_floor, floor_col="_floor")
            if df.empty:
                print(f"\n[WARN] No requests for bucket '{label}'.")
                continue

            print("\n" + "#" * 90)
            print(f"Counts per policy — {label.upper()} demand weeks")
            if weeks_by_floor:
                print(f"Global weeks: {sorted(weeks)}")
                print(f"Per-floor weeks: {self._format_weeks_by_floor(weeks_by_floor)}")
            else:
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

    def plot_wait_time_by_week_bucket(self,
                                      week_buckets: WeekBucketMap,
                                      week_buckets_by_floor: FloorWeekBucketMap | None = None,
                                      policy_subset: list[str] | None = None,
                                      output_subdir_name: str | None = None,
                                      filename_prefix: str = "wait_times",
                                      title_prefix: str = "Wait time by policy") -> None:
        df = self.dailyfloor_stats_df.copy()
        iso = df["_day"].dt.isocalendar()
        df["iso_year"] = iso["year"].astype(int)
        df["iso_week"] = iso["week"].astype(int)

        if week_buckets_by_floor is None:
            week_buckets_by_floor = {}

        labels = list(dict.fromkeys(list(week_buckets) + list(week_buckets_by_floor)))
        for label in labels:
            weeks = set(week_buckets.get(label, set()))
            weeks_by_floor = {
                int(floor): set(week_values)
                for floor, week_values in week_buckets_by_floor.get(label, {}).items()
            }
            out_subdir = self.out_dir / f"{label}"
            if output_subdir_name is not None:
                out_subdir = out_subdir / output_subdir_name
            out_subdir.mkdir(parents=True, exist_ok=True)

            sub = self.filter_to_weeks(
                df.rename(columns={"_day":"day"}),
                weeks,
                weeks_by_floor,
                floor_col="_floor",
            )
            self._warn_missing_policies(policy_subset, sub["policy"].unique().tolist(), f"{title_prefix} ({label})")
            sub = self._filter_to_policy_subset(sub, policy_subset)
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
                plt.ylabel(f"Daily-per-floor {self.daily_stat} wait time (seconds), serviced only")
                safe = self._safe_filename(req_type)
                plt.title(f"{title_prefix} - request_type={req_type} - {label} demand weeks")
                plt.tight_layout()
                plt.savefig(out_subdir / f"{filename_prefix}_{self.daily_stat}_{safe}.png", dpi=200)
                plt.close()
                print(f"[INFO] Wrote plot: {filename_prefix}_{self.daily_stat}_{safe}.png in {out_subdir}")

    def plot_main_comparison_by_week_bucket(self) -> None:
        suffix_label = "with future scheduled requests" if self.include_future_scheduled_requests else "without future scheduled requests"
        self.plot_wait_time_by_week_bucket(
            self.week_buckets,
            self.week_buckets_by_floor,
            policy_subset=self.main_comparison_policies,
            filename_prefix="wait_times",
            title_prefix=f"Main comparison ({suffix_label})",
        )
        self.plot_box_per_policy_by_week_bucket(
            policy_subset=self.main_comparison_policies,
            filename="planning_time_per_request.png",
            title_prefix=f"Main comparison planning time ({suffix_label})",
        )

    def plot_all_rollout_ablation_by_week_bucket(self) -> None:
        output_subdir_name = "ablation_all_rollout_variants"
        self.plot_wait_time_by_week_bucket(
            self.week_buckets,
            self.week_buckets_by_floor,
            policy_subset=self.all_rollout_ablation_policies,
            output_subdir_name=output_subdir_name,
            filename_prefix="all_rollout_variants_wait_times",
            title_prefix="Ablation: all rollout variants plus base policy",
        )
        self.plot_box_per_policy_by_week_bucket(
            policy_subset=self.all_rollout_ablation_policies,
            output_subdir_name=output_subdir_name,
            filename="all_rollout_variants_planning_time_per_request.png",
            title_prefix="Ablation: all rollout variant runtimes plus base policy",
        )
        self.write_average_wait_time_ablation_table_by_load(
            policy_subset=self.all_rollout_ablation_policies,
            output_subdir_name=output_subdir_name,
            filename_prefix="all_rollout_variants_average_wait_time_per_serviced_request_by_load",
            title="Ablation: average wait time per serviced request by load",
        )

    def plot_selected_rollout_ablation_by_week_bucket(self) -> None:
        suffix_label = "future scheduled" if self.include_future_scheduled_requests else "no future scheduled"
        output_subdir_name = (
            "ablation_future_scheduled_rollout_variants"
            if self.include_future_scheduled_requests
            else "ablation_rollout_variants"
        )
        filename_prefix = (
            "future_scheduled_rollout_variants_wait_times"
            if self.include_future_scheduled_requests
            else "rollout_variants_wait_times"
        )
        runtime_filename = (
            "future_scheduled_rollout_variants_planning_time_per_request.png"
            if self.include_future_scheduled_requests
            else "rollout_variants_planning_time_per_request.png"
        )
        table_filename_prefix = (
            "future_scheduled_rollout_variants_average_wait_time_per_serviced_request_by_load"
            if self.include_future_scheduled_requests
            else "rollout_variants_average_wait_time_per_serviced_request_by_load"
        )
        self.plot_wait_time_by_week_bucket(
            self.week_buckets,
            self.week_buckets_by_floor,
            policy_subset=self.selected_rollout_ablation_policies,
            output_subdir_name=output_subdir_name,
            filename_prefix=filename_prefix,
            title_prefix=f"Ablation: rollout variants ({suffix_label})",
        )
        self.plot_box_per_policy_by_week_bucket(
            policy_subset=self.selected_rollout_ablation_policies,
            output_subdir_name=output_subdir_name,
            filename=runtime_filename,
            title_prefix=f"Ablation: rollout variant runtimes ({suffix_label})",
        )
        self.write_average_wait_time_ablation_table_by_load(
            policy_subset=self.selected_rollout_ablation_policies,
            output_subdir_name=output_subdir_name,
            filename_prefix=table_filename_prefix,
            title=f"Ablation: rollout variants ({suffix_label}) average wait time per serviced request by load",
        )


def main():
    parser = argparse.ArgumentParser(description="Plot assignment results from per-policy CSVs and logs.")
    parser.add_argument("--root_dir", type=str, default="results/policies", help="Root directory containing per-policy folders with CSVs.")
    parser.add_argument("--out_dir", type=str, default="results/policies/plots", help="Output directory for plots.")
    parser.add_argument("--logs_root_dir", type=str, default="results/policies/logs", help="Root directory containing per-policy log folders.")
    parser.add_argument("--file_glob", type=str, default="*.out", help="Glob pattern to find log files within each policy's log folder.")
    parser.add_argument("--type_col", type=str, default="request_type", help="Column name in CSVs that indicates request type.")
    parser.add_argument("--daily_stat", type=str, default="p95", choices=["mean", "p95"], help="Whether to compute daily mean or p95 wait times for boxplots.")
    parser.add_argument("--include_future_scheduled_requests", action="store_true", help="Use future-scheduled adaptive rollout variants in the main comparison and selected capped/uncapped ablation plots.")
    args = parser.parse_args()

    plotter = AssignmentResultsPlotter(
        root_dir=args.root_dir,
        out_dir=args.out_dir,
        type_col=args.type_col,
        daily_stat=args.daily_stat,
        logs_root_dir=args.logs_root_dir,
        file_glob=args.file_glob,
        include_future_scheduled_requests=args.include_future_scheduled_requests,
    )

    plotter.print_counts_per_policy_per_request_type()
    plotter.plot_main_comparison_by_week_bucket()
    plotter.plot_all_rollout_ablation_by_week_bucket()
    plotter.plot_selected_rollout_ablation_by_week_bucket()
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
