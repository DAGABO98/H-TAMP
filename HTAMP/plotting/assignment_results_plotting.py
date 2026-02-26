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

        self.logs_df = self._parse_all_logs()

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

    def _ordered_policies(self, policies: list[str]) -> list[str]:
        rank = {p: i for i, p in enumerate(self.policy_order)}
        return sorted(policies, key=lambda p: (rank.get(p, 10**9), p))
    
    def _parse_all_logs(self) -> pd.DataFrame:
        if not self.logs_root_dir.exists():
            raise FileNotFoundError(f"logs_root_dir not found: {self.logs_root_dir}")

        rows = []
        for policy_dir in sorted([p for p in self.logs_root_dir.iterdir() if p.is_dir()]):
            policy = policy_dir.name

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
    
    def print_policy_totals(self) -> None:
        """
        Print total planning time and total requests aggregated across all logs per policy,
        plus overall planning_time_per_request computed from totals.
        """
        g = (
            self.logs_df.groupby("policy", as_index=False)[["planning_time_sec", "total_requests"]]
            .sum()
        )
        g["planning_time_per_request_sec_from_totals"] = g["planning_time_sec"] / g["total_requests"].clip(lower=1)

        policies = self._ordered_policies(g["policy"].tolist())
        g = g.set_index("policy").reindex(policies).reset_index()

        print("\n" + "=" * 80)
        print("Planning time totals per policy (across all log files)")
        print("=" * 80)
        print(f"{'Policy':30s}  {'TotalReq':>10s}  {'PlanSec':>12s}  {'PlanSec/Req':>12s}")
        print("-" * 80)
        for _, r in g.iterrows():
            print(
                f"{r['policy'][:30]:30s}  "
                f"{int(r['total_requests']):10d}  "
                f"{float(r['planning_time_sec']):12.2f}  "
                f"{float(r['planning_time_per_request_sec_from_totals']):12.4f}"
            )
    
    def print_policy_mean_pm_std(self) -> None:
        """
        Mean ± 1 std of planning_time_per_request_sec across (day,floor) rows for each policy.
        Units: seconds per request.
        """
        g = (
            self.logs_df.groupby("policy")["planning_time_per_request_sec"]
            .agg(["count", "mean", "std"])
            .reset_index()
        )

        policies = self._ordered_policies(g["policy"].tolist())
        g = g.set_index("policy").reindex(policies).reset_index()

        print("\n" + "=" * 80)
        print("Planning time per request (across day-floor units): mean ± 1 std")
        print("=" * 80)
        print(f"{'Policy':30s}  {'N(day-floor)':>12s}  {'Mean (s/req)':>14s}  {'Std (s/req)':>14s}  {'Mean ± Std':>24s}")
        print("-" * 80)

        for _, r in g.iterrows():
            policy = str(r["policy"])
            n = int(r["count"])
            mean = float(r["mean"]) if np.isfinite(r["mean"]) else float("nan")
            std = float(r["std"]) if np.isfinite(r["std"]) else 0.0  # std is NaN if n==1

            print(
                f"{policy[:30]:30s}  "
                f"{n:12d}  "
                f"{mean:14.4f}  "
                f"{std:14.4f}  "
                f"{mean:10.4f} ± {std:10.4f}"
            )
    
    def plot_box_per_policy(self, filename: str | None = None) -> None:
        """
        One combined plot:
          x-axis = policy
          each policy's box = distribution of planning_time_per_request across (day,floor)
        """
        filename = filename or "planning_time_per_request_boxplot.png"
        out_png = self.out_dir / filename

        policies = self._ordered_policies(self.logs_df["policy"].unique().tolist())
        data = []
        labels = []
        for p in policies:
            vals = self.logs_df.loc[self.logs_df["policy"] == p, "planning_time_per_request_sec"].to_numpy()
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            labels.append(p)
            data.append(vals)

        if not data:
            print("[WARN] No data to plot.")
            return

        plt.figure(figsize=(max(10, len(labels) * 0.9), 5))
        plt.boxplot(data, tick_labels=labels, showfliers=True)
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("Planning time per request (seconds) [per day-floor]")
        plt.title("Planning time per request by policy (daily-per-floor distribution)")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()
        print(f"[INFO] Wrote plot: {out_png}")

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
        required = {"policy", "request_type", "rejected", "serviced", "total"}
        missing = required - set(self.summary_by_type.columns)
        if missing:
            raise ValueError(f"summary_by_type missing required columns: {sorted(missing)}")

        all_policies = self._ordered_policies(self.summary_by_type["policy"].unique().tolist())

        type_order = (
            self.summary_by_type.groupby("request_type")["total"].sum()
            .sort_values(ascending=False)
            .index.tolist()
        )

        for req_type in type_order:
            g = self.summary_by_type[self.summary_by_type["request_type"] == req_type]
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

    def generate_dailyfloor_wait_time_box_plot_by_type(self):
        """
        For each request_type:
          - x-axis: policy
          - each policy's box: distribution of DAILY-PER-FLOOR wait stats across (day,floor)
        """
        out_dir = self.out_dir / "by_request_type"
        out_dir.mkdir(parents=True, exist_ok=True)

        keys = []
        for (policy, req_type), vals in self.dailyfloor_wait_stats_by_policy_type.items():
            keys.append({"policy": policy, "request_type": req_type, "vals": vals})
        kdf = pd.DataFrame(keys)

        if kdf.empty:
            print("[WARN] No daily-per-floor serviced stats found. Skipping plots.")
            return

        for req_type, g in kdf.groupby("request_type"):
            policies_in_plot = self._ordered_policies(g["policy"].tolist())
            g = g.set_index("policy").reindex(policies_in_plot).reset_index()
            policies = g["policy"].tolist()
            data = g["vals"].tolist()

            # Filter empty arrays (matplotlib boxplot complains)
            policies2, data2 = [], []
            for p, d in zip(policies, data):
                if isinstance(d, np.ndarray) and d.size > 0:
                    policies2.append(p)
                    data2.append(d)

            if not data2:
                continue

            plt.figure(figsize=(max(10, len(policies2) * 0.9), 5))
            plt.boxplot(data2, tick_labels=policies2, showfliers=True)
            plt.xticks(rotation=35, ha="right")
            plt.ylabel(
                f"Daily-per-floor {self.daily_stat} wait time (seconds), serviced only\n"
                "wait = max(0, planned_time - scheduled_time)"
            )
            plt.title(f"Daily-per-floor {self.daily_stat} serviced wait time per policy — request_type={req_type}")
            plt.tight_layout()

            safe = self._safe_filename(req_type)
            out_png = out_dir / f"policy_dailyfloor_{self.daily_stat}_wait_time_box__{safe}.png"
            plt.savefig(out_png, dpi=200)
            plt.close()
            print(f"[INFO] Wrote plot: {out_png}")


def main():
    plotter = AssignmentResultsPlotter(
        root_dir="results/policies",
        out_dir="results/policies/plots",
        type_col="request_type",
        daily_stat="p95",
    )
    plotter.print_counts_per_policy_per_request_type()
    plotter.generate_dailyfloor_wait_time_box_plot_by_type()
    plotter.plot_box_per_policy(filename="planning_time_per_request_boxplot.png")
    plotter.print_policy_totals()
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