
import os
import re
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

class AssignmentResultsPlotter:
    """Helper class to extract team composition counts from log files and plot histograms."""
    # (This is just a wrapper around the functions below, which you can also use directly.)
    def __init__(self, 
                 root_dir: str | Path,
                 out_dir="results/policies/plots"):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = Path(out_dir)
        self.root_dir = Path(root_dir)
        self.summary_by_type, self.serviced_waits_by_policy_type = self._generate_results_summary_from_root_dir(type_col="request_type")
    
    def _generate_results_summary_from_root_dir(self, type_col: str = "request_type"):
        rows = []
        serviced_waits_by_policy_type: dict[tuple[str, str], np.ndarray] = {}

        for policy_dir in sorted([p for p in self.root_dir.iterdir() if p.is_dir()]):
            policy_name = policy_dir.name
            csv_files = sorted(policy_dir.glob("*.csv"))
            if not csv_files:
                continue

            frames = []
            for f in csv_files:
                try:
                    frames.append(pd.read_csv(f))
                except Exception as e:
                    print(f"[WARN] Failed reading {f}: {e}")

            if not frames:
                continue

            df = pd.concat(frames, ignore_index=True)

            required = ["completed", "rejected", "planned_time", "scheduled_time", type_col]
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f"Policy '{policy_name}' is missing columns: {missing}")

            # Normalize request types (avoid NaN types breaking groups)
            df[type_col] = df[type_col].astype(str).fillna("UNKNOWN")

            # Group per request type
            for req_type, g in df.groupby(type_col, dropna=False):
                total = int(len(g))
                serviced = int(g["completed"].fillna(False).astype(bool).sum())
                rejected = int(g["rejected"].fillna(False).astype(bool).sum())

                rows.append({
                    "policy": policy_name,
                    "request_type": req_type,
                    "rejected": rejected,
                    "serviced": serviced,
                    "total": total,
                })

                # wait time for SERVICED in this (policy, type)
                serviced_g = g[g["completed"].fillna(False).astype(bool)].copy()
                planned = pd.to_numeric(serviced_g["planned_time"], errors="coerce")
                scheduled = pd.to_numeric(serviced_g["scheduled_time"], errors="coerce")
                wait_time = (planned - scheduled).clip(lower=0)

                waits = wait_time.to_numpy()
                waits = waits[np.isfinite(waits)]
                serviced_waits_by_policy_type[(policy_name, req_type)] = waits

        summary_by_type = pd.DataFrame(rows)
        if summary_by_type.empty:
            raise ValueError(f"No policy folders with readable CSVs found under: {self.root_dir}")

        summary_by_type = summary_by_type.sort_values(["policy", "request_type"]).reset_index(drop=True)
        return summary_by_type, serviced_waits_by_policy_type
    
    def generate_number_of_requests_plots_by_type(self):
        out_dir = self.out_dir / "by_request_type"
        out_dir.mkdir(parents=True, exist_ok=True)

        for req_type, g in self.summary_by_type.groupby("request_type"):
            g = g.sort_values("policy")
            policies = g["policy"].tolist()
            x = np.arange(len(policies))
            width = 0.25

            plt.figure(figsize=(max(10, len(policies) * 0.9), 5))
            b1 = plt.bar(x - width, g["rejected"], width, label="Rejected")
            b2 = plt.bar(x,         g["serviced"], width, label="Serviced")
            b3 = plt.bar(x + width, g["total"],    width, label="Total")

            plt.xticks(x, policies, rotation=35, ha="right")
            plt.ylabel("Count")
            plt.title(f"Requests per policy — request_type={req_type}")
            plt.legend()

            for bars in (b1, b2, b3):
                for rect in bars:
                    h = rect.get_height()
                    plt.text(rect.get_x() + rect.get_width()/2, h, f"{int(h)}",
                            ha="center", va="bottom", fontsize=8)

            plt.tight_layout()
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(req_type))
            plt.savefig(out_dir / f"policy_request_totals__{safe}.png", dpi=200)
            plt.close()

    def generate_serviced_wait_time_box_plot_by_type(self):
        out_dir = self.out_dir / "by_request_type"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build a DataFrame view for easier grouping
        keys = []
        for (policy, req_type), waits in self.serviced_waits_by_policy_type.items():
            keys.append({"policy": policy, "request_type": req_type, "waits": waits})
        kdf = pd.DataFrame(keys)

        for req_type, g in kdf.groupby("request_type"):
            g = g.sort_values("policy")
            policies = g["policy"].tolist()
            data = g["waits"].tolist()

            plt.figure(figsize=(max(10, len(policies) * 0.9), 5))
            plt.boxplot(data, tick_labels=policies, showfliers=True)
            plt.xticks(rotation=35, ha="right")
            plt.ylabel("Wait time (seconds) = max(0, planned_time - scheduled_time)")
            plt.title(f"Serviced request wait time per policy — request_type={req_type}")
            plt.tight_layout()

            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(req_type))
            plt.savefig(out_dir / f"policy_serviced_wait_time_box__{safe}.png", dpi=200)
            plt.close()

def main():
    plotter = AssignmentResultsPlotter(root_dir="results/policies")
    plotter.generate_number_of_requests_plots_by_type()
    plotter.generate_serviced_wait_time_box_plot_by_type()

if __name__ == "__main__":
    pStart = datetime.now()
    try:
        main()
    except Exception as errorMainContext:
        print("Fail End Process: ", errorMainContext)
        traceback.print_exc()
    pEnd = datetime.now()
    print(f"Total Execution Time: {pEnd - pStart}")