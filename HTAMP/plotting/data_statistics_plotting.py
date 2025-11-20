
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd


class DataStatisticsPlottingHelper:
    @staticmethod
    def plot_per_floor(counts: pd.DataFrame, outdir: Path) -> list[Path]:
        out_paths: list[Path] = []
        for floor, grp in counts.groupby("floor"):
        # Ensure sorted by week
            grp = grp.sort_values("week_start")


            # Single-figure bar chart per floor (no subplots)
            plt.figure(figsize=(10, 5))
            x = grp["week_start"].dt.strftime("%Y-%m-%d")
            y = grp["num_requests"]
            plt.bar(x, y)
            plt.title(f"Weekly scheduled requests — Floor {int(floor)}")
            plt.xlabel("Week starting")
            plt.ylabel("# scheduled requests")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()


            outpath = outdir / f"floor_{int(floor)}_weekly_hist.png"
            plt.savefig(outpath, dpi=150, bbox_inches="tight")
            plt.close()
            out_paths.append(outpath)
        return out_paths