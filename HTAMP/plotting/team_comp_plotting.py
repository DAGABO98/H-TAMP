import os
import re
import traceback
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

class TeamCompPlotter:
        """Helper class to extract team composition counts from log files and plot histograms."""
        # (This is just a wrapper around the functions below, which you can also use directly.)
        def __init__(self, 
                     folder: str | Path, 
                     weeks_to_ignore: list[tuple[int, int]],
                     pattern="*.out", 
                     out_dir="results/team_comp/plots"):
            os.makedirs(out_dir, exist_ok=True)
            self.out_dir = Path(out_dir)
            self.date_in_name_regex = re.compile(r"(\d{4}-\d{2}-\d{2})")  # extract date from filename
            self.df = self.build_counts_df(folder, pattern)
            self.df = self._add_week_columns(self.df)
            self.df = self.filter_out_weeks(self.df, weeks_to_ignore=weeks_to_ignore)
        
        def _extract_valid_team_counts(self, path: str | Path):
            path = Path(path)
            text = path.read_text(errors="ignore")

            valid_marker_regex = re.compile(r"Valid team composition found", re.IGNORECASE)
            counts_line_regex = re.compile(
                r"Monitoring\s*Robots\s*:\s*(\d+)\s*,\s*Delivery\s*Robots\s*:\s*(\d+)",
                re.IGNORECASE
            )

            markers = list(valid_marker_regex.finditer(text))
            if not markers:
                return None

            start = markers[-1].end()
            tail = text[start : start + 2000]

            m = counts_line_regex.search(tail) or counts_line_regex.search(text)
            if not m:
                return None

            return int(m.group(1)), int(m.group(2))  # monitoring, delivery

        def build_counts_df(self, folder: str | Path, pattern="*.out"):
            folder = Path(folder)
            rows = []
            for f in sorted(folder.glob(pattern)):
                counts = self._extract_valid_team_counts(f)
                if counts is None:
                    continue
                monitoring, delivery = counts
                rows.append({"file": f.name, "monitoring_robots": monitoring, "delivery_robots": delivery})
            return pd.DataFrame(rows)

        def _add_week_columns(self, df: pd.DataFrame) -> pd.DataFrame:
            """Add date + ISO week columns based on date in filename."""
            df = df.copy()
            df["date_str"] = df["file"].str.extract(self.date_in_name_regex, expand=False)
            df["date"] = pd.to_datetime(df["date_str"], errors="coerce")

            iso = df["date"].dt.isocalendar()  # year/week/day
            df["iso_year"] = iso["year"].astype("Int64")
            df["iso_week"] = iso["week"].astype("Int64")
            df["year_week"] = df["iso_year"].astype(str) + "-W" + df["iso_week"].astype(str).str.zfill(2)
            return df
        
        def filter_out_weeks(self, df: pd.DataFrame, 
                             weeks_to_ignore: list[tuple[int, int]]) -> pd.DataFrame:
            """
            weeks_to_ignore can be:
            - [(2024, 26), (2024, 27)]
            """
            if df.empty:
                return df.copy()

            df = df.copy()

            if not weeks_to_ignore:
                return df

            ignore = set(weeks_to_ignore)
            mask = ~df.apply(lambda r: (int(r["iso_year"]), int(r["iso_week"])) in ignore, axis=1)
            return df[mask]
        
        def _integer_bins(self, series: pd.Series):
            """Nice bins for integer-valued histograms: one bar per integer."""
            s = series.dropna().astype(int)
            if s.empty:
                return None
            lo, hi = int(s.min()), int(s.max())
            # bins centered on integers: [-0.5, 0.5, 1.5, ...]
            return [x - 0.5 for x in range(lo, hi + 2)]
        
        def hist_with_counts(self, ax, data: pd.Series, bins, xlabel: str, title: str):
            data = pd.to_numeric(data, errors="coerce").dropna().astype(int).to_numpy()
            counts, edges, _ = ax.hist(data, bins=bins)

            ax.set_xlabel(xlabel)
            ax.set_ylabel("Number of files")
            ax.set_title(title)

            # Add count labels on top of each bar
            for c, left, right in zip(counts, edges[:-1], edges[1:]):
                if c == 0:
                    continue
                x = (left + right) / 2
                ax.text(x, c, f"{int(c)}", ha="center", va="bottom")

            # Add a little headroom so labels don't clip
            ymax = counts.max() if len(counts) else 0
            ax.set_ylim(0, ymax * 1.1 + 1)
    
        def plot_histograms(self):
            bins_m = self._integer_bins(self.df["monitoring_robots"])
            bins_d = self._integer_bins(self.df["delivery_robots"])
    
            if bins_m is None or bins_d is None:
                raise ValueError("No valid compositions found (or no counts parsed).")
    
            fig, ax = plt.subplots()
            self.hist_with_counts(
                ax,
                self.df["monitoring_robots"],
                bins=bins_m,
                xlabel="Number of monitoring robots",
                title="Monitoring robots per valid team (filtered weeks)",
            )
            fig.tight_layout()
            plt.savefig(self.out_dir / "monitoring_robots_hist.png", dpi=300, bbox_inches="tight")
            plt.close()
    
            fig, ax = plt.subplots()
            self.hist_with_counts(
                ax,
                self.df["delivery_robots"],
                bins=bins_d,
                xlabel="Number of delivery robots",
                title="Delivery robots per valid team (filtered weeks)",
            )
            fig.tight_layout()
            plt.savefig(self.out_dir / "delivery_robots_hist.png", dpi=300, bbox_inches="tight")
            plt.close()

def main():
    #weeks_to_ignore = [(2024, 26), (2024, 27)]
    weeks_to_ignore = []
    plotter = TeamCompPlotter("results/team_comp/logs", 
                              weeks_to_ignore=weeks_to_ignore)
    plotter.plot_histograms()


if __name__ == "__main__":
    pStart = datetime.now()
    try:
        main()
    except Exception as errorMainContext:
        print("Fail End Process: ", errorMainContext)
        traceback.print_exc()
    pEnd = datetime.now()
    print(f"Total Execution Time: {pEnd - pStart}")