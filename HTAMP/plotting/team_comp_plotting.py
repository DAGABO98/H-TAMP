import os
import re
import traceback
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

IsoWeek = tuple[int, int]

class TeamCompPlotter:
    """Helper class to extract team composition counts from log files and plot histograms."""
    # (This is just a wrapper around the functions below, which you can also use directly.)
    def __init__(self, 
                    folder: str | Path, 
                    weeks_to_ignore: list[IsoWeek] | None = None,
                    weeks_to_ignore_by_floor: dict[int, set[IsoWeek]] | None = None,
                    pattern="*.out", 
                    out_dir="results/team_comp/plots"):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = Path(out_dir)
        self.date_in_name_regex = re.compile(r"(\d{4}-\d{2}-\d{2})")  # extract date from filename
        self.floor_in_name_regex = re.compile(r"_floor(\d+)", re.IGNORECASE)
        self.df = self.build_counts_df(folder, pattern)
        self.df = self._add_week_columns(self.df)
        self.df = self.filter_out_weeks(
            self.df,
            weeks_to_ignore=weeks_to_ignore,
            weeks_to_ignore_by_floor=weeks_to_ignore_by_floor,
        )
    
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
            floor_match = self.floor_in_name_regex.search(f.stem)
            floor = int(floor_match.group(1)) if floor_match else None
            rows.append(
                {
                    "file": f.name,
                    "floor": floor,
                    "monitoring_robots": monitoring,
                    "delivery_robots": delivery,
                }
            )
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
    
    def filter_out_weeks(self,
                         df: pd.DataFrame,
                         weeks_to_ignore: list[IsoWeek] | None = None,
                         weeks_to_ignore_by_floor: dict[int, set[IsoWeek]] | None = None) -> pd.DataFrame:
        """
        weeks_to_ignore can be:
        - [(2024, 26), (2024, 27)]
        """
        if df.empty:
            return df.copy()

        df = df.copy()

        if weeks_to_ignore is None:
            weeks_to_ignore = []
        if weeks_to_ignore_by_floor is None:
            weeks_to_ignore_by_floor = {}

        if not weeks_to_ignore:
            if not weeks_to_ignore_by_floor:
                return df

        ignore = set(weeks_to_ignore)
        ignore_by_floor = {int(floor): set(weeks) for floor, weeks in weeks_to_ignore_by_floor.items()}

        def should_keep(row: pd.Series) -> bool:
            week_key = (int(row["iso_year"]), int(row["iso_week"]))
            floor = row.get("floor")
            if pd.notna(floor):
                floor_ignore = ignore_by_floor.get(int(floor), ignore)
            else:
                floor_ignore = ignore
            return week_key not in floor_ignore

        mask = df.apply(should_keep, axis=1)
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
    weeks_to_ignore = [
    # Weeks with highest demand
    (2024, 40),
    (2025, 5),

    #Weeks with medium demand
    (2024, 44),
    (2025, 14),

    #Weeks with lowest demand
    (2024, 27),
    (2024, 36)]

    # Optional per-floor override. Any floor omitted here falls back to
    # weeks_to_ignore above.
    weeks_to_ignore_by_floor: dict[int, set[IsoWeek]] = {
        2: {
            # Weeks with highest demand
            (2025, 6),
            (2025, 8),
            
            #Weeks with medium demand
            (2024, 44),
            (2025, 14),

            #Weeks with lowest demand
            (2024, 27),
            (2024, 28)
        },
        3: {
            # Weeks with highest demand
            (2025, 5),
            (2025, 10),
            
            #Weeks with medium demand
            (2024, 44),
            (2025, 14),

            #Weeks with lowest demand
            (2024, 46),
            (2025, 2)
        },
        7: {
            # Weeks with highest demand
            (2024, 39),
            (2024, 40),
            
            #Weeks with medium demand
            (2024, 44),
            (2025, 14),

            #Weeks with lowest demand
            (2024, 46),
            (2025, 26)
        },
        9: {
            # Weeks with highest demand
            (2024, 43),
            (2025, 6),
            
            #Weeks with medium demand
            (2024, 44),
            (2025, 14),

            #Weeks with lowest demand
            (2025, 19),
            (2025, 21)
        }
    }

    plotter = TeamCompPlotter("results/team_comp/logs", 
                              weeks_to_ignore=weeks_to_ignore,
                              weeks_to_ignore_by_floor=weeks_to_ignore_by_floor)
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
