
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd


class DataStatisticsPlottingHelper:
    
    @staticmethod
    def plot_distribution(dist: pd.DataFrame, out_png: Path, title_suffix: str = "") -> None:
        """
        Plot relative frequency vs requests_per_day and save as PNG.
        Must not set explicit colors/styles per environment rules.
        """
        # Avoid error on empty
        if dist.empty:
            # Create an empty axis with labels so the file still exists
            plt.figure(figsize=(8, 5))
            plt.xlabel("Requests entering a floor in a day")
            plt.ylabel("Relative frequency")
            ttl = "Distribution of requests per floor-day"
            if title_suffix:
                ttl += f" {title_suffix}"
            plt.title(ttl)
            plt.tight_layout()
            plt.savefig(out_png, dpi=160, bbox_inches="tight")
            plt.close()
            return

        x = dist["requests_per_day"].astype(int)
        y = dist["relative_frequency"].astype(float)

        plt.figure(figsize=(8, 5))
        plt.bar(x, y)  # default style/colors only
        plt.xlabel("Requests entering a floor in a day")
        plt.ylabel("Relative frequency")
        ttl = "Distribution of requests per floor-day"
        if title_suffix:
            ttl += f" {title_suffix}"
        plt.title(ttl)
        plt.tight_layout()
        plt.savefig(out_png, dpi=160, bbox_inches="tight")
        plt.close()
    
    @staticmethod
    def plot_weekly_u_chart(weekly: pd.DataFrame, out_png: Path) -> None:
        """
        Plot u vs time with center line and control limits.
        """
        plt.figure(figsize=(10, 5))
        # x-axis as week_start
        x = weekly["week_start"]
        plt.plot(x, weekly["u"], marker="o")
        plt.plot(x, weekly["ucl"], linestyle="--")
        plt.plot(x, weekly["lcl"], linestyle="--")
        # center line (horizontal) using full span
        if not weekly.empty:
            u_bar = weekly.attrs.get("u_bar", weekly["u"].mean())
            plt.axhline(u_bar, linestyle=":")
        plt.xlabel("ISO Week (by week start)")
        plt.ylabel("Average requests per floor-day (u)")
        plt.title("Weekly Shewhart u-chart: requests per floor-day")
        plt.tight_layout()
        plt.savefig(out_png, dpi=160, bbox_inches="tight")
        plt.close()