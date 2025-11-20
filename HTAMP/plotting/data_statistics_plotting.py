
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
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

    @staticmethod
    def plot_heatmap(weekly: pd.DataFrame, out_png: Path) -> None:
        """
        Pivot to floor × week matrix and plot heatmap with matplotlib.
        """
        if weekly.empty:
            # still create an empty figure so a file exists
            plt.figure(figsize=(10, 6))
            plt.title("Floor × Week total requests (no data)")
            plt.xlabel("ISO week")
            plt.ylabel("Floor")
            plt.tight_layout()
            plt.savefig(out_png, dpi=160, bbox_inches="tight")
            plt.close()
            return

        # order columns by week_start, de-duplicated
        weeks = weekly[["iso_label","week_start"]].drop_duplicates().sort_values("week_start")
        week_order = weeks["iso_label"].tolist()

        # Build pivot table
        pivot = weekly.pivot_table(
            index="__floor__", columns="iso_label", values="total_requests", aggfunc="sum", fill_value=0
        )
        # Reindex columns to chronological order
        pivot = pivot.reindex(columns=week_order)

        # Plot
        plt.figure(figsize=(max(10, len(pivot.columns)*0.6), max(6, len(pivot.index)*0.4)))
        im = plt.imshow(pivot.values, aspect="auto")  # default colormap/styles only
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title("Floor × Week total requests")
        plt.xlabel("ISO week")
        plt.ylabel("Floor")

        # Tick labels
        plt.xticks(ticks=np.arange(pivot.shape[1]), labels=pivot.columns, rotation=90)
        plt.yticks(ticks=np.arange(pivot.shape[0]), labels=pivot.index)

        plt.tight_layout()
        plt.savefig(out_png, dpi=160, bbox_inches="tight")
        plt.close()