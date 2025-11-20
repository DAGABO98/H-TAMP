
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
    def plot_heatmap(weekly_dow: pd.DataFrame, out_png: Path) -> None:
        """
        Pivot to ISO week (rows) × DOW (cols) and plot.
        """
        dow_order = [0,1,2,3,4,5,6]  # Monday..Sunday
        dow_names = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}

        if weekly_dow.empty:
            plt.figure(figsize=(8, 6))
            plt.title("ISO Week × Day-of-Week total requests (no data)")
            plt.xlabel("Day of Week")
            plt.ylabel("ISO Week")
            plt.tight_layout()
            plt.savefig(out_png, dpi=160, bbox_inches="tight")
            plt.close()
            return

        # sort weeks by week_start; build readable labels
        wk = weekly_dow[["iso_year","iso_week","week_start"]].drop_duplicates().sort_values("week_start")
        wk["label"] = wk["iso_year"].astype(str) + "-W" + wk["iso_week"].astype(str).str.zfill(2)

        # pivot table
        pivot = weekly_dow.pivot_table(
            index=["iso_year","iso_week"], columns="dow", values="total_requests", aggfunc="sum", fill_value=0
        )
        # reindex rows to chronological order
        pivot = pivot.reindex(index=list(zip(wk["iso_year"], wk["iso_week"])))

        # ensure all 7 columns in Mon..Sun order
        for d in dow_order:
            if d not in pivot.columns:
                pivot[d] = 0
        pivot = pivot[dow_order]

        # plot
        plt.figure(figsize=(10, max(6, 0.4 * pivot.shape[0])))
        im = plt.imshow(pivot.values, aspect="auto")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xlabel("Day of Week")
        plt.ylabel("ISO Week")
        plt.title("ISO Week × Day-of-Week total requests")

        # ticks
        plt.xticks(ticks=np.arange(7), labels=[dow_names[d] for d in dow_order], rotation=0)

        # y tick labels as iso labels in order
        plt.yticks(ticks=np.arange(pivot.shape[0]), labels=wk["label"] if pivot.shape[0] == len(wk) else None)

        plt.tight_layout()
        plt.savefig(out_png, dpi=160, bbox_inches="tight")
        plt.close()