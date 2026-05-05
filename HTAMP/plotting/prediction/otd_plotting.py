from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np


def plot_otd_results(
    *,
    output_dir: Path,
    detail_path: Path,
    summary_path: Path,
    descending_cost_order: bool = True,
    task_label: str = "TPP",
    log: Callable[[str], None] | None = None,
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception as exc:
        if log is not None:
            log(f"Skipping plots because plotting dependencies could not be imported: {exc}")
        return []

    detail_df = pd.read_csv(detail_path)
    summary_df = pd.read_csv(summary_path)
    summary_df = summary_df[summary_df["status"] == "success"].copy()
    if detail_df.empty or summary_df.empty:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: list[Path] = []
    label_column = "run_name"
    summary_df = summary_df.sort_values(
        "otd_total_mean",
        ascending=not descending_cost_order,
    )

    component_path = output_dir / "otd_component_bar.png"
    labels = summary_df[label_column].tolist()
    x_positions = np.arange(len(labels))
    time_values = summary_df["otd_time_mean"].to_numpy(dtype=float)
    type_values = summary_df["otd_type_mean"].to_numpy(dtype=float)
    delete_values = summary_df.get(
        "otd_delete_mean",
        summary_df["otd_edit_mean"] * 0.0,
    ).to_numpy(dtype=float)
    insert_values = summary_df.get(
        "otd_insert_mean",
        summary_df["otd_edit_mean"] * 0.0,
    ).to_numpy(dtype=float)
    residual_values = np.maximum(
        0.0,
        summary_df["otd_total_mean"].to_numpy(dtype=float)
        - time_values
        - type_values
        - delete_values
        - insert_values,
    )
    total_std = summary_df["otd_total_std"].to_numpy(dtype=float)
    fig_width = max(10.0, 0.55 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, 6.0))
    bottoms = np.zeros_like(time_values)
    ax.bar(x_positions, time_values, bottom=bottoms, label="Time")
    bottoms = bottoms + time_values
    ax.bar(x_positions, type_values, bottom=bottoms, label="Type")
    bottoms = bottoms + type_values
    ax.bar(x_positions, delete_values, bottom=bottoms, label="Deletion")
    bottoms = bottoms + delete_values
    ax.bar(x_positions, insert_values, bottom=bottoms, label="Addition")
    bottoms = bottoms + insert_values
    if np.any(residual_values > 1e-12):
        ax.bar(x_positions, residual_values, bottom=bottoms, label="Other")
        bottoms = bottoms + residual_values
    ax.errorbar(
        x_positions,
        bottoms,
        yerr=total_std,
        fmt="none",
        ecolor="black",
        elinewidth=1,
        capsize=2,
    )
    ax.set_ylabel("Mean type/time OTD")
    ax.set_title(f"{task_label} type/time OTD by model")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(component_path, dpi=180)
    plt.close(fig)
    plot_paths.append(component_path)

    boxplot_path = output_dir / "otd_distribution_boxplot.png"
    detail_df = detail_df[detail_df["run_name"].isin(labels)].copy()
    grouped_values = [
        detail_df.loc[detail_df["run_name"] == run_name, "otd_total"].to_numpy(dtype=float)
        for run_name in labels
    ]
    fig, ax = plt.subplots(figsize=(fig_width, 6.0))
    ax.boxplot(grouped_values, labels=labels, showfliers=False)
    ax.set_ylabel("Sample type/time OTD")
    ax.set_title(f"{task_label} type/time OTD distribution")
    ax.tick_params(axis="x", rotation=45)
    for tick_label in ax.get_xticklabels():
        tick_label.set_ha("right")
    fig.tight_layout()
    fig.savefig(boxplot_path, dpi=180)
    plt.close(fig)
    plot_paths.append(boxplot_path)

    if "inference_milliseconds_mean" in summary_df.columns:
        inference_path = output_dir / "inference_latency_bar.png"
        inference_values = summary_df["inference_milliseconds_mean"].to_numpy(dtype=float)
        inference_std = summary_df["inference_milliseconds_std"].to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(fig_width, 6.0))
        ax.bar(x_positions, inference_values, yerr=inference_std, capsize=2)
        ax.set_ylabel("Mean rollout inference latency (ms)")
        ax.set_title(f"{task_label} rollout latency by model")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(inference_path, dpi=180)
        plt.close(fig)
        plot_paths.append(inference_path)

    return plot_paths
