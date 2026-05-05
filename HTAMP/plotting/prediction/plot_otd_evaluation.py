from __future__ import annotations

import argparse
from pathlib import Path

from HTAMP.plotting.prediction.otd_plotting import plot_otd_results


def _resolve_path(path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    path = Path(path_str)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="OTDEvaluationPlotter",
        description="Regenerate OTD evaluation plots from existing OTD CSV outputs.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help=(
            "Directory containing OTD outputs. By default, plots are written here "
            "and CSVs are read from otd_prefix_samples.csv and otd_summary.csv."
        ),
    )
    parser.add_argument(
        "--detail_path",
        default=None,
        help="Optional path to otd_prefix_samples.csv. Defaults to output_dir/otd_prefix_samples.csv.",
    )
    parser.add_argument(
        "--summary_path",
        default=None,
        help="Optional path to otd_summary.csv. Defaults to output_dir/otd_summary.csv.",
    )
    parser.add_argument(
        "--task_label",
        default="TPP",
        help="Label used in plot titles, e.g. 'Delivery TPP' or 'Vital-sign TPP'.",
    )
    parser.add_argument(
        "--ascending_cost_order",
        action="store_true",
        help="Plot models from lowest to highest mean OTD cost instead of descending cost order.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = _resolve_path(args.output_dir)
    if output_dir is None:
        raise ValueError("--output_dir is required.")
    detail_path = _resolve_path(args.detail_path) or output_dir / "otd_prefix_samples.csv"
    summary_path = _resolve_path(args.summary_path) or output_dir / "otd_summary.csv"

    plot_paths = plot_otd_results(
        output_dir=output_dir,
        detail_path=detail_path,
        summary_path=summary_path,
        descending_cost_order=not args.ascending_cost_order,
        task_label=args.task_label,
        log=print,
    )
    if not plot_paths:
        print("No plots were generated. Check that the CSVs exist and contain successful rows.")
        return 1
    for plot_path in plot_paths:
        print(f"Plot saved to {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
