from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from HTAMP.planning.request_handler import GlobalRequestHandler
from HTAMP.prediction.configs.delivery_request_config import DeliveryRequestDatasetConfig
from HTAMP.prediction.medication_mapping.medication_mapping import resolve_medication_name_column
from HTAMP.prediction.medication_mapping.rxnorm_atc_mapper import RxNormAtcMapper, normalize_text


def _load_json_object(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as config_file:
        payload = json.load(config_file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected '{config_path}' to contain a JSON object.")
    return payload


def _load_dataset_config(config_path: str) -> DeliveryRequestDatasetConfig:
    payload = _load_json_object(config_path=config_path)
    if "dataset_config" in payload:
        payload = payload["dataset_config"]
    return DeliveryRequestDatasetConfig.from_dict(payload)


def _load_medication_names_from_config(
    *,
    config_path: str,
    med_name_col: Optional[str],
) -> list[str]:
    dataset_config = _load_dataset_config(config_path=config_path)
    request_handler = GlobalRequestHandler(
        annotated_data_files=dataset_config.annotated_data_files,
        request_dir=dataset_config.request_dir,
        start_date=dataset_config.start_date,
        end_date=dataset_config.end_date,
        use_saved_data=dataset_config.use_saved_request_data,
        included_tasks=("medication",),
    )
    med_df = request_handler.med_df.copy()
    if med_df.empty:
        return []

    resolved_col = resolve_medication_name_column(
        columns=med_df.columns.tolist(),
        explicit_col=med_name_col or dataset_config.medication_code_col,
    )
    return med_df[resolved_col].astype(str).tolist()


def _load_medication_names_from_csvs(
    *,
    input_csvs: List[str],
    med_name_col: str,
) -> list[str]:
    names: list[str] = []
    for csv_path in input_csvs:
        med_df = pd.read_csv(csv_path, dtype=str).fillna("")
        if med_name_col not in med_df.columns:
            raise ValueError(f"Column '{med_name_col}' was not found in '{csv_path}'.")
        names.extend(med_df[med_name_col].astype(str).tolist())
    return names


def load_manual_overrides(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if path is None:
        return {}
    overrides_df = pd.read_csv(path, dtype=str).fillna("")
    if "raw_name" not in overrides_df.columns:
        raise ValueError("manual overrides CSV must contain a 'raw_name' column")
    overrides: Dict[str, Dict[str, Any]] = {}
    for _, row in overrides_df.iterrows():
        key = normalize_text(str(row["raw_name"]))
        if key:
            overrides[key] = row.to_dict()
    return overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a medication mapping dictionary for the delivery-request pipeline."
    )
    parser.add_argument(
        "--config_path",
        default=None,
        help="Optional delivery JSON config to load repo medication rows.",
    )
    parser.add_argument(
        "--input_csv",
        nargs="+",
        default=None,
        help="Optional CSV files containing medication names.",
    )
    parser.add_argument(
        "--med_name_col",
        default=None,
        help="Medication-name column to use. Defaults to dataset_config.medication_code_col or auto-detection.",
    )
    parser.add_argument("--output_csv", required=True, help="Where to write the mapping dictionary CSV.")
    parser.add_argument(
        "--manual_overrides_csv",
        default=None,
        help="Optional CSV with manual overrides. Must include raw_name and rxnorm_rxcui.",
    )
    parser.add_argument(
        "--rxnav_base_url",
        default="https://rxnav.nlm.nih.gov/REST",
        help="Base URL for RxNorm/RxClass API. Point this to a local RxNav-in-a-Box instance for offline use.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="Optional directory for caching API responses so repeated runs are much faster.",
    )
    parser.add_argument("--timeout_s", type=float, default=20.0)
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for smoke tests.")
    parser.add_argument("--progress_every", type=int, default=50)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.config_path and not args.input_csv:
        raise ValueError("Please provide either --config_path or at least one --input_csv.")
    if args.input_csv and not args.med_name_col and not args.config_path:
        raise ValueError("--med_name_col is required when building from --input_csv without --config_path.")

    medication_names: list[str] = []
    if args.config_path:
        medication_names.extend(
            _load_medication_names_from_config(
                config_path=args.config_path,
                med_name_col=args.med_name_col,
            )
        )
    if args.input_csv:
        if args.med_name_col is None:
            raise ValueError("--med_name_col is required when using --input_csv.")
        medication_names.extend(
            _load_medication_names_from_csvs(
                input_csvs=list(args.input_csv),
                med_name_col=args.med_name_col,
            )
        )

    unique_names = sorted({name for name in medication_names if normalize_text(name)})
    if args.limit is not None:
        unique_names = unique_names[: args.limit]

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cache_dir = args.cache_dir
    if cache_dir is None:
        cache_dir = str(output_path.parent / "rxnav_cache")

    overrides = load_manual_overrides(args.manual_overrides_csv)
    mapper = RxNormAtcMapper(
        base_url=args.rxnav_base_url,
        cache_dir=cache_dir,
        timeout_s=args.timeout_s,
    )

    rows: List[Dict[str, Any]] = []
    for idx, raw_name in enumerate(unique_names, start=1):
        override = overrides.get(normalize_text(raw_name))
        result = mapper.map_name(raw_name, manual_override=override)
        rows.append(result.to_row())

        if idx % args.progress_every == 0 or idx == len(unique_names):
            so_far = pd.DataFrame(rows)
            review_count = int(so_far["review_required"].astype(bool).sum()) if not so_far.empty else 0
            print(
                f"mapped={idx}/{len(unique_names)} "
                f"review_required={review_count} "
                f"manual_overrides={sum(row.get('match_type') == 'manual_override' for row in rows)}"
            )

    output_df = pd.DataFrame(rows)
    output_df = output_df.sort_values(["review_required", "status", "raw_name"], ascending=[False, True, True])
    output_df.to_csv(output_path, index=False)

    review_path = output_path.with_name(output_path.stem + "_needs_review.csv")
    output_df[output_df["review_required"].astype(bool)].to_csv(review_path, index=False)

    print(f"saved mapping dictionary: {output_path}")
    print(f"saved review subset: {review_path}")
    print(output_df["status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
