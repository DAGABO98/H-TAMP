from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from HTAMP.planning.request_handler import GlobalRequestHandler
from HTAMP.prediction.configs.delivery_request_config import DeliveryRequestDatasetConfig
from HTAMP.prediction.medication_mapping.medication_mapping import resolve_medication_name_column
from HTAMP.prediction.medication_mapping.rxnorm_atc_mapper import RxNormAtcMapper, normalize_text

LOGGER = logging.getLogger(__name__)
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def _load_json_object(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    LOGGER.debug("Loading JSON config from %s", path)
    with path.open("r", encoding="utf-8") as config_file:
        payload = json.load(config_file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected '{config_path}' to contain a JSON object.")
    return payload


def _load_dataset_config(config_path: str) -> DeliveryRequestDatasetConfig:
    payload = _load_json_object(config_path=config_path)
    if "dataset_config" in payload:
        payload = payload["dataset_config"]
    dataset_config = DeliveryRequestDatasetConfig.from_dict(payload)
    LOGGER.info(
        "Loaded dataset config from %s for %s to %s",
        config_path,
        dataset_config.start_date,
        dataset_config.end_date,
    )
    return dataset_config


def _load_medication_names_from_config(
    *,
    config_path: str,
    med_name_col: Optional[str],
) -> list[str]:
    dataset_config = _load_dataset_config(config_path=config_path)
    LOGGER.info(
        "Loading medication requests from request_dir=%s use_saved_request_data=%s",
        dataset_config.request_dir,
        dataset_config.use_saved_request_data,
    )
    request_handler = GlobalRequestHandler(
        annotated_data_files=dataset_config.annotated_data_files,
        request_dir=dataset_config.request_dir,
        start_date=dataset_config.start_date,
        end_date=dataset_config.end_date,
        use_saved_data=dataset_config.use_saved_request_data,
        included_tasks=("medication",),
    )
    med_df = request_handler.med_df.copy()
    LOGGER.info("Loaded %d medication request rows from config source", len(med_df))
    if med_df.empty:
        LOGGER.warning("Medication request data is empty for config source %s", config_path)
        return []

    resolved_col = resolve_medication_name_column(
        columns=med_df.columns.tolist(),
        explicit_col=med_name_col or dataset_config.medication_code_col,
    )
    names = med_df[resolved_col].astype(str).tolist()
    LOGGER.info(
        "Using medication-name column '%s' from config source; collected %d values",
        resolved_col,
        len(names),
    )
    return names


def _load_medication_names_from_csvs(
    *,
    input_csvs: List[str],
    med_name_col: str,
) -> list[str]:
    names: list[str] = []
    for csv_path in input_csvs:
        LOGGER.info("Loading medication names from CSV %s", csv_path)
        med_df = pd.read_csv(csv_path, dtype=str).fillna("")
        if med_name_col not in med_df.columns:
            raise ValueError(f"Column '{med_name_col}' was not found in '{csv_path}'.")
        csv_names = med_df[med_name_col].astype(str).tolist()
        names.extend(csv_names)
        LOGGER.info(
            "Loaded %d rows and collected %d medication-name values from %s",
            len(med_df),
            len(csv_names),
            csv_path,
        )
    return names


def load_manual_overrides(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if path is None:
        LOGGER.info("No manual overrides CSV provided")
        return {}
    LOGGER.info("Loading manual medication mapping overrides from %s", path)
    overrides_df = pd.read_csv(path, dtype=str).fillna("")
    if "raw_name" not in overrides_df.columns:
        raise ValueError("manual overrides CSV must contain a 'raw_name' column")
    overrides: Dict[str, Dict[str, Any]] = {}
    for _, row in overrides_df.iterrows():
        key = normalize_text(str(row["raw_name"]))
        if key:
            overrides[key] = row.to_dict()
    LOGGER.info("Loaded %d manual override rows with non-empty raw_name keys", len(overrides))
    return overrides


def configure_logging(*, log_level: str, log_file: Optional[str]) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def _shorten_name(raw_name: str, *, max_length: int = 140) -> str:
    raw_name = str(raw_name).replace("\n", " ").strip()
    if len(raw_name) <= max_length:
        return raw_name
    return raw_name[: max_length - 3] + "..."


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
    parser.add_argument(
        "--log_level",
        default="INFO",
        type=str.upper,
        choices=LOG_LEVELS,
        help="Logging verbosity for this script.",
    )
    parser.add_argument(
        "--log_file",
        default=None,
        help="Optional path to also write logs to a file.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(log_level=args.log_level, log_file=args.log_file)
    LOGGER.info("Starting medication mapping CSV generation")
    LOGGER.info(
        "Inputs: config_path=%s input_csv=%s med_name_col=%s",
        args.config_path,
        args.input_csv,
        args.med_name_col,
    )
    LOGGER.info(
        "Outputs: output_csv=%s log_file=%s",
        args.output_csv,
        args.log_file,
    )

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

    LOGGER.info("Collected %d raw medication-name values before de-duplication", len(medication_names))
    unique_names = sorted({name for name in medication_names if normalize_text(name)})
    LOGGER.info("Found %d unique non-empty medication names to map", len(unique_names))
    if args.limit is not None:
        LOGGER.info(
            "Applying --limit=%d; mapping first %d names after sorting",
            args.limit,
            min(args.limit, len(unique_names)),
        )
        unique_names = unique_names[: args.limit]
    if not unique_names:
        raise ValueError("No non-empty medication names were found to map.")

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cache_dir = args.cache_dir
    if cache_dir is None:
        cache_dir = str(output_path.parent / "rxnav_cache")
    LOGGER.info(
        "RxNav settings: base_url=%s cache_dir=%s timeout_s=%s",
        args.rxnav_base_url,
        cache_dir,
        args.timeout_s,
    )

    overrides = load_manual_overrides(args.manual_overrides_csv)
    mapper = RxNormAtcMapper(
        base_url=args.rxnav_base_url,
        cache_dir=cache_dir,
        timeout_s=args.timeout_s,
    )

    rows: List[Dict[str, Any]] = []
    LOGGER.info("Beginning RxNorm/ATC mapping for %d unique medication names", len(unique_names))
    for idx, raw_name in enumerate(unique_names, start=1):
        override = overrides.get(normalize_text(raw_name))
        LOGGER.debug(
            "Mapping medication %d/%d: %s%s",
            idx,
            len(unique_names),
            _shorten_name(raw_name),
            " using manual override" if override is not None else "",
        )
        try:
            result = mapper.map_name(raw_name, manual_override=override)
        except Exception:
            LOGGER.exception(
                "Failed while mapping medication %d/%d: %s",
                idx,
                len(unique_names),
                _shorten_name(raw_name),
            )
            raise
        rows.append(result.to_row())
        LOGGER.debug(
            "Mapped medication %d/%d status=%s match_type=%s rxcui=%s primary_atc4=%s review_required=%s",
            idx,
            len(unique_names),
            result.status,
            result.match_type,
            result.rxnorm_rxcui,
            result.primary_atc4,
            result.review_required,
        )

        if idx % args.progress_every == 0 or idx == len(unique_names):
            so_far = pd.DataFrame(rows)
            review_count = int(so_far["review_required"].astype(bool).sum()) if not so_far.empty else 0
            LOGGER.info(
                "Progress: mapped=%d/%d review_required=%d manual_overrides=%d",
                idx,
                len(unique_names),
                review_count,
                sum(row.get("match_type") == "manual_override" for row in rows),
            )

    LOGGER.info("Building output DataFrame with %d mapping rows", len(rows))
    output_df = pd.DataFrame(rows)
    output_df = output_df.sort_values(["review_required", "status", "raw_name"], ascending=[False, True, True])
    LOGGER.info("Writing mapping dictionary CSV to %s", output_path)
    output_df.to_csv(output_path, index=False)

    review_path = output_path.with_name(output_path.stem + "_needs_review.csv")
    review_count = int(output_df["review_required"].astype(bool).sum())
    LOGGER.info("Writing %d review-required rows to %s", review_count, review_path)
    output_df[output_df["review_required"].astype(bool)].to_csv(review_path, index=False)

    LOGGER.info("Saved mapping dictionary: %s", output_path)
    LOGGER.info("Saved review subset: %s", review_path)
    LOGGER.info("Mapping status counts:\n%s", output_df["status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
