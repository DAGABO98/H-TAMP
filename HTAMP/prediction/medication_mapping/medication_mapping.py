from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from HTAMP.prediction.medication_mapping.rxnorm_atc_mapper import normalize_text

DEFAULT_MEDICATION_NAME_CANDIDATES = (
    "Medication Generic Name",
    "Medication Name",
    "Medication",
    "Medication Display Name",
    "Medication Description",
    "Order Med Name",
    "Order Medication",
    "Drug Name",
    "Generic Name",
    "med_code",
    "medication_code",
    "order_med_id",
    "Order Med ID",
)
SUPPORTED_MEDICATION_CODE_STRATEGIES = (
    "raw_name",
    "rxnorm_rxcui",
    "primary_atc4",
    "primary_atc3",
)
SUPPORTED_MEDICATION_MAPPING_FALLBACKS = (
    "keep_clean_name",
    "drop",
)
MULTI_VALUE_SPLIT_PATTERN = re.compile(r"[;|,]")


def _normalize_column_name(column_name: str) -> str:
    return "".join(character for character in str(column_name).lower() if character.isalnum())


def _match_first_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    normalized_columns = {
        _normalize_column_name(column_name): str(column_name)
        for column_name in columns
    }
    for candidate in candidates:
        matched_column = normalized_columns.get(_normalize_column_name(candidate))
        if matched_column is not None:
            return matched_column
    return None


def _normalize_string_series(series: pd.Series) -> pd.Series:
    normalized = series.where(pd.notna(series), pd.NA).astype(str).str.strip()
    return normalized.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})


def resolve_medication_name_column(
    columns: Sequence[str],
    explicit_col: Optional[str] = None,
) -> str:
    if explicit_col is not None:
        matched = _match_first_column(columns, [explicit_col])
        if matched is None:
            raise ValueError(
                f"Configured medication_code_col '{explicit_col}' was not found in the medication request data."
            )
        return matched

    matched = _match_first_column(columns, DEFAULT_MEDICATION_NAME_CANDIDATES)
    if matched is not None:
        return matched

    heuristic_columns = [
        str(column_name)
        for column_name in columns
        if "med" in _normalize_column_name(str(column_name))
        and "time" not in _normalize_column_name(str(column_name))
        and "dttm" not in _normalize_column_name(str(column_name))
        and "space" not in _normalize_column_name(str(column_name))
        and "room" not in _normalize_column_name(str(column_name))
    ]
    if heuristic_columns:
        return heuristic_columns[0]

    raise ValueError(
        "Could not detect a medication identifier column in the medication request data. "
        "Please set dataset_config.medication_code_col explicitly."
    )


def _value(row: pd.Series, key: str) -> str:
    value = row.get(key, "")
    if pd.isna(value):
        return ""
    value = str(value).strip()
    return "" if value.lower() == "nan" else value


def _first_multi_value(row: pd.Series, key: str) -> str:
    value = _value(row, key)
    if not value:
        return ""
    return next(
        (
            token.strip()
            for token in MULTI_VALUE_SPLIT_PATTERN.split(value)
            if token.strip()
        ),
        "",
    )


def choose_code(row: pd.Series, strategy: str) -> str:
    if strategy == "raw_name":
        return _value(row, "med_name_normalized")
    if strategy == "rxnorm_rxcui":
        return _value(row, "rxnorm_rxcui")
    if strategy == "primary_atc4":
        return _value(row, "primary_atc4") or _value(row, "primary_atc3") or _value(row, "rxnorm_rxcui")
    if strategy == "primary_atc3":
        return _value(row, "primary_atc3") or _value(row, "rxnorm_rxcui")
    raise ValueError(f"Unsupported medication code strategy: {strategy}")


def choose_code_type(row: pd.Series, strategy: str) -> str:
    if strategy == "raw_name":
        return "RAW_NAME"
    if strategy == "rxnorm_rxcui":
        return "RXNORM" if _value(row, "rxnorm_rxcui") else ""
    if strategy == "primary_atc4":
        if _value(row, "primary_atc4"):
            return "ATC4"
        if _value(row, "primary_atc3"):
            return "ATC3"
        if _value(row, "rxnorm_rxcui"):
            return "RXNORM"
        return ""
    if strategy == "primary_atc3":
        if _value(row, "primary_atc3"):
            return "ATC3"
        if _value(row, "rxnorm_rxcui"):
            return "RXNORM"
        return ""
    return ""


def choose_display_name(row: pd.Series, strategy: str) -> str:
    raw_name = _value(row, "med_name_raw")
    if strategy == "raw_name":
        return raw_name or _value(row, "med_name_normalized")
    if strategy == "rxnorm_rxcui":
        return _value(row, "rxnorm_name") or raw_name or _value(row, "med_name_normalized")
    if strategy == "primary_atc4":
        return (
            _first_multi_value(row, "atc4_names")
            or _first_multi_value(row, "atc3_names")
            or _value(row, "rxnorm_name")
            or raw_name
            or _value(row, "med_name_normalized")
        )
    if strategy == "primary_atc3":
        return (
            _first_multi_value(row, "atc3_names")
            or _value(row, "rxnorm_name")
            or raw_name
            or _value(row, "med_name_normalized")
        )
    return raw_name or _value(row, "med_name_normalized")


@dataclass
class MedicationMappingApplier:
    mapping_csv: Optional[str] = None
    code_strategy: str = "raw_name"
    fallback_strategy: str = "keep_clean_name"

    def __post_init__(self) -> None:
        self.code_strategy = str(self.code_strategy).strip().lower()
        if self.code_strategy not in SUPPORTED_MEDICATION_CODE_STRATEGIES:
            raise ValueError(
                f"Unsupported medication_code_strategy '{self.code_strategy}'. "
                f"Expected one of {SUPPORTED_MEDICATION_CODE_STRATEGIES}."
            )

        self.fallback_strategy = str(self.fallback_strategy).strip().lower()
        if self.fallback_strategy not in SUPPORTED_MEDICATION_MAPPING_FALLBACKS:
            raise ValueError(
                f"Unsupported medication_mapping_fallback '{self.fallback_strategy}'. "
                f"Expected one of {SUPPORTED_MEDICATION_MAPPING_FALLBACKS}."
            )

        self.mapping_path = Path(self.mapping_csv) if self.mapping_csv else None
        self.mapping_df = self._load_mapping_df()

    @classmethod
    def from_dataset_config(cls, dataset_config: Any) -> "MedicationMappingApplier":
        return cls(
            mapping_csv=getattr(dataset_config, "medication_mapping_csv", None),
            code_strategy=getattr(dataset_config, "medication_code_strategy", "raw_name"),
            fallback_strategy=getattr(dataset_config, "medication_mapping_fallback", "keep_clean_name"),
        )

    def _load_mapping_df(self) -> Optional[pd.DataFrame]:
        if self.mapping_path is None:
            return None
        if not self.mapping_path.exists():
            raise FileNotFoundError(f"Medication mapping CSV was not found at '{self.mapping_path}'.")

        mapping_df = pd.read_csv(self.mapping_path, dtype=str).fillna("")
        if "raw_name" not in mapping_df.columns:
            raise ValueError("Medication mapping CSV must contain a 'raw_name' column.")

        mapping_df = mapping_df.copy()
        mapping_df["__mapping_key"] = mapping_df["raw_name"].astype(str).map(normalize_text)
        mapping_df = mapping_df[mapping_df["__mapping_key"] != ""].copy()
        mapping_df = mapping_df.drop_duplicates(subset=["__mapping_key"], keep="first").reset_index(drop=True)
        return mapping_df

    def _fallback_code(self, row: pd.Series) -> str:
        if self.fallback_strategy == "keep_clean_name":
            return _value(row, "med_name_normalized")
        return ""

    def _fallback_code_type(self, row: pd.Series) -> str:
        if self.fallback_strategy == "keep_clean_name" and _value(row, "med_name_normalized"):
            return "RAW_NAME"
        return ""

    def _fallback_display_name(self, row: pd.Series) -> str:
        return _value(row, "med_name_raw") or _value(row, "med_name_normalized")

    def to_metadata(self) -> dict[str, object]:
        return {
            "mapping_csv": str(self.mapping_path) if self.mapping_path is not None else None,
            "code_strategy": self.code_strategy,
            "fallback_strategy": self.fallback_strategy,
            "mapping_row_count": int(len(self.mapping_df)) if self.mapping_df is not None else 0,
        }

    def apply(
        self,
        event_df: pd.DataFrame,
        *,
        med_name_col: str,
    ) -> pd.DataFrame:
        if med_name_col not in event_df.columns:
            raise ValueError(f"Medication name column '{med_name_col}' was not found in the input frame.")

        mapped_df = event_df.copy()
        mapped_df["med_name_raw"] = _normalize_string_series(mapped_df[med_name_col])
        mapped_df["med_name_normalized"] = mapped_df["med_name_raw"].map(
            lambda value: normalize_text(str(value)) if pd.notna(value) else ""
        )
        mapped_df["__mapping_key"] = mapped_df["med_name_normalized"]

        if self.mapping_df is not None:
            mapped_df = mapped_df.merge(self.mapping_df, on="__mapping_key", how="left", suffixes=("", "_map"))
        else:
            mapped_df["review_required"] = False
            mapped_df["status"] = ""

        mapped_df["med_code"] = mapped_df.apply(lambda row: choose_code(row, self.code_strategy), axis=1)
        mapped_df["med_code_type"] = mapped_df.apply(lambda row: choose_code_type(row, self.code_strategy), axis=1)
        mapped_df["med_display_name"] = mapped_df.apply(lambda row: choose_display_name(row, self.code_strategy), axis=1)

        missing_code_mask = mapped_df["med_code"].astype(str).str.strip() == ""
        if missing_code_mask.any():
            mapped_df.loc[missing_code_mask, "med_code"] = mapped_df.loc[missing_code_mask].apply(
                self._fallback_code,
                axis=1,
            )
            mapped_df.loc[missing_code_mask, "med_code_type"] = mapped_df.loc[missing_code_mask].apply(
                self._fallback_code_type,
                axis=1,
            )
            mapped_df.loc[missing_code_mask, "med_display_name"] = mapped_df.loc[missing_code_mask].apply(
                self._fallback_display_name,
                axis=1,
            )

        status_column = mapped_df.get("status")
        if isinstance(status_column, pd.Series):
            mapped_df["med_mapping_status"] = status_column.astype(str)
        else:
            mapped_df["med_mapping_status"] = ""
        review_column = mapped_df.get("review_required", False)
        if isinstance(review_column, pd.Series):
            mapped_df["med_mapping_review_required"] = review_column.astype(str).str.lower().isin({"true", "1"})
        else:
            mapped_df["med_mapping_review_required"] = bool(review_column)

        mapped_df["med_code"] = _normalize_string_series(mapped_df["med_code"])
        mapped_df["med_display_name"] = _normalize_string_series(mapped_df["med_display_name"])
        return mapped_df.drop(columns=["__mapping_key"])
