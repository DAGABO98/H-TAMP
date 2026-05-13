from __future__ import annotations

import argparse
import csv
import datetime
import gzip
import json
import shutil
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO


DEFAULT_INPUT_CSV = "data/prediction/offline_request_prediction_cache.csv"
DEFAULT_OUTPUT_DIR = "data/prediction/offline_request_prediction_cache_by_floor_day"
UNKNOWN_VALUE = "unknown"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_repo_relative_path(path_value: str | Path | None) -> Path | None:
    if path_value is None or not str(path_value).strip():
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return _repo_root() / path


def _log(message: str) -> None:
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", flush=True)


def _safe_component(value: Any, *, unknown: str = UNKNOWN_VALUE) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "<na>", "nat"}:
        text = unknown
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in text)
    return cleaned.strip("_") or unknown


def _normalize_floor(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return UNKNOWN_VALUE
    try:
        numeric_value = float(text)
    except ValueError:
        return _safe_component(text)
    if numeric_value.is_integer():
        return str(int(numeric_value))
    return _safe_component(text)


def _normalize_day(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "<na>", "nat"}:
        return UNKNOWN_VALUE
    try:
        return datetime.date.fromisoformat(text[:10]).isoformat()
    except ValueError:
        return _safe_component(text)


def _row_day(row: dict[str, str], day_column: str) -> str:
    for column_name in (
        day_column,
        "sequence_day",
        "scheduled_dttm",
        "Scheduled DTTM",
        "Medication Scheduled DTTM",
        "prediction_anchor_timestamp",
    ):
        value = row.get(column_name, "")
        normalized = _normalize_day(value)
        if normalized != UNKNOWN_VALUE:
            return normalized
    return UNKNOWN_VALUE


def _shard_relative_path(
    *,
    floor: str,
    day: str,
    compression: str,
) -> Path:
    suffix = ".csv.gz" if compression == "gzip" else ".csv"
    return Path(f"floor_{_safe_component(floor)}") / f"day_{_safe_component(day)}{suffix}"


@dataclass
class _OpenShard:
    handle: TextIO
    writer: csv.DictWriter


class _ShardWriterPool:
    def __init__(
        self,
        *,
        output_dir: Path,
        fieldnames: list[str],
        max_open_files: int,
        compression: str,
    ) -> None:
        self.output_dir = output_dir
        self.fieldnames = fieldnames
        self.max_open_files = max(1, int(max_open_files))
        self.compression = compression
        self.open_shards: OrderedDict[tuple[str, str], _OpenShard] = OrderedDict()
        self.initialized_paths: set[Path] = set()

    def writer_for(self, *, floor: str, day: str) -> tuple[csv.DictWriter, Path]:
        key = (floor, day)
        if key in self.open_shards:
            shard = self.open_shards.pop(key)
            self.open_shards[key] = shard
            return shard.writer, _shard_relative_path(
                floor=floor,
                day=day,
                compression=self.compression,
            )

        while len(self.open_shards) >= self.max_open_files:
            _, old_shard = self.open_shards.popitem(last=False)
            old_shard.handle.close()

        relative_path = _shard_relative_path(
            floor=floor,
            day=day,
            compression=self.compression,
        )
        shard_path = self.output_dir / relative_path
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        first_write = relative_path not in self.initialized_paths
        mode = "wt" if first_write else "at"
        if self.compression == "gzip":
            handle = gzip.open(shard_path, mode, newline="", encoding="utf-8")
        else:
            handle = shard_path.open(mode, newline="", encoding="utf-8")
        writer = csv.DictWriter(handle, fieldnames=self.fieldnames, extrasaction="ignore")
        if first_write:
            writer.writeheader()
            self.initialized_paths.add(relative_path)
        self.open_shards[key] = _OpenShard(handle=handle, writer=writer)
        return writer, relative_path

    def close(self) -> None:
        for shard in self.open_shards.values():
            shard.handle.close()
        self.open_shards.clear()


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _write_manifest(
    *,
    manifest_path: Path,
    shard_counts: dict[tuple[str, str, Path], int],
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=["floor", "day", "relative_path", "row_count"],
        )
        writer.writeheader()
        for floor, day, relative_path in sorted(shard_counts):
            writer.writerow(
                {
                    "floor": floor,
                    "day": day,
                    "relative_path": str(relative_path).replace("\\", "/"),
                    "row_count": int(shard_counts[(floor, day, relative_path)]),
                }
            )


def _write_metadata(
    *,
    metadata_path: Path,
    input_csv: Path,
    output_dir: Path,
    manifest_path: Path,
    total_rows: int,
    shard_count: int,
    args: argparse.Namespace,
) -> None:
    metadata = {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "total_rows": int(total_rows),
        "shard_count": int(shard_count),
        "args": vars(args),
    }
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, default=str)


def split_cache(args: argparse.Namespace) -> int:
    input_csv = _resolve_repo_relative_path(args.input_csv)
    if input_csv is None or not input_csv.exists():
        raise FileNotFoundError(f"Input cache CSV not found: {input_csv}")

    output_dir = _resolve_repo_relative_path(args.output_dir)
    if output_dir is None:
        output_dir = _repo_root() / DEFAULT_OUTPUT_DIR
    manifest_path = (
        _resolve_repo_relative_path(args.manifest_path)
        if args.manifest_path
        else output_dir / "manifest.csv"
    )
    if manifest_path is None:
        manifest_path = output_dir / "manifest.csv"
    metadata_path = output_dir / "metadata.json"

    _prepare_output_dir(output_dir, overwrite=bool(args.overwrite))
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(2**31 - 1)

    total_rows = 0
    shard_counts: dict[tuple[str, str, Path], int] = {}
    started_at = datetime.datetime.now()
    _log(f"Splitting {input_csv} into {output_dir}")
    with input_csv.open("r", newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError(f"Input cache CSV has no header: {input_csv}")
        writer_pool = _ShardWriterPool(
            output_dir=output_dir,
            fieldnames=list(reader.fieldnames),
            max_open_files=int(args.max_open_files),
            compression=str(args.compression),
        )
        try:
            for row in reader:
                floor = _normalize_floor(row.get(args.floor_column, ""))
                day = _row_day(row, args.day_column)
                writer, relative_path = writer_pool.writer_for(floor=floor, day=day)
                writer.writerow(row)
                shard_key = (floor, day, relative_path)
                shard_counts[shard_key] = shard_counts.get(shard_key, 0) + 1
                total_rows += 1
                if args.progress_every and total_rows % int(args.progress_every) == 0:
                    _log(
                        f"Processed {total_rows:,} rows into {len(shard_counts):,} shards."
                    )
        finally:
            writer_pool.close()

    _write_manifest(manifest_path=manifest_path, shard_counts=shard_counts)
    _write_metadata(
        metadata_path=metadata_path,
        input_csv=input_csv,
        output_dir=output_dir,
        manifest_path=manifest_path,
        total_rows=total_rows,
        shard_count=len(shard_counts),
        args=args,
    )
    duration = datetime.datetime.now() - started_at
    _log(
        f"Finished splitting {total_rows:,} rows into {len(shard_counts):,} shards "
        f"in {duration}."
    )
    _log(f"Manifest saved to {manifest_path}")
    _log(f"Metadata saved to {metadata_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="SplitOfflineRequestPredictionCache",
        description=(
            "Stream a large offline request prediction cache into per-floor/per-day "
            "CSV shards without loading the full cache into memory."
        ),
    )
    parser.add_argument("--input_csv", default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest_path", default=None)
    parser.add_argument("--floor_column", default="floor")
    parser.add_argument("--day_column", default="day")
    parser.add_argument("--compression", choices=("none", "gzip"), default="none")
    parser.add_argument("--max_open_files", type=int, default=64)
    parser.add_argument("--progress_every", type=int, default=1_000_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    return split_cache(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
