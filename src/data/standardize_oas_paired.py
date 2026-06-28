"""Standardize local paired OAS files for unknown-target background retrieval."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "oas"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "oas"
OUTPUT_PATH = PROCESSED_DIR / "oas_paired_standardized.csv"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics" / "oas_standardization_metrics.json"

MAX_OUTPUT_ROWS = 50_000
CHUNK_SIZE = 25_000
MIN_HEAVY_LEN = 80
MAX_HEAVY_LEN = 180
MIN_LIGHT_LEN = 60
MAX_LIGHT_LEN = 140

HEAVY_CANDIDATES = [
    "sequence_alignment_aa_heavy",
    "v_sequence_alignment_aa_heavy",
    "sequence_aa_heavy",
    "heavy_sequence",
]
LIGHT_CANDIDATES = [
    "sequence_alignment_aa_light",
    "v_sequence_alignment_aa_light",
    "sequence_aa_light",
    "light_sequence",
]
CANONICAL_PATTERN = re.compile(r"[^ACDEFGHIKLMNPQRSTVWY]")


def relpath(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def normalize_column_name(column: str) -> str:
    """Normalize column names for flexible matching."""
    return column.strip().lower()


def find_column(columns: list[str], candidates: list[str]) -> str | None:
    """Find a candidate column by normalized name."""
    normalized = {normalize_column_name(column): column for column in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def detect_header(path: Path) -> tuple[int, list[str], str | None, str | None]:
    """Detect OAS CSV header row and heavy/light sequence columns."""
    for skiprows in [0, 1, 2]:
        try:
            columns = list(
                pd.read_csv(
                    path,
                    compression="gzip",
                    skiprows=skiprows,
                    nrows=0,
                ).columns
            )
        except Exception:
            continue
        heavy = find_column(columns, HEAVY_CANDIDATES)
        light = find_column(columns, LIGHT_CANDIDATES)
        if heavy and light:
            return skiprows, columns, heavy, light
    return 0, [], None, None


def clean_sequence(values: pd.Series) -> pd.Series:
    """Uppercase, remove whitespace, and retain canonical amino-acid letters."""
    cleaned = values.fillna("").astype(str).str.replace(r"\s+", "", regex=True).str.upper()
    return cleaned.map(lambda value: CANONICAL_PATTERN.sub("", value))


def standardize_file(path: Path, remaining_rows: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Standardize one compressed OAS CSV."""
    skiprows, columns, heavy_col, light_col = detect_header(path)
    metrics: dict[str, Any] = {
        "file_path": relpath(path),
        "header_skiprows": skiprows,
        "column_count": len(columns),
        "heavy_column": heavy_col,
        "light_column": light_col,
        "input_rows_scanned": 0,
        "rows_after_sequence_cleaning": 0,
        "rows_after_length_filter": 0,
        "rows_emitted_before_global_dedup": 0,
        "status": "available" if heavy_col and light_col else "missing_required_columns",
    }
    if not heavy_col or not light_col or remaining_rows <= 0:
        return pd.DataFrame(), metrics

    chunks = []
    for chunk in pd.read_csv(
        path,
        compression="gzip",
        skiprows=skiprows,
        dtype=str,
        keep_default_na=False,
        usecols=[heavy_col, light_col],
        chunksize=CHUNK_SIZE,
        on_bad_lines="skip",
    ):
        metrics["input_rows_scanned"] += int(len(chunk))
        heavy = clean_sequence(chunk[heavy_col])
        light = clean_sequence(chunk[light_col])
        usable = pd.DataFrame(
            {
                "heavy_sequence": heavy,
                "light_sequence": light,
            }
        )
        nonempty = usable["heavy_sequence"].ne("") & usable["light_sequence"].ne("")
        metrics["rows_after_sequence_cleaning"] += int(nonempty.sum())
        lengths = (
            usable["heavy_sequence"].str.len().between(MIN_HEAVY_LEN, MAX_HEAVY_LEN)
            & usable["light_sequence"].str.len().between(MIN_LIGHT_LEN, MAX_LIGHT_LEN)
        )
        usable = usable.loc[nonempty & lengths].copy()
        metrics["rows_after_length_filter"] += int(len(usable))
        if usable.empty:
            continue
        usable["sequence_pair_text"] = usable["heavy_sequence"] + "[SEP]" + usable["light_sequence"]
        usable["source_file"] = path.name
        usable["background_source"] = "OAS_paired_unknown_target_background"
        chunks.append(usable)
        emitted = sum(len(part) for part in chunks)
        if emitted >= remaining_rows:
            break

    if not chunks:
        return pd.DataFrame(), metrics
    output = pd.concat(chunks, ignore_index=True).head(remaining_rows)
    metrics["rows_emitted_before_global_dedup"] = int(len(output))
    return output, metrics


def build_standardized_table() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Standardize all available OAS files up to the global cap."""
    files = sorted(RAW_DIR.glob("*.csv.gz"))
    parts = []
    file_metrics = []
    remaining = MAX_OUTPUT_ROWS
    for path in files:
        part, metrics = standardize_file(path, remaining)
        file_metrics.append(metrics)
        if not part.empty:
            parts.append(part)
            remaining -= len(part)
        if remaining <= 0:
            break

    if parts:
        table = pd.concat(parts, ignore_index=True)
        before_dedup = int(len(table))
        table = table.drop_duplicates(["heavy_sequence", "light_sequence"]).head(MAX_OUTPUT_ROWS)
    else:
        table = pd.DataFrame(
            columns=[
                "heavy_sequence",
                "light_sequence",
                "sequence_pair_text",
                "source_file",
                "background_source",
            ]
        )
        before_dedup = 0

    metrics = {
        "status": "available" if len(table) else "empty",
        "raw_dir": relpath(RAW_DIR),
        "output_path": relpath(OUTPUT_PATH),
        "raw_file_count": len(files),
        "max_output_rows": MAX_OUTPUT_ROWS,
        "rows_before_global_dedup": before_dedup,
        "standardized_row_count": int(len(table)),
        "duplicate_pair_count_removed": int(before_dedup - len(table)),
        "file_metrics": file_metrics,
        "output_columns": list(table.columns),
    }
    return table, metrics


def write_metrics(payload: dict[str, Any]) -> None:
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "reports" / "metrics").mkdir(parents=True, exist_ok=True)
    table, metrics = build_standardized_table()
    table.to_csv(OUTPUT_PATH, index=False)
    write_metrics(metrics)
    mappings = [
        {
            "file_path": item["file_path"],
            "heavy_column": item["heavy_column"],
            "light_column": item["light_column"],
            "status": item["status"],
            "rows_after_length_filter": item["rows_after_length_filter"],
        }
        for item in metrics["file_metrics"]
    ]
    print(
        "standardized_oas="
        f"{metrics['standardized_row_count']}; raw_files={metrics['raw_file_count']}; "
        f"output_path={metrics['output_path']}; column_mappings={mappings}",
        flush=True,
    )
    return 0 if metrics["standardized_row_count"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
