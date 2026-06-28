"""Shared aggregate-only helpers for the final antibody workflow."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MISSING_TEXT_VALUES = {"", "na", "nan", "none", "null", "nd", "n/a", "unknown"}
SEQUENCE_VALUE_COLUMNS = {
    "sequence_a",
    "sequence_b",
    "sequence_a_raw",
    "sequence_pair_text",
    "sequence_heavy_only",
    "heavy_sequence",
    "light_sequence",
    "group_feature_cdr3",
    "group_feature_b_cdr3",
    "existing_cdrh3",
    "existing_cdrl3",
    "cdrh1_seq",
    "cdrh2_seq",
    "cdrh3_seq",
    "cdrl1_seq",
    "cdrl2_seq",
    "cdrl3_seq",
    "marked_heavy_text",
    "marked_light_text",
    "whole_pair_kmer_text",
    "all_cdr_kmer_text",
    "cdrh3_cdrl3_kmer_text",
    "cdrh3",
    "cdrh3_prefix",
    "cdrh3_suffix",
}


def relpath(path: Path) -> str:
    """Return a project-relative path when possible."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def read_csv_text(path: Path) -> pd.DataFrame:
    """Read CSV fields as text while preserving blank values."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def read_csv_columns(path: Path) -> list[str]:
    """Read only CSV column names."""
    return list(pd.read_csv(path, nrows=0).columns)


def load_json(path: Path) -> dict[str, Any] | None:
    """Load JSON if available."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write pretty JSON with stable keys."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    """Write UTF-8 text, creating parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def normalized_text(values: pd.Series) -> pd.Series:
    """Normalize free-text values without printing their contents."""
    return values.fillna("").astype(str).str.strip()


def is_missing_text(value: Any) -> bool:
    """Return true for project-level missing text tokens."""
    return str(value or "").strip().lower() in MISSING_TEXT_VALUES


def optional_column(data: pd.DataFrame, column: str) -> pd.Series:
    """Return a text column or aligned blanks."""
    if column in data.columns:
        return data[column]
    return pd.Series([""] * len(data), index=data.index)


def first_available_column(data: pd.DataFrame, columns: list[str]) -> tuple[pd.Series, str | None]:
    """Return the first available column from a candidate list."""
    for column in columns:
        if column in data.columns:
            return data[column], column
    return pd.Series([""] * len(data), index=data.index), None


def label_series(data: pd.DataFrame) -> pd.Series:
    """Return binary labels from strict or broader neutral tables."""
    raw, _ = first_available_column(data, ["label", "neutralising_label", "extra_col_07"])
    text = normalized_text(raw).replace({"": np.nan})
    return pd.to_numeric(text, errors="coerce")


def bool_from_columns(data: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Parse boolean-like values from the first useful values across columns."""
    combined = pd.Series([np.nan] * len(data), index=data.index, dtype=object)
    for column in columns:
        if column not in data.columns:
            continue
        raw = data[column]
        text = normalized_text(raw).str.lower()
        mapped = text.map(
            {
                "true": True,
                "false": False,
                "yes": True,
                "no": False,
                "1": True,
                "0": False,
            }
        )
        numeric = pd.to_numeric(raw, errors="coerce")
        numeric_bool = numeric.map(lambda value: bool(value) if pd.notna(value) else np.nan)
        parsed = mapped.where(mapped.notna(), numeric_bool)
        fill_mask = combined.isna() & parsed.notna()
        combined.loc[fill_mask] = parsed.loc[fill_mask]
    return combined.fillna(False).astype(bool)


def normalize_sequence_for_features(value: Any) -> str:
    """Compact an existing sequence-like value for internal feature building only."""
    text = re.sub(r"\s+", "", str(value or "")).upper()
    return "" if text.lower() in MISSING_TEXT_VALUES else text


def compact_model_text(value: Any) -> str:
    """Create compact character-kmer text from an existing model-input field."""
    text = str(value or "")
    text = text.replace("[SEP]", "|")
    return re.sub(r"\s+", "", text).upper()


def compact_pair_from_columns(data: pd.DataFrame) -> pd.Series:
    """Build compact heavy/light pair text from existing sequence columns."""
    heavy = optional_column(data, "sequence_a").map(normalize_sequence_for_features)
    light = optional_column(data, "sequence_b").map(normalize_sequence_for_features)
    return pd.Series(
        [f"{h}|{l}" if l else h for h, l in zip(heavy, light)],
        index=data.index,
    )


def target_region_group_value(value: Any) -> str:
    """Map target-region metadata into stable broad groups."""
    text = str(value or "").strip().lower()
    if text in MISSING_TEXT_VALUES:
        return "unknown"
    if "rbd" in text or "receptor binding" in text:
        return "RBD"
    if "ntd" in text or "n-terminal" in text or "n terminal" in text:
        return "NTD"
    if text in {"s", "spike"} or "spike" in text or "s protein" in text:
        return "Spike/S"
    return "other"


def target_region_group_series(data: pd.DataFrame) -> pd.Series:
    """Return broad target-region groups."""
    values, _ = first_available_column(data, ["target_region_group", "metadata_target_region"])
    return normalized_text(values).map(target_region_group_value)


def light_status_series(data: pd.DataFrame) -> pd.Series:
    """Return paired versus light-missing/single-chain status."""
    if "paired_light_status" in data.columns:
        raw = normalized_text(data["paired_light_status"])
        return raw.where(raw.ne(""), "unknown")
    if "missing_light_flag" in data.columns:
        missing = normalized_text(data["missing_light_flag"]).str.lower().isin({"true", "1", "yes"})
    elif "has_light_bool" in data.columns:
        missing = ~bool_from_columns(data, ["has_light_bool"])
    else:
        light = optional_column(data, "sequence_b").map(normalize_sequence_for_features)
        missing = light.eq("")
    return pd.Series(
        np.where(missing, "light_missing_or_single_chain", "paired"),
        index=data.index,
    )


def structure_available_series(data: pd.DataFrame) -> pd.Series:
    """Return structure metadata availability from boolean or text fields."""
    bool_part = bool_from_columns(data, ["has_structure", "extra_col_08", "extra_col_16"])
    text_values, _ = first_available_column(
        data,
        ["metadata_structure", "structure_id", "pdb_id", "extra_col_11"],
    )
    text_part = normalized_text(text_values).map(lambda value: not is_missing_text(value))
    return bool_part | text_part.astype(bool)


def value_counts_dict(values: pd.Series, limit: int | None = None) -> dict[str, int]:
    """Return JSON-safe value counts."""
    counts = values.fillna("missing").astype(str).value_counts(dropna=False)
    if limit is not None:
        counts = counts.head(limit)
    return {str(key): int(value) for key, value in counts.items()}


def label_counts_dict(labels: pd.Series) -> dict[str, int]:
    """Return binary label counts with stable keys."""
    numeric = pd.to_numeric(labels, errors="coerce")
    counts = numeric.value_counts().sort_index()
    return {"0": int(counts.get(0, 0)), "1": int(counts.get(1, 0))}


def missing_counts_dict(data: pd.DataFrame) -> dict[str, int]:
    """Return missing counts per column, treating blank text as missing."""
    counts: dict[str, int] = {}
    for column in data.columns:
        values = data[column]
        missing = values.isna() | normalized_text(values).str.lower().isin(MISSING_TEXT_VALUES)
        counts[column] = int(missing.sum())
    return counts


def numeric_summary(values: pd.Series) -> dict[str, float | None]:
    """Return aggregate numeric summary."""
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == 0:
        return {"min": None, "mean": None, "median": None, "max": None}
    return {
        "min": float(numeric.min()),
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "max": float(numeric.max()),
    }


def safe_output_columns(data: pd.DataFrame) -> list[str]:
    """Return columns safe for aggregate/record-index outputs."""
    return [column for column in data.columns if column not in SEQUENCE_VALUE_COLUMNS]


def markdown_counts_table(title: str, counts: dict[str, int]) -> list[str]:
    """Build a compact Markdown count table."""
    lines = [f"### {title}", "", "| Value | Count |", "|---|---:|"]
    for key, value in counts.items():
        lines.append(f"| {key} | {value} |")
    lines.append("")
    return lines
