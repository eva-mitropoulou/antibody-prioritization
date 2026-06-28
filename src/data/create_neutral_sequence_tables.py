"""Create neutral CSV copies for generic sequence classification.

The script performs column neutralization, sequence-key construction, metadata
carry-forward, and neutral warning generation for existing records.

Run from the project root:

    python src/data/create_neutral_sequence_tables.py
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_SOURCE_PATH = PROJECT_ROOT / "data" / "raw" / "source_sequences.csv"
PREPARED_PATH = PROJECT_ROOT / "data" / "processed" / "prepared_sequences.csv"
ML_PATH = PROJECT_ROOT / "data" / "processed" / "sequence_classification_ml.csv"

NEUTRAL_SOURCE_PATH = (
    PROJECT_ROOT / "data" / "processed" / "neutral_source_sequences.csv"
)
NEUTRAL_PREPARED_PATH = (
    PROJECT_ROOT / "data" / "processed" / "neutral_prepared_sequences.csv"
)
NEUTRAL_ML_PATH = (
    PROJECT_ROOT / "data" / "processed" / "neutral_sequence_classification_ml.csv"
)

SOURCE_POSITIONAL_NAMES = {
    0: "sample_name",
    1: "sample_type",
    2: "metadata_positive_text",
    3: "metadata_negative_text",
    4: "label_positive_text",
    5: "label_negative_text",
    6: "metadata_target_region",
    7: "metadata_origin",
    8: "sequence_a",
    9: "sequence_b",
    10: "group_feature_v",
    11: "group_feature_j",
    12: "group_feature_b_v",
    13: "group_feature_b_j",
    14: "group_feature_cdr3",
    15: "group_feature_b_cdr3",
    16: "metadata_structure",
    17: "metadata_model",
    18: "metadata_sources",
    19: "metadata_date_added",
    20: "metadata_last_updated",
}

ALIAS_NAMES = {
    "sequence_heavy_only": "sequence_a",
    "sequence_pair_text": "sequence_pair_text",
    "label": "label",
    "heavy_or_vhh_sequence": "sequence_a_raw",
    "light_sequence": "sequence_b",
    "cdrh3": "group_feature_cdr3",
    "cdrl3": "group_feature_b_cdr3",
    "protein_epitope": "metadata_target_region",
    "antibody_name": "sample_name",
    "antibody_type": "sample_type",
}

NEUTRAL_COLUMN_ORDER = [
    "sample_name",
    "sample_type",
    "label",
    "label_positive_text",
    "label_negative_text",
    "metadata_positive_text",
    "metadata_negative_text",
    "sequence_a",
    "sequence_b",
    "sequence_a_raw",
    "sequence_pair_text",
    "sequence_key",
    "group_feature_v",
    "group_feature_j",
    "group_feature_b_v",
    "group_feature_b_j",
    "group_feature_cdr3",
    "group_feature_b_cdr3",
    "metadata_target_region",
    "metadata_origin",
    "metadata_structure",
    "metadata_model",
    "metadata_sources",
    "metadata_date_added",
    "metadata_last_updated",
]

SOURCE_MERGE_COLUMNS = [
    "group_feature_v",
    "group_feature_j",
    "group_feature_b_v",
    "group_feature_b_j",
    "group_feature_cdr3",
    "group_feature_b_cdr3",
    "metadata_target_region",
    "metadata_origin",
]


def warn_once(messages: list[str], message: str) -> None:
    """Collect a warning without printing duplicates."""
    if message not in messages:
        messages.append(message)


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV as text while preserving blank fields as blanks."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def is_blank(series: pd.Series) -> pd.Series:
    """Return a mask for empty text-like values."""
    return series.fillna("").astype(str).str.strip().eq("")


def combine_prefer_existing(existing: pd.Series, incoming: pd.Series) -> pd.Series:
    """Fill blank existing values from incoming values."""
    combined = existing.copy()
    incoming = incoming.reindex(existing.index)
    fill_mask = is_blank(combined) & ~is_blank(incoming)
    combined.loc[fill_mask] = incoming.loc[fill_mask]
    return combined


def add_or_fill_column(output: pd.DataFrame, name: str, values: pd.Series) -> None:
    """Add a neutral column, or fill blanks if it already exists."""
    aligned = values.reset_index(drop=True)
    if name in output.columns:
        output[name] = combine_prefer_existing(output[name], aligned)
    else:
        output[name] = aligned


def normalize_sequence(values: pd.Series) -> pd.Series:
    """Normalize sequence text for stable joining and matching."""
    return (
        values.fillna("")
        .astype(str)
        .str.replace(r"\s+", "", regex=True)
        .str.upper()
    )


def ensure_sequence_key(output: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
    """Ensure sequence columns and the neutral sequence key are present."""
    if "sequence_a" not in output.columns and "sequence_a_raw" in output.columns:
        output["sequence_a"] = output["sequence_a_raw"]

    if "sequence_a" not in output.columns:
        output["sequence_a"] = ""
        warn_once(warnings, "WARNING: sequence_a unavailable")

    if "sequence_b" not in output.columns:
        output["sequence_b"] = ""

    output["sequence_key"] = (
        normalize_sequence(output["sequence_a"])
        + "||"
        + normalize_sequence(output["sequence_b"])
    )
    return output


def reorder_columns(output: pd.DataFrame) -> pd.DataFrame:
    """Place neutral columns first and any remaining neutral extras last."""
    ordered = [column for column in NEUTRAL_COLUMN_ORDER if column in output.columns]
    remaining = [column for column in output.columns if column not in ordered]
    return output[ordered + remaining]


def neutralize_source_csv(path: Path, warnings: list[str]) -> pd.DataFrame:
    """Rename source columns by position only."""
    data = read_csv(path)
    renamed_columns = []

    for position, _ in enumerate(data.columns):
        renamed_columns.append(
            SOURCE_POSITIONAL_NAMES.get(position, f"extra_col_{position:02d}")
        )

    for position in SOURCE_POSITIONAL_NAMES:
        if position >= len(data.columns):
            warn_once(
                warnings,
                f"WARNING: missing expected positional column {position}",
            )

    data = data.copy()
    data.columns = renamed_columns
    data = ensure_sequence_key(data, warnings)
    return reorder_columns(data)


def neutralize_alias_csv(path: Path, warnings: list[str]) -> pd.DataFrame:
    """Create a neutral table from a CSV with known neutral alias candidates."""
    data = read_csv(path)
    output = pd.DataFrame(index=data.index)

    for position, source_column in enumerate(data.columns):
        neutral_name = ALIAS_NAMES.get(str(source_column))
        if neutral_name is None:
            neutral_name = f"extra_col_{position:02d}"
        add_or_fill_column(output, neutral_name, data[source_column])

    output = ensure_sequence_key(output, warnings)
    return reorder_columns(output)


def first_non_blank(values: Iterable[object]) -> str:
    """Return the first non-blank value from a grouped column."""
    for value in values:
        text = "" if pd.isna(value) else str(value)
        if text.strip():
            return text
    return ""


def source_metadata_by_key(source: pd.DataFrame) -> pd.DataFrame:
    """Build one source metadata row per neutral sequence key."""
    if "sequence_key" not in source.columns:
        return pd.DataFrame(columns=["sequence_key"])

    available_columns = [
        column for column in SOURCE_MERGE_COLUMNS if column in source.columns
    ]
    if not available_columns:
        return pd.DataFrame(columns=["sequence_key"])

    subset = source[["sequence_key", *available_columns]].copy()
    subset = subset[~is_blank(subset["sequence_key"])].copy()
    if subset.empty:
        return pd.DataFrame(columns=["sequence_key", *available_columns])

    aggregations = {column: first_non_blank for column in available_columns}
    return subset.groupby("sequence_key", as_index=False).agg(aggregations)


def merge_source_metadata(
    ml_table: pd.DataFrame,
    source_table: pd.DataFrame,
    warnings: list[str],
) -> pd.DataFrame:
    """Merge available neutral group metadata from the source table."""
    source_metadata = source_metadata_by_key(source_table)
    if source_metadata.empty:
        merged = ml_table.copy()
    else:
        merged = ml_table.merge(
            source_metadata,
            how="left",
            on="sequence_key",
            suffixes=("", "__source"),
        )

    for column in SOURCE_MERGE_COLUMNS:
        source_column = f"{column}__source"
        if source_column in merged.columns:
            merged[column] = combine_prefer_existing(
                merged[column],
                merged[source_column],
            )
            merged = merged.drop(columns=[source_column])

        if column not in merged.columns:
            warn_once(warnings, f"WARNING: {column} unavailable")
            continue

        if bool(is_blank(merged[column]).all()):
            warn_once(warnings, f"WARNING: {column} unavailable")

    return reorder_columns(merged)


def write_table(path: Path, table: pd.DataFrame) -> None:
    """Write one neutral table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    print(
        f"Wrote {path.relative_to(PROJECT_ROOT)} "
        f"rows={len(table)} columns={len(table.columns)}"
    )


def main() -> None:
    """Create neutral source, prepared, and model-input CSVs."""
    warnings: list[str] = []

    neutral_source = neutralize_source_csv(RAW_SOURCE_PATH, warnings)
    neutral_prepared = neutralize_alias_csv(PREPARED_PATH, warnings)
    neutral_ml = neutralize_alias_csv(ML_PATH, warnings)
    neutral_ml = merge_source_metadata(neutral_ml, neutral_source, warnings)

    write_table(NEUTRAL_SOURCE_PATH, neutral_source)
    write_table(NEUTRAL_PREPARED_PATH, neutral_prepared)
    write_table(NEUTRAL_ML_PATH, neutral_ml)

    for message in warnings:
        print(message)


if __name__ == "__main__":
    main()
