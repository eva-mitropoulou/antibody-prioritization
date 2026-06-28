"""Prepare a clean SARS-CoV-2 CoV-AbDab sequence table.

Run from the project root:

    python src/data/prepare_covabdab_sarscov2.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


# Paths are resolved from this file so the script can be run from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "covabdab.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "covabdab_sarscov2_sequences.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "covabdab_cleaning_report.md"

SARS_COV_2_PATTERN = re.compile(r"sars[\s_-]*cov[\s_-]*2|sarscov2", flags=re.IGNORECASE)
CANONICAL_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

# CoV-AbDab uses placeholders such as ND for unavailable sequence values.
MISSING_MARKERS = {
    "",
    "-",
    "--",
    "na",
    "n/a",
    "nan",
    "nd",
    "n d",
    "none",
    "null",
    "not available",
    "not determined",
    "not reported",
    "not sequenced",
    "unknown",
    "unavailable",
}

COLUMN_RENAMES = {
    "Name": "antibody_name",
    "Ab or Nb": "antibody_type",
    "VHorVHH": "heavy_or_vhh_sequence",
    "VL": "light_sequence",
    "CDRH3": "cdrh3",
    "CDRL3": "cdrl3",
    "Protein + Epitope": "protein_epitope",
    "Origin": "origin",
    "Structures": "structures",
    "Sources": "sources",
    "Date Added": "date_added",
    "Last Updated": "last_updated",
}

SEQUENCE_COLUMNS = [
    "heavy_or_vhh_sequence",
    "light_sequence",
    "cdrh3",
    "cdrl3",
]

OUTPUT_COLUMNS = [
    "antibody_name",
    "antibody_type",
    "heavy_or_vhh_sequence",
    "light_sequence",
    "cdrh3",
    "cdrl3",
    "protein_epitope",
    "neutralising_label",
    "has_structure",
    "sequence_key",
    "origin",
    "structures",
    "sources",
    "date_added",
    "last_updated",
    "valid_heavy_or_vhh_sequence",
    "valid_light_sequence",
    "valid_cdrh3",
    "valid_cdrl3",
    "has_heavy_or_vhh",
    "has_light",
    "has_paired_heavy_light",
    "is_nanobody_like",
    "neutralising_conflict",
    "duplicate_count",
]


def normalize_marker(value: object) -> str:
    """Normalize short placeholder strings for missing-value checks."""
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_missing_value(value: object) -> bool:
    """Return True for pandas missing values and common CoV-AbDab placeholders."""
    if pd.isna(value):
        return True

    return normalize_marker(value) in MISSING_MARKERS


def clean_sequence(value: object) -> object:
    """Remove whitespace, uppercase sequence text, and convert placeholders to NA."""
    if is_missing_value(value):
        return pd.NA

    sequence = re.sub(r"\s+", "", str(value)).upper()
    if normalize_marker(sequence) in MISSING_MARKERS:
        return pd.NA

    return sequence if sequence else pd.NA


def has_value(series: pd.Series) -> pd.Series:
    """Check whether each value is present after applying placeholder rules."""
    return ~series.apply(is_missing_value)


def contains_sars_cov_2(series: pd.Series) -> pd.Series:
    """Find rows where a text column mentions SARS-CoV-2."""
    return series.fillna("").astype(str).str.contains(SARS_COV_2_PATTERN, regex=True)


def is_valid_amino_acid_sequence(value: object) -> bool:
    """Validate a sequence against the 20 canonical amino acids."""
    if is_missing_value(value):
        return False

    sequence = str(value)
    return bool(sequence) and set(sequence).issubset(CANONICAL_AMINO_ACIDS)


def make_sequence_key(row: pd.Series) -> object:
    """Create a key from heavy/VHH alone or heavy/VHH plus light chain."""
    heavy = row["heavy_or_vhh_sequence"]
    light = row["light_sequence"]

    if is_missing_value(heavy):
        return pd.NA

    if not is_missing_value(light):
        return f"{heavy}__{light}"

    return heavy


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Fail early with a clear message if the expected CoV-AbDab columns change."""
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise KeyError(f"Missing required column(s): {missing_text}")


def add_cleaning_columns(df: pd.DataFrame, raw_filtered: pd.DataFrame) -> pd.DataFrame:
    """Add sequence-validity, label, structure, and duplicate helper columns."""
    cleaned = df.copy()

    for column in SEQUENCE_COLUMNS:
        cleaned[column] = cleaned[column].apply(clean_sequence)

    cleaned["valid_heavy_or_vhh_sequence"] = cleaned["heavy_or_vhh_sequence"].apply(
        is_valid_amino_acid_sequence
    )
    cleaned["valid_light_sequence"] = cleaned["light_sequence"].apply(
        is_valid_amino_acid_sequence
    )
    cleaned["valid_cdrh3"] = cleaned["cdrh3"].apply(is_valid_amino_acid_sequence)
    cleaned["valid_cdrl3"] = cleaned["cdrl3"].apply(is_valid_amino_acid_sequence)

    cleaned["has_heavy_or_vhh"] = has_value(cleaned["heavy_or_vhh_sequence"])
    cleaned["has_light"] = has_value(cleaned["light_sequence"])
    cleaned["has_paired_heavy_light"] = cleaned["has_heavy_or_vhh"] & cleaned["has_light"]

    antibody_type = cleaned["antibody_type"].fillna("").astype(str)
    cleaned["is_nanobody_like"] = antibody_type.str.contains(
        r"\b(?:nb|nanobody|vhh)\b",
        case=False,
        regex=True,
    )

    cleaned["has_structure"] = has_value(cleaned["structures"])

    neutralising_sars = contains_sars_cov_2(raw_filtered["Neutralising Vs"])
    not_neutralising_sars = contains_sars_cov_2(raw_filtered["Not Neutralising Vs"])
    cleaned["neutralising_conflict"] = neutralising_sars & not_neutralising_sars

    # Positive labels take precedence when both columns mention SARS-CoV-2; the
    # conflict flag lets later scripts exclude or resolve those ambiguous rows.
    cleaned["neutralising_label"] = pd.Series(pd.NA, index=cleaned.index, dtype="Int64")
    cleaned.loc[neutralising_sars, "neutralising_label"] = 1
    cleaned.loc[not_neutralising_sars & ~neutralising_sars, "neutralising_label"] = 0

    cleaned["sequence_key"] = cleaned.apply(make_sequence_key, axis=1)

    return cleaned


def build_report(metrics: dict[str, int]) -> str:
    """Create the Markdown cleaning report."""
    lines = [
        "# CoV-AbDab SARS-CoV-2 Cleaning Report",
        "",
        f"Source file: `{RAW_PATH.relative_to(PROJECT_ROOT)}`",
        f"Output file: `{OUTPUT_PATH.relative_to(PROJECT_ROOT)}`",
        "",
        "Metrics are counted after filtering to SARS-CoV-2 binders unless noted.",
        "Rows without a heavy/VHH sequence are counted, then excluded from the",
        "final sequence-key table because no usable sequence key can be created.",
        "",
        "## Required Metrics",
        "",
        "| Metric | Count |",
        "|---|---:|",
    ]

    metric_labels = [
        ("raw_row_count", "Raw row count"),
        ("sars_cov_2_filtered_row_count", "SARS-CoV-2 filtered row count"),
        ("rows_with_heavy_or_vhh_sequence", "Rows with heavy/VHH sequence"),
        ("rows_with_light_sequence", "Rows with light sequence"),
        ("paired_heavy_light_rows", "Paired heavy-light rows"),
        ("nanobody_like_rows", "Nanobody-like rows"),
        ("neutralising_label_1", "neutralising_label = 1"),
        ("neutralising_label_0", "neutralising_label = 0"),
        ("neutralising_label_missing", "neutralising_label missing"),
        ("neutralisation_conflicts", "Neutralisation conflicts"),
        ("invalid_heavy_or_vhh_sequences", "Invalid heavy/VHH sequences"),
        ("invalid_light_sequences", "Invalid light sequences"),
        ("duplicate_sequence_keys", "Duplicate sequence keys"),
        ("rows_removed_by_missing_sequence_key", "Rows removed by missing sequence key"),
        ("rows_removed_by_deduplication", "Rows removed by sequence-key deduplication"),
        ("final_deduplicated_row_count", "Final deduplicated row count"),
    ]

    for key, label in metric_labels:
        lines.append(f"| {label} | {metrics[key]} |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `neutralising_label = 1` means `Neutralising Vs` mentions SARS-CoV-2.",
            "- `neutralising_label = 0` means `Not Neutralising Vs` mentions SARS-CoV-2 and the positive column does not.",
            "- `neutralising_conflict = true` marks rows where both neutralising columns mention SARS-CoV-2.",
            "- Sequence validation uses only canonical amino acids: `ACDEFGHIKLMNPQRSTVWY`.",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    """Run the CoV-AbDab SARS-CoV-2 cleaning workflow."""
    raw = pd.read_csv(RAW_PATH)
    require_columns(raw, list(COLUMN_RENAMES) + ["Binds to", "Neutralising Vs", "Not Neutralising Vs"])

    sars_cov_2_binder_mask = contains_sars_cov_2(raw["Binds to"])
    raw_filtered = raw.loc[sars_cov_2_binder_mask].copy()

    cleaned = raw_filtered.rename(columns=COLUMN_RENAMES)[list(COLUMN_RENAMES.values())]
    cleaned = add_cleaning_columns(cleaned, raw_filtered)

    duplicate_source = cleaned.loc[cleaned["has_heavy_or_vhh"]].copy()
    duplicate_counts = duplicate_source["sequence_key"].value_counts(dropna=False)
    duplicate_source["duplicate_count"] = duplicate_source["sequence_key"].map(duplicate_counts)

    final = duplicate_source.drop_duplicates(subset="sequence_key", keep="first").copy()
    final = final[OUTPUT_COLUMNS]

    metrics = {
        "raw_row_count": int(len(raw)),
        "sars_cov_2_filtered_row_count": int(len(cleaned)),
        "rows_with_heavy_or_vhh_sequence": int(cleaned["has_heavy_or_vhh"].sum()),
        "rows_with_light_sequence": int(cleaned["has_light"].sum()),
        "paired_heavy_light_rows": int(cleaned["has_paired_heavy_light"].sum()),
        "nanobody_like_rows": int(cleaned["is_nanobody_like"].sum()),
        "neutralising_label_1": int((cleaned["neutralising_label"] == 1).sum()),
        "neutralising_label_0": int((cleaned["neutralising_label"] == 0).sum()),
        "neutralising_label_missing": int(cleaned["neutralising_label"].isna().sum()),
        "neutralisation_conflicts": int(cleaned["neutralising_conflict"].sum()),
        "invalid_heavy_or_vhh_sequences": int(
            (cleaned["has_heavy_or_vhh"] & ~cleaned["valid_heavy_or_vhh_sequence"]).sum()
        ),
        "invalid_light_sequences": int(
            (cleaned["has_light"] & ~cleaned["valid_light_sequence"]).sum()
        ),
        "duplicate_sequence_keys": int((duplicate_counts > 1).sum()),
        "rows_removed_by_missing_sequence_key": int((~cleaned["has_heavy_or_vhh"]).sum()),
        "rows_removed_by_deduplication": int(len(duplicate_source) - len(final)),
        "final_deduplicated_row_count": int(len(final)),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    final.to_csv(OUTPUT_PATH, index=False)
    REPORT_PATH.write_text(build_report(metrics), encoding="utf-8")

    print(f"Loaded raw rows: {metrics['raw_row_count']}")
    print(f"SARS-CoV-2 binder rows: {metrics['sars_cov_2_filtered_row_count']}")
    print(f"Rows with heavy/VHH sequence: {metrics['rows_with_heavy_or_vhh_sequence']}")
    print(f"Rows with light sequence: {metrics['rows_with_light_sequence']}")
    print(f"Paired heavy-light rows: {metrics['paired_heavy_light_rows']}")
    print(f"Nanobody-like rows: {metrics['nanobody_like_rows']}")
    print(f"neutralising_label = 1: {metrics['neutralising_label_1']}")
    print(f"neutralising_label = 0: {metrics['neutralising_label_0']}")
    print(f"neutralising_label missing: {metrics['neutralising_label_missing']}")
    print(f"Neutralisation conflicts: {metrics['neutralisation_conflicts']}")
    print(f"Invalid heavy/VHH sequences: {metrics['invalid_heavy_or_vhh_sequences']}")
    print(f"Invalid light sequences: {metrics['invalid_light_sequences']}")
    print(f"Duplicate sequence keys: {metrics['duplicate_sequence_keys']}")
    print(f"Final deduplicated rows: {metrics['final_deduplicated_row_count']}")
    print(f"Saved clean table to: {OUTPUT_PATH}")
    print(f"Saved report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
