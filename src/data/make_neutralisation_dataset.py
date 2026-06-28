"""Create a strict ML-ready SARS-CoV-2 neutralisation dataset.

This script performs dataset creation and light EDA only. It does not train a
model.

Run from the project root:

    python src/data/make_neutralisation_dataset.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


# Matplotlib needs a writable cache directory in this environment.
MPLCONFIGDIR = Path("/tmp") / "antibody_prioritization_mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "covabdab_sarscov2_sequences.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "covabdab_neutralisation_ml.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "neutralisation_dataset_report.md"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"


REQUIRED_COLUMNS = [
    "antibody_name",
    "antibody_type",
    "heavy_or_vhh_sequence",
    "light_sequence",
    "cdrh3",
    "cdrl3",
    "protein_epitope",
    "neutralising_label",
    "has_heavy_or_vhh",
    "valid_heavy_or_vhh_sequence",
    "valid_light_sequence",
    "has_light",
    "is_nanobody_like",
    "has_structure",
    "neutralising_conflict",
]

OUTPUT_COLUMNS = [
    "antibody_name",
    "antibody_type",
    "sequence_heavy_only",
    "sequence_pair_text",
    "label",
    "heavy_or_vhh_sequence",
    "light_sequence",
    "cdrh3",
    "cdrl3",
    "protein_epitope",
    "heavy_length",
    "light_length",
    "cdrh3_length",
    "cdrl3_length",
    "has_light",
    "is_nanobody_like",
    "has_structure",
    "targets_rbd",
    "targets_spike",
    "targets_ntd",
    "sequence_key",
    "origin",
    "sources",
    "date_added",
    "last_updated",
]


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Fail early if the prepared CoV-AbDab table does not have expected columns."""
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise KeyError(f"Missing required column(s): {missing_text}")


def as_bool(series: pd.Series) -> pd.Series:
    """Convert CSV-loaded booleans or boolean-like strings into bool values."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    normalized = series.fillna("").astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y", "t"})


def has_sequence(series: pd.Series) -> pd.Series:
    """Check for sequence values that are not missing after CSV loading."""
    return series.notna() & series.astype(str).str.len().gt(0)


def sequence_length(series: pd.Series) -> pd.Series:
    """Return sequence lengths while preserving missing values as NA."""
    return series.where(has_sequence(series)).str.len().astype("Int64")


def make_pair_text(row: pd.Series) -> str:
    """Join heavy and light chains for paired-chain model inputs."""
    heavy = row["heavy_or_vhh_sequence"]
    light = row["light_sequence"]

    if pd.notna(light) and str(light):
        return f"{heavy}[SEP]{light}"

    return str(heavy)


def add_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add model input strings, labels, length features, and target flags."""
    ml = df.copy()

    ml["sequence_heavy_only"] = ml["heavy_or_vhh_sequence"]
    ml["sequence_pair_text"] = ml.apply(make_pair_text, axis=1)
    ml["label"] = ml["neutralising_label"].astype(int)

    ml["heavy_length"] = sequence_length(ml["heavy_or_vhh_sequence"])
    ml["light_length"] = sequence_length(ml["light_sequence"])
    ml["cdrh3_length"] = sequence_length(ml["cdrh3"])
    ml["cdrl3_length"] = sequence_length(ml["cdrl3"])

    epitope = ml["protein_epitope"].fillna("").astype(str)
    ml["targets_rbd"] = epitope.str.contains(r"\brbd\b", case=False, regex=True)
    ml["targets_spike"] = epitope.str.contains(
        r"\bspike\b|(?:^|[;\s,/+])s(?:$|[;\s,/+])",
        case=False,
        regex=True,
    )
    ml["targets_ntd"] = epitope.str.contains(r"\bntd\b", case=False, regex=True)

    return ml


def summarize_numeric(series: pd.Series) -> dict[str, float]:
    """Create a compact numeric summary for report tables."""
    summary = series.dropna().describe(percentiles=[0.25, 0.5, 0.75])
    keys = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    return {key: float(summary.get(key, 0.0)) for key in keys}


def format_summary_table(title: str, summary: dict[str, float]) -> list[str]:
    """Format a length summary as a small Markdown table."""
    lines = [
        f"### {title}",
        "",
        "| Statistic | Value |",
        "|---|---:|",
    ]

    for key, value in summary.items():
        if key == "count":
            formatted_value = f"{int(value)}"
        else:
            formatted_value = f"{value:.2f}"
        lines.append(f"| {key} | {formatted_value} |")

    lines.append("")
    return lines


def make_figures(ml: pd.DataFrame) -> None:
    """Create simple EDA figures for the strict neutralisation dataset."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    label_counts = ml["label"].value_counts().sort_index()
    label_names = ["Non-neutralising", "Neutralising"]

    plt.figure(figsize=(6, 4))
    sns.barplot(x=label_names, y=[label_counts.get(0, 0), label_counts.get(1, 0)])
    plt.xlabel("Label")
    plt.ylabel("Rows")
    plt.title("Neutralisation Label Counts")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "label_counts.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.histplot(data=ml, x="heavy_length", bins=40)
    plt.xlabel("Heavy/VHH sequence length")
    plt.ylabel("Rows")
    plt.title("Heavy/VHH Length Distribution")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "heavy_length_distribution.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.histplot(data=ml, x="cdrh3_length", bins=30)
    plt.xlabel("CDRH3 length")
    plt.ylabel("Rows")
    plt.title("CDRH3 Length Distribution")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "cdrh3_length_distribution.png", dpi=200)
    plt.close()

    antibody_type_counts = ml["antibody_type"].fillna("Unknown").value_counts()
    plt.figure(figsize=(6, 4))
    sns.barplot(x=antibody_type_counts.index, y=antibody_type_counts.values)
    plt.xlabel("Antibody type")
    plt.ylabel("Rows")
    plt.title("Antibody Type Counts")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "antibody_type_counts.png", dpi=200)
    plt.close()


def build_report(metrics: dict[str, object], summaries: dict[str, dict[str, float]]) -> str:
    """Build the neutralisation dataset Markdown report."""
    lines = [
        "# SARS-CoV-2 Neutralisation ML Dataset Report",
        "",
        f"Input file: `{INPUT_PATH.relative_to(PROJECT_ROOT)}`",
        f"Output file: `{OUTPUT_PATH.relative_to(PROJECT_ROOT)}`",
        "",
        "No machine learning is trained in this step.",
        "",
        "## Filtering Summary",
        "",
        "| Metric | Count |",
        "|---|---:|",
        f"| Input row count | {metrics['input_row_count']} |",
        f"| Rows removed because label missing | {metrics['removed_label_missing']} |",
        f"| Rows removed because neutralisation_conflict true | {metrics['removed_conflict']} |",
        f"| Rows removed because missing/invalid heavy sequence | {metrics['removed_invalid_heavy']} |",
        f"| Rows removed because present light sequence is invalid | {metrics['removed_invalid_light']} |",
        f"| Final ML row count | {metrics['final_ml_row_count']} |",
        "",
        "## Label Balance",
        "",
        "| Label | Count | Percentage |",
        "|---|---:|---:|",
        (
            f"| 0: non-neutralising | {metrics['label_0_count']} | "
            f"{metrics['label_0_percent']:.2f}% |"
        ),
        (
            f"| 1: neutralising | {metrics['label_1_count']} | "
            f"{metrics['label_1_percent']:.2f}% |"
        ),
        "",
        "## Dataset Composition",
        "",
        "| Metric | Count |",
        "|---|---:|",
        f"| Paired heavy-light count | {metrics['paired_heavy_light_count']} |",
        f"| Nanobody-like count | {metrics['nanobody_like_count']} |",
        f"| Rows with structure | {metrics['rows_with_structure']} |",
        f"| Targets RBD | {metrics['targets_rbd_count']} |",
        f"| Targets Spike | {metrics['targets_spike_count']} |",
        f"| Targets NTD | {metrics['targets_ntd_count']} |",
        "",
        "## Length Summaries",
        "",
    ]

    for title, summary in summaries.items():
        lines.extend(format_summary_table(title, summary))

    lines.extend(
        [
            "## Figures",
            "",
            "- `reports/figures/label_counts.png`",
            "- `reports/figures/heavy_length_distribution.png`",
            "- `reports/figures/cdrh3_length_distribution.png`",
            "- `reports/figures/antibody_type_counts.png`",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    """Create the strict supervised neutralisation dataset and EDA outputs."""
    data = pd.read_csv(INPUT_PATH)
    require_columns(data, REQUIRED_COLUMNS)

    data["has_heavy_or_vhh"] = as_bool(data["has_heavy_or_vhh"])
    data["valid_heavy_or_vhh_sequence"] = as_bool(data["valid_heavy_or_vhh_sequence"])
    data["has_light"] = as_bool(data["has_light"])
    data["valid_light_sequence"] = as_bool(data["valid_light_sequence"])
    data["is_nanobody_like"] = as_bool(data["is_nanobody_like"])
    data["has_structure"] = as_bool(data["has_structure"])
    data["neutralising_conflict"] = as_bool(data["neutralising_conflict"])

    label_mask = data["neutralising_label"].isin([0, 1])
    removed_label_missing = int((~label_mask).sum())
    labeled = data.loc[label_mask].copy()

    conflict_mask = labeled["neutralising_conflict"]
    removed_conflict = int(conflict_mask.sum())
    non_conflict = labeled.loc[~conflict_mask].copy()

    heavy_ok = (
        non_conflict["has_heavy_or_vhh"]
        & non_conflict["valid_heavy_or_vhh_sequence"]
        & has_sequence(non_conflict["heavy_or_vhh_sequence"])
    )
    removed_invalid_heavy = int((~heavy_ok).sum())
    heavy_valid = non_conflict.loc[heavy_ok].copy()

    light_ok = ~heavy_valid["has_light"] | heavy_valid["valid_light_sequence"]
    removed_invalid_light = int((~light_ok).sum())
    ml = heavy_valid.loc[light_ok].copy()
    ml = add_model_columns(ml)

    # Keep a compact but traceable table for model development.
    optional_columns = [column for column in OUTPUT_COLUMNS if column in ml.columns]
    ml = ml[optional_columns].copy()

    label_counts = ml["label"].value_counts().sort_index()
    final_count = int(len(ml))
    label_0_count = int(label_counts.get(0, 0))
    label_1_count = int(label_counts.get(1, 0))

    metrics: dict[str, object] = {
        "input_row_count": int(len(data)),
        "removed_label_missing": removed_label_missing,
        "removed_conflict": removed_conflict,
        "removed_invalid_heavy": removed_invalid_heavy,
        "removed_invalid_light": removed_invalid_light,
        "final_ml_row_count": final_count,
        "label_0_count": label_0_count,
        "label_1_count": label_1_count,
        "label_0_percent": (label_0_count / final_count * 100) if final_count else 0.0,
        "label_1_percent": (label_1_count / final_count * 100) if final_count else 0.0,
        "paired_heavy_light_count": int(ml["has_light"].sum()),
        "nanobody_like_count": int(ml["is_nanobody_like"].sum()),
        "rows_with_structure": int(ml["has_structure"].sum()),
        "targets_rbd_count": int(ml["targets_rbd"].sum()),
        "targets_spike_count": int(ml["targets_spike"].sum()),
        "targets_ntd_count": int(ml["targets_ntd"].sum()),
    }

    summaries = {
        "heavy_length": summarize_numeric(ml["heavy_length"]),
        "light_length": summarize_numeric(ml["light_length"]),
        "cdrh3_length": summarize_numeric(ml["cdrh3_length"]),
        "cdrl3_length": summarize_numeric(ml["cdrl3_length"]),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    ml.to_csv(OUTPUT_PATH, index=False)
    REPORT_PATH.write_text(build_report(metrics, summaries), encoding="utf-8")
    make_figures(ml)

    print(f"Input row count: {metrics['input_row_count']}")
    print(f"Rows removed because label missing: {metrics['removed_label_missing']}")
    print(f"Rows removed because neutralisation_conflict true: {metrics['removed_conflict']}")
    print(f"Rows removed because missing/invalid heavy sequence: {metrics['removed_invalid_heavy']}")
    print(f"Rows removed because present light sequence is invalid: {metrics['removed_invalid_light']}")
    print(f"Final ML row count: {metrics['final_ml_row_count']}")
    print(f"Label 0 count: {metrics['label_0_count']}")
    print(f"Label 1 count: {metrics['label_1_count']}")
    print(f"Paired heavy-light count: {metrics['paired_heavy_light_count']}")
    print(f"Nanobody-like count: {metrics['nanobody_like_count']}")
    print(f"Rows with structure: {metrics['rows_with_structure']}")
    print(f"Targets RBD: {metrics['targets_rbd_count']}")
    print(f"Targets Spike: {metrics['targets_spike_count']}")
    print(f"Targets NTD: {metrics['targets_ntd_count']}")
    print(f"Saved ML dataset to: {OUTPUT_PATH}")
    print(f"Saved report to: {REPORT_PATH}")
    print(f"Saved figures to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
