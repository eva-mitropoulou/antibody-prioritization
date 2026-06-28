"""Build a diversity-aware shortlist from existing scored records.

This script reads the broader existing-record prioritization table and selects
one high-scoring representative per heuristic diversity group. It uses existing
public dataset records only and does not generate, alter, mutate, optimize, or
propose biological sequences.

Run from the project root:

    python src/analysis/build_diversity_aware_shortlist.py
"""

from __future__ import annotations

import json
import os
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# Matplotlib needs a writable cache directory in this environment.
MPLCONFIGDIR = Path("/tmp") / "antibody_prioritization_mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "reports" / "broader_existing_record_prioritization_table.csv"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "diversity_aware_existing_record_shortlist.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "diversity_aware_shortlist_report.md"
SUMMARY_PATH = PROJECT_ROOT / "reports" / "metrics" / "diversity_aware_shortlist_summary.json"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
TIER_COUNTS_PATH = FIGURE_DIR / "shortlist_tier_counts.png"
RECORD_CATEGORY_COUNTS_PATH = FIGURE_DIR / "shortlist_record_category_counts.png"
TARGET_REGION_COUNTS_PATH = FIGURE_DIR / "shortlist_target_region_counts.png"
PROBABILITY_DISTRIBUTION_PATH = FIGURE_DIR / "shortlist_probability_distribution.png"

PROBABILITY_COLUMN = "predicted_neutralisation_probability"
CONFIDENCE_COLUMN = "confidence_bin"
RISK_BIN_COLUMN = "developability_risk_bin"
RISK_SCORE_COLUMN = "developability_risk_score"
RECORD_CATEGORY_COLUMN = "record_category"
TARGET_REGION_COLUMN = "metadata_target_region"
V_GENE_COLUMN = "group_feature_v"
EXISTING_DIVERSITY_COLUMN = "diversity_group"
CDRH3_COLUMN_CANDIDATES = ["cdrh3", "group_feature_cdr3"]
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
    "cdrh1_seq",
    "cdrh2_seq",
    "cdrh3_seq",
    "cdrl1_seq",
    "cdrl2_seq",
    "cdrl3_seq",
    "cdrh3",
    "cdrh3_prefix",
    "cdrh3_suffix",
}

MIN_CANDIDATE_PROBABILITY = 0.75
MODEL_DISAGREEMENT_PROBABILITY = 0.90
HIGH_CONFIDENCE = "high"
ACCEPTABLE_RISK_BINS = {"low", "medium"}
PRIMARY_RECORD_CATEGORIES = {
    "known_neutralising",
    "missing_label",
    "conflict_label",
}
MODEL_DISAGREEMENT_CATEGORY = "known_non_neutralising"
TOP_TARGET_REGIONS_IN_FIGURE = 20
TOP_TARGET_REGIONS_IN_REPORT = 30
FALLBACK_TOP_K_PER_CATEGORY = 25

MISSING_TEXT_VALUES = {"", "na", "nan", "none", "null", "nd", "n/a", "unknown"}


def read_csv(path: Path) -> pd.DataFrame:
    """Read the input table as text while preserving blank fields."""
    if not path.exists():
        raise FileNotFoundError(f"Required input table not found: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def normalized_text(values: pd.Series) -> pd.Series:
    """Normalize free text for grouping and comparisons."""
    return values.fillna("").astype(str).str.strip()


def normalize_group_text(values: pd.Series, missing_value: str) -> pd.Series:
    """Normalize group labels and replace blank-like values."""
    normalized = normalized_text(values)
    missing = normalized.str.lower().isin(MISSING_TEXT_VALUES)
    normalized = normalized.mask(missing, missing_value)
    return normalized


def normalize_sequence(value: Any) -> str:
    """Normalize an existing sequence-like field."""
    text = "" if pd.isna(value) else str(value)
    text = "".join(text.split()).upper()
    if text.lower() in MISSING_TEXT_VALUES:
        return ""
    return text


def short_hash(value: str) -> str:
    """Return a short stable hash for grouping without exposing sequence text."""
    if not value:
        return "missing_cdrh3"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def require_columns(data: pd.DataFrame, columns: list[str]) -> None:
    """Raise a clear error if a required input column is absent."""
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"Input table is missing required columns: {missing}")


def numeric_column(data: pd.DataFrame, column: str) -> pd.Series:
    """Convert an existing column to numeric values."""
    return pd.to_numeric(data[column], errors="coerce")


def source_cdrh3(data: pd.DataFrame) -> tuple[pd.Series, str]:
    """Return the first available CDRH3-like column."""
    for column in CDRH3_COLUMN_CANDIDATES:
        if column in data.columns:
            return data[column].map(normalize_sequence), column
    return pd.Series([""] * len(data), index=data.index), "missing"


def source_v_gene_group(data: pd.DataFrame) -> tuple[pd.Series, str]:
    """Return V-gene group labels from existing columns when available."""
    if V_GENE_COLUMN in data.columns:
        return normalize_group_text(data[V_GENE_COLUMN], "unknown_v"), V_GENE_COLUMN

    if EXISTING_DIVERSITY_COLUMN in data.columns:
        prefix = (
            data[EXISTING_DIVERSITY_COLUMN]
            .fillna("")
            .astype(str)
            .str.split("|", n=1, regex=False)
            .str[0]
        )
        return normalize_group_text(prefix, "unknown_v"), f"{EXISTING_DIVERSITY_COLUMN}_prefix"

    return pd.Series(["unknown_v"] * len(data), index=data.index), "missing"


def add_filter_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Append typed columns used for filtering."""
    output = data.copy()
    output[PROBABILITY_COLUMN] = numeric_column(output, PROBABILITY_COLUMN)
    output["confidence_score"] = numeric_column(output, "confidence_score")
    output[RISK_SCORE_COLUMN] = numeric_column(output, RISK_SCORE_COLUMN)
    output[CONFIDENCE_COLUMN] = normalized_text(output[CONFIDENCE_COLUMN]).str.lower()
    output[RISK_BIN_COLUMN] = normalized_text(output[RISK_BIN_COLUMN]).str.lower()
    output[RECORD_CATEGORY_COLUMN] = normalized_text(output[RECORD_CATEGORY_COLUMN]).str.lower()
    return output


def build_candidate_pool(data: pd.DataFrame) -> pd.DataFrame:
    """Filter to high-score, high-confidence existing-record candidates."""
    output = add_filter_columns(data)
    high_score = output[PROBABILITY_COLUMN].ge(MIN_CANDIDATE_PROBABILITY)
    high_confidence = output[CONFIDENCE_COLUMN].eq(HIGH_CONFIDENCE)
    acceptable_risk = output[RISK_BIN_COLUMN].isin(ACCEPTABLE_RISK_BINS)
    primary_category = output[RECORD_CATEGORY_COLUMN].isin(PRIMARY_RECORD_CATEGORIES)
    model_disagreement = (
        output[RECORD_CATEGORY_COLUMN].eq(MODEL_DISAGREEMENT_CATEGORY)
        & output[PROBABILITY_COLUMN].ge(MODEL_DISAGREEMENT_PROBABILITY)
    )

    candidate_mask = high_score & high_confidence & acceptable_risk
    candidate_mask &= primary_category | model_disagreement

    candidates = output.loc[candidate_mask].copy()
    candidates["model_disagreement_record"] = candidates[RECORD_CATEGORY_COLUMN].eq(
        MODEL_DISAGREEMENT_CATEGORY
    )
    return candidates


def build_relaxed_candidate_pool(data: pd.DataFrame) -> pd.DataFrame:
    """Relax the score threshold once if the default candidate pool is empty."""
    output = add_filter_columns(data)
    relaxed_score = output[PROBABILITY_COLUMN].ge(0.65)
    acceptable_confidence = output[CONFIDENCE_COLUMN].isin({"medium", "high"})
    acceptable_risk = output[RISK_BIN_COLUMN].isin(ACCEPTABLE_RISK_BINS)
    primary_category = output[RECORD_CATEGORY_COLUMN].isin(PRIMARY_RECORD_CATEGORIES)
    candidates = output.loc[
        relaxed_score & acceptable_confidence & acceptable_risk & primary_category
    ].copy()
    candidates["model_disagreement_record"] = False
    return candidates


def build_top_k_fallback_pool(data: pd.DataFrame) -> pd.DataFrame:
    """Use top-k per record category when threshold filters produce no rows."""
    output = add_filter_columns(data)
    output = output.sort_values(
        [RECORD_CATEGORY_COLUMN, PROBABILITY_COLUMN, "confidence_score"],
        ascending=[True, False, False],
        kind="mergesort",
    )
    candidates = output.groupby(RECORD_CATEGORY_COLUMN, group_keys=False).head(
        FALLBACK_TOP_K_PER_CATEGORY
    ).copy()
    candidates["model_disagreement_record"] = candidates[RECORD_CATEGORY_COLUMN].eq(
        MODEL_DISAGREEMENT_CATEGORY
    )
    return candidates


def add_diversity_columns(candidates: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Append heuristic diversity grouping columns."""
    output = candidates.copy()
    cdrh3, cdrh3_source = source_cdrh3(output)
    v_gene_group, v_gene_source = source_v_gene_group(output)
    cdrh3_length = cdrh3.str.len()

    output["target_region_group"] = normalize_group_text(
        output[TARGET_REGION_COLUMN],
        "unknown_target_region",
    )
    output["v_gene_group"] = v_gene_group
    has_cdrh3 = cdrh3.ne("")
    derived_length_bin = np.select(
        [
            has_cdrh3 & cdrh3_length.le(10),
            has_cdrh3 & cdrh3_length.between(11, 20),
            has_cdrh3 & cdrh3_length.gt(20),
        ],
        ["short", "medium", "long"],
        default="unknown",
    )
    if "cdrh3_length_bin" in output.columns:
        existing_bin = normalize_group_text(output["cdrh3_length_bin"], "unknown")
        output["cdrh3_length_bin"] = existing_bin.where(
            existing_bin.ne("unknown"),
            derived_length_bin,
        )
    else:
        output["cdrh3_length_bin"] = derived_length_bin
    output["cdrh3_pattern_hash"] = cdrh3.map(short_hash)
    output["diversity_group"] = (
        output["target_region_group"]
        + " | "
        + output["v_gene_group"]
        + " | "
        + output["cdrh3_length_bin"].astype(str)
        + " | "
        + output["cdrh3_pattern_hash"]
    )

    source_info = {
        "cdrh3_source": cdrh3_source,
        "v_gene_group_source": v_gene_source,
    }
    return output, source_info


def select_diverse_representatives(candidates: pd.DataFrame) -> pd.DataFrame:
    """Keep the top one representative per diversity group."""
    sorted_candidates = candidates.sort_values(
        by=[
            "diversity_group",
            PROBABILITY_COLUMN,
            "confidence_score",
            RISK_SCORE_COLUMN,
            "row_id",
        ],
        ascending=[True, False, False, True, True],
        kind="mergesort",
    ).copy()
    sorted_candidates["diversity_group_candidate_count"] = sorted_candidates.groupby(
        "diversity_group"
    )["diversity_group"].transform("size")
    sorted_candidates["candidate_rank_within_diversity_group"] = (
        sorted_candidates.groupby("diversity_group").cumcount() + 1
    )
    shortlist = sorted_candidates[
        sorted_candidates["candidate_rank_within_diversity_group"].eq(1)
    ].copy()
    shortlist = shortlist.sort_values(
        by=[
            PROBABILITY_COLUMN,
            "confidence_score",
            RISK_SCORE_COLUMN,
            "target_region_group",
            "sample_name",
        ],
        ascending=[False, False, True, True, True],
        kind="mergesort",
    )
    shortlist.insert(0, "shortlist_rank", np.arange(1, len(shortlist) + 1, dtype=int))
    return shortlist


def add_shortlist_tiers(shortlist: pd.DataFrame) -> pd.DataFrame:
    """Assign mutually exclusive shortlist tiers."""
    output = shortlist.copy()
    review = output[RECORD_CATEGORY_COLUMN].eq("conflict_label") | output[
        "model_disagreement_record"
    ].astype(bool)
    tier_1 = (
        output[PROBABILITY_COLUMN].ge(0.85)
        & output[CONFIDENCE_COLUMN].eq(HIGH_CONFIDENCE)
        & output[RISK_BIN_COLUMN].eq("low")
    )
    tier_2 = (
        output[PROBABILITY_COLUMN].ge(0.75)
        & output[CONFIDENCE_COLUMN].eq(HIGH_CONFIDENCE)
        & output[RISK_BIN_COLUMN].isin(ACCEPTABLE_RISK_BINS)
    )

    output["shortlist_tier"] = "tier_2"
    output.loc[tier_1, "shortlist_tier"] = "tier_1"
    output.loc[tier_2 & ~tier_1, "shortlist_tier"] = "tier_2"
    output.loc[review, "shortlist_tier"] = "review"
    return output


def order_shortlist_columns(shortlist: pd.DataFrame) -> pd.DataFrame:
    """Put shortlist and diversity columns before original record metadata."""
    leading_columns = [
        "shortlist_rank",
        "shortlist_tier",
        "sample_name",
        "sample_type",
        "record_category",
        "model_disagreement_record",
        PROBABILITY_COLUMN,
        "confidence_score",
        CONFIDENCE_COLUMN,
        RISK_SCORE_COLUMN,
        RISK_BIN_COLUMN,
        "target_region_group",
        "v_gene_group",
        "cdrh3_length_bin",
        "cdrh3_pattern_hash",
        "diversity_group",
        "diversity_group_candidate_count",
        "candidate_rank_within_diversity_group",
        "metadata_target_region",
        "group_feature_v",
        "row_id",
    ]
    existing_leading = [column for column in leading_columns if column in shortlist.columns]
    trailing = [
        column
        for column in shortlist.columns
        if column not in existing_leading and column not in SEQUENCE_VALUE_COLUMNS
    ]
    return shortlist[existing_leading + trailing]


def value_counts_dict(values: pd.Series) -> dict[str, int]:
    """Return stable JSON-safe value counts."""
    counts = values.value_counts(dropna=False)
    return {str(key): int(value) for key, value in counts.items()}


def top_counts_dict(values: pd.Series, limit: int | None = None) -> dict[str, int]:
    """Return full or top-N value counts as a JSON-safe dictionary."""
    counts = values.value_counts(dropna=False)
    if limit is not None:
        counts = counts.head(limit)
    return {str(key): int(value) for key, value in counts.items()}


def summarize(
    input_rows: int,
    candidates: pd.DataFrame,
    shortlist: pd.DataFrame,
    source_info: dict[str, str],
    fallback_used: str | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Build the metrics summary."""
    return {
        "status": "available",
        "fallback_used": fallback_used,
        "warnings": warnings or [],
        "input_path": str(INPUT_PATH.relative_to(PROJECT_ROOT)),
        "input_record_count": int(input_rows),
        "candidate_pool_size_before_diversity_filtering": int(len(candidates)),
        "final_shortlist_size": int(len(shortlist)),
        "diversity_group_count": int(candidates["diversity_group"].nunique()),
        "source_columns": source_info,
        "candidate_pool_record_category_counts": value_counts_dict(
            candidates[RECORD_CATEGORY_COLUMN]
        ),
        "candidate_pool_target_region_counts": top_counts_dict(
            candidates["target_region_group"]
        ),
        "shortlist_tier_counts": value_counts_dict(shortlist["shortlist_tier"]),
        "shortlist_record_category_counts": value_counts_dict(
            shortlist[RECORD_CATEGORY_COLUMN]
        ),
        "shortlist_target_region_counts": top_counts_dict(
            shortlist["target_region_group"]
        ),
        "shortlist_top_target_region_counts": top_counts_dict(
            shortlist["target_region_group"],
            TOP_TARGET_REGIONS_IN_REPORT,
        ),
        "missing_label_shortlist_count": int(
            shortlist[RECORD_CATEGORY_COLUMN].eq("missing_label").sum()
        ),
        "conflict_label_shortlist_count": int(
            shortlist[RECORD_CATEGORY_COLUMN].eq("conflict_label").sum()
        ),
        "model_disagreement_shortlist_count": int(
            shortlist["model_disagreement_record"].astype(bool).sum()
        ),
        "probability_summary": {
            "min": float(shortlist[PROBABILITY_COLUMN].min()) if len(shortlist) else None,
            "median": float(shortlist[PROBABILITY_COLUMN].median()) if len(shortlist) else None,
            "mean": float(shortlist[PROBABILITY_COLUMN].mean()) if len(shortlist) else None,
            "max": float(shortlist[PROBABILITY_COLUMN].max()) if len(shortlist) else None,
        },
        "artifacts": {
            "shortlist_csv": str(OUTPUT_PATH.relative_to(PROJECT_ROOT)),
            "report": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
            "summary_json": str(SUMMARY_PATH.relative_to(PROJECT_ROOT)),
            "tier_counts": str(TIER_COUNTS_PATH.relative_to(PROJECT_ROOT)),
            "record_category_counts": str(
                RECORD_CATEGORY_COUNTS_PATH.relative_to(PROJECT_ROOT)
            ),
            "target_region_counts": str(TARGET_REGION_COUNTS_PATH.relative_to(PROJECT_ROOT)),
            "probability_distribution": str(
                PROBABILITY_DISTRIBUTION_PATH.relative_to(PROJECT_ROOT)
            ),
        },
    }


def save_count_figure(
    counts: pd.Series,
    path: Path,
    title: str,
    xlabel: str,
    color: str,
    max_categories: int | None = None,
) -> None:
    """Save a horizontal count bar plot."""
    plot_counts = counts.copy()
    if max_categories is not None and len(plot_counts) > max_categories:
        top = plot_counts.head(max_categories)
        other = int(plot_counts.iloc[max_categories:].sum())
        if other:
            plot_counts = pd.concat([top, pd.Series({"Other": other})])
        else:
            plot_counts = top

    fig_height = max(3.5, 0.34 * max(1, len(plot_counts)) + 1.4)
    fig, ax = plt.subplots(figsize=(9, fig_height))
    plot_counts.sort_values(ascending=True).plot.barh(ax=ax, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_probability_distribution(shortlist: pd.DataFrame) -> None:
    """Save the shortlist probability distribution."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(shortlist[PROBABILITY_COLUMN].dropna(), bins=20, color="#4C78A8", alpha=0.85)
    ax.axvline(0.75, color="#F58518", linestyle="--", linewidth=1.5, label="candidate cutoff")
    ax.axvline(0.85, color="#54A24B", linestyle="--", linewidth=1.5, label="tier 1 cutoff")
    ax.set_title("Shortlist predicted neutralisation probabilities")
    ax.set_xlabel("Predicted neutralisation probability")
    ax.set_ylabel("Existing record count")
    ax.legend(frameon=False)
    fig.tight_layout()
    PROBABILITY_DISTRIBUTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PROBABILITY_DISTRIBUTION_PATH, dpi=200)
    plt.close(fig)


def save_figures(shortlist: pd.DataFrame) -> None:
    """Save all requested figures."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    save_count_figure(
        shortlist["shortlist_tier"].value_counts(),
        TIER_COUNTS_PATH,
        "Shortlist tier counts",
        "Existing record count",
        "#4C78A8",
    )
    save_count_figure(
        shortlist[RECORD_CATEGORY_COLUMN].value_counts(),
        RECORD_CATEGORY_COUNTS_PATH,
        "Shortlist record categories",
        "Existing record count",
        "#54A24B",
    )
    save_count_figure(
        shortlist["target_region_group"].value_counts(),
        TARGET_REGION_COUNTS_PATH,
        "Shortlist target regions",
        "Existing record count",
        "#E45756",
        max_categories=TOP_TARGET_REGIONS_IN_FIGURE,
    )
    save_probability_distribution(shortlist)


def format_counts_table(counts: dict[str, int], name: str) -> list[str]:
    """Format a count dictionary as Markdown rows."""
    lines = [f"| {name} | Count |", "|---|---:|"]
    for key, value in counts.items():
        lines.append(f"| {key} | {value} |")
    return lines


def build_report(summary: dict[str, Any], shortlist: pd.DataFrame) -> str:
    """Build a Markdown report for the diversity-aware shortlist."""
    lines = [
        "# Diversity-Aware Existing-Record Shortlist",
        "",
        "This shortlist contains existing public dataset records only. It does not",
        "create, alter, mutate, optimize, rank newly designed sequences, or propose",
        "sequence changes.",
        "",
        "## Method",
        "",
        (
            "Candidates were filtered to predicted probability >= 0.75, high "
            "confidence, and low or medium sequence-risk bins."
        ),
        (
            "Known neutralising, missing-label, and conflict-label records were kept. "
            "Known non-neutralising records were included only when probability >= 0.90 "
            "and were marked as model-disagreement records."
        ),
        (
            "One representative was selected per diversity group after sorting by "
            "predicted probability, confidence score, and sequence-risk score."
        ),
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Input records | {summary['input_record_count']} |",
        (
            "| Candidate pool size before diversity filtering | "
            f"{summary['candidate_pool_size_before_diversity_filtering']} |"
        ),
        f"| Final shortlist size | {summary['final_shortlist_size']} |",
        f"| Diversity groups | {summary['diversity_group_count']} |",
        f"| Missing-label records in shortlist | {summary['missing_label_shortlist_count']} |",
        f"| Conflict-label records in shortlist | {summary['conflict_label_shortlist_count']} |",
        (
            "| Model-disagreement records in shortlist | "
            f"{summary['model_disagreement_shortlist_count']} |"
        ),
        "",
        "## Source Columns",
        "",
        f"- CDRH3 source: `{summary['source_columns']['cdrh3_source']}`",
        f"- V-gene group source: `{summary['source_columns']['v_gene_group_source']}`",
    ]
    if summary["source_columns"]["v_gene_group_source"] != V_GENE_COLUMN:
        lines.append(
            "- The broader input table did not contain a direct `group_feature_v` column; "
            "the script used only existing table fields and did not infer or assign new "
            "V-gene annotations."
        )
    lines.extend(["", "## Tier Counts", ""])
    lines.extend(format_counts_table(summary["shortlist_tier_counts"], "Tier"))
    lines.extend(["", "## Record Category Counts", ""])
    lines.extend(format_counts_table(summary["shortlist_record_category_counts"], "Record category"))
    lines.extend(["", "## Target Region Counts", ""])
    lines.extend(
        format_counts_table(
            summary["shortlist_top_target_region_counts"],
            "Target region",
        )
    )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Model score is not therapeutic efficacy.",
            "- Diversity grouping is heuristic.",
            "- Labels are heterogeneous literature-derived labels.",
            "- This is retrospective prioritization of existing records only.",
            "- No new sequences are generated, altered, proposed, or optimized.",
            "",
            "## Artifacts",
            "",
            f"- `{OUTPUT_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{SUMMARY_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{TIER_COUNTS_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{RECORD_CATEGORY_COUNTS_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{TARGET_REGION_COUNTS_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{PROBABILITY_DISTRIBUTION_PATH.relative_to(PROJECT_ROOT)}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    """Build and save the diversity-aware shortlist artifacts."""
    data = read_csv(INPUT_PATH)
    require_columns(
        data,
        [
            PROBABILITY_COLUMN,
            "confidence_score",
            CONFIDENCE_COLUMN,
            RISK_SCORE_COLUMN,
            RISK_BIN_COLUMN,
            RECORD_CATEGORY_COLUMN,
            TARGET_REGION_COLUMN,
        ],
    )

    candidates = build_candidate_pool(data)
    fallback_used = None
    warnings: list[str] = []
    if candidates.empty:
        candidates = build_relaxed_candidate_pool(data)
        fallback_used = "relaxed_score_threshold_once"
        warnings.append("Default candidate pool was empty; score threshold was relaxed once.")
    if candidates.empty:
        candidates = build_top_k_fallback_pool(data)
        fallback_used = "top_k_per_category"
        warnings.append("Relaxed candidate pool was empty; using top-k per category fallback.")
    candidates, source_info = add_diversity_columns(candidates)
    shortlist = select_diverse_representatives(candidates)
    shortlist = add_shortlist_tiers(shortlist)
    shortlist = order_shortlist_columns(shortlist)
    summary = summarize(len(data), candidates, shortlist, source_info, fallback_used, warnings)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    shortlist.to_csv(OUTPUT_PATH, index=False)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    REPORT_PATH.write_text(build_report(summary, shortlist))
    save_figures(shortlist)

    print(
        "Diversity-aware shortlist complete: "
        f"candidates={len(candidates)}, shortlist={len(shortlist)}, "
        f"diversity_groups={summary['diversity_group_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
