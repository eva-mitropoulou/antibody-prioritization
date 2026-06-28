"""Aggregate biological sensitivity analysis for final workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from _safe_analysis_utils import (
    PROJECT_ROOT,
    label_counts_dict,
    label_series,
    numeric_summary,
    read_csv_text,
    value_counts_dict,
    write_json,
    write_text,
)


INPUT_PATH = PROJECT_ROOT / "reports" / "broader_existing_record_prioritization_table.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "biological_sensitivity_report.md"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics" / "biological_sensitivity_metrics.json"


def score_column(data: pd.DataFrame) -> str:
    for column in ["primary_probability", "predicted_neutralisation_probability", "kmer_probability"]:
        if column in data.columns:
            return column
    raise ValueError("No score column found for sensitivity analysis.")


def summarize_group(data: pd.DataFrame, group_column: str, score: str) -> dict[str, Any]:
    """Summarize labels and scores by one subgroup column."""
    if group_column not in data.columns:
        return {"available": False, "groups": {}}
    labels = label_series(data)
    groups = data[group_column].fillna("missing").astype(str).replace({"": "missing"})
    output = {}
    for group, indices in groups.groupby(groups).groups.items():
        mask = data.index.isin(indices)
        output[str(group)] = {
            "row_count": int(mask.sum()),
            "label_counts": label_counts_dict(labels.loc[mask]),
            "score_summary": numeric_summary(data.loc[mask, score]),
            "high_score_count": int(pd.to_numeric(data.loc[mask, score], errors="coerce").ge(0.75).sum()),
        }
    return {"available": True, "groups": output}


def build_metrics() -> dict[str, Any]:
    data = read_csv_text(INPUT_PATH)
    score = score_column(data)
    data[score] = pd.to_numeric(data[score], errors="coerce")
    labels = label_series(data)
    subgroup_columns = [
        "target_region_group",
        "paired_light_status",
        "record_category",
        "developability_risk_bin",
        "confidence_bin",
    ]
    return {
        "status": "available",
        "input_path": str(INPUT_PATH.relative_to(PROJECT_ROOT)),
        "row_count": int(len(data)),
        "score_column": score,
        "label_balance": label_counts_dict(labels),
        "missing_label_count": int(labels.isna().sum()),
        "overall_score_summary": numeric_summary(data[score]),
        "subgroups": {
            column: summarize_group(data, column, score) for column in subgroup_columns
        },
    }


def subgroup_section(column: str, summary: dict[str, Any]) -> list[str]:
    lines = [f"## {column}", ""]
    if not summary["available"]:
        lines.extend(["Not available.", ""])
        return lines
    lines.extend(
        [
            "| Group | Rows | Label 0 | Label 1 | High-score count | Median score |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for group, values in summary["groups"].items():
        median = values["score_summary"]["median"]
        median_text = "n/a" if median is None else f"{median:.4f}"
        lines.append(
            f"| {group} | {values['row_count']} | {values['label_counts']['0']} | "
            f"{values['label_counts']['1']} | {values['high_score_count']} | {median_text} |"
        )
    lines.append("")
    return lines


def build_report(metrics: dict[str, Any]) -> str:
    overall = metrics["overall_score_summary"]
    lines = [
        "# Biological Sensitivity Analysis",
        "",
        "Aggregate sensitivity checks across target-region, paired/light-missing,",
        "record-category, confidence, and sequence-risk subgroups.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Rows | {metrics['row_count']} |",
        f"| Label 0 count | {metrics['label_balance']['0']} |",
        f"| Label 1 count | {metrics['label_balance']['1']} |",
        f"| Missing label count | {metrics['missing_label_count']} |",
        f"| Median score | {overall['median']:.4f} |",
        "",
    ]
    for column, summary in metrics["subgroups"].items():
        lines.extend(subgroup_section(column, summary))
    return "\n".join(lines)


def main() -> int:
    metrics = build_metrics()
    write_json(METRICS_PATH, metrics)
    write_text(REPORT_PATH, build_report(metrics))
    print(
        "Biological sensitivity complete: "
        f"rows={metrics['row_count']}, score={metrics['score_column']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
