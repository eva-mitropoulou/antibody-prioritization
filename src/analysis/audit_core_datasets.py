"""Audit core antibody datasets using aggregate-only outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from _safe_analysis_utils import (
    PROJECT_ROOT,
    label_counts_dict,
    label_series,
    light_status_series,
    missing_counts_dict,
    read_csv_text,
    relpath,
    structure_available_series,
    target_region_group_series,
    value_counts_dict,
    write_json,
    write_text,
)


STRICT_PATH = PROJECT_ROOT / "data" / "processed" / "neutral_sequence_classification_ml.csv"
BROADER_PATH = PROJECT_ROOT / "data" / "processed" / "neutral_prepared_sequences.csv"
ANNOTATED_PATH = PROJECT_ROOT / "data" / "processed" / "bioaware_paired_cdr_annotated.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "core_dataset_audit.md"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics" / "core_dataset_audit.json"


DOMAIN_REGION_HINTS = ("cdr", "region", "annotation", "chain_type", "marker")


def domain_region_columns(data: pd.DataFrame) -> list[str]:
    """Return domain-region-related column names only."""
    return [
        column
        for column in data.columns
        if any(hint in column.lower() for hint in DOMAIN_REGION_HINTS)
    ]


def summarize_table(path: Path, strict_labels: bool) -> dict[str, Any]:
    """Build an aggregate-only summary for one table."""
    data = read_csv_text(path)
    labels = label_series(data)
    target_groups = target_region_group_series(data)
    light_status = light_status_series(data)
    structure_available = structure_available_series(data)
    labeled = labels.notna()

    summary = {
        "path": relpath(path),
        "exists": path.exists(),
        "shape": [int(data.shape[0]), int(data.shape[1])],
        "columns": list(data.columns),
        "missing_value_counts": missing_counts_dict(data),
        "label_counts": label_counts_dict(labels),
        "strict_labeled_row_count": int(labeled.sum()) if strict_labels else None,
        "broader_row_count": int(len(data)) if not strict_labels else None,
        "paired_light_status_counts": value_counts_dict(light_status),
        "target_region_group_counts": value_counts_dict(target_groups),
        "target_region_metadata_available_count": int(target_groups.ne("unknown").sum()),
        "structure_available_count": int(structure_available.sum()),
        "structure_missing_count": int((~structure_available).sum()),
        "domain_region_columns": domain_region_columns(data),
    }
    return summary


def build_metrics() -> dict[str, Any]:
    """Build the core dataset audit metrics."""
    strict = summarize_table(STRICT_PATH, strict_labels=True)
    broader = summarize_table(BROADER_PATH, strict_labels=False)
    annotated = summarize_table(ANNOTATED_PATH, strict_labels=True) if ANNOTATED_PATH.exists() else None

    return {
        "status": "available",
        "strict_labeled_dataset": strict,
        "broader_prepared_dataset": broader,
        "annotated_paired_dataset": annotated,
        "quality_gates": {
            "strict_labeled_row_count_reported": strict["strict_labeled_row_count"] is not None,
            "broader_row_count_reported": broader["broader_row_count"] is not None,
            "both_classes_counted": all(
                strict["label_counts"].get(label, 0) > 0 for label in ["0", "1"]
            ),
            "paired_and_light_missing_counted": all(
                key in strict["paired_light_status_counts"]
                for key in ["paired", "light_missing_or_single_chain"]
            ),
        },
    }


def report_table_summary(name: str, summary: dict[str, Any]) -> list[str]:
    """Format one table summary."""
    lines = [
        f"## {name}",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Rows | {summary['shape'][0]} |",
        f"| Columns | {summary['shape'][1]} |",
        f"| Label 0 count | {summary['label_counts'].get('0', 0)} |",
        f"| Label 1 count | {summary['label_counts'].get('1', 0)} |",
        f"| Target-region metadata available | {summary['target_region_metadata_available_count']} |",
        f"| Structure metadata available | {summary['structure_available_count']} |",
        "",
        "### Columns",
        "",
        ", ".join(f"`{column}`" for column in summary["columns"]),
        "",
    ]
    lines.extend(["### Paired/Light-Missing Counts", "", "| Status | Count |", "|---|---:|"])
    for key, value in summary["paired_light_status_counts"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "### Target-Region Group Counts", "", "| Group | Count |", "|---|---:|"])
    for key, value in summary["target_region_group_counts"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "### Domain/Region Columns", ""])
    if summary["domain_region_columns"]:
        lines.append(", ".join(f"`{column}`" for column in summary["domain_region_columns"]))
    else:
        lines.append("No domain-region columns detected.")
    lines.extend(["", "### Missing-Value Counts", "", "| Column | Missing count |", "|---|---:|"])
    for key, value in summary["missing_value_counts"].items():
        lines.append(f"| `{key}` | {value} |")
    lines.append("")
    return lines


def build_report(metrics: dict[str, Any]) -> str:
    """Build the Markdown report."""
    lines = [
        "# Core Dataset Audit",
        "",
        "Aggregate-only audit of the core sequence-record datasets. No raw records,",
        "sequence strings, or source links are included.",
        "",
    ]
    lines.extend(report_table_summary("Strict Labeled ML Dataset", metrics["strict_labeled_dataset"]))
    lines.extend(report_table_summary("Broader Prepared Dataset", metrics["broader_prepared_dataset"]))
    annotated = metrics.get("annotated_paired_dataset")
    if annotated:
        lines.extend(report_table_summary("Annotated Paired Dataset", annotated))
    lines.extend(
        [
            "## Quality Gates",
            "",
            "| Gate | Pass |",
            "|---|---:|",
        ]
    )
    for key, value in metrics["quality_gates"].items():
        lines.append(f"| {key} | {str(bool(value)).lower()} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    metrics = build_metrics()
    write_json(METRICS_PATH, metrics)
    write_text(REPORT_PATH, build_report(metrics))
    print(
        "Core dataset audit complete: "
        f"strict_rows={metrics['strict_labeled_dataset']['shape'][0]}, "
        f"broader_rows={metrics['broader_prepared_dataset']['shape'][0]}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
