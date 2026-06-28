"""Summarize domain-region annotation coverage without rerunning annotation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from _safe_analysis_utils import (
    PROJECT_ROOT,
    label_series,
    light_status_series,
    load_json,
    read_csv_text,
    relpath,
    value_counts_dict,
    write_json,
    write_text,
)


ANNOTATED_PATH = PROJECT_ROOT / "data" / "processed" / "bioaware_paired_cdr_annotated.csv"
EXISTING_REPORT_PATH = PROJECT_ROOT / "reports" / "bioaware_cdr_annotation_report.md"
EXISTING_METRICS_PATH = PROJECT_ROOT / "reports" / "metrics" / "bioaware_cdr_annotation_summary.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "domain_region_annotation_summary.md"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics" / "domain_region_annotation_summary.json"


CDR_FOUND_COLUMNS = [
    "cdrh1_found",
    "cdrh2_found",
    "cdrh3_found",
    "cdrl1_found",
    "cdrl2_found",
    "cdrl3_found",
]


def bool_count(data: pd.DataFrame, column: str) -> int | None:
    """Return true count for a boolean-like column."""
    if column not in data.columns:
        return None
    values = data[column].fillna("").astype(str).str.strip().str.lower()
    return int(values.isin({"true", "1", "yes"}).sum())


def build_metrics() -> dict[str, Any]:
    """Build coverage metrics from existing annotation artifacts."""
    existing = load_json(EXISTING_METRICS_PATH) or {}
    if not ANNOTATED_PATH.exists():
        return {
            "status": "missing_annotation_file",
            "annotated_file": relpath(ANNOTATED_PATH),
            "existing_report_available": EXISTING_REPORT_PATH.exists(),
            "existing_metrics_available": EXISTING_METRICS_PATH.exists(),
        }

    data = read_csv_text(ANNOTATED_PATH)
    light_status = light_status_series(data)
    labels = label_series(data)
    cdr_coverage = {
        column: bool_count(data, column)
        for column in CDR_FOUND_COLUMNS
        if column in data.columns
    }
    all_six = bool_count(data, "all_six_cdrs_found")
    fallback_columns = [
        column
        for column in ["existing_cdrh3", "existing_cdrl3", "group_feature_cdr3", "group_feature_b_cdr3"]
        if column in data.columns
    ]
    mismatch_counts = {}
    for column in ["heavy_annotation_ok", "light_annotation_ok", "marker_insertion_ok"]:
        value = bool_count(data, column)
        if value is not None:
            mismatch_counts[f"{column}_false_or_missing"] = int(len(data) - value)

    return {
        "status": "available",
        "annotated_file": relpath(ANNOTATED_PATH),
        "existing_report_available": EXISTING_REPORT_PATH.exists(),
        "existing_metrics_available": EXISTING_METRICS_PATH.exists(),
        "row_count": int(len(data)),
        "label_counts": {
            "0": int(labels.eq(0).sum()),
            "1": int(labels.eq(1).sum()),
            "missing": int(labels.isna().sum()),
        },
        "paired_row_count": int(light_status.eq("paired").sum()),
        "single_chain_or_light_missing_row_count": int(
            light_status.eq("light_missing_or_single_chain").sum()
        ),
        "paired_light_status_counts": value_counts_dict(light_status),
        "six_region_coverage": cdr_coverage,
        "all_six_cdrs_found_count": all_six,
        "all_six_cdrs_found_fraction": float(all_six / len(data)) if all_six is not None else None,
        "fallback_region3_columns_available": fallback_columns,
        "mismatch_or_missing_counts": mismatch_counts,
        "reused_existing_metrics_keys": sorted(existing.keys()),
        "source_existing_summary": {
            "paired_annotation_candidate_count": existing.get("paired_annotation_candidate_count"),
            "nanobody_like_or_missing_light_count": existing.get(
                "nanobody_like_or_missing_light_count"
            ),
            "status": existing.get("status"),
        },
    }


def build_report(metrics: dict[str, Any]) -> str:
    """Build Markdown report."""
    lines = [
        "# Domain-Region Annotation Summary",
        "",
        "This stage reuses existing annotation artifacts when present. It does not",
        "rerun heavy annotation or print raw sequence values.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Status | {metrics['status']} |",
        f"| Annotated rows | {metrics.get('row_count', 0)} |",
        f"| Paired rows | {metrics.get('paired_row_count', 0)} |",
        (
            "| Single-chain or light-missing rows | "
            f"{metrics.get('single_chain_or_light_missing_row_count', 0)} |"
        ),
        f"| Existing annotation report available | {str(metrics.get('existing_report_available')).lower()} |",
        f"| Existing annotation metrics available | {str(metrics.get('existing_metrics_available')).lower()} |",
        "",
    ]
    if metrics["status"] == "available":
        lines.extend(["## Six-Region Coverage", "", "| Region flag | Found count |", "|---|---:|"])
        for key, value in metrics["six_region_coverage"].items():
            lines.append(f"| `{key}` | {value} |")
        lines.extend(
            [
                f"| `all_six_cdrs_found` | {metrics.get('all_six_cdrs_found_count')} |",
                "",
                "## Fallback Region-3 Columns",
                "",
            ]
        )
        if metrics["fallback_region3_columns_available"]:
            lines.append(", ".join(f"`{column}`" for column in metrics["fallback_region3_columns_available"]))
        else:
            lines.append("No region-3 fallback columns were detected.")
        lines.extend(["", "## Mismatch/Missing Counts", "", "| Check | Count |", "|---|---:|"])
        for key, value in metrics["mismatch_or_missing_counts"].items():
            lines.append(f"| {key} | {value} |")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    metrics = build_metrics()
    write_json(METRICS_PATH, metrics)
    write_text(REPORT_PATH, build_report(metrics))
    print(
        "Domain-region annotation summary complete: "
        f"status={metrics['status']}, paired={metrics.get('paired_row_count', 0)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
