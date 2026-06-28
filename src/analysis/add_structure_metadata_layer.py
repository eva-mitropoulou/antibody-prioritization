"""Add structure metadata availability to the final shortlist."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from _safe_analysis_utils import (
    PROJECT_ROOT,
    read_csv_text,
    safe_output_columns,
    structure_available_series,
    value_counts_dict,
    write_json,
    write_text,
)


INPUT_PATH = PROJECT_ROOT / "reports" / "diversity_aware_existing_record_shortlist.csv"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "structure_annotated_shortlist.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "structure_metadata_report.md"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics" / "structure_metadata_summary.json"
DOCKING_PATH = PROJECT_ROOT / "reports" / "docking_next_steps.md"


def build_outputs() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build structure-annotated shortlist and metrics."""
    data = read_csv_text(INPUT_PATH)
    structure = structure_available_series(data)
    output = data.copy()
    output["structure_available"] = structure
    output["structure_workflow_status"] = output["structure_available"].map(
        lambda value: "structure_metadata_available" if value else "structure_metadata_missing"
    )
    output = output[safe_output_columns(output)]
    target_counts = (
        output.groupby(["target_region_group", "structure_workflow_status"]).size().unstack(fill_value=0)
        if "target_region_group" in output.columns
        else pd.DataFrame()
    )
    metrics = {
        "status": "available",
        "input_path": str(INPUT_PATH.relative_to(PROJECT_ROOT)),
        "row_count": int(len(output)),
        "structure_available_count": int(output["structure_available"].sum()),
        "structure_missing_count": int((~output["structure_available"]).sum()),
        "structure_workflow_status_counts": value_counts_dict(output["structure_workflow_status"]),
        "target_region_by_structure_status": {
            str(index): {str(key): int(value) for key, value in row.items()}
            for index, row in target_counts.iterrows()
        },
        "docking_run_by_default": False,
        "artifacts": {
            "structure_annotated_shortlist": str(OUTPUT_PATH.relative_to(PROJECT_ROOT)),
            "report": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
            "metrics_json": str(METRICS_PATH.relative_to(PROJECT_ROOT)),
            "docking_next_steps": str(DOCKING_PATH.relative_to(PROJECT_ROOT)),
        },
    }
    return output, metrics


def build_report(metrics: dict[str, Any]) -> str:
    lines = [
        "# Structure Metadata Layer",
        "",
        "Structure availability was summarized for the diversity-aware shortlist.",
        "Docking was not run by default.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Shortlist rows | {metrics['row_count']} |",
        f"| Structure metadata available | {metrics['structure_available_count']} |",
        f"| Structure metadata missing | {metrics['structure_missing_count']} |",
        "| Docking run by default | false |",
        "",
        "## Status Counts",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    for key, value in metrics["structure_workflow_status_counts"].items():
        lines.append(f"| {key} | {value} |")
    lines.append("")
    return "\n".join(lines)


def build_docking_next_steps(metrics: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Docking Next Steps",
            "",
            "Docking is intentionally not run in the final workflow.",
            "",
            "Recommended next steps:",
            "",
            "- Select only records with structure metadata or high-confidence homology-model candidates.",
            "- Keep docking as a separate validation workflow from sequence classification metrics.",
            "- Report docking scores separately from neutralisation labels and model probabilities.",
            "- Preserve the existing-record constraint; do not infer therapeutic efficacy from docking alone.",
            "",
            f"Structure-available shortlist records: {metrics['structure_available_count']}.",
            "",
        ]
    )


def main() -> int:
    output, metrics = build_outputs()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)
    write_json(METRICS_PATH, metrics)
    write_text(REPORT_PATH, build_report(metrics))
    write_text(DOCKING_PATH, build_docking_next_steps(metrics))
    print(
        "Structure metadata layer complete: "
        f"available={metrics['structure_available_count']}, "
        f"missing={metrics['structure_missing_count']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
