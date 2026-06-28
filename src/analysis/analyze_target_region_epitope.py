"""Analyze target-region metadata as aggregate subgroup context."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/antibody_prioritization_mplconfig")
import matplotlib.pyplot as plt

from _safe_analysis_utils import (
    PROJECT_ROOT,
    label_counts_dict,
    label_series,
    light_status_series,
    read_csv_text,
    structure_available_series,
    target_region_group_series,
    value_counts_dict,
    write_json,
    write_text,
)


INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "neutral_prepared_sequences.csv"
STRICT_PATH = PROJECT_ROOT / "data" / "processed" / "neutral_sequence_classification_ml.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "target_region_epitope_analysis.md"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics" / "target_region_epitope_analysis.json"
FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "target_region_group_counts.png"


def subgroup_summary(data: pd.DataFrame) -> dict[str, Any]:
    """Summarize label balance and metadata by normalized target-region group."""
    labels = label_series(data)
    target_groups = target_region_group_series(data)
    light_status = light_status_series(data)
    structure = structure_available_series(data)
    rows: dict[str, Any] = {}
    for group in ["RBD", "NTD", "Spike/S", "other", "unknown"]:
        mask = target_groups.eq(group)
        labeled = labels.loc[mask].notna()
        rows[group] = {
            "row_count": int(mask.sum()),
            "labeled_row_count": int(labeled.sum()),
            "label_counts": label_counts_dict(labels.loc[mask]),
            "paired_light_status_counts": value_counts_dict(light_status.loc[mask]),
            "structure_available_count": int(structure.loc[mask].sum()),
            "structure_missing_count": int((~structure.loc[mask]).sum()),
        }
    return rows


def build_metrics() -> dict[str, Any]:
    """Build target-region metrics."""
    broader = read_csv_text(INPUT_PATH)
    strict = read_csv_text(STRICT_PATH)
    broader_groups = target_region_group_series(broader)
    strict_groups = target_region_group_series(strict)
    metrics = {
        "status": "available",
        "input_path": str(INPUT_PATH.relative_to(PROJECT_ROOT)),
        "strict_input_path": str(STRICT_PATH.relative_to(PROJECT_ROOT)),
        "broader_row_count": int(len(broader)),
        "strict_labeled_row_count": int(len(strict)),
        "broader_target_region_group_counts": value_counts_dict(broader_groups),
        "strict_target_region_group_counts": value_counts_dict(strict_groups),
        "broader_unknown_count": int(broader_groups.eq("unknown").sum()),
        "strict_unknown_count": int(strict_groups.eq("unknown").sum()),
        "broader_subgroups": subgroup_summary(broader),
        "strict_labeled_subgroups": subgroup_summary(strict),
    }
    usable_groups = [
        group
        for group, summary in metrics["strict_labeled_subgroups"].items()
        if summary["labeled_row_count"] >= 50 and summary["label_counts"]["0"] > 0 and summary["label_counts"]["1"] > 0
    ]
    metrics["target_region_metadata_useful_for_subgroup_analysis"] = bool(usable_groups)
    metrics["usable_labeled_groups"] = usable_groups
    return metrics


def save_figure(metrics: dict[str, Any]) -> None:
    """Save aggregate target-region group counts."""
    counts = metrics["broader_target_region_group_counts"]
    if not counts:
        return
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ordered = pd.Series(counts).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ordered.plot.barh(ax=ax, color="#4C78A8")
    ax.set_xlabel("Existing record count")
    ax.set_ylabel("")
    ax.set_title("Target-region metadata groups")
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=200)
    plt.close(fig)


def subgroup_table(title: str, groups: dict[str, Any]) -> list[str]:
    """Format subgroup summaries."""
    lines = [
        f"## {title}",
        "",
        (
            "| Group | Rows | Labeled rows | Label 0 | Label 1 | Paired | "
            "Light-missing/single-chain | Structure available |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group, summary in groups.items():
        status = summary["paired_light_status_counts"]
        lines.append(
            f"| {group} | {summary['row_count']} | {summary['labeled_row_count']} | "
            f"{summary['label_counts']['0']} | {summary['label_counts']['1']} | "
            f"{status.get('paired', 0)} | "
            f"{status.get('light_missing_or_single_chain', 0)} | "
            f"{summary['structure_available_count']} |"
        )
    lines.append("")
    return lines


def build_report(metrics: dict[str, Any]) -> str:
    """Build Markdown report."""
    lines = [
        "# Target-Region Metadata Analysis",
        "",
        "Target-region metadata was normalized to RBD, NTD, Spike/S, other, and",
        "unknown groups. This report contains aggregate counts only.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Broader records | {metrics['broader_row_count']} |",
        f"| Strict labeled records | {metrics['strict_labeled_row_count']} |",
        f"| Broader unknown target-region count | {metrics['broader_unknown_count']} |",
        f"| Strict unknown target-region count | {metrics['strict_unknown_count']} |",
        (
            "| Metadata useful for subgroup analysis | "
            f"{str(metrics['target_region_metadata_useful_for_subgroup_analysis']).lower()} |"
        ),
        "",
    ]
    lines.extend(subgroup_table("Broader Prepared Dataset", metrics["broader_subgroups"]))
    lines.extend(subgroup_table("Strict Labeled Dataset", metrics["strict_labeled_subgroups"]))
    lines.extend(
        [
            "## Interpretation",
            "",
            (
                "Target-region metadata is useful for subgroup analysis when a normalized "
                "group has enough labeled rows and contains both label classes."
            ),
            "",
            "## Artifacts",
            "",
            f"- `{METRICS_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{FIGURE_PATH.relative_to(PROJECT_ROOT)}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    metrics = build_metrics()
    write_json(METRICS_PATH, metrics)
    write_text(REPORT_PATH, build_report(metrics))
    save_figure(metrics)
    print(
        "Target-region analysis complete: "
        f"unknown={metrics['broader_unknown_count']}, "
        f"useful={metrics['target_region_metadata_useful_for_subgroup_analysis']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
