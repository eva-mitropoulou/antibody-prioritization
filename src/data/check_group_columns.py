"""Report neutral grouping-column diagnostics.

The script reads only the neutral sequence-classification table and writes
Markdown plus JSON diagnostics. Raw values are represented with deterministic
neutral value IDs in reports and terminal output.

Run from the project root:

    python src/data/check_group_columns.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "neutral_sequence_classification_ml.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "group_column_diagnostics.md"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics" / "group_column_diagnostics.json"

GROUP_COLUMNS = [
    "group_feature_cdr3",
    "group_feature_v",
    "group_feature_j",
    "group_feature_b_v",
    "group_feature_b_j",
    "metadata_target_region",
]

MOSTLY_MISSING_THRESHOLD = 0.50
NEAR_ROW_UNIQUE_ROW_THRESHOLD = 0.80
NEAR_ROW_UNIQUE_NON_MISSING_THRESHOLD = 0.90


def read_csv(path: Path) -> pd.DataFrame:
    """Read the neutral CSV as text while preserving blank fields as blanks."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def normalized_text(values: pd.Series) -> pd.Series:
    """Normalize text for diagnostics without changing the saved table."""
    return values.fillna("").astype(str).str.strip()


def value_id(value: str) -> str:
    """Create a stable neutral identifier for a raw observed value."""
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"value_{digest}"


def useful_for_grouping(
    row_count: int,
    exists: bool,
    missing_count: int,
    non_missing_count: int,
    unique_non_missing_count: int,
) -> bool:
    """Apply neutral grouping usefulness checks."""
    if not exists or row_count == 0 or non_missing_count == 0:
        return False

    missing_ratio = missing_count / row_count
    unique_row_ratio = unique_non_missing_count / row_count
    unique_non_missing_ratio = unique_non_missing_count / non_missing_count
    has_repeated_values = unique_non_missing_count < non_missing_count

    mostly_missing = missing_ratio >= MOSTLY_MISSING_THRESHOLD
    near_row_unique = (
        unique_row_ratio >= NEAR_ROW_UNIQUE_ROW_THRESHOLD
        or unique_non_missing_ratio >= NEAR_ROW_UNIQUE_NON_MISSING_THRESHOLD
    )

    return bool(has_repeated_values and not mostly_missing and not near_row_unique)


def diagnose_column(data: pd.DataFrame, column: str) -> dict[str, Any]:
    """Compute diagnostics for one neutral grouping column."""
    row_count = int(len(data))
    exists = column in data.columns

    if not exists:
        return {
            "exists": False,
            "missing_count": row_count,
            "non_missing_count": 0,
            "unique_non_missing_count": 0,
            "top_20_values": [],
            "useful_for_grouping": False,
        }

    values = normalized_text(data[column])
    non_missing = values[values.ne("")]
    missing_count = int(row_count - len(non_missing))
    unique_non_missing_count = int(non_missing.nunique(dropna=True))

    top_values = [
        {"value_id": value_id(str(value)), "count": int(count)}
        for value, count in non_missing.value_counts().head(20).items()
    ]

    return {
        "exists": True,
        "missing_count": missing_count,
        "non_missing_count": int(len(non_missing)),
        "unique_non_missing_count": unique_non_missing_count,
        "top_20_values": top_values,
        "useful_for_grouping": useful_for_grouping(
            row_count=row_count,
            exists=True,
            missing_count=missing_count,
            non_missing_count=int(len(non_missing)),
            unique_non_missing_count=unique_non_missing_count,
        ),
    }


def build_diagnostics(data: pd.DataFrame) -> dict[str, Any]:
    """Build the full diagnostics payload."""
    return {
        "input_path": str(INPUT_PATH.relative_to(PROJECT_ROOT)),
        "row_count": int(len(data)),
        "value_display": "deterministic neutral value_id",
        "thresholds": {
            "mostly_missing_ratio": MOSTLY_MISSING_THRESHOLD,
            "near_row_unique_row_ratio": NEAR_ROW_UNIQUE_ROW_THRESHOLD,
            "near_row_unique_non_missing_ratio": NEAR_ROW_UNIQUE_NON_MISSING_THRESHOLD,
        },
        "columns": {
            column: diagnose_column(data, column) for column in GROUP_COLUMNS
        },
    }


def yes_no(value: bool) -> str:
    """Format booleans as yes/no."""
    return "yes" if value else "no"


def true_false(value: bool) -> str:
    """Format booleans as true/false."""
    return "true" if value else "false"


def build_report(diagnostics: dict[str, Any]) -> str:
    """Build the Markdown diagnostics report."""
    lines = [
        "# Neutral Group Column Diagnostics",
        "",
        f"Input file: `{diagnostics['input_path']}`",
        f"Rows: {diagnostics['row_count']}",
        "",
        "Observed values are shown as deterministic neutral value IDs.",
        "",
    ]

    for column, stats in diagnostics["columns"].items():
        lines.extend(
            [
                f"## {column}",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| exists | {yes_no(stats['exists'])} |",
                f"| missing count | {stats['missing_count']} |",
                f"| non-missing count | {stats['non_missing_count']} |",
                f"| unique non-missing count | {stats['unique_non_missing_count']} |",
                (
                    f"| useful_for_grouping | "
                    f"{true_false(stats['useful_for_grouping'])} |"
                ),
                "",
                "| Top value ID | Count |",
                "|---|---:|",
            ]
        )
        if stats["top_20_values"]:
            for item in stats["top_20_values"]:
                lines.append(f"| {item['value_id']} | {item['count']} |")
        else:
            lines.append("| n/a | 0 |")
        lines.append("")

    return "\n".join(lines)


def compact_top_values(items: list[dict[str, Any]]) -> str:
    """Format top values compactly for terminal output."""
    if not items:
        return "n/a"
    return ", ".join(f"{item['value_id']}:{item['count']}" for item in items)


def print_diagnostics(diagnostics: dict[str, Any]) -> None:
    """Print compact neutral diagnostics."""
    print("group diagnostics")
    print(f"rows: {diagnostics['row_count']}")
    for column, stats in diagnostics["columns"].items():
        print(
            f"{column}: exists={yes_no(stats['exists'])} "
            f"missing={stats['missing_count']} "
            f"non_missing={stats['non_missing_count']} "
            f"unique_non_missing={stats['unique_non_missing_count']} "
            f"useful_for_grouping={true_false(stats['useful_for_grouping'])} "
            f"top_20={compact_top_values(stats['top_20_values'])}"
        )


def main() -> None:
    """Write neutral grouping diagnostics."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    data = read_csv(INPUT_PATH)
    diagnostics = build_diagnostics(data)

    REPORT_PATH.write_text(build_report(diagnostics), encoding="utf-8")
    METRICS_PATH.write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print_diagnostics(diagnostics)
    print(f"Wrote {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {METRICS_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
