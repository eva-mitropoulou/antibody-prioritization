"""Run matched compact k-mer benchmark audits on existing records."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/antibody_prioritization_mplconfig")
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from _safe_analysis_utils import (
    PROJECT_ROOT,
    compact_model_text,
    compact_pair_from_columns,
    label_counts_dict,
    read_csv_text,
    relpath,
    write_json,
    write_text,
)


STRICT_PATH = PROJECT_ROOT / "data" / "processed" / "neutral_sequence_classification_ml.csv"
ANNOTATED_PATH = PROJECT_ROOT / "data" / "processed" / "bioaware_paired_cdr_annotated.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "matched_kmer_benchmark_audit.md"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics" / "matched_kmer_benchmark_audit.json"
ROC_FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "matched_kmer_roc_auc.png"
PR_FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "matched_kmer_pr_auc.png"

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_SPLIT_ATTEMPTS = 100
GROUP_COLUMNS = ["group_feature_v", "group_feature_cdr3", "sequence_key"]


def make_pipeline() -> Pipeline:
    """Create the requested compact character k-mer model."""
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2)),
            ("classifier", LogisticRegression(max_iter=5000, class_weight="balanced")),
        ]
    )


def positive_scores(model: Pipeline | DummyClassifier, values: pd.Series | np.ndarray) -> np.ndarray:
    """Return positive-class probabilities."""
    class_list = list(model.classes_)
    positive_index = class_list.index(1)
    return model.predict_proba(values)[:, positive_index]


def metric_dict(y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    """Compute scalar metrics."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def compact_whole_pair(data: pd.DataFrame) -> pd.Series:
    """Return compact whole-pair input text."""
    for column in ["whole_pair_kmer_text", "sequence_pair_text"]:
        if column in data.columns:
            values = data[column].map(compact_model_text)
            if values.str.len().gt(0).all():
                return values
    return compact_pair_from_columns(data).map(compact_model_text)


def compact_region_only(data: pd.DataFrame) -> pd.Series:
    """Return compact region-only text from existing annotation fields."""
    for column in ["all_cdr_kmer_text", "cdrh3_cdrl3_kmer_text"]:
        if column in data.columns:
            values = data[column].map(compact_model_text)
            if values.str.len().gt(0).any():
                return values
    region_columns = [
        column
        for column in ["cdrh1_seq", "cdrh2_seq", "cdrh3_seq", "cdrl1_seq", "cdrl2_seq", "cdrl3_seq"]
        if column in data.columns
    ]
    if not region_columns:
        return pd.Series([""] * len(data), index=data.index)
    combined = data[region_columns].fillna("").astype(str).agg("|".join, axis=1)
    return combined.map(compact_model_text)


def length_summary(values: pd.Series) -> dict[str, float | int | None]:
    """Summarize compact input lengths without exposing strings."""
    lengths = values.fillna("").astype(str).str.len()
    if len(lengths) == 0:
        return {"min": None, "mean": None, "median": None, "max": None, "empty_count": 0}
    return {
        "min": int(lengths.min()),
        "mean": float(lengths.mean()),
        "median": float(lengths.median()),
        "max": int(lengths.max()),
        "empty_count": int(lengths.eq(0).sum()),
    }


def choose_group_column(data: pd.DataFrame) -> str:
    """Choose an available grouping column for grouped validation."""
    for column in GROUP_COLUMNS:
        if column not in data.columns:
            continue
        values = data[column].fillna("").astype(str).str.strip()
        if values.nunique(dropna=True) > 1:
            return column
    raise ValueError("No usable group column was available for grouped split.")


def grouped_split(data: pd.DataFrame, group_column: str) -> tuple[pd.Index, pd.Index, pd.Series, int]:
    """Create a grouped split with both labels represented and no group overlap."""
    groups = data[group_column].fillna("").astype(str).str.strip()
    groups = groups.mask(groups.eq(""), "__missing_group__")
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    last_error = "no_valid_grouped_split"
    for offset in range(MAX_SPLIT_ATTEMPTS):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE + offset,
        )
        train_pos, test_pos = next(splitter.split(data, data["label"], groups=groups))
        train_idx = data.index[train_pos]
        test_idx = data.index[test_pos]
        if data.loc[train_idx, "label"].nunique() != 2:
            last_error = "train_split_single_label"
            continue
        if data.loc[test_idx, "label"].nunique() != 2:
            last_error = "test_split_single_label"
            continue
        train_groups = set(groups.loc[train_idx])
        test_groups = set(groups.loc[test_idx])
        if train_groups & test_groups:
            last_error = "group_overlap"
            continue
        return train_idx, test_idx, groups, RANDOM_STATE + offset
    raise ValueError(last_error)


def prepare_block(path: Path, block_id: str, paired_only: bool) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load and prepare one matched benchmark block."""
    data = read_csv_text(path)
    if "label" not in data.columns:
        raise ValueError(f"{relpath(path)} is missing label column.")
    data = data.copy()
    data["label"] = pd.to_numeric(data["label"], errors="coerce")
    data = data[data["label"].isin([0, 1])].copy()
    data["label"] = data["label"].astype(int)
    if paired_only and "has_light_bool" in data.columns:
        paired = data["has_light_bool"].fillna("").astype(str).str.lower().isin({"true", "1", "yes"})
        data = data[paired].copy()
    data["whole_pair_compact_kmer"] = compact_whole_pair(data)
    data["region_only_compact_kmer"] = compact_region_only(data)
    data["whole_pair_plus_region_compact_kmer"] = (
        data["whole_pair_compact_kmer"] + "|REGION|" + data["region_only_compact_kmer"]
    )
    input_columns = {
        "whole_pair_compact_kmer": "whole_pair_compact_kmer",
    }
    if block_id == "paired_annotated_subset":
        input_columns.update(
            {
                "region_only_compact_kmer": "region_only_compact_kmer",
                "whole_pair_plus_region_compact_kmer": "whole_pair_plus_region_compact_kmer",
            }
        )
    required = list(input_columns.values())
    for column in required:
        data = data[data[column].fillna("").astype(str).str.len().gt(0)].copy()
    return data.reset_index(drop=True), input_columns


def evaluate_variant(train_df: pd.DataFrame, test_df: pd.DataFrame, column: str) -> dict[str, Any]:
    """Fit and evaluate one compact k-mer input."""
    model = make_pipeline()
    model.fit(train_df[column], train_df["label"])
    scores = positive_scores(model, test_df[column])
    predictions = model.predict(test_df[column])
    return metric_dict(test_df["label"], predictions, scores)


def evaluate_majority(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, Any]:
    """Evaluate a majority-class baseline on the same split."""
    model = DummyClassifier(strategy="most_frequent")
    train_x = np.zeros((len(train_df), 1))
    test_x = np.zeros((len(test_df), 1))
    model.fit(train_x, train_df["label"])
    scores = positive_scores(model, test_x)
    predictions = model.predict(test_x)
    return metric_dict(test_df["label"], predictions, scores)


def run_block(path: Path, block_id: str, description: str, paired_only: bool) -> dict[str, Any]:
    """Run one matched benchmark block."""
    data, input_columns = prepare_block(path, block_id, paired_only=paired_only)
    if data["label"].nunique() != 2:
        raise ValueError(f"{block_id} lacks both label classes.")
    group_column = choose_group_column(data)
    train_idx, test_idx, groups, split_seed = grouped_split(data, group_column)
    train_df = data.loc[train_idx].copy()
    test_df = data.loc[test_idx].copy()
    train_groups = set(groups.loc[train_idx])
    test_groups = set(groups.loc[test_idx])
    split = {
        "strategy": "GroupShuffleSplit",
        "group_column": group_column,
        "random_state": split_seed,
        "test_size": TEST_SIZE,
        "train_size": int(len(train_df)),
        "test_size_rows": int(len(test_df)),
        "train_label_counts": label_counts_dict(train_df["label"]),
        "test_label_counts": label_counts_dict(test_df["label"]),
        "train_group_count": int(len(train_groups)),
        "test_group_count": int(len(test_groups)),
        "group_overlap_count": int(len(train_groups & test_groups)),
    }
    results = {"majority_baseline": evaluate_majority(train_df, test_df), "kmer_logreg": {}}
    input_summaries = {}
    for variant, column in input_columns.items():
        results["kmer_logreg"][variant] = {
            "input_column": column,
            **evaluate_variant(train_df, test_df, column),
        }
        input_summaries[variant] = length_summary(data[column])
    return {
        "block_id": block_id,
        "description": description,
        "path": relpath(path),
        "row_subset": description,
        "row_count": int(len(data)),
        "label_counts": label_counts_dict(data["label"]),
        "split": split,
        "input_length_summaries": input_summaries,
        "results": results,
    }


def build_metrics() -> dict[str, Any]:
    """Run all matched blocks and return metrics."""
    full = run_block(
        STRICT_PATH,
        "full_strict_dataset",
        "Full strict labeled dataset; whole-pair compact k-mer input.",
        paired_only=False,
    )
    paired = run_block(
        ANNOTATED_PATH,
        "paired_annotated_subset",
        "Paired annotated subset; whole-pair, region-only, and combined compact k-mer inputs.",
        paired_only=True,
    )
    paired_results = paired["results"]["kmer_logreg"]
    whole = paired_results["whole_pair_compact_kmer"]
    region = paired_results["region_only_compact_kmer"]
    combined = paired_results["whole_pair_plus_region_compact_kmer"]
    metrics = {
        "status": "available",
        "model": {
            "vectorizer": 'TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=2)',
            "classifier": 'LogisticRegression(max_iter=5000, class_weight="balanced")',
        },
        "blocks": {
            "full_strict_dataset": full,
            "paired_annotated_subset": paired,
        },
        "region_feature_comparison": {
            "row_subset": "paired_annotated_subset",
            "split_strategy": paired["split"],
            "region_only_delta_roc_auc_vs_whole_pair": float(
                region["roc_auc"] - whole["roc_auc"]
            ),
            "region_only_delta_pr_auc_vs_whole_pair": float(
                region["average_precision"] - whole["average_precision"]
            ),
            "whole_pair_plus_region_delta_roc_auc_vs_whole_pair": float(
                combined["roc_auc"] - whole["roc_auc"]
            ),
            "whole_pair_plus_region_delta_pr_auc_vs_whole_pair": float(
                combined["average_precision"] - whole["average_precision"]
            ),
            "region_features_improved_roc_auc": bool(combined["roc_auc"] > whole["roc_auc"]),
            "region_features_improved_pr_auc": bool(
                combined["average_precision"] > whole["average_precision"]
            ),
        },
    }
    return metrics


def save_figures(metrics: dict[str, Any]) -> None:
    """Save ROC-AUC and PR-AUC comparison figures."""
    records = []
    for block_id, block in metrics["blocks"].items():
        for variant, result in block["results"]["kmer_logreg"].items():
            records.append(
                {
                    "label": f"{block_id}\n{variant}",
                    "roc_auc": result["roc_auc"],
                    "average_precision": result["average_precision"],
                }
            )
    table = pd.DataFrame(records)
    if table.empty:
        return
    for metric, path, title in [
        ("roc_auc", ROC_FIGURE_PATH, "Matched compact k-mer ROC-AUC"),
        ("average_precision", PR_FIGURE_PATH, "Matched compact k-mer PR-AUC"),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.bar(np.arange(len(table)), table[metric], color="#4C78A8")
        ax.set_xticks(np.arange(len(table)))
        ax.set_xticklabels(table["label"], rotation=20, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel(metric)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)


def format_metric(value: float) -> str:
    return f"{float(value):.4f}"


def block_report(block: dict[str, Any]) -> list[str]:
    """Format one benchmark block."""
    split = block["split"]
    lines = [
        f"## {block['block_id']}",
        "",
        block["description"],
        "",
        "| Split detail | Value |",
        "|---|---:|",
        f"| Rows | {block['row_count']} |",
        f"| Label 0 count | {block['label_counts']['0']} |",
        f"| Label 1 count | {block['label_counts']['1']} |",
        f"| Split strategy | {split['strategy']} |",
        f"| Group column | `{split['group_column']}` |",
        f"| Train rows | {split['train_size']} |",
        f"| Test rows | {split['test_size_rows']} |",
        f"| Train group count | {split['train_group_count']} |",
        f"| Test group count | {split['test_group_count']} |",
        f"| Group overlap | {split['group_overlap_count']} |",
        "",
        "### Input Length Summaries",
        "",
        "| Input | Min | Mean | Median | Max | Empty count |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant, summary in block["input_length_summaries"].items():
        lines.append(
            f"| {variant} | {summary['min']} | {summary['mean']:.2f} | "
            f"{summary['median']:.2f} | {summary['max']} | {summary['empty_count']} |"
        )
    lines.extend(
        [
            "",
            "### Metrics",
            "",
            "| Model | Input | ROC-AUC | PR-AUC | Balanced accuracy | F1 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    baseline = block["results"]["majority_baseline"]
    lines.append(
        f"| majority_baseline | n/a | {format_metric(baseline['roc_auc'])} | "
        f"{format_metric(baseline['average_precision'])} | "
        f"{format_metric(baseline['balanced_accuracy'])} | {format_metric(baseline['f1'])} |"
    )
    for variant, result in block["results"]["kmer_logreg"].items():
        lines.append(
            f"| kmer_logreg | {variant} | {format_metric(result['roc_auc'])} | "
            f"{format_metric(result['average_precision'])} | "
            f"{format_metric(result['balanced_accuracy'])} | {format_metric(result['f1'])} |"
        )
    lines.append("")
    return lines


def build_report(metrics: dict[str, Any]) -> str:
    """Build Markdown report."""
    comparison = metrics["region_feature_comparison"]
    lines = [
        "# Matched Compact K-mer Benchmark Audit",
        "",
        "All comparisons in this report use compact character k-mer inputs and",
        "grouped splits with zero train/test group overlap. Full-dataset and",
        "paired-subset results are reported separately.",
        "",
    ]
    for block in metrics["blocks"].values():
        lines.extend(block_report(block))
    lines.extend(
        [
            "## Region Feature Comparison",
            "",
            "Subset: paired annotated rows only. Split: same grouped split as the paired block.",
            "",
            "| Comparison | Delta ROC-AUC | Delta PR-AUC |",
            "|---|---:|---:|",
            (
                "| region-only minus whole-pair | "
                f"{comparison['region_only_delta_roc_auc_vs_whole_pair']:.4f} | "
                f"{comparison['region_only_delta_pr_auc_vs_whole_pair']:.4f} |"
            ),
            (
                "| whole-pair plus region minus whole-pair | "
                f"{comparison['whole_pair_plus_region_delta_roc_auc_vs_whole_pair']:.4f} | "
                f"{comparison['whole_pair_plus_region_delta_pr_auc_vs_whole_pair']:.4f} |"
            ),
            "",
            "## Artifacts",
            "",
            f"- `{METRICS_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{ROC_FIGURE_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{PR_FIGURE_PATH.relative_to(PROJECT_ROOT)}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    metrics = build_metrics()
    write_json(METRICS_PATH, metrics)
    write_text(REPORT_PATH, build_report(metrics))
    save_figures(metrics)
    comparison = metrics["region_feature_comparison"]
    print(
        "Matched k-mer audit complete: "
        f"full_rows={metrics['blocks']['full_strict_dataset']['row_count']}, "
        f"paired_rows={metrics['blocks']['paired_annotated_subset']['row_count']}, "
        f"region_pr_improved={comparison['region_features_improved_pr_auc']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
