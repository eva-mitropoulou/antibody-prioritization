"""Run neutral grouped validation for the generic k-mer sequence baseline.

This script evaluates an existing supervised TF-IDF + logistic-regression model
under random and neutral grouped train/test splits.

Run from the project root:

    python src/models/train_kmer_grouped_validation.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# Matplotlib needs a writable cache directory in this environment.
MPLCONFIGDIR = Path("/tmp") / "antibody_prioritization_mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

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
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "neutral_sequence_classification_ml.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "grouped_validation_report.md"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics" / "grouped_validation_metrics.json"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
ROC_AUC_FIGURE_PATH = FIGURE_DIR / "grouped_validation_roc_auc_comparison.png"
PR_AUC_FIGURE_PATH = FIGURE_DIR / "grouped_validation_pr_auc_comparison.png"
F1_FIGURE_PATH = FIGURE_DIR / "grouped_validation_f1_comparison.png"

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_GROUP_SPLIT_ATTEMPTS = 50

INPUT_VARIANTS = {
    "sequence_a": "sequence_a",
    "sequence_pair_text": "sequence_pair_text",
}
PRIMARY_VARIANT = "sequence_pair_text"
GROUP_COLUMNS = ["group_feature_cdr3", "group_feature_v"]
REQUIRED_COLUMNS = ["sequence_a", "sequence_pair_text", "label"]

MOSTLY_MISSING_THRESHOLD = 0.50
NEAR_ROW_UNIQUE_ROW_THRESHOLD = 0.80
NEAR_ROW_UNIQUE_NON_MISSING_THRESHOLD = 0.90


def read_csv(path: Path) -> pd.DataFrame:
    """Read the neutral CSV as text while preserving blank fields as blanks."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Fail early if required neutral input columns are unavailable."""
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise KeyError(f"Missing required neutral column(s): {missing_text}")


def normalized_text(values: pd.Series) -> pd.Series:
    """Normalize text for split checks without changing the saved table."""
    return values.fillna("").astype(str).str.strip()


def prepare_input(df: pd.DataFrame) -> pd.DataFrame:
    """Validate labels and neutral model input columns."""
    require_columns(df, REQUIRED_COLUMNS)

    available_group_columns = [column for column in GROUP_COLUMNS if column in df.columns]
    keep_columns = list(dict.fromkeys([*REQUIRED_COLUMNS, *available_group_columns]))
    data = df[keep_columns].copy()
    data = data.dropna(subset=["label"]).copy()
    data = data[normalized_text(data["label"]).ne("")].copy()
    data["label"] = data["label"].astype(int)

    unexpected_labels = sorted(set(data["label"]) - {0, 1})
    if unexpected_labels:
        raise ValueError(f"Expected binary labels 0/1, found: {unexpected_labels}")

    label_counts = data["label"].value_counts()
    if len(label_counts) != 2 or label_counts.min() < 2:
        raise ValueError("Train/test evaluation requires at least two rows per label.")

    for column in INPUT_VARIANTS.values():
        data[column] = data[column].fillna("").astype(str)
        empty_count = int(data[column].str.len().eq(0).sum())
        if empty_count:
            raise ValueError(
                f"Neutral column {column} has {empty_count} empty input strings."
            )

    return data.reset_index(drop=True)


def label_counts(labels: pd.Series) -> dict[str, int]:
    """Return stable binary label counts for JSON and Markdown output."""
    counts = labels.value_counts().sort_index()
    return {"0": int(counts.get(0, 0)), "1": int(counts.get(1, 0))}


def group_column_status(data: pd.DataFrame, column: str) -> dict[str, Any]:
    """Decide whether a neutral group column can support grouped validation."""
    row_count = int(len(data))
    if column not in data.columns:
        return {
            "exists": False,
            "missing_count": row_count,
            "non_missing_count": 0,
            "unique_non_missing_count": 0,
            "useful_for_grouping": False,
            "reason": "missing",
        }

    values = normalized_text(data[column])
    non_missing = values[values.ne("")]
    missing_count = int(row_count - len(non_missing))
    non_missing_count = int(len(non_missing))
    unique_non_missing_count = int(non_missing.nunique(dropna=True))

    if row_count == 0 or non_missing_count == 0:
        reason = "empty"
        useful = False
    else:
        missing_ratio = missing_count / row_count
        unique_row_ratio = unique_non_missing_count / row_count
        unique_non_missing_ratio = unique_non_missing_count / non_missing_count
        has_repeated_values = unique_non_missing_count < non_missing_count

        if missing_ratio >= MOSTLY_MISSING_THRESHOLD:
            reason = "mostly_empty"
            useful = False
        elif not has_repeated_values:
            reason = "no_repeated_values"
            useful = False
        elif (
            unique_row_ratio >= NEAR_ROW_UNIQUE_ROW_THRESHOLD
            or unique_non_missing_ratio >= NEAR_ROW_UNIQUE_NON_MISSING_THRESHOLD
        ):
            reason = "near_row_unique"
            useful = False
        else:
            reason = "ok"
            useful = True

    return {
        "exists": True,
        "missing_count": missing_count,
        "non_missing_count": non_missing_count,
        "unique_non_missing_count": unique_non_missing_count,
        "useful_for_grouping": useful,
        "reason": reason,
    }


def split_group_values(data: pd.DataFrame, column: str) -> pd.Series:
    """Return group values without using row-level fallback IDs."""
    values = normalized_text(data[column]).copy()
    values.loc[values.eq("")] = "__missing_group__"
    return values


def grouped_split(
    data: pd.DataFrame,
    group_column: str,
) -> tuple[pd.Index, pd.Index, pd.Series, int]:
    """Create a meaningful grouped train/test split or raise ValueError."""
    groups = split_group_values(data, group_column)
    last_error = "no_valid_grouped_split"

    for offset in range(MAX_GROUP_SPLIT_ATTEMPTS):
        split_random_state = RANDOM_STATE + offset
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=TEST_SIZE,
            random_state=split_random_state,
        )
        try:
            train_positions, test_positions = next(
                splitter.split(data, data["label"], groups=groups)
            )
        except ValueError as exc:
            last_error = str(exc)
            break

        train_idx = data.index[train_positions]
        test_idx = data.index[test_positions]
        train_labels = data.loc[train_idx, "label"]
        test_labels = data.loc[test_idx, "label"]

        if train_labels.nunique() != 2:
            last_error = "train_split_single_label"
            continue
        if test_labels.nunique() != 2:
            last_error = "test_split_single_label"
            continue

        train_group_set = set(groups.loc[train_idx].astype(str))
        test_group_set = set(groups.loc[test_idx].astype(str))
        if train_group_set & test_group_set:
            last_error = "group_overlap"
            continue

        return train_idx, test_idx, groups, split_random_state

    raise ValueError(last_error)


def random_split(data: pd.DataFrame) -> tuple[pd.Index, pd.Index]:
    """Create the random stratified train/test split."""
    train_idx, test_idx = train_test_split(
        data.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data["label"],
    )
    return pd.Index(train_idx), pd.Index(test_idx)


def split_diagnostics(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    groups: pd.Series | None = None,
) -> dict[str, Any]:
    """Summarize split sizes, label balance, and optional group overlap."""
    diagnostics: dict[str, Any] = {
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "train_label_counts": label_counts(train_df["label"]),
        "test_label_counts": label_counts(test_df["label"]),
        "train_group_count": None,
        "test_group_count": None,
        "group_overlap_count": None,
    }

    if groups is not None:
        train_group_set = set(groups.loc[train_df.index].astype(str))
        test_group_set = set(groups.loc[test_df.index].astype(str))
        diagnostics.update(
            {
                "train_group_count": int(len(train_group_set)),
                "test_group_count": int(len(test_group_set)),
                "group_overlap_count": int(len(train_group_set & test_group_set)),
            }
        )

    return diagnostics


def make_pipeline() -> Pipeline:
    """Create the k-mer TF-IDF and logistic-regression pipeline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=5000, class_weight="balanced"),
            ),
        ]
    )


def positive_scores(model: Pipeline | DummyClassifier, values: pd.Series | np.ndarray) -> np.ndarray:
    """Return scores for label 1 from estimators with binary predict_proba output."""
    class_list = list(model.classes_)
    if 1 not in class_list:
        raise ValueError("Estimator was not fitted with positive class label 1.")

    positive_index = class_list.index(1)
    return model.predict_proba(values)[:, positive_index]


def metric_dict(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, Any]:
    """Compute scalar classification metrics and the confusion matrix."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    has_both_labels = y_true.nunique() == 2

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if has_both_labels else None,
        "average_precision": (
            float(average_precision_score(y_true, y_score)) if has_both_labels else None
        ),
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def train_and_evaluate_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    input_column: str,
) -> dict[str, Any]:
    """Fit one k-mer model and return its metrics."""
    model = make_pipeline()
    model.fit(train_df[input_column], train_df["label"])

    y_pred = model.predict(test_df[input_column])
    y_score = positive_scores(model, test_df[input_column])

    return metric_dict(test_df["label"], y_pred, y_score)


def evaluate_majority_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    """Fit and evaluate a majority-class baseline for one test split."""
    baseline = DummyClassifier(strategy="most_frequent")
    placeholder_train = np.zeros((len(train_df), 1))
    placeholder_test = np.zeros((len(test_df), 1))
    baseline.fit(placeholder_train, train_df["label"])

    y_pred = baseline.predict(placeholder_test)
    y_score = positive_scores(baseline, placeholder_test)

    return metric_dict(test_df["label"], y_pred, y_score)


def evaluate_valid_split(
    data: pd.DataFrame,
    train_idx: pd.Index,
    test_idx: pd.Index,
    groups: pd.Series | None,
) -> dict[str, Any]:
    """Evaluate all configured inputs on one valid split."""
    train_df = data.loc[train_idx].copy()
    test_df = data.loc[test_idx].copy()
    diagnostics = split_diagnostics(train_df, test_df, groups)

    split_results: dict[str, Any] = {
        "split": diagnostics,
        "majority_baseline": {
            **diagnostics,
            **evaluate_majority_baseline(train_df, test_df),
        },
        "kmer_logreg": {},
    }

    for variant_name, input_column in INPUT_VARIANTS.items():
        split_results["kmer_logreg"][variant_name] = {
            "input_column": input_column,
            **diagnostics,
            **train_and_evaluate_model(train_df, test_df, input_column),
        }

    return split_results


def evaluate_all(data: pd.DataFrame) -> dict[str, Any]:
    """Evaluate random validation and valid neutral grouped validations."""
    results: dict[str, Any] = {}
    group_status = {
        column: group_column_status(data, column) for column in GROUP_COLUMNS
    }

    random_train_idx, random_test_idx = random_split(data)
    results["random"] = {
        "valid": True,
        "meaningful": True,
        "reason": "ok",
        "group_column": None,
        "split_random_state": RANDOM_STATE,
        **evaluate_valid_split(
            data=data,
            train_idx=random_train_idx,
            test_idx=random_test_idx,
            groups=None,
        ),
    }

    for column in GROUP_COLUMNS:
        status = group_status[column]
        if not status["useful_for_grouping"]:
            results[column] = {
                "valid": False,
                "meaningful": False,
                "reason": status["reason"],
                "group_column": column,
                "group_column_status": status,
            }
            continue

        try:
            train_idx, test_idx, groups, split_random_state = grouped_split(data, column)
        except ValueError as exc:
            results[column] = {
                "valid": False,
                "meaningful": False,
                "reason": str(exc),
                "group_column": column,
                "group_column_status": status,
            }
            continue

        split_results = evaluate_valid_split(
            data=data,
            train_idx=train_idx,
            test_idx=test_idx,
            groups=groups,
        )
        overlap = split_results["split"]["group_overlap_count"]
        results[column] = {
            "valid": overlap == 0,
            "meaningful": overlap == 0,
            "reason": "ok" if overlap == 0 else "group_overlap",
            "group_column": column,
            "group_column_status": status,
            "split_random_state": split_random_state,
            **split_results,
        }

    return results


def metric_value(value: float | None) -> float:
    """Convert optional metrics to NaN for plotting."""
    return np.nan if value is None else float(value)


def flatten_kmer_results(results: dict[str, Any]) -> pd.DataFrame:
    """Create a tabular view of valid k-mer metrics for reports and figures."""
    records = []

    for split_name, split_results in results.items():
        if not split_results.get("valid"):
            continue
        for variant_name, metrics in split_results["kmer_logreg"].items():
            records.append(
                {
                    "split_strategy": split_name,
                    "input_variant": variant_name,
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "roc_auc": metric_value(metrics["roc_auc"]),
                    "average_precision": metric_value(metrics["average_precision"]),
                }
            )

    return pd.DataFrame.from_records(records)


def save_metric_comparison(
    kmer_table: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Save a grouped bar chart comparing one metric across valid split strategies."""
    split_order = [
        split_name
        for split_name in ["random", *GROUP_COLUMNS]
        if split_name in set(kmer_table["split_strategy"])
    ]
    variant_order = list(INPUT_VARIANTS)
    x = np.arange(len(split_order))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for offset, variant_name in zip([-width / 2, width / 2], variant_order):
        values = []
        for split_name in split_order:
            row = kmer_table[
                (kmer_table["split_strategy"] == split_name)
                & (kmer_table["input_variant"] == variant_name)
            ]
            values.append(float(row.iloc[0][metric]) if not row.empty else np.nan)

        ax.bar(x + offset, values, width=width, label=variant_name)

    ax.set_xticks(x)
    ax.set_xticklabels(split_order, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_figures(results: dict[str, Any]) -> None:
    """Save comparison figures for valid k-mer model metrics."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    kmer_table = flatten_kmer_results(results)
    if kmer_table.empty:
        return

    save_metric_comparison(
        kmer_table,
        metric="roc_auc",
        title="Grouped Validation ROC-AUC Comparison",
        ylabel="ROC-AUC",
        output_path=ROC_AUC_FIGURE_PATH,
    )
    save_metric_comparison(
        kmer_table,
        metric="average_precision",
        title="Grouped Validation PR-AUC Comparison",
        ylabel="Average precision / PR-AUC",
        output_path=PR_AUC_FIGURE_PATH,
    )
    save_metric_comparison(
        kmer_table,
        metric="f1",
        title="Grouped Validation F1 Comparison",
        ylabel="F1",
        output_path=F1_FIGURE_PATH,
    )


def format_metric(value: float | None) -> str:
    """Format optional metric values for Markdown and terminal output."""
    return "n/a" if value is None else f"{value:.4f}"


def format_counts(counts: dict[str, int]) -> str:
    """Format label counts compactly."""
    return f"0={counts['0']}, 1={counts['1']}"


def format_nullable(value: Any) -> str:
    """Format nullable diagnostics."""
    return "n/a" if value is None else str(value)


def true_false(value: bool) -> str:
    """Format booleans as true/false."""
    return "true" if value else "false"


def format_status_rows(results: dict[str, Any]) -> list[str]:
    """Format split validity and group-overlap diagnostics."""
    lines = [
        (
            "| Split | Group column | Valid | Meaningful | Reason | Train groups | "
            "Test groups | Group overlap |"
        ),
        "|---|---|---:|---:|---|---:|---:|---:|",
    ]

    for split_name, split_results in results.items():
        split = split_results.get("split", {})
        lines.append(
            f"| {split_name} | {format_nullable(split_results.get('group_column'))} | "
            f"{true_false(bool(split_results.get('valid')))} | "
            f"{true_false(bool(split_results.get('meaningful')))} | "
            f"{split_results.get('reason', 'n/a')} | "
            f"{format_nullable(split.get('train_group_count'))} | "
            f"{format_nullable(split.get('test_group_count'))} | "
            f"{format_nullable(split.get('group_overlap_count'))} |"
        )

    return lines


def format_model_rows(results: dict[str, Any]) -> list[str]:
    """Format valid k-mer and majority-baseline metrics as Markdown rows."""
    lines = [
        (
            "| Split | Model | Input | Train size | Test size | Train labels | "
            "Test labels | Train groups | Test groups | Group overlap | Accuracy | "
            "Balanced accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |"
        ),
        "|---|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for split_name, split_results in results.items():
        if not split_results.get("valid"):
            continue

        baseline = split_results["majority_baseline"]
        lines.append(format_one_row(split_name, "majority_baseline", "n/a", baseline))

        for variant_name, metrics in split_results["kmer_logreg"].items():
            lines.append(format_one_row(split_name, "kmer_logreg", variant_name, metrics))

    return lines


def format_one_row(
    split_name: str,
    model_name: str,
    input_variant: str,
    metrics: dict[str, Any],
) -> str:
    """Format one metric row for the Markdown report."""
    return (
        f"| {split_name} | {model_name} | {input_variant} | "
        f"{metrics['train_size']} | {metrics['test_size']} | "
        f"{format_counts(metrics['train_label_counts'])} | "
        f"{format_counts(metrics['test_label_counts'])} | "
        f"{format_nullable(metrics['train_group_count'])} | "
        f"{format_nullable(metrics['test_group_count'])} | "
        f"{format_nullable(metrics['group_overlap_count'])} | "
        f"{metrics['accuracy']:.4f} | "
        f"{metrics['balanced_accuracy']:.4f} | {metrics['precision']:.4f} | "
        f"{metrics['recall']:.4f} | {metrics['f1']:.4f} | "
        f"{format_metric(metrics['roc_auc'])} | "
        f"{format_metric(metrics['average_precision'])} |"
    )


def format_confusion_matrices(results: dict[str, Any]) -> list[str]:
    """Format primary-input confusion matrices for the Markdown report."""
    lines = ["## Confusion Matrices", ""]
    for split_name, split_results in results.items():
        if not split_results.get("valid"):
            continue
        matrix = split_results["kmer_logreg"][PRIMARY_VARIANT]["confusion_matrix"]
        lines.extend(
            [
                f"### {split_name}",
                "",
                "| True label | Predicted 0 | Predicted 1 |",
                "|---|---:|---:|",
                f"| 0 | {matrix[0][0]} | {matrix[0][1]} |",
                f"| 1 | {matrix[1][0]} | {matrix[1][1]} |",
                "",
            ]
        )
    return lines


def build_conclusion(results: dict[str, Any]) -> list[str]:
    """Create a short neutral validation conclusion."""
    lines = ["## Validation Conclusion", ""]
    valid_group_splits = [
        split_name
        for split_name in GROUP_COLUMNS
        if results.get(split_name, {}).get("valid")
    ]
    invalid_group_splits = [
        split_name
        for split_name in GROUP_COLUMNS
        if not results.get(split_name, {}).get("valid")
    ]

    if valid_group_splits:
        lines.append(
            "At least one neutral grouped split is valid, with zero train/test "
            "group overlap."
        )
    else:
        lines.append(
            "No neutral grouped split is valid for this table, so grouped "
            "validation should not be interpreted."
        )

    if invalid_group_splits:
        reasons = ", ".join(
            f"{name}={results[name]['reason']}" for name in invalid_group_splits
        )
        lines.append(f"Skipped grouped split(s): {reasons}.")

    lines.append("")
    return lines


def build_report(data: pd.DataFrame, results: dict[str, Any]) -> str:
    """Build the grouped-validation Markdown report."""
    lines = [
        "# Neutral Grouped K-mer Validation",
        "",
        f"Input file: `{INPUT_PATH.relative_to(PROJECT_ROOT)}`",
        "",
        "This report evaluates a generic supervised sequence-classification baseline",
        "on existing labeled rows using neutral column names.",
        "",
        "## Data",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Rows | {len(data)} |",
        f"| Label 0 count | {label_counts(data['label'])['0']} |",
        f"| Label 1 count | {label_counts(data['label'])['1']} |",
        "",
        "## Model",
        "",
        "- `TfidfVectorizer(analyzer=\"char\", ngram_range=(3, 5), min_df=2)`",
        "- `LogisticRegression(max_iter=5000, class_weight=\"balanced\")`",
        "- Majority-class baseline: `DummyClassifier(strategy=\"most_frequent\")`",
        "",
        "## Split Validity",
        "",
    ]

    lines.extend(format_status_rows(results))
    lines.extend(["", "## Metrics", ""])
    lines.extend(format_model_rows(results))
    lines.extend([""])
    lines.extend(format_confusion_matrices(results))
    lines.extend(build_conclusion(results))
    lines.extend(
        [
            "## Artifacts",
            "",
            f"- `{METRICS_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{ROC_AUC_FIGURE_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{PR_AUC_FIGURE_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{F1_FIGURE_PATH.relative_to(PROJECT_ROOT)}`",
            "",
        ]
    )

    return "\n".join(lines)


def print_primary_summary(results: dict[str, Any]) -> None:
    """Print requested terminal metrics for the primary neutral input."""
    scalar_names = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    ]

    for split_name in ["random", *GROUP_COLUMNS]:
        split_results = results[split_name]
        if not split_results.get("valid"):
            print(f"\n{split_name} validation")
            print("valid: false")
            print("meaningful: false")
            print(f"reason: {split_results['reason']}")
            print("group_overlap_count: n/a")
            continue

        metrics = split_results["kmer_logreg"][PRIMARY_VARIANT]
        print(f"\n{split_name} metrics")
        print("valid: true")
        print(f"meaningful: {true_false(bool(split_results.get('meaningful')))}")
        print(f"group_column: {format_nullable(split_results.get('group_column'))}")
        print(f"train_size: {metrics['train_size']}")
        print(f"test_size: {metrics['test_size']}")
        print(f"train_label_counts: {metrics['train_label_counts']}")
        print(f"test_label_counts: {metrics['test_label_counts']}")
        print(f"train_group_count: {format_nullable(metrics['train_group_count'])}")
        print(f"test_group_count: {format_nullable(metrics['test_group_count'])}")
        print(f"group_overlap_count: {format_nullable(metrics['group_overlap_count'])}")
        for metric_name in scalar_names:
            print(f"{metric_name}: {format_metric(metrics[metric_name])}")

    print("\nconfusion matrices")
    for split_name in ["random", *GROUP_COLUMNS]:
        split_results = results[split_name]
        if not split_results.get("valid"):
            continue
        matrix = split_results["kmer_logreg"][PRIMARY_VARIANT]["confusion_matrix"]
        print(f"{split_name}: {matrix}")


def main() -> None:
    """Run neutral grouped validation and save reports, metrics, and figures."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    raw_data = read_csv(INPUT_PATH)
    data = prepare_input(raw_data)
    results = evaluate_all(data)

    metrics_payload = {
        "input_path": str(INPUT_PATH.relative_to(PROJECT_ROOT)),
        "row_count": int(len(data)),
        "label_counts": label_counts(data["label"]),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "input_variants": INPUT_VARIANTS,
        "primary_variant": PRIMARY_VARIANT,
        "group_columns": GROUP_COLUMNS,
        "group_validity_thresholds": {
            "mostly_missing_ratio": MOSTLY_MISSING_THRESHOLD,
            "near_row_unique_row_ratio": NEAR_ROW_UNIQUE_ROW_THRESHOLD,
            "near_row_unique_non_missing_ratio": NEAR_ROW_UNIQUE_NON_MISSING_THRESHOLD,
        },
        "results": results,
        "artifacts": {
            "report": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
            "metrics_json": str(METRICS_PATH.relative_to(PROJECT_ROOT)),
            "roc_auc_comparison": str(ROC_AUC_FIGURE_PATH.relative_to(PROJECT_ROOT)),
            "pr_auc_comparison": str(PR_AUC_FIGURE_PATH.relative_to(PROJECT_ROOT)),
            "f1_comparison": str(F1_FIGURE_PATH.relative_to(PROJECT_ROOT)),
        },
    }

    METRICS_PATH.write_text(
        json.dumps(metrics_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(build_report(data, results), encoding="utf-8")
    save_figures(results)
    print_primary_summary(results)


if __name__ == "__main__":
    main()
