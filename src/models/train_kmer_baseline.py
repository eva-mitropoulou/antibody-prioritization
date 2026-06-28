"""Train generic k-mer logistic-regression baselines for sequence classification.

This script trains standard supervised classifiers on existing labeled sequence
strings. It does not generate, design, mutate, optimize, rank, or propose
biological sequences.

Run from the project root:

    python src/models/train_kmer_baseline.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
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
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "covabdab_neutralisation_ml.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "kmer_baseline_report.md"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics" / "kmer_baseline_metrics.json"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
MODEL_DIR = PROJECT_ROOT / "models"

HEAVY_MODEL_PATH = MODEL_DIR / "kmer_logreg_heavy_only.joblib"
PAIR_MODEL_PATH = MODEL_DIR / "kmer_logreg_pair_text.joblib"
HEAVY_CM_PATH = FIGURE_DIR / "kmer_baseline_confusion_matrix_heavy_only.png"
PAIR_CM_PATH = FIGURE_DIR / "kmer_baseline_confusion_matrix_pair_text.png"
ROC_PATH = FIGURE_DIR / "kmer_baseline_roc_curve.png"
PR_PATH = FIGURE_DIR / "kmer_baseline_pr_curve.png"

RANDOM_STATE = 42
TEST_SIZE = 0.2
REQUIRED_COLUMNS = ["sequence_heavy_only", "sequence_pair_text", "label"]


MODEL_CONFIGS = {
    "heavy_only": {
        "display_name": "Heavy-only k-mer logistic regression",
        "input_column": "sequence_heavy_only",
        "model_path": HEAVY_MODEL_PATH,
        "confusion_matrix_path": HEAVY_CM_PATH,
    },
    "pair_text": {
        "display_name": "Pair-text k-mer logistic regression",
        "input_column": "sequence_pair_text",
        "model_path": PAIR_MODEL_PATH,
        "confusion_matrix_path": PAIR_CM_PATH,
    },
}


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Fail early if the expected supervised input columns are unavailable."""
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise KeyError(f"Missing required column(s): {missing_text}")


def prepare_input(df: pd.DataFrame) -> pd.DataFrame:
    """Validate labels and convert model input columns to plain strings."""
    require_columns(df, REQUIRED_COLUMNS)

    data = df[REQUIRED_COLUMNS].copy()
    data = data.dropna(subset=["label"]).copy()
    data["label"] = data["label"].astype(int)

    unexpected_labels = sorted(set(data["label"]) - {0, 1})
    if unexpected_labels:
        raise ValueError(f"Expected binary labels 0/1, found: {unexpected_labels}")

    label_counts = data["label"].value_counts()
    if len(label_counts) != 2 or label_counts.min() < 2:
        raise ValueError("Stratified split requires at least two rows for each label.")

    for column in ["sequence_heavy_only", "sequence_pair_text"]:
        data[column] = data[column].fillna("").astype(str)
        empty_count = int(data[column].str.len().eq(0).sum())
        if empty_count:
            raise ValueError(f"Column {column} has {empty_count} empty sequence strings.")

    return data


def make_pipeline() -> Pipeline:
    """Create the required k-mer TF-IDF and logistic-regression pipeline."""
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


def positive_scores(model: Pipeline | DummyClassifier, values: pd.Series) -> np.ndarray:
    """Return scores for label 1 from estimators with binary predict_proba output."""
    if not hasattr(model, "predict_proba"):
        raise TypeError("Estimator must expose predict_proba for ROC and PR metrics.")

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


def train_and_evaluate_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    input_column: str,
) -> tuple[Pipeline, dict[str, Any], np.ndarray]:
    """Fit one configured k-mer model and return the estimator and metrics."""
    model = make_pipeline()
    model.fit(train_df[input_column], train_df["label"])

    y_pred = model.predict(test_df[input_column])
    y_score = positive_scores(model, test_df[input_column])
    metrics = metric_dict(test_df["label"], y_pred, y_score)

    return model, metrics, y_score


def train_majority_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[DummyClassifier, dict[str, Any], np.ndarray]:
    """Fit and evaluate a majority-class classifier on the shared split."""
    baseline = DummyClassifier(strategy="most_frequent")
    placeholder_train = np.zeros((len(train_df), 1))
    placeholder_test = np.zeros((len(test_df), 1))
    baseline.fit(placeholder_train, train_df["label"])

    y_pred = baseline.predict(placeholder_test)
    y_score = positive_scores(baseline, placeholder_test)
    metrics = metric_dict(test_df["label"], y_pred, y_score)

    return baseline, metrics, y_score


def save_confusion_matrix(
    matrix: list[list[int]],
    title: str,
    output_path: Path,
) -> None:
    """Save a labeled confusion-matrix figure."""
    display = ConfusionMatrixDisplay(
        confusion_matrix=np.asarray(matrix),
        display_labels=["0", "1"],
    )
    display.plot(cmap="Blues", values_format="d", colorbar=False)
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_roc_curve(
    y_true: pd.Series,
    score_by_model: dict[str, np.ndarray],
    metrics_by_model: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    """Save a combined ROC curve for all evaluated classifiers."""
    plt.figure(figsize=(6, 5))

    for model_key, scores in score_by_model.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(y_true, scores)
        auc = metrics_by_model[model_key]["roc_auc"]
        label = f"{model_key} (AUC={auc:.3f})"
        plt.plot(false_positive_rate, true_positive_rate, label=label)

    plt.plot([0, 1], [0, 1], linestyle="--", color="0.5", label="chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_pr_curve(
    y_true: pd.Series,
    score_by_model: dict[str, np.ndarray],
    metrics_by_model: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    """Save a combined precision-recall curve for all evaluated classifiers."""
    plt.figure(figsize=(6, 5))

    prevalence = float(np.mean(y_true))
    plt.axhline(
        prevalence,
        linestyle="--",
        color="0.5",
        label=f"prevalence={prevalence:.3f}",
    )

    for model_key, scores in score_by_model.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        average_precision = metrics_by_model[model_key]["average_precision"]
        label = f"{model_key} (AP={average_precision:.3f})"
        plt.plot(recall, precision, label=label)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def format_metric_table(metrics_by_model: dict[str, dict[str, Any]]) -> list[str]:
    """Format classifier metrics as Markdown table rows."""
    lines = [
        (
            "| Model | Accuracy | Balanced accuracy | Precision | Recall | F1 | "
            "ROC-AUC | Average precision |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    metric_order = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    ]

    for model_key, metrics in metrics_by_model.items():
        values = " | ".join(f"{metrics[name]:.4f}" for name in metric_order)
        lines.append(f"| {model_key} | {values} |")

    return lines


def format_confusion_matrix(model_key: str, matrix: list[list[int]]) -> list[str]:
    """Format one confusion matrix as a compact Markdown table."""
    return [
        f"### {model_key}",
        "",
        "| True label | Predicted 0 | Predicted 1 |",
        "|---|---:|---:|",
        f"| 0 | {matrix[0][0]} | {matrix[0][1]} |",
        f"| 1 | {matrix[1][0]} | {matrix[1][1]} |",
        "",
    ]


def build_report(
    row_count: int,
    label_counts: dict[int, int],
    metrics_by_model: dict[str, dict[str, Any]],
) -> str:
    """Build the Markdown report for the generic supervised baseline."""
    lines = [
        "# K-mer Sequence Classification Baseline",
        "",
        f"Input file: `{INPUT_PATH.relative_to(PROJECT_ROOT)}`",
        "",
        "This report describes a generic supervised sequence-classification baseline",
        "trained on an existing labeled table. The workflow preserves source sequence",
        "fields and produces aggregate benchmark outputs.",
        "",
        "## Data",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Rows | {row_count} |",
        f"| Label 0 count | {label_counts.get(0, 0)} |",
        f"| Label 1 count | {label_counts.get(1, 0)} |",
        "",
        "## Split",
        "",
        "| Setting | Value |",
        "|---|---:|",
        f"| test_size | {TEST_SIZE} |",
        f"| random_state | {RANDOM_STATE} |",
        "| stratify | label |",
        "",
        "## Model",
        "",
        "- `TfidfVectorizer(analyzer=\"char\", ngram_range=(3, 5), min_df=2)`",
        "- `LogisticRegression(max_iter=5000, class_weight=\"balanced\")`",
        "- Majority-class baseline: `DummyClassifier(strategy=\"most_frequent\")`",
        "",
        "## Metrics",
        "",
    ]

    lines.extend(format_metric_table(metrics_by_model))
    lines.extend(["", "## Confusion Matrices", ""])

    for model_key, metrics in metrics_by_model.items():
        lines.extend(format_confusion_matrix(model_key, metrics["confusion_matrix"]))

    lines.extend(
        [
            "## Artifacts",
            "",
            f"- `{METRICS_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{HEAVY_CM_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{PAIR_CM_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{ROC_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{PR_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{HEAVY_MODEL_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{PAIR_MODEL_PATH.relative_to(PROJECT_ROOT)}`",
            "",
        ]
    )

    return "\n".join(lines)


def print_metrics(name: str, metrics: dict[str, Any]) -> None:
    """Print metrics and confusion matrix in a readable terminal format."""
    scalar_names = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    ]

    print(f"\n{name} metrics")
    for metric_name in scalar_names:
        print(f"{metric_name}: {metrics[metric_name]:.4f}")
    print("confusion_matrix:")
    for row in metrics["confusion_matrix"]:
        print(row)


def main() -> None:
    """Train, evaluate, and save the generic sequence-classification baselines."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    raw_data = pd.read_csv(INPUT_PATH)
    data = prepare_input(raw_data)

    train_df, test_df = train_test_split(
        data,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data["label"],
    )

    metrics_by_model: dict[str, dict[str, Any]] = {}
    score_by_model: dict[str, np.ndarray] = {}

    _, majority_metrics, majority_scores = train_majority_baseline(train_df, test_df)
    metrics_by_model["majority_baseline"] = majority_metrics
    score_by_model["majority_baseline"] = majority_scores

    for model_key, config in MODEL_CONFIGS.items():
        model, metrics, scores = train_and_evaluate_model(
            train_df=train_df,
            test_df=test_df,
            input_column=config["input_column"],
        )
        metrics_by_model[model_key] = metrics
        score_by_model[model_key] = scores

        joblib.dump(model, config["model_path"])
        save_confusion_matrix(
            metrics["confusion_matrix"],
            title=config["display_name"],
            output_path=config["confusion_matrix_path"],
        )

    save_roc_curve(
        test_df["label"],
        score_by_model,
        metrics_by_model,
        ROC_PATH,
    )
    save_pr_curve(
        test_df["label"],
        score_by_model,
        metrics_by_model,
        PR_PATH,
    )

    label_counts = {
        int(label): int(count)
        for label, count in data["label"].value_counts().sort_index().items()
    }
    metrics_payload = {
        "input_path": str(INPUT_PATH.relative_to(PROJECT_ROOT)),
        "row_count": int(len(data)),
        "label_counts": {str(label): count for label, count in label_counts.items()},
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratify": "label",
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
        },
        "models": metrics_by_model,
        "artifacts": {
            "report": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
            "metrics_json": str(METRICS_PATH.relative_to(PROJECT_ROOT)),
            "heavy_only_confusion_matrix": str(
                HEAVY_CM_PATH.relative_to(PROJECT_ROOT)
            ),
            "pair_text_confusion_matrix": str(PAIR_CM_PATH.relative_to(PROJECT_ROOT)),
            "roc_curve": str(ROC_PATH.relative_to(PROJECT_ROOT)),
            "pr_curve": str(PR_PATH.relative_to(PROJECT_ROOT)),
            "heavy_only_model": str(HEAVY_MODEL_PATH.relative_to(PROJECT_ROOT)),
            "pair_text_model": str(PAIR_MODEL_PATH.relative_to(PROJECT_ROOT)),
        },
    }

    METRICS_PATH.write_text(
        json.dumps(metrics_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        build_report(
            row_count=int(len(data)),
            label_counts=label_counts,
            metrics_by_model=metrics_by_model,
        ),
        encoding="utf-8",
    )

    print_metrics("Heavy-only", metrics_by_model["heavy_only"])
    print_metrics("Pair-text", metrics_by_model["pair_text"])
    print("\nConfusion matrices")
    print(f"heavy_only: {metrics_by_model['heavy_only']['confusion_matrix']}")
    print(f"pair_text: {metrics_by_model['pair_text']['confusion_matrix']}")


if __name__ == "__main__":
    main()
