"""Train logistic-regression baselines on cached AbLang2 embeddings.

This script is a supervised benchmark on existing labeled rows. It does not
generate, design, mutate, optimize, rank, or propose biological sequences.

Run from the project root:

    python src/models/train_embedding_baseline.py
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
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HEAVY_EMBEDDING_PATH = PROJECT_ROOT / "data" / "processed" / "embeddings_ablang2_heavy.npy"
PAIR_EMBEDDING_PATH = PROJECT_ROOT / "data" / "processed" / "embeddings_ablang2_pair.npy"
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "embeddings_ablang2_metadata.csv"
KMER_METRICS_PATH = PROJECT_ROOT / "reports" / "metrics" / "grouped_validation_metrics.json"

REPORT_PATH = PROJECT_ROOT / "reports" / "embedding_baseline_report.md"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics" / "embedding_baseline_metrics.json"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
ROC_AUC_FIGURE_PATH = FIGURE_DIR / "embedding_vs_kmer_roc_auc.png"
PR_AUC_FIGURE_PATH = FIGURE_DIR / "embedding_vs_kmer_pr_auc.png"
MODEL_DIR = PROJECT_ROOT / "models"
HEAVY_MODEL_PATH = MODEL_DIR / "embedding_logreg_heavy.joblib"
PAIR_MODEL_PATH = MODEL_DIR / "embedding_logreg_pair.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_GROUP_SPLIT_ATTEMPTS = 50
GROUP_COLUMN = "group_feature_v"
MOSTLY_MISSING_THRESHOLD = 0.50
NEAR_ROW_UNIQUE_ROW_THRESHOLD = 0.80
NEAR_ROW_UNIQUE_NON_MISSING_THRESHOLD = 0.90


def read_metadata(path: Path) -> pd.DataFrame:
    """Read embedding metadata as text while preserving blank fields as blanks."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def require_files(paths: list[Path]) -> None:
    """Fail clearly when cached embeddings are missing."""
    missing = [path for path in paths if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path.relative_to(PROJECT_ROOT)) for path in missing)
        raise FileNotFoundError(
            f"Missing embedding artifact(s): {missing_text}. "
            "Run python src/models/embed_with_ablang2.py first."
        )


def normalized_text(values: pd.Series) -> pd.Series:
    """Normalize text for split checks."""
    return values.fillna("").astype(str).str.strip()


def load_inputs() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Load metadata and cached embedding matrices."""
    require_files([HEAVY_EMBEDDING_PATH, METADATA_PATH])
    metadata = read_metadata(METADATA_PATH)
    embeddings = {"heavy": np.load(HEAVY_EMBEDDING_PATH)}

    if PAIR_EMBEDDING_PATH.exists():
        embeddings["pair"] = np.load(PAIR_EMBEDDING_PATH)

    row_count = len(metadata)
    for name, values in embeddings.items():
        if values.shape[0] != row_count:
            raise ValueError(
                f"{name} embedding rows ({values.shape[0]}) do not match "
                f"metadata rows ({row_count})."
            )

    metadata = metadata.copy()
    metadata = metadata[normalized_text(metadata["label"]).ne("")].copy()
    metadata["label"] = metadata["label"].astype(int)

    unexpected_labels = sorted(set(metadata["label"]) - {0, 1})
    if unexpected_labels:
        raise ValueError(f"Expected binary labels 0/1, found: {unexpected_labels}")

    label_counts = metadata["label"].value_counts()
    if len(label_counts) != 2 or label_counts.min() < 2:
        raise ValueError("Train/test evaluation requires at least two rows per label.")

    selected_positions = metadata.index.to_numpy()
    embeddings = {name: values[selected_positions] for name, values in embeddings.items()}
    metadata = metadata.reset_index(drop=True)

    return metadata, embeddings


def label_counts(labels: pd.Series) -> dict[str, int]:
    """Return stable binary label counts for JSON and Markdown output."""
    counts = labels.value_counts().sort_index()
    return {"0": int(counts.get(0, 0)), "1": int(counts.get(1, 0))}


def group_column_status(data: pd.DataFrame) -> dict[str, Any]:
    """Decide whether the requested neutral group column supports validation."""
    row_count = int(len(data))
    if GROUP_COLUMN not in data.columns:
        return {
            "exists": False,
            "missing_count": row_count,
            "non_missing_count": 0,
            "unique_non_missing_count": 0,
            "useful_for_grouping": False,
            "reason": "missing",
        }

    values = normalized_text(data[GROUP_COLUMN])
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


def split_group_values(data: pd.DataFrame) -> pd.Series:
    """Return group values without row-level fallback groups."""
    values = normalized_text(data[GROUP_COLUMN]).copy()
    values.loc[values.eq("")] = "__missing_group__"
    return values


def random_split(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Create a random stratified split."""
    train_idx, test_idx = train_test_split(
        np.arange(len(data)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data["label"],
    )
    return np.asarray(train_idx), np.asarray(test_idx)


def grouped_split(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.Series, int]:
    """Create a meaningful grouped split or raise ValueError."""
    groups = split_group_values(data)
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

        train_labels = data.iloc[train_positions]["label"]
        test_labels = data.iloc[test_positions]["label"]
        if train_labels.nunique() != 2:
            last_error = "train_split_single_label"
            continue
        if test_labels.nunique() != 2:
            last_error = "test_split_single_label"
            continue

        train_group_set = set(groups.iloc[train_positions].astype(str))
        test_group_set = set(groups.iloc[test_positions].astype(str))
        if train_group_set & test_group_set:
            last_error = "group_overlap"
            continue

        return (
            np.asarray(train_positions),
            np.asarray(test_positions),
            groups,
            split_random_state,
        )

    raise ValueError(last_error)


def split_diagnostics(
    data: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    groups: pd.Series | None,
) -> dict[str, Any]:
    """Summarize split size, label balance, and group overlap."""
    train_labels = data.iloc[train_idx]["label"]
    test_labels = data.iloc[test_idx]["label"]
    diagnostics: dict[str, Any] = {
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "train_label_counts": label_counts(train_labels),
        "test_label_counts": label_counts(test_labels),
        "train_group_count": None,
        "test_group_count": None,
        "group_overlap_count": None,
    }

    if groups is not None:
        train_group_set = set(groups.iloc[train_idx].astype(str))
        test_group_set = set(groups.iloc[test_idx].astype(str))
        diagnostics.update(
            {
                "train_group_count": int(len(train_group_set)),
                "test_group_count": int(len(test_group_set)),
                "group_overlap_count": int(len(train_group_set & test_group_set)),
            }
        )

    return diagnostics


def make_pipeline() -> Pipeline:
    """Create a dense-embedding logistic-regression classifier."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=5000, class_weight="balanced"),
            ),
        ]
    )


def positive_scores(model: Pipeline, values: np.ndarray) -> np.ndarray:
    """Return scores for label 1."""
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
    """Compute scalar metrics and a confusion matrix."""
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


def train_and_evaluate(
    data: pd.DataFrame,
    embeddings: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict[str, Any]:
    """Fit one embedding classifier and return metrics."""
    model = make_pipeline()
    model.fit(embeddings[train_idx], data.iloc[train_idx]["label"])

    y_true = data.iloc[test_idx]["label"]
    y_pred = model.predict(embeddings[test_idx])
    y_score = positive_scores(model, embeddings[test_idx])

    return metric_dict(y_true, y_pred, y_score)


def evaluate_split(
    data: pd.DataFrame,
    embeddings_by_model: dict[str, np.ndarray],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    groups: pd.Series | None,
) -> dict[str, Any]:
    """Evaluate all available embedding variants on one split."""
    diagnostics = split_diagnostics(data, train_idx, test_idx, groups)
    model_results = {}

    for model_name, embeddings in embeddings_by_model.items():
        model_results[model_name] = {
            "embedding_shape": list(embeddings.shape),
            **diagnostics,
            **train_and_evaluate(
                data=data,
                embeddings=embeddings,
                train_idx=train_idx,
                test_idx=test_idx,
            ),
        }

    return {
        "split": diagnostics,
        "models": model_results,
    }


def evaluate_all(
    data: pd.DataFrame,
    embeddings_by_model: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Evaluate random and group_feature_v splits."""
    results: dict[str, Any] = {}

    random_train_idx, random_test_idx = random_split(data)
    results["random"] = {
        "valid": True,
        "meaningful": True,
        "reason": "ok",
        "group_column": None,
        "split_random_state": RANDOM_STATE,
        **evaluate_split(
            data=data,
            embeddings_by_model=embeddings_by_model,
            train_idx=random_train_idx,
            test_idx=random_test_idx,
            groups=None,
        ),
    }

    status = group_column_status(data)
    if not status["useful_for_grouping"]:
        results[GROUP_COLUMN] = {
            "valid": False,
            "meaningful": False,
            "reason": status["reason"],
            "group_column": GROUP_COLUMN,
            "group_column_status": status,
        }
        return results

    try:
        train_idx, test_idx, groups, split_random_state = grouped_split(data)
    except ValueError as exc:
        results[GROUP_COLUMN] = {
            "valid": False,
            "meaningful": False,
            "reason": str(exc),
            "group_column": GROUP_COLUMN,
            "group_column_status": status,
        }
        return results

    split_results = evaluate_split(
        data=data,
        embeddings_by_model=embeddings_by_model,
        train_idx=train_idx,
        test_idx=test_idx,
        groups=groups,
    )
    overlap = split_results["split"]["group_overlap_count"]
    results[GROUP_COLUMN] = {
        "valid": overlap == 0,
        "meaningful": overlap == 0,
        "reason": "ok" if overlap == 0 else "group_overlap",
        "group_column": GROUP_COLUMN,
        "group_column_status": status,
        "split_random_state": split_random_state,
        **split_results,
    }

    return results


def save_full_data_models(
    data: pd.DataFrame,
    embeddings_by_model: dict[str, np.ndarray],
) -> None:
    """Fit final classifiers on all labeled rows for saved artifacts."""
    model_paths = {
        "heavy": HEAVY_MODEL_PATH,
        "pair": PAIR_MODEL_PATH,
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for model_name, embeddings in embeddings_by_model.items():
        model = make_pipeline()
        model.fit(embeddings, data["label"])
        joblib.dump(model, model_paths[model_name])


def load_kmer_comparison() -> dict[str, Any]:
    """Load k-mer baseline metrics if available."""
    if not KMER_METRICS_PATH.exists():
        return {"available": False, "reason": "missing"}

    payload = json.loads(KMER_METRICS_PATH.read_text(encoding="utf-8"))
    results = payload.get("results", {})
    comparison: dict[str, Any] = {"available": True, "metrics": {}}

    for split_name in ["random", GROUP_COLUMN]:
        split_results = results.get(split_name, {})
        if not split_results.get("valid"):
            continue
        model_metrics = (
            split_results.get("kmer_logreg", {})
            .get("sequence_pair_text", {})
        )
        if model_metrics:
            comparison["metrics"][split_name] = {
                "roc_auc": model_metrics.get("roc_auc"),
                "average_precision": model_metrics.get("average_precision"),
                "f1": model_metrics.get("f1"),
            }

    return comparison


def build_comparison(
    embedding_results: dict[str, Any],
    kmer_comparison: dict[str, Any],
) -> dict[str, Any]:
    """Compare embedding models against the k-mer pair-text baseline."""
    comparison = {
        "kmer_available": bool(kmer_comparison.get("available")),
        "by_split": {},
    }
    kmer_metrics = kmer_comparison.get("metrics", {})

    for split_name in ["random", GROUP_COLUMN]:
        split_results = embedding_results.get(split_name, {})
        if not split_results.get("valid"):
            continue

        split_comparison: dict[str, Any] = {
            "kmer_pair_text": kmer_metrics.get(split_name),
            "embedding": {},
        }
        for model_name, metrics in split_results["models"].items():
            current = {
                "roc_auc": metrics.get("roc_auc"),
                "average_precision": metrics.get("average_precision"),
                "f1": metrics.get("f1"),
            }
            baseline = split_comparison["kmer_pair_text"]
            if baseline:
                current["delta_roc_auc_vs_kmer"] = (
                    current["roc_auc"] - baseline["roc_auc"]
                    if current["roc_auc"] is not None
                    and baseline["roc_auc"] is not None
                    else None
                )
                current["delta_average_precision_vs_kmer"] = (
                    current["average_precision"] - baseline["average_precision"]
                    if current["average_precision"] is not None
                    and baseline["average_precision"] is not None
                    else None
                )
            split_comparison["embedding"][model_name] = current

        comparison["by_split"][split_name] = split_comparison

    return comparison


def metric_value(value: float | None) -> float:
    """Convert optional metrics to NaN for plotting."""
    return np.nan if value is None else float(value)


def plot_comparison(
    embedding_results: dict[str, Any],
    kmer_comparison: dict[str, Any],
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    """Save a bar chart comparing embedding models and k-mer metrics."""
    records = []
    kmer_metrics = kmer_comparison.get("metrics", {})

    for split_name in ["random", GROUP_COLUMN]:
        if split_name in kmer_metrics:
            records.append(
                {
                    "split": split_name,
                    "model": "kmer_pair_text",
                    "value": metric_value(kmer_metrics[split_name].get(metric)),
                }
            )

        split_results = embedding_results.get(split_name, {})
        if not split_results.get("valid"):
            continue
        for model_name, metrics in split_results["models"].items():
            records.append(
                {
                    "split": split_name,
                    "model": f"embedding_{model_name}",
                    "value": metric_value(metrics.get(metric)),
                }
            )

    if not records:
        return

    table = pd.DataFrame.from_records(records)
    split_order = [name for name in ["random", GROUP_COLUMN] if name in set(table["split"])]
    model_order = ["kmer_pair_text", "embedding_heavy", "embedding_pair"]
    width = 0.25
    x = np.arange(len(split_order))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for offset_index, model_name in enumerate(model_order):
        offset = (offset_index - 1) * width
        values = []
        for split_name in split_order:
            row = table[(table["split"] == split_name) & (table["model"] == model_name)]
            values.append(float(row.iloc[0]["value"]) if not row.empty else np.nan)
        ax.bar(x + offset, values, width=width, label=model_name)

    ax.set_xticks(x)
    ax.set_xticklabels(split_order, rotation=10, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_figures(
    embedding_results: dict[str, Any],
    kmer_comparison: dict[str, Any],
) -> None:
    """Save comparison figures."""
    plot_comparison(
        embedding_results=embedding_results,
        kmer_comparison=kmer_comparison,
        metric="roc_auc",
        ylabel="ROC-AUC",
        title="Embedding vs K-mer ROC-AUC",
        output_path=ROC_AUC_FIGURE_PATH,
    )
    plot_comparison(
        embedding_results=embedding_results,
        kmer_comparison=kmer_comparison,
        metric="average_precision",
        ylabel="Average precision / PR-AUC",
        title="Embedding vs K-mer PR-AUC",
        output_path=PR_AUC_FIGURE_PATH,
    )


def format_metric(value: float | None) -> str:
    """Format optional metrics."""
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


def format_model_rows(results: dict[str, Any]) -> list[str]:
    """Format embedding metrics as Markdown rows."""
    lines = [
        (
            "| Split | Model | Train size | Test size | Train labels | Test labels | "
            "Train groups | Test groups | Group overlap | Accuracy | Balanced accuracy | "
            "Precision | Recall | F1 | ROC-AUC | PR-AUC |"
        ),
        "|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for split_name, split_results in results.items():
        if not split_results.get("valid"):
            continue
        for model_name, metrics in split_results["models"].items():
            lines.append(
                f"| {split_name} | {model_name} | "
                f"{metrics['train_size']} | {metrics['test_size']} | "
                f"{format_counts(metrics['train_label_counts'])} | "
                f"{format_counts(metrics['test_label_counts'])} | "
                f"{format_nullable(metrics['train_group_count'])} | "
                f"{format_nullable(metrics['test_group_count'])} | "
                f"{format_nullable(metrics['group_overlap_count'])} | "
                f"{metrics['accuracy']:.4f} | {metrics['balanced_accuracy']:.4f} | "
                f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | "
                f"{metrics['f1']:.4f} | {format_metric(metrics['roc_auc'])} | "
                f"{format_metric(metrics['average_precision'])} |"
            )

    return lines


def format_confusion_matrices(results: dict[str, Any]) -> list[str]:
    """Format confusion matrices for the Markdown report."""
    lines = ["## Confusion Matrices", ""]
    for split_name, split_results in results.items():
        if not split_results.get("valid"):
            continue
        for model_name, metrics in split_results["models"].items():
            matrix = metrics["confusion_matrix"]
            lines.extend(
                [
                    f"### {split_name} / {model_name}",
                    "",
                    "| True label | Predicted 0 | Predicted 1 |",
                    "|---|---:|---:|",
                    f"| 0 | {matrix[0][0]} | {matrix[0][1]} |",
                    f"| 1 | {matrix[1][0]} | {matrix[1][1]} |",
                    "",
                ]
            )
    return lines


def format_comparison_rows(comparison: dict[str, Any]) -> list[str]:
    """Format embedding-vs-kmer comparison rows."""
    lines = [
        (
            "| Split | Embedding model | K-mer ROC-AUC | Embedding ROC-AUC | "
            "Delta ROC-AUC | K-mer PR-AUC | Embedding PR-AUC | Delta PR-AUC |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for split_name, split_comparison in comparison["by_split"].items():
        baseline = split_comparison.get("kmer_pair_text")
        for model_name, metrics in split_comparison["embedding"].items():
            lines.append(
                f"| {split_name} | {model_name} | "
                f"{format_metric(None if not baseline else baseline.get('roc_auc'))} | "
                f"{format_metric(metrics.get('roc_auc'))} | "
                f"{format_metric(metrics.get('delta_roc_auc_vs_kmer'))} | "
                f"{format_metric(None if not baseline else baseline.get('average_precision'))} | "
                f"{format_metric(metrics.get('average_precision'))} | "
                f"{format_metric(metrics.get('delta_average_precision_vs_kmer'))} |"
            )

    return lines


def build_report(
    data: pd.DataFrame,
    embeddings_by_model: dict[str, np.ndarray],
    results: dict[str, Any],
    comparison: dict[str, Any],
) -> str:
    """Build the embedding baseline Markdown report."""
    lines = [
        "# AbLang2 Embedding Baseline",
        "",
        f"Input embeddings: `{HEAVY_EMBEDDING_PATH.relative_to(PROJECT_ROOT)}`",
        "",
        "This report evaluates logistic-regression classifiers on cached pretrained",
        "AbLang2 embeddings for existing labeled rows.",
        "",
        "## Data",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Rows | {len(data)} |",
        f"| Label 0 count | {label_counts(data['label'])['0']} |",
        f"| Label 1 count | {label_counts(data['label'])['1']} |",
    ]
    for model_name, embeddings in embeddings_by_model.items():
        lines.append(f"| {model_name} embedding shape | {list(embeddings.shape)} |")

    lines.extend(
        [
            "",
            "## Split Validity",
            "",
            "| Split | Group column | Valid | Meaningful | Reason | Group overlap |",
            "|---|---|---:|---:|---|---:|",
        ]
    )
    for split_name, split_results in results.items():
        split = split_results.get("split", {})
        lines.append(
            f"| {split_name} | {format_nullable(split_results.get('group_column'))} | "
            f"{true_false(bool(split_results.get('valid')))} | "
            f"{true_false(bool(split_results.get('meaningful')))} | "
            f"{split_results.get('reason', 'n/a')} | "
            f"{format_nullable(split.get('group_overlap_count'))} |"
        )

    lines.extend(["", "## Metrics", ""])
    lines.extend(format_model_rows(results))
    lines.extend(["", "## Comparison vs K-mer", ""])
    lines.extend(format_comparison_rows(comparison))
    lines.extend([""])
    lines.extend(format_confusion_matrices(results))
    lines.extend(
        [
            "## Artifacts",
            "",
            f"- `{METRICS_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{ROC_AUC_FIGURE_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{PR_AUC_FIGURE_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{HEAVY_MODEL_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{PAIR_MODEL_PATH.relative_to(PROJECT_ROOT)}`",
            "",
        ]
    )

    return "\n".join(lines)


def print_metrics(results: dict[str, Any], comparison: dict[str, Any]) -> None:
    """Print requested terminal metrics and comparisons."""
    scalar_names = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    ]

    for split_name in ["random", GROUP_COLUMN]:
        split_results = results[split_name]
        if not split_results.get("valid"):
            print(f"\n{split_name} metrics")
            print("valid: false")
            print(f"reason: {split_results['reason']}")
            continue

        print(f"\n{split_name} metrics")
        print("valid: true")
        print(f"group_overlap_count: {format_nullable(split_results['split']['group_overlap_count'])}")
        for model_name, metrics in split_results["models"].items():
            print(f"{model_name}:")
            for metric_name in scalar_names:
                print(f"  {metric_name}: {format_metric(metrics[metric_name])}")
            print(f"  confusion_matrix: {metrics['confusion_matrix']}")

    print("\ncomparison vs k-mer")
    if not comparison["kmer_available"]:
        print("kmer metrics unavailable")
        return
    for split_name, split_comparison in comparison["by_split"].items():
        baseline = split_comparison.get("kmer_pair_text")
        print(f"{split_name}:")
        if not baseline:
            print("  kmer_pair_text: unavailable")
            continue
        print(
            "  kmer_pair_text: "
            f"roc_auc={format_metric(baseline.get('roc_auc'))}, "
            f"average_precision={format_metric(baseline.get('average_precision'))}"
        )
        for model_name, metrics in split_comparison["embedding"].items():
            print(
                f"  embedding_{model_name}: "
                f"roc_auc={format_metric(metrics.get('roc_auc'))} "
                f"(delta={format_metric(metrics.get('delta_roc_auc_vs_kmer'))}), "
                f"average_precision={format_metric(metrics.get('average_precision'))} "
                f"(delta={format_metric(metrics.get('delta_average_precision_vs_kmer'))})"
            )


def main() -> None:
    """Train and evaluate embedding logistic-regression baselines."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    metadata, embeddings_by_model = load_inputs()
    results = evaluate_all(metadata, embeddings_by_model)
    save_full_data_models(metadata, embeddings_by_model)
    kmer_comparison = load_kmer_comparison()
    comparison = build_comparison(results, kmer_comparison)
    save_figures(results, kmer_comparison)

    payload = {
        "metadata_path": str(METADATA_PATH.relative_to(PROJECT_ROOT)),
        "row_count": int(len(metadata)),
        "label_counts": label_counts(metadata["label"]),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "embedding_shapes": {
            name: list(values.shape) for name, values in embeddings_by_model.items()
        },
        "results": results,
        "comparison_vs_kmer": comparison,
        "artifacts": {
            "report": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
            "metrics_json": str(METRICS_PATH.relative_to(PROJECT_ROOT)),
            "roc_auc_comparison": str(ROC_AUC_FIGURE_PATH.relative_to(PROJECT_ROOT)),
            "pr_auc_comparison": str(PR_AUC_FIGURE_PATH.relative_to(PROJECT_ROOT)),
            "heavy_model": str(HEAVY_MODEL_PATH.relative_to(PROJECT_ROOT)),
            "pair_model": str(PAIR_MODEL_PATH.relative_to(PROJECT_ROOT)),
        },
    }
    METRICS_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        build_report(metadata, embeddings_by_model, results, comparison),
        encoding="utf-8",
    )

    print_metrics(results, comparison)


if __name__ == "__main__":
    main()
