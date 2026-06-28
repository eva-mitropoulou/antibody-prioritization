"""Run source-holdout validation and calibration diagnostics.

This reporting module evaluates a compact k-mer primary model family under
source/study holdout controls and calibration checks. It never writes raw
sequence values or raw source strings.
"""

from __future__ import annotations

import hashlib
import os
import re
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
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from _safe_analysis_utils import PROJECT_ROOT, load_json, write_json, write_text


INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "neutral_sequence_classification_ml.csv"
MODEL_REGISTRY_PATH = PROJECT_ROOT / "reports" / "metrics" / "model_registry.json"
SOURCE_REPORT_PATH = PROJECT_ROOT / "reports" / "source_holdout_validation_report.md"
CALIBRATION_REPORT_PATH = PROJECT_ROOT / "reports" / "calibration_threshold_report.md"
SOURCE_METRICS_PATH = PROJECT_ROOT / "reports" / "metrics" / "source_holdout_validation_metrics.json"
CALIBRATION_METRICS_PATH = PROJECT_ROOT / "reports" / "metrics" / "calibration_threshold_metrics.json"
ROC_FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "source_holdout_roc_auc_by_group.png"
PR_FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "source_holdout_pr_auc_by_group.png"
CALIBRATION_FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "calibration_curve.png"
THRESHOLD_FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "threshold_precision_recall.png"
HISTOGRAM_FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "probability_histogram_by_label.png"

RANDOM_STATE = 42
TEST_SIZE = 0.2
MIN_TEST_ROWS_PER_SOURCE = 30
MIN_CLASS_ROWS_FOR_METRICS = 5
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
TOP_K_VALUES = [25, 50, 100, 250]
SOURCE_CANDIDATES = [
    "sources",
    "Sources",
    "source",
    "study",
    "paper",
    "publication",
    "metadata_source",
    "Date Added",
    "date_added",
]
SOURCE_FALLBACK_CANDIDATES = ["metadata_origin"]
DATE_COLUMNS = {"Date Added", "date_added"}


def relpath(path: Path) -> str:
    """Return project-relative path."""
    return str(path.relative_to(PROJECT_ROOT))


def compact_sequence_text(value: Any) -> str:
    """Remove whitespace and uppercase an existing compact sequence text."""
    return re.sub(r"\s+", "", str(value or "")).upper()


def source_token(value: Any) -> str:
    """Build a stable primary source token without printing it."""
    text = str(value or "").strip()
    if not text:
        return "missing_source"
    parts = re.split(r"[;,|]", text)
    for part in parts:
        token = part.strip()
        if token:
            return token
    return text


def stable_hash(value: str) -> str:
    """Hash source or sequence-like values for safe IDs."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def detect_source_column(data: pd.DataFrame) -> tuple[str | None, str]:
    """Detect source/study column, preferring source-like fields over dates."""
    columns = set(data.columns)
    non_date_candidates = [c for c in SOURCE_CANDIDATES if c not in DATE_COLUMNS]
    for candidate in non_date_candidates:
        if candidate in columns:
            return candidate, "explicit_source_or_study"
    for candidate in SOURCE_FALLBACK_CANDIDATES:
        if candidate in columns:
            return candidate, "source_like_fallback"
    for candidate in DATE_COLUMNS:
        if candidate in columns:
            return candidate, "date_fallback"
    return None, "none_available"


def make_source_groups(data: pd.DataFrame, source_column: str | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Append hashed source group IDs."""
    output = data.copy()
    if source_column:
        raw = output[source_column].fillna("").astype(str)
    else:
        raw = pd.Series(["missing_source"] * len(output), index=output.index)
    tokens = raw.map(source_token)
    hashes = tokens.map(stable_hash)
    unique_hashes = sorted(hashes.unique())
    short_ids = {value: f"source_{index + 1:03d}" for index, value in enumerate(unique_hashes)}
    output["source_group_hash"] = hashes
    output["source_group_short_id"] = hashes.map(short_ids)
    diagnostics = {
        "source_column": source_column,
        "source_column_detection": "available" if source_column else "missing",
        "source_group_count": int(output["source_group_short_id"].nunique()),
        "raw_source_values_written": False,
        "source_group_hashes_written": True,
    }
    return output, diagnostics


def prepare_data() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load strict labeled data, compact input text, and safe source groups."""
    data = pd.read_csv(INPUT_PATH, dtype=str, keep_default_na=False)
    if "label" not in data.columns:
        raise ValueError("Input table is missing label column.")
    if "sequence_pair_text" not in data.columns:
        raise ValueError("Input table is missing sequence_pair_text column.")
    data = data.copy()
    data["label"] = pd.to_numeric(data["label"], errors="coerce")
    data = data[data["label"].isin([0, 1])].copy()
    data["label"] = data["label"].astype(int)
    data["compact_sequence_text"] = data["sequence_pair_text"].map(compact_sequence_text)
    data = data[data["compact_sequence_text"].str.len().gt(0)].copy()
    source_column, source_detection_reason = detect_source_column(data)
    data, diagnostics = make_source_groups(data, source_column)
    diagnostics.update(
        {
            "source_detection_reason": source_detection_reason,
            "input_path": relpath(INPUT_PATH),
            "row_count": int(len(data)),
            "columns": list(pd.read_csv(INPUT_PATH, nrows=0).columns),
            "label_counts": label_counts(data["label"]),
            "source_group_size_distribution": numeric_distribution(
                data.groupby("source_group_short_id").size()
            ),
            "source_group_label_balance": source_label_balance(data),
            "usable_source_group_count": int(
                sum(
                    group["row_count"] >= MIN_TEST_ROWS_PER_SOURCE
                    for group in source_label_balance(data).values()
                )
            ),
        }
    )
    registry = load_json(MODEL_REGISTRY_PATH) or {}
    diagnostics["primary_broad_scorer_from_registry"] = (
        (registry.get("primary_broad_scorer") or {}).get("model_id")
    )
    diagnostics["model_family_used"] = "compact_char_kmer_logreg"
    return data.reset_index(drop=True), diagnostics


def label_counts(labels: pd.Series) -> dict[str, int]:
    """Return stable label counts."""
    counts = labels.value_counts().sort_index()
    return {"0": int(counts.get(0, 0)), "1": int(counts.get(1, 0))}


def numeric_distribution(values: pd.Series) -> dict[str, float | int | None]:
    """Summarize numeric values."""
    if len(values) == 0:
        return {"min": None, "median": None, "mean": None, "max": None}
    return {
        "min": int(values.min()),
        "median": float(values.median()),
        "mean": float(values.mean()),
        "max": int(values.max()),
    }


def source_label_balance(data: pd.DataFrame) -> dict[str, Any]:
    """Summarize safe source group label balance."""
    output: dict[str, Any] = {}
    grouped = data.groupby("source_group_short_id", sort=True)
    for group_id, group in grouped:
        output[str(group_id)] = {
            "source_group_hash": str(group["source_group_hash"].iloc[0]),
            "row_count": int(len(group)),
            "label_counts": label_counts(group["label"]),
        }
    return output


def make_model() -> Pipeline:
    """Create the requested k-mer model."""
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2)),
            ("classifier", LogisticRegression(max_iter=5000, class_weight="balanced")),
        ]
    )


def positive_scores(model: Pipeline | DummyClassifier, values: pd.Series | np.ndarray) -> np.ndarray:
    """Return label-1 probabilities."""
    class_list = list(model.classes_)
    if 1 not in class_list:
        return np.zeros(len(values), dtype=float)
    return model.predict_proba(values)[:, class_list.index(1)]


def metric_payload(y_true: pd.Series, y_score: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Compute metrics, with ROC/PR only when class counts are sufficient."""
    counts = label_counts(y_true)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    valid_auc = counts["0"] >= MIN_CLASS_ROWS_FOR_METRICS and counts["1"] >= MIN_CLASS_ROWS_FOR_METRICS
    accuracy = float(accuracy_score(y_true, y_pred))
    balanced_accuracy = (
        float(balanced_accuracy_score(y_true, y_pred))
        if y_true.nunique() == 2
        else accuracy
    )
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if valid_auc else None,
        "pr_auc": float(average_precision_score(y_true, y_score)) if valid_auc else None,
        "confusion_matrix": matrix.astype(int).tolist(),
        "test_positive_fraction": float(y_true.mean()) if len(y_true) else None,
        "pr_baseline": float(y_true.mean()) if len(y_true) else None,
        "valid_auc_metrics": bool(valid_auc),
    }


def majority_baseline(train_y: pd.Series, test_y: pd.Series) -> dict[str, Any]:
    """Evaluate majority-class baseline."""
    model = DummyClassifier(strategy="most_frequent")
    train_x = np.zeros((len(train_y), 1))
    test_x = np.zeros((len(test_y), 1))
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    score = positive_scores(model, test_x)
    return metric_payload(test_y, score, pred)


def fit_evaluate(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    """Fit and evaluate the k-mer model on one split."""
    model = make_model()
    model.fit(train["compact_sequence_text"], train["label"])
    score = positive_scores(model, test["compact_sequence_text"])
    pred = (score >= 0.5).astype(int)
    metrics = metric_payload(test["label"], score, pred)
    metrics["majority_baseline"] = majority_baseline(train["label"], test["label"])
    return metrics


def run_leave_source_out(data: pd.DataFrame) -> dict[str, Any]:
    """Run leave-one-source-group-out validation."""
    results: dict[str, Any] = {}
    skipped: dict[str, int] = {}
    for group_id, test in data.groupby("source_group_short_id", sort=True):
        test = test.copy()
        if len(test) < MIN_TEST_ROWS_PER_SOURCE:
            results[group_id] = {
                "status": "skipped",
                "reason": "too_few_test_rows",
                "test_size": int(len(test)),
                "test_label_counts": label_counts(test["label"]),
            }
            skipped["too_few_test_rows"] = skipped.get("too_few_test_rows", 0) + 1
            continue
        train = data[data["source_group_short_id"].ne(group_id)].copy()
        train_counts = label_counts(train["label"])
        test_counts = label_counts(test["label"])
        if min(train_counts.values()) == 0:
            results[group_id] = {
                "status": "skipped",
                "reason": "train_single_class",
                "test_size": int(len(test)),
                "train_label_counts": train_counts,
                "test_label_counts": test_counts,
            }
            skipped["train_single_class"] = skipped.get("train_single_class", 0) + 1
            continue
        train_groups = set(train["source_group_short_id"])
        test_groups = set(test["source_group_short_id"])
        group_overlap = int(len(train_groups & test_groups))
        metrics = fit_evaluate(train, test)
        reason = "ok" if metrics["valid_auc_metrics"] else "test_insufficient_class_counts"
        if reason != "ok":
            skipped[reason] = skipped.get(reason, 0) + 1
        results[group_id] = {
            "status": "valid" if reason == "ok" else "metrics_limited",
            "reason": reason,
            "group_overlap_count": group_overlap,
            "train_size": int(len(train)),
            "test_size": int(len(test)),
            "train_label_counts": train_counts,
            "test_label_counts": test_counts,
            "metrics": metrics,
        }
    return {
        "per_group": results,
        "aggregate": aggregate_leave_source_out(results),
        "skipped_group_reasons": skipped,
    }


def aggregate_leave_source_out(results: dict[str, Any]) -> dict[str, Any]:
    """Aggregate valid held-out source group metrics."""
    valid = [
        item for item in results.values() if item.get("metrics", {}).get("valid_auc_metrics")
    ]
    if not valid:
        return {
            "valid_heldout_source_group_count": 0,
            "skipped_group_count": int(len(results)),
            "macro_mean": {},
            "weighted_mean_by_test_size": {},
        }
    metric_names = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    macro: dict[str, float] = {}
    weighted: dict[str, float] = {}
    weights = np.asarray([item["test_size"] for item in valid], dtype=float)
    for metric in metric_names:
        values = np.asarray([item["metrics"][metric] for item in valid], dtype=float)
        macro[metric] = float(np.mean(values))
        weighted[metric] = float(np.average(values, weights=weights))
    return {
        "valid_heldout_source_group_count": int(len(valid)),
        "skipped_group_count": int(len(results) - len(valid)),
        "macro_mean": macro,
        "weighted_mean_by_test_size": weighted,
    }


def source_grouped_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Create source-grouped fallback split with zero group overlap."""
    groups = data["source_group_short_id"].astype(str)
    last_reason = "no_valid_split"
    for offset in range(100):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE + offset,
        )
        train_pos, test_pos = next(splitter.split(data, data["label"], groups=groups))
        train = data.iloc[train_pos].copy()
        test = data.iloc[test_pos].copy()
        if train["label"].nunique() != 2:
            last_reason = "train_single_class"
            continue
        if test["label"].nunique() != 2:
            last_reason = "test_single_class"
            continue
        overlap = set(train["source_group_short_id"]) & set(test["source_group_short_id"])
        if overlap:
            last_reason = "group_overlap"
            continue
        split = {
            "strategy": "GroupShuffleSplit",
            "group_column": "source_group_short_id",
            "random_state": RANDOM_STATE + offset,
            "test_size": TEST_SIZE,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_label_counts": label_counts(train["label"]),
            "test_label_counts": label_counts(test["label"]),
            "train_group_count": int(train["source_group_short_id"].nunique()),
            "test_group_count": int(test["source_group_short_id"].nunique()),
            "group_overlap_count": 0,
            "fallback_reason": None,
        }
        return train, test, split
    return group_feature_v_split(data, last_reason)


def group_feature_v_split(data: pd.DataFrame, source_failure_reason: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fallback to existing group_feature_v grouped split if source split is unsuitable."""
    if "group_feature_v" not in data.columns:
        raise ValueError(f"Source grouped split failed and group_feature_v is unavailable: {source_failure_reason}")
    groups = data["group_feature_v"].fillna("").astype(str)
    groups = groups.mask(groups.eq(""), "missing_group_feature_v")
    for offset in range(100):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE + offset,
        )
        train_pos, test_pos = next(splitter.split(data, data["label"], groups=groups))
        train = data.iloc[train_pos].copy()
        test = data.iloc[test_pos].copy()
        if train["label"].nunique() != 2 or test["label"].nunique() != 2:
            continue
        train_groups = set(groups.iloc[train_pos])
        test_groups = set(groups.iloc[test_pos])
        if train_groups & test_groups:
            continue
        return train, test, {
            "strategy": "GroupShuffleSplit",
            "group_column": "group_feature_v",
            "random_state": RANDOM_STATE + offset,
            "test_size": TEST_SIZE,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_label_counts": label_counts(train["label"]),
            "test_label_counts": label_counts(test["label"]),
            "train_group_count": int(len(train_groups)),
            "test_group_count": int(len(test_groups)),
            "group_overlap_count": 0,
            "fallback_reason": f"source_split_unsuitable:{source_failure_reason}",
        }
    raise ValueError("No valid grouped fallback split was available.")


def run_fallback_validation(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], Pipeline, np.ndarray]:
    """Run source-grouped fallback validation and return fitted model/probabilities."""
    train, test, split = source_grouped_split(data)
    model = make_model()
    model.fit(train["compact_sequence_text"], train["label"])
    scores = positive_scores(model, test["compact_sequence_text"])
    preds = (scores >= 0.5).astype(int)
    metrics = metric_payload(test["label"], scores, preds)
    metrics["majority_baseline"] = majority_baseline(train["label"], test["label"])
    split["metrics"] = metrics
    return train, test, split, model, scores


def calibration_curve_payload(y_true: pd.Series, y_score: np.ndarray, n_bins: int = 10) -> tuple[list[dict[str, Any]], float]:
    """Build 10-bin calibration curve and ECE approximation."""
    y = y_true.to_numpy()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, Any]] = []
    ece = 0.0
    for index in range(n_bins):
        low = bins[index]
        high = bins[index + 1]
        if index == n_bins - 1:
            mask = (y_score >= low) & (y_score <= high)
        else:
            mask = (y_score >= low) & (y_score < high)
        count = int(mask.sum())
        if count:
            mean_pred = float(np.mean(y_score[mask]))
            observed = float(np.mean(y[mask]))
            ece += (count / len(y_score)) * abs(mean_pred - observed)
        else:
            mean_pred = None
            observed = None
        rows.append(
            {
                "bin_index": index + 1,
                "probability_low": float(low),
                "probability_high": float(high),
                "count": count,
                "mean_predicted_probability": mean_pred,
                "observed_positive_fraction": observed,
            }
        )
    return rows, float(ece)


def threshold_table(y_true: pd.Series, y_score: np.ndarray) -> list[dict[str, Any]]:
    """Compute threshold operating points."""
    output = []
    for threshold in THRESHOLDS:
        pred = (y_score >= threshold).astype(int)
        output.append(
            {
                "threshold": float(threshold),
                "predicted_positive_count": int(pred.sum()),
                "precision": float(precision_score(y_true, pred, zero_division=0)),
                "recall": float(recall_score(y_true, pred, zero_division=0)),
                "f1": float(f1_score(y_true, pred, zero_division=0)),
            }
        )
    return output


def topk_precision(y_true: pd.Series, y_score: np.ndarray) -> dict[str, Any]:
    """Compute top-k precision where possible."""
    order = np.argsort(-y_score)
    output: dict[str, Any] = {}
    y = y_true.to_numpy()
    for k in TOP_K_VALUES:
        if len(y) < k:
            continue
        selected = y[order[:k]]
        output[str(k)] = {
            "selected_count": int(k),
            "positive_count": int(selected.sum()),
            "precision": float(np.mean(selected)),
        }
    return output


def high_confidence_threshold_summary(thresholds: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick a reasonable high-confidence review threshold."""
    candidates = [
        row for row in thresholds if row["predicted_positive_count"] > 0 and row["precision"] >= 0.8
    ]
    if candidates:
        selected = max(candidates, key=lambda row: (row["threshold"], row["recall"]))
        return {
            "selection_rule": "highest threshold with precision >= 0.8 and at least one predicted positive",
            **selected,
        }
    selected = max(thresholds, key=lambda row: (row["precision"], row["threshold"]))
    return {
        "selection_rule": "highest observed precision among evaluated thresholds",
        **selected,
    }


def run_calibration(data: pd.DataFrame) -> dict[str, Any]:
    """Run calibration and threshold diagnostics."""
    train, test, split, _model, scores = run_fallback_validation(data)
    curve, ece = calibration_curve_payload(test["label"], scores)
    thresholds = threshold_table(test["label"], scores)
    topk = topk_precision(test["label"], scores)
    brier = float(brier_score_loss(test["label"], scores))
    return {
        "status": "available",
        "input_path": relpath(INPUT_PATH),
        "split": split,
        "brier_score": brier,
        "expected_calibration_error": ece,
        "calibration_curve": curve,
        "threshold_metrics": thresholds,
        "topk_precision": topk,
        "best_high_confidence_threshold": high_confidence_threshold_summary(thresholds),
        "calibrated_model": {
            "status": "skipped",
            "reason": "Uncalibrated source-grouped test diagnostics were prioritized; no final-test calibration fitting was performed.",
        },
        "artifacts": {
            "report": relpath(CALIBRATION_REPORT_PATH),
            "metrics_json": relpath(CALIBRATION_METRICS_PATH),
            "calibration_curve": relpath(CALIBRATION_FIGURE_PATH),
            "threshold_precision_recall": relpath(THRESHOLD_FIGURE_PATH),
            "probability_histogram_by_label": relpath(HISTOGRAM_FIGURE_PATH),
        },
    }


def source_metrics_payload(data: pd.DataFrame, diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Run source holdout and fallback validation."""
    leave_source_out = run_leave_source_out(data)
    train, test, fallback_split, _model, _scores = run_fallback_validation(data)
    return {
        "status": "available",
        "source_diagnostics": diagnostics,
        "leave_source_out": leave_source_out,
        "source_grouped_fallback": fallback_split,
        "quality_gates": {
            "source_group_counts_reported": diagnostics["source_group_count"] > 0,
            "validation_result_produced": bool(fallback_split.get("metrics")),
            "group_overlap_zero": fallback_split.get("group_overlap_count") == 0,
            "raw_source_strings_written": False,
            "raw_sequence_strings_written": False,
        },
        "artifacts": {
            "report": relpath(SOURCE_REPORT_PATH),
            "metrics_json": relpath(SOURCE_METRICS_PATH),
            "roc_auc_by_group": relpath(ROC_FIGURE_PATH),
            "pr_auc_by_group": relpath(PR_FIGURE_PATH),
        },
    }


def fmt(value: Any) -> str:
    """Format optional metrics."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def build_source_report(metrics: dict[str, Any]) -> str:
    """Build source-holdout validation report."""
    diag = metrics["source_diagnostics"]
    aggregate = metrics["leave_source_out"]["aggregate"]
    fallback = metrics["source_grouped_fallback"]
    fallback_metrics = fallback["metrics"]
    lines = [
        "# Source-Holdout Validation",
        "",
        "Source/study groups are represented only by hashes and short IDs. Raw",
        "source strings, DOI/source URLs, and sequence strings are not written.",
        "",
        "## Source Diagnostics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Source column | {diag.get('source_column') or 'unavailable'} |",
        f"| Source detection reason | {diag.get('source_detection_reason')} |",
        f"| Row count | {diag['row_count']} |",
        f"| Source group count | {diag['source_group_count']} |",
        f"| Usable source group count | {diag['usable_source_group_count']} |",
        f"| Source group size min | {diag['source_group_size_distribution']['min']} |",
        f"| Source group size median | {fmt(diag['source_group_size_distribution']['median'])} |",
        f"| Source group size max | {diag['source_group_size_distribution']['max']} |",
        "",
        "## Leave-Source-Out Aggregate",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Valid held-out source groups | {aggregate['valid_heldout_source_group_count']} |",
        f"| Skipped/limited groups | {aggregate['skipped_group_count']} |",
        f"| Macro ROC-AUC | {fmt(aggregate.get('macro_mean', {}).get('roc_auc'))} |",
        f"| Macro PR-AUC | {fmt(aggregate.get('macro_mean', {}).get('pr_auc'))} |",
        f"| Weighted ROC-AUC | {fmt(aggregate.get('weighted_mean_by_test_size', {}).get('roc_auc'))} |",
        f"| Weighted PR-AUC | {fmt(aggregate.get('weighted_mean_by_test_size', {}).get('pr_auc'))} |",
        "",
        "## Source-Grouped Fallback Split",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Split strategy | {fallback['strategy']} |",
        f"| Group column | {fallback['group_column']} |",
        f"| Train rows | {fallback['train_rows']} |",
        f"| Test rows | {fallback['test_rows']} |",
        f"| Group overlap | {fallback['group_overlap_count']} |",
        f"| ROC-AUC | {fmt(fallback_metrics.get('roc_auc'))} |",
        f"| PR-AUC | {fmt(fallback_metrics.get('pr_auc'))} |",
        f"| F1 | {fmt(fallback_metrics.get('f1'))} |",
        "",
        "## Skipped Group Reasons",
        "",
        "| Reason | Count |",
        "|---|---:|",
    ]
    for reason, count in metrics["leave_source_out"]["skipped_group_reasons"].items():
        lines.append(f"| {reason} | {count} |")
    lines.extend(
        [
            "",
            "## Per-Source Results",
            "",
            "| Source group | Status | Reason | Test rows | Label 0 | Label 1 | ROC-AUC | PR-AUC |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for group_id, result in metrics["leave_source_out"]["per_group"].items():
        counts = result.get("test_label_counts", {})
        result_metrics = result.get("metrics", {})
        lines.append(
            f"| {group_id} | {result.get('status')} | {result.get('reason')} | "
            f"{result.get('test_size')} | {counts.get('0', 0)} | {counts.get('1', 0)} | "
            f"{fmt(result_metrics.get('roc_auc'))} | {fmt(result_metrics.get('pr_auc'))} |"
        )
    lines.extend(["", "## Artifacts", ""])
    for path in metrics["artifacts"].values():
        lines.append(f"- `{path}`")
    lines.append("")
    return "\n".join(lines)


def build_calibration_report(metrics: dict[str, Any]) -> str:
    """Build calibration/threshold report."""
    split = metrics["split"]
    split_metrics = split["metrics"]
    best = metrics["best_high_confidence_threshold"]
    lines = [
        "# Calibration And Threshold Analysis",
        "",
        "Calibration was evaluated on a held-out grouped split. Probability scores",
        "are interpreted as prioritization/ranking signals unless calibration is",
        "shown to be reliable.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Split group column | {split['group_column']} |",
        f"| Train rows | {split['train_rows']} |",
        f"| Test rows | {split['test_rows']} |",
        f"| Group overlap | {split['group_overlap_count']} |",
        f"| ROC-AUC | {fmt(split_metrics.get('roc_auc'))} |",
        f"| PR-AUC | {fmt(split_metrics.get('pr_auc'))} |",
        f"| Brier score | {metrics['brier_score']:.4f} |",
        f"| Expected calibration error | {metrics['expected_calibration_error']:.4f} |",
        "",
        "## Threshold Table",
        "",
        "| Threshold | Predicted positives | Precision | Recall | F1 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in metrics["threshold_metrics"]:
        lines.append(
            f"| {row['threshold']:.1f} | {row['predicted_positive_count']} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Top-k Precision",
            "",
            "| k | Positive count | Precision |",
            "|---:|---:|---:|",
        ]
    )
    for k, row in metrics["topk_precision"].items():
        lines.append(f"| {k} | {row['positive_count']} | {row['precision']:.4f} |")
    lines.extend(
        [
            "",
            "## High-Confidence Review Threshold",
            "",
            (
                f"Selected threshold {best['threshold']:.1f}: precision "
                f"{best['precision']:.4f}, recall {best['recall']:.4f}, "
                f"predicted positives {best['predicted_positive_count']}."
            ),
            "",
            "## Calibration Interpretation",
            "",
            "Use the model primarily for ranking unless a target use case accepts the",
            "reported Brier score and threshold tradeoffs.",
            "",
            "## Artifacts",
            "",
        ]
    )
    for path in metrics["artifacts"].values():
        lines.append(f"- `{path}`")
    lines.append("")
    return "\n".join(lines)


def save_holdout_figures(metrics: dict[str, Any]) -> None:
    """Save per-source ROC and PR figures."""
    rows = []
    for group_id, result in metrics["leave_source_out"]["per_group"].items():
        m = result.get("metrics", {})
        if m.get("valid_auc_metrics"):
            rows.append(
                {
                    "source_group": group_id,
                    "roc_auc": m["roc_auc"],
                    "pr_auc": m["pr_auc"],
                    "test_size": result["test_size"],
                }
            )
    table = pd.DataFrame(rows).sort_values("source_group") if rows else pd.DataFrame()
    if table.empty:
        return
    for metric, path, title in [
        ("roc_auc", ROC_FIGURE_PATH, "Leave-source-out ROC-AUC by source group"),
        ("pr_auc", PR_FIGURE_PATH, "Leave-source-out PR-AUC by source group"),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(max(8, 0.22 * len(table)), 4.8))
        ax.bar(np.arange(len(table)), table[metric], color="#4C78A8")
        ax.set_xticks(np.arange(len(table)))
        ax.set_xticklabels(table["source_group"], rotation=90, fontsize=6)
        ax.set_ylim(0, 1)
        ax.set_ylabel(metric)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)


def save_calibration_figures(metrics: dict[str, Any], data: pd.DataFrame) -> None:
    """Save calibration curve, threshold curve, and probability histogram."""
    _train, test, split, _model, scores = run_fallback_validation(data)
    curve = pd.DataFrame(metrics["calibration_curve"])
    CALIBRATION_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    valid = curve.dropna(subset=["mean_predicted_probability", "observed_positive_fraction"])
    if not valid.empty:
        ax.plot(
            valid["mean_predicted_probability"],
            valid["observed_positive_fraction"],
            marker="o",
            color="#4C78A8",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.5")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive fraction")
    ax.set_title("Calibration curve")
    fig.tight_layout()
    fig.savefig(CALIBRATION_FIGURE_PATH, dpi=200)
    plt.close(fig)

    threshold_df = pd.DataFrame(metrics["threshold_metrics"])
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(threshold_df["threshold"], threshold_df["precision"], marker="o", label="precision")
    ax.plot(threshold_df["threshold"], threshold_df["recall"], marker="o", label="recall")
    ax.plot(threshold_df["threshold"], threshold_df["f1"], marker="o", label="F1")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric")
    ax.set_title("Threshold precision/recall tradeoff")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(THRESHOLD_FIGURE_PATH, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, color in [(0, "#4C78A8"), (1, "#F58518")]:
        values = scores[test["label"].eq(label)]
        if len(values):
            ax.hist(values, bins=30, alpha=0.65, color=color, label=f"label {label}")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Held-out row count")
    ax.set_title("Probability histogram by true label")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(HISTOGRAM_FIGURE_PATH, dpi=200)
    plt.close(fig)


def main() -> int:
    data, diagnostics = prepare_data()
    source_metrics = source_metrics_payload(data, diagnostics)
    calibration_metrics = run_calibration(data)
    write_json(SOURCE_METRICS_PATH, source_metrics)
    write_text(SOURCE_REPORT_PATH, build_source_report(source_metrics))
    write_json(CALIBRATION_METRICS_PATH, calibration_metrics)
    write_text(CALIBRATION_REPORT_PATH, build_calibration_report(calibration_metrics))
    save_holdout_figures(source_metrics)
    save_calibration_figures(calibration_metrics, data)

    aggregate = source_metrics["leave_source_out"]["aggregate"]
    fallback_metrics = source_metrics["source_grouped_fallback"]["metrics"]
    best_threshold = calibration_metrics["best_high_confidence_threshold"]
    output_paths = {
        "source_report": relpath(SOURCE_REPORT_PATH),
        "calibration_report": relpath(CALIBRATION_REPORT_PATH),
        "source_metrics": relpath(SOURCE_METRICS_PATH),
        "calibration_metrics": relpath(CALIBRATION_METRICS_PATH),
        "source_roc_figure": relpath(ROC_FIGURE_PATH),
        "source_pr_figure": relpath(PR_FIGURE_PATH),
        "calibration_curve": relpath(CALIBRATION_FIGURE_PATH),
        "threshold_precision_recall": relpath(THRESHOLD_FIGURE_PATH),
        "probability_histogram_by_label": relpath(HISTOGRAM_FIGURE_PATH),
    }
    print(
        "source_groups="
        f"{diagnostics['source_group_count']}; "
        f"valid_heldout_source_groups={aggregate['valid_heldout_source_group_count']}; "
        f"source_holdout_mean_roc_auc={fmt(aggregate.get('macro_mean', {}).get('roc_auc'))}; "
        f"source_holdout_mean_pr_auc={fmt(aggregate.get('macro_mean', {}).get('pr_auc'))}; "
        f"source_grouped_fallback_roc_auc={fmt(fallback_metrics.get('roc_auc'))}; "
        f"source_grouped_fallback_pr_auc={fmt(fallback_metrics.get('pr_auc'))}; "
        f"brier_score={calibration_metrics['brier_score']:.4f}; "
        f"best_high_confidence_threshold={best_threshold}; "
        f"outputs={output_paths}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
