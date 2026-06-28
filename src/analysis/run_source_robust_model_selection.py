"""Source-robust k-mer model selection and failure analysis.

This CPU-only module evaluates conservative k-mer model variants under
source/study holdout controls, calibration diagnostics, and abstention analysis.
It never writes raw sequence values or raw source strings.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
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


STRICT_PATH = PROJECT_ROOT / "data" / "processed" / "neutral_sequence_classification_ml.csv"
SOURCE_BASELINE_PATH = PROJECT_ROOT / "reports" / "metrics" / "source_holdout_validation_metrics.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "source_robust_model_selection_report.md"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics" / "source_robust_model_selection_metrics.json"
COMPARISON_CSV_PATH = PROJECT_ROOT / "reports" / "source_robust_model_comparison.csv"
FAILURE_CSV_PATH = PROJECT_ROOT / "reports" / "source_holdout_failure_analysis.csv"
COMPARISON_FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "source_robust_model_comparison.png"
PR_FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "source_robust_pr_auc_by_model.png"
ROC_FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "source_robust_roc_auc_by_model.png"
ABSTENTION_FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "abstention_precision_coverage.png"
FAILURE_FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "source_failure_summary.png"

RANDOM_STATE = 42
TEST_SIZE = 0.2
MIN_TEST_ROWS_PER_SOURCE = 30
MIN_CLASS_ROWS_FOR_METRICS = 5
MIN_VARIANT_ROWS = 300
MIN_VARIANT_SOURCE_GROUPS = 4
MAX_FEATURES = 50_000
FALLBACK_MAX_FEATURES = 30_000
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
TOP_K_VALUES = [25, 50, 100, 250]
SIMILAR_METRIC_MARGIN = 0.005
MISSING_TEXT_VALUES = {"", "na", "nan", "none", "null", "nd", "n/a", "unknown"}

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
HEAVY_CANDIDATES = ["sequence_a", "heavy_sequence", "sequence_heavy_only", "vh_sequence"]
LIGHT_CANDIDATES = ["sequence_b", "light_sequence", "vl_sequence"]
CDR_CANDIDATES = [
    "cdrh1_seq",
    "cdrh2_seq",
    "cdrh3_seq",
    "cdrl1_seq",
    "cdrl2_seq",
    "cdrl3_seq",
    "existing_cdrh3",
    "existing_cdrl3",
    "group_feature_cdr3",
    "group_feature_b_cdr3",
]


@dataclass
class Variant:
    """Model variant definition."""

    model_id: str
    display_name: str
    row_subset: str
    selection_eligible: bool
    simplicity_rank: int
    data: pd.DataFrame


def relpath(path: Path) -> str:
    """Return project-relative path."""
    return str(path.relative_to(PROJECT_ROOT))


def stable_hash(value: str) -> str:
    """Hash unsafe source values."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_text(value: Any) -> str:
    """Normalize general text."""
    return str(value or "").strip()


def normalize_sequence(value: Any) -> str:
    """Remove whitespace and uppercase sequence-like strings for internal modeling."""
    text = re.sub(r"\s+", "", str(value or "")).upper()
    return "" if text.lower() in MISSING_TEXT_VALUES else text


def compact_existing_pair(value: Any) -> str:
    """Compact an existing pair-text field."""
    return re.sub(r"\s+", "", str(value or "")).upper()


def source_token(value: Any) -> str:
    """Return stable primary source token without exposing it."""
    text = normalize_text(value)
    if not text:
        return "missing_source"
    parts = re.split(r"[;,|]", text)
    for part in parts:
        token = part.strip()
        if token:
            return token
    return text


def detect_source_column(data: pd.DataFrame) -> tuple[str | None, str]:
    """Detect source/study column, preferring source-like fields over dates."""
    columns = set(data.columns)
    for candidate in [c for c in SOURCE_CANDIDATES if c not in DATE_COLUMNS]:
        if candidate in columns:
            return candidate, "explicit_source_or_study"
    for candidate in SOURCE_FALLBACK_CANDIDATES:
        if candidate in columns:
            return candidate, "source_like_fallback"
    for candidate in DATE_COLUMNS:
        if candidate in columns:
            return candidate, "date_fallback"
    return None, "none_available"


def first_available_column(data: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return first available column."""
    for column in candidates:
        if column in data.columns:
            return column
    return None


def label_counts(labels: pd.Series) -> dict[str, int]:
    """Return stable label counts."""
    counts = labels.value_counts().sort_index()
    return {"0": int(counts.get(0, 0)), "1": int(counts.get(1, 0))}


def count_dict(values: pd.Series) -> dict[str, int]:
    """Return JSON-safe counts."""
    counts = values.fillna("missing").astype(str).value_counts().sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def numeric_summary(values: pd.Series) -> dict[str, float | int | None]:
    """Return numeric summary."""
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == 0:
        return {"min": None, "median": None, "mean": None, "max": None}
    return {
        "min": float(numeric.min()),
        "median": float(numeric.median()),
        "mean": float(numeric.mean()),
        "max": float(numeric.max()),
    }


def target_category(value: Any) -> str:
    """Map target metadata to coarse categories."""
    text = str(value or "").strip().lower()
    if text in MISSING_TEXT_VALUES:
        return "other_unknown"
    if "rbd" in text or "receptor binding" in text:
        return "RBD"
    if "spike" in text or text in {"s", "s protein"}:
        return "Spike non-RBD"
    return "other_unknown"


def load_strict_data() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load strict labeled ML data and append safe metadata."""
    raw = pd.read_csv(STRICT_PATH, dtype=str, keep_default_na=False)
    data = raw.copy()
    data["label"] = pd.to_numeric(data["label"], errors="coerce")
    data = data[data["label"].isin([0, 1])].copy()
    data["label"] = data["label"].astype(int)
    source_column, source_reason = detect_source_column(data)
    source_values = (
        data[source_column].fillna("").astype(str)
        if source_column
        else pd.Series(["missing_source"] * len(data), index=data.index)
    )
    source_hashes = source_values.map(source_token).map(stable_hash)
    source_short_ids = {
        value: f"source_{index + 1:03d}"
        for index, value in enumerate(sorted(source_hashes.unique()))
    }
    data["source_group_hash"] = source_hashes
    data["source_group_short_id"] = source_hashes.map(source_short_ids)

    heavy_col = first_available_column(data, HEAVY_CANDIDATES)
    light_col = first_available_column(data, LIGHT_CANDIDATES)
    heavy = data[heavy_col].map(normalize_sequence) if heavy_col else pd.Series([""] * len(data), index=data.index)
    light = data[light_col].map(normalize_sequence) if light_col else pd.Series([""] * len(data), index=data.index)
    data["heavy_length"] = heavy.str.len().astype(int)
    data["light_length"] = light.str.len().astype(int)
    data["has_light"] = light.ne("")
    data["paired_light_status"] = np.where(data["has_light"], "paired", "light_missing_or_single_chain")
    if "metadata_target_region" in data.columns:
        data["target_category"] = data["metadata_target_region"].map(target_category)
    else:
        data["target_category"] = "other_unknown"

    relevant_columns = [
        column
        for column in [
            source_column,
            "metadata_target_region",
            heavy_col,
            light_col,
            "sequence_pair_text",
            "group_feature_cdr3",
            "group_feature_b_cdr3",
        ]
        if column
    ]
    missing_counts = {
        column: int(data[column].fillna("").astype(str).str.strip().eq("").sum())
        for column in sorted(set(relevant_columns))
    }
    source_sizes = data.groupby("source_group_short_id").size()
    diagnostics = {
        "input_path": relpath(STRICT_PATH),
        "shape": [int(data.shape[0]), int(data.shape[1])],
        "columns": list(raw.columns),
        "label_counts": label_counts(data["label"]),
        "metadata_missing_counts": missing_counts,
        "source_column": source_column,
        "source_detection_reason": source_reason,
        "source_group_count": int(data["source_group_short_id"].nunique()),
        "source_group_size_distribution": numeric_summary(source_sizes),
        "source_group_label_balance": source_group_label_balance(data),
        "raw_source_strings_written": False,
        "raw_sequence_strings_written": False,
    }
    return data.reset_index(drop=True), diagnostics


def source_group_label_balance(data: pd.DataFrame) -> dict[str, Any]:
    """Return safe label balance by source group."""
    output: dict[str, Any] = {}
    for group_id, group in data.groupby("source_group_short_id", sort=True):
        output[group_id] = {
            "source_group_hash": str(group["source_group_hash"].iloc[0]),
            "row_count": int(len(group)),
            "label_counts": label_counts(group["label"]),
        }
    return output


def build_pair_text(data: pd.DataFrame) -> pd.Series:
    """Build whole-pair compact model text."""
    if "sequence_pair_text" in data.columns:
        text = data["sequence_pair_text"].map(compact_existing_pair)
        if text.str.len().gt(0).any():
            return text
    heavy_col = first_available_column(data, HEAVY_CANDIDATES)
    light_col = first_available_column(data, LIGHT_CANDIDATES)
    if heavy_col:
        heavy = data[heavy_col].map(normalize_sequence)
        light = data[light_col].map(normalize_sequence) if light_col else pd.Series([""] * len(data), index=data.index)
        return pd.Series(
            [f"{h}[SEP]{l}" if l else h for h, l in zip(heavy, light)],
            index=data.index,
        )
    for column in data.columns:
        if "sequence" in column.lower():
            return data[column].map(normalize_sequence)
    return pd.Series([""] * len(data), index=data.index)


def build_heavy_text(data: pd.DataFrame) -> pd.Series:
    """Build heavy-only text."""
    heavy_col = first_available_column(data, HEAVY_CANDIDATES)
    if not heavy_col:
        return pd.Series([""] * len(data), index=data.index)
    return data[heavy_col].map(normalize_sequence)


def build_cdr_text(data: pd.DataFrame) -> tuple[pd.Series, list[str]]:
    """Build CDR/region text from available columns."""
    available = [column for column in CDR_CANDIDATES if column in data.columns]
    if not available:
        return pd.Series([""] * len(data), index=data.index), []
    parts = []
    for column in available:
        parts.append(data[column].map(normalize_sequence))
    cdr = pd.concat(parts, axis=1).agg("|".join, axis=1).map(compact_existing_pair)
    return cdr, available


def variant_frame(data: pd.DataFrame, model_text: pd.Series) -> pd.DataFrame:
    """Return frame with safe metadata and internal model text."""
    frame = data[
        [
            "label",
            "source_group_short_id",
            "source_group_hash",
            "target_category",
            "paired_light_status",
            "heavy_length",
            "light_length",
            "has_light",
        ]
    ].copy()
    frame["model_text"] = model_text.fillna("").astype(str)
    frame = frame[frame["model_text"].str.len().gt(0)].copy()
    return frame.reset_index(drop=True)


def build_variants(data: pd.DataFrame) -> tuple[list[Variant], dict[str, Any]]:
    """Build model variants where possible."""
    variants: list[Variant] = []
    build_info: dict[str, Any] = {}

    pair_text = build_pair_text(data)
    whole = variant_frame(data, pair_text)
    variants.append(
        Variant("whole_pair_kmer", "Whole-pair k-mer", "strict_labeled_all_rows", True, 1, whole)
    )
    build_info["whole_pair_kmer"] = {"row_count": int(len(whole)), "status": "available"}

    heavy_text = build_heavy_text(data)
    heavy = variant_frame(data, heavy_text)
    if len(heavy) >= MIN_VARIANT_ROWS:
        variants.append(
            Variant("heavy_only_kmer", "Heavy-only k-mer", "strict_labeled_all_rows", True, 2, heavy)
        )
        build_info["heavy_only_kmer"] = {"row_count": int(len(heavy)), "status": "available"}

    paired_mask = data["has_light"].astype(bool)
    paired = variant_frame(data.loc[paired_mask].copy(), pair_text.loc[paired_mask])
    if len(paired) >= MIN_VARIANT_ROWS and paired["source_group_short_id"].nunique() >= MIN_VARIANT_SOURCE_GROUPS:
        variants.append(
            Variant(
                "paired_only_whole_pair_kmer",
                "Paired-only whole-pair k-mer",
                "strict_labeled_paired_rows_only",
                False,
                3,
                paired,
            )
        )
        build_info["paired_only_whole_pair_kmer"] = {"row_count": int(len(paired)), "status": "available"}
    else:
        build_info["paired_only_whole_pair_kmer"] = {"row_count": int(len(paired)), "status": "skipped_too_few_rows_or_sources"}

    cdr_text, cdr_columns = build_cdr_text(data)
    cdr = variant_frame(data, cdr_text)
    if len(cdr) >= MIN_VARIANT_ROWS and cdr["source_group_short_id"].nunique() >= MIN_VARIANT_SOURCE_GROUPS:
        variants.append(
            Variant("cdr_region_kmer", "CDR/region k-mer", "strict_labeled_rows_with_cdr_text", True, 4, cdr)
        )
        build_info["cdr_region_kmer"] = {
            "row_count": int(len(cdr)),
            "status": "available",
            "cdr_columns_used": cdr_columns,
        }
        whole_plus_cdr = variant_frame(data, pair_text + "|REGION|" + cdr_text)
        variants.append(
            Variant(
                "whole_plus_cdr_kmer",
                "Whole-pair plus CDR k-mer",
                "strict_labeled_all_rows",
                True,
                5,
                whole_plus_cdr,
            )
        )
        build_info["whole_plus_cdr_kmer"] = {"row_count": int(len(whole_plus_cdr)), "status": "available"}
    else:
        build_info["cdr_region_kmer"] = {
            "row_count": int(len(cdr)),
            "status": "skipped_too_few_cdr_rows",
            "cdr_columns_used": cdr_columns,
        }
        build_info["whole_plus_cdr_kmer"] = {"status": "skipped_cdr_unavailable"}

    rbd_mask = data["target_category"].eq("RBD")
    rbd = variant_frame(data.loc[rbd_mask].copy(), pair_text.loc[rbd_mask])
    if len(rbd) >= MIN_VARIANT_ROWS and rbd["source_group_short_id"].nunique() >= MIN_VARIANT_SOURCE_GROUPS:
        variants.append(
            Variant(
                "rbd_or_target_region_subset_kmer",
                "RBD/target-region subset k-mer",
                "strict_labeled_rbd_subset",
                False,
                6,
                rbd,
            )
        )
        build_info["rbd_or_target_region_subset_kmer"] = {"row_count": int(len(rbd)), "status": "available"}
    else:
        build_info["rbd_or_target_region_subset_kmer"] = {
            "row_count": int(len(rbd)),
            "status": "skipped_too_few_rbd_rows_or_sources",
        }

    return variants, build_info


def base_valid_source_groups(data: pd.DataFrame) -> list[str]:
    """Return source groups valid on the full strict table."""
    valid = []
    for group_id, group in data.groupby("source_group_short_id", sort=True):
        counts = label_counts(group["label"])
        if (
            len(group) >= MIN_TEST_ROWS_PER_SOURCE
            and counts["0"] >= MIN_CLASS_ROWS_FOR_METRICS
            and counts["1"] >= MIN_CLASS_ROWS_FOR_METRICS
        ):
            valid.append(str(group_id))
    return valid


def make_model(max_features: int = MAX_FEATURES) -> Pipeline:
    """Create compact k-mer logistic model."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=max_features,
                ),
            ),
            ("classifier", LogisticRegression(max_iter=5000, class_weight="balanced")),
        ]
    )


def positive_scores(model: Pipeline | DummyClassifier, values: pd.Series | np.ndarray) -> np.ndarray:
    """Return positive-class probabilities."""
    class_list = list(model.classes_)
    if 1 not in class_list:
        return np.zeros(len(values), dtype=float)
    return model.predict_proba(values)[:, class_list.index(1)]


def metric_payload(y_true: pd.Series, y_score: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Compute binary classification metrics."""
    counts = label_counts(y_true)
    valid_auc = counts["0"] >= MIN_CLASS_ROWS_FOR_METRICS and counts["1"] >= MIN_CLASS_ROWS_FOR_METRICS
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    accuracy = float(accuracy_score(y_true, y_pred))
    balanced = float(balanced_accuracy_score(y_true, y_pred)) if y_true.nunique() == 2 else accuracy
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
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
    """Evaluate a majority baseline."""
    model = DummyClassifier(strategy="most_frequent")
    train_x = np.zeros((len(train_y), 1))
    test_x = np.zeros((len(test_y), 1))
    model.fit(train_x, train_y)
    score = positive_scores(model, test_x)
    pred = model.predict(test_x)
    return metric_payload(test_y, score, pred)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int]:
    """Fit model and return scores/predictions/max_features used."""
    for max_features in [MAX_FEATURES, FALLBACK_MAX_FEATURES]:
        try:
            model = make_model(max_features=max_features)
            model.fit(train["model_text"], train["label"])
            scores = positive_scores(model, test["model_text"])
            return scores, (scores >= 0.5).astype(int), max_features
        except ValueError:
            if max_features == FALLBACK_MAX_FEATURES:
                raise
    raise RuntimeError("unreachable")


def group_descriptor(group: pd.DataFrame) -> dict[str, Any]:
    """Return safe source group descriptors for failure analysis."""
    target_counts = count_dict(group["target_category"]) if "target_category" in group.columns else {}
    paired_counts = count_dict(group["paired_light_status"]) if "paired_light_status" in group.columns else {}
    return {
        "target_category_distribution": target_counts,
        "paired_light_missing_counts": paired_counts,
        "mean_heavy_length": float(group["heavy_length"].mean()) if len(group) else None,
        "mean_light_length": float(group["light_length"].mean()) if len(group) else None,
    }


def evaluate_leave_source_out(
    variant: Variant,
    candidate_source_groups: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate one variant on candidate source holdout groups."""
    per_group: dict[str, Any] = {}
    failure_rows: list[dict[str, Any]] = []
    skipped_reasons: dict[str, int] = {}
    data = variant.data
    for group_id in candidate_source_groups:
        test = data[data["source_group_short_id"].eq(group_id)].copy()
        descriptor = group_descriptor(test) if len(test) else {}
        test_counts = label_counts(test["label"]) if len(test) else {"0": 0, "1": 0}
        if len(test) < MIN_TEST_ROWS_PER_SOURCE:
            reason = "too_few_test_rows_for_variant"
            per_group[group_id] = {
                "status": "skipped",
                "reason": reason,
                "test_size": int(len(test)),
                "test_label_counts": test_counts,
                **descriptor,
            }
            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
            failure_rows.append(failure_row(variant, group_id, per_group[group_id]))
            continue
        if test_counts["0"] < MIN_CLASS_ROWS_FOR_METRICS or test_counts["1"] < MIN_CLASS_ROWS_FOR_METRICS:
            reason = "insufficient_test_class_counts"
            per_group[group_id] = {
                "status": "skipped",
                "reason": reason,
                "test_size": int(len(test)),
                "test_label_counts": test_counts,
                **descriptor,
            }
            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
            failure_rows.append(failure_row(variant, group_id, per_group[group_id]))
            continue
        train = data[data["source_group_short_id"].ne(group_id)].copy()
        train_counts = label_counts(train["label"])
        if train_counts["0"] == 0 or train_counts["1"] == 0:
            reason = "train_single_class"
            per_group[group_id] = {
                "status": "skipped",
                "reason": reason,
                "train_label_counts": train_counts,
                "test_size": int(len(test)),
                "test_label_counts": test_counts,
                **descriptor,
            }
            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
            failure_rows.append(failure_row(variant, group_id, per_group[group_id]))
            continue
        train_groups = set(train["source_group_short_id"])
        test_groups = set(test["source_group_short_id"])
        scores, preds, max_features = fit_predict(train, test)
        metrics = metric_payload(test["label"], scores, preds)
        metrics["majority_baseline"] = majority_baseline(train["label"], test["label"])
        result = {
            "status": "valid",
            "reason": "ok",
            "source_group_short_id": group_id,
            "group_overlap_count": int(len(train_groups & test_groups)),
            "train_size": int(len(train)),
            "test_size": int(len(test)),
            "train_label_counts": train_counts,
            "test_label_counts": test_counts,
            "max_features_used": int(max_features),
            "metrics": metrics,
            **descriptor,
        }
        per_group[group_id] = result
        failure_rows.append(failure_row(variant, group_id, result))
    return {
        "candidate_source_groups": candidate_source_groups,
        "evaluated_source_groups": [
            group_id for group_id, item in per_group.items() if item.get("status") == "valid"
        ],
        "per_group": per_group,
        "aggregate": aggregate_per_group(per_group),
        "skipped_reasons": skipped_reasons,
    }, failure_rows


def failure_row(variant: Variant, group_id: str, result: dict[str, Any]) -> dict[str, Any]:
    """Build one safe failure-analysis row."""
    metrics = result.get("metrics", {})
    counts = result.get("test_label_counts", {})
    return {
        "model_variant": variant.model_id,
        "source_group_short_id": group_id,
        "status": result.get("status"),
        "skip_reason": result.get("reason"),
        "test_row_count": int(result.get("test_size", 0) or 0),
        "positive_count": int(counts.get("1", 0) or 0),
        "negative_count": int(counts.get("0", 0) or 0),
        "target_category_distribution": json.dumps(result.get("target_category_distribution", {}), sort_keys=True),
        "paired_light_missing_counts": json.dumps(result.get("paired_light_missing_counts", {}), sort_keys=True),
        "mean_heavy_length": result.get("mean_heavy_length"),
        "mean_light_length": result.get("mean_light_length"),
        "roc_auc": metrics.get("roc_auc"),
        "pr_auc": metrics.get("pr_auc"),
        "f1": metrics.get("f1"),
    }


def aggregate_per_group(per_group: dict[str, Any]) -> dict[str, Any]:
    """Aggregate source holdout metrics."""
    valid = [
        result
        for result in per_group.values()
        if result.get("status") == "valid" and result.get("metrics", {}).get("valid_auc_metrics")
    ]
    if not valid:
        return {
            "valid_heldout_source_group_count": 0,
            "skipped_group_count": int(len(per_group)),
            "macro_mean": {},
            "weighted_mean_by_test_size": {},
        }
    metric_names = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    weights = np.asarray([result["test_size"] for result in valid], dtype=float)
    macro: dict[str, float] = {}
    weighted: dict[str, float] = {}
    for metric in metric_names:
        values = np.asarray([result["metrics"][metric] for result in valid], dtype=float)
        macro[metric] = float(np.mean(values))
        weighted[metric] = float(np.average(values, weights=weights))
    return {
        "valid_heldout_source_group_count": int(len(valid)),
        "skipped_group_count": int(len(per_group) - len(valid)),
        "macro_mean": macro,
        "weighted_mean_by_test_size": weighted,
    }


def source_grouped_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Create source-grouped fallback split."""
    groups = data["source_group_short_id"].astype(str)
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
        train_groups = set(train["source_group_short_id"])
        test_groups = set(test["source_group_short_id"])
        if train_groups & test_groups:
            continue
        return train, test, {
            "strategy": "GroupShuffleSplit",
            "group_column": "source_group_short_id",
            "random_state": RANDOM_STATE + offset,
            "test_size": TEST_SIZE,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_label_counts": label_counts(train["label"]),
            "test_label_counts": label_counts(test["label"]),
            "train_group_count": int(len(train_groups)),
            "test_group_count": int(len(test_groups)),
            "group_overlap_count": 0,
        }
    raise ValueError("No valid source-grouped split was available.")


def calibration_curve_payload(y_true: pd.Series, y_score: np.ndarray) -> tuple[list[dict[str, Any]], float]:
    """Build calibration curve and ECE approximation."""
    y = y_true.to_numpy()
    bins = np.linspace(0.0, 1.0, 11)
    rows = []
    ece = 0.0
    for index in range(10):
        low = bins[index]
        high = bins[index + 1]
        if index == 9:
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


def threshold_metrics(y_true: pd.Series, y_score: np.ndarray) -> list[dict[str, Any]]:
    """Build threshold/abstention table."""
    rows = []
    total = len(y_true)
    for threshold in THRESHOLDS:
        pred = (y_score >= threshold).astype(int)
        selected = int(pred.sum())
        rows.append(
            {
                "threshold": float(threshold),
                "predicted_positive_count": selected,
                "coverage_fraction": float(selected / total) if total else 0.0,
                "precision": float(precision_score(y_true, pred, zero_division=0)),
                "recall": float(recall_score(y_true, pred, zero_division=0)),
                "f1": float(f1_score(y_true, pred, zero_division=0)),
            }
        )
    return rows


def topk_precision(y_true: pd.Series, y_score: np.ndarray) -> dict[str, Any]:
    """Compute top-k precision."""
    order = np.argsort(-y_score)
    labels = y_true.to_numpy()
    output: dict[str, Any] = {}
    for k in TOP_K_VALUES:
        if len(labels) < k:
            continue
        selected = labels[order[:k]]
        output[str(k)] = {
            "selected_count": int(k),
            "positive_count": int(selected.sum()),
            "precision": float(np.mean(selected)),
        }
    return output


def best_threshold(thresholds: list[dict[str, Any]]) -> dict[str, Any]:
    """Identify best high-confidence threshold."""
    candidates = [
        row
        for row in thresholds
        if row["precision"] >= 0.80 and row["predicted_positive_count"] > 0
    ]
    if candidates:
        selected = max(candidates, key=lambda row: (row["threshold"], row["recall"]))
        return {
            "selection_rule": "highest threshold with precision >= 0.80 and at least one selected record",
            **selected,
        }
    candidates = [row for row in thresholds if row["predicted_positive_count"] >= 25]
    if candidates:
        selected = max(candidates, key=lambda row: (row["precision"], row["threshold"]))
        return {
            "selection_rule": "highest precision with at least 25 selected records",
            **selected,
        }
    selected = max(thresholds, key=lambda row: (row["precision"], row["predicted_positive_count"]))
    return {"selection_rule": "highest precision among evaluated thresholds", **selected}


def evaluate_calibration(variant: Variant) -> dict[str, Any]:
    """Evaluate calibration/abstention for one variant on source-grouped split."""
    train, test, split = source_grouped_split(variant.data)
    scores, preds, max_features = fit_predict(train, test)
    metrics = metric_payload(test["label"], scores, preds)
    curve, ece = calibration_curve_payload(test["label"], scores)
    thresholds = threshold_metrics(test["label"], scores)
    return {
        "split": {**split, "metrics": metrics, "max_features_used": int(max_features)},
        "brier_score": float(brier_score_loss(test["label"], scores)),
        "expected_calibration_error": ece,
        "calibration_curve": curve,
        "threshold_metrics": thresholds,
        "topk_precision": topk_precision(test["label"], scores),
        "best_high_confidence_threshold": best_threshold(thresholds),
        "calibrated_model": {
            "status": "skipped",
            "reason": "No calibration model was fit on final test rows; uncalibrated ranking diagnostics reported.",
        },
    }


def failure_correlations(rows: list[dict[str, Any]], selected_model: str) -> dict[str, Any]:
    """Compute simple correlations with selected-model source failures."""
    table = pd.DataFrame(rows)
    table = table[(table["model_variant"] == selected_model) & table["roc_auc"].notna()].copy()
    if table.empty:
        return {}
    table["positive_fraction"] = table["positive_count"] / (
        table["positive_count"] + table["negative_count"]
    )
    signals = ["test_row_count", "positive_fraction", "mean_heavy_length", "mean_light_length"]
    output: dict[str, Any] = {}
    for metric in ["roc_auc", "pr_auc"]:
        output[metric] = {}
        for signal in signals:
            valid = table[[metric, signal]].dropna()
            if len(valid) >= 3 and valid[signal].nunique() > 1:
                output[metric][signal] = float(valid[metric].corr(valid[signal]))
            else:
                output[metric][signal] = None
    return output


def select_model(comparison_rows: list[dict[str, Any]], calibration: dict[str, Any]) -> dict[str, Any]:
    """Select most defensible model by source-holdout metrics."""
    eligible = [
        row
        for row in comparison_rows
        if row["selection_eligible"]
        and row["valid_heldout_source_group_count"] > 0
        and row["weighted_pr_auc"] is not None
        and row["weighted_roc_auc"] is not None
    ]
    if not eligible:
        return {
            "selected_model": None,
            "selection_reason": "No selection-eligible source-holdout model had valid metrics.",
            "meaningful_improvement_over_previous": False,
        }
    eligible = sorted(
        eligible,
        key=lambda row: (
            row["weighted_pr_auc"],
            row["weighted_roc_auc"],
            -row["brier_score"] if row["brier_score"] is not None else -999,
            -row["simplicity_rank"],
        ),
        reverse=True,
    )
    selected = eligible[0]
    previous = previous_source_baseline()
    previous_pr = previous.get("macro_pr_auc")
    previous_roc = previous.get("macro_roc_auc")
    improvement_pr = (
        selected["weighted_pr_auc"] - previous_pr
        if isinstance(previous_pr, (int, float))
        else None
    )
    improvement_roc = (
        selected["weighted_roc_auc"] - previous_roc
        if isinstance(previous_roc, (int, float))
        else None
    )
    meaningful = bool(
        improvement_pr is not None
        and improvement_roc is not None
        and improvement_pr >= 0.02
        and improvement_roc >= 0.02
    )
    return {
        "selected_model": selected["model_variant"],
        "selection_priority": [
            "highest weighted leave-source-out PR-AUC",
            "highest weighted leave-source-out ROC-AUC",
            "lower Brier score",
            "simpler model if metrics are similar",
        ],
        "selection_reason": (
            f"Selected {selected['model_variant']} by weighted leave-source-out "
            f"PR-AUC={selected['weighted_pr_auc']:.4f}, "
            f"ROC-AUC={selected['weighted_roc_auc']:.4f}, "
            f"Brier={selected['brier_score']:.4f}."
        ),
        "previous_source_holdout_baseline": previous,
        "improvement_over_previous": {
            "weighted_pr_auc_minus_previous_macro_pr_auc": improvement_pr,
            "weighted_roc_auc_minus_previous_macro_roc_auc": improvement_roc,
        },
        "meaningful_improvement_over_previous": meaningful,
        "use_recommendation": (
            "Treat scores as ranking/prioritization evidence rather than calibrated prospective prediction."
        ),
    }


def previous_source_baseline() -> dict[str, Any]:
    """Load previous source-holdout baseline metrics."""
    baseline = load_json(SOURCE_BASELINE_PATH) or {}
    aggregate = baseline.get("leave_source_out", {}).get("aggregate", {})
    macro = aggregate.get("macro_mean", {})
    weighted = aggregate.get("weighted_mean_by_test_size", {})
    return {
        "macro_roc_auc": macro.get("roc_auc"),
        "macro_pr_auc": macro.get("pr_auc"),
        "weighted_roc_auc": weighted.get("roc_auc"),
        "weighted_pr_auc": weighted.get("pr_auc"),
        "valid_heldout_source_group_count": aggregate.get("valid_heldout_source_group_count"),
    }


def comparison_row(variant: Variant, holdout: dict[str, Any], calibration: dict[str, Any]) -> dict[str, Any]:
    """Build one comparison table row."""
    aggregate = holdout["aggregate"]
    macro = aggregate.get("macro_mean", {})
    weighted = aggregate.get("weighted_mean_by_test_size", {})
    split_metrics = calibration["split"]["metrics"]
    return {
        "model_variant": variant.model_id,
        "display_name": variant.display_name,
        "row_subset": variant.row_subset,
        "selection_eligible": bool(variant.selection_eligible),
        "simplicity_rank": int(variant.simplicity_rank),
        "row_count": int(len(variant.data)),
        "source_group_count": int(variant.data["source_group_short_id"].nunique()),
        "valid_heldout_source_group_count": aggregate.get("valid_heldout_source_group_count", 0),
        "macro_roc_auc": macro.get("roc_auc"),
        "macro_pr_auc": macro.get("pr_auc"),
        "weighted_roc_auc": weighted.get("roc_auc"),
        "weighted_pr_auc": weighted.get("pr_auc"),
        "fallback_roc_auc": split_metrics.get("roc_auc"),
        "fallback_pr_auc": split_metrics.get("pr_auc"),
        "brier_score": calibration.get("brier_score"),
        "expected_calibration_error": calibration.get("expected_calibration_error"),
        "best_threshold": calibration["best_high_confidence_threshold"]["threshold"],
        "best_threshold_precision": calibration["best_high_confidence_threshold"]["precision"],
        "best_threshold_recall": calibration["best_high_confidence_threshold"]["recall"],
        "best_threshold_coverage": calibration["best_high_confidence_threshold"]["coverage_fraction"],
    }


def run_all() -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Run all source-robust model selection diagnostics."""
    data, data_audit = load_strict_data()
    variants, build_info = build_variants(data)
    candidates = base_valid_source_groups(data)
    holdouts: dict[str, Any] = {}
    calibrations: dict[str, Any] = {}
    failure_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    for variant in variants:
        holdout, rows = evaluate_leave_source_out(variant, candidates)
        calibration = evaluate_calibration(variant)
        holdouts[variant.model_id] = holdout
        calibrations[variant.model_id] = calibration
        failure_rows.extend(rows)
        comparison_rows.append(comparison_row(variant, holdout, calibration))

    selection = select_model(comparison_rows, calibrations)
    selected_model = selection.get("selected_model")
    failure_summary = {
        "best_generalizing_source_groups": best_worst_groups(failure_rows, selected_model, best=True),
        "worst_generalizing_source_groups": best_worst_groups(failure_rows, selected_model, best=False),
        "failure_correlations": failure_correlations(failure_rows, selected_model) if selected_model else {},
    }
    metrics = {
        "status": "available",
        "data_audit": data_audit,
        "variant_build_info": build_info,
        "candidate_source_groups": candidates,
        "candidate_source_group_count": int(len(candidates)),
        "model": {
            "vectorizer": 'TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=2, max_features=50000)',
            "classifier": 'LogisticRegression(max_iter=5000, class_weight="balanced")',
            "neural_training_used": False,
            "gpu_used": False,
        },
        "holdout_results": holdouts,
        "calibration_results": calibrations,
        "model_selection": selection,
        "failure_analysis": failure_summary,
        "quality_gates": {
            "raw_sequence_strings_written": False,
            "raw_source_urls_or_dois_written": False,
            "same_candidate_source_groups_used_where_applicable": True,
            "vectorizers_fit_only_on_training_data": True,
            "group_overlap_zero": all(
                cal["split"]["group_overlap_count"] == 0 for cal in calibrations.values()
            ),
            "selected_model_justified_by_source_holdout": bool(selection.get("selected_model")),
            "no_prospective_therapeutic_prediction_claim": True,
        },
        "artifacts": {
            "report": relpath(REPORT_PATH),
            "metrics_json": relpath(METRICS_PATH),
            "comparison_csv": relpath(COMPARISON_CSV_PATH),
            "failure_analysis_csv": relpath(FAILURE_CSV_PATH),
            "comparison_figure": relpath(COMPARISON_FIGURE_PATH),
            "pr_auc_figure": relpath(PR_FIGURE_PATH),
            "roc_auc_figure": relpath(ROC_FIGURE_PATH),
            "abstention_figure": relpath(ABSTENTION_FIGURE_PATH),
            "failure_summary_figure": relpath(FAILURE_FIGURE_PATH),
        },
    }
    return metrics, pd.DataFrame(comparison_rows), pd.DataFrame(failure_rows)


def best_worst_groups(rows: list[dict[str, Any]], selected_model: str | None, best: bool) -> list[dict[str, Any]]:
    """Return best or worst selected-model source groups by PR-AUC."""
    if not selected_model:
        return []
    table = pd.DataFrame(rows)
    table = table[(table["model_variant"] == selected_model) & table["pr_auc"].notna()].copy()
    if table.empty:
        return []
    table = table.sort_values(["pr_auc", "roc_auc"], ascending=[not best, not best])
    output = []
    for _, row in table.head(5).iterrows():
        output.append(
            {
                "source_group_short_id": row["source_group_short_id"],
                "test_row_count": int(row["test_row_count"]),
                "positive_count": int(row["positive_count"]),
                "negative_count": int(row["negative_count"]),
                "roc_auc": float(row["roc_auc"]),
                "pr_auc": float(row["pr_auc"]),
            }
        )
    return output


def fmt(value: Any) -> str:
    """Format optional metric."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def build_report(metrics: dict[str, Any], comparison: pd.DataFrame) -> str:
    """Build Markdown report."""
    selection = metrics["model_selection"]
    previous = selection.get("previous_source_holdout_baseline", {})
    selected_model = selection.get("selected_model")
    selected_row = (
        comparison[comparison["model_variant"].eq(selected_model)].iloc[0].to_dict()
        if selected_model and not comparison[comparison["model_variant"].eq(selected_model)].empty
        else {}
    )
    lines = [
        "# Source-Robust Model Selection",
        "",
        "This module evaluates CPU-only compact k-mer models under source/study",
        "holdout controls, calibration diagnostics, and abstention analysis. It does",
        "not train neural models and does not write raw source strings or sequences.",
        "",
        "## Data Audit",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Input shape rows | {metrics['data_audit']['shape'][0]} |",
        f"| Input shape columns | {metrics['data_audit']['shape'][1]} |",
        f"| Label 0 count | {metrics['data_audit']['label_counts']['0']} |",
        f"| Label 1 count | {metrics['data_audit']['label_counts']['1']} |",
        f"| Source groups | {metrics['data_audit']['source_group_count']} |",
        f"| Candidate source-holdout groups | {metrics['candidate_source_group_count']} |",
        "",
        "## Model Comparison",
        "",
        (
            "| Model | Selection eligible | Rows | Valid source groups | Weighted PR-AUC | "
            "Weighted ROC-AUC | Brier | Best threshold | Precision | Recall |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in comparison.sort_values(["selection_eligible", "weighted_pr_auc"], ascending=[False, False]).iterrows():
        lines.append(
            f"| {row['model_variant']} | {str(bool(row['selection_eligible'])).lower()} | "
            f"{int(row['row_count'])} | {int(row['valid_heldout_source_group_count'])} | "
            f"{fmt(row['weighted_pr_auc'])} | {fmt(row['weighted_roc_auc'])} | "
            f"{fmt(row['brier_score'])} | {fmt(row['best_threshold'])} | "
            f"{fmt(row['best_threshold_precision'])} | {fmt(row['best_threshold_recall'])} |"
        )
    lines.extend(
        [
            "",
            "## Selection",
            "",
            f"Selected model: `{selected_model}`.",
            "",
            selection.get("selection_reason", ""),
            "",
            (
                f"Previous source-holdout baseline macro ROC-AUC/PR-AUC: "
                f"{fmt(previous.get('macro_roc_auc'))}/{fmt(previous.get('macro_pr_auc'))}."
            ),
            (
                f"Selected weighted source-holdout ROC-AUC/PR-AUC: "
                f"{fmt(selected_row.get('weighted_roc_auc'))}/{fmt(selected_row.get('weighted_pr_auc'))}."
            ),
            (
                "Meaningful improvement over previous baseline: "
                f"{str(selection.get('meaningful_improvement_over_previous')).lower()}."
            ),
            "",
            selection.get("use_recommendation", ""),
            "",
            "## Failure Analysis",
            "",
            "Best and worst groups are listed by sanitized source ID only.",
            "",
            "### Best-Generalizing Source Groups",
            "",
            "| Source group | Test rows | Positives | Negatives | ROC-AUC | PR-AUC |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in metrics["failure_analysis"]["best_generalizing_source_groups"]:
        lines.append(
            f"| {row['source_group_short_id']} | {row['test_row_count']} | "
            f"{row['positive_count']} | {row['negative_count']} | "
            f"{fmt(row['roc_auc'])} | {fmt(row['pr_auc'])} |"
        )
    lines.extend(
        [
            "",
            "### Worst-Generalizing Source Groups",
            "",
            "| Source group | Test rows | Positives | Negatives | ROC-AUC | PR-AUC |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in metrics["failure_analysis"]["worst_generalizing_source_groups"]:
        lines.append(
            f"| {row['source_group_short_id']} | {row['test_row_count']} | "
            f"{row['positive_count']} | {row['negative_count']} | "
            f"{fmt(row['roc_auc'])} | {fmt(row['pr_auc'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            source_robust_interpretation(metrics, selected_row),
            "",
            "## Artifacts",
            "",
        ]
    )
    for path in metrics["artifacts"].values():
        lines.append(f"- `{path}`")
    lines.append("")
    return "\n".join(lines)


def source_robust_interpretation(metrics: dict[str, Any], selected_row: dict[str, Any]) -> str:
    """Build concise interpretation."""
    selection = metrics["model_selection"]
    selected_model = selection.get("selected_model")
    cdr_rows = [
        row
        for row in source_robust_rows(metrics)
        if row.get("model_variant") in {"cdr_region_kmer", "whole_plus_cdr_kmer"}
    ]
    cdr_best = max(
        (row.get("weighted_pr_auc") or -1 for row in cdr_rows),
        default=None,
    )
    selected_pr = selected_row.get("weighted_pr_auc")
    if selection.get("meaningful_improvement_over_previous"):
        first = "Source-robust model selection improved cross-source performance meaningfully."
    else:
        first = (
            "No model materially improved source-holdout performance enough to remove "
            "concern about source/study effects."
        )
    if cdr_best is not None and selected_pr is not None and cdr_best >= selected_pr - SIMILAR_METRIC_MARGIN:
        cdr_text = "CDR/region models were competitive for source robustness."
    else:
        cdr_text = "CDR/region models did not clearly improve source robustness."
    return (
        f"{first} {cdr_text} The selected score should still be treated as a "
        "ranking/prioritization signal rather than calibrated prospective therapeutic "
        "prediction. High-confidence review thresholds should be chosen by precision "
        "and coverage tradeoff, not by assuming probabilities are perfectly calibrated."
    )


def source_robust_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten comparison rows from metrics when needed."""
    rows = []
    for model_id, holdout in metrics.get("holdout_results", {}).items():
        aggregate = holdout.get("aggregate", {})
        calibration = metrics.get("calibration_results", {}).get(model_id, {})
        rows.append(
            {
                "model_variant": model_id,
                "weighted_pr_auc": aggregate.get("weighted_mean_by_test_size", {}).get("pr_auc"),
                "weighted_roc_auc": aggregate.get("weighted_mean_by_test_size", {}).get("roc_auc"),
                "brier_score": calibration.get("brier_score"),
            }
        )
    return rows


def save_figures(comparison: pd.DataFrame, failure: pd.DataFrame, metrics: dict[str, Any]) -> None:
    """Save model comparison and failure-analysis figures."""
    if comparison.empty:
        return
    for metric, path, title, color in [
        ("weighted_pr_auc", PR_FIGURE_PATH, "Weighted leave-source-out PR-AUC", "#4C78A8"),
        ("weighted_roc_auc", ROC_FIGURE_PATH, "Weighted leave-source-out ROC-AUC", "#F58518"),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        plot = comparison.sort_values(metric, ascending=True)
        fig, ax = plt.subplots(figsize=(8.5, max(4, 0.42 * len(plot) + 1.2)))
        ax.barh(plot["model_variant"], plot[metric], color=color)
        ax.set_xlim(0, 1)
        ax.set_xlabel(metric)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)

    plot = comparison.sort_values("weighted_pr_auc", ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.46 * len(plot) + 1.2)))
    y = np.arange(len(plot))
    ax.barh(y - 0.16, plot["weighted_pr_auc"], height=0.32, label="PR-AUC", color="#4C78A8")
    ax.barh(y + 0.16, plot["weighted_roc_auc"], height=0.32, label="ROC-AUC", color="#F58518")
    ax.set_yticks(y)
    ax.set_yticklabels(plot["model_variant"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Metric")
    ax.set_title("Source-robust model comparison")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(COMPARISON_FIGURE_PATH, dpi=200)
    plt.close(fig)

    selected = metrics["model_selection"].get("selected_model")
    calibration = metrics["calibration_results"].get(selected, {}) if selected else {}
    threshold_df = pd.DataFrame(calibration.get("threshold_metrics", []))
    if not threshold_df.empty:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(threshold_df["coverage_fraction"], threshold_df["precision"], marker="o")
        for _, row in threshold_df.iterrows():
            ax.text(row["coverage_fraction"], row["precision"], f"{row['threshold']:.1f}", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Coverage fraction")
        ax.set_ylabel("Precision")
        ax.set_title("Abstention precision vs coverage")
        fig.tight_layout()
        fig.savefig(ABSTENTION_FIGURE_PATH, dpi=200)
        plt.close(fig)

    if not failure.empty and selected:
        subset = failure[(failure["model_variant"] == selected) & failure["pr_auc"].notna()].copy()
        if not subset.empty:
            subset = subset.sort_values("pr_auc", ascending=True)
            fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(subset) + 1.2)))
            ax.barh(subset["source_group_short_id"], subset["pr_auc"], color="#E45756")
            ax.set_xlim(0, 1)
            ax.set_xlabel("PR-AUC")
            ax.set_title("Selected-model source failure summary")
            fig.tight_layout()
            fig.savefig(FAILURE_FIGURE_PATH, dpi=200)
            plt.close(fig)


def main() -> int:
    metrics, comparison, failure = run_all()
    comparison.to_csv(COMPARISON_CSV_PATH, index=False)
    failure.to_csv(FAILURE_CSV_PATH, index=False)
    write_json(METRICS_PATH, metrics)
    write_text(REPORT_PATH, build_report(metrics, comparison))
    save_figures(comparison, failure, metrics)

    selected = metrics["model_selection"].get("selected_model")
    selected_row = comparison[comparison["model_variant"].eq(selected)].iloc[0].to_dict()
    previous = metrics["model_selection"].get("previous_source_holdout_baseline", {})
    print(
        f"source_groups={metrics['data_audit']['source_group_count']}; "
        f"valid_heldout_source_groups={selected_row['valid_heldout_source_group_count']}; "
        f"model_comparison_table={comparison.to_dict(orient='records')}; "
        f"selected_model={selected}; "
        f"previous_source_holdout_roc_auc={fmt(previous.get('macro_roc_auc'))}; "
        f"previous_source_holdout_pr_auc={fmt(previous.get('macro_pr_auc'))}; "
        f"new_selected_source_holdout_roc_auc={fmt(selected_row.get('weighted_roc_auc'))}; "
        f"new_selected_source_holdout_pr_auc={fmt(selected_row.get('weighted_pr_auc'))}; "
        f"brier_score={fmt(selected_row.get('brier_score'))}; "
        f"best_high_confidence_threshold={metrics['calibration_results'][selected]['best_high_confidence_threshold']}; "
        f"output_paths={metrics['artifacts']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
