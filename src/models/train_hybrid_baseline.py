"""Train hybrid sequence-classification baselines.

This script benchmarks combinations of k-mer TF-IDF features, cached AbLang2
embeddings, and simple row-level features on existing labeled rows.

Run from the project root:

    python src/models/train_hybrid_baseline.py
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits


# Matplotlib needs a writable cache directory in this environment.
MPLCONFIGDIR = Path("/tmp") / "antibody_prioritization_mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt
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
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "neutral_sequence_classification_ml.csv"
PAIR_EMBEDDING_PATH = PROJECT_ROOT / "data" / "processed" / "embeddings_ablang2_pair.npy"
KMER_METRICS_PATH = PROJECT_ROOT / "reports" / "metrics" / "grouped_validation_metrics.json"
EMBEDDING_METRICS_PATH = (
    PROJECT_ROOT / "reports" / "metrics" / "embedding_baseline_metrics.json"
)

REPORT_PATH = PROJECT_ROOT / "reports" / "hybrid_baseline_report.md"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics" / "hybrid_baseline_metrics.json"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
ROC_AUC_FIGURE_PATH = FIGURE_DIR / "hybrid_roc_auc_comparison.png"
PR_AUC_FIGURE_PATH = FIGURE_DIR / "hybrid_pr_auc_comparison.png"
F1_FIGURE_PATH = FIGURE_DIR / "hybrid_f1_comparison.png"
MODEL_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODEL_DIR / "hybrid_best_model.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_GROUP_SPLIT_ATTEMPTS = 50
GROUP_COLUMN = "group_feature_v"

MOSTLY_MISSING_THRESHOLD = 0.50
NEAR_ROW_UNIQUE_ROW_THRESHOLD = 0.80
NEAR_ROW_UNIQUE_NON_MISSING_THRESHOLD = 0.90

FEATURE_SETS = {
    "kmer_only": {"kmer": True, "ablang2_pair": False, "simple": False},
    "ablang2_pair_only": {"kmer": False, "ablang2_pair": True, "simple": False},
    "simple_features_only": {"kmer": False, "ablang2_pair": False, "simple": True},
    "hybrid_kmer_plus_simple": {"kmer": True, "ablang2_pair": False, "simple": True},
    "hybrid_kmer_plus_ablang2": {"kmer": True, "ablang2_pair": True, "simple": False},
    "hybrid_all": {"kmer": True, "ablang2_pair": True, "simple": True},
}

N_JOBS = int(os.environ.get("HYBRID_N_JOBS", str(min(len(FEATURE_SETS), os.cpu_count() or 1))))

SIMPLE_FEATURE_COLUMNS = [
    "heavy_length",
    "light_length",
    "cdrh3_length",
    "cdrl3_length",
    "has_light",
    "is_nanobody_like",
    "has_structure",
    "targets_rbd",
    "targets_spike",
    "targets_ntd",
]


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV as text while preserving blank fields as blanks."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def normalized_text(values: pd.Series) -> pd.Series:
    """Normalize text for feature and split checks."""
    return values.fillna("").astype(str).str.strip()


def normalized_sequence(values: pd.Series) -> pd.Series:
    """Normalize sequence-like text for length features."""
    return (
        values.fillna("")
        .astype(str)
        .str.replace(r"\s+", "", regex=True)
        .str.upper()
    )


def optional_column(data: pd.DataFrame, column: str) -> pd.Series:
    """Return a column or aligned blanks."""
    if column in data.columns:
        return data[column]
    return pd.Series([""] * len(data), index=data.index)


def numeric_or_none(data: pd.DataFrame, column: str) -> pd.Series | None:
    """Return an existing numeric/boolean column as float values, if present."""
    if column not in data.columns:
        return None

    values = normalized_text(data[column]).str.lower()
    mapped = values.map(
        {
            "true": 1.0,
            "false": 0.0,
            "yes": 1.0,
            "no": 0.0,
            "1": 1.0,
            "0": 0.0,
        }
    )
    numeric = pd.to_numeric(data[column], errors="coerce")
    combined = numeric.where(numeric.notna(), mapped)
    return combined.fillna(0.0).astype(float)


def length_feature(
    data: pd.DataFrame,
    feature_name: str,
    source_column: str,
) -> pd.Series:
    """Use an existing length feature, or compute length from a neutral source."""
    existing = numeric_or_none(data, feature_name)
    if existing is not None:
        return existing
    return normalized_sequence(optional_column(data, source_column)).str.len().astype(float)


def boolean_feature(
    data: pd.DataFrame,
    feature_name: str,
    source_column: str,
) -> pd.Series:
    """Use an existing boolean feature, or compute presence from a neutral source."""
    existing = numeric_or_none(data, feature_name)
    if existing is not None:
        return existing
    return normalized_text(optional_column(data, source_column)).ne("").astype(float)


def target_feature(
    data: pd.DataFrame,
    feature_name: str,
    pattern: str,
) -> pd.Series:
    """Use an existing target flag, or compute one from neutral target metadata."""
    existing = numeric_or_none(data, feature_name)
    if existing is not None:
        return existing
    values = normalized_text(optional_column(data, "metadata_target_region")).str.lower()
    return values.str.contains(pattern, regex=True, na=False).astype(float)


def build_simple_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build the fixed simple feature matrix requested for the benchmark."""
    has_light_existing = numeric_or_none(data, "has_light")
    if has_light_existing is None:
        has_light = normalized_sequence(optional_column(data, "sequence_b")).ne("").astype(float)
    else:
        has_light = has_light_existing

    nanobody_existing = numeric_or_none(data, "is_nanobody_like")
    if nanobody_existing is None:
        is_nanobody_like = (1.0 - has_light).astype(float)
    else:
        is_nanobody_like = nanobody_existing

    simple = pd.DataFrame(
        {
            "heavy_length": length_feature(data, "heavy_length", "sequence_a"),
            "light_length": length_feature(data, "light_length", "sequence_b"),
            "cdrh3_length": length_feature(
                data,
                "cdrh3_length",
                "group_feature_cdr3",
            ),
            "cdrl3_length": length_feature(
                data,
                "cdrl3_length",
                "group_feature_b_cdr3",
            ),
            "has_light": has_light,
            "is_nanobody_like": is_nanobody_like,
            "has_structure": boolean_feature(
                data,
                "has_structure",
                "metadata_structure",
            ),
            "targets_rbd": target_feature(data, "targets_rbd", r"\brbd\b"),
            "targets_spike": target_feature(data, "targets_spike", r"\bspike\b|^s$"),
            "targets_ntd": target_feature(data, "targets_ntd", r"\bntd\b"),
        }
    )
    return simple[SIMPLE_FEATURE_COLUMNS].astype(float)


def load_inputs() -> tuple[pd.DataFrame, np.ndarray]:
    """Load neutral model input rows plus cached pair embeddings."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_PATH.relative_to(PROJECT_ROOT)}")
    if not PAIR_EMBEDDING_PATH.exists():
        raise FileNotFoundError(
            f"Missing pair embeddings: {PAIR_EMBEDDING_PATH.relative_to(PROJECT_ROOT)}. "
            "Run python src/models/embed_with_ablang2.py first."
        )

    raw_data = read_csv(INPUT_PATH)
    pair_embeddings = np.load(PAIR_EMBEDDING_PATH)
    if len(raw_data) != pair_embeddings.shape[0]:
        raise ValueError(
            "Neutral input rows and pair embedding rows do not match: "
            f"{len(raw_data)} vs {pair_embeddings.shape[0]}."
        )

    required_columns = ["sequence_pair_text", "label"]
    missing_columns = [column for column in required_columns if column not in raw_data.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise KeyError(f"Missing required neutral column(s): {missing_text}")

    data = raw_data.copy()
    data = data[normalized_text(data["label"]).ne("")].copy()
    data["label"] = data["label"].astype(int)
    data["sequence_pair_text"] = data["sequence_pair_text"].fillna("").astype(str)

    empty_pair_text = int(data["sequence_pair_text"].str.len().eq(0).sum())
    if empty_pair_text:
        raise ValueError(f"sequence_pair_text has {empty_pair_text} empty rows.")

    unexpected_labels = sorted(set(data["label"]) - {0, 1})
    if unexpected_labels:
        raise ValueError(f"Expected binary labels 0/1, found: {unexpected_labels}")

    label_count_values = data["label"].value_counts()
    if len(label_count_values) != 2 or label_count_values.min() < 2:
        raise ValueError("Train/test evaluation requires at least two rows per label.")

    selected_positions = data.index.to_numpy()
    data = data.reset_index(drop=True)
    data[SIMPLE_FEATURE_COLUMNS] = build_simple_features(data)
    pair_embeddings = pair_embeddings[selected_positions]

    return data, pair_embeddings


def label_counts(labels: pd.Series) -> dict[str, int]:
    """Return stable binary label counts for JSON and Markdown output."""
    counts = labels.value_counts().sort_index()
    return {"0": int(counts.get(0, 0)), "1": int(counts.get(1, 0))}


def group_column_status(data: pd.DataFrame) -> dict[str, Any]:
    """Decide whether group_feature_v can support grouped validation."""
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
    """Create a meaningful group_feature_v split or raise ValueError."""
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


def make_classifier() -> LogisticRegression:
    """Create the shared logistic-regression classifier."""
    return LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="liblinear",
        random_state=RANDOM_STATE,
    )


def fit_feature_transformers(
    feature_set: str,
    data: pd.DataFrame,
    pair_embeddings: np.ndarray,
    train_idx: np.ndarray,
) -> dict[str, Any]:
    """Fit vectorizers/scalers for one feature set."""
    config = FEATURE_SETS[feature_set]
    components: dict[str, Any] = {"feature_set": feature_set, "config": config}

    if config["kmer"]:
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2)
        vectorizer.fit(data.iloc[train_idx]["sequence_pair_text"])
        components["kmer_vectorizer"] = vectorizer

    if config["ablang2_pair"]:
        embedding_scaler = StandardScaler()
        embedding_scaler.fit(pair_embeddings[train_idx])
        components["embedding_scaler"] = embedding_scaler

    if config["simple"]:
        simple_scaler = StandardScaler()
        simple_scaler.fit(data.iloc[train_idx][SIMPLE_FEATURE_COLUMNS])
        components["simple_scaler"] = simple_scaler

    return components


def transform_features(
    components: dict[str, Any],
    data: pd.DataFrame,
    pair_embeddings: np.ndarray,
    indices: np.ndarray,
) -> sparse.csr_matrix:
    """Transform one row subset into a sparse design matrix."""
    config = components["config"]
    parts = []

    if config["kmer"]:
        kmer_part = components["kmer_vectorizer"].transform(
            data.iloc[indices]["sequence_pair_text"]
        )
        parts.append(kmer_part)

    if config["ablang2_pair"]:
        embedding_part = components["embedding_scaler"].transform(pair_embeddings[indices])
        parts.append(sparse.csr_matrix(embedding_part))

    if config["simple"]:
        simple_part = components["simple_scaler"].transform(
            data.iloc[indices][SIMPLE_FEATURE_COLUMNS]
        )
        parts.append(sparse.csr_matrix(simple_part))

    if not parts:
        raise ValueError("Feature set has no active features.")
    if len(parts) == 1:
        return parts[0].tocsr()
    return sparse.hstack(parts, format="csr")


def fit_design_matrices(
    feature_set: str,
    data: pd.DataFrame,
    pair_embeddings: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, dict[str, Any]]:
    """Fit feature transformers and return train/test matrices."""
    components = fit_feature_transformers(feature_set, data, pair_embeddings, train_idx)
    x_train = transform_features(components, data, pair_embeddings, train_idx)
    x_test = transform_features(components, data, pair_embeddings, test_idx)
    return x_train, x_test, components


def positive_scores(model: LogisticRegression, values: sparse.csr_matrix) -> np.ndarray:
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


def evaluate_feature_set(
    feature_set: str,
    data: pd.DataFrame,
    pair_embeddings: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict[str, Any]:
    """Fit and evaluate one feature set on one split."""
    x_train, x_test, _ = fit_design_matrices(
        feature_set,
        data,
        pair_embeddings,
        train_idx,
        test_idx,
    )
    model = make_classifier()
    with threadpool_limits(limits=1):
        model.fit(x_train, data.iloc[train_idx]["label"])

    y_true = data.iloc[test_idx]["label"]
    y_pred = model.predict(x_test)
    y_score = positive_scores(model, x_test)
    return {
        "feature_count": int(x_train.shape[1]),
        **metric_dict(y_true, y_pred, y_score),
    }


def evaluate_split(
    data: pd.DataFrame,
    pair_embeddings: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    groups: pd.Series | None,
) -> dict[str, Any]:
    """Evaluate all feature sets on one split."""
    diagnostics = split_diagnostics(data, train_idx, test_idx, groups)

    def evaluate_one(feature_set: str) -> tuple[str, dict[str, Any]]:
        print(f"Evaluating {feature_set}", flush=True)
        return feature_set, {
            **diagnostics,
            **evaluate_feature_set(
                feature_set,
                data,
                pair_embeddings,
                train_idx,
                test_idx,
            ),
        }

    evaluated = joblib.Parallel(n_jobs=N_JOBS, backend="loky")(
        joblib.delayed(evaluate_one)(feature_set) for feature_set in FEATURE_SETS
    )
    model_results = dict(evaluated)

    return {
        "split": diagnostics,
        "feature_sets": model_results,
    }


def evaluate_all(
    data: pd.DataFrame,
    pair_embeddings: np.ndarray,
) -> dict[str, Any]:
    """Evaluate random and group_feature_v validation."""
    results: dict[str, Any] = {}

    random_train_idx, random_test_idx = random_split(data)
    print("Split: random", flush=True)
    results["random"] = {
        "valid": True,
        "meaningful": True,
        "reason": "ok",
        "group_column": None,
        "split_random_state": RANDOM_STATE,
        **evaluate_split(
            data=data,
            pair_embeddings=pair_embeddings,
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

    print(f"Split: {GROUP_COLUMN}", flush=True)
    split_results = evaluate_split(
        data=data,
        pair_embeddings=pair_embeddings,
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


def metric_value(value: float | None) -> float:
    """Convert optional metrics to NaN for plotting."""
    return np.nan if value is None else float(value)


def best_grouped_model(results: dict[str, Any], metric: str) -> dict[str, Any] | None:
    """Return the best valid group_feature_v result by one metric."""
    grouped = results.get(GROUP_COLUMN, {})
    if not grouped.get("valid"):
        return None

    best_name = None
    best_metrics = None
    best_value = -np.inf
    for feature_set, metrics in grouped["feature_sets"].items():
        value = metrics.get(metric)
        if value is not None and value > best_value:
            best_name = feature_set
            best_metrics = metrics
            best_value = float(value)

    if best_name is None or best_metrics is None:
        return None
    return {"feature_set": best_name, "metric": metric, "value": best_value, "metrics": best_metrics}


def load_previous_metrics() -> dict[str, Any]:
    """Load previous k-mer and embedding benchmark summaries if available."""
    previous: dict[str, Any] = {}

    if KMER_METRICS_PATH.exists():
        previous["kmer"] = json.loads(KMER_METRICS_PATH.read_text(encoding="utf-8"))
    else:
        previous["kmer"] = None

    if EMBEDDING_METRICS_PATH.exists():
        previous["embedding"] = json.loads(
            EMBEDDING_METRICS_PATH.read_text(encoding="utf-8")
        )
    else:
        previous["embedding"] = None

    return previous


def previous_kmer_pair_metrics(previous: dict[str, Any], split_name: str) -> dict[str, Any] | None:
    """Extract previous pair-text k-mer metrics for one split."""
    payload = previous.get("kmer")
    if not payload:
        return None
    split_results = payload.get("results", {}).get(split_name, {})
    if not split_results.get("valid"):
        return None
    return split_results.get("kmer_logreg", {}).get("sequence_pair_text")


def build_interpretation(
    results: dict[str, Any],
    previous: dict[str, Any],
) -> list[str]:
    """Build a short interpretation for the report."""
    grouped = results.get(GROUP_COLUMN, {})
    lines = ["## Interpretation", ""]

    if not grouped.get("valid"):
        lines.extend(
            [
                f"`{GROUP_COLUMN}` grouped validation was invalid: {grouped.get('reason')}.",
                "",
            ]
        )
        return lines

    grouped_results = grouped["feature_sets"]
    kmer_only = grouped_results["kmer_only"]
    kmer_plus_simple = grouped_results["hybrid_kmer_plus_simple"]
    kmer_plus_ablang2 = grouped_results["hybrid_kmer_plus_ablang2"]
    hybrid_all = grouped_results["hybrid_all"]
    best_roc = best_grouped_model(results, "roc_auc")
    best_pr = best_grouped_model(results, "average_precision")

    ablang2_adds = max(
        metric_value(kmer_plus_ablang2["roc_auc"]),
        metric_value(hybrid_all["roc_auc"]),
    ) > metric_value(kmer_only["roc_auc"])
    simple_helps = metric_value(kmer_plus_simple["roc_auc"]) > metric_value(
        kmer_only["roc_auc"]
    )

    previous_kmer = previous_kmer_pair_metrics(previous, GROUP_COLUMN)
    if previous_kmer:
        best_grouped_roc = metric_value(best_roc["value"] if best_roc else None)
        improves_previous = best_grouped_roc > metric_value(previous_kmer.get("roc_auc"))
    else:
        improves_previous = False

    lines.extend(
        [
            (
                "AbLang2 adds value beyond k-mers: "
                f"{'yes' if ablang2_adds else 'no'}."
            ),
            (
                "Simple features improve grouped ROC-AUC over k-mers alone: "
                f"{'yes' if simple_helps else 'no'}."
            ),
            (
                "Final hybrid improves over the previous k-mer baseline: "
                f"{'yes' if improves_previous else 'no'}."
            ),
        ]
    )

    if best_roc:
        lines.append(
            f"Best grouped ROC-AUC: `{best_roc['feature_set']}` "
            f"({best_roc['value']:.4f})."
        )
    if best_pr:
        lines.append(
            f"Best grouped PR-AUC: `{best_pr['feature_set']}` "
            f"({best_pr['value']:.4f})."
        )

    lines.append("")
    return lines


def save_best_model(
    data: pd.DataFrame,
    pair_embeddings: np.ndarray,
    results: dict[str, Any],
) -> dict[str, Any] | None:
    """Fit the best grouped ROC-AUC feature set on all rows and save it."""
    best = best_grouped_model(results, "roc_auc")
    if best is None:
        return None

    all_idx = np.arange(len(data))
    components = fit_feature_transformers(
        best["feature_set"],
        data,
        pair_embeddings,
        all_idx,
    )
    x_all = transform_features(components, data, pair_embeddings, all_idx)
    model = make_classifier()
    model.fit(x_all, data["label"])

    artifact = {
        "feature_set": best["feature_set"],
        "feature_columns": SIMPLE_FEATURE_COLUMNS,
        "components": components,
        "classifier": model,
        "input_path": str(INPUT_PATH.relative_to(PROJECT_ROOT)),
        "pair_embedding_path": str(PAIR_EMBEDDING_PATH.relative_to(PROJECT_ROOT)),
    }
    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, BEST_MODEL_PATH)
    return {"feature_set": best["feature_set"], "path": str(BEST_MODEL_PATH.relative_to(PROJECT_ROOT))}


def flatten_results(results: dict[str, Any]) -> pd.DataFrame:
    """Create one row per split and feature set."""
    records = []
    for split_name, split_results in results.items():
        if not split_results.get("valid"):
            continue
        for feature_set, metrics in split_results["feature_sets"].items():
            records.append(
                {
                    "split": split_name,
                    "feature_set": feature_set,
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "roc_auc": metric_value(metrics["roc_auc"]),
                    "average_precision": metric_value(metrics["average_precision"]),
                    "confusion_matrix": metrics["confusion_matrix"],
                }
            )
    return pd.DataFrame.from_records(records)


def save_metric_figure(
    table: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Save a feature-set comparison figure for one metric."""
    if table.empty:
        return

    feature_order = list(FEATURE_SETS)
    split_order = [split for split in ["random", GROUP_COLUMN] if split in set(table["split"])]
    x = np.arange(len(feature_order))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 4.8))
    offsets = np.linspace(-width / 2, width / 2, num=max(len(split_order), 1))

    for offset, split_name in zip(offsets, split_order):
        values = []
        for feature_set in feature_order:
            row = table[
                (table["split"] == split_name) & (table["feature_set"] == feature_set)
            ]
            values.append(float(row.iloc[0][metric]) if not row.empty else np.nan)
        ax.bar(x + offset, values, width=width, label=split_name)

    ax.set_xticks(x)
    ax.set_xticklabels(feature_order, rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_figures(results: dict[str, Any]) -> None:
    """Save all hybrid comparison figures."""
    table = flatten_results(results)
    save_metric_figure(
        table,
        metric="roc_auc",
        title="Hybrid Feature ROC-AUC Comparison",
        ylabel="ROC-AUC",
        output_path=ROC_AUC_FIGURE_PATH,
    )
    save_metric_figure(
        table,
        metric="average_precision",
        title="Hybrid Feature PR-AUC Comparison",
        ylabel="Average precision / PR-AUC",
        output_path=PR_AUC_FIGURE_PATH,
    )
    save_metric_figure(
        table,
        metric="f1",
        title="Hybrid Feature F1 Comparison",
        ylabel="F1",
        output_path=F1_FIGURE_PATH,
    )


def format_metric(value: float | None) -> str:
    """Format optional metric values."""
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


def format_comparison_table(results: dict[str, Any]) -> list[str]:
    """Format the full feature-set comparison table."""
    lines = [
        (
            "| Split | Feature set | Features | Train size | Test size | "
            "Group overlap | Accuracy | Balanced accuracy | Precision | Recall | "
            "F1 | ROC-AUC | PR-AUC | Confusion matrix |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    for split_name, split_results in results.items():
        if not split_results.get("valid"):
            lines.append(
                f"| {split_name} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | "
                f"n/a | n/a | n/a | n/a | n/a | invalid: {split_results.get('reason')} |"
            )
            continue
        for feature_set, metrics in split_results["feature_sets"].items():
            lines.append(
                f"| {split_name} | {feature_set} | {metrics['feature_count']} | "
                f"{metrics['train_size']} | {metrics['test_size']} | "
                f"{format_nullable(metrics['group_overlap_count'])} | "
                f"{metrics['accuracy']:.4f} | {metrics['balanced_accuracy']:.4f} | "
                f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | "
                f"{metrics['f1']:.4f} | {format_metric(metrics['roc_auc'])} | "
                f"{format_metric(metrics['average_precision'])} | "
                f"{metrics['confusion_matrix']} |"
            )

    return lines


def build_report(
    data: pd.DataFrame,
    pair_embeddings: np.ndarray,
    results: dict[str, Any],
    previous: dict[str, Any],
    best_model_artifact: dict[str, Any] | None,
) -> str:
    """Build the Markdown report."""
    best_roc = best_grouped_model(results, "roc_auc")
    best_pr = best_grouped_model(results, "average_precision")

    lines = [
        "# Hybrid Feature Baseline",
        "",
        f"Input file: `{INPUT_PATH.relative_to(PROJECT_ROOT)}`",
        f"Pair embeddings: `{PAIR_EMBEDDING_PATH.relative_to(PROJECT_ROOT)}`",
        "",
        "This report evaluates logistic-regression classifiers on existing labeled",
        "rows using k-mer TF-IDF features, cached AbLang2 embeddings, simple",
        "features, and hybrid combinations.",
        "",
        "## Data",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Rows | {len(data)} |",
        f"| Label 0 count | {label_counts(data['label'])['0']} |",
        f"| Label 1 count | {label_counts(data['label'])['1']} |",
        f"| Pair embedding shape | {list(pair_embeddings.shape)} |",
        "",
        "## Best Grouped Models",
        "",
        "| Selection | Feature set | Value |",
        "|---|---|---:|",
        (
            f"| Grouped ROC-AUC | {best_roc['feature_set']} | {best_roc['value']:.4f} |"
            if best_roc
            else "| Grouped ROC-AUC | n/a | n/a |"
        ),
        (
            f"| Grouped PR-AUC | {best_pr['feature_set']} | {best_pr['value']:.4f} |"
            if best_pr
            else "| Grouped PR-AUC | n/a | n/a |"
        ),
        "",
        "## Comparison Table",
        "",
    ]

    lines.extend(format_comparison_table(results))
    lines.extend([""])
    lines.extend(build_interpretation(results, previous))
    lines.extend(
        [
            "## Artifacts",
            "",
            f"- `{METRICS_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{ROC_AUC_FIGURE_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{PR_AUC_FIGURE_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{F1_FIGURE_PATH.relative_to(PROJECT_ROOT)}`",
        ]
    )
    if best_model_artifact:
        lines.append(f"- `{best_model_artifact['path']}`")
    lines.append("")

    return "\n".join(lines)


def print_comparison_table(results: dict[str, Any]) -> None:
    """Print the requested comparison table."""
    print("\ncomparison table")
    for line in format_comparison_table(results):
        print(line)


def main() -> None:
    """Run the full hybrid benchmark."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    data, pair_embeddings = load_inputs()
    results = evaluate_all(data, pair_embeddings)
    previous = load_previous_metrics()
    best_model_artifact = save_best_model(data, pair_embeddings, results)
    save_figures(results)

    payload = {
        "input_path": str(INPUT_PATH.relative_to(PROJECT_ROOT)),
        "pair_embedding_path": str(PAIR_EMBEDDING_PATH.relative_to(PROJECT_ROOT)),
        "row_count": int(len(data)),
        "label_counts": label_counts(data["label"]),
        "pair_embedding_shape": list(pair_embeddings.shape),
        "simple_feature_columns": SIMPLE_FEATURE_COLUMNS,
        "feature_sets": FEATURE_SETS,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "results": results,
        "best_grouped_roc_auc": best_grouped_model(results, "roc_auc"),
        "best_grouped_average_precision": best_grouped_model(
            results,
            "average_precision",
        ),
        "best_model_artifact": best_model_artifact,
        "artifacts": {
            "report": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
            "metrics_json": str(METRICS_PATH.relative_to(PROJECT_ROOT)),
            "roc_auc_comparison": str(ROC_AUC_FIGURE_PATH.relative_to(PROJECT_ROOT)),
            "pr_auc_comparison": str(PR_AUC_FIGURE_PATH.relative_to(PROJECT_ROOT)),
            "f1_comparison": str(F1_FIGURE_PATH.relative_to(PROJECT_ROOT)),
            "best_model": str(BEST_MODEL_PATH.relative_to(PROJECT_ROOT)),
        },
    }
    METRICS_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        build_report(data, pair_embeddings, results, previous, best_model_artifact),
        encoding="utf-8",
    )

    print_comparison_table(results)


if __name__ == "__main__":
    main()
