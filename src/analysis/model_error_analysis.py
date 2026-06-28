"""Compare grouped-test errors for k-mer and frozen-embedding MLP models.

This module evaluates existing labeled rows only. It does not generate,
mutate, optimize, rank, or propose biological sequences.

Run from the project root:

    python src/analysis/model_error_analysis.py
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd


# Matplotlib needs a writable cache directory in this environment.
MPLCONFIGDIR = Path("/tmp") / "antibody_prioritization_mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt
import torch
from sklearn.exceptions import UndefinedMetricWarning
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
from threadpoolctl import threadpool_limits
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "neutral_sequence_classification_ml.csv"
PAIR_EMBEDDING_PATH = PROJECT_ROOT / "data" / "processed" / "embeddings_ablang2_pair.npy"
MLP_PAIR_MODEL_PATH = PROJECT_ROOT / "models" / "pytorch_mlp_pair.pt"
GROUPED_METRICS_PATH = PROJECT_ROOT / "reports" / "metrics" / "grouped_validation_metrics.json"
PYTORCH_METRICS_PATH = PROJECT_ROOT / "reports" / "metrics" / "pytorch_embedding_mlp_metrics.json"

PREDICTIONS_PATH = PROJECT_ROOT / "reports" / "model_error_analysis_predictions.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "model_error_analysis_report.md"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics" / "model_error_analysis_metrics.json"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
SUBGROUP_ROC_AUC_FIGURE_PATH = FIGURE_DIR / "subgroup_roc_auc_comparison.png"
SUBGROUP_PR_AUC_FIGURE_PATH = FIGURE_DIR / "subgroup_pr_auc_comparison.png"
ERROR_COUNTS_FIGURE_PATH = FIGURE_DIR / "error_counts_by_target_region.png"
PROBABILITY_BY_LABEL_FIGURE_PATH = FIGURE_DIR / "predicted_probability_by_true_label.png"
PROBABILITY_SCATTER_FIGURE_PATH = FIGURE_DIR / "kmer_vs_mlp_probability_scatter.png"

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_GROUP_SPLIT_ATTEMPTS = 50
GROUP_COLUMN = "group_feature_v"
MIN_SUBGROUP_SIZE_FOR_SUMMARY = 10
MAX_SUBGROUPS_PER_FIGURE = 20
BATCH_SIZE = 256

MOSTLY_MISSING_THRESHOLD = 0.50
NEAR_ROW_UNIQUE_ROW_THRESHOLD = 0.80
NEAR_ROW_UNIQUE_NON_MISSING_THRESHOLD = 0.90

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

PREDICTION_COLUMNS = [
    "row_id",
    "antibody_name",
    "true_label",
    "kmer_pred_label",
    "kmer_pred_proba",
    "mlp_pair_pred_label",
    "mlp_pair_pred_proba",
    "group_feature_v",
    "group_feature_j",
    "group_feature_b_v",
    "group_feature_b_j",
    "metadata_target_region",
    "has_light",
    "is_nanobody_like",
    "has_structure",
    "heavy_length",
    "light_length",
    "cdrh3_length",
    "cdrl3_length",
    "targets_rbd",
    "targets_spike",
    "targets_ntd",
]


def read_csv(path: Path) -> pd.DataFrame:
    """Read CSV text while preserving blank fields."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def normalized_text(values: pd.Series) -> pd.Series:
    """Normalize text for split checks and subgroup labels."""
    return values.fillna("").astype(str).str.strip()


def normalized_sequence(values: pd.Series) -> pd.Series:
    """Normalize sequence-like text only for length and presence features."""
    return (
        values.fillna("")
        .astype(str)
        .str.replace(r"\s+", "", regex=True)
        .str.upper()
    )


def optional_column(data: pd.DataFrame, column: str) -> pd.Series:
    """Return an aligned column or aligned blanks when unavailable."""
    if column in data.columns:
        return data[column]
    return pd.Series([""] * len(data), index=data.index)


def numeric_or_none(data: pd.DataFrame, column: str) -> pd.Series | None:
    """Return an existing numeric or boolean feature, if present."""
    if column not in data.columns:
        return None

    text_values = normalized_text(data[column]).str.lower()
    mapped = text_values.map(
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


def boolean_presence_feature(
    data: pd.DataFrame,
    feature_name: str,
    source_column: str,
) -> pd.Series:
    """Use an existing boolean feature, or compute presence from metadata."""
    existing = numeric_or_none(data, feature_name)
    if existing is not None:
        return existing
    return normalized_text(optional_column(data, source_column)).ne("").astype(float)


def target_feature(data: pd.DataFrame, feature_name: str, pattern: str) -> pd.Series:
    """Use an existing target flag, or compute one from neutral target metadata."""
    existing = numeric_or_none(data, feature_name)
    if existing is not None:
        return existing
    values = normalized_text(optional_column(data, "metadata_target_region")).str.lower()
    return values.str.contains(pattern, regex=True, na=False).astype(float)


def build_simple_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build fixed numeric row-level features used for subgroup analysis."""
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
            "has_structure": boolean_presence_feature(
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


def require_files(paths: list[Path]) -> None:
    """Fail clearly if required artifacts are unavailable."""
    missing = [path for path in paths if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path.relative_to(PROJECT_ROOT)) for path in missing)
        raise FileNotFoundError(f"Missing required artifact(s): {missing_text}")


def load_inputs() -> tuple[pd.DataFrame, np.ndarray]:
    """Load neutral rows, cached pair embeddings, labels, and feature flags."""
    require_files([INPUT_PATH, PAIR_EMBEDDING_PATH, MLP_PAIR_MODEL_PATH])
    raw_data = read_csv(INPUT_PATH)
    pair_embeddings = np.load(PAIR_EMBEDDING_PATH).astype(np.float32)
    if len(raw_data) != pair_embeddings.shape[0]:
        raise ValueError(
            "Neutral input rows and pair embedding rows do not match: "
            f"{len(raw_data)} vs {pair_embeddings.shape[0]}."
        )

    required_columns = ["sequence_pair_text", "label", GROUP_COLUMN]
    missing_columns = [column for column in required_columns if column not in raw_data.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise KeyError(f"Missing required neutral column(s): {missing_text}")

    data = raw_data.copy()
    data["row_id"] = np.arange(len(data), dtype=int)
    data = data[normalized_text(data["label"]).ne("")].copy()
    selected_positions = data.index.to_numpy()
    data = data.reset_index(drop=True)
    pair_embeddings = pair_embeddings[selected_positions]

    data["label"] = data["label"].astype(int)
    unexpected_labels = sorted(set(data["label"]) - {0, 1})
    if unexpected_labels:
        raise ValueError(f"Expected binary labels 0/1, found: {unexpected_labels}")

    label_counts = data["label"].value_counts()
    if len(label_counts) != 2 or label_counts.min() < 2:
        raise ValueError("Evaluation requires at least two rows per label.")

    data["sequence_pair_text"] = data["sequence_pair_text"].fillna("").astype(str)
    empty_inputs = int(data["sequence_pair_text"].str.len().eq(0).sum())
    if empty_inputs:
        raise ValueError(f"sequence_pair_text has {empty_inputs} empty rows.")

    data[SIMPLE_FEATURE_COLUMNS] = build_simple_features(data)
    return data, pair_embeddings


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


def grouped_split(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.Series, int]:
    """Create the same meaningful group_feature_v split used by model scripts."""
    status = group_column_status(data)
    if not status["useful_for_grouping"]:
        raise ValueError(f"grouped validation invalid: {status['reason']}")

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


def label_counts(labels: pd.Series) -> dict[str, int]:
    """Return stable binary label counts."""
    counts = labels.value_counts().sort_index()
    return {"0": int(counts.get(0, 0)), "1": int(counts.get(1, 0))}


def split_diagnostics(
    data: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    groups: pd.Series,
) -> dict[str, Any]:
    """Summarize grouped split size, label balance, and group overlap."""
    train_group_set = set(groups.iloc[train_idx].astype(str))
    test_group_set = set(groups.iloc[test_idx].astype(str))
    return {
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "train_label_counts": label_counts(data.iloc[train_idx]["label"]),
        "test_label_counts": label_counts(data.iloc[test_idx]["label"]),
        "train_group_count": int(len(train_group_set)),
        "test_group_count": int(len(test_group_set)),
        "group_overlap_count": int(len(train_group_set & test_group_set)),
    }


def make_kmer_pipeline() -> Pipeline:
    """Create the grouped-validation k-mer TF-IDF classifier."""
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


def positive_scores(model: Pipeline, values: pd.Series) -> np.ndarray:
    """Return positive-class probabilities from a fitted sklearn model."""
    class_list = list(model.classes_)
    if 1 not in class_list:
        raise ValueError("Estimator was not fitted with positive class label 1.")
    return model.predict_proba(values)[:, class_list.index(1)]


def fit_score_kmer(
    data: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the best classical model on train rows and score test rows."""
    model = make_kmer_pipeline()
    with threadpool_limits(limits=1):
        model.fit(
            data.iloc[train_idx]["sequence_pair_text"],
            data.iloc[train_idx]["label"],
        )
    scores = positive_scores(model, data.iloc[test_idx]["sequence_pair_text"])
    labels = (scores >= 0.5).astype(int)
    return labels, scores


class EmbeddingMLP(nn.Module):
    """Small MLP matching the saved frozen-embedding classifier."""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Return one logit per row."""
        return self.network(values).squeeze(-1)


def torch_load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a PyTorch checkpoint across torch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def score_mlp_pair(
    pair_embeddings: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load the saved pair MLP and score the grouped test rows."""
    checkpoint = torch_load_checkpoint(MLP_PAIR_MODEL_PATH)
    input_dim = int(checkpoint.get("input_dim", pair_embeddings.shape[1]))
    if input_dim != pair_embeddings.shape[1]:
        raise ValueError(
            "Pair embedding dimension does not match checkpoint input_dim: "
            f"{pair_embeddings.shape[1]} vs {input_dim}."
        )

    hidden_layers = [int(value) for value in checkpoint.get("hidden_layers", [256, 64])]
    dropout = float(checkpoint.get("dropout", 0.2))
    scaler_mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float32)
    scaler_scale = np.asarray(checkpoint["scaler_scale"], dtype=np.float32)
    scaler_scale = np.where(scaler_scale == 0, 1.0, scaler_scale)

    x_test = ((pair_embeddings[test_idx] - scaler_mean) / scaler_scale).astype(np.float32)
    model = EmbeddingMLP(input_dim=input_dim, hidden_layers=hidden_layers, dropout=dropout)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scores = []
    with torch.no_grad():
        for start in range(0, len(x_test), BATCH_SIZE):
            batch = torch.from_numpy(x_test[start : start + BATCH_SIZE])
            logits = model(batch)
            scores.append(torch.sigmoid(logits).numpy())

    probabilities = np.concatenate(scores)
    labels = (probabilities >= 0.5).astype(int)
    return labels, probabilities, checkpoint


def metric_dict(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, Any]:
    """Compute scalar metrics and a confusion matrix."""
    y_true_array = np.asarray(y_true, dtype=int)
    has_both_labels = len(set(y_true_array.tolist())) == 2
    matrix = confusion_matrix(y_true_array, y_pred, labels=[0, 1])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        balanced_accuracy = balanced_accuracy_score(y_true_array, y_pred)

    return {
        "accuracy": float(accuracy_score(y_true_array, y_pred)),
        "balanced_accuracy": float(balanced_accuracy),
        "precision": float(precision_score(y_true_array, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_array, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_array, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true_array, y_score)) if has_both_labels else None,
        "average_precision": (
            float(average_precision_score(y_true_array, y_score))
            if has_both_labels
            else None
        ),
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def output_name_column(data: pd.DataFrame) -> pd.Series:
    """Return an optional public row name under the requested output alias."""
    if "sample_name" in data.columns:
        return data["sample_name"].fillna("").astype(str)
    if "antibody_name" in data.columns:
        return data["antibody_name"].fillna("").astype(str)
    return pd.Series([""] * len(data), index=data.index)


def add_length_bins(predictions: pd.DataFrame) -> pd.DataFrame:
    """Add fixed bins for subgroup analysis."""
    output = predictions.copy()

    cdrh3 = pd.to_numeric(output["cdrh3_length"], errors="coerce")
    output["cdrh3_length_bin"] = np.select(
        [cdrh3 <= 10, (cdrh3 >= 11) & (cdrh3 <= 20), cdrh3 > 20],
        ["short", "medium", "long"],
        default="missing",
    )

    heavy_length = pd.to_numeric(output["heavy_length"], errors="coerce")
    output["heavy_length_bin"] = np.select(
        [heavy_length <= 115, (heavy_length >= 116) & (heavy_length <= 135), heavy_length > 135],
        ["short", "normal", "long"],
        default="missing",
    )
    return output


def int_like(values: pd.Series) -> pd.Series:
    """Return 0/1 numeric flags as integer-looking values."""
    return pd.to_numeric(values, errors="coerce").fillna(0).round().astype(int)


def create_prediction_table(
    data: pd.DataFrame,
    test_idx: np.ndarray,
    kmer_labels: np.ndarray,
    kmer_scores: np.ndarray,
    mlp_labels: np.ndarray,
    mlp_scores: np.ndarray,
) -> pd.DataFrame:
    """Create the requested row-level test-set prediction table."""
    test_data = data.iloc[test_idx].copy()
    predictions = pd.DataFrame(index=test_data.index)
    predictions["row_id"] = test_data["row_id"].astype(int)
    predictions["antibody_name"] = output_name_column(test_data)
    predictions["true_label"] = test_data["label"].astype(int)
    predictions["kmer_pred_label"] = kmer_labels.astype(int)
    predictions["kmer_pred_proba"] = kmer_scores.astype(float)
    predictions["mlp_pair_pred_label"] = mlp_labels.astype(int)
    predictions["mlp_pair_pred_proba"] = mlp_scores.astype(float)

    for column in [
        "group_feature_v",
        "group_feature_j",
        "group_feature_b_v",
        "group_feature_b_j",
        "metadata_target_region",
    ]:
        predictions[column] = optional_column(test_data, column).fillna("").astype(str)

    for column in SIMPLE_FEATURE_COLUMNS:
        predictions[column] = test_data[column].astype(float)

    for column in ["has_light", "is_nanobody_like", "has_structure", "targets_rbd", "targets_spike", "targets_ntd"]:
        predictions[column] = int_like(predictions[column])

    for column in ["heavy_length", "light_length", "cdrh3_length", "cdrl3_length"]:
        predictions[column] = pd.to_numeric(predictions[column], errors="coerce").fillna(0).round().astype(int)

    predictions = add_length_bins(predictions)
    return predictions[PREDICTION_COLUMNS + ["cdrh3_length_bin", "heavy_length_bin"]]


def subgroup_value_series(predictions: pd.DataFrame, column: str) -> pd.Series:
    """Normalize subgroup values for stable reporting."""
    values = predictions[column]
    if column in {
        "targets_rbd",
        "targets_ntd",
        "has_light",
        "is_nanobody_like",
        "has_structure",
    }:
        return int_like(values).map({0: "false", 1: "true"})
    text_values = normalized_text(values)
    text_values.loc[text_values.eq("")] = "missing"
    return text_values


def subgroup_metrics(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    """Compute requested subgroup metrics for both models."""
    subgroup_columns = [
        "metadata_target_region",
        "targets_rbd",
        "targets_ntd",
        "has_light",
        "is_nanobody_like",
        "has_structure",
        "cdrh3_length_bin",
        "heavy_length_bin",
    ]
    model_specs = {
        "kmer": ("kmer_pred_label", "kmer_pred_proba"),
        "mlp_pair": ("mlp_pair_pred_label", "mlp_pair_pred_proba"),
    }

    records: list[dict[str, Any]] = []
    y_true_all = predictions["true_label"].astype(int)

    for subgroup_column in subgroup_columns:
        values = subgroup_value_series(predictions, subgroup_column)
        for subgroup_value in sorted(values.unique().tolist()):
            mask = values.eq(subgroup_value)
            if not mask.any():
                continue
            y_true = y_true_all.loc[mask].to_numpy(dtype=int)
            size = int(mask.sum())
            positive_fraction = float(np.mean(y_true)) if size else None

            for model_name, (label_column, score_column) in model_specs.items():
                y_pred = predictions.loc[mask, label_column].to_numpy(dtype=int)
                y_score = predictions.loc[mask, score_column].to_numpy(dtype=float)
                records.append(
                    {
                        "subgroup_type": subgroup_column,
                        "subgroup_value": str(subgroup_value),
                        "model": model_name,
                        "subgroup_size": size,
                        "positive_label_fraction": positive_fraction,
                        **metric_dict(y_true, y_pred, y_score),
                    }
                )
    return records


def subgroup_records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert subgroup metric records to a DataFrame."""
    if not records:
        return pd.DataFrame()
    return pd.DataFrame.from_records(records)


def paired_subgroup_deltas(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Return k-mer minus MLP and MLP minus k-mer subgroup deltas."""
    frame = subgroup_records_to_frame(records)
    if frame.empty:
        return frame

    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    ]
    index_columns = ["subgroup_type", "subgroup_value", "subgroup_size", "positive_label_fraction"]
    pivot = frame.pivot_table(
        index=index_columns,
        columns="model",
        values=metric_columns,
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{model}" for metric, model in pivot.columns]
    pivot = pivot.reset_index()

    for metric in metric_columns:
        kmer_column = f"{metric}_kmer"
        mlp_column = f"{metric}_mlp_pair"
        if kmer_column in pivot.columns and mlp_column in pivot.columns:
            pivot[f"delta_{metric}_mlp_minus_kmer"] = (
                pivot[mlp_column] - pivot[kmer_column]
            )
    return pivot


def metric_or_none(value: Any) -> float | None:
    """Normalize optional numeric values."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return float(value)


def format_metric(value: Any) -> str:
    """Format optional metric values."""
    numeric = metric_or_none(value)
    return "n/a" if numeric is None else f"{numeric:.4f}"


def format_fraction(value: Any) -> str:
    """Format optional fractions."""
    numeric = metric_or_none(value)
    return "n/a" if numeric is None else f"{numeric:.3f}"


def format_confusion_matrix(value: list[list[int]]) -> str:
    """Format a 2x2 confusion matrix compactly."""
    return f"[[{value[0][0]}, {value[0][1]}], [{value[1][0]}, {value[1][1]}]]"


def compact_value(value: str, max_length: int = 48) -> str:
    """Keep long metadata values readable in Markdown tables."""
    if len(value) <= max_length:
        return value
    return value[: max_length - 3] + "..."


def save_placeholder_figure(output_path: Path, message: str) -> None:
    """Save a small placeholder figure when no plottable data exists."""
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_subgroup_metric_figure(
    records: list[dict[str, Any]],
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Save a grouped bar plot for subgroup ROC-AUC or PR-AUC."""
    deltas = paired_subgroup_deltas(records)
    kmer_column = f"{metric}_kmer"
    mlp_column = f"{metric}_mlp_pair"
    if deltas.empty or kmer_column not in deltas.columns or mlp_column not in deltas.columns:
        save_placeholder_figure(output_path, f"No {ylabel} subgroup comparison available")
        return

    plot_data = deltas[
        (deltas["subgroup_size"] >= MIN_SUBGROUP_SIZE_FOR_SUMMARY)
        & deltas[kmer_column].notna()
        & deltas[mlp_column].notna()
    ].copy()
    if plot_data.empty:
        save_placeholder_figure(output_path, f"No {ylabel} subgroup comparison available")
        return

    plot_data["label"] = (
        plot_data["subgroup_type"].astype(str)
        + "="
        + plot_data["subgroup_value"].astype(str).map(compact_value)
    )
    plot_data["abs_delta"] = (plot_data[mlp_column] - plot_data[kmer_column]).abs()
    plot_data = plot_data.sort_values(["abs_delta", "subgroup_size"], ascending=False).head(
        MAX_SUBGROUPS_PER_FIGURE
    )
    plot_data = plot_data.iloc[::-1]

    y_positions = np.arange(len(plot_data))
    height = 0.38
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(plot_data) + 1.5)))
    ax.barh(y_positions - height / 2, plot_data[kmer_column], height=height, label="k-mer")
    ax.barh(y_positions + height / 2, plot_data[mlp_column], height=height, label="MLP pair")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_data["label"])
    ax.set_xlim(0, 1)
    ax.set_xlabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_error_counts_by_target_region(predictions: pd.DataFrame, output_path: Path) -> None:
    """Save error counts by neutral target-region metadata."""
    target_region = normalized_text(predictions["metadata_target_region"])
    target_region.loc[target_region.eq("")] = "missing"
    errors = pd.DataFrame(
        {
            "metadata_target_region": target_region,
            "kmer": predictions["true_label"].ne(predictions["kmer_pred_label"]).astype(int),
            "mlp_pair": predictions["true_label"].ne(predictions["mlp_pair_pred_label"]).astype(int),
        }
    )
    grouped = errors.groupby("metadata_target_region", dropna=False)[["kmer", "mlp_pair"]].sum()
    grouped["total_errors"] = grouped["kmer"] + grouped["mlp_pair"]
    grouped = grouped.sort_values("total_errors", ascending=False).head(15)

    if grouped.empty:
        save_placeholder_figure(output_path, "No target-region error counts available")
        return

    labels = [compact_value(str(value), max_length=36) for value in grouped.index.tolist()]
    x = np.arange(len(grouped))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(9, 0.55 * len(grouped)), 5))
    ax.bar(x - width / 2, grouped["kmer"], width=width, label="k-mer")
    ax.bar(x + width / 2, grouped["mlp_pair"], width=width, label="MLP pair")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Incorrect predictions")
    ax.set_title("Error Counts by Target-Region Metadata")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_probability_by_label(predictions: pd.DataFrame, output_path: Path) -> None:
    """Save predicted-probability distributions by true label."""
    data = []
    labels = []
    for model_name, score_column in [
        ("k-mer", "kmer_pred_proba"),
        ("MLP pair", "mlp_pair_pred_proba"),
    ]:
        for label in [0, 1]:
            values = predictions.loc[
                predictions["true_label"].eq(label),
                score_column,
            ].astype(float)
            data.append(values.to_numpy())
            labels.append(f"{model_name}\ntrue={label}")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    try:
        ax.boxplot(data, tick_labels=labels, showfliers=False)
    except TypeError:
        ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Predicted probability for label 1")
    ax.set_title("Predicted Probability by True Label")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_probability_scatter(predictions: pd.DataFrame, output_path: Path) -> None:
    """Save k-mer vs MLP probability scatter on grouped test rows."""
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for label, color in [(0, "#4c78a8"), (1, "#f58518")]:
        subset = predictions[predictions["true_label"].eq(label)]
        ax.scatter(
            subset["kmer_pred_proba"],
            subset["mlp_pair_pred_proba"],
            s=18,
            alpha=0.65,
            label=f"true={label}",
            color=color,
            edgecolors="none",
        )
    ax.plot([0, 1], [0, 1], color="black", linewidth=1, linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("k-mer predicted probability")
    ax.set_ylabel("MLP pair predicted probability")
    ax.set_title("k-mer vs MLP Pair Probabilities")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_figures(predictions: pd.DataFrame, records: list[dict[str, Any]]) -> None:
    """Save all requested diagnostic figures."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    save_subgroup_metric_figure(
        records,
        metric="roc_auc",
        title="Subgroup ROC-AUC Comparison",
        ylabel="ROC-AUC",
        output_path=SUBGROUP_ROC_AUC_FIGURE_PATH,
    )
    save_subgroup_metric_figure(
        records,
        metric="average_precision",
        title="Subgroup PR-AUC Comparison",
        ylabel="Average precision / PR-AUC",
        output_path=SUBGROUP_PR_AUC_FIGURE_PATH,
    )
    save_error_counts_by_target_region(predictions, ERROR_COUNTS_FIGURE_PATH)
    save_probability_by_label(predictions, PROBABILITY_BY_LABEL_FIGURE_PATH)
    save_probability_scatter(predictions, PROBABILITY_SCATTER_FIGURE_PATH)


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    """Load a JSON file if it exists."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def previous_reference_metrics() -> dict[str, Any]:
    """Extract compact previous baseline metrics for context."""
    previous: dict[str, Any] = {
        "grouped_kmer_pair": None,
        "pytorch_mlp_pair": None,
    }
    grouped = load_json_if_exists(GROUPED_METRICS_PATH)
    if grouped:
        previous["grouped_kmer_pair"] = (
            grouped.get("results", {})
            .get(GROUP_COLUMN, {})
            .get("kmer_logreg", {})
            .get("sequence_pair_text")
        )

    pytorch = load_json_if_exists(PYTORCH_METRICS_PATH)
    if pytorch:
        previous["pytorch_mlp_pair"] = (
            pytorch.get("results", {})
            .get(GROUP_COLUMN, {})
            .get("models", {})
            .get("pair", {})
            .get("metrics")
        )
    return previous


def summarize_improvement_rows(
    deltas: pd.DataFrame,
    delta_column: str,
    ascending: bool,
    limit: int = 8,
) -> pd.DataFrame:
    """Select subgroup rows with meaningful size and finite deltas."""
    if deltas.empty or delta_column not in deltas.columns:
        return pd.DataFrame()
    selected = deltas[
        (deltas["subgroup_size"] >= MIN_SUBGROUP_SIZE_FOR_SUMMARY)
        & deltas[delta_column].notna()
    ].copy()
    if selected.empty:
        return selected
    return selected.sort_values(delta_column, ascending=ascending).head(limit)


def subgroup_table_rows(records: list[dict[str, Any]], limit: int = 120) -> list[str]:
    """Format subgroup metric records for Markdown."""
    frame = subgroup_records_to_frame(records)
    if frame.empty:
        return ["No subgroup metrics were computed."]

    frame = frame[frame["subgroup_size"] >= MIN_SUBGROUP_SIZE_FOR_SUMMARY].copy()
    frame = frame.sort_values(["subgroup_type", "subgroup_value", "model"]).head(limit)
    lines = [
        "| Subgroup | Value | Model | n | Positive fraction | Balanced accuracy | F1 | ROC-AUC | PR-AUC |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"| {row['subgroup_type']} | {compact_value(str(row['subgroup_value']))} | "
            f"{row['model']} | {int(row['subgroup_size'])} | "
            f"{format_fraction(row['positive_label_fraction'])} | "
            f"{format_metric(row['balanced_accuracy'])} | "
            f"{format_metric(row['f1'])} | {format_metric(row['roc_auc'])} | "
            f"{format_metric(row['average_precision'])} |"
        )
    return lines


def delta_table_rows(
    rows: pd.DataFrame,
    delta_column: str,
    title_label: str,
) -> list[str]:
    """Format subgroup delta rows for Markdown."""
    if rows.empty:
        return [f"No {title_label} subgroup differences met the reporting threshold."]

    lines = [
        "| Subgroup | Value | n | k-mer balanced accuracy | MLP balanced accuracy | Delta MLP-kmer |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, row in rows.iterrows():
        lines.append(
            f"| {row['subgroup_type']} | {compact_value(str(row['subgroup_value']))} | "
            f"{int(row['subgroup_size'])} | "
            f"{format_metric(row.get('balanced_accuracy_kmer'))} | "
            f"{format_metric(row.get('balanced_accuracy_mlp_pair'))} | "
            f"{format_metric(row.get(delta_column))} |"
        )
    return lines


def hardest_table_rows(records: list[dict[str, Any]], model_name: str) -> list[str]:
    """Format hardest subgroups by balanced accuracy."""
    frame = subgroup_records_to_frame(records)
    if frame.empty:
        return [f"No subgroup metrics available for {model_name}."]
    selected = frame[
        (frame["model"].eq(model_name))
        & (frame["subgroup_size"] >= MIN_SUBGROUP_SIZE_FOR_SUMMARY)
    ].copy()
    if selected.empty:
        return [f"No subgroup metrics above size threshold for {model_name}."]
    selected = selected.sort_values(["balanced_accuracy", "subgroup_size"], ascending=[True, False]).head(8)
    lines = [
        "| Subgroup | Value | n | Positive fraction | Balanced accuracy | F1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, row in selected.iterrows():
        lines.append(
            f"| {row['subgroup_type']} | {compact_value(str(row['subgroup_value']))} | "
            f"{int(row['subgroup_size'])} | "
            f"{format_fraction(row['positive_label_fraction'])} | "
            f"{format_metric(row['balanced_accuracy'])} | {format_metric(row['f1'])} |"
        )
    return lines


def report_markdown(
    overall_metrics: dict[str, dict[str, Any]],
    split_info: dict[str, Any],
    group_status: dict[str, Any],
    records: list[dict[str, Any]],
    previous: dict[str, Any],
) -> str:
    """Build the Markdown report."""
    deltas = paired_subgroup_deltas(records)
    mlp_improves = summarize_improvement_rows(
        deltas,
        "delta_balanced_accuracy_mlp_minus_kmer",
        ascending=False,
    )
    mlp_improves = mlp_improves[
        mlp_improves["delta_balanced_accuracy_mlp_minus_kmer"] > 0
    ]
    kmer_stronger = summarize_improvement_rows(
        deltas,
        "delta_balanced_accuracy_mlp_minus_kmer",
        ascending=True,
    )
    kmer_stronger = kmer_stronger[
        kmer_stronger["delta_balanced_accuracy_mlp_minus_kmer"] < 0
    ]

    lines = [
        "# Model Error Analysis",
        "",
        "This analysis compares the best classical k-mer baseline and the saved PyTorch pair-embedding MLP on the same group_feature_v held-out split.",
        "",
        "## Grouped Split",
        "",
        f"- train rows: {split_info['train_size']}",
        f"- test rows: {split_info['test_size']}",
        f"- train groups: {split_info['train_group_count']}",
        f"- test groups: {split_info['test_group_count']}",
        f"- train/test group overlap: {split_info['group_overlap_count']}",
        f"- group-column status: {group_status['reason']}",
        "",
        "## Overall Metrics",
        "",
        "| Model | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Confusion matrix |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model_name, metrics in overall_metrics.items():
        lines.append(
            f"| {model_name} | {format_metric(metrics['accuracy'])} | "
            f"{format_metric(metrics['balanced_accuracy'])} | "
            f"{format_metric(metrics['precision'])} | {format_metric(metrics['recall'])} | "
            f"{format_metric(metrics['f1'])} | {format_metric(metrics['roc_auc'])} | "
            f"{format_metric(metrics['average_precision'])} | "
            f"{format_confusion_matrix(metrics['confusion_matrix'])} |"
        )

    kmer_previous = previous.get("grouped_kmer_pair") or {}
    mlp_previous = previous.get("pytorch_mlp_pair") or {}
    lines.extend(
        [
            "",
            "## Comparison With Saved Benchmarks",
            "",
            f"- saved grouped k-mer ROC-AUC: {format_metric(kmer_previous.get('roc_auc'))}",
            f"- reconstructed grouped k-mer ROC-AUC: {format_metric(overall_metrics['kmer']['roc_auc'])}",
            f"- saved PyTorch pair MLP ROC-AUC: {format_metric(mlp_previous.get('roc_auc'))}",
            f"- reconstructed PyTorch pair MLP ROC-AUC: {format_metric(overall_metrics['mlp_pair']['roc_auc'])}",
            "",
            "## Hardest Subgroups",
            "",
            "Hardest subgroups are sorted by low balanced accuracy, with subgroup size at least 10.",
            "",
            "### k-mer",
            "",
            *hardest_table_rows(records, "kmer"),
            "",
            "### MLP pair",
            "",
            *hardest_table_rows(records, "mlp_pair"),
            "",
            "## Where MLP Improves Over k-mer",
            "",
            *delta_table_rows(
                mlp_improves,
                "delta_balanced_accuracy_mlp_minus_kmer",
                "MLP-improved",
            ),
            "",
            "## Where k-mer Is Stronger",
            "",
            *delta_table_rows(
                kmer_stronger,
                "delta_balanced_accuracy_mlp_minus_kmer",
                "k-mer-stronger",
            ),
            "",
            "## Subgroup Metrics",
            "",
            *subgroup_table_rows(records),
            "",
            "## Interpretation",
            "",
        ]
    )

    if overall_metrics["kmer"]["roc_auc"] > overall_metrics["mlp_pair"]["roc_auc"]:
        lines.append(
            "The k-mer model is stronger overall on the same V-gene grouped test set by ROC-AUC."
        )
    else:
        lines.append(
            "The pair-embedding MLP is stronger overall on the same V-gene grouped test set by ROC-AUC."
        )

    if mlp_improves.empty:
        lines.append(
            "No sufficiently sized subgroup showed a balanced-accuracy gain for the MLP over k-mer in this analysis."
        )
    else:
        lines.append(
            "Some subgroups show higher MLP balanced accuracy, but this should be interpreted with subgroup size and label balance in mind."
        )

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Labels are literature-derived.",
            "- Assay conditions are heterogeneous.",
            "- Grouped validation by V-gene reduces but does not eliminate all sequence-family leakage.",
            "- This is classification of existing labeled antibodies, not sequence design.",
            "",
            "## Artifacts",
            "",
            f"- predictions: `{PREDICTIONS_PATH.relative_to(PROJECT_ROOT)}`",
            f"- metrics: `{METRICS_PATH.relative_to(PROJECT_ROOT)}`",
            f"- subgroup ROC-AUC figure: `{SUBGROUP_ROC_AUC_FIGURE_PATH.relative_to(PROJECT_ROOT)}`",
            f"- subgroup PR-AUC figure: `{SUBGROUP_PR_AUC_FIGURE_PATH.relative_to(PROJECT_ROOT)}`",
            f"- error counts figure: `{ERROR_COUNTS_FIGURE_PATH.relative_to(PROJECT_ROOT)}`",
            f"- probability by label figure: `{PROBABILITY_BY_LABEL_FIGURE_PATH.relative_to(PROJECT_ROOT)}`",
            f"- probability scatter figure: `{PROBABILITY_SCATTER_FIGURE_PATH.relative_to(PROJECT_ROOT)}`",
            "",
        ]
    )
    return "\n".join(lines)


def to_json_safe(value: Any) -> Any:
    """Convert numpy and pandas scalar values for JSON output."""
    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [to_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return to_json_safe(value.tolist())
    if pd.isna(value) if not isinstance(value, (list, dict, tuple)) else False:
        return None
    return value


def save_outputs(
    predictions: pd.DataFrame,
    records: list[dict[str, Any]],
    overall_metrics: dict[str, dict[str, Any]],
    split_info: dict[str, Any],
    group_status: dict[str, Any],
    split_random_state: int,
    checkpoint: dict[str, Any],
) -> None:
    """Save CSV, JSON, Markdown, and figures."""
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    predictions[PREDICTION_COLUMNS].to_csv(PREDICTIONS_PATH, index=False)
    save_figures(predictions, records)
    previous = previous_reference_metrics()

    metrics_payload = {
        "input_path": str(INPUT_PATH.relative_to(PROJECT_ROOT)),
        "pair_embedding_path": str(PAIR_EMBEDDING_PATH.relative_to(PROJECT_ROOT)),
        "mlp_pair_model_path": str(MLP_PAIR_MODEL_PATH.relative_to(PROJECT_ROOT)),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "group_column": GROUP_COLUMN,
        "split_random_state": split_random_state,
        "group_column_status": group_status,
        "split": split_info,
        "overall_metrics": overall_metrics,
        "subgroup_metrics": records,
        "previous_reference_metrics": previous,
        "mlp_checkpoint": {
            "embedding_name": checkpoint.get("embedding_name"),
            "split_name": checkpoint.get("split_name"),
            "input_dim": checkpoint.get("input_dim"),
            "hidden_layers": checkpoint.get("hidden_layers"),
            "dropout": checkpoint.get("dropout"),
            "best_epoch": checkpoint.get("best_epoch"),
        },
        "artifacts": {
            "predictions": str(PREDICTIONS_PATH.relative_to(PROJECT_ROOT)),
            "report": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
            "metrics_json": str(METRICS_PATH.relative_to(PROJECT_ROOT)),
            "subgroup_roc_auc_comparison": str(SUBGROUP_ROC_AUC_FIGURE_PATH.relative_to(PROJECT_ROOT)),
            "subgroup_pr_auc_comparison": str(SUBGROUP_PR_AUC_FIGURE_PATH.relative_to(PROJECT_ROOT)),
            "error_counts_by_target_region": str(ERROR_COUNTS_FIGURE_PATH.relative_to(PROJECT_ROOT)),
            "predicted_probability_by_true_label": str(PROBABILITY_BY_LABEL_FIGURE_PATH.relative_to(PROJECT_ROOT)),
            "kmer_vs_mlp_probability_scatter": str(PROBABILITY_SCATTER_FIGURE_PATH.relative_to(PROJECT_ROOT)),
        },
    }
    METRICS_PATH.write_text(
        json.dumps(to_json_safe(metrics_payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        report_markdown(
            overall_metrics=overall_metrics,
            split_info=split_info,
            group_status=group_status,
            records=records,
            previous=previous,
        ),
        encoding="utf-8",
    )


def main() -> None:
    """Run grouped model comparison and subgroup error analysis."""
    print("Loading neutral input and cached pair embeddings")
    data, pair_embeddings = load_inputs()
    group_status = group_column_status(data)

    print("Building group_feature_v split")
    train_idx, test_idx, groups, split_random_state = grouped_split(data)
    split_info = split_diagnostics(data, train_idx, test_idx, groups)
    if split_info["group_overlap_count"] != 0:
        raise ValueError("Grouped split has train/test group overlap.")

    print("Fitting grouped k-mer baseline")
    kmer_labels, kmer_scores = fit_score_kmer(data, train_idx, test_idx)

    print("Scoring saved PyTorch pair MLP")
    mlp_labels, mlp_scores, checkpoint = score_mlp_pair(pair_embeddings, test_idx)

    y_true = data.iloc[test_idx]["label"].to_numpy(dtype=int)
    overall_metrics = {
        "kmer": metric_dict(y_true, kmer_labels, kmer_scores),
        "mlp_pair": metric_dict(y_true, mlp_labels, mlp_scores),
    }

    predictions = create_prediction_table(
        data=data,
        test_idx=test_idx,
        kmer_labels=kmer_labels,
        kmer_scores=kmer_scores,
        mlp_labels=mlp_labels,
        mlp_scores=mlp_scores,
    )
    records = subgroup_metrics(predictions)

    print("Saving reports, metrics, predictions, and figures")
    save_outputs(
        predictions=predictions,
        records=records,
        overall_metrics=overall_metrics,
        split_info=split_info,
        group_status=group_status,
        split_random_state=split_random_state,
        checkpoint=checkpoint,
    )

    print("Model error analysis complete")
    print(f"group_overlap_count: {split_info['group_overlap_count']}")
    print(
        "kmer ROC-AUC: "
        f"{format_metric(overall_metrics['kmer']['roc_auc'])}, "
        "PR-AUC: "
        f"{format_metric(overall_metrics['kmer']['average_precision'])}"
    )
    print(
        "mlp_pair ROC-AUC: "
        f"{format_metric(overall_metrics['mlp_pair']['roc_auc'])}, "
        "PR-AUC: "
        f"{format_metric(overall_metrics['mlp_pair']['average_precision'])}"
    )
    print(f"predictions: {PREDICTIONS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"metrics: {METRICS_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
