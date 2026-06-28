"""Train classifiers on frozen Hugging Face sequence-model embeddings.

This script is a supervised benchmark on existing labeled rows. It does not
fine-tune the pretrained model and works with existing public sequence records for benchmark analysis.

Run from the project root:

    python src/models/train_pretrained_frozen_baseline.py
"""

from __future__ import annotations

import json
import os
import random
import sys
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
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMBEDDING_DIR = PROJECT_ROOT / "data" / "processed" / "pretrained_embeddings"
HEAVY_EMBEDDING_PATH = EMBEDDING_DIR / "heavy.npy"
PAIR_EMBEDDING_PATH = EMBEDDING_DIR / "pair.npy"
METADATA_PATH = EMBEDDING_DIR / "metadata.csv"
EMBEDDING_REPORT_PATH = PROJECT_ROOT / "reports" / "pretrained_embedding_report.md"

REPORT_PATH = PROJECT_ROOT / "reports" / "pretrained_frozen_baseline_report.md"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics" / "pretrained_frozen_baseline_metrics.json"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
ROC_AUC_FIGURE_PATH = FIGURE_DIR / "pretrained_frozen_roc_auc_comparison.png"
PR_AUC_FIGURE_PATH = FIGURE_DIR / "pretrained_frozen_pr_auc_comparison.png"
MODEL_DIR = PROJECT_ROOT / "models"
HEAVY_MLP_MODEL_PATH = MODEL_DIR / "pretrained_frozen_mlp_heavy.pt"
PAIR_MLP_MODEL_PATH = MODEL_DIR / "pretrained_frozen_mlp_pair.pt"

RANDOM_STATE = 42
TEST_SIZE = 0.2
INNER_VALIDATION_SIZE = 0.15
MAX_GROUP_SPLIT_ATTEMPTS = 50
GROUP_COLUMN = "group_feature_v"

HIDDEN_LAYERS = [256, 64]
DROPOUT = 0.2
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 12
MIN_DELTA = 1e-4

MOSTLY_MISSING_THRESHOLD = 0.50
NEAR_ROW_UNIQUE_ROW_THRESHOLD = 0.80
NEAR_ROW_UNIQUE_NON_MISSING_THRESHOLD = 0.90

KMER_GROUPED_ROC_AUC = 0.7810
KMER_GROUPED_PR_AUC = 0.8236
ABLANG2_PAIR_MLP_GROUPED_ROC_AUC = 0.7573
ABLANG2_PAIR_MLP_GROUPED_PR_AUC = 0.8099

MLP_MODEL_PATHS = {
    "heavy": HEAVY_MLP_MODEL_PATH,
    "pair": PAIR_MLP_MODEL_PATH,
}


def read_metadata(path: Path) -> pd.DataFrame:
    """Read embedding metadata as text while preserving blank fields."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def normalized_text(values: pd.Series) -> pd.Series:
    """Normalize text for split checks."""
    return values.fillna("").astype(str).str.strip()


def parse_bool(value: Any, default: bool = False) -> bool:
    """Parse common boolean strings from CSV metadata."""
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", ""}:
        return False
    return default


def first_metadata_value(metadata: pd.DataFrame, column: str, default: str = "") -> str:
    """Return a stable metadata value when present."""
    if column not in metadata.columns or metadata.empty:
        return default
    values = sorted(set(normalized_text(metadata[column])) - {""})
    if not values:
        return default
    if len(values) == 1:
        return values[0]
    return "; ".join(values[:5])


def require_files(paths: list[Path]) -> None:
    """Fail clearly when required cached artifacts are missing."""
    missing = [path for path in paths if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path.relative_to(PROJECT_ROOT)) for path in missing)
        raise FileNotFoundError(
            f"Missing required artifact(s): {missing_text}. "
            "Run python src/models/embed_with_pretrained_sequence_model.py first."
        )


def load_inputs() -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    """Load cached embeddings and aligned metadata."""
    require_files([HEAVY_EMBEDDING_PATH, METADATA_PATH])
    metadata = read_metadata(METADATA_PATH)
    embeddings: dict[str, np.ndarray] = {
        "heavy": np.load(HEAVY_EMBEDDING_PATH).astype(np.float32)
    }

    pair_metadata_available = True
    if "pair_embedding_available" in metadata.columns and not metadata.empty:
        pair_metadata_available = parse_bool(metadata["pair_embedding_available"].iloc[0])
    pair_skip_reason = first_metadata_value(metadata, "pair_skip_reason", default="")
    if PAIR_EMBEDDING_PATH.exists() and pair_metadata_available:
        embeddings["pair"] = np.load(PAIR_EMBEDDING_PATH).astype(np.float32)

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

    counts = metadata["label"].value_counts()
    if len(counts) != 2 or counts.min() < 2:
        raise ValueError("Train/test evaluation requires at least two rows per label.")

    selected_positions = metadata.index.to_numpy()
    embeddings = {name: values[selected_positions] for name, values in embeddings.items()}
    metadata = metadata.reset_index(drop=True)

    model_status = {
        "available": True,
        "reason": "cached_embeddings_loaded",
        "model_name": first_metadata_value(metadata, "pretrained_model_name", "unknown"),
        "tokenizer_class": first_metadata_value(metadata, "tokenizer_class", "unknown"),
        "model_class": first_metadata_value(metadata, "model_class", "unknown"),
        "tokenization_style": first_metadata_value(
            metadata,
            "tokenization_style",
            "unknown",
        ),
        "embedding_report": (
            str(EMBEDDING_REPORT_PATH.relative_to(PROJECT_ROOT))
            if EMBEDDING_REPORT_PATH.exists()
            else None
        ),
        "pair_embedding_available": "pair" in embeddings,
        "pair_skip_reason": pair_skip_reason,
    }
    return metadata, embeddings, model_status


def set_seed(seed: int) -> None:
    """Set deterministic seeds for numpy, Python, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def label_counts(labels: pd.Series) -> dict[str, int]:
    """Return stable binary label counts for JSON and Markdown output."""
    counts = labels.value_counts().sort_index()
    return {"0": int(counts.get(0, 0)), "1": int(counts.get(1, 0))}


def group_column_status(data: pd.DataFrame) -> dict[str, Any]:
    """Decide whether group_feature_v supports grouped validation."""
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
    """Create a random stratified train/test split."""
    train_idx, test_idx = train_test_split(
        np.arange(len(data)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data["label"],
    )
    return np.asarray(train_idx), np.asarray(test_idx)


def grouped_split(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.Series, int]:
    """Create a meaningful group_feature_v split."""
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


def build_splits(data: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Build random and grouped split definitions."""
    train_idx, test_idx = random_split(data)
    splits: dict[str, dict[str, Any]] = {
        "random": {
            "valid": True,
            "meaningful": True,
            "reason": "ok",
            "group_column": None,
            "train_idx": train_idx,
            "test_idx": test_idx,
            "groups": None,
            "split_random_state": RANDOM_STATE,
        }
    }

    status = group_column_status(data)
    if not status["useful_for_grouping"]:
        splits[GROUP_COLUMN] = {
            "valid": False,
            "meaningful": False,
            "reason": status["reason"],
            "group_column": GROUP_COLUMN,
            "group_column_status": status,
        }
        return splits

    try:
        train_idx, test_idx, groups, split_random_state = grouped_split(data)
    except ValueError as exc:
        splits[GROUP_COLUMN] = {
            "valid": False,
            "meaningful": False,
            "reason": str(exc),
            "group_column": GROUP_COLUMN,
            "group_column_status": status,
        }
        return splits

    splits[GROUP_COLUMN] = {
        "valid": True,
        "meaningful": True,
        "reason": "ok",
        "group_column": GROUP_COLUMN,
        "group_column_status": status,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "groups": groups,
        "split_random_state": split_random_state,
    }
    return splits


def split_diagnostics(
    data: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    groups: pd.Series | None,
) -> dict[str, Any]:
    """Summarize split sizes, labels, and optional group overlap."""
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


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    """Compute scalar metrics and a confusion matrix."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    has_both_labels = len(set(y_true.tolist())) == 2
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


def make_logreg_pipeline() -> Pipeline:
    """Create the requested logistic-regression classifier."""
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
    """Return probabilities for label 1."""
    class_list = list(model.classes_)
    if 1 not in class_list:
        raise ValueError("Estimator was not fitted with positive class label 1.")
    positive_index = class_list.index(1)
    return model.predict_proba(values)[:, positive_index]


def train_logreg(
    metadata: pd.DataFrame,
    embeddings: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    """Train and evaluate logistic regression on one embedding matrix."""
    model = make_logreg_pipeline()
    y_all = metadata["label"].to_numpy(dtype=np.int64)
    model.fit(embeddings[train_idx], y_all[train_idx])

    y_true = y_all[test_idx]
    y_pred = model.predict(embeddings[test_idx])
    y_score = positive_scores(model, embeddings[test_idx])
    return {
        "classifier": "logistic_regression",
        "model_config": {
            "max_iter": 5000,
            "class_weight": "balanced",
            "scaler": "StandardScaler",
        },
        "metrics": {**diagnostics, **metric_dict(y_true, y_pred, y_score)},
    }


class EmbeddingMLP(nn.Module):
    """Small MLP for binary classification from frozen embeddings."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_LAYERS[0]),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_LAYERS[0], HIDDEN_LAYERS[1]),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_LAYERS[1], 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Return one logit per row."""
        return self.network(values).squeeze(-1)


def make_loader(
    x_values: np.ndarray,
    y_values: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Create a TensorDataset DataLoader."""
    dataset = TensorDataset(
        torch.from_numpy(x_values.astype(np.float32)),
        torch.from_numpy(y_values.astype(np.float32)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def predict_scores_and_loss(
    model: nn.Module,
    x_values: np.ndarray,
    y_values: np.ndarray,
    loss_fn: nn.Module | None,
    device: torch.device,
) -> tuple[np.ndarray, float | None]:
    """Predict probabilities and, when requested, average BCE loss."""
    model.eval()
    scores = []
    losses = []
    loader = make_loader(x_values, y_values.astype(np.float32), BATCH_SIZE * 4, shuffle=False)
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            scores.append(torch.sigmoid(logits).cpu().numpy())
            if loss_fn is not None:
                loss = loss_fn(logits, batch_y)
                losses.append(float(loss.detach().cpu().item()) * len(batch_x))

    score_values = np.concatenate(scores)
    if loss_fn is None or not losses:
        return score_values, None
    return score_values, float(sum(losses) / len(x_values))


def train_mlp(
    embedding_name: str,
    split_name: str,
    metadata: pd.DataFrame,
    embeddings: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    diagnostics: dict[str, Any],
    device_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train and evaluate one MLP, returning report data and checkpoint."""
    set_seed(RANDOM_STATE)
    if device_name == "cpu":
        torch.set_num_threads(1)

    device = torch.device(device_name)
    y_all = metadata["label"].to_numpy(dtype=np.int64)
    train_core_idx, val_idx = train_test_split(
        train_idx,
        test_size=INNER_VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_all[train_idx],
    )
    train_core_idx = np.asarray(train_core_idx)
    val_idx = np.asarray(val_idx)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(embeddings[train_core_idx]).astype(np.float32)
    x_val = scaler.transform(embeddings[val_idx]).astype(np.float32)
    x_test = scaler.transform(embeddings[test_idx]).astype(np.float32)
    y_train = y_all[train_core_idx].astype(np.float32)
    y_val = y_all[val_idx].astype(np.int64)
    y_test = y_all[test_idx].astype(np.int64)

    positives = float(np.sum(y_train == 1))
    negatives = float(np.sum(y_train == 0))
    pos_weight_value = negatives / positives if positives > 0 else 1.0

    model = EmbeddingMLP(input_dim=x_train.shape[1]).to(device)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    train_loader = make_loader(x_train, y_train, BATCH_SIZE, shuffle=True)

    best_state = None
    best_epoch = 0
    best_val_auc = -np.inf
    best_val_loss = np.inf
    stale_epochs = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        batch_losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))

        val_scores, val_loss = predict_scores_and_loss(
            model=model,
            x_values=x_val,
            y_values=y_val,
            loss_fn=loss_fn,
            device=device,
        )
        val_pred = (val_scores >= 0.5).astype(int)
        val_metrics = metric_dict(y_val, val_pred, val_scores)
        current_val_auc = (
            -np.inf if val_metrics["roc_auc"] is None else float(val_metrics["roc_auc"])
        )
        current_val_loss = float("inf") if val_loss is None else float(val_loss)

        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(np.mean(batch_losses)),
                "val_loss": None if val_loss is None else float(val_loss),
                "val_roc_auc": None if current_val_auc == -np.inf else current_val_auc,
                "val_average_precision": val_metrics["average_precision"],
            }
        )

        improved_auc = current_val_auc > best_val_auc + MIN_DELTA
        improved_loss = current_val_loss < best_val_loss - MIN_DELTA
        if improved_auc or (best_val_auc == -np.inf and improved_loss):
            best_val_auc = current_val_auc
            best_val_loss = current_val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= EARLY_STOPPING_PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_scores, _ = predict_scores_and_loss(
        model=model,
        x_values=x_test,
        y_values=y_test,
        loss_fn=None,
        device=device,
    )
    test_pred = (test_scores >= 0.5).astype(int)
    test_metrics = metric_dict(y_test, test_pred, test_scores)

    checkpoint = {
        "embedding_name": embedding_name,
        "split_name": split_name,
        "model_state_dict": {
            key: value.cpu() for key, value in model.state_dict().items()
        },
        "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
        "input_dim": int(x_train.shape[1]),
        "hidden_layers": HIDDEN_LAYERS,
        "dropout": DROPOUT,
        "best_epoch": int(best_epoch),
        "test_metrics": test_metrics,
    }

    result = {
        "classifier": "pytorch_mlp",
        "device": device_name,
        "input_dim": int(x_train.shape[1]),
        "inner_train_size": int(len(train_core_idx)),
        "validation_size": int(len(val_idx)),
        "best_epoch": int(best_epoch),
        "epochs_trained": int(len(history)),
        "pos_weight": float(pos_weight_value),
        "history": history,
        "model_config": {
            "hidden_layers": HIDDEN_LAYERS,
            "dropout": DROPOUT,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "optimizer": "AdamW",
            "loss": "BCEWithLogitsLoss",
        },
        "metrics": {**diagnostics, **test_metrics},
    }
    return result, checkpoint


def evaluate_all(
    metadata: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
    device_name: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Evaluate all embeddings, splits, and classifier types."""
    splits = build_splits(metadata)
    results: dict[str, Any] = {}
    checkpoints: dict[str, dict[str, Any]] = {}

    for split_name, split_info in splits.items():
        results[split_name] = {
            "valid": split_info.get("valid", False),
            "meaningful": split_info.get("meaningful", False),
            "reason": split_info.get("reason", "unknown"),
            "group_column": split_info.get("group_column"),
            "split_random_state": split_info.get("split_random_state"),
            "models": {},
        }
        if "group_column_status" in split_info:
            results[split_name]["group_column_status"] = split_info["group_column_status"]
        if not split_info.get("valid"):
            continue

        train_idx = split_info["train_idx"]
        test_idx = split_info["test_idx"]
        groups = split_info.get("groups")
        diagnostics = split_diagnostics(metadata, train_idx, test_idx, groups)
        results[split_name]["split"] = diagnostics

        for embedding_name, values in embeddings.items():
            print(f"Training {embedding_name} / {split_name} / logistic regression", flush=True)
            logreg_result = train_logreg(
                metadata=metadata,
                embeddings=values,
                train_idx=train_idx,
                test_idx=test_idx,
                diagnostics=diagnostics,
            )

            print(f"Training {embedding_name} / {split_name} / PyTorch MLP", flush=True)
            mlp_result, checkpoint = train_mlp(
                embedding_name=embedding_name,
                split_name=split_name,
                metadata=metadata,
                embeddings=values,
                train_idx=train_idx,
                test_idx=test_idx,
                diagnostics=diagnostics,
                device_name=device_name,
            )

            results[split_name]["models"][embedding_name] = {
                "embedding_shape": list(values.shape),
                "logistic_regression": logreg_result,
                "pytorch_mlp": mlp_result,
            }

            if split_name == GROUP_COLUMN or embedding_name not in checkpoints:
                checkpoints[embedding_name] = checkpoint

    return results, checkpoints


def save_mlp_checkpoints(checkpoints: dict[str, dict[str, Any]]) -> dict[str, str | None]:
    """Save requested MLP model artifacts."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths: dict[str, str | None] = {"heavy": None, "pair": None}
    for embedding_name, checkpoint in checkpoints.items():
        path = MLP_MODEL_PATHS[embedding_name]
        torch.save(checkpoint, path)
        saved_paths[embedding_name] = str(path.relative_to(PROJECT_ROOT))
    return saved_paths


def grouped_records(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten grouped split model metrics for comparison and summaries."""
    grouped = results.get(GROUP_COLUMN, {})
    if not grouped.get("valid"):
        return []
    records = []
    for embedding_name, embedding_result in grouped.get("models", {}).items():
        for classifier_name in ["logistic_regression", "pytorch_mlp"]:
            model_result = embedding_result[classifier_name]
            metrics = model_result["metrics"]
            records.append(
                {
                    "embedding": embedding_name,
                    "classifier": classifier_name,
                    "label": f"{classifier_name}_{embedding_name}",
                    "roc_auc": metrics.get("roc_auc"),
                    "average_precision": metrics.get("average_precision"),
                    "f1": metrics.get("f1"),
                    "metrics": metrics,
                }
            )
    return records


def best_grouped_record(results: dict[str, Any]) -> dict[str, Any] | None:
    """Return the best grouped frozen model by ROC-AUC, then PR-AUC."""
    records = [
        record
        for record in grouped_records(results)
        if record["roc_auc"] is not None and record["average_precision"] is not None
    ]
    if not records:
        return None
    records.sort(
        key=lambda record: (
            float(record["roc_auc"]),
            float(record["average_precision"]),
        ),
        reverse=True,
    )
    return records[0]


def build_comparison(results: dict[str, Any]) -> dict[str, Any]:
    """Compare frozen pretrained classifiers with requested baselines."""
    records = grouped_records(results)
    frozen_models = {}
    for record in records:
        roc_auc = record.get("roc_auc")
        pr_auc = record.get("average_precision")
        frozen_models[record["label"]] = {
            "embedding": record["embedding"],
            "classifier": record["classifier"],
            "roc_auc": roc_auc,
            "average_precision": pr_auc,
            "f1": record.get("f1"),
            "delta_roc_auc_vs_kmer": (
                float(roc_auc) - KMER_GROUPED_ROC_AUC if roc_auc is not None else None
            ),
            "delta_average_precision_vs_kmer": (
                float(pr_auc) - KMER_GROUPED_PR_AUC if pr_auc is not None else None
            ),
            "delta_roc_auc_vs_ablang2_pair_mlp": (
                float(roc_auc) - ABLANG2_PAIR_MLP_GROUPED_ROC_AUC
                if roc_auc is not None
                else None
            ),
            "delta_average_precision_vs_ablang2_pair_mlp": (
                float(pr_auc) - ABLANG2_PAIR_MLP_GROUPED_PR_AUC
                if pr_auc is not None
                else None
            ),
        }

    best = best_grouped_record(results)
    beats_kmer_roc = bool(best and best["roc_auc"] > KMER_GROUPED_ROC_AUC)
    beats_kmer_pr = bool(best and best["average_precision"] > KMER_GROUPED_PR_AUC)
    return {
        "baselines": {
            "kmer_tfidf_logreg_pair_text": {
                "grouped_roc_auc": KMER_GROUPED_ROC_AUC,
                "grouped_average_precision": KMER_GROUPED_PR_AUC,
            },
            "ablang2_pair_mlp": {
                "grouped_roc_auc": ABLANG2_PAIR_MLP_GROUPED_ROC_AUC,
                "grouped_average_precision": ABLANG2_PAIR_MLP_GROUPED_PR_AUC,
            },
        },
        "frozen_models": frozen_models,
        "best_grouped_model": best,
        "beats_kmer_grouped_roc_auc": beats_kmer_roc,
        "beats_kmer_grouped_average_precision": beats_kmer_pr,
        "beats_kmer_on_both_primary_metrics": beats_kmer_roc and beats_kmer_pr,
    }


def metric_value(value: float | None) -> float:
    """Convert optional metrics to NaN for plotting."""
    return np.nan if value is None else float(value)


def save_metric_comparison_figure(
    results: dict[str, Any],
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    """Save grouped comparison bars for one metric."""
    baseline_value = (
        KMER_GROUPED_ROC_AUC if metric == "roc_auc" else KMER_GROUPED_PR_AUC
    )
    ablang2_value = (
        ABLANG2_PAIR_MLP_GROUPED_ROC_AUC
        if metric == "roc_auc"
        else ABLANG2_PAIR_MLP_GROUPED_PR_AUC
    )
    records = [
        {"label": "k-mer logreg", "value": baseline_value},
        {"label": "AbLang2 pair MLP", "value": ablang2_value},
    ]
    for record in grouped_records(results):
        records.append(
            {
                "label": (
                    f"HF {record['classifier'].replace('_', ' ')} "
                    f"{record['embedding']}"
                ),
                "value": metric_value(record.get(metric)),
            }
        )

    if len(records) <= 2:
        return

    labels = [record["label"] for record in records]
    values = [record["value"] for record in records]
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(records))
    bars = ax.bar(x, values, color=colors[: len(records)])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for bar, value in zip(bars, values):
        if np.isfinite(value):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(value + 0.015, 0.98),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_figures(results: dict[str, Any]) -> None:
    """Save requested comparison figures."""
    save_metric_comparison_figure(
        results=results,
        metric="roc_auc",
        ylabel="Grouped ROC-AUC",
        title="Frozen Pretrained Embedding ROC-AUC Comparison",
        output_path=ROC_AUC_FIGURE_PATH,
    )
    save_metric_comparison_figure(
        results=results,
        metric="average_precision",
        ylabel="Grouped average precision / PR-AUC",
        title="Frozen Pretrained Embedding PR-AUC Comparison",
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


def format_metrics_table(results: dict[str, Any]) -> list[str]:
    """Format random and grouped metrics as Markdown rows."""
    lines = [
        (
            "| Split | Input | Classifier | Train size | Test size | Train labels | "
            "Test labels | Group overlap | Accuracy | Balanced accuracy | Precision | "
            "Recall | F1 | ROC-AUC | PR-AUC | Confusion matrix |"
        ),
        "|---|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for split_name, split_result in results.items():
        if not split_result.get("valid"):
            lines.append(
                f"| {split_name} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | "
                f"n/a | n/a | n/a | n/a | n/a | n/a | n/a | invalid: {split_result.get('reason')} |"
            )
            continue
        for embedding_name, embedding_result in split_result["models"].items():
            for classifier_name in ["logistic_regression", "pytorch_mlp"]:
                model_result = embedding_result[classifier_name]
                metrics = model_result["metrics"]
                lines.append(
                    f"| {split_name} | {embedding_name} | {classifier_name} | "
                    f"{metrics['train_size']} | {metrics['test_size']} | "
                    f"{format_counts(metrics['train_label_counts'])} | "
                    f"{format_counts(metrics['test_label_counts'])} | "
                    f"{format_nullable(metrics['group_overlap_count'])} | "
                    f"{metrics['accuracy']:.4f} | {metrics['balanced_accuracy']:.4f} | "
                    f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | "
                    f"{metrics['f1']:.4f} | {format_metric(metrics['roc_auc'])} | "
                    f"{format_metric(metrics['average_precision'])} | "
                    f"{metrics['confusion_matrix']} |"
                )
    return lines


def format_grouped_comparison(comparison: dict[str, Any]) -> list[str]:
    """Format grouped baseline comparisons."""
    lines = [
        (
            "| Model | Grouped ROC-AUC | Delta ROC-AUC vs k-mer | "
            "Grouped PR-AUC | Delta PR-AUC vs k-mer |"
        ),
        "|---|---:|---:|---:|---:|",
        (
            f"| k-mer TF-IDF + logistic regression | "
            f"{KMER_GROUPED_ROC_AUC:.4f} | 0.0000 | "
            f"{KMER_GROUPED_PR_AUC:.4f} | 0.0000 |"
        ),
        (
            f"| frozen AbLang2 pair MLP | "
            f"{ABLANG2_PAIR_MLP_GROUPED_ROC_AUC:.4f} | "
            f"{ABLANG2_PAIR_MLP_GROUPED_ROC_AUC - KMER_GROUPED_ROC_AUC:.4f} | "
            f"{ABLANG2_PAIR_MLP_GROUPED_PR_AUC:.4f} | "
            f"{ABLANG2_PAIR_MLP_GROUPED_PR_AUC - KMER_GROUPED_PR_AUC:.4f} |"
        ),
    ]
    for label, metrics in comparison["frozen_models"].items():
        lines.append(
            f"| {label} | {format_metric(metrics.get('roc_auc'))} | "
            f"{format_metric(metrics.get('delta_roc_auc_vs_kmer'))} | "
            f"{format_metric(metrics.get('average_precision'))} | "
            f"{format_metric(metrics.get('delta_average_precision_vs_kmer'))} |"
        )
    return lines


def input_winner_text(results: dict[str, Any]) -> str:
    """Describe whether heavy-only or pair inputs performed better."""
    by_input: dict[str, tuple[str, float, float]] = {}
    for record in grouped_records(results):
        roc_auc = record.get("roc_auc")
        pr_auc = record.get("average_precision")
        if roc_auc is None or pr_auc is None:
            continue
        current = by_input.get(record["embedding"])
        candidate = (record["classifier"], float(roc_auc), float(pr_auc))
        if current is None or (candidate[1], candidate[2]) > (current[1], current[2]):
            by_input[record["embedding"]] = candidate

    if not by_input:
        return "Grouped input comparison is unavailable."
    if "pair" not in by_input:
        classifier, roc_auc, pr_auc = by_input["heavy"]
        return (
            "Only heavy-only embeddings were available. Best grouped heavy model: "
            f"{classifier} ROC-AUC {roc_auc:.4f}, PR-AUC {pr_auc:.4f}."
        )
    heavy = by_input.get("heavy")
    pair = by_input.get("pair")
    if heavy is None:
        classifier, roc_auc, pr_auc = pair
        return (
            "Only pair embeddings were evaluated. Best grouped pair model: "
            f"{classifier} ROC-AUC {roc_auc:.4f}, PR-AUC {pr_auc:.4f}."
        )
    winner = "pair" if (pair[1], pair[2]) > (heavy[1], heavy[2]) else "heavy-only"
    return (
        f"{winner} worked better by grouped ROC-AUC. "
        f"Heavy best: {heavy[0]} ROC-AUC {heavy[1]:.4f}, PR-AUC {heavy[2]:.4f}; "
        f"pair best: {pair[0]} ROC-AUC {pair[1]:.4f}, PR-AUC {pair[2]:.4f}."
    )


def mlp_improvement_text(results: dict[str, Any]) -> str:
    """Describe whether MLPs improved over logistic regression."""
    grouped = results.get(GROUP_COLUMN, {})
    if not grouped.get("valid"):
        return "Grouped MLP-vs-logistic comparison is unavailable."
    statements = []
    any_improvement = False
    for embedding_name, embedding_result in grouped.get("models", {}).items():
        logreg = embedding_result["logistic_regression"]["metrics"]
        mlp = embedding_result["pytorch_mlp"]["metrics"]
        delta_roc = (
            mlp["roc_auc"] - logreg["roc_auc"]
            if mlp["roc_auc"] is not None and logreg["roc_auc"] is not None
            else None
        )
        delta_pr = (
            mlp["average_precision"] - logreg["average_precision"]
            if mlp["average_precision"] is not None
            and logreg["average_precision"] is not None
            else None
        )
        if delta_roc is not None and delta_roc > 0:
            any_improvement = True
        statements.append(
            f"{embedding_name}: delta ROC-AUC {format_metric(delta_roc)}, "
            f"delta PR-AUC {format_metric(delta_pr)}"
        )
    prefix = "The MLP improved over logistic regression by ROC-AUC for at least one input." if any_improvement else "The MLP did not improve over logistic regression by grouped ROC-AUC."
    return f"{prefix} {'; '.join(statements)}."


def build_interpretation(results: dict[str, Any], comparison: dict[str, Any]) -> list[str]:
    """Build a short, honest interpretation for the report."""
    best = comparison.get("best_grouped_model")
    lines = ["## Interpretation", ""]
    lines.append(input_winner_text(results))
    lines.append("")
    lines.append(mlp_improvement_text(results))
    lines.append("")
    if best:
        lines.append(
            "Best frozen pretrained grouped model: "
            f"`{best['label']}` with ROC-AUC {best['roc_auc']:.4f} and "
            f"PR-AUC {best['average_precision']:.4f}."
        )
    lines.append(
        "Beats k-mer grouped ROC-AUC: "
        f"{'yes' if comparison['beats_kmer_grouped_roc_auc'] else 'no'}."
    )
    lines.append(
        "Beats k-mer grouped PR-AUC: "
        f"{'yes' if comparison['beats_kmer_grouped_average_precision'] else 'no'}."
    )
    if comparison["beats_kmer_on_both_primary_metrics"]:
        lines.append("Conclusion: the frozen pretrained representation improves on the k-mer baseline.")
    else:
        lines.append(
            "Conclusion: the frozen pretrained representation is a useful benchmark, "
            "but it should not be claimed to beat the k-mer baseline unless both "
            "grouped ROC-AUC and PR-AUC improve."
        )
    lines.append("")
    return lines


def build_report(
    metadata: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
    model_status: dict[str, Any],
    results: dict[str, Any],
    comparison: dict[str, Any],
    device_name: str,
    saved_model_paths: dict[str, str | None],
) -> str:
    """Build the Markdown report."""
    lines = [
        "# Frozen Pretrained Sequence Model Baseline",
        "",
        "This report evaluates classifiers on frozen Hugging Face sequence-model",
        "representations for existing labeled rows only. No pretrained parameters",
        "were fine-tuned.",
        "",
        "## Model Availability",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Available | `{model_status['available']}` |",
        f"| Reason | `{model_status['reason']}` |",
        f"| Model name | `{model_status['model_name']}` |",
        f"| Tokenizer class | `{model_status['tokenizer_class']}` |",
        f"| Model class | `{model_status['model_class']}` |",
        f"| Tokenization style | `{model_status['tokenization_style']}` |",
        f"| Pair embedding available | `{model_status['pair_embedding_available']}` |",
        f"| Pair skip reason | `{model_status.get('pair_skip_reason') or 'n/a'}` |",
        f"| Classifier device | `{device_name}` |",
        "",
        "## Data",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Rows | {len(metadata)} |",
        f"| Label 0 count | {label_counts(metadata['label'])['0']} |",
        f"| Label 1 count | {label_counts(metadata['label'])['1']} |",
    ]
    for name, values in embeddings.items():
        lines.append(f"| {name} embedding shape | {list(values.shape)} |")

    lines.extend(["", "## Metrics", ""])
    lines.extend(format_metrics_table(results))
    lines.extend(["", "## Grouped Baseline Comparison", ""])
    lines.extend(format_grouped_comparison(comparison))
    lines.extend([""])
    lines.extend(build_interpretation(results, comparison))
    lines.extend(
        [
            "## Artifacts",
            "",
            f"- `{METRICS_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{ROC_AUC_FIGURE_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{PR_AUC_FIGURE_PATH.relative_to(PROJECT_ROOT)}`",
        ]
    )
    for path in saved_model_paths.values():
        if path:
            lines.append(f"- `{path}`")
    lines.append("")
    return "\n".join(lines)


def unavailable_report(reason: str) -> str:
    """Build a report for missing cached embeddings or dependencies."""
    return "\n".join(
        [
            "# Frozen Pretrained Sequence Model Baseline",
            "",
            "status: `unavailable`",
            f"reason: `{reason}`",
            "",
            "Run the embedding step first:",
            "",
            "```bash",
            "python src/models/embed_with_pretrained_sequence_model.py",
            "```",
            "",
            "No classifier artifacts were created.",
            "",
        ]
    )


def write_unavailable(reason: str) -> None:
    """Persist unavailable report and metrics JSON."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "unavailable",
        "reason": reason,
        "required_artifacts": {
            "heavy_embeddings": str(HEAVY_EMBEDDING_PATH.relative_to(PROJECT_ROOT)),
            "metadata": str(METADATA_PATH.relative_to(PROJECT_ROOT)),
        },
        "artifacts": {
            "report": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
            "metrics_json": str(METRICS_PATH.relative_to(PROJECT_ROOT)),
        },
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    REPORT_PATH.write_text(unavailable_report(reason), encoding="utf-8")


def print_summary(results: dict[str, Any], comparison: dict[str, Any]) -> None:
    """Print compact terminal metrics and conclusion."""
    print("\nFrozen pretrained classifier metrics")
    for line in format_metrics_table(results):
        print(line)
    print("\nGrouped comparison")
    for line in format_grouped_comparison(comparison):
        print(line)
    for line in build_interpretation(results, comparison):
        if line and not line.startswith("##"):
            print(line)


def main() -> int:
    """Run the frozen embedding classifier benchmark."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        metadata, embeddings, model_status = load_inputs()
    except Exception as exc:
        write_unavailable(str(exc))
        print(f"pretrained frozen baseline unavailable: {exc}")
        return 1

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device_name}", flush=True)
    results, checkpoints = evaluate_all(metadata, embeddings, device_name)
    saved_model_paths = save_mlp_checkpoints(checkpoints)
    comparison = build_comparison(results)
    save_figures(results)

    payload = {
        "status": "available",
        "metadata_path": str(METADATA_PATH.relative_to(PROJECT_ROOT)),
        "embedding_paths": {
            "heavy": str(HEAVY_EMBEDDING_PATH.relative_to(PROJECT_ROOT)),
            "pair": (
                str(PAIR_EMBEDDING_PATH.relative_to(PROJECT_ROOT))
                if "pair" in embeddings
                else None
            ),
        },
        "row_count": int(len(metadata)),
        "label_counts": label_counts(metadata["label"]),
        "embedding_shapes": {name: list(values.shape) for name, values in embeddings.items()},
        "model_status": model_status,
        "device": device_name,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "inner_validation_size": INNER_VALIDATION_SIZE,
        "results": results,
        "comparison": comparison,
        "artifacts": {
            "report": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
            "metrics_json": str(METRICS_PATH.relative_to(PROJECT_ROOT)),
            "roc_auc_comparison": str(ROC_AUC_FIGURE_PATH.relative_to(PROJECT_ROOT)),
            "pr_auc_comparison": str(PR_AUC_FIGURE_PATH.relative_to(PROJECT_ROOT)),
            "heavy_mlp_model": saved_model_paths.get("heavy"),
            "pair_mlp_model": saved_model_paths.get("pair"),
        },
    }
    METRICS_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        build_report(
            metadata=metadata,
            embeddings=embeddings,
            model_status=model_status,
            results=results,
            comparison=comparison,
            device_name=device_name,
            saved_model_paths=saved_model_paths,
        ),
        encoding="utf-8",
    )
    print_summary(results, comparison)
    return 0


if __name__ == "__main__":
    sys.exit(main())
