"""Train PyTorch MLP classifiers on frozen AbLang2 embeddings.

This script benchmarks small supervised neural-network classifiers on cached
AbLang2 embeddings for existing labeled rows. It does not fine-tune AbLang2 and
does not generate, mutate, optimize, rank, or propose biological sequences.

Run from the project root:

    python src/models/train_pytorch_embedding_mlp.py
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd


# Matplotlib needs a writable cache directory in this environment.
MPLCONFIGDIR = Path("/tmp") / "antibody_prioritization_mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt
import torch
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
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HEAVY_EMBEDDING_PATH = PROJECT_ROOT / "data" / "processed" / "embeddings_ablang2_heavy.npy"
PAIR_EMBEDDING_PATH = PROJECT_ROOT / "data" / "processed" / "embeddings_ablang2_pair.npy"
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "embeddings_ablang2_metadata.csv"
KMER_METRICS_PATH = PROJECT_ROOT / "reports" / "metrics" / "grouped_validation_metrics.json"
EMBEDDING_METRICS_PATH = (
    PROJECT_ROOT / "reports" / "metrics" / "embedding_baseline_metrics.json"
)
HYBRID_METRICS_PATH = PROJECT_ROOT / "reports" / "metrics" / "hybrid_baseline_metrics.json"

REPORT_PATH = PROJECT_ROOT / "reports" / "pytorch_embedding_mlp_report.md"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics" / "pytorch_embedding_mlp_metrics.json"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
TRAINING_CURVE_PATH = FIGURE_DIR / "pytorch_mlp_training_curves.png"
ROC_AUC_FIGURE_PATH = FIGURE_DIR / "pytorch_mlp_roc_auc_comparison.png"
PR_AUC_FIGURE_PATH = FIGURE_DIR / "pytorch_mlp_pr_auc_comparison.png"
MODEL_DIR = PROJECT_ROOT / "models"
HEAVY_MODEL_PATH = MODEL_DIR / "pytorch_mlp_heavy.pt"
PAIR_MODEL_PATH = MODEL_DIR / "pytorch_mlp_pair.pt"

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

EMBEDDING_MODEL_PATHS = {
    "heavy": HEAVY_MODEL_PATH,
    "pair": PAIR_MODEL_PATH,
}


def n_jobs_for_device(device_name: str) -> int:
    """Choose safe parallelism for CPU or GPU training."""
    requested = int(os.environ.get("PYTORCH_MLP_N_JOBS", "0"))
    if requested > 0:
        return requested
    if device_name == "cuda":
        return 1
    return min(4, os.cpu_count() or 1)


def read_metadata(path: Path) -> pd.DataFrame:
    """Read embedding metadata as text while preserving blank fields."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def normalized_text(values: pd.Series) -> pd.Series:
    """Normalize text for split checks."""
    return values.fillna("").astype(str).str.strip()


def require_files(paths: list[Path]) -> None:
    """Fail clearly when required cached artifacts are missing."""
    missing = [path for path in paths if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path.relative_to(PROJECT_ROOT)) for path in missing)
        raise FileNotFoundError(
            f"Missing required artifact(s): {missing_text}. "
            "Run python src/models/embed_with_ablang2.py first."
        )


def load_inputs() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Load labels, grouping metadata, and frozen embedding matrices."""
    require_files([HEAVY_EMBEDDING_PATH, PAIR_EMBEDDING_PATH, METADATA_PATH])

    metadata = read_metadata(METADATA_PATH)
    embeddings = {
        "heavy": np.load(HEAVY_EMBEDDING_PATH).astype(np.float32),
        "pair": np.load(PAIR_EMBEDDING_PATH).astype(np.float32),
    }

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
    return metadata, embeddings


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


def predict_scores(
    model: nn.Module,
    x_values: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Predict probabilities for label 1."""
    model.eval()
    scores = []
    loader = make_loader(
        x_values,
        np.zeros(len(x_values), dtype=np.float32),
        batch_size=BATCH_SIZE * 4,
        shuffle=False,
    )
    with torch.no_grad():
        for batch_x, _ in loader:
            logits = model(batch_x.to(device))
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores)


def metric_dict(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    """Compute classification metrics from scores."""
    y_pred = (y_score >= 0.5).astype(int)
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


def train_one_model(
    embedding_name: str,
    split_name: str,
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    split_info: dict[str, Any],
    device_name: str,
) -> dict[str, Any]:
    """Train one embedding MLP for one split."""
    set_seed(RANDOM_STATE)
    if device_name == "cpu":
        torch.set_num_threads(1)

    device = torch.device(device_name)
    train_idx = split_info["train_idx"]
    test_idx = split_info["test_idx"]
    groups = split_info.get("groups")
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

        val_scores = predict_scores(model, x_val, device)
        val_metrics = metric_dict(y_val, val_scores)
        val_loss = float(
            loss_fn(
                torch.logit(
                    torch.from_numpy(np.clip(val_scores, 1e-6, 1 - 1e-6)).to(device),
                    eps=1e-6,
                ),
                torch.from_numpy(y_val.astype(np.float32)).to(device),
            )
            .detach()
            .cpu()
            .item()
        )
        current_val_auc = (
            -np.inf if val_metrics["roc_auc"] is None else float(val_metrics["roc_auc"])
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(batch_losses)),
                "val_loss": val_loss,
                "val_roc_auc": None if current_val_auc == -np.inf else current_val_auc,
                "val_average_precision": val_metrics["average_precision"],
            }
        )

        improved_auc = current_val_auc > best_val_auc + MIN_DELTA
        improved_loss = val_loss < best_val_loss - MIN_DELTA
        if improved_auc or (best_val_auc == -np.inf and improved_loss):
            best_val_auc = current_val_auc
            best_val_loss = val_loss
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

    test_scores = predict_scores(model, x_test, device)
    test_metrics = metric_dict(y_test, test_scores)
    diagnostics = split_diagnostics(metadata, train_idx, test_idx, groups)

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

    return {
        "embedding_name": embedding_name,
        "split_name": split_name,
        "valid": True,
        "meaningful": True,
        "reason": "ok",
        "device": device_name,
        "input_dim": int(x_train.shape[1]),
        "inner_train_size": int(len(train_core_idx)),
        "validation_size": int(len(val_idx)),
        "best_epoch": int(best_epoch),
        "epochs_trained": int(len(history)),
        "pos_weight": float(pos_weight_value),
        "history": history,
        "split": diagnostics,
        "metrics": {**diagnostics, **test_metrics},
        "checkpoint": checkpoint,
    }


def run_training_jobs(
    metadata: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
    splits: dict[str, dict[str, Any]],
    device_name: str,
) -> dict[str, Any]:
    """Run all valid embedding/split jobs, parallelized on CPU."""
    jobs = []
    results: dict[str, Any] = {
        split_name: {
            "valid": split_info.get("valid", False),
            "meaningful": split_info.get("meaningful", False),
            "reason": split_info.get("reason", "unknown"),
            "group_column": split_info.get("group_column"),
            "models": {},
        }
        for split_name, split_info in splits.items()
    }

    for split_name, split_info in splits.items():
        if not split_info.get("valid"):
            continue
        for embedding_name, values in embeddings.items():
            jobs.append((embedding_name, split_name, values, split_info))

    def run_job(job: tuple[str, str, np.ndarray, dict[str, Any]]) -> dict[str, Any]:
        embedding_name, split_name, values, split_info = job
        print(f"Training {embedding_name} / {split_name}", flush=True)
        return train_one_model(
            embedding_name=embedding_name,
            split_name=split_name,
            embeddings=values,
            metadata=metadata,
            split_info=split_info,
            device_name=device_name,
        )

    n_jobs = n_jobs_for_device(device_name)
    if n_jobs == 1:
        completed = [run_job(job) for job in jobs]
    else:
        completed = joblib.Parallel(n_jobs=n_jobs, backend="loky")(
            joblib.delayed(run_job)(job) for job in jobs
        )

    for item in completed:
        split_name = item["split_name"]
        embedding_name = item["embedding_name"]
        results[split_name]["models"][embedding_name] = {
            key: value
            for key, value in item.items()
            if key not in {"checkpoint", "split_name", "embedding_name"}
        }
        if "split" not in results[split_name]:
            results[split_name]["split"] = item["split"]
        results[split_name]["valid"] = True
        results[split_name]["meaningful"] = True
        results[split_name]["reason"] = "ok"

        checkpoint_path = EMBEDDING_MODEL_PATHS[embedding_name]
        if split_name == GROUP_COLUMN or not checkpoint_path.exists():
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(item["checkpoint"], checkpoint_path)

    return results


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    """Load a JSON metrics file if available."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def extract_previous_metrics() -> dict[str, Any]:
    """Extract compact baseline metrics for comparison."""
    previous: dict[str, Any] = {
        "kmer": {},
        "embedding_logreg": {},
        "hybrid": {},
    }
    kmer = load_json_if_exists(KMER_METRICS_PATH)
    if kmer:
        for split_name in ["random", GROUP_COLUMN]:
            split = kmer.get("results", {}).get(split_name, {})
            if split.get("valid"):
                previous["kmer"][split_name] = (
                    split.get("kmer_logreg", {}).get("sequence_pair_text", {})
                )

    embedding = load_json_if_exists(EMBEDDING_METRICS_PATH)
    if embedding:
        for split_name in ["random", GROUP_COLUMN]:
            split = embedding.get("results", {}).get(split_name, {})
            if split.get("valid"):
                previous["embedding_logreg"][split_name] = split.get("models", {})

    hybrid = load_json_if_exists(HYBRID_METRICS_PATH)
    if hybrid:
        for split_name in ["random", GROUP_COLUMN]:
            split = hybrid.get("results", {}).get(split_name, {})
            if split.get("valid"):
                previous["hybrid"][split_name] = split.get("feature_sets", {})

    return previous


def best_model_by_grouped_metric(results: dict[str, Any], metric: str) -> dict[str, Any] | None:
    """Return the best grouped MLP result by one metric."""
    grouped = results.get(GROUP_COLUMN, {})
    if not grouped.get("valid"):
        return None
    best_name = None
    best_metrics = None
    best_value = -np.inf
    for embedding_name, model_result in grouped.get("models", {}).items():
        value = model_result["metrics"].get(metric)
        if value is not None and value > best_value:
            best_name = embedding_name
            best_metrics = model_result["metrics"]
            best_value = float(value)
    if best_name is None:
        return None
    return {
        "embedding_name": best_name,
        "metric": metric,
        "value": best_value,
        "metrics": best_metrics,
    }


def build_comparison(results: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    """Compare MLPs with k-mer, logistic embedding, and hybrid baselines."""
    comparison: dict[str, Any] = {}
    for split_name in ["random", GROUP_COLUMN]:
        split_result = results.get(split_name, {})
        if not split_result.get("valid"):
            continue
        comparison[split_name] = {
            "kmer_pair_text": previous["kmer"].get(split_name),
            "embedding_logreg": previous["embedding_logreg"].get(split_name, {}),
            "hybrid": previous["hybrid"].get(split_name, {}),
            "mlp": {},
        }
        for embedding_name, model_result in split_result["models"].items():
            metrics = model_result["metrics"]
            logreg = comparison[split_name]["embedding_logreg"].get(embedding_name)
            comparison[split_name]["mlp"][embedding_name] = {
                "roc_auc": metrics.get("roc_auc"),
                "average_precision": metrics.get("average_precision"),
                "f1": metrics.get("f1"),
                "delta_roc_auc_vs_embedding_logreg": (
                    metrics.get("roc_auc") - logreg.get("roc_auc")
                    if logreg
                    and metrics.get("roc_auc") is not None
                    and logreg.get("roc_auc") is not None
                    else None
                ),
                "delta_average_precision_vs_embedding_logreg": (
                    metrics.get("average_precision") - logreg.get("average_precision")
                    if logreg
                    and metrics.get("average_precision") is not None
                    and logreg.get("average_precision") is not None
                    else None
                ),
                "delta_roc_auc_vs_kmer": (
                    metrics.get("roc_auc")
                    - comparison[split_name]["kmer_pair_text"].get("roc_auc")
                    if comparison[split_name]["kmer_pair_text"]
                    and metrics.get("roc_auc") is not None
                    and comparison[split_name]["kmer_pair_text"].get("roc_auc") is not None
                    else None
                ),
            }
    return comparison


def save_training_curves(results: dict[str, Any]) -> None:
    """Save validation ROC-AUC and loss curves for all MLP runs."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for split_name, split_result in results.items():
        if not split_result.get("valid"):
            continue
        for embedding_name, model_result in split_result["models"].items():
            history = pd.DataFrame(model_result["history"])
            label = f"{split_name}/{embedding_name}"
            axes[0].plot(history["epoch"], history["val_loss"], label=label)
            axes[1].plot(history["epoch"], history["val_roc_auc"], label=label)

    axes[0].set_title("Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Validation ROC-AUC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("ROC-AUC")
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(TRAINING_CURVE_PATH, dpi=200)
    plt.close(fig)


def metric_value(value: float | None) -> float:
    """Convert optional metrics to NaN for plotting."""
    return np.nan if value is None else float(value)


def save_comparison_figure(
    results: dict[str, Any],
    previous: dict[str, Any],
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Save MLP vs baseline comparison for one metric."""
    records = []
    for split_name in ["random", GROUP_COLUMN]:
        kmer = previous["kmer"].get(split_name)
        if kmer:
            records.append(
                {
                    "split": split_name,
                    "model": "kmer",
                    "value": metric_value(kmer.get(metric)),
                }
            )
        split_result = results.get(split_name, {})
        if not split_result.get("valid"):
            continue
        for embedding_name, model_result in split_result["models"].items():
            records.append(
                {
                    "split": split_name,
                    "model": f"mlp_{embedding_name}",
                    "value": metric_value(model_result["metrics"].get(metric)),
                }
            )

    if not records:
        return

    table = pd.DataFrame.from_records(records)
    split_order = [name for name in ["random", GROUP_COLUMN] if name in set(table["split"])]
    model_order = ["kmer", "mlp_heavy", "mlp_pair"]
    x = np.arange(len(split_order))
    width = 0.26
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


def save_figures(results: dict[str, Any], previous: dict[str, Any]) -> None:
    """Save all requested MLP figures."""
    save_training_curves(results)
    save_comparison_figure(
        results,
        previous,
        metric="roc_auc",
        title="PyTorch MLP ROC-AUC Comparison",
        ylabel="ROC-AUC",
        output_path=ROC_AUC_FIGURE_PATH,
    )
    save_comparison_figure(
        results,
        previous,
        metric="average_precision",
        title="PyTorch MLP PR-AUC Comparison",
        ylabel="Average precision / PR-AUC",
        output_path=PR_AUC_FIGURE_PATH,
    )


def format_metric(value: float | None) -> str:
    """Format optional metric values."""
    return "n/a" if value is None else f"{value:.4f}"


def format_counts(counts: dict[str, int]) -> str:
    """Format label counts compactly."""
    return f"0={counts['0']}, 1={counts['1']}"


def format_nullable(value: Any) -> str:
    """Format nullable values."""
    return "n/a" if value is None else str(value)


def true_false(value: bool) -> str:
    """Format booleans as true/false."""
    return "true" if value else "false"


def format_metrics_table(results: dict[str, Any]) -> list[str]:
    """Format MLP metric rows."""
    lines = [
        (
            "| Split | Embedding | Device | Epochs | Best epoch | Train size | "
            "Test size | Group overlap | Accuracy | Balanced accuracy | Precision | "
            "Recall | F1 | ROC-AUC | PR-AUC | Confusion matrix |"
        ),
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for split_name, split_result in results.items():
        if not split_result.get("valid"):
            lines.append(
                f"| {split_name} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | "
                f"n/a | n/a | n/a | n/a | n/a | n/a | n/a | invalid: {split_result.get('reason')} |"
            )
            continue
        for embedding_name, model_result in split_result["models"].items():
            metrics = model_result["metrics"]
            lines.append(
                f"| {split_name} | {embedding_name} | {model_result['device']} | "
                f"{model_result['epochs_trained']} | {model_result['best_epoch']} | "
                f"{metrics['train_size']} | {metrics['test_size']} | "
                f"{format_nullable(metrics['group_overlap_count'])} | "
                f"{metrics['accuracy']:.4f} | {metrics['balanced_accuracy']:.4f} | "
                f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | "
                f"{metrics['f1']:.4f} | {format_metric(metrics['roc_auc'])} | "
                f"{format_metric(metrics['average_precision'])} | "
                f"{metrics['confusion_matrix']} |"
            )
    return lines


def format_comparison_table(comparison: dict[str, Any]) -> list[str]:
    """Format MLP comparison against previous baselines."""
    lines = [
        (
            "| Split | Embedding | K-mer ROC-AUC | MLP ROC-AUC | Delta vs k-mer | "
            "Embedding logreg ROC-AUC | Delta vs embedding logreg | K-mer PR-AUC | "
            "MLP PR-AUC | Delta PR-AUC vs embedding logreg |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split_name, split_comparison in comparison.items():
        kmer = split_comparison.get("kmer_pair_text")
        for embedding_name, metrics in split_comparison["mlp"].items():
            logreg = split_comparison.get("embedding_logreg", {}).get(embedding_name)
            lines.append(
                f"| {split_name} | {embedding_name} | "
                f"{format_metric(None if not kmer else kmer.get('roc_auc'))} | "
                f"{format_metric(metrics.get('roc_auc'))} | "
                f"{format_metric(metrics.get('delta_roc_auc_vs_kmer'))} | "
                f"{format_metric(None if not logreg else logreg.get('roc_auc'))} | "
                f"{format_metric(metrics.get('delta_roc_auc_vs_embedding_logreg'))} | "
                f"{format_metric(None if not kmer else kmer.get('average_precision'))} | "
                f"{format_metric(metrics.get('average_precision'))} | "
                f"{format_metric(metrics.get('delta_average_precision_vs_embedding_logreg'))} |"
            )
    return lines


def build_interpretation(results: dict[str, Any], comparison: dict[str, Any]) -> list[str]:
    """Build a short, honest interpretation."""
    best_roc = best_model_by_grouped_metric(results, "roc_auc")
    best_pr = best_model_by_grouped_metric(results, "average_precision")
    grouped_comparison = comparison.get(GROUP_COLUMN, {})
    kmer = grouped_comparison.get("kmer_pair_text")

    improves_kmer = False
    if best_roc and kmer and kmer.get("roc_auc") is not None:
        improves_kmer = best_roc["value"] > float(kmer["roc_auc"])

    improves_logreg = []
    for embedding_name, metrics in grouped_comparison.get("mlp", {}).items():
        delta = metrics.get("delta_roc_auc_vs_embedding_logreg")
        if delta is not None and delta > 0:
            improves_logreg.append(embedding_name)

    lines = [
        "## Interpretation",
        "",
        (
            "The PyTorch MLP improves over logistic regression on the same embeddings: "
            f"{', '.join(improves_logreg) if improves_logreg else 'no'}."
        ),
        (
            "The PyTorch MLP improves over the k-mer grouped ROC-AUC baseline: "
            f"{'yes' if improves_kmer else 'no'}."
        ),
    ]
    if best_roc:
        lines.append(
            f"Best grouped MLP ROC-AUC: `{best_roc['embedding_name']}` "
            f"({best_roc['value']:.4f})."
        )
    if best_pr:
        lines.append(
            f"Best grouped MLP PR-AUC: `{best_pr['embedding_name']}` "
            f"({best_pr['value']:.4f})."
        )
    lines.append(
        "Frozen AbLang2 embeddings remain useful for neural-network benchmarking, "
        "but this result should not be overclaimed if k-mer features stay stronger."
    )
    lines.append("")
    return lines


def build_report(
    metadata: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
    results: dict[str, Any],
    comparison: dict[str, Any],
    previous: dict[str, Any],
    device_name: str,
    n_jobs: int,
) -> str:
    """Build the Markdown report."""
    lines = [
        "# PyTorch AbLang2 Embedding MLP",
        "",
        f"Metadata: `{METADATA_PATH.relative_to(PROJECT_ROOT)}`",
        f"Device: `{device_name}`",
        f"Parallel jobs: `{n_jobs}`",
        "",
        "This report evaluates small PyTorch MLP classifiers on frozen AbLang2",
        "embeddings for existing labeled rows.",
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

    lines.extend(["", "## MLP Metrics", ""])
    lines.extend(format_metrics_table(results))
    lines.extend(["", "## Comparison Against Baselines", ""])
    lines.extend(format_comparison_table(comparison))
    lines.extend([""])
    lines.extend(build_interpretation(results, comparison))
    lines.extend(
        [
            "## Artifacts",
            "",
            f"- `{METRICS_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{TRAINING_CURVE_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{ROC_AUC_FIGURE_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{PR_AUC_FIGURE_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{HEAVY_MODEL_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{PAIR_MODEL_PATH.relative_to(PROJECT_ROOT)}`",
            "",
        ]
    )
    return "\n".join(lines)


def print_summary(results: dict[str, Any], comparison: dict[str, Any]) -> None:
    """Print compact requested terminal output."""
    print("\nPyTorch MLP metrics")
    for line in format_metrics_table(results):
        print(line)
    print("\ncomparison against baselines")
    for line in format_comparison_table(comparison):
        print(line)


def main() -> None:
    """Run the PyTorch embedding MLP benchmark."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    metadata, embeddings = load_inputs()
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    n_jobs = n_jobs_for_device(device_name)
    print(f"Device: {device_name}", flush=True)
    print(f"Parallel jobs: {n_jobs}", flush=True)

    splits = build_splits(metadata)
    results = run_training_jobs(metadata, embeddings, splits, device_name)
    previous = extract_previous_metrics()
    comparison = build_comparison(results, previous)
    save_figures(results, previous)

    payload = {
        "metadata_path": str(METADATA_PATH.relative_to(PROJECT_ROOT)),
        "embedding_paths": {
            "heavy": str(HEAVY_EMBEDDING_PATH.relative_to(PROJECT_ROOT)),
            "pair": str(PAIR_EMBEDDING_PATH.relative_to(PROJECT_ROOT)),
        },
        "row_count": int(len(metadata)),
        "label_counts": label_counts(metadata["label"]),
        "embedding_shapes": {name: list(values.shape) for name, values in embeddings.items()},
        "device": device_name,
        "n_jobs": n_jobs,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "inner_validation_size": INNER_VALIDATION_SIZE,
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
        "results": results,
        "comparison": comparison,
        "best_grouped_roc_auc": best_model_by_grouped_metric(results, "roc_auc"),
        "best_grouped_average_precision": best_model_by_grouped_metric(
            results,
            "average_precision",
        ),
        "artifacts": {
            "report": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
            "metrics_json": str(METRICS_PATH.relative_to(PROJECT_ROOT)),
            "training_curves": str(TRAINING_CURVE_PATH.relative_to(PROJECT_ROOT)),
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
        build_report(metadata, embeddings, results, comparison, previous, device_name, n_jobs),
        encoding="utf-8",
    )
    print_summary(results, comparison)


if __name__ == "__main__":
    main()
