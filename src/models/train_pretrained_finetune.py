"""Fine-tune a Hugging Face pretrained sequence model on pair inputs.

This script uses only existing labeled rows from the neutral ML table. It does
not create, mutate, optimize, rank, or propose biological sequences.

Run from the project root:

    python src/models/train_pretrained_finetune.py

Optionally choose a Hugging Face model:

    PRETRAINED_SEQUENCE_MODEL=Exscientia/IgBert \
        python src/models/train_pretrained_finetune.py
"""

from __future__ import annotations

import copy
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
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from transformers import AutoModel, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.embed_with_pretrained_sequence_model import (
    construct_pair_text,
    ensure_pad_token,
    last_hidden_state,
    max_length_from_model,
    mean_pool,
    normalize_sequence_text,
    optional_text_column,
    read_input,
    run_model_forward,
    split_pair_text,
    text_column,
    tokenize_pairs,
    choose_tokenization_style,
)


INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "neutral_sequence_classification_ml.csv"

REPORT_PATH = PROJECT_ROOT / "reports" / "pretrained_finetune_report.md"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics" / "pretrained_finetune_metrics.json"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
TRAINING_CURVE_PATH = FIGURE_DIR / "pretrained_finetune_training_curves.png"
ROC_AUC_FIGURE_PATH = FIGURE_DIR / "pretrained_finetune_roc_auc_comparison.png"
PR_AUC_FIGURE_PATH = FIGURE_DIR / "pretrained_finetune_pr_auc_comparison.png"
MODEL_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODEL_DIR / "pretrained_finetune_best.pt"

DEFAULT_MODEL_NAME = "Exscientia/IgBert"
MODEL_ENV_VAR = "PRETRAINED_SEQUENCE_MODEL"
MODEL_NAME = os.environ.get(MODEL_ENV_VAR, DEFAULT_MODEL_NAME)

RANDOM_STATE = 42
TEST_SIZE = 0.2
INNER_VALIDATION_SIZE = 0.15
MAX_GROUP_SPLIT_ATTEMPTS = 50
GROUP_COLUMN = "group_feature_v"

DROP_OUT = 0.2
HEAD_LEARNING_RATE = 1e-3
BACKBONE_LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
MAX_EPOCHS = 20
BATCH_SIZE = int(os.environ.get("PRETRAINED_FINETUNE_BATCH_SIZE", "8"))
EARLY_STOPPING_PATIENCE = 4
MIN_DELTA = 1e-4
MAX_GRAD_NORM = 1.0

MOSTLY_MISSING_THRESHOLD = 0.50
NEAR_ROW_UNIQUE_ROW_THRESHOLD = 0.80
NEAR_ROW_UNIQUE_NON_MISSING_THRESHOLD = 0.90

KMER_GROUPED_ROC_AUC = 0.7810
KMER_GROUPED_PR_AUC = 0.8236
FROZEN_PAIR_MLP_GROUPED_ROC_AUC = 0.7541
FROZEN_PAIR_MLP_GROUPED_PR_AUC = 0.8078

TRAINING_MODES = ["head_only", "last_1_layer", "last_2_layers"]


class PairTextDataset(Dataset):
    """Dataset of existing pair-text rows and binary labels."""

    def __init__(self, pair_texts: list[str], labels: np.ndarray) -> None:
        self.pair_texts = pair_texts
        self.labels = labels.astype(np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[str, np.float32]:
        return self.pair_texts[index], np.float32(self.labels[index])


class SequenceFineTuner(nn.Module):
    """Mean-pooled sequence model plus a one-logit classification head."""

    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        dropout: float,
        freeze_backbone_forward: bool,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone_forward = freeze_backbone_forward
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return one logit per pair input."""
        if self.freeze_backbone_forward:
            # The backbone is fully frozen in head-only mode, so keep its
            # dropout disabled and avoid building an autograd graph for it.
            self.backbone.eval()
            with torch.no_grad():
                outputs = run_model_forward(self.backbone, batch)
        else:
            outputs = run_model_forward(self.backbone, batch)

        hidden = last_hidden_state(outputs)
        pooled = mean_pool(hidden, batch.get("attention_mask"))
        return self.classifier(self.dropout(pooled)).squeeze(-1)


def set_seed(seed: int) -> None:
    """Set deterministic seeds for numpy, Python, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalized_text(values: pd.Series) -> pd.Series:
    """Normalize text for split checks."""
    return values.fillna("").astype(str).str.strip()


def label_counts(labels: pd.Series | np.ndarray) -> dict[str, int]:
    """Return stable binary label counts for JSON and Markdown output."""
    series = pd.Series(labels).astype(int)
    counts = series.value_counts().sort_index()
    return {"0": int(counts.get(0, 0)), "1": int(counts.get(1, 0))}


def require_batch_size() -> None:
    """Keep the fine-tuning batch size in the requested small-batch range."""
    if BATCH_SIZE not in {8, 16}:
        raise ValueError(
            "PRETRAINED_FINETUNE_BATCH_SIZE must be 8 or 16 for this benchmark."
        )


def load_dataset() -> pd.DataFrame:
    """Load labels, pair text, and grouping metadata from existing rows."""
    data = read_input(INPUT_PATH)
    heavy = text_column(data, "sequence_heavy_only", ["sequence_a"]).map(
        normalize_sequence_text
    )
    light = optional_text_column(
        data,
        "sequence_b",
        ["sequence_light_only", "light_sequence"],
    )
    if "sequence_pair_text" in data.columns:
        pair = data["sequence_pair_text"].fillna("").astype(str)
    else:
        pair = construct_pair_text(heavy, light)

    output = pd.DataFrame(
        {
            "row_id": np.arange(len(data), dtype=int),
            "sequence_pair_text": pair.astype(str),
            "label": text_column(data, "label", []).astype(int),
            GROUP_COLUMN: optional_text_column(data, GROUP_COLUMN).astype(str),
        }
    )

    output = output[normalized_text(output["sequence_pair_text"]).ne("")].copy()
    unexpected_labels = sorted(set(output["label"]) - {0, 1})
    if unexpected_labels:
        raise ValueError(f"Expected binary labels 0/1, found: {unexpected_labels}")
    counts = output["label"].value_counts()
    if len(counts) != 2 or counts.min() < 2:
        raise ValueError("Fine-tuning requires at least two rows per label.")
    return output.reset_index(drop=True)


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


def grouped_train_test_split(data: pd.DataFrame) -> dict[str, Any]:
    """Create the existing group_feature_v zero-overlap train/test split."""
    status = group_column_status(data)
    if not status["useful_for_grouping"]:
        raise ValueError(f"group_feature_v is not useful for grouping: {status['reason']}")

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
            train_idx, test_idx = next(
                splitter.split(data, data["label"], groups=groups)
            )
        except ValueError as exc:
            last_error = str(exc)
            break

        train_labels = data.iloc[train_idx]["label"]
        test_labels = data.iloc[test_idx]["label"]
        if train_labels.nunique() != 2:
            last_error = "train_split_single_label"
            continue
        if test_labels.nunique() != 2:
            last_error = "test_split_single_label"
            continue

        train_group_set = set(groups.iloc[train_idx].astype(str))
        test_group_set = set(groups.iloc[test_idx].astype(str))
        overlap = train_group_set & test_group_set
        if overlap:
            last_error = "group_overlap"
            continue

        return {
            "train_idx": np.asarray(train_idx),
            "test_idx": np.asarray(test_idx),
            "groups": groups,
            "split_random_state": split_random_state,
            "group_column_status": status,
        }
    raise ValueError(last_error)


def inner_validation_split(
    data: pd.DataFrame,
    train_idx: np.ndarray,
    groups: pd.Series,
) -> dict[str, Any]:
    """Create a grouped validation split inside train, falling back if needed."""
    train_data = data.iloc[train_idx].reset_index(drop=True)
    train_groups = groups.iloc[train_idx].reset_index(drop=True)
    last_error = "no_valid_inner_grouped_split"

    for offset in range(MAX_GROUP_SPLIT_ATTEMPTS):
        split_random_state = RANDOM_STATE + 100 + offset
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=INNER_VALIDATION_SIZE,
            random_state=split_random_state,
        )
        try:
            inner_train_local, val_local = next(
                splitter.split(train_data, train_data["label"], groups=train_groups)
            )
        except ValueError as exc:
            last_error = str(exc)
            break

        inner_train_labels = train_data.iloc[inner_train_local]["label"]
        val_labels = train_data.iloc[val_local]["label"]
        if inner_train_labels.nunique() != 2:
            last_error = "inner_train_split_single_label"
            continue
        if val_labels.nunique() != 2:
            last_error = "validation_split_single_label"
            continue

        inner_train_group_set = set(train_groups.iloc[inner_train_local].astype(str))
        val_group_set = set(train_groups.iloc[val_local].astype(str))
        if inner_train_group_set & val_group_set:
            last_error = "inner_group_overlap"
            continue

        return {
            "method": "grouped",
            "reason": "ok",
            "train_core_idx": train_idx[np.asarray(inner_train_local)],
            "val_idx": train_idx[np.asarray(val_local)],
            "split_random_state": split_random_state,
            "group_overlap_count": 0,
        }

    inner_train_idx, val_idx = train_test_split(
        train_idx,
        test_size=INNER_VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=data.iloc[train_idx]["label"],
    )
    inner_train_idx = np.asarray(inner_train_idx)
    val_idx = np.asarray(val_idx)
    train_group_set = set(groups.iloc[inner_train_idx].astype(str))
    val_group_set = set(groups.iloc[val_idx].astype(str))
    return {
        "method": "random_stratified_fallback",
        "reason": last_error,
        "train_core_idx": inner_train_idx,
        "val_idx": val_idx,
        "split_random_state": RANDOM_STATE,
        "group_overlap_count": int(len(train_group_set & val_group_set)),
    }


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
        "train_label_counts": label_counts(train_labels.to_numpy()),
        "test_label_counts": label_counts(test_labels.to_numpy()),
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


def nested_getattr(root: Any, path: str) -> Any | None:
    """Safely retrieve a dotted module path."""
    value = root
    for part in path.split("."):
        if not hasattr(value, part):
            return None
        value = getattr(value, part)
    return value


def identify_transformer_layers(backbone: nn.Module) -> dict[str, Any]:
    """Find a ModuleList/list of transformer layers when the architecture is known."""
    candidates = [
        "encoder.layer",
        "bert.encoder.layer",
        "roberta.encoder.layer",
        "base_model.encoder.layer",
        "model.encoder.layers",
        "encoder.layers",
    ]
    for path in candidates:
        value = nested_getattr(backbone, path)
        if value is None:
            continue
        if isinstance(value, (nn.ModuleList, list, tuple)) and value:
            if all(isinstance(layer, nn.Module) for layer in value):
                return {
                    "available": True,
                    "path": path,
                    "layer_count": int(len(value)),
                    "layers": list(value),
                    "reason": "ok",
                }
    return {
        "available": False,
        "path": None,
        "layer_count": 0,
        "layers": [],
        "reason": "could_not_identify_transformer_layer_list",
    }


def configure_trainable_parameters(
    model: SequenceFineTuner,
    mode: str,
    layer_info: dict[str, Any],
) -> dict[str, Any]:
    """Freeze/unfreeze the requested model parts."""
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(False)

    unfreezed_layer_count = 0
    skipped_reason = None
    if mode == "head_only":
        pass
    elif mode == "last_1_layer":
        if not layer_info["available"] or layer_info["layer_count"] < 1:
            skipped_reason = layer_info["reason"]
        else:
            for parameter in layer_info["layers"][-1].parameters():
                parameter.requires_grad_(True)
            unfreezed_layer_count = 1
    elif mode == "last_2_layers":
        if not layer_info["available"] or layer_info["layer_count"] < 2:
            skipped_reason = layer_info["reason"]
        else:
            for layer in layer_info["layers"][-2:]:
                for parameter in layer.parameters():
                    parameter.requires_grad_(True)
            unfreezed_layer_count = 2
    else:
        skipped_reason = f"unknown_mode:{mode}"

    for parameter in model.classifier.parameters():
        parameter.requires_grad_(True)

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return {
        "mode": mode,
        "skipped": skipped_reason is not None,
        "skip_reason": skipped_reason,
        "unfreezed_transformer_layer_count": unfreezed_layer_count,
        "trainable_parameter_count": int(trainable),
        "total_parameter_count": int(total),
        "trainable_parameter_fraction": float(trainable / total) if total else 0.0,
    }


def make_collate_fn(
    tokenizer: Any,
    style: str,
    max_length: int | None,
) -> Any:
    """Create a collate function that dynamically tokenizes pair text."""

    def collate(batch: list[tuple[str, np.float32]]) -> dict[str, Any]:
        texts = [item[0] for item in batch]
        labels = torch.tensor([float(item[1]) for item in batch], dtype=torch.float32)
        encoded = tokenize_pairs(tokenizer, texts, style, max_length)
        return {"encoded": encoded, "labels": labels}

    return collate


def make_loader(
    data: pd.DataFrame,
    indices: np.ndarray,
    tokenizer: Any,
    style: str,
    max_length: int | None,
    shuffle: bool,
) -> DataLoader:
    """Build a dataloader for pair-text fine-tuning."""
    selected = data.iloc[indices]
    dataset = PairTextDataset(
        selected["sequence_pair_text"].astype(str).tolist(),
        selected["label"].to_numpy(dtype=np.float32),
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        collate_fn=make_collate_fn(tokenizer, style, max_length),
    )


def move_encoded_to_device(encoded: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tokenizer tensors onto the selected device."""
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in encoded.items()
    }


def evaluate_model(
    model: SequenceFineTuner,
    loader: DataLoader,
    loss_fn: nn.Module | None,
    device: torch.device,
) -> tuple[dict[str, Any], float | None, np.ndarray, np.ndarray]:
    """Evaluate a model and return metrics, loss, labels, and scores."""
    model.eval()
    losses = []
    labels = []
    scores = []

    with torch.no_grad():
        for batch in loader:
            encoded = move_encoded_to_device(batch["encoded"], device)
            y = batch["labels"].to(device)
            logits = model(encoded)
            if loss_fn is not None:
                loss = loss_fn(logits, y)
                losses.append(float(loss.detach().cpu().item()) * len(y))
            scores.append(torch.sigmoid(logits).detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

    y_true = np.concatenate(labels).astype(int)
    y_score = np.concatenate(scores)
    y_pred = (y_score >= 0.5).astype(int)
    avg_loss = None if loss_fn is None else float(sum(losses) / len(y_true))
    return metric_dict(y_true, y_pred, y_score), avg_loss, y_true, y_score


def train_one_mode(
    mode: str,
    data: pd.DataFrame,
    split_info: dict[str, Any],
    validation_info: dict[str, Any],
    tokenizer: Any,
    style: str,
    max_length: int | None,
    device_name: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Fine-tune one controlled mode and return metrics plus checkpoint."""
    set_seed(RANDOM_STATE)
    device = torch.device(device_name)
    backbone = AutoModel.from_pretrained(MODEL_NAME)
    backbone.to(device)
    hidden_size = int(getattr(backbone.config, "hidden_size"))
    layer_info = identify_transformer_layers(backbone)
    model = SequenceFineTuner(
        backbone=backbone,
        hidden_size=hidden_size,
        dropout=DROP_OUT,
        freeze_backbone_forward=(mode == "head_only"),
    ).to(device)

    trainability = configure_trainable_parameters(model, mode, layer_info)
    result_base = {
        "mode": mode,
        "valid": not trainability["skipped"],
        "reason": trainability["skip_reason"] or "ok",
        "layer_identification": {
            key: value
            for key, value in layer_info.items()
            if key != "layers"
        },
        "trainability": trainability,
    }
    if trainability["skipped"]:
        return result_base, None

    train_idx = split_info["train_idx"]
    test_idx = split_info["test_idx"]
    groups = split_info["groups"]
    train_core_idx = validation_info["train_core_idx"]
    val_idx = validation_info["val_idx"]

    train_loader = make_loader(data, train_core_idx, tokenizer, style, max_length, shuffle=True)
    val_loader = make_loader(data, val_idx, tokenizer, style, max_length, shuffle=False)
    test_loader = make_loader(data, test_idx, tokenizer, style, max_length, shuffle=False)

    y_train = data.iloc[train_core_idx]["label"].to_numpy(dtype=np.float32)
    positives = float(np.sum(y_train == 1))
    negatives = float(np.sum(y_train == 0))
    pos_weight_value = negatives / positives if positives > 0 else 1.0
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    )

    head_params = list(model.classifier.parameters())
    head_param_ids = {id(parameter) for parameter in head_params}
    backbone_params = [
        parameter
        for name, parameter in model.backbone.named_parameters()
        if parameter.requires_grad and id(parameter) not in head_param_ids
    ]
    optimizer_groups = [
        {
            "params": head_params,
            "lr": HEAD_LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
        }
    ]
    if backbone_params:
        optimizer_groups.append(
            {
                "params": backbone_params,
                "lr": BACKBONE_LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
            }
        )
    optimizer = torch.optim.AdamW(optimizer_groups)

    best_state = None
    best_epoch = 0
    best_val_auc = -np.inf
    best_val_loss = np.inf
    stale_epochs = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        batch_losses = []
        progress = tqdm(train_loader, desc=f"{mode} epoch {epoch}", leave=False)
        for batch in progress:
            encoded = move_encoded_to_device(batch["encoded"], device)
            y = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(encoded)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                MAX_GRAD_NORM,
            )
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))

        val_metrics, val_loss, _, _ = evaluate_model(model, val_loader, loss_fn, device)
        current_val_auc = (
            -np.inf if val_metrics["roc_auc"] is None else float(val_metrics["roc_auc"])
        )
        current_val_loss = float("inf") if val_loss is None else float(val_loss)
        train_loss = float(np.mean(batch_losses))

        history.append(
            {
                "epoch": int(epoch),
                "train_loss": train_loss,
                "validation_loss": None if val_loss is None else float(val_loss),
                "validation_roc_auc": None if current_val_auc == -np.inf else current_val_auc,
                "validation_average_precision": val_metrics["average_precision"],
                "validation_f1": val_metrics["f1"],
            }
        )
        print(
            f"{mode} epoch {epoch}: train_loss={train_loss:.4f} "
            f"val_loss={current_val_loss:.4f} val_roc_auc={current_val_auc:.4f}",
            flush=True,
        )

        improved_auc = current_val_auc > best_val_auc + MIN_DELTA
        improved_loss = current_val_loss < best_val_loss - MIN_DELTA
        if improved_auc or (best_val_auc == -np.inf and improved_loss):
            best_val_auc = current_val_auc
            best_val_loss = current_val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= EARLY_STOPPING_PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics, test_loss, _, _ = evaluate_model(model, test_loader, loss_fn, device)
    split_diag = split_diagnostics(data, train_idx, test_idx, groups)
    val_diag = split_diagnostics(data, train_core_idx, val_idx, groups)
    checkpoint = {
        "model_name": MODEL_NAME,
        "mode": mode,
        "tokenization_style": style,
        "max_length": max_length,
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "hidden_size": hidden_size,
        "dropout": DROP_OUT,
        "best_epoch": int(best_epoch),
        "test_metrics": test_metrics,
        "trainability": trainability,
    }

    overfit = overfitting_summary(history)
    result = {
        **result_base,
        "device": device_name,
        "hidden_size": hidden_size,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "epochs_trained": int(len(history)),
        "best_epoch": int(best_epoch),
        "pos_weight": float(pos_weight_value),
        "history": history,
        "split": split_diag,
        "validation_split": {
            **val_diag,
            "method": validation_info["method"],
            "reason": validation_info["reason"],
            "split_random_state": validation_info["split_random_state"],
        },
        "test_loss": test_loss,
        "metrics": {**split_diag, **test_metrics},
        "overfitting": overfit,
        "optimizer": {
            "type": "AdamW",
            "head_learning_rate": HEAD_LEARNING_RATE,
            "backbone_learning_rate": BACKBONE_LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_grad_norm": MAX_GRAD_NORM,
        },
    }
    return result, checkpoint


def overfitting_summary(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize simple evidence of overfitting from the training history."""
    if not history:
        return {"available": False, "evidence": False, "reason": "empty_history"}

    valid_roc = [
        item for item in history if item.get("validation_roc_auc") is not None
    ]
    valid_loss = [
        item for item in history if item.get("validation_loss") is not None
    ]
    if not valid_roc or not valid_loss:
        return {"available": False, "evidence": False, "reason": "missing_validation_metrics"}

    best_roc_item = max(valid_roc, key=lambda item: float(item["validation_roc_auc"]))
    best_loss_item = min(valid_loss, key=lambda item: float(item["validation_loss"]))
    last_item = history[-1]
    roc_drop = (
        float(best_roc_item["validation_roc_auc"])
        - float(last_item.get("validation_roc_auc") or best_roc_item["validation_roc_auc"])
    )
    loss_increase = (
        float(last_item.get("validation_loss") or best_loss_item["validation_loss"])
        - float(best_loss_item["validation_loss"])
    )
    train_loss_drop = float(history[0]["train_loss"]) - float(last_item["train_loss"])
    evidence = bool(train_loss_drop > 0 and (roc_drop > 0.02 or loss_increase > 0.05))
    return {
        "available": True,
        "evidence": evidence,
        "best_validation_roc_auc": float(best_roc_item["validation_roc_auc"]),
        "best_validation_roc_auc_epoch": int(best_roc_item["epoch"]),
        "last_validation_roc_auc": last_item.get("validation_roc_auc"),
        "validation_roc_auc_drop_from_best": float(roc_drop),
        "best_validation_loss": float(best_loss_item["validation_loss"]),
        "best_validation_loss_epoch": int(best_loss_item["epoch"]),
        "last_validation_loss": last_item.get("validation_loss"),
        "validation_loss_increase_from_best": float(loss_increase),
        "train_loss_drop": float(train_loss_drop),
    }


def build_comparison(results: dict[str, Any]) -> dict[str, Any]:
    """Compare fine-tuning modes against current grouped baselines."""
    modes = {}
    for mode, result in results.items():
        if not result.get("valid"):
            modes[mode] = {"valid": False, "reason": result.get("reason", "unknown")}
            continue
        metrics = result["metrics"]
        roc_auc = metrics.get("roc_auc")
        pr_auc = metrics.get("average_precision")
        modes[mode] = {
            "valid": True,
            "roc_auc": roc_auc,
            "average_precision": pr_auc,
            "f1": metrics.get("f1"),
            "delta_roc_auc_vs_kmer": (
                float(roc_auc) - KMER_GROUPED_ROC_AUC if roc_auc is not None else None
            ),
            "delta_average_precision_vs_kmer": (
                float(pr_auc) - KMER_GROUPED_PR_AUC if pr_auc is not None else None
            ),
            "delta_roc_auc_vs_frozen_pair_mlp": (
                float(roc_auc) - FROZEN_PAIR_MLP_GROUPED_ROC_AUC
                if roc_auc is not None
                else None
            ),
            "delta_average_precision_vs_frozen_pair_mlp": (
                float(pr_auc) - FROZEN_PAIR_MLP_GROUPED_PR_AUC
                if pr_auc is not None
                else None
            ),
        }

    valid_modes = [
        (mode, item)
        for mode, item in modes.items()
        if item.get("valid") and item.get("roc_auc") is not None
    ]
    best = None
    if valid_modes:
        mode, item = max(
            valid_modes,
            key=lambda pair: (
                float(pair[1]["roc_auc"]),
                float(pair[1]["average_precision"] or -np.inf),
            ),
        )
        best = {"mode": mode, **item}

    return {
        "baselines": {
            "kmer_tfidf_logreg_pair_text": {
                "grouped_roc_auc": KMER_GROUPED_ROC_AUC,
                "grouped_average_precision": KMER_GROUPED_PR_AUC,
            },
            "frozen_pretrained_pair_mlp": {
                "grouped_roc_auc": FROZEN_PAIR_MLP_GROUPED_ROC_AUC,
                "grouped_average_precision": FROZEN_PAIR_MLP_GROUPED_PR_AUC,
            },
        },
        "modes": modes,
        "best_mode": best,
        "beats_kmer_grouped_roc_auc": bool(
            best and best.get("roc_auc") is not None and best["roc_auc"] > KMER_GROUPED_ROC_AUC
        ),
        "beats_kmer_grouped_average_precision": bool(
            best
            and best.get("average_precision") is not None
            and best["average_precision"] > KMER_GROUPED_PR_AUC
        ),
        "beats_frozen_pair_mlp_grouped_roc_auc": bool(
            best
            and best.get("roc_auc") is not None
            and best["roc_auc"] > FROZEN_PAIR_MLP_GROUPED_ROC_AUC
        ),
        "beats_frozen_pair_mlp_grouped_average_precision": bool(
            best
            and best.get("average_precision") is not None
            and best["average_precision"] > FROZEN_PAIR_MLP_GROUPED_PR_AUC
        ),
    }


def metric_value(value: float | None) -> float:
    """Convert optional metrics to NaN for plotting."""
    return np.nan if value is None else float(value)


def save_training_curves(results: dict[str, Any]) -> None:
    """Save train loss, validation loss, and validation ROC-AUC curves."""
    records = []
    for mode, result in results.items():
        if not result.get("valid"):
            continue
        for item in result.get("history", []):
            records.append({"mode": mode, **item})
    if not records:
        return

    table = pd.DataFrame.from_records(records)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for mode, group in table.groupby("mode"):
        axes[0].plot(group["epoch"], group["train_loss"], label=mode)
        axes[1].plot(group["epoch"], group["validation_loss"], label=mode)
        axes[2].plot(group["epoch"], group["validation_roc_auc"], label=mode)
    axes[0].set_title("Train Loss")
    axes[1].set_title("Validation Loss")
    axes[2].set_title("Validation ROC-AUC")
    for axis in axes:
        axis.set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_ylabel("Loss")
    axes[2].set_ylabel("ROC-AUC")
    axes[2].set_ylim(0, 1)
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    TRAINING_CURVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(TRAINING_CURVE_PATH, dpi=200)
    plt.close(fig)


def save_comparison_figure(
    comparison: dict[str, Any],
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Save a grouped metric comparison figure."""
    baseline_value = KMER_GROUPED_ROC_AUC if metric == "roc_auc" else KMER_GROUPED_PR_AUC
    frozen_value = (
        FROZEN_PAIR_MLP_GROUPED_ROC_AUC
        if metric == "roc_auc"
        else FROZEN_PAIR_MLP_GROUPED_PR_AUC
    )
    records = [
        {"label": "k-mer logreg", "value": baseline_value},
        {"label": "frozen pair MLP", "value": frozen_value},
    ]
    for mode, result in comparison["modes"].items():
        if result.get("valid"):
            records.append({"label": mode, "value": metric_value(result.get(metric))})
    if len(records) <= 2:
        return

    labels = [record["label"] for record in records]
    values = [record["value"] for record in records]
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(records))
    bars = ax.bar(x, values, color=colors[: len(records)])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
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


def save_figures(results: dict[str, Any], comparison: dict[str, Any]) -> None:
    """Save all requested fine-tuning figures."""
    save_training_curves(results)
    save_comparison_figure(
        comparison,
        metric="roc_auc",
        title="Fine-Tuned Pretrained Model ROC-AUC Comparison",
        ylabel="Grouped ROC-AUC",
        output_path=ROC_AUC_FIGURE_PATH,
    )
    save_comparison_figure(
        comparison,
        metric="average_precision",
        title="Fine-Tuned Pretrained Model PR-AUC Comparison",
        ylabel="Grouped average precision / PR-AUC",
        output_path=PR_AUC_FIGURE_PATH,
    )


def format_metric(value: float | None) -> str:
    """Format optional metrics."""
    return "n/a" if value is None else f"{value:.4f}"


def format_counts(counts: dict[str, int]) -> str:
    """Format label counts compactly."""
    return f"0={counts['0']}, 1={counts['1']}"


def format_metrics_table(results: dict[str, Any]) -> list[str]:
    """Format fine-tuning metrics as Markdown rows."""
    lines = [
        (
            "| Mode | Valid | Reason | Epochs | Best epoch | Trainable params | "
            "Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC | "
            "PR-AUC | Confusion matrix |"
        ),
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for mode, result in results.items():
        if not result.get("valid"):
            trainable = result.get("trainability", {}).get("trainable_parameter_count", 0)
            lines.append(
                f"| {mode} | false | {result.get('reason', 'unknown')} | n/a | n/a | "
                f"{trainable} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
            )
            continue
        metrics = result["metrics"]
        lines.append(
            f"| {mode} | true | ok | {result['epochs_trained']} | "
            f"{result['best_epoch']} | "
            f"{result['trainability']['trainable_parameter_count']} | "
            f"{metrics['accuracy']:.4f} | {metrics['balanced_accuracy']:.4f} | "
            f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | "
            f"{metrics['f1']:.4f} | {format_metric(metrics['roc_auc'])} | "
            f"{format_metric(metrics['average_precision'])} | "
            f"{metrics['confusion_matrix']} |"
        )
    return lines


def format_comparison_table(comparison: dict[str, Any]) -> list[str]:
    """Format comparison against current baselines."""
    lines = [
        (
            "| Model | Grouped ROC-AUC | Delta ROC-AUC vs k-mer | "
            "Delta ROC-AUC vs frozen pair MLP | Grouped PR-AUC | "
            "Delta PR-AUC vs k-mer | Delta PR-AUC vs frozen pair MLP |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| k-mer TF-IDF + logistic regression | {KMER_GROUPED_ROC_AUC:.4f} | "
            f"0.0000 | n/a | {KMER_GROUPED_PR_AUC:.4f} | 0.0000 | n/a |"
        ),
        (
            f"| frozen pretrained pair MLP | {FROZEN_PAIR_MLP_GROUPED_ROC_AUC:.4f} | "
            f"{FROZEN_PAIR_MLP_GROUPED_ROC_AUC - KMER_GROUPED_ROC_AUC:.4f} | "
            f"0.0000 | {FROZEN_PAIR_MLP_GROUPED_PR_AUC:.4f} | "
            f"{FROZEN_PAIR_MLP_GROUPED_PR_AUC - KMER_GROUPED_PR_AUC:.4f} | "
            f"0.0000 |"
        ),
    ]
    for mode, result in comparison["modes"].items():
        if not result.get("valid"):
            continue
        lines.append(
            f"| fine-tune {mode} | {format_metric(result.get('roc_auc'))} | "
            f"{format_metric(result.get('delta_roc_auc_vs_kmer'))} | "
            f"{format_metric(result.get('delta_roc_auc_vs_frozen_pair_mlp'))} | "
            f"{format_metric(result.get('average_precision'))} | "
            f"{format_metric(result.get('delta_average_precision_vs_kmer'))} | "
            f"{format_metric(result.get('delta_average_precision_vs_frozen_pair_mlp'))} |"
        )
    return lines


def format_overfitting_table(results: dict[str, Any]) -> list[str]:
    """Format overfitting diagnostics."""
    lines = [
        (
            "| Mode | Evidence | Best val ROC-AUC | Last val ROC-AUC | "
            "Val ROC-AUC drop | Best val loss | Last val loss | Val loss increase |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, result in results.items():
        if not result.get("valid"):
            continue
        overfit = result["overfitting"]
        lines.append(
            f"| {mode} | {str(overfit['evidence']).lower()} | "
            f"{format_metric(overfit.get('best_validation_roc_auc'))} | "
            f"{format_metric(overfit.get('last_validation_roc_auc'))} | "
            f"{format_metric(overfit.get('validation_roc_auc_drop_from_best'))} | "
            f"{format_metric(overfit.get('best_validation_loss'))} | "
            f"{format_metric(overfit.get('last_validation_loss'))} | "
            f"{format_metric(overfit.get('validation_loss_increase_from_best'))} |"
        )
    return lines


def build_interpretation(comparison: dict[str, Any], results: dict[str, Any]) -> list[str]:
    """Build a concise interpretation of fine-tuning results."""
    best = comparison.get("best_mode")
    lines = ["## Interpretation", ""]
    if not best:
        lines.append("No fine-tuning mode completed successfully.")
        lines.append("")
        return lines

    lines.append(
        f"Best fine-tuned mode: `{best['mode']}` with grouped ROC-AUC "
        f"{best['roc_auc']:.4f} and PR-AUC {best['average_precision']:.4f}."
    )
    lines.append(
        "Fine-tuning improves over the frozen pretrained pair MLP by ROC-AUC: "
        f"{'yes' if comparison['beats_frozen_pair_mlp_grouped_roc_auc'] else 'no'}."
    )
    lines.append(
        "Fine-tuning improves over the frozen pretrained pair MLP by PR-AUC: "
        f"{'yes' if comparison['beats_frozen_pair_mlp_grouped_average_precision'] else 'no'}."
    )
    lines.append(
        "Fine-tuning beats the k-mer grouped ROC-AUC baseline: "
        f"{'yes' if comparison['beats_kmer_grouped_roc_auc'] else 'no'}."
    )
    lines.append(
        "Fine-tuning beats the k-mer grouped PR-AUC baseline: "
        f"{'yes' if comparison['beats_kmer_grouped_average_precision'] else 'no'}."
    )
    overfit_modes = [
        mode
        for mode, result in results.items()
        if result.get("valid") and result.get("overfitting", {}).get("evidence")
    ]
    lines.append(
        "Evidence of overfitting: "
        f"{', '.join(overfit_modes) if overfit_modes else 'none by the simple validation diagnostics'}."
    )
    lines.append("")
    return lines


def build_report(
    data: pd.DataFrame,
    tokenizer: Any,
    style: str,
    max_length: int | None,
    split_info: dict[str, Any],
    validation_info: dict[str, Any],
    results: dict[str, Any],
    comparison: dict[str, Any],
    device_name: str,
    best_checkpoint_path: str | None,
) -> str:
    """Build the Markdown fine-tuning report."""
    test_diag = split_diagnostics(
        data,
        split_info["train_idx"],
        split_info["test_idx"],
        split_info["groups"],
    )
    val_diag = split_diagnostics(
        data,
        validation_info["train_core_idx"],
        validation_info["val_idx"],
        split_info["groups"],
    )
    lines = [
        "# Pretrained Sequence Model Fine-Tuning",
        "",
        "This report fine-tunes a Hugging Face pretrained sequence model on existing",
        "heavy-light pair inputs only. No input sequences were created or altered.",
        "",
        "## Setup",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Model name | `{MODEL_NAME}` |",
        f"| Tokenizer class | `{tokenizer.__class__.__name__}` |",
        f"| Device | `{device_name}` |",
        f"| Tokenization style | `{style}` |",
        f"| Max sequence length | `{max_length}` |",
        f"| Batch size | `{BATCH_SIZE}` |",
        f"| Max epochs | `{MAX_EPOCHS}` |",
        f"| Dropout | `{DROP_OUT}` |",
        "",
        "## Data And Splits",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Rows | {len(data)} |",
        f"| Label 0 count | {label_counts(data['label'].to_numpy())['0']} |",
        f"| Label 1 count | {label_counts(data['label'].to_numpy())['1']} |",
        f"| Grouped train size | {test_diag['train_size']} |",
        f"| Grouped test size | {test_diag['test_size']} |",
        f"| Train groups | {test_diag['train_group_count']} |",
        f"| Test groups | {test_diag['test_group_count']} |",
        f"| Train/test group overlap | {test_diag['group_overlap_count']} |",
        f"| Inner validation method | {validation_info['method']} |",
        f"| Inner train size | {val_diag['train_size']} |",
        f"| Validation size | {val_diag['test_size']} |",
        f"| Inner train/validation group overlap | {val_diag['group_overlap_count']} |",
        "",
        "## Metrics",
        "",
    ]
    lines.extend(format_metrics_table(results))
    lines.extend(["", "## Baseline Comparison", ""])
    lines.extend(format_comparison_table(comparison))
    lines.extend(["", "## Overfitting Diagnostics", ""])
    lines.extend(format_overfitting_table(results))
    lines.extend([""])
    lines.extend(build_interpretation(comparison, results))
    lines.extend(
        [
            "## Artifacts",
            "",
            f"- `{METRICS_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{TRAINING_CURVE_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{ROC_AUC_FIGURE_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{PR_AUC_FIGURE_PATH.relative_to(PROJECT_ROOT)}`",
        ]
    )
    if best_checkpoint_path:
        lines.append(f"- `{best_checkpoint_path}`")
    lines.append("")
    return "\n".join(lines)


def unavailable_report(reason: str) -> str:
    """Build a clear unavailable report."""
    return "\n".join(
        [
            "# Pretrained Sequence Model Fine-Tuning",
            "",
            "status: `unavailable`",
            f"reason: `{reason}`",
            "",
            "No fine-tuning artifacts were created.",
            "",
        ]
    )


def write_unavailable(reason: str) -> None:
    """Persist unavailable status files."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "unavailable",
        "reason": reason,
        "model_name": MODEL_NAME,
        "input_path": str(INPUT_PATH.relative_to(PROJECT_ROOT)),
        "artifacts": {
            "report": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
            "metrics_json": str(METRICS_PATH.relative_to(PROJECT_ROOT)),
        },
    }
    REPORT_PATH.write_text(unavailable_report(reason), encoding="utf-8")
    METRICS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def select_best_checkpoint(
    checkpoints: dict[str, dict[str, Any]],
    comparison: dict[str, Any],
) -> tuple[str | None, str | None]:
    """Save the best fine-tuned checkpoint by validation-selected mode."""
    best = comparison.get("best_mode")
    if not best:
        return None, None
    mode = best["mode"]
    checkpoint = checkpoints.get(mode)
    if checkpoint is None:
        return mode, None
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, BEST_MODEL_PATH)
    return mode, str(BEST_MODEL_PATH.relative_to(PROJECT_ROOT))


def print_summary(results: dict[str, Any], comparison: dict[str, Any]) -> None:
    """Print compact terminal summary."""
    print("\nFine-tuning metrics")
    for line in format_metrics_table(results):
        print(line)
    print("\nBaseline comparison")
    for line in format_comparison_table(comparison):
        print(line)


def main() -> int:
    """Run the controlled pair-input fine-tuning benchmark."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        require_batch_size()
        set_seed(RANDOM_STATE)
        data = load_dataset()
        split_info = grouped_train_test_split(data)
        validation_info = inner_validation_split(
            data,
            split_info["train_idx"],
            split_info["groups"],
        )

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        probe_backbone = AutoModel.from_pretrained(MODEL_NAME)
        ensure_pad_token(tokenizer)
        max_length = max_length_from_model(tokenizer, probe_backbone)
        style, tokenization_attempts = choose_tokenization_style(
            tokenizer=tokenizer,
            heavy_examples=[
                split_pair_text(value)[0]
                for value in data["sequence_pair_text"].head(5).astype(str).tolist()
            ],
            pair_examples=data["sequence_pair_text"].head(5).astype(str).tolist(),
            max_length=max_length,
        )
        del probe_backbone
    except Exception as exc:
        write_unavailable(str(exc))
        print(f"fine-tuning unavailable: {exc}")
        return 1

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device_name}", flush=True)
    if device_name == "cpu":
        print(
            "Warning: running fine-tuning on CPU can be very slow for this model.",
            flush=True,
        )

    results: dict[str, Any] = {}
    checkpoints: dict[str, dict[str, Any]] = {}
    skipped_due_to_layers = False

    for mode in TRAINING_MODES:
        if skipped_due_to_layers and mode != "head_only":
            results[mode] = {
                "mode": mode,
                "valid": False,
                "reason": "transformer_layers_not_identified_safely",
            }
            continue

        print(f"Starting fine-tuning mode: {mode}", flush=True)
        try:
            result, checkpoint = train_one_mode(
                mode=mode,
                data=data,
                split_info=split_info,
                validation_info=validation_info,
                tokenizer=tokenizer,
                style=style,
                max_length=max_length,
                device_name=device_name,
            )
            results[mode] = result
            if checkpoint is not None:
                checkpoints[mode] = checkpoint
            if mode in {"last_1_layer", "last_2_layers"} and not result.get("valid"):
                skipped_due_to_layers = True
        except Exception as exc:
            results[mode] = {
                "mode": mode,
                "valid": False,
                "reason": str(exc),
            }
            print(f"Mode {mode} failed: {exc}", flush=True)

    comparison = build_comparison(results)
    best_mode, best_checkpoint_path = select_best_checkpoint(checkpoints, comparison)
    save_figures(results, comparison)

    payload = {
        "status": "available",
        "model_name": MODEL_NAME,
        "environment_variable": MODEL_ENV_VAR,
        "input_path": str(INPUT_PATH.relative_to(PROJECT_ROOT)),
        "row_count": int(len(data)),
        "label_counts": label_counts(data["label"].to_numpy()),
        "input_variant": "sequence_pair_text",
        "device": device_name,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "inner_validation_size": INNER_VALIDATION_SIZE,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "tokenization_style": style,
        "tokenization_attempts": tokenization_attempts,
        "max_sequence_length": max_length,
        "split": {
            "group_column": GROUP_COLUMN,
            "train_test": split_diagnostics(
                data,
                split_info["train_idx"],
                split_info["test_idx"],
                split_info["groups"],
            ),
            "validation": {
                **split_diagnostics(
                    data,
                    validation_info["train_core_idx"],
                    validation_info["val_idx"],
                    split_info["groups"],
                ),
                "method": validation_info["method"],
                "reason": validation_info["reason"],
            },
        },
        "results": results,
        "comparison": comparison,
        "best_mode": best_mode,
        "artifacts": {
            "report": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
            "metrics_json": str(METRICS_PATH.relative_to(PROJECT_ROOT)),
            "training_curves": str(TRAINING_CURVE_PATH.relative_to(PROJECT_ROOT)),
            "roc_auc_comparison": str(ROC_AUC_FIGURE_PATH.relative_to(PROJECT_ROOT)),
            "pr_auc_comparison": str(PR_AUC_FIGURE_PATH.relative_to(PROJECT_ROOT)),
            "best_model": best_checkpoint_path,
        },
    }
    METRICS_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        build_report(
            data=data,
            tokenizer=tokenizer,
            style=style,
            max_length=max_length,
            split_info=split_info,
            validation_info=validation_info,
            results=results,
            comparison=comparison,
            device_name=device_name,
            best_checkpoint_path=best_checkpoint_path,
        ),
        encoding="utf-8",
    )
    print_summary(results, comparison)
    return 0


if __name__ == "__main__":
    sys.exit(main())
