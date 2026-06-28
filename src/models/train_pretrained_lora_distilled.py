"""LoRA fine-tuning with k-mer teacher distillation for pair inputs.

This script uses only existing labeled rows from the neutral ML table. It does
not create, mutate, optimize, rank, or propose biological sequences.

Run from the project root:

    python src/models/train_pretrained_lora_distilled.py

Optionally choose a Hugging Face model:

    PRETRAINED_SEQUENCE_MODEL=Exscientia/IgBert \
        python src/models/train_pretrained_lora_distilled.py
"""

from __future__ import annotations

import copy
import gc
import json
import os
import random
import sys
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
import torch.nn.functional as F
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover - depends on sklearn version.
    StratifiedGroupKFold = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import train_pretrained_finetune as ft
from src.models.embed_with_pretrained_sequence_model import (
    choose_tokenization_style,
    ensure_pad_token,
    last_hidden_state,
    max_length_from_model,
    run_model_forward,
    split_pair_text,
    tokenize_pairs,
)


INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "neutral_sequence_classification_ml.csv"
KMER_TEACHER_PATH = PROJECT_ROOT / "models" / "kmer_logreg_pair_text.joblib"

REPORT_PATH = PROJECT_ROOT / "reports" / "pretrained_lora_distilled_report.md"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics" / "pretrained_lora_distilled_metrics.json"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
TRAINING_CURVE_PATH = FIGURE_DIR / "pretrained_lora_distilled_training_curves.png"
ROC_PR_FIGURE_PATH = FIGURE_DIR / "pretrained_lora_distilled_roc_pr_seed_summary.png"
MODEL_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODEL_DIR / "pretrained_lora_distilled_best.pt"

DEFAULT_MODEL_NAME = "Exscientia/IgBert"
MODEL_ENV_VAR = "PRETRAINED_SEQUENCE_MODEL"
MODEL_NAME = os.environ.get(MODEL_ENV_VAR, DEFAULT_MODEL_NAME)

TRAINING_SEEDS = [1, 7, 42]
STD_DDOF = 1
BATCH_SIZE = int(os.environ.get("PRETRAINED_LORA_BATCH_SIZE", "8"))
MAX_EPOCHS = 15
EARLY_STOPPING_PATIENCE = 4
MIN_DELTA = 1e-4
MAX_GRAD_NORM = 1.0
WARMUP_RATIO = 0.10

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.10
LORA_LEARNING_RATE = 1e-4
HEAD_LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.05
HEAD_DROPOUT = 0.4
HEAD_HIDDEN_SIZE = 256
SUPERVISED_LOSS_WEIGHT = 0.6
DISTILLATION_LOSS_WEIGHT = 0.4
LABEL_SMOOTHING_POSITIVE = 0.95
LABEL_SMOOTHING_NEGATIVE = 0.05
TEACHER_OOF_FOLDS = 5

KMER_GROUPED_ROC_AUC = 0.7810
KMER_GROUPED_PR_AUC = 0.8236
FROZEN_PAIR_MLP_GROUPED_ROC_AUC = 0.7541
FROZEN_PAIR_MLP_GROUPED_PR_AUC = 0.8078
DIRECT_FINETUNE_SEED_MEAN_ROC_AUC = 0.7443
DIRECT_FINETUNE_SEED_MEAN_PR_AUC = 0.8151
DIRECT_FINETUNE_METRICS_PATH = (
    PROJECT_ROOT / "reports" / "metrics" / "pretrained_finetune_seed_check_metrics.json"
)

COMMON_LORA_TARGET_NAMES = [
    "query",
    "key",
    "value",
    "dense",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "out_proj",
]


class PairDistillationDataset(Dataset):
    """Dataset of existing pair text, true labels, and teacher probabilities."""

    def __init__(
        self,
        pair_texts: list[str],
        labels: np.ndarray,
        teacher_probabilities: np.ndarray | None = None,
    ) -> None:
        self.pair_texts = pair_texts
        self.labels = labels.astype(np.float32)
        self.teacher_probabilities = None
        if teacher_probabilities is not None:
            self.teacher_probabilities = teacher_probabilities.astype(np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item: dict[str, Any] = {
            "pair_text": self.pair_texts[index],
            "label": np.float32(self.labels[index]),
        }
        if self.teacher_probabilities is not None:
            item["teacher_probability"] = np.float32(self.teacher_probabilities[index])
        return item


class LoraDistilledClassifier(nn.Module):
    """PEFT backbone plus CLS/mean/max pooled one-logit classifier."""

    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        include_cls: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.include_cls = include_cls
        pooled_parts = 3 if include_cls else 2
        classifier_input_size = hidden_size * pooled_parts
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_size),
            nn.Dropout(dropout),
            nn.Linear(classifier_input_size, HEAD_HIDDEN_SIZE),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(HEAD_HIDDEN_SIZE, 1),
        )

    @staticmethod
    def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        """Mean-pool token states over non-padding positions only."""
        if attention_mask is None:
            return hidden.mean(dim=1)
        mask = attention_mask.to(hidden.device).unsqueeze(-1).type_as(hidden)
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts

    @staticmethod
    def max_pool(hidden: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        """Max-pool token states while excluding padding tokens."""
        if attention_mask is None:
            return hidden.max(dim=1).values
        mask = attention_mask.to(hidden.device).unsqueeze(-1).bool()
        masked_hidden = hidden.masked_fill(~mask, torch.finfo(hidden.dtype).min)
        pooled = masked_hidden.max(dim=1).values
        return torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return one logit per existing pair input."""
        outputs = run_model_forward(self.backbone, batch)
        hidden = last_hidden_state(outputs)
        attention_mask = batch.get("attention_mask")
        pooled_parts = []
        if self.include_cls:
            pooled_parts.append(hidden[:, 0, :])
        pooled_parts.append(self.mean_pool(hidden, attention_mask))
        pooled_parts.append(self.max_pool(hidden, attention_mask))
        pooled = torch.cat(pooled_parts, dim=1)
        return self.classifier(pooled).squeeze(-1)


def set_seed(seed: int) -> None:
    """Set deterministic seeds for numpy, Python, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_batch_size() -> None:
    """Keep the LoRA fine-tuning batch size in the requested small-batch range."""
    if BATCH_SIZE not in {8, 16}:
        raise ValueError("PRETRAINED_LORA_BATCH_SIZE must be 8 or 16.")


def load_peft() -> dict[str, Any]:
    """Import PEFT lazily and fail clearly if unavailable."""
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError(
            "PEFT is required for LoRA fine-tuning but is not installed. "
            "Install dependencies with `pip install -r requirements.txt`. "
            "No normal fine-tuning fallback was run."
        ) from exc
    return {
        "LoraConfig": LoraConfig,
        "TaskType": TaskType,
        "get_peft_model": get_peft_model,
    }


def make_kmer_teacher_pipeline() -> Pipeline:
    """Construct the leakage-safe k-mer teacher estimator."""
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


def teacher_positive_scores(model: Any, values: pd.Series) -> np.ndarray:
    """Return teacher positive-class probabilities."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(values.astype(str))
        classes = list(getattr(model, "classes_", [0, 1]))
        positive_index = classes.index(1) if 1 in classes else 1
        return probabilities[:, positive_index].astype(np.float32)
    scores = model.decision_function(values.astype(str))
    return (1.0 / (1.0 + np.exp(-scores))).astype(np.float32)


def load_teacher_template() -> tuple[Any, dict[str, Any]]:
    """Load the saved k-mer model as a cloneable template when available."""
    if KMER_TEACHER_PATH.exists():
        try:
            saved = joblib.load(KMER_TEACHER_PATH)
            return saved, {
                "saved_model_available": True,
                "saved_model_path": str(KMER_TEACHER_PATH.relative_to(PROJECT_ROOT)),
                "template_source": "loaded_saved_model_cloned_for_oof_refits",
                "load_error": None,
            }
        except Exception as exc:
            return make_kmer_teacher_pipeline(), {
                "saved_model_available": False,
                "saved_model_path": str(KMER_TEACHER_PATH.relative_to(PROJECT_ROOT)),
                "template_source": "reconstructed_default_kmer_pipeline",
                "load_error": str(exc),
            }
    return make_kmer_teacher_pipeline(), {
        "saved_model_available": False,
        "saved_model_path": str(KMER_TEACHER_PATH.relative_to(PROJECT_ROOT)),
        "template_source": "reconstructed_default_kmer_pipeline",
        "load_error": "saved model path not found",
    }


def make_group_oof_splitter(group_count: int) -> tuple[Any, int, str]:
    """Create a grouped OOF splitter, stratified when sklearn supports it."""
    n_splits = max(2, min(TEACHER_OOF_FOLDS, group_count))
    if StratifiedGroupKFold is not None:
        return (
            StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=ft.RANDOM_STATE,
            ),
            n_splits,
            "StratifiedGroupKFold",
        )
    return GroupKFold(n_splits=n_splits), n_splits, "GroupKFold"


def build_teacher_probabilities(
    data: pd.DataFrame,
    split_info: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Create leakage-safe teacher probabilities for outer-training rows."""
    template, teacher_info = load_teacher_template()
    train_idx = np.asarray(split_info["train_idx"])
    groups = split_info["groups"].iloc[train_idx].reset_index(drop=True).astype(str)
    train_data = data.iloc[train_idx].reset_index(drop=True)
    y_train = train_data["label"].astype(int).to_numpy()
    x_train = train_data["sequence_pair_text"].astype(str)

    probabilities = np.full(len(data), np.nan, dtype=np.float32)
    unique_group_count = int(groups.nunique())

    try:
        splitter, n_splits, splitter_name = make_group_oof_splitter(unique_group_count)
        fold_records = []
        for fold_index, (fit_local, predict_local) in enumerate(
            splitter.split(train_data, y_train, groups=groups),
            start=1,
        ):
            fit_labels = y_train[np.asarray(fit_local)]
            if len(set(fit_labels.tolist())) != 2:
                raise ValueError(f"OOF fold {fold_index} fit split has one label class.")

            teacher = clone(template)
            teacher.fit(x_train.iloc[fit_local], fit_labels)
            global_predict_idx = train_idx[np.asarray(predict_local)]
            probabilities[global_predict_idx] = teacher_positive_scores(
                teacher,
                x_train.iloc[predict_local],
            )
            fold_group_set = set(groups.iloc[predict_local].astype(str))
            fit_group_set = set(groups.iloc[fit_local].astype(str))
            fold_records.append(
                {
                    "fold": int(fold_index),
                    "fit_size": int(len(fit_local)),
                    "predict_size": int(len(predict_local)),
                    "fit_label_counts": ft.label_counts(fit_labels),
                    "predict_label_counts": ft.label_counts(
                        y_train[np.asarray(predict_local)]
                    ),
                    "fit_group_count": int(len(fit_group_set)),
                    "predict_group_count": int(len(fold_group_set)),
                    "group_overlap_count": int(len(fit_group_set & fold_group_set)),
                }
            )

        if np.isnan(probabilities[train_idx]).any():
            missing = int(np.isnan(probabilities[train_idx]).sum())
            raise ValueError(f"OOF teacher left {missing} outer-training rows unset.")

        teacher_info.update(
            {
                "method": "group_aware_out_of_fold_on_outer_training_set",
                "out_of_fold": True,
                "splitter": splitter_name,
                "n_splits": int(n_splits),
                "final_test_used_for_teacher_fit": False,
                "limitation": None,
                "folds": fold_records,
            }
        )
        return probabilities, teacher_info
    except Exception as exc:
        teacher = clone(template)
        teacher.fit(x_train, y_train)
        probabilities[train_idx] = teacher_positive_scores(teacher, x_train)
        teacher_info.update(
            {
                "method": "outer_training_set_fit_in_sample_probabilities",
                "out_of_fold": False,
                "splitter": None,
                "n_splits": None,
                "final_test_used_for_teacher_fit": False,
                "limitation": (
                    "OOF teacher probability generation failed; teacher was fit only "
                    "on the outer training set, so final test rows were still not used. "
                    f"OOF error: {exc}"
                ),
                "folds": [],
            }
        )
        return probabilities, teacher_info


def terminal_module_name(module_name: str) -> str:
    """Return the terminal component of a dotted module name."""
    return module_name.rsplit(".", 1)[-1]


def find_lora_target_modules(backbone: nn.Module) -> dict[str, Any]:
    """Find common attention projection names present as linear modules."""
    found: dict[str, list[str]] = {}
    for module_name, module in backbone.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        terminal = terminal_module_name(module_name)
        if terminal in COMMON_LORA_TARGET_NAMES:
            found.setdefault(terminal, []).append(module_name)

    target_modules = [
        name for name in COMMON_LORA_TARGET_NAMES if name in found
    ]
    return {
        "target_modules": target_modules,
        "matched_module_count": int(sum(len(values) for values in found.values())),
        "matched_modules_by_target": found,
    }


def apply_lora(backbone: nn.Module, peft_api: dict[str, Any], targets: list[str]) -> nn.Module:
    """Freeze the base model and attach LoRA adapters."""
    if not targets:
        raise RuntimeError(
            "No common LoRA target modules were found in the pretrained backbone."
        )
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)

    config = peft_api["LoraConfig"](
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=targets,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=peft_api["TaskType"].FEATURE_EXTRACTION,
    )
    return peft_api["get_peft_model"](backbone, config)


def trainable_parameter_summary(model: nn.Module) -> dict[str, Any]:
    """Count trainable and total parameters."""
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return {
        "trainable_parameter_count": int(trainable),
        "total_parameter_count": int(total),
        "trainable_parameter_fraction": float(trainable / total) if total else 0.0,
    }


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Copy trainable parameters only for compact PEFT checkpointing."""
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    state = model.state_dict()
    return {
        name: state[name].detach().cpu().clone()
        for name in sorted(trainable_names)
        if name in state
    }


def restore_trainable_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Restore trainable parameters captured by trainable_state_dict."""
    current_state = model.state_dict()
    for name, tensor in state.items():
        current_state[name].copy_(tensor.to(current_state[name].device))
    model.load_state_dict(current_state)


def make_collate_fn(
    tokenizer: Any,
    style: str,
    max_length: int | None,
) -> Any:
    """Create a collate function that dynamically tokenizes pair text."""

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        texts = [str(item["pair_text"]) for item in batch]
        labels = torch.tensor([float(item["label"]) for item in batch], dtype=torch.float32)
        encoded = tokenize_pairs(tokenizer, texts, style, max_length)
        output: dict[str, Any] = {"encoded": encoded, "labels": labels}
        if "teacher_probability" in batch[0]:
            output["teacher_probabilities"] = torch.tensor(
                [float(item["teacher_probability"]) for item in batch],
                dtype=torch.float32,
            )
        return output

    return collate


def make_loader(
    data: pd.DataFrame,
    indices: np.ndarray,
    teacher_probabilities: np.ndarray | None,
    tokenizer: Any,
    style: str,
    max_length: int | None,
    shuffle: bool,
) -> DataLoader:
    """Build a dataloader for pair-text LoRA fine-tuning."""
    selected = data.iloc[indices]
    selected_teacher = (
        None if teacher_probabilities is None else teacher_probabilities[np.asarray(indices)]
    )
    dataset = PairDistillationDataset(
        selected["sequence_pair_text"].astype(str).tolist(),
        selected["label"].to_numpy(dtype=np.float32),
        selected_teacher,
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
    model: LoraDistilledClassifier,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[dict[str, Any], float, np.ndarray, np.ndarray]:
    """Evaluate with supervised validation/test loss and scalar metrics."""
    model.eval()
    losses = []
    labels = []
    scores = []
    with torch.no_grad():
        for batch in loader:
            encoded = move_encoded_to_device(batch["encoded"], device)
            y = batch["labels"].to(device)
            logits = model(encoded)
            loss = loss_fn(logits, y)
            losses.append(float(loss.detach().cpu().item()) * len(y))
            scores.append(torch.sigmoid(logits).detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

    y_true = np.concatenate(labels).astype(int)
    y_score = np.concatenate(scores)
    y_pred = (y_score >= 0.5).astype(int)
    return ft.metric_dict(y_true, y_pred, y_score), float(sum(losses) / len(y_true)), y_true, y_score


def smoothed_labels(labels: torch.Tensor) -> torch.Tensor:
    """Apply simple binary label smoothing."""
    return labels * (LABEL_SMOOTHING_POSITIVE - LABEL_SMOOTHING_NEGATIVE) + LABEL_SMOOTHING_NEGATIVE


def combined_training_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    teacher_probabilities: torch.Tensor,
    supervised_loss_fn: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return combined, supervised, and teacher-distillation losses."""
    supervised = supervised_loss_fn(logits, smoothed_labels(labels))
    distillation = F.binary_cross_entropy_with_logits(logits, teacher_probabilities)
    combined = SUPERVISED_LOSS_WEIGHT * supervised + DISTILLATION_LOSS_WEIGHT * distillation
    return combined, supervised, distillation


def build_optimizer_and_scheduler(
    model: LoraDistilledClassifier,
    train_loader: DataLoader,
) -> tuple[torch.optim.Optimizer, Any]:
    """Create AdamW with separate LoRA/head rates and linear warmup."""
    head_params = list(model.classifier.parameters())
    lora_params = [
        parameter
        for parameter in model.backbone.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": lora_params,
                "lr": LORA_LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
            },
            {
                "params": head_params,
                "lr": HEAD_LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
            },
        ]
    )
    total_steps = max(1, MAX_EPOCHS * len(train_loader))
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler


def train_one_seed(
    seed: int,
    data: pd.DataFrame,
    split_info: dict[str, Any],
    validation_info: dict[str, Any],
    teacher_probabilities: np.ndarray,
    tokenizer: Any,
    style: str,
    max_length: int | None,
    peft_api: dict[str, Any],
    device_name: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Train the LoRA-distilled student for one seed."""
    set_seed(seed)
    device = torch.device(device_name)
    backbone = AutoModel.from_pretrained(MODEL_NAME)
    hidden_size = int(getattr(backbone.config, "hidden_size"))
    lora_targets = find_lora_target_modules(backbone)
    peft_backbone = apply_lora(backbone, peft_api, lora_targets["target_modules"])
    include_cls = getattr(tokenizer, "cls_token_id", None) is not None
    model = LoraDistilledClassifier(
        backbone=peft_backbone,
        hidden_size=hidden_size,
        include_cls=include_cls,
        dropout=HEAD_DROPOUT,
    ).to(device)

    for parameter in model.classifier.parameters():
        parameter.requires_grad_(True)

    trainability = trainable_parameter_summary(model)
    train_core_idx = validation_info["train_core_idx"]
    val_idx = validation_info["val_idx"]
    test_idx = split_info["test_idx"]

    train_loader = make_loader(
        data,
        train_core_idx,
        teacher_probabilities,
        tokenizer,
        style,
        max_length,
        shuffle=True,
    )
    val_loader = make_loader(
        data,
        val_idx,
        None,
        tokenizer,
        style,
        max_length,
        shuffle=False,
    )
    test_loader = make_loader(
        data,
        test_idx,
        None,
        tokenizer,
        style,
        max_length,
        shuffle=False,
    )

    y_train = data.iloc[train_core_idx]["label"].to_numpy(dtype=np.float32)
    positives = float(np.sum(y_train == 1))
    negatives = float(np.sum(y_train == 0))
    pos_weight_value = negatives / positives if positives > 0 else 1.0
    supervised_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    )
    validation_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    )
    optimizer, scheduler = build_optimizer_and_scheduler(model, train_loader)

    best_state: dict[str, torch.Tensor] | None = None
    best_record: dict[str, Any] | None = None
    best_val_pr = -np.inf
    best_val_roc = -np.inf
    stale_epochs = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        combined_losses = []
        supervised_losses = []
        distillation_losses = []
        progress = tqdm(train_loader, desc=f"seed {seed} LoRA epoch {epoch}", leave=False)
        for batch in progress:
            encoded = move_encoded_to_device(batch["encoded"], device)
            labels = batch["labels"].to(device)
            teacher = batch["teacher_probabilities"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(encoded)
            loss, supervised_loss, distillation_loss = combined_training_loss(
                logits,
                labels,
                teacher,
                supervised_loss_fn,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                MAX_GRAD_NORM,
            )
            optimizer.step()
            scheduler.step()
            combined_losses.append(float(loss.detach().cpu().item()))
            supervised_losses.append(float(supervised_loss.detach().cpu().item()))
            distillation_losses.append(float(distillation_loss.detach().cpu().item()))

        val_metrics, val_loss, _, _ = evaluate_model(
            model,
            val_loader,
            validation_loss_fn,
            device,
        )
        current_val_pr = (
            -np.inf
            if val_metrics["average_precision"] is None
            else float(val_metrics["average_precision"])
        )
        current_val_roc = (
            -np.inf if val_metrics["roc_auc"] is None else float(val_metrics["roc_auc"])
        )
        record = {
            "epoch": int(epoch),
            "train_loss": float(np.mean(combined_losses)),
            "train_supervised_loss": float(np.mean(supervised_losses)),
            "train_distillation_loss": float(np.mean(distillation_losses)),
            "validation_loss": float(val_loss),
            "validation_roc_auc": None
            if current_val_roc == -np.inf
            else current_val_roc,
            "validation_average_precision": None
            if current_val_pr == -np.inf
            else current_val_pr,
            "validation_f1": val_metrics["f1"],
        }
        history.append(record)
        print(
            f"seed={seed} epoch {epoch}: train_loss={record['train_loss']:.4f} "
            f"val_loss={val_loss:.4f} val_pr_auc={current_val_pr:.4f} "
            f"val_roc_auc={current_val_roc:.4f}",
            flush=True,
        )

        improved_pr = current_val_pr > best_val_pr + MIN_DELTA
        tied_pr_better_roc = (
            abs(current_val_pr - best_val_pr) <= MIN_DELTA
            and current_val_roc > best_val_roc + MIN_DELTA
        )
        if improved_pr or tied_pr_better_roc:
            best_val_pr = current_val_pr
            best_val_roc = current_val_roc
            best_record = dict(record)
            best_state = trainable_state_dict(model)
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= EARLY_STOPPING_PATIENCE:
            break

    if best_state is not None:
        restore_trainable_state(model, best_state)

    test_metrics, test_loss, _, _ = evaluate_model(
        model,
        test_loader,
        validation_loss_fn,
        device,
    )
    overfit = ft.overfitting_summary(history)
    split_diag = ft.split_diagnostics(
        data,
        split_info["train_idx"],
        test_idx,
        split_info["groups"],
    )
    val_diag = ft.split_diagnostics(
        data,
        train_core_idx,
        val_idx,
        split_info["groups"],
    )

    checkpoint = {
        "model_name": MODEL_NAME,
        "seed": int(seed),
        "tokenization_style": style,
        "max_length": max_length,
        "hidden_size": hidden_size,
        "include_cls": bool(include_cls),
        "pooling": ["cls", "mean", "max"] if include_cls else ["mean", "max"],
        "head_hidden_size": HEAD_HIDDEN_SIZE,
        "head_dropout": HEAD_DROPOUT,
        "lora": {
            "r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "target_modules": lora_targets["target_modules"],
        },
        "trainable_state_dict": trainable_state_dict(model),
        "best_epoch": None if best_record is None else int(best_record["epoch"]),
        "best_validation_average_precision": None
        if best_record is None
        else best_record["validation_average_precision"],
        "test_metrics": test_metrics,
    }
    result = {
        "seed": int(seed),
        "valid": True,
        "reason": "ok",
        "model_name": MODEL_NAME,
        "device": device_name,
        "hidden_size": hidden_size,
        "include_cls_pool": bool(include_cls),
        "pooling": ["cls", "mean", "max"] if include_cls else ["mean", "max"],
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "epochs_trained": int(len(history)),
        "best_epoch": None if best_record is None else int(best_record["epoch"]),
        "best_epoch_train_loss": None
        if best_record is None
        else float(best_record["train_loss"]),
        "best_epoch_validation_loss": None
        if best_record is None
        else float(best_record["validation_loss"]),
        "best_epoch_validation_roc_auc": None
        if best_record is None
        else best_record["validation_roc_auc"],
        "best_epoch_validation_average_precision": None
        if best_record is None
        else best_record["validation_average_precision"],
        "pos_weight": float(pos_weight_value),
        "history": history,
        "split": split_diag,
        "validation_split": {
            **val_diag,
            "method": validation_info["method"],
            "reason": validation_info["reason"],
            "split_random_state": validation_info["split_random_state"],
        },
        "test_loss": float(test_loss),
        "metrics": {**split_diag, **test_metrics},
        "overfitting": overfit,
        "lora": {
            "r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            **lora_targets,
        },
        "trainability": trainability,
        "optimizer": {
            "type": "AdamW",
            "lora_learning_rate": LORA_LEARNING_RATE,
            "classifier_head_learning_rate": HEAD_LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "max_grad_norm": MAX_GRAD_NORM,
        },
        "loss": {
            "supervised_weight": SUPERVISED_LOSS_WEIGHT,
            "distillation_weight": DISTILLATION_LOSS_WEIGHT,
            "label_smoothing_positive": LABEL_SMOOTHING_POSITIVE,
            "label_smoothing_negative": LABEL_SMOOTHING_NEGATIVE,
        },
    }
    return result, checkpoint


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute mean/std across valid seed runs."""
    valid = [result for result in results if result.get("valid")]
    metrics = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "f1": "f1",
        "balanced_accuracy": "balanced_accuracy",
    }
    aggregate: dict[str, Any] = {
        "valid_seed_count": int(len(valid)),
        "std_ddof": STD_DDOF,
    }
    for output_name, metric_name in metrics.items():
        values = [
            float(result["metrics"][metric_name])
            for result in valid
            if result.get("metrics", {}).get(metric_name) is not None
        ]
        aggregate[output_name] = {
            "values": values,
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values, ddof=STD_DDOF)) if len(values) > 1 else 0.0,
            "min": float(np.min(values)) if values else None,
            "max": float(np.max(values)) if values else None,
        }
    return aggregate


def load_previous_direct_finetune_context() -> dict[str, Any]:
    """Load previous direct fine-tuning seed-check context if available."""
    context = {
        "roc_auc": DIRECT_FINETUNE_SEED_MEAN_ROC_AUC,
        "pr_auc": DIRECT_FINETUNE_SEED_MEAN_PR_AUC,
        "overfit_seed_count": None,
        "source": "hardcoded_current_fact",
    }
    if DIRECT_FINETUNE_METRICS_PATH.exists():
        try:
            payload = json.loads(DIRECT_FINETUNE_METRICS_PATH.read_text())
            seed_mean = payload.get("comparison", {}).get("seed_mean", {})
            context.update(
                {
                    "roc_auc": float(seed_mean.get("roc_auc", context["roc_auc"])),
                    "pr_auc": float(seed_mean.get("pr_auc", context["pr_auc"])),
                    "overfit_seed_count": payload.get("comparison", {}).get(
                        "overfit_seed_count"
                    ),
                    "source": str(DIRECT_FINETUNE_METRICS_PATH.relative_to(PROJECT_ROOT)),
                }
            )
        except Exception as exc:
            context["load_error"] = str(exc)
    return context


def build_comparison(
    aggregate: dict[str, Any],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare LoRA distillation against current grouped baselines."""
    direct = load_previous_direct_finetune_context()
    mean_roc = aggregate["roc_auc"]["mean"]
    mean_pr = aggregate["pr_auc"]["mean"]
    overfit_seed_count = int(
        sum(
            1
            for result in results
            if result.get("valid") and result.get("overfitting", {}).get("evidence")
        )
    )
    direct_overfit = direct.get("overfit_seed_count")
    return {
        "baselines": {
            "kmer_tfidf_logreg_pair_text": {
                "grouped_roc_auc": KMER_GROUPED_ROC_AUC,
                "grouped_pr_auc": KMER_GROUPED_PR_AUC,
            },
            "frozen_pretrained_pair_mlp": {
                "grouped_roc_auc": FROZEN_PAIR_MLP_GROUPED_ROC_AUC,
                "grouped_pr_auc": FROZEN_PAIR_MLP_GROUPED_PR_AUC,
            },
            "previous_direct_finetuning_seed_mean": direct,
        },
        "seed_mean": {
            "roc_auc": mean_roc,
            "pr_auc": mean_pr,
            "delta_roc_auc_vs_kmer": None
            if mean_roc is None
            else float(mean_roc - KMER_GROUPED_ROC_AUC),
            "delta_pr_auc_vs_kmer": None if mean_pr is None else float(mean_pr - KMER_GROUPED_PR_AUC),
            "delta_roc_auc_vs_frozen_pair_mlp": None
            if mean_roc is None
            else float(mean_roc - FROZEN_PAIR_MLP_GROUPED_ROC_AUC),
            "delta_pr_auc_vs_frozen_pair_mlp": None
            if mean_pr is None
            else float(mean_pr - FROZEN_PAIR_MLP_GROUPED_PR_AUC),
            "delta_roc_auc_vs_previous_direct_finetuning": None
            if mean_roc is None
            else float(mean_roc - direct["roc_auc"]),
            "delta_pr_auc_vs_previous_direct_finetuning": None
            if mean_pr is None
            else float(mean_pr - direct["pr_auc"]),
        },
        "beats_kmer_grouped_roc_auc": bool(mean_roc is not None and mean_roc > KMER_GROUPED_ROC_AUC),
        "beats_kmer_grouped_pr_auc": bool(mean_pr is not None and mean_pr > KMER_GROUPED_PR_AUC),
        "beats_frozen_pair_mlp_grouped_roc_auc": bool(
            mean_roc is not None and mean_roc > FROZEN_PAIR_MLP_GROUPED_ROC_AUC
        ),
        "beats_frozen_pair_mlp_grouped_pr_auc": bool(
            mean_pr is not None and mean_pr > FROZEN_PAIR_MLP_GROUPED_PR_AUC
        ),
        "beats_previous_direct_finetuning_roc_auc": bool(
            mean_roc is not None and mean_roc > direct["roc_auc"]
        ),
        "beats_previous_direct_finetuning_pr_auc": bool(
            mean_pr is not None and mean_pr > direct["pr_auc"]
        ),
        "overfit_seed_count": overfit_seed_count,
        "overfitting_reduced_vs_previous_direct_finetuning": (
            None if direct_overfit is None else bool(overfit_seed_count < int(direct_overfit))
        ),
    }


def select_best_checkpoint(
    checkpoints: list[dict[str, Any]],
) -> tuple[int | None, str | None]:
    """Save the best seed checkpoint by validation PR-AUC."""
    valid = [
        checkpoint
        for checkpoint in checkpoints
        if checkpoint.get("best_validation_average_precision") is not None
    ]
    if not valid:
        return None, None
    best = max(
        valid,
        key=lambda item: float(item["best_validation_average_precision"]),
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best, BEST_MODEL_PATH)
    return int(best["seed"]), str(BEST_MODEL_PATH.relative_to(PROJECT_ROOT))


def metric_or_na(value: Any) -> str:
    """Format optional numeric metrics."""
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def format_seed_results_table(results: list[dict[str, Any]]) -> list[str]:
    """Format seed-wise metrics as Markdown."""
    lines = [
        (
            "| Seed | Best epoch | Train loss | Val loss | Val ROC-AUC | Val PR-AUC | "
            "Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC | "
            "PR-AUC | Confusion matrix |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        if not result.get("valid"):
            lines.append(
                f"| {result.get('seed')} | n/a | n/a | n/a | n/a | n/a | "
                "n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
            )
            continue
        metrics = result["metrics"]
        lines.append(
            f"| {result['seed']} | {result['best_epoch']} | "
            f"{metric_or_na(result['best_epoch_train_loss'])} | "
            f"{metric_or_na(result['best_epoch_validation_loss'])} | "
            f"{metric_or_na(result['best_epoch_validation_roc_auc'])} | "
            f"{metric_or_na(result['best_epoch_validation_average_precision'])} | "
            f"{metrics['accuracy']:.4f} | {metrics['balanced_accuracy']:.4f} | "
            f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | "
            f"{metrics['f1']:.4f} | {metric_or_na(metrics['roc_auc'])} | "
            f"{metric_or_na(metrics['average_precision'])} | "
            f"{metrics['confusion_matrix']} |"
        )
    return lines


def format_aggregate_table(aggregate: dict[str, Any]) -> list[str]:
    """Format aggregate metrics as Markdown."""
    lines = ["| Metric | Mean | Std | Min | Max |", "|---|---:|---:|---:|---:|"]
    for key in ["roc_auc", "pr_auc", "f1", "balanced_accuracy"]:
        item = aggregate[key]
        lines.append(
            f"| {key} | {metric_or_na(item['mean'])} | {metric_or_na(item['std'])} | "
            f"{metric_or_na(item['min'])} | {metric_or_na(item['max'])} |"
        )
    return lines


def save_training_curves(results: list[dict[str, Any]]) -> None:
    """Save train/validation curves for each seed."""
    records = []
    for result in results:
        if not result.get("valid"):
            continue
        for item in result.get("history", []):
            records.append({"seed": result["seed"], **item})
    if not records:
        return
    table = pd.DataFrame.from_records(records)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    metrics = [
        ("train_loss", "Train combined loss", axes[0, 0]),
        ("validation_loss", "Validation supervised loss", axes[0, 1]),
        ("validation_average_precision", "Validation PR-AUC", axes[1, 0]),
        ("validation_roc_auc", "Validation ROC-AUC", axes[1, 1]),
    ]
    for seed, group in table.groupby("seed"):
        for column, title, axis in metrics:
            axis.plot(group["epoch"], group[column], marker="o", label=f"seed {seed}")
            axis.set_title(title)
            axis.set_xlabel("Epoch")
            axis.grid(alpha=0.25)
    axes[0, 0].set_ylabel("Loss")
    axes[0, 1].set_ylabel("Loss")
    axes[1, 0].set_ylabel("PR-AUC")
    axes[1, 1].set_ylabel("ROC-AUC")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].legend(fontsize=8)
    fig.tight_layout()
    TRAINING_CURVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(TRAINING_CURVE_PATH, dpi=200)
    plt.close(fig)


def save_seed_summary_figure(results: list[dict[str, Any]], aggregate: dict[str, Any]) -> None:
    """Save seed-level ROC/PR summary against baselines."""
    valid = [result for result in results if result.get("valid")]
    if not valid:
        return
    seeds = [str(result["seed"]) for result in valid]
    roc_values = [result["metrics"]["roc_auc"] for result in valid]
    pr_values = [result["metrics"]["average_precision"] for result in valid]
    x = np.arange(len(valid))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, roc_values, width, label="ROC-AUC", color="#4C78A8")
    ax.bar(x + width / 2, pr_values, width, label="PR-AUC", color="#54A24B")
    ax.axhline(KMER_GROUPED_ROC_AUC, color="#4C78A8", linestyle="--", linewidth=1.5, label="k-mer ROC-AUC")
    ax.axhline(KMER_GROUPED_PR_AUC, color="#54A24B", linestyle="--", linewidth=1.5, label="k-mer PR-AUC")
    ax.axhline(FROZEN_PAIR_MLP_GROUPED_ROC_AUC, color="#F58518", linestyle=":", linewidth=1.5, label="frozen ROC-AUC")
    ax.axhline(FROZEN_PAIR_MLP_GROUPED_PR_AUC, color="#E45756", linestyle=":", linewidth=1.5, label="frozen PR-AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Grouped held-out test metric")
    ax.set_title("LoRA-distilled pretrained model seed summary")
    ax.legend(fontsize=8, ncol=2)
    summary_text = (
        f"mean ROC={metric_or_na(aggregate['roc_auc']['mean'])}, "
        f"mean PR={metric_or_na(aggregate['pr_auc']['mean'])}"
    )
    ax.text(0.01, 0.02, summary_text, transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    ROC_PR_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ROC_PR_FIGURE_PATH, dpi=200)
    plt.close(fig)


def save_figures(results: list[dict[str, Any]], aggregate: dict[str, Any]) -> None:
    """Save requested figures."""
    save_training_curves(results)
    save_seed_summary_figure(results, aggregate)


def build_report(payload: dict[str, Any]) -> str:
    """Build the Markdown report."""
    if payload["status"] != "available":
        return "\n".join(
            [
                "# LoRA-Distilled Pretrained Sequence Model",
                "",
                "status: `unavailable`",
                f"reason: `{payload.get('reason', 'unknown')}`",
                "",
                "No normal fine-tuning fallback was run.",
                "",
            ]
        )

    aggregate = payload["aggregate"]
    comparison = payload["comparison"]
    teacher = payload["teacher"]
    split = payload["split"]
    validation = payload["validation_split"]
    trainability = payload["trainability"]
    lora = payload["lora"]
    lines = [
        "# LoRA-Distilled Pretrained Sequence Model",
        "",
        "This benchmark trains LoRA adapters plus a classifier head on existing",
        "heavy-light pair inputs only. No input sequences were created or altered.",
        "",
        "## Setup",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Model name | `{payload['model_name']}` |",
        f"| Tokenizer class | `{payload['tokenizer_class']}` |",
        f"| Device | `{payload['device']}` |",
        f"| Tokenization style | `{payload['tokenization_style']}` |",
        f"| Max sequence length | `{payload['max_sequence_length']}` |",
        f"| Batch size | `{payload['batch_size']}` |",
        f"| Max epochs | `{payload['max_epochs']}` |",
        f"| Seeds | `{payload['seeds']}` |",
        f"| Pooling | `{payload['pooling']}` |",
        "",
        "## LoRA",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| r | `{LORA_R}` |",
        f"| alpha | `{LORA_ALPHA}` |",
        f"| dropout | `{LORA_DROPOUT}` |",
        f"| Target modules found | `{lora['target_modules']}` |",
        f"| Matched module count | `{lora['matched_module_count']}` |",
        f"| Trainable parameters | `{trainability['trainable_parameter_count']}` |",
        f"| Total parameters | `{trainability['total_parameter_count']}` |",
        f"| Trainable fraction | `{trainability['trainable_parameter_fraction']:.6f}` |",
        "",
        "## Teacher",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Method | `{teacher['method']}` |",
        f"| Saved model available | `{teacher['saved_model_available']}` |",
        f"| Template source | `{teacher['template_source']}` |",
        f"| Out-of-fold probabilities | `{teacher['out_of_fold']}` |",
        f"| Final test used for teacher fit | `{teacher['final_test_used_for_teacher_fit']}` |",
        f"| Splitter | `{teacher.get('splitter')}` |",
        f"| Folds | `{teacher.get('n_splits')}` |",
    ]
    if teacher.get("limitation"):
        lines.append(f"| Limitation | `{teacher['limitation']}` |")
    lines.extend(
        [
            "",
            "## Data And Splits",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Rows | {payload['row_count']} |",
            f"| Label 0 count | {payload['label_counts']['0']} |",
            f"| Label 1 count | {payload['label_counts']['1']} |",
            f"| Grouped train size | {split['train_size']} |",
            f"| Grouped test size | {split['test_size']} |",
            f"| Train/test group overlap | {split['group_overlap_count']} |",
            f"| Inner validation method | {validation['method']} |",
            f"| Inner train size | {validation['train_size']} |",
            f"| Validation size | {validation['test_size']} |",
            f"| Inner train/validation group overlap | {validation['group_overlap_count']} |",
            "",
            "## Seed-Wise Results",
            "",
        ]
    )
    lines.extend(format_seed_results_table(payload["results"]))
    lines.extend(["", "## Mean +/- Std Across Seeds", ""])
    lines.extend(format_aggregate_table(aggregate))
    seed_mean = comparison["seed_mean"]
    lines.extend(
        [
            "",
            "## Baseline Comparison",
            "",
            "| Comparison | Result |",
            "|---|---|",
            f"| Mean ROC-AUC vs k-mer 0.7810 | `{metric_or_na(seed_mean['delta_roc_auc_vs_kmer'])}` |",
            f"| Mean PR-AUC vs k-mer 0.8236 | `{metric_or_na(seed_mean['delta_pr_auc_vs_kmer'])}` |",
            f"| Mean ROC-AUC vs frozen pair MLP 0.7541 | `{metric_or_na(seed_mean['delta_roc_auc_vs_frozen_pair_mlp'])}` |",
            f"| Mean PR-AUC vs frozen pair MLP 0.8078 | `{metric_or_na(seed_mean['delta_pr_auc_vs_frozen_pair_mlp'])}` |",
            f"| Mean ROC-AUC vs direct fine-tune 0.7443 | `{metric_or_na(seed_mean['delta_roc_auc_vs_previous_direct_finetuning'])}` |",
            f"| Mean PR-AUC vs direct fine-tune 0.8151 | `{metric_or_na(seed_mean['delta_pr_auc_vs_previous_direct_finetuning'])}` |",
            f"| Beats k-mer ROC-AUC | `{comparison['beats_kmer_grouped_roc_auc']}` |",
            f"| Beats k-mer PR-AUC | `{comparison['beats_kmer_grouped_pr_auc']}` |",
            f"| Overfitting reduced vs previous direct fine-tuning | `{comparison['overfitting_reduced_vs_previous_direct_finetuning']}` |",
            "",
            "## Conclusion",
            "",
            (
                "LoRA distillation beats the k-mer ROC-AUC baseline: "
                f"{'yes' if comparison['beats_kmer_grouped_roc_auc'] else 'no'}."
            ),
            (
                "LoRA distillation beats the k-mer PR-AUC baseline: "
                f"{'yes' if comparison['beats_kmer_grouped_pr_auc'] else 'no'}."
            ),
            (
                "Overfitting appears reduced compared with previous direct fine-tuning: "
                f"{comparison['overfitting_reduced_vs_previous_direct_finetuning']}."
            ),
            "",
            "## Artifacts",
            "",
            f"- `{METRICS_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{TRAINING_CURVE_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{ROC_PR_FIGURE_PATH.relative_to(PROJECT_ROOT)}`",
        ]
    )
    if payload.get("best_checkpoint_path"):
        lines.append(f"- `{payload['best_checkpoint_path']}`")
    lines.append("")
    return "\n".join(lines)


def write_unavailable(reason: str) -> None:
    """Persist a clear unavailable status."""
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
    METRICS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    REPORT_PATH.write_text(build_report(payload), encoding="utf-8")


def main() -> int:
    """Run the LoRA + k-mer teacher distillation benchmark."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        require_batch_size()
        peft_api = load_peft()
        set_seed(ft.RANDOM_STATE)
        data = ft.load_dataset()
        split_info = ft.grouped_train_test_split(data)
        validation_info = ft.inner_validation_split(
            data,
            split_info["train_idx"],
            split_info["groups"],
        )
        teacher_probabilities, teacher_info = build_teacher_probabilities(data, split_info)

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
        probe_targets = find_lora_target_modules(probe_backbone)
        if not probe_targets["target_modules"]:
            raise RuntimeError("No LoRA target modules found in the pretrained model.")
        include_cls = getattr(tokenizer, "cls_token_id", None) is not None
        del probe_backbone
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        write_unavailable(str(exc))
        print(f"LoRA distillation unavailable: {exc}", flush=True)
        return 1

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device_name}", flush=True)
    if device_name == "cpu":
        print(
            "Warning: running LoRA fine-tuning on CPU can be very slow.",
            flush=True,
        )

    results: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    first_trainability: dict[str, Any] | None = None
    first_lora: dict[str, Any] | None = None

    for seed in TRAINING_SEEDS:
        print(f"Starting LoRA distillation seed: {seed}", flush=True)
        try:
            result, checkpoint = train_one_seed(
                seed=seed,
                data=data,
                split_info=split_info,
                validation_info=validation_info,
                teacher_probabilities=teacher_probabilities,
                tokenizer=tokenizer,
                style=style,
                max_length=max_length,
                peft_api=peft_api,
                device_name=device_name,
            )
            results.append(result)
            if checkpoint is not None:
                checkpoints.append(checkpoint)
            if first_trainability is None:
                first_trainability = result["trainability"]
            if first_lora is None:
                first_lora = result["lora"]
        except Exception as exc:
            results.append(
                {
                    "seed": int(seed),
                    "valid": False,
                    "reason": str(exc),
                }
            )
            print(f"Seed {seed} failed: {exc}", flush=True)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    aggregate = aggregate_results(results)
    comparison = build_comparison(aggregate, results)
    best_seed, best_checkpoint_path = select_best_checkpoint(checkpoints)
    split_diag = ft.split_diagnostics(
        data,
        split_info["train_idx"],
        split_info["test_idx"],
        split_info["groups"],
    )
    val_diag = ft.split_diagnostics(
        data,
        validation_info["train_core_idx"],
        validation_info["val_idx"],
        split_info["groups"],
    )
    validation_summary = {
        **val_diag,
        "method": validation_info["method"],
        "reason": validation_info["reason"],
        "split_random_state": validation_info["split_random_state"],
    }
    payload = {
        "status": "available" if aggregate["valid_seed_count"] else "unavailable",
        "model_name": MODEL_NAME,
        "environment_variable": MODEL_ENV_VAR,
        "input_path": str(INPUT_PATH.relative_to(PROJECT_ROOT)),
        "row_count": int(len(data)),
        "label_counts": ft.label_counts(data["label"].to_numpy()),
        "device": device_name,
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenization_style": style,
        "tokenization_attempts": tokenization_attempts,
        "max_sequence_length": max_length,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "seeds": TRAINING_SEEDS,
        "pooling": ["cls", "mean", "max"] if include_cls else ["mean", "max"],
        "teacher": teacher_info,
        "split": split_diag,
        "validation_split": validation_summary,
        "lora": first_lora or probe_targets,
        "trainability": first_trainability
        or {
            "trainable_parameter_count": 0,
            "total_parameter_count": 0,
            "trainable_parameter_fraction": 0.0,
        },
        "results": results,
        "aggregate": aggregate,
        "comparison": comparison,
        "best_checkpoint_seed": best_seed,
        "best_checkpoint_path": best_checkpoint_path,
        "artifacts": {
            "report": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
            "metrics_json": str(METRICS_PATH.relative_to(PROJECT_ROOT)),
            "training_curves": str(TRAINING_CURVE_PATH.relative_to(PROJECT_ROOT)),
            "roc_pr_seed_summary": str(ROC_PR_FIGURE_PATH.relative_to(PROJECT_ROOT)),
            "best_checkpoint": best_checkpoint_path,
        },
    }

    METRICS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    REPORT_PATH.write_text(build_report(payload), encoding="utf-8")
    save_figures(results, aggregate)

    print("\nLoRA-distilled seed metrics")
    for line in format_seed_results_table(results):
        print(line)
    print("\nAggregate")
    for line in format_aggregate_table(aggregate):
        print(line)
    print(f"\nBest checkpoint seed: {best_seed}")
    return 0 if aggregate["valid_seed_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
