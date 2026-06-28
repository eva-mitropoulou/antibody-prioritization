"""Seed robustness check for the best pretrained fine-tuning configuration.

This script uses only existing labeled rows from the neutral ML table. It does
not create, mutate, optimize, rank, or propose biological sequences.

Run from the project root:

    python src/models/train_pretrained_finetune_seed_check.py

Optionally choose a Hugging Face model:

    PRETRAINED_SEQUENCE_MODEL=Exscientia/IgBert \
        python src/models/train_pretrained_finetune_seed_check.py
"""

from __future__ import annotations

import gc
import json
import os
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
from torch import nn
from transformers import AutoModel, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import train_pretrained_finetune as ft
from src.models.embed_with_pretrained_sequence_model import (
    ensure_pad_token,
    max_length_from_model,
    split_pair_text,
    choose_tokenization_style,
)


REPORT_PATH = PROJECT_ROOT / "reports" / "pretrained_finetune_seed_check_report.md"
METRICS_PATH = (
    PROJECT_ROOT / "reports" / "metrics" / "pretrained_finetune_seed_check_metrics.json"
)
FIGURE_PATH = (
    PROJECT_ROOT / "reports" / "figures" / "pretrained_finetune_seed_check_roc_pr.png"
)

MODE = "last_1_layer"
TRAINING_SEEDS = [1, 7, 42, 123, 2026]
STD_DDOF = 1


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Copy trainable parameters only; frozen backbone weights do not change."""
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


def train_one_seed(
    seed: int,
    data: pd.DataFrame,
    split_info: dict[str, Any],
    validation_info: dict[str, Any],
    tokenizer: Any,
    style: str,
    max_length: int | None,
    device_name: str,
) -> dict[str, Any]:
    """Train last_1_layer for one seed and return validation-selected metrics."""
    ft.set_seed(seed)
    device = torch.device(device_name)

    backbone = AutoModel.from_pretrained(ft.MODEL_NAME)
    backbone.to(device)
    hidden_size = int(getattr(backbone.config, "hidden_size"))
    layer_info = ft.identify_transformer_layers(backbone)
    model = ft.SequenceFineTuner(
        backbone=backbone,
        hidden_size=hidden_size,
        dropout=ft.DROP_OUT,
        freeze_backbone_forward=False,
    ).to(device)

    trainability = ft.configure_trainable_parameters(model, MODE, layer_info)
    result_base: dict[str, Any] = {
        "seed": int(seed),
        "mode": MODE,
        "valid": not trainability["skipped"],
        "reason": trainability["skip_reason"] or "ok",
        "layer_identification": {
            key: value for key, value in layer_info.items() if key != "layers"
        },
        "trainability": trainability,
    }
    if trainability["skipped"]:
        return result_base

    train_core_idx = validation_info["train_core_idx"]
    val_idx = validation_info["val_idx"]
    test_idx = split_info["test_idx"]

    train_loader = ft.make_loader(
        data, train_core_idx, tokenizer, style, max_length, shuffle=True
    )
    val_loader = ft.make_loader(data, val_idx, tokenizer, style, max_length, shuffle=False)
    test_loader = ft.make_loader(
        data, test_idx, tokenizer, style, max_length, shuffle=False
    )

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
        for parameter in model.backbone.parameters()
        if parameter.requires_grad and id(parameter) not in head_param_ids
    ]
    optimizer_groups = [
        {
            "params": head_params,
            "lr": ft.HEAD_LEARNING_RATE,
            "weight_decay": ft.WEIGHT_DECAY,
        }
    ]
    if backbone_params:
        optimizer_groups.append(
            {
                "params": backbone_params,
                "lr": ft.BACKBONE_LEARNING_RATE,
                "weight_decay": ft.WEIGHT_DECAY,
            }
        )
    optimizer = torch.optim.AdamW(optimizer_groups)

    best_state: dict[str, torch.Tensor] | None = None
    best_record: dict[str, Any] | None = None
    best_val_auc = -np.inf
    best_val_loss = np.inf
    stale_epochs = 0
    history = []

    for epoch in range(1, ft.MAX_EPOCHS + 1):
        model.train()
        batch_losses = []
        for batch in train_loader:
            encoded = ft.move_encoded_to_device(batch["encoded"], device)
            y = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(encoded)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                ft.MAX_GRAD_NORM,
            )
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))

        val_metrics, val_loss, _, _ = ft.evaluate_model(model, val_loader, loss_fn, device)
        current_val_auc = (
            -np.inf if val_metrics["roc_auc"] is None else float(val_metrics["roc_auc"])
        )
        current_val_loss = float("inf") if val_loss is None else float(val_loss)
        train_loss = float(np.mean(batch_losses))
        record = {
            "epoch": int(epoch),
            "train_loss": train_loss,
            "validation_loss": None if val_loss is None else float(val_loss),
            "validation_roc_auc": None
            if current_val_auc == -np.inf
            else current_val_auc,
            "validation_average_precision": val_metrics["average_precision"],
            "validation_f1": val_metrics["f1"],
        }
        history.append(record)
        print(
            f"seed={seed} {MODE} epoch {epoch}: "
            f"train_loss={train_loss:.4f} "
            f"val_loss={current_val_loss:.4f} "
            f"val_roc_auc={current_val_auc:.4f}",
            flush=True,
        )

        improved_auc = current_val_auc > best_val_auc + ft.MIN_DELTA
        improved_loss = current_val_loss < best_val_loss - ft.MIN_DELTA
        if improved_auc or (best_val_auc == -np.inf and improved_loss):
            best_val_auc = current_val_auc
            best_val_loss = current_val_loss
            best_record = dict(record)
            best_state = trainable_state_dict(model)
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= ft.EARLY_STOPPING_PATIENCE:
            break

    if best_state is not None:
        restore_trainable_state(model, best_state)

    test_metrics, test_loss, _, _ = ft.evaluate_model(model, test_loader, loss_fn, device)
    overfit = ft.overfitting_summary(history)

    result = {
        **result_base,
        "device": device_name,
        "hidden_size": hidden_size,
        "batch_size": ft.BATCH_SIZE,
        "max_epochs": ft.MAX_EPOCHS,
        "epochs_trained": int(len(history)),
        "best_epoch": None if best_record is None else int(best_record["epoch"]),
        "best_epoch_train_loss": None
        if best_record is None
        else float(best_record["train_loss"]),
        "best_epoch_validation_loss": None
        if best_record is None
        else best_record["validation_loss"],
        "best_epoch_validation_roc_auc": None
        if best_record is None
        else best_record["validation_roc_auc"],
        "history": history,
        "test_loss": test_loss,
        "metrics": test_metrics,
        "overfitting": overfit,
        "pos_weight": float(pos_weight_value),
    }

    del model, backbone, optimizer, loss_fn
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def aggregate_seed_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute mean/std across valid seed runs."""
    metrics = ["roc_auc", "average_precision", "f1", "balanced_accuracy"]
    valid_results = [result for result in results if result.get("valid")]
    aggregate: dict[str, Any] = {
        "valid_seed_count": int(len(valid_results)),
        "std_ddof": STD_DDOF,
    }
    for metric in metrics:
        values = [
            float(result["metrics"][metric])
            for result in valid_results
            if result.get("metrics", {}).get(metric) is not None
        ]
        metric_key = "pr_auc" if metric == "average_precision" else metric
        aggregate[metric_key] = {
            "values": values,
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values, ddof=STD_DDOF)) if len(values) > 1 else 0.0,
            "min": float(np.min(values)) if values else None,
            "max": float(np.max(values)) if values else None,
        }
    return aggregate


def build_comparison(aggregate: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare seed-averaged fine-tuning to current baselines."""
    roc_mean = aggregate["roc_auc"]["mean"]
    pr_mean = aggregate["pr_auc"]["mean"]
    valid_results = [result for result in results if result.get("valid")]
    roc_values = [
        float(result["metrics"]["roc_auc"])
        for result in valid_results
        if result.get("metrics", {}).get("roc_auc") is not None
    ]
    pr_values = [
        float(result["metrics"]["average_precision"])
        for result in valid_results
        if result.get("metrics", {}).get("average_precision") is not None
    ]
    f1_std = aggregate["f1"]["std"]
    balanced_std = aggregate["balanced_accuracy"]["std"]
    roc_std = aggregate["roc_auc"]["std"]
    pr_std = aggregate["pr_auc"]["std"]

    return {
        "baselines": {
            "kmer_tfidf_logreg_pair_text": {
                "grouped_roc_auc": ft.KMER_GROUPED_ROC_AUC,
                "grouped_pr_auc": ft.KMER_GROUPED_PR_AUC,
            },
            "frozen_pretrained_pair_mlp": {
                "grouped_roc_auc": ft.FROZEN_PAIR_MLP_GROUPED_ROC_AUC,
                "grouped_pr_auc": ft.FROZEN_PAIR_MLP_GROUPED_PR_AUC,
            },
        },
        "seed_mean": {
            "roc_auc": roc_mean,
            "pr_auc": pr_mean,
            "delta_roc_auc_vs_kmer": None
            if roc_mean is None
            else float(roc_mean - ft.KMER_GROUPED_ROC_AUC),
            "delta_pr_auc_vs_kmer": None
            if pr_mean is None
            else float(pr_mean - ft.KMER_GROUPED_PR_AUC),
            "delta_roc_auc_vs_frozen_pair_mlp": None
            if roc_mean is None
            else float(roc_mean - ft.FROZEN_PAIR_MLP_GROUPED_ROC_AUC),
            "delta_pr_auc_vs_frozen_pair_mlp": None
            if pr_mean is None
            else float(pr_mean - ft.FROZEN_PAIR_MLP_GROUPED_PR_AUC),
        },
        "all_seeds_beat_frozen_roc_auc": bool(
            roc_values
            and all(value > ft.FROZEN_PAIR_MLP_GROUPED_ROC_AUC for value in roc_values)
        ),
        "all_seeds_beat_frozen_pr_auc": bool(
            pr_values
            and all(value > ft.FROZEN_PAIR_MLP_GROUPED_PR_AUC for value in pr_values)
        ),
        "all_seeds_beat_kmer_pr_auc": bool(
            pr_values and all(value > ft.KMER_GROUPED_PR_AUC for value in pr_values)
        ),
        "all_seeds_beat_kmer_roc_auc": bool(
            roc_values and all(value > ft.KMER_GROUPED_ROC_AUC for value in roc_values)
        ),
        "stability_rule": "stable if ROC-AUC std <= 0.02 and PR-AUC std <= 0.02",
        "stable_by_rule": bool(
            roc_std is not None
            and pr_std is not None
            and roc_std <= 0.02
            and pr_std <= 0.02
        ),
        "secondary_variability": {
            "f1_std": f1_std,
            "balanced_accuracy_std": balanced_std,
        },
        "overfit_seed_count": int(
            sum(
                1
                for result in valid_results
                if result.get("overfitting", {}).get("evidence")
            )
        ),
    }


def format_metric(value: float | None) -> str:
    """Format optional numeric metrics for Markdown."""
    return "n/a" if value is None else f"{value:.4f}"


def save_figure(results: list[dict[str, Any]], aggregate: dict[str, Any]) -> None:
    """Save seed-wise ROC-AUC and PR-AUC with baseline reference lines."""
    valid_results = [result for result in results if result.get("valid")]
    if not valid_results:
        return

    seeds = [int(result["seed"]) for result in valid_results]
    roc_values = [float(result["metrics"]["roc_auc"]) for result in valid_results]
    pr_values = [float(result["metrics"]["average_precision"]) for result in valid_results]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
    plot_specs = [
        (
            axes[0],
            roc_values,
            "Grouped ROC-AUC",
            ft.KMER_GROUPED_ROC_AUC,
            ft.FROZEN_PAIR_MLP_GROUPED_ROC_AUC,
            aggregate["roc_auc"]["mean"],
        ),
        (
            axes[1],
            pr_values,
            "Grouped PR-AUC",
            ft.KMER_GROUPED_PR_AUC,
            ft.FROZEN_PAIR_MLP_GROUPED_PR_AUC,
            aggregate["pr_auc"]["mean"],
        ),
    ]
    for axis, values, ylabel, kmer_value, frozen_value, mean_value in plot_specs:
        axis.plot(seeds, values, marker="o", color="#4C78A8", label="fine-tune seed")
        axis.axhline(kmer_value, color="#E45756", linestyle="--", linewidth=1.5, label="k-mer")
        axis.axhline(
            frozen_value,
            color="#F58518",
            linestyle=":",
            linewidth=1.8,
            label="frozen pair MLP",
        )
        if mean_value is not None:
            axis.axhline(
                mean_value,
                color="#54A24B",
                linestyle="-.",
                linewidth=1.5,
                label="seed mean",
            )
        axis.set_title(ylabel)
        axis.set_xlabel("Training seed")
        axis.set_ylabel(ylabel)
        axis.set_xticks(seeds)
        axis.set_ylim(0.65, 0.88)
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=200)
    plt.close(fig)


def per_seed_table(results: list[dict[str, Any]]) -> list[str]:
    """Format the required per-seed metrics."""
    lines = [
        (
            "| Seed | Best epoch | Train loss at best | Val loss at best | "
            "Val ROC-AUC at best | Accuracy | Balanced accuracy | Precision | "
            "Recall | F1 | ROC-AUC | PR-AUC | Confusion matrix |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        if not result.get("valid"):
            lines.append(
                f"| {result['seed']} | n/a | n/a | n/a | n/a | n/a | n/a | "
                f"n/a | n/a | n/a | n/a | n/a | {result.get('reason', 'failed')} |"
            )
            continue
        metrics = result["metrics"]
        lines.append(
            f"| {result['seed']} | {result['best_epoch']} | "
            f"{format_metric(result['best_epoch_train_loss'])} | "
            f"{format_metric(result['best_epoch_validation_loss'])} | "
            f"{format_metric(result['best_epoch_validation_roc_auc'])} | "
            f"{metrics['accuracy']:.4f} | {metrics['balanced_accuracy']:.4f} | "
            f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | "
            f"{metrics['f1']:.4f} | {format_metric(metrics['roc_auc'])} | "
            f"{format_metric(metrics['average_precision'])} | "
            f"{metrics['confusion_matrix']} |"
        )
    return lines


def aggregate_table(aggregate: dict[str, Any]) -> list[str]:
    """Format aggregate mean/std metrics."""
    rows = [
        ("ROC-AUC", aggregate["roc_auc"]),
        ("PR-AUC", aggregate["pr_auc"]),
        ("F1", aggregate["f1"]),
        ("Balanced accuracy", aggregate["balanced_accuracy"]),
    ]
    lines = [
        "| Metric | Mean | Std | Min | Max |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, stats in rows:
        lines.append(
            f"| {label} | {format_metric(stats['mean'])} | "
            f"{format_metric(stats['std'])} | {format_metric(stats['min'])} | "
            f"{format_metric(stats['max'])} |"
        )
    return lines


def comparison_table(comparison: dict[str, Any]) -> list[str]:
    """Format seed-mean comparisons against baselines."""
    seed_mean = comparison["seed_mean"]
    return [
        (
            "| Model | ROC-AUC | Delta ROC-AUC vs k-mer | "
            "Delta ROC-AUC vs frozen pair MLP | PR-AUC | "
            "Delta PR-AUC vs k-mer | Delta PR-AUC vs frozen pair MLP |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| k-mer TF-IDF + logistic regression | {ft.KMER_GROUPED_ROC_AUC:.4f} | "
            f"0.0000 | n/a | {ft.KMER_GROUPED_PR_AUC:.4f} | 0.0000 | n/a |"
        ),
        (
            f"| frozen pretrained pair MLP | {ft.FROZEN_PAIR_MLP_GROUPED_ROC_AUC:.4f} | "
            f"{ft.FROZEN_PAIR_MLP_GROUPED_ROC_AUC - ft.KMER_GROUPED_ROC_AUC:.4f} | "
            f"0.0000 | {ft.FROZEN_PAIR_MLP_GROUPED_PR_AUC:.4f} | "
            f"{ft.FROZEN_PAIR_MLP_GROUPED_PR_AUC - ft.KMER_GROUPED_PR_AUC:.4f} | "
            f"0.0000 |"
        ),
        (
            f"| fine-tune last_1_layer seed mean | "
            f"{format_metric(seed_mean['roc_auc'])} | "
            f"{format_metric(seed_mean['delta_roc_auc_vs_kmer'])} | "
            f"{format_metric(seed_mean['delta_roc_auc_vs_frozen_pair_mlp'])} | "
            f"{format_metric(seed_mean['pr_auc'])} | "
            f"{format_metric(seed_mean['delta_pr_auc_vs_kmer'])} | "
            f"{format_metric(seed_mean['delta_pr_auc_vs_frozen_pair_mlp'])} |"
        ),
    ]


def conclusion_lines(comparison: dict[str, Any], aggregate: dict[str, Any]) -> list[str]:
    """Build the requested conclusion from observed seed variability."""
    overfit_count = comparison["overfit_seed_count"]
    valid_count = aggregate["valid_seed_count"]
    return [
        "## Conclusion",
        "",
        (
            "Stability: "
            f"{'stable' if comparison['stable_by_rule'] else 'not clearly stable'} "
            f"by the predefined std rule "
            f"(ROC-AUC std {format_metric(aggregate['roc_auc']['std'])}, "
            f"PR-AUC std {format_metric(aggregate['pr_auc']['std'])})."
        ),
        (
            "Reliably beats frozen embeddings: "
            f"{'yes' if comparison['all_seeds_beat_frozen_roc_auc'] and comparison['all_seeds_beat_frozen_pr_auc'] else 'no'} "
            "(requires every valid seed to beat frozen pair MLP on both ROC-AUC and PR-AUC)."
        ),
        (
            "Reliably beats k-mer on PR-AUC: "
            f"{'yes' if comparison['all_seeds_beat_kmer_pr_auc'] else 'no'} "
            "(requires every valid seed to beat the k-mer PR-AUC baseline)."
        ),
        (
            "Reliably beats k-mer on ROC-AUC: "
            f"{'yes' if comparison['all_seeds_beat_kmer_roc_auc'] else 'no'}."
        ),
        (
            "Overfitting: "
            f"{'present' if overfit_count else 'not detected by the simple validation diagnostics'} "
            f"in {overfit_count}/{valid_count} valid seed runs."
        ),
        "",
    ]


def build_report(
    data: pd.DataFrame,
    tokenizer: Any,
    style: str,
    max_length: int | None,
    split_info: dict[str, Any],
    validation_info: dict[str, Any],
    results: list[dict[str, Any]],
    aggregate: dict[str, Any],
    comparison: dict[str, Any],
    device_name: str,
) -> str:
    """Build the Markdown seed-check report."""
    test_diag = ft.split_diagnostics(
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
    lines = [
        "# Pretrained Fine-Tuning Seed Check",
        "",
        "This report repeats only the best fine-tuning configuration,",
        "`last_1_layer`, on existing heavy-light pair inputs. No input sequences",
        "were created or altered.",
        "",
        "## Setup",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Model name | `{ft.MODEL_NAME}` |",
        f"| Tokenizer class | `{tokenizer.__class__.__name__}` |",
        f"| Device | `{device_name}` |",
        f"| Mode | `{MODE}` |",
        f"| Training seeds | `{', '.join(str(seed) for seed in TRAINING_SEEDS)}` |",
        f"| Batch size | `{ft.BATCH_SIZE}` |",
        f"| Max epochs | `{ft.MAX_EPOCHS}` |",
        f"| Early stopping patience | `{ft.EARLY_STOPPING_PATIENCE}` |",
        f"| Tokenization style | `{style}` |",
        f"| Max sequence length | `{max_length}` |",
        "",
        "## Data And Splits",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Rows | {len(data)} |",
        f"| Label 0 count | {ft.label_counts(data['label'].to_numpy())['0']} |",
        f"| Label 1 count | {ft.label_counts(data['label'].to_numpy())['1']} |",
        f"| Grouped train size | {test_diag['train_size']} |",
        f"| Grouped test size | {test_diag['test_size']} |",
        f"| Train/test split random state | {split_info['split_random_state']} |",
        f"| Train groups | {test_diag['train_group_count']} |",
        f"| Test groups | {test_diag['test_group_count']} |",
        f"| Train/test group overlap | {test_diag['group_overlap_count']} |",
        f"| Inner validation method | {validation_info['method']} |",
        f"| Inner validation split random state | {validation_info['split_random_state']} |",
        f"| Inner train size | {val_diag['train_size']} |",
        f"| Validation size | {val_diag['test_size']} |",
        f"| Inner train/validation group overlap | {val_diag['group_overlap_count']} |",
        "",
        "The outer grouped train/test split uses the same helper and random state",
        "as `train_pretrained_finetune.py`, so the test split is held fixed across",
        "all seed runs.",
        "",
        "## Per-Seed Metrics",
        "",
    ]
    lines.extend(per_seed_table(results))
    lines.extend(["", "## Aggregate Metrics", ""])
    lines.extend(aggregate_table(aggregate))
    lines.extend(["", "## Baseline Comparison", ""])
    lines.extend(comparison_table(comparison))
    lines.extend([""])
    lines.extend(conclusion_lines(comparison, aggregate))
    lines.extend(
        [
            "## Artifacts",
            "",
            f"- `{METRICS_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{FIGURE_PATH.relative_to(PROJECT_ROOT)}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_unavailable(reason: str) -> None:
    """Persist a clear unavailable status if setup fails."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "unavailable",
        "reason": reason,
        "model_name": ft.MODEL_NAME,
        "mode": MODE,
        "seeds": TRAINING_SEEDS,
        "input_path": str(ft.INPUT_PATH.relative_to(PROJECT_ROOT)),
        "artifacts": {
            "report": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
            "metrics_json": str(METRICS_PATH.relative_to(PROJECT_ROOT)),
        },
    }
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# Pretrained Fine-Tuning Seed Check",
                "",
                "status: `unavailable`",
                f"reason: `{reason}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    METRICS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    """Run the seed robustness check."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        ft.require_batch_size()
        ft.set_seed(ft.RANDOM_STATE)
        data = ft.load_dataset()
        split_info = ft.grouped_train_test_split(data)
        validation_info = ft.inner_validation_split(
            data,
            split_info["train_idx"],
            split_info["groups"],
        )

        tokenizer = AutoTokenizer.from_pretrained(ft.MODEL_NAME)
        probe_backbone = AutoModel.from_pretrained(ft.MODEL_NAME)
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
        layer_probe = ft.identify_transformer_layers(probe_backbone)
        del probe_backbone
        gc.collect()
        if not layer_probe["available"] or layer_probe["layer_count"] < 1:
            raise RuntimeError(
                "Could not safely identify a final transformer layer for last_1_layer: "
                f"{layer_probe['reason']}"
            )
    except Exception as exc:
        write_unavailable(str(exc))
        print(f"seed check unavailable: {exc}")
        return 1

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device_name}", flush=True)
    if device_name == "cpu":
        print(
            "Warning: running fine-tuning seed checks on CPU can be very slow.",
            flush=True,
        )

    results = []
    for seed in TRAINING_SEEDS:
        print(f"Starting seed check: seed={seed}, mode={MODE}", flush=True)
        try:
            results.append(
                train_one_seed(
                    seed=seed,
                    data=data,
                    split_info=split_info,
                    validation_info=validation_info,
                    tokenizer=tokenizer,
                    style=style,
                    max_length=max_length,
                    device_name=device_name,
                )
            )
        except Exception as exc:
            results.append(
                {
                    "seed": int(seed),
                    "mode": MODE,
                    "valid": False,
                    "reason": str(exc),
                }
            )
            print(f"Seed {seed} failed: {exc}", flush=True)

    aggregate = aggregate_seed_metrics(results)
    comparison = build_comparison(aggregate, results)
    save_figure(results, aggregate)

    payload = {
        "status": "available",
        "model_name": ft.MODEL_NAME,
        "environment_variable": ft.MODEL_ENV_VAR,
        "input_path": str(ft.INPUT_PATH.relative_to(PROJECT_ROOT)),
        "row_count": int(len(data)),
        "label_counts": ft.label_counts(data["label"].to_numpy()),
        "input_variant": "sequence_pair_text",
        "mode": MODE,
        "seeds": TRAINING_SEEDS,
        "device": device_name,
        "batch_size": ft.BATCH_SIZE,
        "max_epochs": ft.MAX_EPOCHS,
        "early_stopping_patience": ft.EARLY_STOPPING_PATIENCE,
        "tokenization_style": style,
        "tokenization_attempts": tokenization_attempts,
        "max_sequence_length": max_length,
        "split": {
            "group_column": ft.GROUP_COLUMN,
            "train_test": {
                **ft.split_diagnostics(
                    data,
                    split_info["train_idx"],
                    split_info["test_idx"],
                    split_info["groups"],
                ),
                "split_random_state": split_info["split_random_state"],
            },
            "validation": {
                **ft.split_diagnostics(
                    data,
                    validation_info["train_core_idx"],
                    validation_info["val_idx"],
                    split_info["groups"],
                ),
                "method": validation_info["method"],
                "reason": validation_info["reason"],
                "split_random_state": validation_info["split_random_state"],
            },
        },
        "results": results,
        "aggregate": aggregate,
        "comparison": comparison,
        "artifacts": {
            "report": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
            "metrics_json": str(METRICS_PATH.relative_to(PROJECT_ROOT)),
            "roc_pr_figure": str(FIGURE_PATH.relative_to(PROJECT_ROOT)),
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
            aggregate=aggregate,
            comparison=comparison,
            device_name=device_name,
        ),
        encoding="utf-8",
    )

    print("\nSeed-check metrics")
    for line in per_seed_table(results):
        print(line)
    print("\nAggregate metrics")
    for line in aggregate_table(aggregate):
        print(line)
    print("\nBaseline comparison")
    for line in comparison_table(comparison):
        print(line)

    return 0


if __name__ == "__main__":
    sys.exit(main())
