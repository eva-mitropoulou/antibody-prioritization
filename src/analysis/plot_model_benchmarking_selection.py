"""Create README figures for model benchmarking and broad-model selection.

The script reads committed metric summaries only. It does not train models,
score records, or inspect raw sequence tables.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

MPLCONFIGDIR = Path("/tmp") / "antibody_prioritization_mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"
ASSET_DIR = PROJECT_ROOT / "docs" / "assets"

MODEL_REGISTRY_PATH = METRICS_DIR / "model_registry.json"
LM_REGISTRY_PATH = METRICS_DIR / "lm_benchmark_registry.json"
PRETRAINED_FINETUNE_PATH = METRICS_DIR / "pretrained_finetune_metrics.json"
PRETRAINED_SEED_CHECK_PATH = METRICS_DIR / "pretrained_finetune_seed_check_metrics.json"
PRETRAINED_LORA_PATH = METRICS_DIR / "pretrained_lora_distilled_metrics.json"
SOURCE_ROBUST_PATH = METRICS_DIR / "source_robust_model_selection_metrics.json"
CALIBRATION_THRESHOLD_PATH = METRICS_DIR / "calibration_threshold_metrics.json"

BROAD_BENCHMARK_PATH = ASSET_DIR / "broad_model_benchmark.png"
FOLLOWUP_PATH = ASSET_DIR / "kmer_vs_igbert_followup.png"
ROBUSTNESS_PATH = ASSET_DIR / "selected_model_robustness.png"

TEXT = "#27313a"
MUTED = "#5c6872"
GRID = "#d7dde3"
BLUE = "#5f87a6"
BLUE_LIGHT = "#8fb3c8"
ORANGE = "#d0693c"
ORANGE_LIGHT = "#f0a35d"
GREY = "#aab4bd"


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metric_label(value: float) -> str:
    return f"{value:.4f}"


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#c8d1da")
    ax.tick_params(axis="y", length=0, colors=TEXT)
    ax.tick_params(axis="x", colors=TEXT)


def add_bar_labels(ax: plt.Axes, bars, values: list[float], x_pad: float = 0.004) -> None:
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            value + x_pad,
            bar.get_y() + bar.get_height() / 2,
            metric_label(value),
            va="center",
            ha="left",
            fontsize=8.5,
            color=TEXT,
        )


def source_robust_comparison_path(source_robust: dict) -> Path:
    return PROJECT_ROOT / source_robust["artifacts"]["comparison_csv"]


def collect_broad_models(model_registry: dict, lm_registry: dict) -> list[dict]:
    primary = model_registry["primary_broad_scorer"]
    rows = [
        {
            "model_id": primary["model_id"],
            "name": "Whole-pair k-mer",
            "roc_auc": primary["metrics"]["roc_auc"],
            "pr_auc": primary["metrics"]["pr_auc"],
            "selected": True,
        }
    ]

    names = {
        "pretrained_finetune": "IgBERT fine-tune",
        "embedding_baseline": "AbLang2 embedding logreg",
        "pytorch_embedding_mlp": "AbLang2 embedding MLP",
        "pretrained_frozen_baseline": "Frozen IgBERT MLP",
    }
    wanted = set(names)
    for entry in lm_registry["entries"]:
        model_id = entry["model_id"]
        if model_id not in wanted:
            continue
        if entry.get("matched_kmer_reference_id") != "full_strict_dataset":
            continue
        if entry.get("row_count") != primary["row_count"]:
            continue
        rows.append(
            {
                "model_id": model_id,
                "name": names[model_id],
                "roc_auc": entry["metrics"]["roc_auc"],
                "pr_auc": entry["metrics"]["pr_auc"],
                "selected": False,
            }
        )

    order = [
        "kmer_tfidf_logreg_pair_text",
        "pretrained_finetune",
        "embedding_baseline",
        "pytorch_embedding_mlp",
        "pretrained_frozen_baseline",
    ]
    return sorted(rows, key=lambda row: order.index(row["model_id"]))


def plot_broad_model_benchmark(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.3), sharey=True)
    names = [row["name"] for row in rows]
    y = np.arange(len(rows))
    colors = [ORANGE if row["selected"] else BLUE for row in rows]

    for ax, metric, title in [
        (axes[0], "roc_auc", "ROC-AUC"),
        (axes[1], "pr_auc", "PR-AUC"),
    ]:
        values = [row[metric] for row in rows]
        bars = ax.barh(y, values, color=colors, height=0.58)
        add_bar_labels(ax, bars, values)
        ax.set_xlim(0.70, 0.85)
        ax.set_title(f"{title}\nhigher is better", loc="left", fontsize=11, color=TEXT)
        ax.set_xlabel(title, fontsize=10, color=TEXT)
        style_axis(ax)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names, fontsize=9.5)
    axes[0].invert_yaxis()
    axes[1].tick_params(labelleft=False)

    axes[1].annotate(
        "IgBERT had slightly higher PR-AUC\nbut lower ROC-AUC.",
        xy=(0.8317, 1),
        xytext=(0.765, 1.75),
        textcoords="data",
        fontsize=9,
        color=TEXT,
        arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 1.2},
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#f4f8fb",
            "edgecolor": "#b5cadd",
            "linewidth": 1.0,
        },
    )

    fig.suptitle(
        "Broad model benchmark",
        x=0.08,
        y=0.97,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color=TEXT,
    )
    fig.text(
        0.08,
        0.025,
        "Same-subset comparison on the full strict labelled dataset.",
        fontsize=9.5,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.25, right=0.98, top=0.83, bottom=0.15, wspace=0.16)
    fig.savefig(BROAD_BENCHMARK_PATH, dpi=220)
    plt.close(fig)


def collect_followup_rows(
    model_registry: dict,
    pretrained_finetune: dict,
    seed_check: dict,
    lora_distilled: dict,
) -> list[dict]:
    primary = model_registry["primary_broad_scorer"]
    best_single = pretrained_finetune["comparison"]["best_mode"]
    seed_mean = seed_check["comparison"]["seed_mean"]
    lora_mean = lora_distilled["comparison"]["seed_mean"]

    return [
        {
            "name": "Whole-pair k-mer",
            "roc_auc": primary["metrics"]["roc_auc"],
            "pr_auc": primary["metrics"]["pr_auc"],
            "selected": True,
        },
        {
            "name": "IgBERT fine-tune\nbest single run",
            "roc_auc": best_single["roc_auc"],
            "pr_auc": best_single["average_precision"],
            "selected": False,
        },
        {
            "name": "IgBERT fine-tune\n5-seed mean",
            "roc_auc": seed_mean["roc_auc"],
            "pr_auc": seed_mean["pr_auc"],
            "selected": False,
        },
        {
            "name": "LoRA/distilled IgBERT\n3-seed mean",
            "roc_auc": lora_mean["roc_auc"],
            "pr_auc": lora_mean["pr_auc"],
            "selected": False,
        },
    ]


def plot_followup(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    y = np.arange(len(rows))
    height = 0.32
    roc_values = [row["roc_auc"] for row in rows]
    pr_values = [row["pr_auc"] for row in rows]
    selected = [row["selected"] for row in rows]

    roc_colors = [ORANGE if flag else BLUE_LIGHT for flag in selected]
    pr_colors = [ORANGE_LIGHT if flag else BLUE for flag in selected]
    roc_bars = ax.barh(y - height / 2, roc_values, height=height, color=roc_colors, label="ROC-AUC")
    pr_bars = ax.barh(y + height / 2, pr_values, height=height, color=pr_colors, label="PR-AUC")
    add_bar_labels(ax, roc_bars, roc_values)
    add_bar_labels(ax, pr_bars, pr_values)

    ax.set_yticks(y)
    ax.set_yticklabels([row["name"] for row in rows], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0.70, 0.85)
    ax.set_xlabel("AUC", fontsize=10, color=TEXT)
    style_axis(ax)
    ax.legend(loc="lower right", frameon=False, fontsize=9.5)

    ax.annotate(
        "single run looked competitive",
        xy=(0.8317, 1 + height / 2),
        xytext=(0.785, 1.55),
        fontsize=9,
        color=TEXT,
        arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 1.2},
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#f4f8fb",
            "edgecolor": "#b5cadd",
            "linewidth": 1.0,
        },
    )
    ax.annotate(
        "5-seed mean fell below k-mer",
        xy=(0.8151, 2 + height / 2),
        xytext=(0.775, 2.55),
        fontsize=9,
        color=TEXT,
        arrowprops={"arrowstyle": "->", "color": "#7c8790", "lw": 1.2},
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#f7f8f9",
            "edgecolor": "#c8d1da",
            "linewidth": 1.0,
        },
    )

    fig.suptitle(
        "Why the k-mer scorer was retained",
        x=0.08,
        y=0.97,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color=TEXT,
    )
    fig.text(
        0.08,
        0.025,
        "Follow-up IgBERT checks used the same strict labelled dataset; seed-averaged results did not beat the k-mer baseline overall.",
        fontsize=9.5,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.27, right=0.96, top=0.86, bottom=0.14)
    fig.savefig(FOLLOWUP_PATH, dpi=220)
    plt.close(fig)


def collect_robustness_rows(model_registry: dict, source_robust: dict) -> tuple[list[dict], list[dict]]:
    primary = model_registry["primary_broad_scorer"]
    comparison_path = source_robust_comparison_path(source_robust)
    comparison = pd.read_csv(comparison_path)
    selected_model = source_robust["model_selection"]["selected_model"]
    selected_row = comparison.loc[comparison["model_variant"].eq(selected_model)].iloc[0]

    auc_rows = [
        {
            "name": "Grouped benchmark",
            "roc_auc": primary["metrics"]["roc_auc"],
            "pr_auc": primary["metrics"]["pr_auc"],
        },
        {
            "name": "Source/study holdout",
            "roc_auc": float(selected_row["weighted_roc_auc"]),
            "pr_auc": float(selected_row["weighted_pr_auc"]),
        },
    ]
    threshold_rows = [
        {"name": "Precision", "value": float(selected_row["best_threshold_precision"])},
        {"name": "Recall", "value": float(selected_row["best_threshold_recall"])},
        {"name": "Coverage", "value": float(selected_row["best_threshold_coverage"])},
    ]
    return auc_rows, threshold_rows


def plot_robustness(auc_rows: list[dict], threshold_rows: list[dict]) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.0, 5.3),
        gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.34},
    )

    y = np.arange(len(auc_rows))
    height = 0.32
    roc_values = [row["roc_auc"] for row in auc_rows]
    pr_values = [row["pr_auc"] for row in auc_rows]
    roc_bars = axes[0].barh(y - height / 2, roc_values, height=height, color=BLUE_LIGHT, label="ROC-AUC")
    pr_bars = axes[0].barh(y + height / 2, pr_values, height=height, color=BLUE, label="PR-AUC")
    add_bar_labels(axes[0], roc_bars, roc_values)
    add_bar_labels(axes[0], pr_bars, pr_values)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([row["name"] for row in auc_rows], fontsize=9.5)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0.55, 0.86)
    axes[0].set_xlabel("AUC", fontsize=10, color=TEXT)
    axes[0].set_title("A. Selected k-mer validation", loc="left", fontsize=11, fontweight="bold", color=TEXT)
    axes[0].legend(loc="lower right", frameon=False, fontsize=9.5)
    style_axis(axes[0])

    metric_names = [row["name"] for row in threshold_rows]
    values = [row["value"] for row in threshold_rows]
    colors = [ORANGE_LIGHT, GREY, GREY]
    bars = axes[1].bar(metric_names, values, color=colors, width=0.58)
    for bar, value in zip(bars, values, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            metric_label(value),
            ha="center",
            va="bottom",
            fontsize=9,
            color=TEXT,
        )
    axes[1].set_ylim(0, 0.95)
    axes[1].set_ylabel("Metric value", fontsize=10, color=TEXT)
    axes[1].set_title("B. Threshold 0.7 review cutoff", loc="left", fontsize=11, fontweight="bold", color=TEXT)
    axes[1].grid(axis="y", color=GRID, linewidth=0.8)
    axes[1].set_axisbelow(True)
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].spines[["bottom", "left"]].set_color("#c8d1da")
    axes[1].tick_params(colors=TEXT)

    fig.suptitle(
        "Selected model robustness and review cutoff",
        x=0.08,
        y=0.97,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color=TEXT,
    )
    fig.text(
        0.08,
        0.025,
        "Performance drops under source/study holdout, so scores are used for review ranking rather than as final biological labels.",
        fontsize=9.5,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.20, right=0.96, top=0.82, bottom=0.17)
    fig.savefig(ROBUSTNESS_PATH, dpi=220)
    plt.close(fig)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    model_registry = read_json(MODEL_REGISTRY_PATH)
    lm_registry = read_json(LM_REGISTRY_PATH)
    pretrained_finetune = read_json(PRETRAINED_FINETUNE_PATH)
    seed_check = read_json(PRETRAINED_SEED_CHECK_PATH)
    lora_distilled = read_json(PRETRAINED_LORA_PATH)
    source_robust = read_json(SOURCE_ROBUST_PATH)
    read_json(CALIBRATION_THRESHOLD_PATH)

    broad_rows = collect_broad_models(model_registry, lm_registry)
    plot_broad_model_benchmark(broad_rows)

    followup_rows = collect_followup_rows(
        model_registry,
        pretrained_finetune,
        seed_check,
        lora_distilled,
    )
    plot_followup(followup_rows)

    auc_rows, threshold_rows = collect_robustness_rows(model_registry, source_robust)
    plot_robustness(auc_rows, threshold_rows)

    for path in [BROAD_BENCHMARK_PATH, FOLLOWUP_PATH, ROBUSTNESS_PATH]:
        print(f"Wrote {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
