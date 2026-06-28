"""Run OAS unknown-target background retrieval diagnostics.

This module compares project records against local paired OAS natural background
records. OAS rows are unknown-target background for enrichment analysis, and the
metrics are kept separate from the main neutralisation classification benchmark.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/antibody_prioritization_mplconfig")
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from _safe_analysis_utils import PROJECT_ROOT, write_json, write_text


OAS_PATH = PROJECT_ROOT / "data" / "processed" / "oas" / "oas_paired_standardized.csv"
PROJECT_PATH = PROJECT_ROOT / "data" / "processed" / "neutral_prepared_sequences.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "oas_background_retrieval_report.md"
SKIP_REPORT_PATH = PROJECT_ROOT / "reports" / "oas_background_skipped.md"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics" / "oas_background_retrieval_metrics.json"
SCORES_PATH = PROJECT_ROOT / "reports" / "oas_background_retrieval_scores.csv"
SCORE_DISTRIBUTION_PATH = PROJECT_ROOT / "reports" / "figures" / "oas_retrieval_score_distribution.png"
TOPK_FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "oas_retrieval_topk_enrichment.png"
SPACE_FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "oas_background_sequence_space.png"

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_ROWS_PER_CLASS = 50_000
TOP_K_VALUES = [50, 100, 500]
MISSING_TEXT_VALUES = {"", "na", "nan", "none", "null", "nd", "n/a", "unknown"}


def relpath(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def normalize_sequence(value: Any) -> str:
    """Normalize existing sequence text internally without printing it."""
    text = re.sub(r"\s+", "", str(value or "")).upper()
    return "" if text.lower() in MISSING_TEXT_VALUES else text


def compact_pair_text(heavy: Any, light: Any = "") -> str:
    """Build compact pair text for character k-mers."""
    heavy_text = normalize_sequence(heavy)
    light_text = normalize_sequence(light)
    if light_text:
        return f"{heavy_text}|{light_text}"
    return heavy_text


def compact_model_text(value: Any) -> str:
    """Compact an existing pair-text field."""
    text = str(value or "").replace("[SEP]", "|")
    return re.sub(r"\s+", "", text).upper()


def sequence_hash(value: str) -> str:
    """Return a stable sequence-pair hash for sequence-redacted outputs."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_oas_background() -> pd.DataFrame:
    """Load standardized OAS background rows."""
    if not OAS_PATH.exists():
        raise FileNotFoundError(f"Missing standardized OAS file: {relpath(OAS_PATH)}")
    data = pd.read_csv(OAS_PATH, dtype=str, keep_default_na=False)
    required = {"heavy_sequence", "light_sequence", "sequence_pair_text", "source_file", "background_source"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Standardized OAS file missing columns: {missing}")
    output = data.copy()
    output["compact_pair_text"] = output["sequence_pair_text"].map(compact_model_text)
    output = output[output["compact_pair_text"].str.len().gt(0)].copy()
    output["source_class"] = "oas_unknown_background"
    output["retrieval_label"] = 0
    return output[["compact_pair_text", "source_class", "retrieval_label", "source_file"]]


def load_project_records() -> pd.DataFrame:
    """Load project records and build compact pair text."""
    data = pd.read_csv(PROJECT_PATH, dtype=str, keep_default_na=False)
    if "sequence_a" not in data.columns:
        raise ValueError("Project table missing sequence_a column.")
    light = data["sequence_b"] if "sequence_b" in data.columns else pd.Series([""] * len(data), index=data.index)
    output = pd.DataFrame(
        {
            "compact_pair_text": [
                compact_pair_text(heavy, light_value)
                for heavy, light_value in zip(data["sequence_a"], light)
            ],
            "source_class": "project_record",
            "retrieval_label": 1,
            "source_file": PROJECT_PATH.name,
        }
    )
    output = output[output["compact_pair_text"].str.len().gt(0)].copy()
    output = output.drop_duplicates("compact_pair_text")
    return output.reset_index(drop=True)


def build_retrieval_table() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build balanced project-vs-OAS retrieval table."""
    project = load_project_records()
    oas = load_oas_background()
    project_pairs = set(project["compact_pair_text"])
    overlap_mask = oas["compact_pair_text"].isin(project_pairs)
    exact_overlap_count = int(overlap_mask.sum())
    oas = oas.loc[~overlap_mask].drop_duplicates("compact_pair_text").reset_index(drop=True)

    n_per_class = min(len(project), len(oas), MAX_ROWS_PER_CLASS)
    if n_per_class < 10:
        raise ValueError("Not enough project/OAS rows after overlap removal.")
    project_sample = project.sample(n=n_per_class, random_state=RANDOM_STATE)
    oas_sample = oas.sample(n=n_per_class, random_state=RANDOM_STATE)
    table = pd.concat([project_sample, oas_sample], ignore_index=True)
    table = table.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    table["sequence_pair_hash"] = table["compact_pair_text"].map(sequence_hash)
    diagnostics = {
        "project_row_count_before_balance": int(len(project)),
        "oas_row_count_before_overlap_removal": int(len(load_oas_background())),
        "oas_row_count_after_overlap_removal": int(len(oas)),
        "exact_overlap_count": exact_overlap_count,
        "balanced_rows_per_class": int(n_per_class),
        "retrieval_table_row_count": int(len(table)),
    }
    return table, diagnostics


def make_model() -> Pipeline:
    """Create the requested k-mer retrieval model."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=50_000,
                ),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=5000, class_weight="balanced"),
            ),
        ]
    )


def topk_enrichment(scores: pd.DataFrame, baseline_fraction: float) -> dict[str, Any]:
    """Compute top-k project-record enrichment."""
    ranked = scores.sort_values("project_retrieval_probability", ascending=False)
    output: dict[str, Any] = {}
    for k in TOP_K_VALUES:
        selected = ranked.head(min(k, len(ranked)))
        precision = float(selected["retrieval_label"].mean()) if len(selected) else None
        enrichment = (
            float(precision / baseline_fraction)
            if precision is not None and baseline_fraction > 0
            else None
        )
        output[str(k)] = {
            "selected_count": int(len(selected)),
            "project_record_count": int(selected["retrieval_label"].sum()),
            "project_record_fraction": precision,
            "enrichment_over_random_baseline": enrichment,
        }
    return output


def run_retrieval() -> tuple[pd.DataFrame, dict[str, Any], Pipeline, pd.DataFrame]:
    """Fit and evaluate OAS-vs-project retrieval."""
    table, diagnostics = build_retrieval_table()
    train_idx, test_idx = train_test_split(
        table.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=table["retrieval_label"],
    )
    train = table.loc[train_idx].copy()
    test = table.loc[test_idx].copy()
    model = make_model()
    model.fit(train["compact_pair_text"], train["retrieval_label"])
    probabilities = model.predict_proba(test["compact_pair_text"])[:, list(model.classes_).index(1)]
    scores = test[
        ["sequence_pair_hash", "source_class", "retrieval_label", "source_file"]
    ].copy()
    scores["split"] = "test"
    scores["project_retrieval_probability"] = probabilities.astype(float)
    baseline_fraction = float(test["retrieval_label"].mean())
    metrics = {
        "status": "available",
        "input_paths": {
            "oas_background": relpath(OAS_PATH),
            "project_records": relpath(PROJECT_PATH),
        },
        "background_label_semantics": "OAS paired records are unknown-target natural background for enrichment analysis.",
        "classification_task_mixed_with_main_task": False,
        "model": {
            "vectorizer": 'TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=2, max_features=50000)',
            "classifier": 'LogisticRegression(max_iter=5000, class_weight="balanced")',
        },
        "split": {
            "strategy": "stratified_train_test_split",
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_label_counts": {
                "oas_unknown_background": int(train["retrieval_label"].eq(0).sum()),
                "project_record": int(train["retrieval_label"].eq(1).sum()),
            },
            "test_label_counts": {
                "oas_unknown_background": int(test["retrieval_label"].eq(0).sum()),
                "project_record": int(test["retrieval_label"].eq(1).sum()),
            },
        },
        **diagnostics,
        "random_baseline_positive_fraction": baseline_fraction,
        "roc_auc": float(roc_auc_score(test["retrieval_label"], probabilities)),
        "pr_auc": float(average_precision_score(test["retrieval_label"], probabilities)),
        "topk_enrichment": topk_enrichment(scores, baseline_fraction),
        "artifacts": {
            "report": relpath(REPORT_PATH),
            "metrics_json": relpath(METRICS_PATH),
            "scores_csv": relpath(SCORES_PATH),
            "score_distribution_figure": relpath(SCORE_DISTRIBUTION_PATH),
            "topk_enrichment_figure": relpath(TOPK_FIGURE_PATH),
            "sequence_space_figure": relpath(SPACE_FIGURE_PATH),
        },
    }
    return scores, metrics, model, test


def build_skip_report(reason: str) -> dict[str, Any]:
    """Build skip metrics if local OAS is unavailable."""
    return {
        "status": "skipped",
        "reason": reason,
        "input_paths": {
            "oas_background": relpath(OAS_PATH),
            "project_records": relpath(PROJECT_PATH),
        },
        "background_label_semantics": "OAS paired records are unknown-target natural background used for enrichment analysis.",
        "classification_task_mixed_with_main_task": False,
    }


def save_score_distribution(scores: pd.DataFrame) -> None:
    """Save score distributions by source class."""
    SCORE_DISTRIBUTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for source_class, color in [
        ("oas_unknown_background", "#4C78A8"),
        ("project_record", "#F58518"),
    ]:
        values = scores.loc[
            scores["source_class"].eq(source_class),
            "project_retrieval_probability",
        ]
        if len(values):
            ax.hist(values, bins=30, alpha=0.65, label=source_class, color=color)
    ax.set_xlabel("Project retrieval probability")
    ax.set_ylabel("Test record count")
    ax.set_title("OAS background retrieval score distribution")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(SCORE_DISTRIBUTION_PATH, dpi=200)
    plt.close(fig)


def save_topk_figure(metrics: dict[str, Any]) -> None:
    """Save top-k enrichment figure."""
    TOPK_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    topk = metrics["topk_enrichment"]
    labels = list(topk)
    values = [topk[label]["enrichment_over_random_baseline"] for label in labels]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color="#54A24B")
    ax.axhline(1.0, color="0.4", linestyle="--", linewidth=1)
    ax.set_xlabel("Top-k")
    ax.set_ylabel("Enrichment over random baseline")
    ax.set_title("Top-k retrieval enrichment")
    fig.tight_layout()
    fig.savefig(TOPK_FIGURE_PATH, dpi=200)
    plt.close(fig)


def save_sequence_space_figure(model: Pipeline, test: pd.DataFrame, scores: pd.DataFrame) -> None:
    """Save a 2D sequence-space view from k-mer TF-IDF/SVD features."""
    SPACE_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    matrix = model.named_steps["tfidf"].transform(test["compact_pair_text"])
    n_components = 2 if matrix.shape[1] > 2 else max(1, matrix.shape[1])
    if n_components < 2:
        return
    svd = TruncatedSVD(n_components=2, random_state=RANDOM_STATE)
    coords = svd.fit_transform(matrix)
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = scores["retrieval_label"].map({0: "#4C78A8", 1: "#F58518"}).to_numpy()
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=9, alpha=0.55, linewidths=0)
    ax.set_xlabel("SVD component 1")
    ax.set_ylabel("SVD component 2")
    ax.set_title("OAS background versus project sequence space")
    fig.tight_layout()
    fig.savefig(SPACE_FIGURE_PATH, dpi=200)
    plt.close(fig)


def build_report(metrics: dict[str, Any]) -> str:
    """Build Markdown retrieval report."""
    lines = [
        "# OAS Background Retrieval",
        "",
        "This diagnostic treats paired OAS records as unknown-target natural",
        "background for enrichment analysis. Metrics are kept separate from the main",
        "neutralisation classification benchmark.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Status | {metrics['status']} |",
        f"| Project rows before balance | {metrics.get('project_row_count_before_balance', 0)} |",
        f"| OAS rows before overlap removal | {metrics.get('oas_row_count_before_overlap_removal', 0)} |",
        f"| Exact overlap count | {metrics.get('exact_overlap_count', 0)} |",
        f"| OAS rows after overlap removal | {metrics.get('oas_row_count_after_overlap_removal', 0)} |",
        f"| Balanced rows per class | {metrics.get('balanced_rows_per_class', 0)} |",
        f"| Train rows | {metrics.get('split', {}).get('train_rows', 0)} |",
        f"| Test rows | {metrics.get('split', {}).get('test_rows', 0)} |",
        f"| Random baseline positive fraction | {metrics.get('random_baseline_positive_fraction', 0):.4f} |",
        f"| ROC-AUC | {metrics.get('roc_auc', 0):.4f} |",
        f"| PR-AUC | {metrics.get('pr_auc', 0):.4f} |",
        "",
        "## Top-k Enrichment",
        "",
        "| k | Selected | Project records | Project fraction | Enrichment over random |",
        "|---:|---:|---:|---:|---:|",
    ]
    for k, item in metrics.get("topk_enrichment", {}).items():
        lines.append(
            f"| {k} | {item['selected_count']} | {item['project_record_count']} | "
            f"{item['project_record_fraction']:.4f} | "
            f"{item['enrichment_over_random_baseline']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
        ]
    )
    for path in metrics.get("artifacts", {}).values():
        lines.append(f"- `{path}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    for folder in [
        PROJECT_ROOT / "reports" / "figures",
        PROJECT_ROOT / "reports" / "metrics",
    ]:
        folder.mkdir(parents=True, exist_ok=True)
    if not OAS_PATH.exists():
        metrics = build_skip_report("Standardized OAS background file is missing.")
        write_json(METRICS_PATH, metrics)
        text = build_report(metrics)
        write_text(REPORT_PATH, text)
        write_text(SKIP_REPORT_PATH, text)
        print(
            "oas_retrieval_status=skipped; reason=standardized_oas_missing; "
            f"oas_path={relpath(OAS_PATH)}",
            flush=True,
        )
        return 0

    scores, metrics, model, test = run_retrieval()
    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(SCORES_PATH, index=False)
    write_json(METRICS_PATH, metrics)
    write_text(REPORT_PATH, build_report(metrics))
    save_score_distribution(scores)
    save_topk_figure(metrics)
    save_sequence_space_figure(model, test, scores)
    print(
        "oas_retrieval_status=available; "
        f"project_rows={metrics['project_row_count_before_balance']}; "
        f"oas_rows={metrics['oas_row_count_after_overlap_removal']}; "
        f"exact_overlap_count={metrics['exact_overlap_count']}; "
        f"roc_auc={metrics['roc_auc']:.4f}; pr_auc={metrics['pr_auc']:.4f}; "
        f"topk_enrichment={metrics['topk_enrichment']}; "
        f"metrics_path={relpath(METRICS_PATH)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
