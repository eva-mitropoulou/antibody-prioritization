"""Run hard matched OAS unknown-target background retrieval control."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/antibody_prioritization_mplconfig")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from _safe_analysis_utils import PROJECT_ROOT, write_json, write_text


PROJECT_PATH = PROJECT_ROOT / "data" / "processed" / "neutral_prepared_sequences.csv"
OAS_PATH = PROJECT_ROOT / "data" / "processed" / "oas" / "oas_paired_standardized.csv"
AUDIT_PATH = PROJECT_ROOT / "reports" / "oas_matched_background_audit.md"
AUDIT_METRICS_PATH = PROJECT_ROOT / "reports" / "metrics" / "oas_matched_background_audit.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "oas_matched_background_retrieval_report.md"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics" / "oas_matched_background_retrieval_metrics.json"
SCORES_PATH = PROJECT_ROOT / "reports" / "oas_matched_background_retrieval_scores.csv"
SCORE_DISTRIBUTION_PATH = (
    PROJECT_ROOT / "reports" / "figures" / "oas_matched_retrieval_score_distribution.png"
)
TOPK_FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "oas_matched_retrieval_topk_enrichment.png"

RANDOM_STATE = 42
TEST_SIZE = 0.2
TOP_K_VALUES = [50, 100, 500]
MISSING_TEXT_VALUES = {"", "na", "nan", "none", "null", "nd", "n/a", "unknown"}


def relpath(path: Path) -> str:
    """Return project-relative path."""
    return str(path.relative_to(PROJECT_ROOT))


def normalize_sequence(value: Any) -> str:
    """Compact an existing sequence-like value for internal analysis."""
    text = re.sub(r"\s+", "", str(value or "")).upper()
    return "" if text.lower() in MISSING_TEXT_VALUES else text


def compact_pair_text(heavy: Any, light: Any = "") -> str:
    """Build compact heavy/light pair text."""
    heavy_text = normalize_sequence(heavy)
    light_text = normalize_sequence(light)
    return f"{heavy_text}[SEP]{light_text}" if light_text else heavy_text


def compact_existing_pair(value: Any) -> str:
    """Compact an existing sequence-pair field."""
    return re.sub(r"\s+", "", str(value or "")).upper()


def sequence_hash(value: str) -> str:
    """Return a stable redacted sequence-pair key."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def length_bin(value: int, width: int) -> int:
    """Return numeric bin start."""
    return int((int(value) // width) * width)


def build_bins(data: pd.DataFrame) -> pd.DataFrame:
    """Append coarse matching bins."""
    output = data.copy()
    output["heavy_length_bin"] = output["heavy_length"].map(lambda value: length_bin(value, 5))
    output["light_length_bin"] = output["light_length"].map(lambda value: length_bin(value, 5))
    output["total_length_bin"] = output["total_pair_length"].map(lambda value: length_bin(value, 10))
    output["matching_bin"] = (
        output["heavy_length_bin"].astype(str)
        + "|"
        + output["light_length_bin"].astype(str)
        + "|"
        + output["total_length_bin"].astype(str)
        + "|"
        + output["has_light"].astype(str)
    )
    return output


def load_project_records() -> pd.DataFrame:
    """Load project records and compute matching features."""
    data = pd.read_csv(PROJECT_PATH, dtype=str, keep_default_na=False)
    if "sequence_a" not in data.columns:
        raise ValueError("Project input is missing sequence_a.")
    light_values = (
        data["sequence_b"]
        if "sequence_b" in data.columns
        else pd.Series([""] * len(data), index=data.index)
    )
    heavy = data["sequence_a"].map(normalize_sequence)
    light = light_values.map(normalize_sequence)
    cdrh3_source = "group_feature_cdr3" if "group_feature_cdr3" in data.columns else None
    cdrh3_length = (
        data[cdrh3_source].map(normalize_sequence).str.len()
        if cdrh3_source
        else pd.Series([np.nan] * len(data), index=data.index)
    )
    output = pd.DataFrame(
        {
            "compact_pair_text": [
                compact_pair_text(heavy_value, light_value)
                for heavy_value, light_value in zip(heavy, light)
            ],
            "heavy_length": heavy.str.len().astype(int),
            "light_length": light.str.len().astype(int),
            "has_light": light.ne(""),
            "cdrh3_length": cdrh3_length,
            "heavy_prefix_3": heavy.str[:3],
            "light_prefix_3": light.str[:3],
            "source": "project_record",
            "retrieval_label": 1,
        }
    )
    output["total_pair_length"] = output["heavy_length"] + output["light_length"]
    output = output[output["compact_pair_text"].str.len().gt(0)].drop_duplicates(
        "compact_pair_text"
    )
    return build_bins(output).reset_index(drop=True)


def load_oas_records() -> pd.DataFrame:
    """Load standardized OAS background and compute matching features."""
    data = pd.read_csv(OAS_PATH, dtype=str, keep_default_na=False)
    required = {"heavy_sequence", "light_sequence", "sequence_pair_text"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"OAS input is missing required columns: {missing}")
    heavy = data["heavy_sequence"].map(normalize_sequence)
    light = data["light_sequence"].map(normalize_sequence)
    output = pd.DataFrame(
        {
            "compact_pair_text": data["sequence_pair_text"].map(compact_existing_pair),
            "heavy_length": heavy.str.len().astype(int),
            "light_length": light.str.len().astype(int),
            "has_light": light.ne(""),
            "cdrh3_length": np.nan,
            "heavy_prefix_3": heavy.str[:3],
            "light_prefix_3": light.str[:3],
            "source": "oas_unknown_target_background",
            "retrieval_label": 0,
        }
    )
    output["total_pair_length"] = output["heavy_length"] + output["light_length"]
    output = output[output["compact_pair_text"].str.len().gt(0)].drop_duplicates(
        "compact_pair_text"
    )
    return build_bins(output).reset_index(drop=True)


def sample_indices(indices: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample indices without DataFrame.sample."""
    if len(indices) <= n:
        return indices
    selected_positions = rng.choice(np.arange(len(indices)), size=n, replace=False)
    return indices[selected_positions]


def build_matched_dataset() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a length/status-matched OAS background retrieval table."""
    rng = np.random.default_rng(RANDOM_STATE)
    project = load_project_records()
    raw_oas = load_oas_records()
    project_pairs = set(project["compact_pair_text"])
    exact_overlap_mask = raw_oas["compact_pair_text"].isin(project_pairs)
    exact_overlap_count = int(exact_overlap_mask.sum())
    oas = raw_oas.loc[~exact_overlap_mask].copy().reset_index(drop=True)

    project_groups = project.groupby("matching_bin", sort=True).indices
    oas_groups = oas.groupby("matching_bin", sort=True).indices
    matched_project_parts = []
    matched_oas_parts = []
    bin_audit: dict[str, Any] = {}
    skipped_project_rows = 0
    project_rows_not_selected_due_to_oas_shortage = 0

    for matching_bin, project_idx in project_groups.items():
        project_indices = np.asarray(project_idx, dtype=int)
        oas_indices = np.asarray(oas_groups.get(matching_bin, []), dtype=int)
        project_count = int(len(project_indices))
        oas_count = int(len(oas_indices))
        if oas_count == 0:
            skipped_project_rows += project_count
            bin_audit[str(matching_bin)] = {
                "project_rows": project_count,
                "oas_rows_available": 0,
                "matched_rows_per_class": 0,
            }
            continue
        n = min(project_count, oas_count)
        project_rows_not_selected_due_to_oas_shortage += max(0, project_count - n)
        selected_project = sample_indices(project_indices, n, rng)
        selected_oas = sample_indices(oas_indices, n, rng)
        matched_project_parts.append(project.iloc[selected_project])
        matched_oas_parts.append(oas.iloc[selected_oas])
        bin_audit[str(matching_bin)] = {
            "project_rows": project_count,
            "oas_rows_available": oas_count,
            "matched_rows_per_class": int(n),
        }

    if not matched_project_parts or not matched_oas_parts:
        raise ValueError("No matched OAS/project bins were available.")

    matched_project = pd.concat(matched_project_parts, ignore_index=True)
    matched_oas = pd.concat(matched_oas_parts, ignore_index=True)
    matched = pd.concat([matched_project, matched_oas], ignore_index=True)
    permutation = rng.permutation(len(matched))
    matched = matched.iloc[permutation].reset_index(drop=True)
    matched["hashed_sequence_key"] = matched["compact_pair_text"].map(sequence_hash)

    audit = {
        "status": "available",
        "input_paths": {
            "project_records": relpath(PROJECT_PATH),
            "oas_background": relpath(OAS_PATH),
        },
        "background_label_semantics": "OAS is unknown-target background for enrichment analysis.",
        "main_neutralisation_benchmark_mixed": False,
        "project_row_count": int(len(project)),
        "raw_oas_row_count": int(len(raw_oas)),
        "oas_row_count_after_overlap_removal": int(len(oas)),
        "matched_project_row_count": int(len(matched_project)),
        "matched_oas_row_count": int(len(matched_oas)),
        "skipped_project_rows_due_to_no_matched_oas_bin": int(skipped_project_rows),
        "project_rows_not_selected_due_to_oas_bin_shortage": int(
            project_rows_not_selected_due_to_oas_shortage
        ),
        "exact_overlap_count_removed": exact_overlap_count,
        "matching_bin_count": int(len(bin_audit)),
        "matched_nonempty_bin_count": int(
            sum(1 for item in bin_audit.values() if item["matched_rows_per_class"] > 0)
        ),
        "class_balance": {
            "project_record": int(matched["retrieval_label"].eq(1).sum()),
            "oas_unknown_target_background": int(matched["retrieval_label"].eq(0).sum()),
        },
        "length_bin_distributions": {
            "project_heavy_length_bin": count_dict(matched_project["heavy_length_bin"]),
            "oas_heavy_length_bin": count_dict(matched_oas["heavy_length_bin"]),
            "project_light_length_bin": count_dict(matched_project["light_length_bin"]),
            "oas_light_length_bin": count_dict(matched_oas["light_length_bin"]),
            "project_total_length_bin": count_dict(matched_project["total_length_bin"]),
            "oas_total_length_bin": count_dict(matched_oas["total_length_bin"]),
            "project_has_light": count_dict(matched_project["has_light"]),
            "oas_has_light": count_dict(matched_oas["has_light"]),
        },
        "bin_audit": bin_audit,
        "audit_report": relpath(AUDIT_PATH),
        "audit_metrics": relpath(AUDIT_METRICS_PATH),
    }
    return matched, audit


def count_dict(values: pd.Series) -> dict[str, int]:
    """Return JSON-safe counts."""
    counts = values.astype(str).value_counts().sort_index()
    return {str(key): int(value) for key, value in counts.items()}


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
            ("classifier", LogisticRegression(max_iter=5000, class_weight="balanced")),
        ]
    )


def topk_enrichment(scores: pd.DataFrame, baseline: float) -> dict[str, Any]:
    """Compute top-k enrichment on test-set scores."""
    ranked = scores.sort_values("retrieval_score", ascending=False)
    output: dict[str, Any] = {}
    for k in TOP_K_VALUES:
        if len(ranked) < k:
            continue
        selected = ranked.head(k)
        project_fraction = float(selected["retrieval_label"].mean())
        output[str(k)] = {
            "selected_count": int(k),
            "project_record_count": int(selected["retrieval_label"].sum()),
            "project_record_fraction": project_fraction,
            "enrichment_over_random_baseline": float(project_fraction / baseline)
            if baseline > 0
            else None,
        }
    return output


def run_model(matched: pd.DataFrame, audit: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Train and evaluate matched retrieval model."""
    train_idx, test_idx = train_test_split(
        matched.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=matched["retrieval_label"],
    )
    train = matched.loc[train_idx].copy()
    test = matched.loc[test_idx].copy()
    model = make_model()
    model.fit(train["compact_pair_text"], train["retrieval_label"])
    positive_index = list(model.classes_).index(1)
    scores_array = model.predict_proba(test["compact_pair_text"])[:, positive_index]
    predictions = (scores_array >= 0.5).astype(int)
    matrix = confusion_matrix(test["retrieval_label"], predictions, labels=[0, 1])
    scores = test[
        [
            "hashed_sequence_key",
            "source",
            "retrieval_label",
            "heavy_length_bin",
            "light_length_bin",
            "total_length_bin",
            "has_light",
            "matching_bin",
        ]
    ].copy()
    scores["split"] = "test"
    scores["retrieval_score"] = scores_array.astype(float)
    baseline = float(test["retrieval_label"].mean())
    metrics = {
        "status": "available",
        "background_label_semantics": audit["background_label_semantics"],
        "main_neutralisation_benchmark_mixed": False,
        "input_paths": audit["input_paths"],
        "matched_project_row_count": audit["matched_project_row_count"],
        "matched_oas_row_count": audit["matched_oas_row_count"],
        "skipped_project_rows_due_to_no_matched_oas_bin": audit[
            "skipped_project_rows_due_to_no_matched_oas_bin"
        ],
        "project_rows_not_selected_due_to_oas_bin_shortage": audit[
            "project_rows_not_selected_due_to_oas_bin_shortage"
        ],
        "exact_overlap_count_removed": audit["exact_overlap_count_removed"],
        "matching_bin_count": audit["matching_bin_count"],
        "matched_nonempty_bin_count": audit["matched_nonempty_bin_count"],
        "class_balance": audit["class_balance"],
        "split": {
            "strategy": "stratified_train_test_split",
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_label_counts": count_dict(train["retrieval_label"]),
            "test_label_counts": count_dict(test["retrieval_label"]),
        },
        "model": {
            "vectorizer": 'TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=2, max_features=50000)',
            "classifier": 'LogisticRegression(max_iter=5000, class_weight="balanced")',
        },
        "random_baseline_positive_fraction": baseline,
        "roc_auc": float(roc_auc_score(test["retrieval_label"], scores_array)),
        "pr_auc": float(average_precision_score(test["retrieval_label"], scores_array)),
        "confusion_matrix": matrix.astype(int).tolist(),
        "topk_enrichment": topk_enrichment(scores, baseline),
        "artifacts": {
            "audit_report": relpath(AUDIT_PATH),
            "audit_metrics": relpath(AUDIT_METRICS_PATH),
            "retrieval_report": relpath(REPORT_PATH),
            "retrieval_metrics": relpath(METRICS_PATH),
            "scores_csv": relpath(SCORES_PATH),
            "score_distribution_figure": relpath(SCORE_DISTRIBUTION_PATH),
            "topk_enrichment_figure": relpath(TOPK_FIGURE_PATH),
        },
    }
    return scores, metrics


def build_audit_report(audit: dict[str, Any]) -> str:
    """Build aggregate-only matching audit report."""
    lines = [
        "# OAS Matched Background Audit",
        "",
        "OAS rows are treated as unknown-target background for enrichment analysis.",
        "This matching audit reports aggregate matching fields and public-safe identifiers.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Project row count | {audit['project_row_count']} |",
        f"| Raw OAS row count | {audit['raw_oas_row_count']} |",
        f"| Matched project row count | {audit['matched_project_row_count']} |",
        f"| Matched OAS row count | {audit['matched_oas_row_count']} |",
        (
            "| Skipped project rows due to no matched OAS bin | "
            f"{audit['skipped_project_rows_due_to_no_matched_oas_bin']} |"
        ),
        (
            "| Project rows not selected due to OAS bin shortage | "
            f"{audit['project_rows_not_selected_due_to_oas_bin_shortage']} |"
        ),
        f"| Exact overlap count removed | {audit['exact_overlap_count_removed']} |",
        f"| Matching bins | {audit['matching_bin_count']} |",
        f"| Non-empty matched bins | {audit['matched_nonempty_bin_count']} |",
        "",
        "## Class Balance",
        "",
        "| Class | Count |",
        "|---|---:|",
    ]
    for key, value in audit["class_balance"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Length-Bin Distributions", ""])
    for name, counts in audit["length_bin_distributions"].items():
        lines.extend([f"### {name}", "", "| Bin | Count |", "|---|---:|"])
        for key, value in counts.items():
            lines.append(f"| {key} | {value} |")
        lines.append("")
    return "\n".join(lines)


def build_retrieval_report(metrics: dict[str, Any]) -> str:
    """Build aggregate-only matched retrieval report."""
    lines = [
        "# OAS Matched Background Retrieval",
        "",
        "This hard control compares project records against length/status-matched",
        "OAS unknown-target background. It is separate from the main neutralisation",
        "classification benchmark.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Matched project rows | {metrics['matched_project_row_count']} |",
        f"| Matched OAS rows | {metrics['matched_oas_row_count']} |",
        f"| Skipped project rows | {metrics['skipped_project_rows_due_to_no_matched_oas_bin']} |",
        f"| Exact overlap count removed | {metrics['exact_overlap_count_removed']} |",
        f"| Train rows | {metrics['split']['train_rows']} |",
        f"| Test rows | {metrics['split']['test_rows']} |",
        f"| Random baseline positive fraction | {metrics['random_baseline_positive_fraction']:.4f} |",
        f"| ROC-AUC | {metrics['roc_auc']:.4f} |",
        f"| PR-AUC | {metrics['pr_auc']:.4f} |",
        "",
        "## Confusion Matrix",
        "",
        "| True label | Predicted OAS background | Predicted project |",
        "|---|---:|---:|",
        (
            f"| OAS unknown-target background | {metrics['confusion_matrix'][0][0]} | "
            f"{metrics['confusion_matrix'][0][1]} |"
        ),
        (
            f"| Project record | {metrics['confusion_matrix'][1][0]} | "
            f"{metrics['confusion_matrix'][1][1]} |"
        ),
        "",
        "## Top-k Enrichment",
        "",
        "| k | Project records | Project fraction | Enrichment over random |",
        "|---:|---:|---:|---:|",
    ]
    for k, item in metrics["topk_enrichment"].items():
        lines.append(
            f"| {k} | {item['project_record_count']} | "
            f"{item['project_record_fraction']:.4f} | "
            f"{item['enrichment_over_random_baseline']:.4f} |"
        )
    lines.extend(["", "## Artifacts", ""])
    for path in metrics["artifacts"].values():
        lines.append(f"- `{path}`")
    lines.append("")
    return "\n".join(lines)


def save_score_distribution(scores: pd.DataFrame) -> None:
    """Save matched retrieval score distribution."""
    SCORE_DISTRIBUTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for source, color in [
        ("oas_unknown_target_background", "#4C78A8"),
        ("project_record", "#F58518"),
    ]:
        values = scores.loc[scores["source"].eq(source), "retrieval_score"]
        if len(values):
            ax.hist(values, bins=30, alpha=0.65, color=color, label=source)
    ax.set_xlabel("Project retrieval score")
    ax.set_ylabel("Test record count")
    ax.set_title("Matched OAS retrieval score distribution")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(SCORE_DISTRIBUTION_PATH, dpi=200)
    plt.close(fig)


def save_topk_figure(metrics: dict[str, Any]) -> None:
    """Save top-k enrichment figure."""
    if not metrics["topk_enrichment"]:
        return
    TOPK_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    labels = list(metrics["topk_enrichment"])
    values = [
        metrics["topk_enrichment"][label]["enrichment_over_random_baseline"]
        for label in labels
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color="#54A24B")
    ax.axhline(1.0, color="0.4", linestyle="--", linewidth=1)
    ax.set_xlabel("Top-k")
    ax.set_ylabel("Enrichment over random baseline")
    ax.set_title("Matched OAS top-k enrichment")
    fig.tight_layout()
    fig.savefig(TOPK_FIGURE_PATH, dpi=200)
    plt.close(fig)


def main() -> int:
    matched, audit = build_matched_dataset()
    scores, metrics = run_model(matched, audit)
    write_json(AUDIT_METRICS_PATH, audit)
    write_text(AUDIT_PATH, build_audit_report(audit))
    write_json(METRICS_PATH, metrics)
    write_text(REPORT_PATH, build_retrieval_report(metrics))
    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(SCORES_PATH, index=False)
    save_score_distribution(scores)
    save_topk_figure(metrics)
    print(
        "matched_project_rows="
        f"{metrics['matched_project_row_count']}; "
        f"matched_oas_rows={metrics['matched_oas_row_count']}; "
        f"skipped_project_rows={metrics['skipped_project_rows_due_to_no_matched_oas_bin']}; "
        f"exact_overlap_count_removed={metrics['exact_overlap_count_removed']}; "
        f"roc_auc={metrics['roc_auc']:.4f}; pr_auc={metrics['pr_auc']:.4f}; "
        f"topk_enrichment={metrics['topk_enrichment']}; "
        f"outputs={metrics['artifacts']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
