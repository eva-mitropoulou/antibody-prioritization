"""Run retrospective selection-loop simulations on existing scored records."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/antibody_prioritization_mplconfig")
import matplotlib.pyplot as plt

from _safe_analysis_utils import (
    PROJECT_ROOT,
    label_series,
    read_csv_text,
    safe_output_columns,
    value_counts_dict,
    write_json,
    write_text,
)


INPUT_PATH = PROJECT_ROOT / "reports" / "broader_existing_record_prioritization_table.csv"
CLUSTER_PATH = PROJECT_ROOT / "reports" / "unsupervised_antibody_clusters.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "active_learning_simulation_report.md"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics" / "active_learning_simulation_metrics.json"
SELECTED_PATH = PROJECT_ROOT / "reports" / "active_learning_selected_records.csv"
FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "active_learning_strategy_hits.png"

RANDOM_STATE = 42
RANDOM_REPEATS = 100
BUDGETS = [25, 50, 100, 200, 500]


def probability_column(data: pd.DataFrame) -> str:
    """Return the preferred score column."""
    for column in ["primary_probability", "predicted_neutralisation_probability", "kmer_probability"]:
        if column in data.columns:
            return column
    raise ValueError("No model probability column found.")


def prepare_data() -> tuple[pd.DataFrame, str]:
    """Load labeled scored records and optional clusters."""
    data = read_csv_text(INPUT_PATH)
    score_column = probability_column(data)
    data[score_column] = pd.to_numeric(data[score_column], errors="coerce")
    data["label_numeric"] = label_series(data)
    data = data[data["label_numeric"].isin([0, 1]) & data[score_column].notna()].copy()
    data["label_numeric"] = data["label_numeric"].astype(int)
    if CLUSTER_PATH.exists() and "sample_name" in data.columns:
        clusters = read_csv_text(CLUSTER_PATH)
        keep = [column for column in ["sample_name", "cluster_id"] if column in clusters.columns]
        if len(keep) == 2:
            clusters = clusters[keep].drop_duplicates("sample_name")
            data = data.merge(clusters, on="sample_name", how="left", suffixes=("", "_cluster"))
    return data.reset_index(drop=True), score_column


def round_robin_groups(data: pd.DataFrame, group_column: str, score_column: str) -> pd.DataFrame:
    """Order records by cycling through groups with high scores first."""
    if group_column not in data.columns:
        return data.sort_values(score_column, ascending=False, kind="mergesort")
    temp = data.copy()
    temp[group_column] = temp[group_column].fillna("missing").astype(str)
    temp = temp.sort_values([group_column, score_column], ascending=[True, False], kind="mergesort")
    temp["_rank_in_group"] = temp.groupby(group_column).cumcount()
    return temp.sort_values(["_rank_in_group", score_column], ascending=[True, False], kind="mergesort")


def strategy_order(data: pd.DataFrame, strategy: str, score_column: str) -> pd.DataFrame:
    """Return ordered candidates for a deterministic strategy."""
    if strategy == "highest_score":
        return data.sort_values(score_column, ascending=False, kind="mergesort")
    if strategy == "uncertainty":
        temp = data.copy()
        temp["_uncertainty"] = (temp[score_column] - 0.5).abs()
        return temp.sort_values(["_uncertainty", score_column], ascending=[True, False], kind="mergesort")
    if strategy == "diversity_aware_high_score":
        return round_robin_groups(data, "diversity_group", score_column)
    if strategy == "target_region_stratified_high_score":
        return round_robin_groups(data, "target_region_group", score_column)
    if strategy == "cluster_aware_high_score":
        return round_robin_groups(data.dropna(subset=["cluster_id"]), "cluster_id", score_column)
    raise ValueError(f"Unknown strategy: {strategy}")


def evaluate_ordered(data: pd.DataFrame, strategy: str, ordered: pd.DataFrame, budgets: list[int]) -> dict[str, Any]:
    """Evaluate cumulative hit rates for an ordered list."""
    budget_metrics = {}
    for budget in budgets:
        selected = ordered.head(min(budget, len(ordered)))
        positives = int(selected["label_numeric"].eq(1).sum())
        selected_count = int(len(selected))
        budget_metrics[str(budget)] = {
            "selected_count": selected_count,
            "positive_count": positives,
            "precision": float(positives / selected_count) if selected_count else None,
        }
    return {
        "strategy": strategy,
        "budget_metrics": budget_metrics,
        "available_candidate_count": int(len(ordered)),
    }


def evaluate_random(data: pd.DataFrame, budgets: list[int]) -> dict[str, Any]:
    """Evaluate random selection with repeats."""
    rng = np.random.default_rng(RANDOM_STATE)
    metrics = {}
    for budget in budgets:
        selected_count = min(budget, len(data))
        positives = []
        precisions = []
        for _ in range(RANDOM_REPEATS):
            indices = rng.choice(data.index.to_numpy(), size=selected_count, replace=False)
            labels = data.loc[indices, "label_numeric"]
            hits = int(labels.eq(1).sum())
            positives.append(hits)
            precisions.append(hits / selected_count if selected_count else 0.0)
        metrics[str(budget)] = {
            "selected_count": int(selected_count),
            "positive_count_mean": float(np.mean(positives)),
            "positive_count_std": float(np.std(positives, ddof=1)),
            "precision_mean": float(np.mean(precisions)),
            "precision_std": float(np.std(precisions, ddof=1)),
        }
    return {
        "strategy": "random",
        "repeats": RANDOM_REPEATS,
        "budget_metrics": metrics,
        "available_candidate_count": int(len(data)),
    }


def selected_records_for_best(data: pd.DataFrame, score_column: str, strategy: str, budget: int) -> pd.DataFrame:
    """Return the selected records for the winning deterministic strategy."""
    ordered = strategy_order(data, strategy, score_column).head(budget).copy()
    ordered.insert(0, "selection_rank", np.arange(1, len(ordered) + 1, dtype=int))
    ordered.insert(0, "selection_strategy", strategy)
    keep = [
        "selection_strategy",
        "selection_rank",
        "row_id",
        "sample_name",
        "label_numeric",
        score_column,
        "record_category",
        "target_region_group",
        "paired_light_status",
        "confidence_bin",
        "developability_risk_bin",
        "has_structure",
        "cluster_id",
    ]
    keep = [column for column in keep if column in ordered.columns]
    return ordered[keep]


def build_metrics() -> tuple[dict[str, Any], pd.DataFrame]:
    """Run all strategies."""
    data, score_column = prepare_data()
    budgets = [budget for budget in BUDGETS if budget <= len(data)]
    if not budgets:
        budgets = [len(data)]
    random_result = evaluate_random(data, budgets)
    strategies = ["highest_score", "uncertainty", "diversity_aware_high_score"]
    if "target_region_group" in data.columns and data["target_region_group"].nunique(dropna=True) > 1:
        strategies.append("target_region_stratified_high_score")
    if "cluster_id" in data.columns and data["cluster_id"].notna().sum() >= min(budgets):
        strategies.append("cluster_aware_high_score")
    results = {"random": random_result}
    for strategy in strategies:
        ordered = strategy_order(data, strategy, score_column)
        results[strategy] = evaluate_ordered(data, strategy, ordered, budgets)
    largest_budget = str(max(budgets))
    random_hits = random_result["budget_metrics"][largest_budget]["positive_count_mean"]
    deterministic = {
        strategy: result["budget_metrics"][largest_budget]["positive_count"]
        for strategy, result in results.items()
        if strategy != "random"
    }
    best_strategy = max(deterministic, key=deterministic.get)
    best_hits = deterministic[best_strategy]
    selected = selected_records_for_best(data, score_column, best_strategy, int(largest_budget))
    metrics = {
        "status": "available",
        "input_path": str(INPUT_PATH.relative_to(PROJECT_ROOT)),
        "row_subset": "broader prioritization table, labeled records only",
        "split_strategy": "retrospective ranking simulation; no model refit",
        "label_balance": {
            "0": int(data["label_numeric"].eq(0).sum()),
            "1": int(data["label_numeric"].eq(1).sum()),
        },
        "candidate_count": int(len(data)),
        "score_column": score_column,
        "budgets": budgets,
        "strategy_results": results,
        "best_strategy": best_strategy,
        "best_strategy_positive_count_at_largest_budget": int(best_hits),
        "random_positive_count_mean_at_largest_budget": float(random_hits),
        "best_strategy_beats_random_mean": bool(best_hits > random_hits),
        "available_strategy_count": int(len(results)),
        "artifacts": {
            "report": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
            "metrics_json": str(METRICS_PATH.relative_to(PROJECT_ROOT)),
            "selected_records_csv": str(SELECTED_PATH.relative_to(PROJECT_ROOT)),
            "figure": str(FIGURE_PATH.relative_to(PROJECT_ROOT)),
        },
    }
    return metrics, selected


def save_figure(metrics: dict[str, Any]) -> None:
    """Save positives selected by strategy at largest budget."""
    largest_budget = str(max(metrics["budgets"]))
    rows = []
    for strategy, result in metrics["strategy_results"].items():
        item = result["budget_metrics"][largest_budget]
        hits = item.get("positive_count", item.get("positive_count_mean"))
        rows.append({"strategy": strategy, "positive_count": hits})
    table = pd.DataFrame(rows)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(table["strategy"], table["positive_count"], color="#4C78A8")
    ax.set_xlabel("Positive labels selected")
    ax.set_title(f"Retrospective selection at budget {largest_budget}")
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=200)
    plt.close(fig)


def build_report(metrics: dict[str, Any]) -> str:
    """Build Markdown report."""
    largest_budget = str(max(metrics["budgets"]))
    lines = [
        "# Retrospective Selection-Loop Simulation",
        "",
        "This simulation compares selection strategies on existing labeled records.",
        "It is retrospective and does not claim prospective efficacy.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Candidate records | {metrics['candidate_count']} |",
        f"| Label 0 count | {metrics['label_balance']['0']} |",
        f"| Label 1 count | {metrics['label_balance']['1']} |",
        f"| Strategies evaluated | {metrics['available_strategy_count']} |",
        f"| Best strategy | {metrics['best_strategy']} |",
        f"| Best beats random mean | {str(metrics['best_strategy_beats_random_mean']).lower()} |",
        "",
        "## Strategy Results",
        "",
        "| Strategy | Selected at largest budget | Positive labels | Precision |",
        "|---|---:|---:|---:|",
    ]
    for strategy, result in metrics["strategy_results"].items():
        item = result["budget_metrics"][largest_budget]
        if strategy == "random":
            positives = item["positive_count_mean"]
            precision = item["precision_mean"]
        else:
            positives = item["positive_count"]
            precision = item["precision"]
        lines.append(
            f"| {strategy} | {item['selected_count']} | {float(positives):.2f} | "
            f"{float(precision):.4f} |"
        )
    if not metrics["best_strategy_beats_random_mean"]:
        lines.extend(
            [
                "",
                "The best informed strategy did not exceed the random mean at the",
                "largest evaluated budget; this is reported without adjustment.",
            ]
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- `{METRICS_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{SELECTED_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{FIGURE_PATH.relative_to(PROJECT_ROOT)}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    metrics, selected = build_metrics()
    SELECTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(SELECTED_PATH, index=False)
    write_json(METRICS_PATH, metrics)
    write_text(REPORT_PATH, build_report(metrics))
    save_figure(metrics)
    print(
        "Active-learning simulation complete: "
        f"best={metrics['best_strategy']}, "
        f"beats_random={metrics['best_strategy_beats_random_mean']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
