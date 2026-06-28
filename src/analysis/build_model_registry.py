"""Build the final model registry and primary scorer selections."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from _safe_analysis_utils import PROJECT_ROOT, load_json, relpath, write_json, write_text


MATCHED_KMER_PATH = PROJECT_ROOT / "reports" / "metrics" / "matched_kmer_benchmark_audit.json"
LM_REGISTRY_PATH = PROJECT_ROOT / "reports" / "metrics" / "lm_benchmark_registry.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "model_registry.md"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics" / "model_registry.json"
KMER_MODEL_PATH = PROJECT_ROOT / "models" / "kmer_logreg_pair_text.joblib"


def metric_pair(result: dict[str, Any]) -> dict[str, float | None]:
    """Normalize metric names."""
    return {
        "roc_auc": result.get("roc_auc"),
        "pr_auc": result.get("average_precision", result.get("pr_auc")),
        "balanced_accuracy": result.get("balanced_accuracy"),
        "f1": result.get("f1"),
    }


def all_kmer_candidates(matched: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten all matched k-mer candidates."""
    candidates: list[dict[str, Any]] = []
    for block_id, block in matched.get("blocks", {}).items():
        split = block.get("split", {})
        for variant, result in block.get("results", {}).get("kmer_logreg", {}).items():
            candidates.append(
                {
                    "model_id": f"kmer_tfidf_logreg__{block_id}__{variant}",
                    "family": "compact_char_kmer_logreg",
                    "row_subset": block.get("row_subset"),
                    "block_id": block_id,
                    "input_variant": variant,
                    "row_count": block.get("row_count"),
                    "label_balance": block.get("label_counts"),
                    "split": split,
                    "metrics": metric_pair(result),
                    "artifact": relpath(KMER_MODEL_PATH) if KMER_MODEL_PATH.exists() else None,
                }
            )
    return candidates


def best_by_pr_auc(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Select the highest PR-AUC candidate, breaking ties by ROC-AUC."""
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            item["metrics"]["pr_auc"] if item["metrics"]["pr_auc"] is not None else -1,
            item["metrics"]["roc_auc"] if item["metrics"]["roc_auc"] is not None else -1,
        ),
    )


def choose_paired_region_scorer(matched: dict[str, Any]) -> dict[str, Any] | None:
    """Choose the best paired annotated compact k-mer scorer by matched metrics."""
    block = matched.get("blocks", {}).get("paired_annotated_subset")
    if not block:
        return None
    candidates = [
        candidate
        for candidate in all_kmer_candidates({"blocks": {"paired_annotated_subset": block}})
        if candidate["block_id"] == "paired_annotated_subset"
    ]
    selected = best_by_pr_auc(candidates)
    if selected:
        selected["artifact"] = None
        selected["selection_note"] = (
            "Selected as primary paired-region scorer by matched grouped validation on "
            "the paired annotated subset; region-only compact k-mer achieved the best "
            "paired annotated ROC-AUC and PR-AUC."
        )
    return selected


def build_registry() -> dict[str, Any]:
    """Build final model selection registry."""
    matched = load_json(MATCHED_KMER_PATH) or {}
    lm = load_json(LM_REGISTRY_PATH) or {}
    kmer_candidates = all_kmer_candidates(matched)
    full_candidates = [
        item for item in kmer_candidates if item["block_id"] == "full_strict_dataset"
    ]
    primary_broad = best_by_pr_auc(full_candidates)
    if primary_broad:
        primary_broad["model_id"] = "kmer_tfidf_logreg_pair_text"
        primary_broad["selection_note"] = (
            "Selected as broad primary scorer by matched grouped validation on the "
            "full strict labeled subset; simple model retained over neural benchmarks "
            "unless a same-subset result clearly wins."
        )
    paired_region = choose_paired_region_scorer(matched)
    best_pretrained = lm.get("best_pretrained_or_embedding_result")
    best_kmer = best_by_pr_auc(kmer_candidates)
    return {
        "status": "available" if primary_broad else "no_primary_broad_scorer",
        "primary_broad_scorer": primary_broad,
        "primary_paired_region_scorer": paired_region,
        "best_pretrained_model_result": best_pretrained,
        "best_kmer_result": best_kmer,
        "region_feature_comparison": matched.get("region_feature_comparison"),
        "pretrained_models_beat_matched_kmer_baselines": lm.get(
            "pretrained_models_beat_matched_kmer_baselines"
        ),
        "selection_rules": [
            "Use matched validation performance.",
            "Do not force pretrained models to win.",
            "Demote unstable neural models.",
            "Keep different row subsets separated.",
            "Prefer the simpler model when performance is practically tied.",
            "Exclude diagnostic error-analysis artifacts from pretrained-model selection.",
        ],
        "excluded_diagnostic_results": lm.get("excluded_diagnostic_results", []),
        "pretrained_selection_note": (
            "Actual pretrained and embedding benchmarks are retained as benchmark evidence only; "
            "none reliably replaces the matched k-mer references on both primary metrics."
        ),
        "invalid_cross_subset_comparisons": (
            "Full strict dataset metrics and paired annotated subset metrics are "
            "reported separately and are not treated as directly comparable."
        ),
    }


def metric_text(value: float | None) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def model_row(label: str, model: dict[str, Any] | None) -> str:
    """Format one model row."""
    if not model:
        return f"| {label} | unavailable | n/a | n/a | n/a | n/a | n/a |"
    split = model.get("split", {})
    metrics = model.get("metrics", {})
    return (
        f"| {label} | {model.get('model_id')} | {model.get('row_subset')} | "
        f"{model.get('row_count')} | {split.get('group_overlap_count')} | "
        f"{metric_text(metrics.get('roc_auc'))} | {metric_text(metrics.get('pr_auc'))} |"
    )


def build_report(registry: dict[str, Any]) -> str:
    """Build Markdown report."""
    lines = [
        "# Final Model Registry",
        "",
        "The registry separates full strict dataset comparisons from paired annotated",
        "subset comparisons. Metrics from different row subsets are kept separate.",
        "",
        "| Role | Model | Row subset | Rows | Group overlap | ROC-AUC | PR-AUC |",
        "|---|---|---|---:|---:|---:|---:|",
        model_row("Primary broad scorer", registry.get("primary_broad_scorer")),
        model_row("Primary paired/region scorer", registry.get("primary_paired_region_scorer")),
        model_row("Best k-mer result", registry.get("best_kmer_result")),
    ]
    best_pretrained = registry.get("best_pretrained_model_result") or {}
    lines.extend(
        [
            "",
            "## Best Pretrained/Embedding Benchmark",
            "",
            "| Model/result | Row subset | Rows | ROC-AUC | PR-AUC | Beats matched k-mer |",
            "|---|---|---:|---:|---:|---|",
            (
                f"| {best_pretrained.get('model_id', 'unavailable')} | "
                f"`{best_pretrained.get('row_subset', 'n/a')}` | "
                f"{best_pretrained.get('row_count', 'n/a')} | "
                f"{metric_text((best_pretrained.get('metrics') or {}).get('roc_auc'))} | "
                f"{metric_text((best_pretrained.get('metrics') or {}).get('pr_auc'))} | "
                f"{best_pretrained.get('beats_matched_kmer_reference')} |"
            ),
            "",
            "## Selection Rationale",
            "",
        ]
    )
    for rule in registry["selection_rules"]:
        lines.append(f"- {rule}")
    lines.extend(
        [
            "",
            registry["invalid_cross_subset_comparisons"],
            "",
            registry["pretrained_selection_note"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    registry = build_registry()
    write_json(METRICS_PATH, registry)
    write_text(REPORT_PATH, build_report(registry))
    primary = registry.get("primary_broad_scorer") or {}
    paired = registry.get("primary_paired_region_scorer") or {}
    print(
        "Model registry complete: "
        f"primary={primary.get('model_id', 'unavailable')}, "
        f"paired={paired.get('model_id', 'unavailable')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
