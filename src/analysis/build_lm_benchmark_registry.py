"""Build a registry of pretrained/embedding sequence-model benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from _safe_analysis_utils import PROJECT_ROOT, load_json, relpath, write_json, write_text


METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"
MATCHED_KMER_PATH = METRICS_DIR / "matched_kmer_benchmark_audit.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "lm_benchmark_registry.md"
REGISTRY_PATH = METRICS_DIR / "lm_benchmark_registry.json"


SOURCE_FILES = {
    "embedding_baseline": "embedding_baseline_metrics.json",
    "pytorch_embedding_mlp": "pytorch_embedding_mlp_metrics.json",
    "pretrained_frozen_baseline": "pretrained_frozen_baseline_metrics.json",
    "pretrained_finetune": "pretrained_finetune_metrics.json",
    "pretrained_finetune_seed_check": "pretrained_finetune_seed_check_metrics.json",
    "pretrained_lora_distilled": "pretrained_lora_distilled_metrics.json",
    "bioaware_igbert_final": "bioaware_igbert_final_metrics.json",
    "hybrid_baseline": "hybrid_baseline_metrics.json",
}


def scalar_metric(value: Any) -> float | None:
    """Extract a scalar metric from raw or aggregate metric payloads."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for key in ["mean", "value", "max"]:
            if key in value and isinstance(value[key], (int, float)):
                return float(value[key])
    return None


def metric_pair_from_payload(payload: dict[str, Any]) -> dict[str, float | None]:
    """Extract ROC-AUC and PR-AUC-like metrics from a dict."""
    roc = scalar_metric(payload.get("roc_auc"))
    pr = scalar_metric(payload.get("pr_auc"))
    if pr is None:
        pr = scalar_metric(payload.get("average_precision"))
    return {"roc_auc": roc, "pr_auc": pr}


def split_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Return aggregate split context."""
    split = payload.get("split", {})
    if "train_test" in split:
        split = split.get("train_test", {})
    return {
        "split_strategy": split.get("strategy")
        or split.get("split_strategy")
        or split.get("group_column")
        or payload.get("test_size")
        or "reported_in_source_metrics",
        "group_overlap_count": split.get("group_overlap_count"),
        "train_label_counts": split.get("train_label_counts"),
        "test_label_counts": split.get("test_label_counts"),
        "train_size": split.get("train_size"),
        "test_size": split.get("test_size") or split.get("test_size_rows"),
    }


def flatten_metric_candidates(obj: Any, path: str = "") -> list[dict[str, Any]]:
    """Find metric dictionaries recursively, excluding obvious k-mer baselines."""
    records: list[dict[str, Any]] = []
    if isinstance(obj, dict):
        metrics = metric_pair_from_payload(obj)
        if metrics["roc_auc"] is not None or metrics["pr_auc"] is not None:
            lower_path = path.lower()
            if "baseline" not in lower_path and "kmer" not in lower_path and "majority" not in lower_path:
                records.append({"path": path.strip("."), "metrics": metrics})
        for key, value in obj.items():
            records.extend(flatten_metric_candidates(value, f"{path}.{key}" if path else str(key)))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            records.extend(flatten_metric_candidates(value, f"{path}.{index}"))
    return records


def best_candidate(payload: dict[str, Any]) -> dict[str, Any]:
    """Select a best available metric candidate within one source file."""
    aggregate = payload.get("aggregate")
    if isinstance(aggregate, dict):
        metrics = metric_pair_from_payload(aggregate)
        if metrics["roc_auc"] is not None or metrics["pr_auc"] is not None:
            return {"path": "aggregate", "metrics": metrics}

    for key in ["best_grouped_average_precision", "best_grouped_roc_auc"]:
        item = payload.get(key)
        if isinstance(item, dict):
            metrics = metric_pair_from_payload(item.get("metrics", item))
            if metrics["roc_auc"] is None and key in payload:
                metrics["roc_auc"] = scalar_metric(payload.get("best_grouped_roc_auc", {}).get("value"))
            if metrics["pr_auc"] is None and key == "best_grouped_average_precision":
                metrics["pr_auc"] = scalar_metric(item.get("value"))
            if metrics["roc_auc"] is not None or metrics["pr_auc"] is not None:
                return {"path": key, "metrics": metrics}

    candidates = flatten_metric_candidates(payload)
    if not candidates:
        return {"path": None, "metrics": {"roc_auc": None, "pr_auc": None}}
    return max(
        candidates,
        key=lambda item: (
            item["metrics"]["pr_auc"] if item["metrics"]["pr_auc"] is not None else -1,
            item["metrics"]["roc_auc"] if item["metrics"]["roc_auc"] is not None else -1,
        ),
    )


def matched_references() -> dict[str, Any]:
    """Load matched k-mer references for same-subset comparisons."""
    matched = load_json(MATCHED_KMER_PATH) or {}
    refs: dict[str, Any] = {}
    for block_id, block in matched.get("blocks", {}).items():
        result = block.get("results", {}).get("kmer_logreg", {}).get("whole_pair_compact_kmer")
        if result:
            refs[block_id] = {
                "row_count": block.get("row_count"),
                "row_subset": block.get("row_subset"),
                "metrics": {
                    "roc_auc": result.get("roc_auc"),
                    "pr_auc": result.get("average_precision"),
                },
            }
    return refs


def reference_for_row_count(row_count: int | None, refs: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    """Find a same-row-count matched reference."""
    if row_count is None:
        return None, None
    for ref_id, ref in refs.items():
        if ref.get("row_count") == row_count:
            return ref_id, ref
    return None, None


def build_entry(model_id: str, filename: str, refs: dict[str, Any]) -> dict[str, Any]:
    """Build one registry entry from a source metric JSON."""
    path = METRICS_DIR / filename
    payload = load_json(path) or {}
    candidate = best_candidate(payload)
    row_count = payload.get("row_count")
    ref_id, ref = reference_for_row_count(row_count, refs)
    metrics = candidate["metrics"]
    beats = None
    if ref and metrics["roc_auc"] is not None and metrics["pr_auc"] is not None:
        beats = {
            "roc_auc": bool(metrics["roc_auc"] > ref["metrics"]["roc_auc"]),
            "pr_auc": bool(metrics["pr_auc"] > ref["metrics"]["pr_auc"]),
            "both_primary_metrics": bool(
                metrics["roc_auc"] > ref["metrics"]["roc_auc"]
                and metrics["pr_auc"] > ref["metrics"]["pr_auc"]
            ),
        }
    entry = {
        "model_id": model_id,
        "source_metrics_file": relpath(path),
        "status": payload.get("status", "available" if path.exists() else "missing"),
        "row_subset": payload.get("input_path")
        or payload.get("metadata_path")
        or "reported_in_source_metrics",
        "row_count": row_count,
        "label_balance": payload.get("label_counts"),
        "split": split_summary(payload),
        "selected_metric_path": candidate["path"],
        "metrics": metrics,
        "matched_kmer_reference_id": ref_id,
        "matched_kmer_reference": ref,
        "beats_matched_kmer_reference": beats,
        "notes": "Benchmark evidence only; not automatically selected as primary scorer.",
    }
    return entry


def build_registry() -> dict[str, Any]:
    """Build the full benchmark registry."""
    refs = matched_references()
    entries = [build_entry(model_id, filename, refs) for model_id, filename in SOURCE_FILES.items()]
    valid_pretrained = [
        entry
        for entry in entries
        if entry["metrics"]["roc_auc"] is not None or entry["metrics"]["pr_auc"] is not None
    ]
    best = None
    if valid_pretrained:
        best = max(
            valid_pretrained,
            key=lambda entry: (
                entry["metrics"]["pr_auc"] if entry["metrics"]["pr_auc"] is not None else -1,
                entry["metrics"]["roc_auc"] if entry["metrics"]["roc_auc"] is not None else -1,
            ),
        )
    any_beats = any(
        bool(entry["beats_matched_kmer_reference"].get("both_primary_metrics"))
        for entry in entries
        if isinstance(entry.get("beats_matched_kmer_reference"), dict)
    )
    return {
        "status": "available",
        "matched_kmer_references": refs,
        "entries": entries,
        "best_pretrained_or_embedding_result": best,
        "pretrained_models_beat_matched_kmer_baselines": bool(any_beats),
        "excluded_diagnostic_results": [
            {
                "model_id": "model_error_analysis",
                "reason": "Diagnostic error-analysis artifact, not a pretrained or language-model benchmark.",
            }
        ],
        "interpretation": (
            "Pretrained and embedding models are benchmark evidence, not automatically "
            "primary scorers. None reliably replaces the matched k-mer references on both primary metrics."
        ),
    }


def metric_text(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def build_report(registry: dict[str, Any]) -> str:
    """Build Markdown report."""
    lines = [
        "# Pretrained Sequence-Model Benchmark Registry",
        "",
        "This registry uses existing metric files only. It does not rerun neural",
        "training. Each row states the reported subset, split context, label",
        "balance, and whether a same-row-count matched k-mer reference was beaten.",
        "",
        "| Model/result | Row subset | Row count | Split | Group overlap | ROC-AUC | PR-AUC | Beats matched k-mer |",
        "|---|---|---:|---|---:|---:|---:|---|",
    ]
    for entry in registry["entries"]:
        split = entry["split"]
        beats = entry["beats_matched_kmer_reference"]
        if isinstance(beats, dict):
            beat_text = str(beats["both_primary_metrics"]).lower()
        else:
            beat_text = "not same-subset comparable"
        lines.append(
            f"| {entry['model_id']} | `{entry['row_subset']}` | "
            f"{entry.get('row_count') or 'n/a'} | {split['split_strategy']} | "
            f"{split.get('group_overlap_count')} | "
            f"{metric_text(entry['metrics']['roc_auc'])} | "
            f"{metric_text(entry['metrics']['pr_auc'])} | {beat_text} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            registry["interpretation"],
            "",
            "Invalid cross-subset comparisons are not used for model selection.",
            "",
            "Diagnostic error-analysis artifacts are excluded from pretrained-model selection.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    registry = build_registry()
    write_json(REGISTRY_PATH, registry)
    write_text(REPORT_PATH, build_report(registry))
    best = registry.get("best_pretrained_or_embedding_result") or {}
    print(
        "LM benchmark registry complete: "
        f"entries={len(registry['entries'])}, "
        f"best={best.get('model_id', 'n/a')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
