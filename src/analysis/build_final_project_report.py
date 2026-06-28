"""Build the final project report from existing metric artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from _safe_analysis_utils import PROJECT_ROOT, load_json, write_text


REPORT_PATH = PROJECT_ROOT / "reports" / "final_project_report.md"

METRICS = {
    "core": PROJECT_ROOT / "reports" / "metrics" / "core_dataset_audit.json",
    "annotation": PROJECT_ROOT / "reports" / "metrics" / "domain_region_annotation_summary.json",
    "target": PROJECT_ROOT / "reports" / "metrics" / "target_region_epitope_analysis.json",
    "matched": PROJECT_ROOT / "reports" / "metrics" / "matched_kmer_benchmark_audit.json",
    "lm": PROJECT_ROOT / "reports" / "metrics" / "lm_benchmark_registry.json",
    "model": PROJECT_ROOT / "reports" / "metrics" / "model_registry.json",
    "prioritization": PROJECT_ROOT / "reports" / "metrics" / "broader_existing_record_prioritization_summary.json",
    "shortlist": PROJECT_ROOT / "reports" / "metrics" / "diversity_aware_shortlist_summary.json",
    "background": PROJECT_ROOT / "reports" / "metrics" / "oas_background_retrieval_metrics.json",
    "matched_background": PROJECT_ROOT
    / "reports"
    / "metrics"
    / "oas_matched_background_retrieval_metrics.json",
    "source_holdout": PROJECT_ROOT
    / "reports"
    / "metrics"
    / "source_holdout_validation_metrics.json",
    "calibration": PROJECT_ROOT / "reports" / "metrics" / "calibration_threshold_metrics.json",
    "source_robust": PROJECT_ROOT
    / "reports"
    / "metrics"
    / "source_robust_model_selection_metrics.json",
    "landscape": PROJECT_ROOT / "reports" / "metrics" / "unsupervised_antibody_landscape_metrics.json",
    "active": PROJECT_ROOT / "reports" / "metrics" / "active_learning_simulation_metrics.json",
    "sensitivity": PROJECT_ROOT / "reports" / "metrics" / "biological_sensitivity_metrics.json",
    "structure": PROJECT_ROOT / "reports" / "metrics" / "structure_metadata_summary.json",
}


def metric_text(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def load_all() -> dict[str, Any]:
    return {key: load_json(path) or {} for key, path in METRICS.items()}


def model_summary(model: dict[str, Any] | None) -> str:
    if not model:
        return "unavailable"
    metrics = model.get("metrics", {})
    return (
        f"{model.get('model_id')} on {model.get('row_subset')} "
        f"(ROC-AUC {metric_text(metrics.get('roc_auc'))}, PR-AUC {metric_text(metrics.get('pr_auc'))})"
    )


def oas_interpretation(
    broad_background: dict[str, Any],
    matched_background: dict[str, Any],
) -> str:
    """Return conservative OAS retrieval interpretation text."""
    matched_roc = matched_background.get("roc_auc")
    matched_pr = matched_background.get("pr_auc")
    broad_roc = broad_background.get("roc_auc")
    broad_pr = broad_background.get("pr_auc")
    if isinstance(matched_roc, (int, float)) and isinstance(matched_pr, (int, float)):
        if matched_roc >= 0.9 and matched_pr >= 0.9:
            return (
                "OAS records are unknown-target natural antibody background. "
                "The OAS retrieval task is a background/enrichment diagnostic kept "
                "separate from the main neutralisation benchmark. High OAS retrieval "
                "separability likely reflects source/domain differences between project "
                "records and natural repertoire background."
            )
        if (
            isinstance(broad_roc, (int, float))
            and isinstance(broad_pr, (int, float))
            and (broad_roc - matched_roc >= 0.10 or broad_pr - matched_pr >= 0.10)
        ):
            return (
                "The original OAS retrieval task was partly driven by dataset/source "
                "differences; matched-background retrieval is the more conservative estimate. "
                "OAS records remain unknown-target background for enrichment analysis."
            )
    return (
        "Matched OAS retrieval is reported as a conservative background-control "
        "diagnostic separate from the main neutralisation benchmark. OAS records remain "
        "unknown-target background for enrichment analysis."
    )


def source_generalization_interpretation(source_holdout: dict[str, Any]) -> str:
    """Summarize whether source-holdout validation supports generalization."""
    aggregate = source_holdout.get("leave_source_out", {}).get("aggregate", {})
    macro = aggregate.get("macro_mean", {})
    valid_groups = aggregate.get("valid_heldout_source_group_count", 0)
    roc = macro.get("roc_auc")
    pr = macro.get("pr_auc")
    if valid_groups and isinstance(roc, (int, float)) and isinstance(pr, (int, float)):
        if roc >= 0.7 and pr >= 0.7:
            return (
                "Leave-source-out validation supports some cross-source generalization, "
                "but source/study effects may still exist because public labels and "
                "record construction are heterogeneous."
            )
        return (
            "Leave-source-out validation is weaker than the matched grouped benchmark, "
            "so source/study effects may materially affect apparent model performance."
        )
    return (
        "Leave-source-out validation was limited by source group structure; the "
        "source-grouped fallback is the main skeptical validation result."
    )


def calibration_interpretation(calibration: dict[str, Any]) -> str:
    """Summarize calibration quality and threshold use."""
    brier = calibration.get("brier_score")
    ece = calibration.get("expected_calibration_error")
    if isinstance(brier, (int, float)) and isinstance(ece, (int, float)):
        if brier <= 0.18 and ece <= 0.08:
            return (
                "Scores are reasonably calibrated for prioritization, but threshold "
                "choices should still be tied to review capacity and false-positive tolerance."
            )
        return (
            "Scores are more reliable for ranking than as absolute probabilities; "
            "thresholds should be treated as review cutoffs rather than calibrated risk estimates."
        )
    return "Calibration diagnostics were unavailable or incomplete."


def source_robust_interpretation(source_robust: dict[str, Any]) -> str:
    """Summarize source-robust model-selection outcome."""
    selection = source_robust.get("model_selection", {})
    selected = selection.get("selected_model")
    if not selected:
        return "Source-robust model selection was unavailable or inconclusive."
    if selection.get("meaningful_improvement_over_previous"):
        improvement = "improved cross-source performance meaningfully"
    else:
        improvement = "did not materially improve cross-source performance"
    cdr_results = []
    for model_id in ["cdr_region_kmer", "whole_plus_cdr_kmer"]:
        holdout = source_robust.get("holdout_results", {}).get(model_id, {})
        weighted = holdout.get("aggregate", {}).get("weighted_mean_by_test_size", {})
        if weighted.get("pr_auc") is not None:
            cdr_results.append(weighted["pr_auc"])
    selected_weighted = (
        source_robust.get("holdout_results", {})
        .get(selected, {})
        .get("aggregate", {})
        .get("weighted_mean_by_test_size", {})
    )
    if cdr_results and selected_weighted.get("pr_auc") is not None and max(cdr_results) >= selected_weighted["pr_auc"] - 0.005:
        cdr_text = "CDR/region features were competitive for source robustness."
    else:
        cdr_text = "CDR/region features did not clearly improve source robustness."
    return (
        f"Source-robust selection chose `{selected}` and {improvement}. "
        f"{cdr_text} Scores remain ranking and prioritization signals for existing records."
    )


def source_robust_threshold_summary(source_robust: dict[str, Any]) -> str:
    """Summarize selected source-robust threshold evidence."""
    selected = source_robust.get("model_selection", {}).get("selected_model")
    if not selected:
        return "Source-robust threshold summary unavailable."
    calibration = source_robust.get("calibration_results", {}).get(selected, {})
    threshold = calibration.get("best_high_confidence_threshold", {})
    weighted = (
        source_robust.get("holdout_results", {})
        .get(selected, {})
        .get("aggregate", {})
        .get("weighted_mean_by_test_size", {})
    )
    return (
        f"Selected weighted leave-source-out ROC-AUC: "
        f"{metric_text(weighted.get('roc_auc'))}. Selected weighted leave-source-out "
        f"PR-AUC: {metric_text(weighted.get('pr_auc'))}. "
        f"High-confidence threshold: {metric_text(threshold.get('threshold'))} "
        f"with precision {metric_text(threshold.get('precision'))}, recall "
        f"{metric_text(threshold.get('recall'))}, and coverage "
        f"{metric_text(threshold.get('coverage_fraction'))}."
    )


def build_report(data: dict[str, Any]) -> str:
    core = data["core"]
    model = data["model"]
    matched = data["matched"]
    lm = data["lm"]
    prioritization = data["prioritization"]
    shortlist = data["shortlist"]
    landscape = data["landscape"]
    active = data["active"]
    target = data["target"]
    background = data["background"]
    matched_background = data["matched_background"]
    source_holdout = data["source_holdout"]
    calibration = data["calibration"]
    source_robust = data["source_robust"]
    structure = data["structure"]
    strict = core.get("strict_labeled_dataset", {})
    broader = core.get("broader_prepared_dataset", {})
    annotation = data["annotation"]
    region_comparison = matched.get("region_feature_comparison", {})
    lines = [
        "# Final Project Report",
        "",
        "## Project Goal",
        "",
        (
            "Build a public-data antibody sequence-record ML workflow for neutralisation "
            "classification benchmarking, sequence-space analysis, retrospective "
            "selection simulation, and prioritization of existing records."
        ),
        "",
        "The workflow preserves source sequence fields and supports retrospective",
        "benchmarking, review prioritization, and background retrieval analysis.",
        "",
        "## Datasets",
        "",
        "| Dataset | Rows | Columns | Label 0 | Label 1 |",
        "|---|---:|---:|---:|---:|",
        (
            f"| Strict labeled ML table | {strict.get('shape', [0, 0])[0]} | "
            f"{strict.get('shape', [0, 0])[1]} | {strict.get('label_counts', {}).get('0', 0)} | "
            f"{strict.get('label_counts', {}).get('1', 0)} |"
        ),
        (
            f"| Broader prepared table | {broader.get('shape', [0, 0])[0]} | "
            f"{broader.get('shape', [0, 0])[1]} | {broader.get('label_counts', {}).get('0', 0)} | "
            f"{broader.get('label_counts', {}).get('1', 0)} |"
        ),
        "",
        "## Cleaning And Labels",
        "",
        (
            "Labels were used as existing binary record metadata. Missing-label and "
            "conflict-label records were preserved for prioritization rather than "
            "discarded from the broader scoring table."
        ),
        "",
        "## Paired/Light-Missing Handling",
        "",
        (
            f"Strict paired/light-missing counts: {strict.get('paired_light_status_counts', {})}. "
            f"Broader paired/light-missing counts: {broader.get('paired_light_status_counts', {})}."
        ),
        "",
        "## Domain-Region Annotation",
        "",
        (
            f"Annotation status: {annotation.get('status', 'missing')}. "
            f"Paired annotated rows: {annotation.get('paired_row_count', 'n/a')}. "
            f"Single-chain/light-missing rows: "
            f"{annotation.get('single_chain_or_light_missing_row_count', 'n/a')}."
        ),
        "",
        "## Target-Region Metadata Analysis",
        "",
        (
            f"Target-region analysis status: {target.get('status', 'missing')}. "
            f"Unknown broader count: {target.get('broader_unknown_count', 'n/a')}. "
            f"Useful for subgroup analysis: "
            f"{target.get('target_region_metadata_useful_for_subgroup_analysis', 'n/a')}."
        ),
        "",
        "## Matched Benchmark Results",
        "",
        (
            "Matched k-mer baselines used compact character strings, grouped splits, "
            "zero group overlap, and separate full strict versus paired annotated subsets."
        ),
        "",
        (
            "Region features improved paired matched ROC-AUC: "
            f"{region_comparison.get('region_features_improved_roc_auc', 'n/a')}; "
            "improved paired matched PR-AUC: "
            f"{region_comparison.get('region_features_improved_pr_auc', 'n/a')}."
        ),
        "",
        "## Pretrained Sequence-Model Benchmarks",
        "",
        (
            "Pretrained and embedding models were treated as benchmark evidence, not "
            "automatic primary scorers. Same-row-count matched k-mer comparisons were "
            "used when available."
        ),
        "",
        (
            f"Pretrained models beat matched k-mer baselines on both primary metrics: "
            f"{lm.get('pretrained_models_beat_matched_kmer_baselines', 'n/a')}."
        ),
        "",
        "## Final Model Selection",
        "",
        f"Primary broad scorer: {model_summary(model.get('primary_broad_scorer'))}.",
        "",
        f"Primary paired/region scorer: {model_summary(model.get('primary_paired_region_scorer'))}.",
        "",
        "## Skeptical Validation Controls",
        "",
        "### Leave-Source/Leave-Study-Out Validation",
        "",
        (
            f"Detected source groups: "
            f"{source_holdout.get('source_diagnostics', {}).get('source_group_count', 'n/a')}. "
            f"Valid held-out source groups: "
            f"{source_holdout.get('leave_source_out', {}).get('aggregate', {}).get('valid_heldout_source_group_count', 'n/a')}. "
            f"Macro source-holdout ROC-AUC: "
            f"{metric_text(source_holdout.get('leave_source_out', {}).get('aggregate', {}).get('macro_mean', {}).get('roc_auc'))}. "
            f"Macro source-holdout PR-AUC: "
            f"{metric_text(source_holdout.get('leave_source_out', {}).get('aggregate', {}).get('macro_mean', {}).get('pr_auc'))}."
        ),
        "",
        (
            f"Source-grouped fallback ROC-AUC: "
            f"{metric_text(source_holdout.get('source_grouped_fallback', {}).get('metrics', {}).get('roc_auc'))}. "
            f"Source-grouped fallback PR-AUC: "
            f"{metric_text(source_holdout.get('source_grouped_fallback', {}).get('metrics', {}).get('pr_auc'))}."
        ),
        "",
        source_generalization_interpretation(source_holdout),
        "",
        "### Calibration And Threshold Analysis",
        "",
        (
            f"Brier score: {metric_text(calibration.get('brier_score'))}. "
            f"Expected calibration error: "
            f"{metric_text(calibration.get('expected_calibration_error'))}. "
            f"High-confidence review threshold: "
            f"{metric_text(calibration.get('best_high_confidence_threshold', {}).get('threshold'))} "
            f"with precision "
            f"{metric_text(calibration.get('best_high_confidence_threshold', {}).get('precision'))} "
            f"and recall "
            f"{metric_text(calibration.get('best_high_confidence_threshold', {}).get('recall'))}."
        ),
        "",
        calibration_interpretation(calibration),
        "",
        "### Source-Robust Model Selection",
        "",
        (
            f"Selected source-robust model: "
            f"{source_robust.get('model_selection', {}).get('selected_model', 'n/a')}. "
            f"Meaningful improvement over previous source-holdout baseline: "
            f"{source_robust.get('model_selection', {}).get('meaningful_improvement_over_previous', 'n/a')}."
        ),
        "",
        source_robust_interpretation(source_robust),
        "",
        source_robust_threshold_summary(source_robust),
        "",
        "## Existing-Record Prioritization",
        "",
        (
            f"Broader scored records: {prioritization.get('scored_record_count', 'n/a')}. "
            f"Missing-label records: {prioritization.get('unlabeled_record_count', 'n/a')}. "
            f"Diversity groups: {prioritization.get('diversity_group_count', 'n/a')}."
        ),
        "",
        "## Diversity-Aware Shortlist",
        "",
        (
            f"Shortlist size: {shortlist.get('final_shortlist_size', 'n/a')} from "
            f"{shortlist.get('candidate_pool_size_before_diversity_filtering', 'n/a')} "
            "candidate records before diversity filtering."
        ),
        "",
        "## Unsupervised Landscape",
        "",
        (
            f"Landscape status: {landscape.get('status', 'missing')}. "
            f"Feature source: {landscape.get('feature_source', 'n/a')}. "
            f"Cluster count: {(landscape.get('cluster_info') or {}).get('n_clusters', 'n/a')}."
        ),
        "",
        "## Background Retrieval Status",
        "",
        (
            f"Background retrieval status: {background.get('status', 'missing')}. "
            "Background retrieval metrics were kept separate from the main classification task."
        ),
        "",
        "### Broad OAS Retrieval",
        "",
        (
            f"Broad OAS retrieval used OAS paired rows as unknown-target background. "
            f"Project rows: {background.get('project_row_count_before_balance', 'n/a')}. "
            f"OAS rows after overlap removal: "
            f"{background.get('oas_row_count_after_overlap_removal', 'n/a')}. "
            f"Exact overlaps removed: {background.get('exact_overlap_count', 'n/a')}. "
            f"ROC-AUC: {metric_text(background.get('roc_auc'))}. "
            f"PR-AUC: {metric_text(background.get('pr_auc'))}."
        ),
        "",
        "### Hard Matched OAS Retrieval",
        "",
        (
            f"Matched OAS retrieval used coarse heavy-length, light-length, total-length, "
            f"and light-status bins. Matched project rows: "
            f"{matched_background.get('matched_project_row_count', 'n/a')}. "
            f"Matched OAS rows: {matched_background.get('matched_oas_row_count', 'n/a')}. "
            f"Skipped project rows: "
            f"{matched_background.get('skipped_project_rows_due_to_no_matched_oas_bin', 'n/a')}. "
            f"Exact overlaps removed: "
            f"{matched_background.get('exact_overlap_count_removed', 'n/a')}. "
            f"ROC-AUC: {metric_text(matched_background.get('roc_auc'))}. "
            f"PR-AUC: {metric_text(matched_background.get('pr_auc'))}."
        ),
        "",
        "### Interpretation",
        "",
        oas_interpretation(background, matched_background),
        "",
        "## Retrospective Selection-Loop Simulation",
        "",
        (
            f"Best strategy: {active.get('best_strategy', 'n/a')}. "
            f"Best beats random mean: {active.get('best_strategy_beats_random_mean', 'n/a')}."
        ),
        "",
        "## Structure Metadata",
        "",
        (
            f"Structure metadata available in shortlist: "
            f"{structure.get('structure_available_count', 'n/a')}. Docking was not run by default."
        ),
        "",
        "## Limitations",
        "",
        "- Public source labels are heterogeneous and retrospective.",
        "- Model probabilities are prioritization signals for existing-record review.",
        "- Subset-specific metrics are not directly comparable across row subsets.",
        "- Diversity and sequence-risk features are heuristic.",
        "- Background retrieval is optional and local-data dependent.",
        "- Docking remains a separate future validation workflow.",
        "",
        "## Next Steps",
        "",
        "- Curate clearer target-region and structure metadata where available.",
        "- Add prospective-style validation only when new external records are available.",
        "- Keep benchmark comparisons matched by row subset and split strategy.",
        "- Use the shortlist as an inspection queue for existing records.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    data = load_all()
    write_text(REPORT_PATH, build_report(data))
    print("Final project report complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
