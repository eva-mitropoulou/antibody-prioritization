# Source-Robust Model Selection

This module evaluates CPU-only compact k-mer models under source/study
holdout controls, calibration diagnostics, and abstention analysis. It uses
compact CPU baselines and aggregate/source-safe outputs.

## Data Audit

| Metric | Value |
|---|---:|
| Input shape rows | 5573 |
| Input shape columns | 38 |
| Label 0 count | 2292 |
| Label 1 count | 3281 |
| Source groups | 84 |
| Candidate source-holdout groups | 11 |

## Model Comparison

| Model | Selection eligible | Rows | Valid source groups | Weighted PR-AUC | Weighted ROC-AUC | Brier | Best threshold | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| whole_pair_kmer | true | 5573 | 11 | 0.6363 | 0.6095 | 0.2637 | 0.7000 | 0.8266 | 0.3062 |
| whole_plus_cdr_kmer | true | 5573 | 11 | 0.6345 | 0.6059 | 0.2632 | 0.8000 | 0.8062 | 0.2227 |
| heavy_only_kmer | true | 5573 | 11 | 0.6243 | 0.5977 | 0.2707 | 0.7000 | 0.8304 | 0.3041 |
| cdr_region_kmer | true | 5573 | 11 | 0.6091 | 0.5773 | 0.2492 | 0.8000 | 1.0000 | 0.0107 |
| rbd_or_target_region_subset_kmer | false | 4820 | 10 | 0.6628 | 0.6170 | 0.2164 | 0.9000 | 0.8667 | 0.0349 |
| paired_only_whole_pair_kmer | false | 5092 | 8 | 0.6023 | 0.6060 | 0.2719 | 0.9000 | 1.0000 | 0.0066 |

## Selection

Selected model: `whole_pair_kmer`.

Selected whole_pair_kmer by weighted leave-source-out PR-AUC=0.6363, ROC-AUC=0.6095, Brier=0.2637.

Previous source-holdout baseline macro ROC-AUC/PR-AUC: 0.5605/0.6454.
Selected weighted source-holdout ROC-AUC/PR-AUC: 0.6095/0.6363.
Meaningful improvement over previous baseline: false.

Treat scores as ranking/prioritization evidence rather than calibrated prospective prediction.

## Failure Analysis

Best and worst groups are listed by sanitized source ID only.

### Best-Generalizing Source Groups

| Source group | Test rows | Positives | Negatives | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| source_080 | 164 | 142 | 22 | 0.7948 | 0.9648 |
| source_060 | 101 | 95 | 6 | 0.3035 | 0.9058 |
| source_041 | 113 | 108 | 5 | 0.1333 | 0.8961 |
| source_048 | 136 | 74 | 62 | 0.8019 | 0.8421 |
| source_074 | 100 | 72 | 28 | 0.4182 | 0.7025 |

### Worst-Generalizing Source Groups

| Source group | Test rows | Positives | Negatives | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| source_043 | 358 | 11 | 347 | 0.4776 | 0.0299 |
| source_021 | 39 | 11 | 28 | 0.5877 | 0.3151 |
| source_052 | 94 | 32 | 62 | 0.6729 | 0.5136 |
| source_061 | 80 | 38 | 42 | 0.5558 | 0.5523 |
| source_083 | 3062 | 1599 | 1463 | 0.6259 | 0.6649 |

## Interpretation

Source/study effects remain visible under source-holdout validation. CDR/region models were competitive for source robustness. The selected score is used as a ranking and prioritization signal for existing records. High-confidence review thresholds are chosen by precision and coverage tradeoff.

## Artifacts

- `reports/source_robust_model_selection_report.md`
- `reports/metrics/source_robust_model_selection_metrics.json`
- `reports/source_robust_model_comparison.csv`
- `reports/source_holdout_failure_analysis.csv`
- `reports/figures/source_robust_model_comparison.png`
- `reports/figures/source_robust_pr_auc_by_model.png`
- `reports/figures/source_robust_roc_auc_by_model.png`
- `reports/figures/abstention_precision_coverage.png`
- `reports/figures/source_failure_summary.png`
