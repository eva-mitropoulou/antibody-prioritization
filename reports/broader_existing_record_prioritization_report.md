# Broader Existing-Record Prioritization

This analysis scores and annotates broader cleaned public dataset records
with usable existing heavy sequences and preserved source sequence fields.

## Model Context

| Model | Grouped ROC-AUC | Grouped PR-AUC | Role |
|---|---:|---:|---|
| k-mer TF-IDF + logistic regression | 0.7810 | 0.8236 | main scoring model |
| IgBert last_1_layer seed mean | 0.7443 | 0.8151 | benchmark comparison |

Scoring model source: `loaded_saved_model`.
Loaded model path: `models/kmer_logreg_pair_text.joblib`.

## Summary

| Metric | Value |
|---|---:|
| Strict labeled records scored | 5573 |
| Broader records scored | 11747 |
| Labeled broader records | 7937 |
| Missing-label broader records | 3810 |
| Diversity groups | 73 |
| Records with structure | 659 |
| Missing-light records | 1762 |
| Nanobody-like records | 1762 |
| Mean predicted probability | 0.5188 |
| Median predicted probability | 0.4993 |
| Missing-label high-score/high-confidence records | 330 |
| Conflict-label high-score/high-confidence records | 165 |

## Record Category Counts

| Record category | Count |
|---|---:|
| missing_label | 3810 |
| known_neutralising | 3282 |
| conflict_label | 2363 |
| known_non_neutralising | 2292 |

## Priority Category Counts

| Priority category | Count |
|---|---:|
| uncertain_prediction | 6557 |
| lower_priority | 4095 |
| high_confidence_known_positive | 600 |
| high_score_missing_label | 329 |
| high_score_conflict_label | 165 |
| high_score_but_sequence_risk | 1 |

## Target Region Group Counts

| Target region group | Count |
|---|---:|
| RBD | 8146 |
| other | 2773 |
| NTD | 820 |
| unknown | 7 |
| Spike/S | 1 |

## Paired/Light-Missing Counts

| Status | Count |
|---|---:|
| paired | 9985 |
| light_missing_or_single_chain | 1762 |

## Developability Risk Counts

| Risk bin | Count |
|---|---:|
| low | 9294 |
| medium | 2433 |
| high | 20 |

## Limitations

- Model score is used for existing-record prioritization.
- Sequence-risk flags are heuristic.
- Labels are heterogeneous literature-derived labels.
- Retrospective scoring of existing records.
- Existing source sequences are preserved.

## Artifacts

- `reports/broader_existing_record_prioritization_table.csv`
- `reports/metrics/broader_existing_record_prioritization_summary.json`
- `reports/figures/broader_priority_category_counts.png`
- `reports/figures/broader_developability_risk_counts.png`
- `reports/figures/broader_probability_by_record_category.png`
- `reports/figures/broader_probability_vs_risk_score.png`
