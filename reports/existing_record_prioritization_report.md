# Existing-Record Prioritization

This analysis scores and annotates existing public dataset records only.
It does not generate, alter, mutate, optimize, rank newly designed
sequences, or propose sequence changes.

## Model Context

| Model | Grouped ROC-AUC | Grouped PR-AUC | Role |
|---|---:|---:|---|
| k-mer TF-IDF + logistic regression | 0.7810 | 0.8236 | main scoring model |
| frozen pretrained pair MLP | 0.7541 | 0.8078 | benchmark comparison |
| IgBert last_1_layer seed mean | 0.7443 | 0.8151 | benchmark comparison |

Scoring model source: `loaded_saved_model`.
Loaded model path: `models/kmer_logreg_pair_text.joblib`.

## Summary

| Metric | Value |
|---|---:|
| Scored records | 5573 |
| Labeled records | 5573 |
| Unlabeled records | 0 |
| Diversity groups | 476 |
| Records with structure | 330 |
| Missing-light records | 481 |
| Nanobody-like records | 481 |
| Mean predicted probability | 0.5214 |
| Median predicted probability | 0.5014 |

## Priority Category Counts

| Priority category | Count |
|---|---:|
| uncertain_prediction | 2911 |
| lower_priority | 2062 |
| high_confidence_known_positive | 600 |

## Target Region Group Counts

| Target region group | Count |
|---|---:|
| RBD | 4820 |
| NTD | 431 |
| other | 314 |
| unknown | 7 |
| Spike/S | 1 |

## Paired/Light-Missing Counts

| Status | Count |
|---|---:|
| paired | 5092 |
| light_missing_or_single_chain | 481 |

## Developability Risk Counts

| Risk bin | Count |
|---|---:|
| low | 4436 |
| medium | 1126 |
| high | 11 |

## Limitations

- Model score is not therapeutic efficacy.
- Sequence-risk flags are heuristic.
- Labels are heterogeneous literature-derived labels.
- This is retrospective scoring of existing records only.
- No new sequences are generated, altered, proposed, or optimized.

## Artifacts

- `reports/existing_record_prioritization_table.csv`
- `reports/metrics/existing_record_prioritization_summary.json`
- `reports/figures/prioritization_probability_distribution.png`
- `reports/figures/developability_risk_counts.png`
- `reports/figures/priority_category_counts.png`
- `reports/figures/probability_vs_developability_risk.png`
