# OAS Matched Background Retrieval

This hard control compares project records against length/status-matched
OAS unknown-target background. It is separate from the main neutralisation
classification benchmark.

| Metric | Value |
|---|---:|
| Matched project rows | 7192 |
| Matched OAS rows | 7192 |
| Skipped project rows | 1818 |
| Exact overlap count removed | 5 |
| Train rows | 11507 |
| Test rows | 2877 |
| Random baseline positive fraction | 0.4998 |
| ROC-AUC | 0.9911 |
| PR-AUC | 0.9893 |

## Confusion Matrix

| True label | Predicted OAS background | Predicted project |
|---|---:|---:|
| OAS unknown-target background | 1346 | 93 |
| Project record | 24 | 1414 |

## Top-k Enrichment

| k | Project records | Project fraction | Enrichment over random |
|---:|---:|---:|---:|
| 50 | 50 | 1.0000 | 2.0007 |
| 100 | 100 | 1.0000 | 2.0007 |
| 500 | 498 | 0.9960 | 1.9927 |

## Artifacts

- `reports/oas_matched_background_audit.md`
- `reports/metrics/oas_matched_background_audit.json`
- `reports/oas_matched_background_retrieval_report.md`
- `reports/metrics/oas_matched_background_retrieval_metrics.json`
- `reports/oas_matched_background_retrieval_scores.csv`
- `reports/figures/oas_matched_retrieval_score_distribution.png`
- `reports/figures/oas_matched_retrieval_topk_enrichment.png`
