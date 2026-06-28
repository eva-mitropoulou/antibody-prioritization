# OAS Background Retrieval

This diagnostic treats paired OAS records as unknown-target natural
background, not assayed negative-class data. Metrics are not mixed with the main
neutralisation classification benchmark.

| Metric | Value |
|---|---:|
| Status | available |
| Project rows before balance | 11748 |
| OAS rows before overlap removal | 17882 |
| Exact overlap count | 5 |
| OAS rows after overlap removal | 17877 |
| Balanced rows per class | 11748 |
| Train rows | 18796 |
| Test rows | 4700 |
| Random baseline positive fraction | 0.5000 |
| ROC-AUC | 0.9921 |
| PR-AUC | 0.9897 |

## Top-k Enrichment

| k | Selected | Project records | Project fraction | Enrichment over random |
|---:|---:|---:|---:|---:|
| 50 | 50 | 50 | 1.0000 | 2.0000 |
| 100 | 100 | 100 | 1.0000 | 2.0000 |
| 500 | 500 | 498 | 0.9960 | 1.9920 |

## Artifacts

- `reports/oas_background_retrieval_report.md`
- `reports/metrics/oas_background_retrieval_metrics.json`
- `reports/oas_background_retrieval_scores.csv`
- `reports/figures/oas_retrieval_score_distribution.png`
- `reports/figures/oas_retrieval_topk_enrichment.png`
- `reports/figures/oas_background_sequence_space.png`
