# Retrospective Selection-Loop Simulation

This simulation compares selection strategies on existing labeled records.
It is retrospective and evaluates existing-record selection behavior.

| Metric | Value |
|---|---:|
| Candidate records | 7937 |
| Label 0 count | 2292 |
| Label 1 count | 5645 |
| Strategies evaluated | 6 |
| Best strategy | highest_score |
| Best beats random mean | true |

## Strategy Results

| Strategy | Selected at largest budget | Positive labels | Precision |
|---|---:|---:|---:|
| random | 500 | 354.98 | 0.7100 |
| highest_score | 500 | 491.00 | 0.9820 |
| uncertainty | 500 | 380.00 | 0.7600 |
| diversity_aware_high_score | 500 | 305.00 | 0.6100 |
| target_region_stratified_high_score | 500 | 375.00 | 0.7500 |
| cluster_aware_high_score | 500 | 461.00 | 0.9220 |

## Artifacts

- `reports/metrics/active_learning_simulation_metrics.json`
- `reports/active_learning_selected_records.csv`
- `reports/figures/active_learning_strategy_hits.png`
