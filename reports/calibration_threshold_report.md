# Calibration And Threshold Analysis

Calibration was evaluated on a held-out grouped split. Probability scores
are interpreted as prioritization/ranking signals unless calibration is
shown to be reliable.

| Metric | Value |
|---|---:|
| Split group column | source_group_short_id |
| Train rows | 5006 |
| Test rows | 567 |
| Group overlap | 0 |
| ROC-AUC | 0.5247 |
| PR-AUC | 0.8189 |
| Brier score | 0.2636 |
| Expected calibration error | 0.3034 |

## Threshold Table

| Threshold | Predicted positives | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|
| 0.5 | 300 | 0.8500 | 0.5460 | 0.6649 |
| 0.6 | 219 | 0.8493 | 0.3983 | 0.5423 |
| 0.7 | 171 | 0.8187 | 0.2998 | 0.4389 |
| 0.8 | 132 | 0.7955 | 0.2248 | 0.3506 |
| 0.9 | 19 | 0.7368 | 0.0300 | 0.0576 |

## Top-k Precision

| k | Positive count | Precision |
|---:|---:|---:|
| 25 | 18 | 0.7200 |
| 50 | 36 | 0.7200 |
| 100 | 75 | 0.7500 |
| 250 | 213 | 0.8520 |

## High-Confidence Review Threshold

Selected threshold 0.7: precision 0.8187, recall 0.2998, predicted positives 171.

## Calibration Interpretation

Use the model primarily for ranking unless a target use case accepts the
reported Brier score and threshold tradeoffs.

## Artifacts

- `reports/calibration_threshold_report.md`
- `reports/metrics/calibration_threshold_metrics.json`
- `reports/figures/calibration_curve.png`
- `reports/figures/threshold_precision_recall.png`
- `reports/figures/probability_histogram_by_label.png`
