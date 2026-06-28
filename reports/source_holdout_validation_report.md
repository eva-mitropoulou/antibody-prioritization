# Source-Holdout Validation

Source/study groups are represented only by hashes and short IDs. Raw
source strings, DOI/source URLs, and sequence strings are not written.

## Source Diagnostics

| Metric | Value |
|---|---:|
| Source column | metadata_origin |
| Source detection reason | source_like_fallback |
| Row count | 5573 |
| Source group count | 84 |
| Usable source group count | 16 |
| Source group size min | 1 |
| Source group size median | 4.0000 |
| Source group size max | 3062 |

## Leave-Source-Out Aggregate

| Metric | Value |
|---|---:|
| Valid held-out source groups | 11 |
| Skipped/limited groups | 73 |
| Macro ROC-AUC | 0.5605 |
| Macro PR-AUC | 0.6454 |
| Weighted ROC-AUC | 0.6104 |
| Weighted PR-AUC | 0.6370 |

## Source-Grouped Fallback Split

| Metric | Value |
|---|---:|
| Split strategy | GroupShuffleSplit |
| Group column | source_group_short_id |
| Train rows | 5006 |
| Test rows | 567 |
| Group overlap | 0 |
| ROC-AUC | 0.5247 |
| PR-AUC | 0.8189 |
| F1 | 0.6649 |

## Skipped Group Reasons

| Reason | Count |
|---|---:|
| test_insufficient_class_counts | 5 |
| too_few_test_rows | 68 |

## Per-Source Results

| Source group | Status | Reason | Test rows | Label 0 | Label 1 | ROC-AUC | PR-AUC |
|---|---|---|---:|---:|---:|---:|---:|
| source_001 | metrics_limited | test_insufficient_class_counts | 165 | 0 | 165 | n/a | n/a |
| source_002 | skipped | too_few_test_rows | 2 | 0 | 2 | n/a | n/a |
| source_003 | skipped | too_few_test_rows | 3 | 0 | 3 | n/a | n/a |
| source_004 | skipped | too_few_test_rows | 7 | 0 | 7 | n/a | n/a |
| source_005 | skipped | too_few_test_rows | 1 | 0 | 1 | n/a | n/a |
| source_006 | metrics_limited | test_insufficient_class_counts | 192 | 0 | 192 | n/a | n/a |
| source_007 | skipped | too_few_test_rows | 1 | 0 | 1 | n/a | n/a |
| source_008 | metrics_limited | test_insufficient_class_counts | 31 | 0 | 31 | n/a | n/a |
| source_009 | metrics_limited | test_insufficient_class_counts | 267 | 0 | 267 | n/a | n/a |
| source_010 | skipped | too_few_test_rows | 1 | 0 | 1 | n/a | n/a |
| source_011 | skipped | too_few_test_rows | 1 | 0 | 1 | n/a | n/a |
| source_012 | skipped | too_few_test_rows | 4 | 0 | 4 | n/a | n/a |
| source_013 | skipped | too_few_test_rows | 5 | 3 | 2 | n/a | n/a |
| source_014 | skipped | too_few_test_rows | 2 | 0 | 2 | n/a | n/a |
| source_015 | skipped | too_few_test_rows | 2 | 0 | 2 | n/a | n/a |
| source_016 | skipped | too_few_test_rows | 2 | 0 | 2 | n/a | n/a |
| source_017 | skipped | too_few_test_rows | 1 | 0 | 1 | n/a | n/a |
| source_018 | skipped | too_few_test_rows | 1 | 0 | 1 | n/a | n/a |
| source_019 | skipped | too_few_test_rows | 1 | 0 | 1 | n/a | n/a |
| source_020 | skipped | too_few_test_rows | 6 | 3 | 3 | n/a | n/a |
| source_021 | valid | ok | 39 | 28 | 11 | 0.5909 | 0.3158 |
| source_022 | skipped | too_few_test_rows | 1 | 0 | 1 | n/a | n/a |
| source_023 | skipped | too_few_test_rows | 6 | 0 | 6 | n/a | n/a |
| source_024 | skipped | too_few_test_rows | 4 | 2 | 2 | n/a | n/a |
| source_025 | skipped | too_few_test_rows | 24 | 2 | 22 | n/a | n/a |
| source_026 | skipped | too_few_test_rows | 1 | 0 | 1 | n/a | n/a |
| source_027 | skipped | too_few_test_rows | 1 | 0 | 1 | n/a | n/a |
| source_028 | skipped | too_few_test_rows | 11 | 2 | 9 | n/a | n/a |
| source_029 | skipped | too_few_test_rows | 1 | 0 | 1 | n/a | n/a |
| source_030 | skipped | too_few_test_rows | 1 | 0 | 1 | n/a | n/a |
| source_031 | skipped | too_few_test_rows | 2 | 0 | 2 | n/a | n/a |
| source_032 | skipped | too_few_test_rows | 4 | 0 | 4 | n/a | n/a |
| source_033 | skipped | too_few_test_rows | 13 | 0 | 13 | n/a | n/a |
| source_034 | skipped | too_few_test_rows | 3 | 0 | 3 | n/a | n/a |
| source_035 | skipped | too_few_test_rows | 13 | 0 | 13 | n/a | n/a |
| source_036 | skipped | too_few_test_rows | 1 | 0 | 1 | n/a | n/a |
| source_037 | skipped | too_few_test_rows | 10 | 1 | 9 | n/a | n/a |
| source_038 | valid | ok | 299 | 191 | 108 | 0.7580 | 0.6933 |
| source_039 | skipped | too_few_test_rows | 1 | 0 | 1 | n/a | n/a |
| source_040 | skipped | too_few_test_rows | 2 | 0 | 2 | n/a | n/a |
| source_041 | valid | ok | 113 | 5 | 108 | 0.1222 | 0.8941 |
| source_042 | skipped | too_few_test_rows | 18 | 15 | 3 | n/a | n/a |
| source_043 | valid | ok | 358 | 347 | 11 | 0.4747 | 0.0301 |
| source_044 | skipped | too_few_test_rows | 3 | 0 | 3 | n/a | n/a |
| source_045 | skipped | too_few_test_rows | 5 | 0 | 5 | n/a | n/a |
| source_046 | skipped | too_few_test_rows | 3 | 0 | 3 | n/a | n/a |
| source_047 | skipped | too_few_test_rows | 5 | 0 | 5 | n/a | n/a |
| source_048 | valid | ok | 136 | 62 | 74 | 0.8117 | 0.8468 |
| source_049 | skipped | too_few_test_rows | 2 | 0 | 2 | n/a | n/a |
| source_050 | skipped | too_few_test_rows | 6 | 0 | 6 | n/a | n/a |
| source_051 | skipped | too_few_test_rows | 1 | 0 | 1 | n/a | n/a |
| source_052 | valid | ok | 94 | 62 | 32 | 0.6663 | 0.5083 |
| source_053 | skipped | too_few_test_rows | 2 | 0 | 2 | n/a | n/a |
| source_054 | skipped | too_few_test_rows | 2 | 0 | 2 | n/a | n/a |
| source_055 | skipped | too_few_test_rows | 4 | 2 | 2 | n/a | n/a |
| source_056 | skipped | too_few_test_rows | 2 | 0 | 2 | n/a | n/a |
| source_057 | skipped | too_few_test_rows | 5 | 0 | 5 | n/a | n/a |
| source_058 | skipped | too_few_test_rows | 1 | 0 | 1 | n/a | n/a |
| source_059 | skipped | too_few_test_rows | 1 | 0 | 1 | n/a | n/a |
| source_060 | valid | ok | 101 | 6 | 95 | 0.2947 | 0.9042 |
| source_061 | valid | ok | 80 | 42 | 38 | 0.5639 | 0.5510 |
| source_062 | skipped | too_few_test_rows | 1 | 0 | 1 | n/a | n/a |
| source_063 | skipped | too_few_test_rows | 1 | 0 | 1 | n/a | n/a |
| source_064 | skipped | too_few_test_rows | 7 | 0 | 7 | n/a | n/a |
| source_065 | skipped | too_few_test_rows | 7 | 0 | 7 | n/a | n/a |
| source_066 | skipped | too_few_test_rows | 8 | 0 | 8 | n/a | n/a |
| source_067 | skipped | too_few_test_rows | 3 | 0 | 3 | n/a | n/a |
| source_068 | skipped | too_few_test_rows | 2 | 0 | 2 | n/a | n/a |
| source_069 | skipped | too_few_test_rows | 1 | 0 | 1 | n/a | n/a |
| source_070 | skipped | too_few_test_rows | 5 | 0 | 5 | n/a | n/a |
| source_071 | skipped | too_few_test_rows | 18 | 0 | 18 | n/a | n/a |
| source_072 | skipped | too_few_test_rows | 3 | 0 | 3 | n/a | n/a |
| source_073 | skipped | too_few_test_rows | 2 | 2 | 0 | n/a | n/a |
| source_074 | valid | ok | 100 | 28 | 72 | 0.4563 | 0.7258 |
| source_075 | skipped | too_few_test_rows | 6 | 0 | 6 | n/a | n/a |
| source_076 | skipped | too_few_test_rows | 1 | 0 | 1 | n/a | n/a |
| source_077 | skipped | too_few_test_rows | 5 | 0 | 5 | n/a | n/a |
| source_078 | skipped | too_few_test_rows | 4 | 2 | 2 | n/a | n/a |
| source_079 | metrics_limited | test_insufficient_class_counts | 87 | 2 | 85 | n/a | n/a |
| source_080 | valid | ok | 164 | 22 | 142 | 0.8006 | 0.9657 |
| source_081 | skipped | too_few_test_rows | 2 | 0 | 2 | n/a | n/a |
| source_082 | skipped | too_few_test_rows | 2 | 0 | 2 | n/a | n/a |
| source_083 | valid | ok | 3062 | 1463 | 1599 | 0.6259 | 0.6646 |
| source_084 | skipped | too_few_test_rows | 8 | 0 | 8 | n/a | n/a |

## Artifacts

- `reports/source_holdout_validation_report.md`
- `reports/metrics/source_holdout_validation_metrics.json`
- `reports/figures/source_holdout_roc_auc_by_group.png`
- `reports/figures/source_holdout_pr_auc_by_group.png`
