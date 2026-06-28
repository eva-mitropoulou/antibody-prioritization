# Biological Sensitivity Analysis

Aggregate sensitivity checks across target-region, paired/light-missing,
record-category, confidence, and sequence-risk subgroups.

| Metric | Value |
|---|---:|
| Rows | 11747 |
| Label 0 count | 2292 |
| Label 1 count | 5645 |
| Missing label count | 3810 |
| Median score | 0.4993 |

## target_region_group

| Group | Rows | Label 0 | Label 1 | High-score count | Median score |
|---|---:|---:|---:|---:|---:|
| NTD | 820 | 304 | 190 | 27 | 0.4669 |
| RBD | 8146 | 1754 | 5356 | 1476 | 0.5264 |
| Spike/S | 1 | 0 | 1 | 0 | 0.7156 |
| other | 2773 | 233 | 92 | 174 | 0.4424 |
| unknown | 7 | 1 | 6 | 1 | 0.6095 |

## paired_light_status

| Group | Rows | Label 0 | Label 1 | High-score count | Median score |
|---|---:|---:|---:|---:|---:|
| light_missing_or_single_chain | 1762 | 60 | 448 | 722 | 0.7040 |
| paired | 9985 | 2232 | 5197 | 956 | 0.4694 |

## record_category

| Group | Rows | Label 0 | Label 1 | High-score count | Median score |
|---|---:|---:|---:|---:|---:|
| conflict_label | 2363 | 0 | 2363 | 281 | 0.5055 |
| known_neutralising | 3282 | 0 | 3282 | 861 | 0.6166 |
| known_non_neutralising | 2292 | 2292 | 0 | 31 | 0.3587 |
| missing_label | 3810 | 0 | 0 | 505 | 0.4941 |

## developability_risk_bin

| Group | Rows | Label 0 | Label 1 | High-score count | Median score |
|---|---:|---:|---:|---:|---:|
| high | 20 | 1 | 13 | 2 | 0.5076 |
| low | 9294 | 1778 | 4546 | 1400 | 0.5066 |
| medium | 2433 | 513 | 1086 | 276 | 0.4720 |

## confidence_bin

| Group | Rows | Label 0 | Label 1 | High-score count | Median score |
|---|---:|---:|---:|---:|---:|
| high | 1249 | 81 | 774 | 1109 | 0.8355 |
| low | 6557 | 1157 | 3263 | 0 | 0.4874 |
| medium | 3941 | 1054 | 1608 | 569 | 0.3403 |
