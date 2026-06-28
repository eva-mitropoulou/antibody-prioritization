# Neutral Grouped K-mer Validation

Input file: `data/processed/neutral_sequence_classification_ml.csv`

This report evaluates a generic supervised sequence-classification baseline
on existing labeled rows using neutral column names.

## Data

| Metric | Value |
|---|---:|
| Rows | 5573 |
| Label 0 count | 2292 |
| Label 1 count | 3281 |

## Model

- `TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2)`
- `LogisticRegression(max_iter=5000, class_weight="balanced")`
- Majority-class baseline: `DummyClassifier(strategy="most_frequent")`

## Split Validity

| Split | Group column | Valid | Meaningful | Reason | Train groups | Test groups | Group overlap |
|---|---|---:|---:|---|---:|---:|---:|
| random | n/a | true | true | ok | n/a | n/a | n/a |
| group_feature_cdr3 | group_feature_cdr3 | false | false | near_row_unique | n/a | n/a | n/a |
| group_feature_v | group_feature_v | true | true | ok | 75 | 19 | 0 |

## Metrics

| Split | Model | Input | Train size | Test size | Train labels | Test labels | Train groups | Test groups | Group overlap | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | majority_baseline | n/a | 4458 | 1115 | 0=1833, 1=2625 | 0=459, 1=656 | n/a | n/a | n/a | 0.5883 | 0.5000 | 0.5883 | 1.0000 | 0.7408 | 0.5000 | 0.5883 |
| random | kmer_logreg | sequence_a | 4458 | 1115 | 0=1833, 1=2625 | 0=459, 1=656 | n/a | n/a | n/a | 0.6574 | 0.6653 | 0.7537 | 0.6204 | 0.6806 | 0.7263 | 0.7934 |
| random | kmer_logreg | sequence_pair_text | 4458 | 1115 | 0=1833, 1=2625 | 0=459, 1=656 | n/a | n/a | n/a | 0.6816 | 0.6931 | 0.7878 | 0.6280 | 0.6989 | 0.7534 | 0.8157 |
| group_feature_v | majority_baseline | n/a | 4751 | 822 | 0=1961, 1=2790 | 0=331, 1=491 | 75 | 19 | 0 | 0.5973 | 0.5000 | 0.5973 | 1.0000 | 0.7479 | 0.5000 | 0.5973 |
| group_feature_v | kmer_logreg | sequence_a | 4751 | 822 | 0=1961, 1=2790 | 0=331, 1=491 | 75 | 19 | 0 | 0.6703 | 0.6359 | 0.6903 | 0.8126 | 0.7465 | 0.7636 | 0.8238 |
| group_feature_v | kmer_logreg | sequence_pair_text | 4751 | 822 | 0=1961, 1=2790 | 0=331, 1=491 | 75 | 19 | 0 | 0.7007 | 0.6747 | 0.7231 | 0.8086 | 0.7635 | 0.7810 | 0.8236 |

## Confusion Matrices

### random

| True label | Predicted 0 | Predicted 1 |
|---|---:|---:|
| 0 | 348 | 111 |
| 1 | 244 | 412 |

### group_feature_v

| True label | Predicted 0 | Predicted 1 |
|---|---:|---:|
| 0 | 179 | 152 |
| 1 | 94 | 397 |

## Validation Conclusion

At least one neutral grouped split is valid, with zero train/test group overlap.
Skipped grouped split(s): group_feature_cdr3=near_row_unique.

## Artifacts

- `reports/metrics/grouped_validation_metrics.json`
- `reports/figures/grouped_validation_roc_auc_comparison.png`
- `reports/figures/grouped_validation_pr_auc_comparison.png`
- `reports/figures/grouped_validation_f1_comparison.png`
