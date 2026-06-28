# AbLang2 Embedding Baseline

Input embeddings: `data/processed/embeddings_ablang2_heavy.npy`

This report evaluates logistic-regression classifiers on cached pretrained
AbLang2 embeddings for existing labeled rows.

## Data

| Metric | Value |
|---|---:|
| Rows | 5573 |
| Label 0 count | 2292 |
| Label 1 count | 3281 |
| heavy embedding shape | [5573, 480] |
| pair embedding shape | [5573, 960] |

## Split Validity

| Split | Group column | Valid | Meaningful | Reason | Group overlap |
|---|---|---:|---:|---|---:|
| random | n/a | true | true | ok | n/a |
| group_feature_v | group_feature_v | true | true | ok | 0 |

## Metrics

| Split | Model | Train size | Test size | Train labels | Test labels | Train groups | Test groups | Group overlap | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | heavy | 4458 | 1115 | 0=1833, 1=2625 | 0=459, 1=656 | n/a | n/a | n/a | 0.6206 | 0.6184 | 0.6958 | 0.6311 | 0.6619 | 0.6714 | 0.7436 |
| random | pair | 4458 | 1115 | 0=1833, 1=2625 | 0=459, 1=656 | n/a | n/a | n/a | 0.6700 | 0.6704 | 0.7449 | 0.6677 | 0.7042 | 0.7079 | 0.7593 |
| group_feature_v | heavy | 4751 | 822 | 0=1961, 1=2790 | 0=331, 1=491 | 75 | 19 | 0 | 0.6727 | 0.6429 | 0.6982 | 0.7963 | 0.7441 | 0.7382 | 0.8211 |
| group_feature_v | pair | 4751 | 822 | 0=1961, 1=2790 | 0=331, 1=491 | 75 | 19 | 0 | 0.6740 | 0.6537 | 0.7140 | 0.7576 | 0.7352 | 0.7112 | 0.7858 |

## Comparison vs K-mer

| Split | Embedding model | K-mer ROC-AUC | Embedding ROC-AUC | Delta ROC-AUC | K-mer PR-AUC | Embedding PR-AUC | Delta PR-AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| random | heavy | 0.7534 | 0.6714 | -0.0820 | 0.8157 | 0.7436 | -0.0722 |
| random | pair | 0.7534 | 0.7079 | -0.0455 | 0.8157 | 0.7593 | -0.0564 |
| group_feature_v | heavy | 0.7810 | 0.7382 | -0.0428 | 0.8236 | 0.8211 | -0.0025 |
| group_feature_v | pair | 0.7810 | 0.7112 | -0.0698 | 0.8236 | 0.7858 | -0.0378 |

## Confusion Matrices

### random / heavy

| True label | Predicted 0 | Predicted 1 |
|---|---:|---:|
| 0 | 278 | 181 |
| 1 | 242 | 414 |

### random / pair

| True label | Predicted 0 | Predicted 1 |
|---|---:|---:|
| 0 | 309 | 150 |
| 1 | 218 | 438 |

### group_feature_v / heavy

| True label | Predicted 0 | Predicted 1 |
|---|---:|---:|
| 0 | 162 | 169 |
| 1 | 100 | 391 |

### group_feature_v / pair

| True label | Predicted 0 | Predicted 1 |
|---|---:|---:|
| 0 | 182 | 149 |
| 1 | 119 | 372 |

## Artifacts

- `reports/metrics/embedding_baseline_metrics.json`
- `reports/figures/embedding_vs_kmer_roc_auc.png`
- `reports/figures/embedding_vs_kmer_pr_auc.png`
- `models/embedding_logreg_heavy.joblib`
- `models/embedding_logreg_pair.joblib`
