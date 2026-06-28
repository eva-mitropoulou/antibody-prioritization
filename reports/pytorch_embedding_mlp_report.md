# PyTorch AbLang2 Embedding MLP

Metadata: `data/processed/embeddings_ablang2_metadata.csv`
Device: `cpu`
Parallel jobs: `4`

This report evaluates small PyTorch MLP classifiers on frozen AbLang2
embeddings for existing labeled rows.

## Data

| Metric | Value |
|---|---:|
| Rows | 5573 |
| Label 0 count | 2292 |
| Label 1 count | 3281 |
| heavy embedding shape | [5573, 480] |
| pair embedding shape | [5573, 960] |

## MLP Metrics

| Split | Embedding | Device | Epochs | Best epoch | Train size | Test size | Group overlap | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Confusion matrix |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| random | heavy | cpu | 23 | 11 | 4458 | 1115 | n/a | 0.6547 | 0.6493 | 0.7182 | 0.6799 | 0.6985 | 0.6976 | 0.7665 | [[284, 175], [210, 446]] |
| random | pair | cpu | 26 | 14 | 4458 | 1115 | n/a | 0.6664 | 0.6726 | 0.7572 | 0.6372 | 0.6921 | 0.7261 | 0.7688 | [[325, 134], [238, 418]] |
| group_feature_v | heavy | cpu | 41 | 29 | 4751 | 822 | 0 | 0.6667 | 0.6457 | 0.7075 | 0.7536 | 0.7298 | 0.7293 | 0.8098 | [[178, 153], [121, 370]] |
| group_feature_v | pair | cpu | 21 | 9 | 4751 | 822 | 0 | 0.7056 | 0.6842 | 0.7345 | 0.7943 | 0.7632 | 0.7573 | 0.8099 | [[190, 141], [101, 390]] |

## Comparison Against Baselines

| Split | Embedding | K-mer ROC-AUC | MLP ROC-AUC | Delta vs k-mer | Embedding logreg ROC-AUC | Delta vs embedding logreg | K-mer PR-AUC | MLP PR-AUC | Delta PR-AUC vs embedding logreg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random | heavy | 0.7534 | 0.6976 | -0.0558 | 0.6714 | 0.0263 | 0.8157 | 0.7665 | 0.0230 |
| random | pair | 0.7534 | 0.7261 | -0.0273 | 0.7079 | 0.0182 | 0.8157 | 0.7688 | 0.0095 |
| group_feature_v | heavy | 0.7810 | 0.7293 | -0.0516 | 0.7382 | -0.0089 | 0.8236 | 0.8098 | -0.0113 |
| group_feature_v | pair | 0.7810 | 0.7573 | -0.0237 | 0.7112 | 0.0461 | 0.8236 | 0.8099 | 0.0240 |

## Interpretation

The PyTorch MLP improves over logistic regression on the same embeddings: pair.
The PyTorch MLP improves over the k-mer grouped ROC-AUC baseline: no.
Best grouped MLP ROC-AUC: `pair` (0.7573).
Best grouped MLP PR-AUC: `pair` (0.8099).
Frozen AbLang2 embeddings remain useful for neural-network benchmarking, but this result should not be overclaimed if k-mer features stay stronger.

## Artifacts

- `reports/metrics/pytorch_embedding_mlp_metrics.json`
- `reports/figures/pytorch_mlp_training_curves.png`
- `reports/figures/pytorch_mlp_roc_auc_comparison.png`
- `reports/figures/pytorch_mlp_pr_auc_comparison.png`
- `models/pytorch_mlp_heavy.pt`
- `models/pytorch_mlp_pair.pt`
