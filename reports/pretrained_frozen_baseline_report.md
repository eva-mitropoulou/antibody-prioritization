# Frozen Pretrained Sequence Model Baseline

This report evaluates classifiers on frozen Hugging Face sequence-model
representations for existing labeled rows only. No pretrained parameters
were fine-tuned.

## Model Availability

| Field | Value |
|---|---|
| Available | `True` |
| Reason | `cached_embeddings_loaded` |
| Model name | `Exscientia/IgBert` |
| Tokenizer class | `BertTokenizer` |
| Model class | `BertModel` |
| Tokenization style | `spaced` |
| Pair embedding available | `True` |
| Pair skip reason | `n/a` |
| Classifier device | `cpu` |

## Data

| Metric | Value |
|---|---:|
| Rows | 5573 |
| Label 0 count | 2292 |
| Label 1 count | 3281 |
| heavy embedding shape | [5573, 1024] |
| pair embedding shape | [5573, 1024] |

## Metrics

| Split | Input | Classifier | Train size | Test size | Train labels | Test labels | Group overlap | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Confusion matrix |
|---|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| random | heavy | logistic_regression | 4458 | 1115 | 0=1833, 1=2625 | 0=459, 1=656 | n/a | 0.6269 | 0.6250 | 0.7020 | 0.6357 | 0.6672 | 0.6808 | 0.7628 | [[282, 177], [239, 417]] |
| random | heavy | pytorch_mlp | 4458 | 1115 | 0=1833, 1=2625 | 0=459, 1=656 | n/a | 0.6475 | 0.6507 | 0.7319 | 0.6326 | 0.6787 | 0.7102 | 0.7892 | [[307, 152], [241, 415]] |
| random | pair | logistic_regression | 4458 | 1115 | 0=1833, 1=2625 | 0=459, 1=656 | n/a | 0.6691 | 0.6667 | 0.7372 | 0.6799 | 0.7074 | 0.7122 | 0.7743 | [[300, 159], [210, 446]] |
| random | pair | pytorch_mlp | 4458 | 1115 | 0=1833, 1=2625 | 0=459, 1=656 | n/a | 0.6888 | 0.6884 | 0.7588 | 0.6905 | 0.7231 | 0.7361 | 0.7970 | [[315, 144], [203, 453]] |
| group_feature_v | heavy | logistic_regression | 4751 | 822 | 0=1961, 1=2790 | 0=331, 1=491 | 0 | 0.6448 | 0.6072 | 0.6695 | 0.8004 | 0.7291 | 0.6835 | 0.7497 | [[137, 194], [98, 393]] |
| group_feature_v | heavy | pytorch_mlp | 4751 | 822 | 0=1961, 1=2790 | 0=331, 1=491 | 0 | 0.6764 | 0.6661 | 0.7339 | 0.7189 | 0.7263 | 0.7165 | 0.7894 | [[203, 128], [138, 353]] |
| group_feature_v | pair | logistic_regression | 4751 | 822 | 0=1961, 1=2790 | 0=331, 1=491 | 0 | 0.6594 | 0.6376 | 0.7010 | 0.7495 | 0.7244 | 0.7151 | 0.7920 | [[174, 157], [123, 368]] |
| group_feature_v | pair | pytorch_mlp | 4751 | 822 | 0=1961, 1=2790 | 0=331, 1=491 | 0 | 0.7019 | 0.6801 | 0.7312 | 0.7923 | 0.7605 | 0.7541 | 0.8078 | [[188, 143], [102, 389]] |

## Grouped Baseline Comparison

| Model | Grouped ROC-AUC | Delta ROC-AUC vs k-mer | Grouped PR-AUC | Delta PR-AUC vs k-mer |
|---|---:|---:|---:|---:|
| k-mer TF-IDF + logistic regression | 0.7810 | 0.0000 | 0.8236 | 0.0000 |
| frozen AbLang2 pair MLP | 0.7573 | -0.0237 | 0.8099 | -0.0137 |
| logistic_regression_heavy | 0.6835 | -0.0975 | 0.7497 | -0.0739 |
| pytorch_mlp_heavy | 0.7165 | -0.0645 | 0.7894 | -0.0342 |
| logistic_regression_pair | 0.7151 | -0.0659 | 0.7920 | -0.0316 |
| pytorch_mlp_pair | 0.7541 | -0.0269 | 0.8078 | -0.0158 |

## Interpretation

pair worked better by grouped ROC-AUC. Heavy best: pytorch_mlp ROC-AUC 0.7165, PR-AUC 0.7894; pair best: pytorch_mlp ROC-AUC 0.7541, PR-AUC 0.8078.

The MLP improved over logistic regression by ROC-AUC for at least one input. heavy: delta ROC-AUC 0.0330, delta PR-AUC 0.0396; pair: delta ROC-AUC 0.0390, delta PR-AUC 0.0158.

Best frozen pretrained grouped model: `pytorch_mlp_pair` with ROC-AUC 0.7541 and PR-AUC 0.8078.
Beats k-mer grouped ROC-AUC: no.
Beats k-mer grouped PR-AUC: no.
Conclusion: the frozen pretrained representation is a useful benchmark, but it should not be claimed to beat the k-mer baseline unless both grouped ROC-AUC and PR-AUC improve.

## Artifacts

- `reports/metrics/pretrained_frozen_baseline_metrics.json`
- `reports/figures/pretrained_frozen_roc_auc_comparison.png`
- `reports/figures/pretrained_frozen_pr_auc_comparison.png`
- `models/pretrained_frozen_mlp_heavy.pt`
- `models/pretrained_frozen_mlp_pair.pt`
