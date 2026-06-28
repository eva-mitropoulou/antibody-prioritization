# Pretrained Fine-Tuning Seed Check

This report repeats only the best fine-tuning configuration,
`last_1_layer`, on existing heavy-light pair inputs. No input sequences
were created or altered.

## Setup

| Field | Value |
|---|---|
| Model name | `Exscientia/IgBert` |
| Tokenizer class | `BertTokenizer` |
| Device | `cuda` |
| Mode | `last_1_layer` |
| Training seeds | `1, 7, 42, 123, 2026` |
| Batch size | `8` |
| Max epochs | `20` |
| Early stopping patience | `4` |
| Tokenization style | `spaced` |
| Max sequence length | `40000` |

## Data And Splits

| Metric | Value |
|---|---:|
| Rows | 5573 |
| Label 0 count | 2292 |
| Label 1 count | 3281 |
| Grouped train size | 4751 |
| Grouped test size | 822 |
| Train/test split random state | 42 |
| Train groups | 75 |
| Test groups | 19 |
| Train/test group overlap | 0 |
| Inner validation method | grouped |
| Inner validation split random state | 142 |
| Inner train size | 3639 |
| Validation size | 1112 |
| Inner train/validation group overlap | 0 |

The outer grouped train/test split uses the same helper and random state
as `train_pretrained_finetune.py`, so the test split is held fixed across
all seed runs.

## Per-Seed Metrics

| Seed | Best epoch | Train loss at best | Val loss at best | Val ROC-AUC at best | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Confusion matrix |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 7 | 0.4009 | 0.5619 | 0.6567 | 0.6606 | 0.6726 | 0.7732 | 0.6110 | 0.6826 | 0.7468 | 0.8182 | [[243, 88], [191, 300]] |
| 7 | 10 | 0.3534 | 0.5923 | 0.6540 | 0.6800 | 0.6529 | 0.7073 | 0.7923 | 0.7474 | 0.7444 | 0.8151 | [[170, 161], [102, 389]] |
| 42 | 11 | 0.3546 | 0.6284 | 0.6438 | 0.6910 | 0.6852 | 0.7548 | 0.7149 | 0.7343 | 0.7462 | 0.8155 | [[217, 114], [140, 351]] |
| 123 | 10 | 0.3571 | 0.5844 | 0.6502 | 0.6873 | 0.6576 | 0.7082 | 0.8106 | 0.7559 | 0.7448 | 0.8169 | [[167, 164], [93, 398]] |
| 2026 | 15 | 0.2918 | 0.9031 | 0.6519 | 0.6691 | 0.6147 | 0.6662 | 0.8941 | 0.7635 | 0.7392 | 0.8097 | [[111, 220], [52, 439]] |

## Aggregate Metrics

| Metric | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| ROC-AUC | 0.7443 | 0.0030 | 0.7392 | 0.7468 |
| PR-AUC | 0.8151 | 0.0032 | 0.8097 | 0.8182 |
| F1 | 0.7367 | 0.0321 | 0.6826 | 0.7635 |
| Balanced accuracy | 0.6566 | 0.0267 | 0.6147 | 0.6852 |

## Baseline Comparison

| Model | ROC-AUC | Delta ROC-AUC vs k-mer | Delta ROC-AUC vs frozen pair MLP | PR-AUC | Delta PR-AUC vs k-mer | Delta PR-AUC vs frozen pair MLP |
|---|---:|---:|---:|---:|---:|---:|
| k-mer TF-IDF + logistic regression | 0.7810 | 0.0000 | n/a | 0.8236 | 0.0000 | n/a |
| frozen pretrained pair MLP | 0.7541 | -0.0269 | 0.0000 | 0.8078 | -0.0158 | 0.0000 |
| fine-tune last_1_layer seed mean | 0.7443 | -0.0367 | -0.0098 | 0.8151 | -0.0085 | 0.0073 |

## Conclusion

Stability: stable by the predefined std rule (ROC-AUC std 0.0030, PR-AUC std 0.0032).
Reliably beats frozen embeddings: no (requires every valid seed to beat frozen pair MLP on both ROC-AUC and PR-AUC).
Reliably beats k-mer on PR-AUC: no (requires every valid seed to beat the k-mer PR-AUC baseline).
Reliably beats k-mer on ROC-AUC: no.
Overfitting: present in 5/5 valid seed runs.

## Artifacts

- `reports/metrics/pretrained_finetune_seed_check_metrics.json`
- `reports/figures/pretrained_finetune_seed_check_roc_pr.png`
