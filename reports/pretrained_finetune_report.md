# Pretrained Sequence Model Fine-Tuning

This report fine-tunes a Hugging Face pretrained sequence model on existing
heavy-light pair inputs only. No input sequences were created or altered.

## Setup

| Field | Value |
|---|---|
| Model name | `Exscientia/IgBert` |
| Tokenizer class | `BertTokenizer` |
| Device | `cuda` |
| Tokenization style | `spaced` |
| Max sequence length | `40000` |
| Batch size | `8` |
| Max epochs | `20` |
| Dropout | `0.2` |

## Data And Splits

| Metric | Value |
|---|---:|
| Rows | 5573 |
| Label 0 count | 2292 |
| Label 1 count | 3281 |
| Grouped train size | 4751 |
| Grouped test size | 822 |
| Train groups | 75 |
| Test groups | 19 |
| Train/test group overlap | 0 |
| Inner validation method | grouped |
| Inner train size | 3639 |
| Validation size | 1112 |
| Inner train/validation group overlap | 0 |

## Metrics

| Mode | Valid | Reason | Epochs | Best epoch | Trainable params | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Confusion matrix |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| head_only | true | ok | 20 | 16 | 1025 | 0.6180 | 0.6423 | 0.7674 | 0.5173 | 0.6180 | 0.6926 | 0.7663 | [[254, 77], [237, 254]] |
| last_1_layer | true | ok | 12 | 8 | 12597249 | 0.7032 | 0.6742 | 0.7201 | 0.8228 | 0.7681 | 0.7695 | 0.8317 | [[174, 157], [87, 404]] |
| last_2_layers | true | ok | 6 | 2 | 25193473 | 0.6436 | 0.5884 | 0.6505 | 0.8717 | 0.7450 | 0.7445 | 0.8211 | [[101, 230], [63, 428]] |

## Baseline Comparison

| Model | Grouped ROC-AUC | Delta ROC-AUC vs k-mer | Delta ROC-AUC vs frozen pair MLP | Grouped PR-AUC | Delta PR-AUC vs k-mer | Delta PR-AUC vs frozen pair MLP |
|---|---:|---:|---:|---:|---:|---:|
| k-mer TF-IDF + logistic regression | 0.7810 | 0.0000 | n/a | 0.8236 | 0.0000 | n/a |
| frozen pretrained pair MLP | 0.7541 | -0.0269 | 0.0000 | 0.8078 | -0.0158 | 0.0000 |
| fine-tune head_only | 0.6926 | -0.0884 | -0.0615 | 0.7663 | -0.0573 | -0.0415 |
| fine-tune last_1_layer | 0.7695 | -0.0115 | 0.0154 | 0.8317 | 0.0081 | 0.0239 |
| fine-tune last_2_layers | 0.7445 | -0.0365 | -0.0096 | 0.8211 | -0.0025 | 0.0133 |

## Overfitting Diagnostics

| Mode | Evidence | Best val ROC-AUC | Last val ROC-AUC | Val ROC-AUC drop | Best val loss | Last val loss | Val loss increase |
|---|---:|---:|---:|---:|---:|---:|---:|
| head_only | false | 0.6293 | 0.6282 | 0.0011 | 0.5198 | 0.5226 | 0.0028 |
| last_1_layer | true | 0.6413 | 0.6411 | 0.0002 | 0.5390 | 0.6754 | 0.1365 |
| last_2_layers | true | 0.6344 | 0.6088 | 0.0256 | 0.6268 | 1.6008 | 0.9740 |

## Interpretation

Best fine-tuned mode: `last_1_layer` with grouped ROC-AUC 0.7695 and PR-AUC 0.8317.
Fine-tuning improves over the frozen pretrained pair MLP by ROC-AUC: yes.
Fine-tuning improves over the frozen pretrained pair MLP by PR-AUC: yes.
Fine-tuning beats the k-mer grouped ROC-AUC baseline: no.
Fine-tuning beats the k-mer grouped PR-AUC baseline: yes.
Evidence of overfitting: last_1_layer, last_2_layers.

## Artifacts

- `reports/metrics/pretrained_finetune_metrics.json`
- `reports/figures/pretrained_finetune_training_curves.png`
- `reports/figures/pretrained_finetune_roc_auc_comparison.png`
- `reports/figures/pretrained_finetune_pr_auc_comparison.png`
- `models/pretrained_finetune_best.pt`
