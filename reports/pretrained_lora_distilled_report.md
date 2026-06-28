# LoRA-Distilled Pretrained Sequence Model

This benchmark trains LoRA adapters plus a classifier head on existing
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
| Max epochs | `15` |
| Seeds | `[1, 7, 42]` |
| Pooling | `['cls', 'mean', 'max']` |

## LoRA

| Field | Value |
|---|---|
| r | `8` |
| alpha | `16` |
| dropout | `0.1` |
| Target modules found | `['query', 'key', 'value', 'dense']` |
| Matched module count | `181` |
| Trainable parameters | `5233153` |
| Total parameters | `425164289` |
| Trainable fraction | `0.012309` |

## Teacher

| Field | Value |
|---|---|
| Method | `group_aware_out_of_fold_on_outer_training_set` |
| Saved model available | `True` |
| Template source | `loaded_saved_model_cloned_for_oof_refits` |
| Out-of-fold probabilities | `True` |
| Final test used for teacher fit | `False` |
| Splitter | `StratifiedGroupKFold` |
| Folds | `5` |

## Data And Splits

| Metric | Value |
|---|---:|
| Rows | 5573 |
| Label 0 count | 2292 |
| Label 1 count | 3281 |
| Grouped train size | 4751 |
| Grouped test size | 822 |
| Train/test group overlap | 0 |
| Inner validation method | grouped |
| Inner train size | 3639 |
| Validation size | 1112 |
| Inner train/validation group overlap | 0 |

## Seed-Wise Results

| Seed | Best epoch | Train loss | Val loss | Val ROC-AUC | Val PR-AUC | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Confusion matrix |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 9 | 0.4865 | 0.5197 | 0.6699 | 0.6965 | 0.6788 | 0.6770 | 0.7539 | 0.6864 | 0.7186 | 0.7313 | 0.8205 | [[221, 110], [154, 337]] |
| 7 | 10 | 0.4808 | 0.5385 | 0.6533 | 0.6818 | 0.6582 | 0.6252 | 0.6842 | 0.7943 | 0.7352 | 0.7102 | 0.7809 | [[151, 180], [101, 390]] |
| 42 | 6 | 0.5141 | 0.5417 | 0.6708 | 0.6934 | 0.6703 | 0.6266 | 0.6786 | 0.8513 | 0.7552 | 0.7357 | 0.8008 | [[133, 198], [73, 418]] |

## Mean +/- Std Across Seeds

| Metric | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| roc_auc | 0.7258 | 0.0136 | 0.7102 | 0.7357 |
| pr_auc | 0.8007 | 0.0198 | 0.7809 | 0.8205 |
| f1 | 0.7363 | 0.0183 | 0.7186 | 0.7552 |
| balanced_accuracy | 0.6429 | 0.0295 | 0.6252 | 0.6770 |

## Baseline Comparison

| Comparison | Result |
|---|---|
| Mean ROC-AUC vs k-mer 0.7810 | `-0.0552` |
| Mean PR-AUC vs k-mer 0.8236 | `-0.0229` |
| Mean ROC-AUC vs frozen pair MLP 0.7541 | `-0.0283` |
| Mean PR-AUC vs frozen pair MLP 0.8078 | `-0.0071` |
| Mean ROC-AUC vs direct fine-tune 0.7443 | `-0.0185` |
| Mean PR-AUC vs direct fine-tune 0.8151 | `-0.0144` |
| Beats k-mer ROC-AUC | `False` |
| Beats k-mer PR-AUC | `False` |
| Overfitting reduced vs previous direct fine-tuning | `True` |

## Conclusion

LoRA distillation beats the k-mer ROC-AUC baseline: no.
LoRA distillation beats the k-mer PR-AUC baseline: no.
Overfitting appears reduced compared with previous direct fine-tuning: True.

## Artifacts

- `reports/metrics/pretrained_lora_distilled_metrics.json`
- `reports/figures/pretrained_lora_distilled_training_curves.png`
- `reports/figures/pretrained_lora_distilled_roc_pr_seed_summary.png`
- `models/pretrained_lora_distilled_best.pt`
