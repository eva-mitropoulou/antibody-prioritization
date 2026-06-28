# Pretrained Sequence-Model Benchmark Registry

This registry uses existing metric files only. It does not rerun neural
training. Each row states the reported subset, split context, label
balance, and whether a same-row-count matched k-mer reference was beaten.

| Model/result | Row subset | Row count | Split | Group overlap | ROC-AUC | PR-AUC | Beats matched k-mer |
|---|---|---:|---|---:|---:|---:|---|
| embedding_baseline | `data/processed/embeddings_ablang2_metadata.csv` | 5573 | 0.2 | None | 0.7382 | 0.8211 | false |
| pytorch_embedding_mlp | `data/processed/embeddings_ablang2_metadata.csv` | 5573 | 0.2 | None | 0.7573 | 0.8099 | false |
| pretrained_frozen_baseline | `data/processed/pretrained_embeddings/metadata.csv` | 5573 | 0.2 | None | 0.7541 | 0.8078 | false |
| pretrained_finetune | `data/processed/neutral_sequence_classification_ml.csv` | 5573 | 0.2 | 0 | 0.7695 | 0.8317 | false |
| pretrained_finetune_seed_check | `data/processed/neutral_sequence_classification_ml.csv` | 5573 | reported_in_source_metrics | 0 | 0.7443 | 0.8151 | false |
| pretrained_lora_distilled | `data/processed/neutral_sequence_classification_ml.csv` | 5573 | reported_in_source_metrics | 0 | 0.7258 | 0.8007 | false |
| bioaware_igbert_final | `data/processed/neutral_sequence_classification_ml.csv` | n/a | reported_in_source_metrics | 0 | 0.6061 | 0.6056 | not same-subset comparable |
| hybrid_baseline | `data/processed/neutral_sequence_classification_ml.csv` | 5573 | 0.2 | None | 0.7510 | 0.8226 | false |

## Interpretation

Pretrained and embedding models are benchmark evidence, not automatically primary scorers. None reliably replaces the matched k-mer references on both primary metrics.

Invalid cross-subset comparisons are not used for model selection.

Diagnostic error-analysis artifacts are excluded from pretrained-model selection.
