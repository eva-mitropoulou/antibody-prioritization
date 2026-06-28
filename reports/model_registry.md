# Final Model Registry

The registry separates full strict dataset comparisons from paired annotated
subset comparisons. Metrics from different row subsets are kept separate.

| Role | Model | Row subset | Rows | Group overlap | ROC-AUC | PR-AUC |
|---|---|---|---:|---:|---:|---:|
| Primary broad scorer | kmer_tfidf_logreg_pair_text | Full strict labeled dataset; whole-pair compact k-mer input. | 5573 | 0 | 0.7800 | 0.8233 |
| Primary paired/region scorer | kmer_tfidf_logreg__paired_annotated_subset__region_only_compact_kmer | Paired annotated subset; whole-pair, region-only, and combined compact k-mer inputs. | 5092 | 0 | 0.6629 | 0.6330 |
| Best k-mer result | kmer_tfidf_logreg_pair_text | Full strict labeled dataset; whole-pair compact k-mer input. | 5573 | 0 | 0.7800 | 0.8233 |

## Best Pretrained/Embedding Benchmark

| Model/result | Row subset | Rows | ROC-AUC | PR-AUC | Beats matched k-mer |
|---|---|---:|---:|---:|---|
| pretrained_finetune | `data/processed/neutral_sequence_classification_ml.csv` | 5573 | 0.7695 | 0.8317 | {'both_primary_metrics': False, 'pr_auc': True, 'roc_auc': False} |

## Selection Rationale

- Use matched validation performance.
- Do not force pretrained models to win.
- Demote unstable neural models.
- Keep different row subsets separated.
- Prefer the simpler model when performance is practically tied.
- Exclude diagnostic error-analysis artifacts from pretrained-model selection.

Full strict dataset metrics and paired annotated subset metrics are reported separately and are not treated as directly comparable.

Actual pretrained and embedding benchmarks are retained as benchmark evidence only; none reliably replaces the matched k-mer references on both primary metrics.
