# Final Model Registry

The registry separates full strict dataset comparisons from paired annotated
subset comparisons. It does not treat metrics from different row subsets as
directly comparable.

| Role | Model | Row subset | Rows | Group overlap | ROC-AUC | PR-AUC |
|---|---|---|---:|---:|---:|---:|
| Primary broad scorer | kmer_tfidf_logreg_pair_text | Full strict labeled dataset; whole-pair compact k-mer input. | 5573 | 0 | 0.7800 | 0.8233 |
| Primary paired/region scorer | kmer_tfidf_logreg__paired_annotated_subset__whole_pair_plus_region_compact_kmer | Paired annotated subset; whole-pair, region-only, and combined compact k-mer inputs. | 5092 | 0 | 0.6550 | 0.6145 |
| Best k-mer result | kmer_tfidf_logreg_pair_text | Full strict labeled dataset; whole-pair compact k-mer input. | 5573 | 0 | 0.7800 | 0.8233 |

## Best Pretrained/Embedding Benchmark

| Model/result | Row subset | Rows | ROC-AUC | PR-AUC | Beats matched k-mer |
|---|---|---:|---:|---:|---|
| model_error_analysis | `data/processed/neutral_sequence_classification_ml.csv` | None | 1.0000 | 1.0000 | None |

## Selection Rationale

- Use matched validation performance.
- Do not force pretrained models to win.
- Demote unstable neural models.
- Keep different row subsets separated.
- Prefer the simpler model when performance is practically tied.

Full strict dataset metrics and paired annotated subset metrics are reported separately and are not treated as directly comparable.
