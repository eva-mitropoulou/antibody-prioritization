# Matched Compact K-mer Benchmark Audit

All comparisons in this report use compact character k-mer inputs and
grouped splits with zero train/test group overlap. Full-dataset and
paired-subset results are reported separately.

## full_strict_dataset

Full strict labeled dataset; whole-pair compact k-mer input.

| Split detail | Value |
|---|---:|
| Rows | 5573 |
| Label 0 count | 2292 |
| Label 1 count | 3281 |
| Split strategy | GroupShuffleSplit |
| Group column | `group_feature_v` |
| Train rows | 4751 |
| Test rows | 822 |
| Train group count | 75 |
| Test group count | 19 |
| Group overlap | 0 |

### Input Length Summaries

| Input | Min | Mean | Median | Max | Empty count |
|---|---:|---:|---:|---:|---:|
| whole_pair_compact_kmer | 88 | 222.94 | 232.00 | 442 | 0 |

### Metrics

| Model | Input | ROC-AUC | PR-AUC | Balanced accuracy | F1 |
|---|---|---:|---:|---:|---:|
| majority_baseline | n/a | 0.5000 | 0.5973 | 0.5000 | 0.7479 |
| kmer_logreg | whole_pair_compact_kmer | 0.7800 | 0.8233 | 0.6772 | 0.7654 |

## paired_annotated_subset

Paired annotated subset; whole-pair, region-only, and combined compact k-mer inputs.

| Split detail | Value |
|---|---:|
| Rows | 5092 |
| Label 0 count | 2232 |
| Label 1 count | 2860 |
| Split strategy | GroupShuffleSplit |
| Group column | `group_feature_v` |
| Train rows | 4368 |
| Test rows | 724 |
| Train group count | 70 |
| Test group count | 18 |
| Group overlap | 0 |

### Input Length Summaries

| Input | Min | Mean | Median | Max | Empty count |
|---|---:|---:|---:|---:|---:|
| whole_pair_compact_kmer | 208 | 232.44 | 232.00 | 442 | 0 |
| region_only_compact_kmer | 40 | 52.96 | 53.00 | 101 | 0 |
| whole_pair_plus_region_compact_kmer | 264 | 293.39 | 293.00 | 502 | 0 |

### Metrics

| Model | Input | ROC-AUC | PR-AUC | Balanced accuracy | F1 |
|---|---|---:|---:|---:|---:|
| majority_baseline | n/a | 0.5000 | 0.4890 | 0.5000 | 0.6568 |
| kmer_logreg | whole_pair_compact_kmer | 0.6209 | 0.5889 | 0.5725 | 0.5232 |
| kmer_logreg | region_only_compact_kmer | 0.6629 | 0.6330 | 0.6281 | 0.6174 |
| kmer_logreg | whole_pair_plus_region_compact_kmer | 0.6550 | 0.6145 | 0.6149 | 0.5900 |

## Region Feature Comparison

Subset: paired annotated rows only. Split: same grouped split as the paired block.

| Comparison | Delta ROC-AUC | Delta PR-AUC |
|---|---:|---:|
| region-only minus whole-pair | 0.0420 | 0.0442 |
| whole-pair plus region minus whole-pair | 0.0341 | 0.0256 |

## Artifacts

- `reports/metrics/matched_kmer_benchmark_audit.json`
- `reports/figures/matched_kmer_roc_auc.png`
- `reports/figures/matched_kmer_pr_auc.png`
