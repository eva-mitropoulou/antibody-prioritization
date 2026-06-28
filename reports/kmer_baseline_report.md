# K-mer Sequence Classification Baseline

Input file: `data/processed/covabdab_neutralisation_ml.csv`

This report describes a generic supervised sequence-classification baseline
trained on an existing labeled table. The workflow does not generate,
design, mutate, optimize, rank, or propose biological sequences.

## Data

| Metric | Value |
|---|---:|
| Rows | 5573 |
| Label 0 count | 2292 |
| Label 1 count | 3281 |

## Split

| Setting | Value |
|---|---:|
| test_size | 0.2 |
| random_state | 42 |
| stratify | label |

## Model

- `TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2)`
- `LogisticRegression(max_iter=5000, class_weight="balanced")`
- Majority-class baseline: `DummyClassifier(strategy="most_frequent")`

## Metrics

| Model | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC | Average precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| majority_baseline | 0.5883 | 0.5000 | 0.5883 | 1.0000 | 0.7408 | 0.5000 | 0.5883 |
| heavy_only | 0.6574 | 0.6653 | 0.7537 | 0.6204 | 0.6806 | 0.7263 | 0.7934 |
| pair_text | 0.6816 | 0.6931 | 0.7878 | 0.6280 | 0.6989 | 0.7534 | 0.8157 |

## Confusion Matrices

### majority_baseline

| True label | Predicted 0 | Predicted 1 |
|---|---:|---:|
| 0 | 0 | 459 |
| 1 | 0 | 656 |

### heavy_only

| True label | Predicted 0 | Predicted 1 |
|---|---:|---:|
| 0 | 326 | 133 |
| 1 | 249 | 407 |

### pair_text

| True label | Predicted 0 | Predicted 1 |
|---|---:|---:|
| 0 | 348 | 111 |
| 1 | 244 | 412 |

## Artifacts

- `reports/metrics/kmer_baseline_metrics.json`
- `reports/figures/kmer_baseline_confusion_matrix_heavy_only.png`
- `reports/figures/kmer_baseline_confusion_matrix_pair_text.png`
- `reports/figures/kmer_baseline_roc_curve.png`
- `reports/figures/kmer_baseline_pr_curve.png`
- `models/kmer_logreg_heavy_only.joblib`
- `models/kmer_logreg_pair_text.joblib`
