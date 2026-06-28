# Hybrid Feature Baseline

Input file: `data/processed/neutral_sequence_classification_ml.csv`
Pair embeddings: `data/processed/embeddings_ablang2_pair.npy`

This report evaluates logistic-regression classifiers on existing labeled
rows using k-mer TF-IDF features, cached AbLang2 embeddings, simple
features, and hybrid combinations.

## Data

| Metric | Value |
|---|---:|
| Rows | 5573 |
| Label 0 count | 2292 |
| Label 1 count | 3281 |
| Pair embedding shape | [5573, 960] |

## Best Grouped Models

| Selection | Feature set | Value |
|---|---|---:|
| Grouped ROC-AUC | kmer_only | 0.7794 |
| Grouped PR-AUC | hybrid_kmer_plus_simple | 0.8226 |

## Comparison Table

| Split | Feature set | Features | Train size | Test size | Group overlap | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Confusion matrix |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| random | kmer_only | 73178 | 4458 | 1115 | n/a | 0.6816 | 0.6934 | 0.7889 | 0.6265 | 0.6984 | 0.7531 | 0.8156 | [[349, 110], [245, 411]] |
| random | ablang2_pair_only | 960 | 4458 | 1115 | n/a | 0.6700 | 0.6701 | 0.7441 | 0.6692 | 0.7047 | 0.7079 | 0.7591 | [[308, 151], [217, 439]] |
| random | simple_features_only | 10 | 4458 | 1115 | n/a | 0.6269 | 0.6113 | 0.6770 | 0.6997 | 0.6882 | 0.6700 | 0.7521 | [[240, 219], [197, 459]] |
| random | hybrid_kmer_plus_simple | 73188 | 4458 | 1115 | n/a | 0.6843 | 0.6892 | 0.7695 | 0.6616 | 0.7115 | 0.7668 | 0.8252 | [[329, 130], [222, 434]] |
| random | hybrid_kmer_plus_ablang2 | 74138 | 4458 | 1115 | n/a | 0.6717 | 0.6723 | 0.7466 | 0.6692 | 0.7058 | 0.7182 | 0.7680 | [[310, 149], [217, 439]] |
| random | hybrid_all | 74148 | 4458 | 1115 | n/a | 0.6735 | 0.6702 | 0.7386 | 0.6890 | 0.7129 | 0.7278 | 0.7786 | [[299, 160], [204, 452]] |
| group_feature_v | kmer_only | 76154 | 4751 | 822 | 0 | 0.6946 | 0.6681 | 0.7182 | 0.8045 | 0.7589 | 0.7794 | 0.8223 | [[176, 155], [96, 395]] |
| group_feature_v | ablang2_pair_only | 960 | 4751 | 822 | 0 | 0.6727 | 0.6517 | 0.7118 | 0.7597 | 0.7350 | 0.7114 | 0.7859 | [[180, 151], [118, 373]] |
| group_feature_v | simple_features_only | 10 | 4751 | 822 | 0 | 0.6387 | 0.6134 | 0.6810 | 0.7434 | 0.7108 | 0.7009 | 0.7697 | [[160, 171], [126, 365]] |
| group_feature_v | hybrid_kmer_plus_simple | 76164 | 4751 | 822 | 0 | 0.6740 | 0.6513 | 0.7100 | 0.7678 | 0.7378 | 0.7510 | 0.8226 | [[177, 154], [114, 377]] |
| group_feature_v | hybrid_kmer_plus_ablang2 | 77114 | 4751 | 822 | 0 | 0.6752 | 0.6528 | 0.7113 | 0.7678 | 0.7385 | 0.7227 | 0.7953 | [[178, 153], [114, 377]] |
| group_feature_v | hybrid_all | 77124 | 4751 | 822 | 0 | 0.6715 | 0.6547 | 0.7179 | 0.7413 | 0.7295 | 0.7194 | 0.7950 | [[188, 143], [127, 364]] |

## Interpretation

AbLang2 adds value beyond k-mers: no.
Simple features improve grouped ROC-AUC over k-mers alone: no.
Final hybrid improves over the previous k-mer baseline: no.
Best grouped ROC-AUC: `kmer_only` (0.7794).
Best grouped PR-AUC: `hybrid_kmer_plus_simple` (0.8226).

## Artifacts

- `reports/metrics/hybrid_baseline_metrics.json`
- `reports/figures/hybrid_roc_auc_comparison.png`
- `reports/figures/hybrid_pr_auc_comparison.png`
- `reports/figures/hybrid_f1_comparison.png`
- `models/hybrid_best_model.joblib`
