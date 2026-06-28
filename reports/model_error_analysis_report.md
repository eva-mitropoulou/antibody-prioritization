# Model Error Analysis

This analysis compares the best classical k-mer baseline and the saved PyTorch pair-embedding MLP on the same group_feature_v held-out split.

## Grouped Split

- train rows: 4751
- test rows: 822
- train groups: 75
- test groups: 19
- train/test group overlap: 0
- group-column status: ok

## Overall Metrics

| Model | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Confusion matrix |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| kmer | 0.7007 | 0.6747 | 0.7231 | 0.8086 | 0.7635 | 0.7810 | 0.8236 | [[179, 152], [94, 397]] |
| mlp_pair | 0.7056 | 0.6842 | 0.7345 | 0.7943 | 0.7632 | 0.7573 | 0.8099 | [[190, 141], [101, 390]] |

## Comparison With Saved Benchmarks

- saved grouped k-mer ROC-AUC: 0.7810
- reconstructed grouped k-mer ROC-AUC: 0.7810
- saved PyTorch pair MLP ROC-AUC: 0.7573
- reconstructed PyTorch pair MLP ROC-AUC: 0.7573

## Hardest Subgroups

Hardest subgroups are sorted by low balanced accuracy, with subgroup size at least 10.

### k-mer

| Subgroup | Value | n | Positive fraction | Balanced accuracy | F1 |
|---|---|---:|---:|---:|---:|
| metadata_target_region | S; S2 | 33 | 0.030 | 0.4375 | 0.0000 |
| has_light | false | 21 | 0.905 | 0.5000 | 0.9500 |
| is_nanobody_like | true | 21 | 0.905 | 0.5000 | 0.9500 |
| metadata_target_region | S; non-RBD | 59 | 0.169 | 0.5255 | 0.2703 |
| cdrh3_length_bin | long | 91 | 0.571 | 0.5769 | 0.5895 |
| metadata_target_region | S; NTD | 112 | 0.554 | 0.5981 | 0.5818 |
| targets_ntd | true | 113 | 0.558 | 0.6019 | 0.5893 |
| targets_rbd | false | 159 | 0.440 | 0.6422 | 0.5802 |

### MLP pair

| Subgroup | Value | n | Positive fraction | Balanced accuracy | F1 |
|---|---|---:|---:|---:|---:|
| metadata_target_region | S; non-RBD | 59 | 0.169 | 0.3949 | 0.1579 |
| has_light | false | 21 | 0.905 | 0.5000 | 0.9500 |
| is_nanobody_like | true | 21 | 0.905 | 0.5000 | 0.9500 |
| cdrh3_length_bin | long | 91 | 0.571 | 0.5288 | 0.5567 |
| metadata_target_region | S; NTD | 112 | 0.554 | 0.5626 | 0.6250 |
| targets_ntd | true | 113 | 0.558 | 0.5654 | 0.6308 |
| targets_rbd | false | 159 | 0.440 | 0.6151 | 0.5974 |
| cdrh3_length_bin | short | 65 | 0.646 | 0.6263 | 0.8000 |

## Where MLP Improves Over k-mer

| Subgroup | Value | n | k-mer balanced accuracy | MLP balanced accuracy | Delta MLP-kmer |
|---|---|---:|---:|---:|---:|
| metadata_target_region | S; S2 | 33 | 0.4375 | 0.8438 | 0.4062 |
| metadata_target_region | S; RBD | 601 | 0.6504 | 0.7143 | 0.0639 |
| targets_rbd | true | 663 | 0.6598 | 0.6957 | 0.0359 |
| cdrh3_length_bin | medium | 666 | 0.6826 | 0.7086 | 0.0260 |
| targets_ntd | false | 709 | 0.6832 | 0.7031 | 0.0198 |
| heavy_length_bin | normal | 789 | 0.6743 | 0.6867 | 0.0124 |
| has_structure | false | 822 | 0.6747 | 0.6842 | 0.0095 |
| is_nanobody_like | false | 801 | 0.6725 | 0.6818 | 0.0093 |

## Where k-mer Is Stronger

| Subgroup | Value | n | k-mer balanced accuracy | MLP balanced accuracy | Delta MLP-kmer |
|---|---|---:|---:|---:|---:|
| metadata_target_region | S; non-RBD | 59 | 0.5255 | 0.3949 | -0.1306 |
| cdrh3_length_bin | short | 65 | 0.7055 | 0.6263 | -0.0792 |
| cdrh3_length_bin | long | 91 | 0.5769 | 0.5288 | -0.0481 |
| targets_ntd | true | 113 | 0.6019 | 0.5654 | -0.0365 |
| metadata_target_region | S; NTD | 112 | 0.5981 | 0.5626 | -0.0355 |
| heavy_length_bin | short | 31 | 0.6818 | 0.6523 | -0.0295 |
| targets_rbd | false | 159 | 0.6422 | 0.6151 | -0.0271 |

## Subgroup Metrics

| Subgroup | Value | Model | n | Positive fraction | Balanced accuracy | F1 | ROC-AUC | PR-AUC |
|---|---|---|---:|---:|---:|---:|---:|---:|
| cdrh3_length_bin | long | kmer | 91 | 0.571 | 0.5769 | 0.5895 | 0.6563 | 0.7441 |
| cdrh3_length_bin | long | mlp_pair | 91 | 0.571 | 0.5288 | 0.5567 | 0.5888 | 0.6556 |
| cdrh3_length_bin | medium | kmer | 666 | 0.596 | 0.6826 | 0.7727 | 0.7942 | 0.8310 |
| cdrh3_length_bin | medium | mlp_pair | 666 | 0.596 | 0.7086 | 0.7831 | 0.7794 | 0.8237 |
| cdrh3_length_bin | short | kmer | 65 | 0.646 | 0.7055 | 0.8542 | 0.7992 | 0.8454 |
| cdrh3_length_bin | short | mlp_pair | 65 | 0.646 | 0.6263 | 0.8000 | 0.7070 | 0.8137 |
| has_light | false | kmer | 21 | 0.905 | 0.5000 | 0.9500 | 1.0000 | 1.0000 |
| has_light | false | mlp_pair | 21 | 0.905 | 0.5000 | 0.9500 | 0.9474 | 0.9950 |
| has_light | true | kmer | 801 | 0.589 | 0.6725 | 0.7560 | 0.7749 | 0.8130 |
| has_light | true | mlp_pair | 801 | 0.589 | 0.6818 | 0.7556 | 0.7532 | 0.8036 |
| has_structure | false | kmer | 822 | 0.597 | 0.6747 | 0.7635 | 0.7810 | 0.8236 |
| has_structure | false | mlp_pair | 822 | 0.597 | 0.6842 | 0.7632 | 0.7573 | 0.8099 |
| heavy_length_bin | normal | kmer | 789 | 0.594 | 0.6743 | 0.7596 | 0.7815 | 0.8240 |
| heavy_length_bin | normal | mlp_pair | 789 | 0.594 | 0.6867 | 0.7636 | 0.7593 | 0.8089 |
| heavy_length_bin | short | kmer | 31 | 0.645 | 0.6818 | 0.8511 | 0.7955 | 0.8848 |
| heavy_length_bin | short | mlp_pair | 31 | 0.645 | 0.6523 | 0.7907 | 0.7409 | 0.8527 |
| is_nanobody_like | false | kmer | 801 | 0.589 | 0.6725 | 0.7560 | 0.7749 | 0.8130 |
| is_nanobody_like | false | mlp_pair | 801 | 0.589 | 0.6818 | 0.7556 | 0.7532 | 0.8036 |
| is_nanobody_like | true | kmer | 21 | 0.905 | 0.5000 | 0.9500 | 1.0000 | 1.0000 |
| is_nanobody_like | true | mlp_pair | 21 | 0.905 | 0.5000 | 0.9500 | 0.9474 | 0.9950 |
| metadata_target_region | S; NTD | kmer | 112 | 0.554 | 0.5981 | 0.5818 | 0.6800 | 0.6894 |
| metadata_target_region | S; NTD | mlp_pair | 112 | 0.554 | 0.5626 | 0.6250 | 0.5997 | 0.6600 |
| metadata_target_region | S; RBD | kmer | 601 | 0.682 | 0.6504 | 0.8115 | 0.7785 | 0.8595 |
| metadata_target_region | S; RBD | mlp_pair | 601 | 0.682 | 0.7143 | 0.8237 | 0.7768 | 0.8584 |
| metadata_target_region | S; S2 | kmer | 33 | 0.030 | 0.4375 | 0.0000 | 0.7500 | 0.1111 |
| metadata_target_region | S; S2 | mlp_pair | 33 | 0.030 | 0.8438 | 0.1667 | 0.6875 | 0.0909 |
| metadata_target_region | S; non-RBD | kmer | 59 | 0.169 | 0.5255 | 0.2703 | 0.5184 | 0.1783 |
| metadata_target_region | S; non-RBD | mlp_pair | 59 | 0.169 | 0.3949 | 0.1579 | 0.4878 | 0.2557 |
| targets_ntd | false | kmer | 709 | 0.604 | 0.6832 | 0.7845 | 0.8011 | 0.8372 |
| targets_ntd | false | mlp_pair | 709 | 0.604 | 0.7031 | 0.7825 | 0.7752 | 0.8239 |
| targets_ntd | true | kmer | 113 | 0.558 | 0.6019 | 0.5893 | 0.6803 | 0.6916 |
| targets_ntd | true | mlp_pair | 113 | 0.558 | 0.5654 | 0.6308 | 0.6038 | 0.6656 |
| targets_rbd | false | kmer | 159 | 0.440 | 0.6422 | 0.5802 | 0.7448 | 0.6688 |
| targets_rbd | false | mlp_pair | 159 | 0.440 | 0.6151 | 0.5974 | 0.6469 | 0.5829 |
| targets_rbd | true | kmer | 663 | 0.635 | 0.6598 | 0.7899 | 0.7827 | 0.8400 |
| targets_rbd | true | mlp_pair | 663 | 0.635 | 0.6957 | 0.7926 | 0.7718 | 0.8364 |

## Interpretation

The k-mer model is stronger overall on the same V-gene grouped test set by ROC-AUC.
Some subgroups show higher MLP balanced accuracy, but this should be interpreted with subgroup size and label balance in mind.

## Limitations

- Labels are literature-derived.
- Assay conditions are heterogeneous.
- Grouped validation by V-gene reduces but does not eliminate all sequence-family leakage.
- This is classification of existing labeled antibodies, not sequence design.

## Artifacts

- predictions: `reports/model_error_analysis_predictions.csv`
- metrics: `reports/metrics/model_error_analysis_metrics.json`
- subgroup ROC-AUC figure: `reports/figures/subgroup_roc_auc_comparison.png`
- subgroup PR-AUC figure: `reports/figures/subgroup_pr_auc_comparison.png`
- error counts figure: `reports/figures/error_counts_by_target_region.png`
- probability by label figure: `reports/figures/predicted_probability_by_true_label.png`
- probability scatter figure: `reports/figures/kmer_vs_mlp_probability_scatter.png`
