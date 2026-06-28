# Bioaware IgBert Final Report

This final benchmark trains a biologically informed IgBert classifier on
paired antibodies only, with explicit all-six-CDR marker tokens and CDR
token pooling. Target-region metadata is used only for analysis.

## Setup

| Field | Value |
|---|---|
| Model name | `Exscientia/IgBert` |
| Tokenizer class | `BertTokenizer` |
| Model class | `BertModel` |
| Device | `cuda` |
| Training mode | `lora_cdr_marked` |
| Marker tokens available | `True` |
| New marker tokens added | `12` |
| Batch size | `8` |
| Max epochs | `20` |
| Seeds | `[1, 7, 42]` |

## Annotation And Dataset

| Metric | Value |
|---|---:|
| All-six-CDR annotation worked | True |
| All six CDRs found | 5091 |
| All-six-CDR coverage | 99.98% |
| Paired-antibody rows retained | 5091 |
| Nanobody-like/light-missing rows excluded | 481 |
| Annotation failure count | 1 |
| Marker insertion failure count | 1 |

Nanobody-like rows were excluded because the primary model is a paired
heavy-light model and nanobodies lack a paired light chain. They require a
separate heavy-only/VHH model.

## Tokenization Verification

Special token IDs: `{'<CDRH1_START>': 30, '<CDRH1_END>': 31, '<CDRH2_START>': 32, '<CDRH2_END>': 33, '<CDRH3_START>': 34, '<CDRH3_END>': 35, '<CDRL1_START>': 36, '<CDRL1_END>': 37, '<CDRL2_START>': 38, '<CDRL2_END>': 39, '<CDRL3_START>': 40, '<CDRL3_END>': 41}`

Example marked heavy sequence:

```text
V Q L V E S G G G L V Q P G G S L R L S C A A S <CDRH1_START> G L T V S S N Y <CDRH1_END> M N W V R Q A P G K G L E W V S V <CDRH2_START> F Y P G G S T <CDRH2_END> F Y A D S V R G R F T I S R D N S K N T L Y L Q M N S L R A E D T A V Y Y C <CDRH3_START> A R D H S G H A L D I <CDRH3_END> W G Q G T M V T V S
```

Example marked light sequence:

```text
D I Q M T Q S P S F L S A S V G D R V T I T C R A S <CDRL1_START> Q G I S S Y <CDRL1_END> L A W Y Q Q K P G K A P K L L I Y <CDRL2_START> A A S <CDRL2_END> T L Q S G V P S R F S G S G S G T E F T L T I S S L Q P E D F A T Y Y C <CDRL3_START> Q H L N S Y P S M Y T <CDRL3_END> F G Q G T K V D I
```

Tokenized example snippet:

```text
[CLS] V Q L V E S G G G L V Q P G G S L R L S C A A S <CDRH1_START> G L T V S S N Y <CDRH1_END> M N W V R Q A P G K G L E W V S V <CDRH2_START> F Y P G G S T <CDRH2_END> F Y A D S V R G R F T I S R D N S K N T L Y L Q M N S L R A E D T A V Y Y C <CDRH3_START> A R D H S G H A L D I <CDRH3_END> W G Q G T M V T
```

## Leakage Diagnostics

| Metric | Value |
|---|---:|
| Outer train rows | 4368 |
| Outer test rows | 723 |
| Train/test group overlap | 0 |
| Inner validation method | grouped |
| Inner train/validation group overlap | 0 |
| Clonotype-like grouping usable | False |
| Clonotype-like grouping reason | `near_row_unique` |

## Seed-Wise Results

| Seed | Best epoch | Train loss | Val loss | Val ROC-AUC | Val PR-AUC | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Confusion matrix |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 4 | 0.5261 | 0.6261 | 0.6341 | 0.7353 | 0.5754 | 0.5774 | 0.5545 | 0.6751 | 0.6089 | 0.6241 | 0.6441 | [[177, 192], [115, 239]] |
| 7 | 5 | 0.4485 | 0.7488 | 0.5799 | 0.6815 | 0.5615 | 0.5638 | 0.5421 | 0.6723 | 0.6003 | 0.5914 | 0.5833 | [[168, 201], [116, 238]] |
| 42 | 3 | 0.6035 | 0.6136 | 0.6109 | 0.7046 | 0.5823 | 0.5819 | 0.5747 | 0.5650 | 0.5698 | 0.6029 | 0.5895 | [[221, 148], [154, 200]] |

## Mean +/- Std Across Seeds

| Metric | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| roc_auc | 0.6061 | 0.0166 | 0.5914 | 0.6241 |
| pr_auc | 0.6056 | 0.0335 | 0.5833 | 0.6441 |
| f1 | 0.5930 | 0.0205 | 0.5698 | 0.6089 |
| balanced_accuracy | 0.5744 | 0.0094 | 0.5638 | 0.5819 |

## Same-Subset K-mer Baselines

| Baseline | ROC-AUC | PR-AUC | F1 | Balanced accuracy | Confusion matrix |
|---|---:|---:|---:|---:|---|
| whole_pair_kmer | 0.6208 | 0.5891 | 0.5263 | 0.5748 | [[247, 122], [184, 170]] |
| all_cdr_kmer | 0.6714 | 0.6408 | 0.6398 | 0.6419 | [[234, 135], [124, 230]] |
| cdrh3_cdrl3_kmer | 0.5946 | 0.5728 | 0.5460 | 0.5690 | [[225, 144], [167, 187]] |

## Subgroup Analysis

| Subgroup | Rows | Positive fraction | ROC-AUC | PR-AUC | Reason if unavailable |
|---|---:|---:|---:|---:|---|
| metadata_target_region:N | 1 | 1.0000 | n/a | n/a | too_few_rows |
| metadata_target_region:S; NTD | 132 | 0.4470 | 0.4676 | 0.4510 |  |
| metadata_target_region:S; RBD | 454 | 0.6101 | 0.6401 | 0.7464 |  |
| metadata_target_region:S; RBD/non-RBD | 1 | 1.0000 | n/a | n/a | too_few_rows |
| metadata_target_region:S; S1 non-RBD | 2 | 0.0000 | n/a | n/a | too_few_rows |
| metadata_target_region:S; S2 | 45 | 0.0222 | 0.1364 | 0.0256 |  |
| metadata_target_region:S; S2' Cleavage Site/Fusion Peptide NTD | 1 | 1.0000 | n/a | n/a | too_few_rows |
| metadata_target_region:S; Unk | 9 | 0.4444 | n/a | n/a | too_few_rows |
| metadata_target_region:S; non-RBD | 69 | 0.1449 | 0.4627 | 0.1517 |  |
| metadata_target_region:S; non-S1 | 9 | 0.0000 | n/a | n/a | too_few_rows |
| structure:has_structure | 0 | n/a | n/a | n/a | too_few_rows |
| structure:without_structure | 723 | 0.4896 | 0.6241 | 0.6441 |  |

## Required Conclusions

1. Did all-six-CDR annotation work? Yes.
2. Paired-antibody rows retained: 5091.
3. Nanobody-like/light-missing rows excluded: 481; they require a separate heavy-only/VHH model.
4. Did all-CDR k-mer beat whole-pair k-mer? ROC-AUC: True; PR-AUC: True.
5. Did CDR-aware IgBert beat paired-subset k-mer on ROC-AUC? False.
6. Did CDR-aware IgBert beat paired-subset k-mer on PR-AUC? True.
7. Did explicit CDR markers and CDR pooling reduce overfitting? True.
8. Did target-region subgroup analysis show biological heterogeneity? See subgroup metrics above; heterogeneous class balance or metrics indicate target-region dependence.
9. Honest final conclusion: if k-mers remain stronger on the same paired subset, the sequence-local CDR motif signal is still better captured by sparse k-mer features than by this parameter-efficient IgBert setup under grouped validation.

## Artifacts

- `data/processed/bioaware_paired_cdr_annotated.csv`
- `reports/bioaware_cdr_annotation_report.md`
- `reports/metrics/bioaware_igbert_final_metrics.json`
- `reports/figures/bioaware_igbert_final_seed_summary.png`
- `reports/figures/bioaware_igbert_final_training_curves.png`
- `reports/figures/bioaware_igbert_final_subgroup_metrics.png`
- `reports/figures/bioaware_cdr_coverage.png`
- `models/bioaware_igbert_final_best.pt`
