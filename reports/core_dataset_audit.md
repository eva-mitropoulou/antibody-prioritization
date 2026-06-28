# Core Dataset Audit

Aggregate-only audit of the core sequence-record datasets. No raw records,
sequence strings, or source links are included.

## Strict Labeled ML Dataset

| Metric | Value |
|---|---:|
| Rows | 5573 |
| Columns | 31 |
| Label 0 count | 2292 |
| Label 1 count | 3281 |
| Target-region metadata available | 5563 |
| Structure metadata available | 5164 |

### Columns

`sample_name`, `sample_type`, `label`, `sequence_a`, `sequence_b`, `sequence_a_raw`, `sequence_pair_text`, `sequence_key`, `group_feature_v`, `group_feature_j`, `group_feature_b_v`, `group_feature_b_j`, `group_feature_cdr3`, `group_feature_b_cdr3`, `metadata_target_region`, `metadata_origin`, `extra_col_10`, `extra_col_11`, `extra_col_12`, `extra_col_13`, `extra_col_14`, `extra_col_15`, `extra_col_16`, `extra_col_17`, `extra_col_18`, `extra_col_19`, `extra_col_20`, `extra_col_21`, `extra_col_22`, `extra_col_23`, `extra_col_24`

### Paired/Light-Missing Counts

| Status | Count |
|---|---:|
| paired | 5092 |
| light_missing_or_single_chain | 481 |

### Target-Region Group Counts

| Group | Count |
|---|---:|
| RBD | 4820 |
| NTD | 431 |
| other | 311 |
| unknown | 10 |
| Spike/S | 1 |

### Domain/Region Columns

`group_feature_cdr3`, `group_feature_b_cdr3`, `metadata_target_region`

### Missing-Value Counts

| Column | Missing count |
|---|---:|
| `sample_name` | 0 |
| `sample_type` | 0 |
| `label` | 0 |
| `sequence_a` | 0 |
| `sequence_b` | 481 |
| `sequence_a_raw` | 0 |
| `sequence_pair_text` | 0 |
| `sequence_key` | 0 |
| `group_feature_v` | 163 |
| `group_feature_j` | 163 |
| `group_feature_b_v` | 481 |
| `group_feature_b_j` | 481 |
| `group_feature_cdr3` | 0 |
| `group_feature_b_cdr3` | 326 |
| `metadata_target_region` | 10 |
| `metadata_origin` | 183 |
| `extra_col_10` | 0 |
| `extra_col_11` | 481 |
| `extra_col_12` | 0 |
| `extra_col_13` | 326 |
| `extra_col_14` | 0 |
| `extra_col_15` | 0 |
| `extra_col_16` | 0 |
| `extra_col_17` | 0 |
| `extra_col_18` | 0 |
| `extra_col_19` | 0 |
| `extra_col_20` | 0 |
| `extra_col_21` | 20 |
| `extra_col_22` | 0 |
| `extra_col_23` | 0 |
| `extra_col_24` | 0 |

## Broader Prepared Dataset

| Metric | Value |
|---|---:|
| Rows | 11748 |
| Columns | 27 |
| Label 0 count | 2292 |
| Label 1 count | 5646 |
| Target-region metadata available | 11738 |
| Structure metadata available | 659 |

### Columns

`sample_name`, `sample_type`, `sequence_a`, `sequence_b`, `sequence_a_raw`, `sequence_key`, `group_feature_cdr3`, `group_feature_b_cdr3`, `metadata_target_region`, `extra_col_07`, `extra_col_08`, `extra_col_09`, `extra_col_10`, `extra_col_11`, `extra_col_12`, `extra_col_13`, `extra_col_14`, `extra_col_15`, `extra_col_16`, `extra_col_17`, `extra_col_18`, `extra_col_19`, `extra_col_20`, `extra_col_21`, `extra_col_22`, `extra_col_23`, `extra_col_24`

### Paired/Light-Missing Counts

| Status | Count |
|---|---:|
| paired | 9986 |
| light_missing_or_single_chain | 1762 |

### Target-Region Group Counts

| Group | Count |
|---|---:|
| RBD | 8147 |
| other | 2770 |
| NTD | 820 |
| unknown | 10 |
| Spike/S | 1 |

### Domain/Region Columns

`group_feature_cdr3`, `group_feature_b_cdr3`, `metadata_target_region`

### Missing-Value Counts

| Column | Missing count |
|---|---:|
| `sample_name` | 0 |
| `sample_type` | 0 |
| `sequence_a` | 0 |
| `sequence_b` | 1762 |
| `sequence_a_raw` | 0 |
| `sequence_key` | 0 |
| `group_feature_cdr3` | 0 |
| `group_feature_b_cdr3` | 676 |
| `metadata_target_region` | 10 |
| `extra_col_07` | 3810 |
| `extra_col_08` | 0 |
| `extra_col_09` | 0 |
| `extra_col_10` | 45 |
| `extra_col_11` | 11089 |
| `extra_col_12` | 0 |
| `extra_col_13` | 0 |
| `extra_col_14` | 0 |
| `extra_col_15` | 0 |
| `extra_col_16` | 0 |
| `extra_col_17` | 0 |
| `extra_col_18` | 0 |
| `extra_col_19` | 0 |
| `extra_col_20` | 0 |
| `extra_col_21` | 0 |
| `extra_col_22` | 0 |
| `extra_col_23` | 0 |
| `extra_col_24` | 0 |

## Annotated Paired Dataset

| Metric | Value |
|---|---:|
| Rows | 5092 |
| Columns | 80 |
| Label 0 count | 2232 |
| Label 1 count | 2860 |
| Target-region metadata available | 5082 |
| Structure metadata available | 5092 |

### Columns

`sample_name`, `sample_type`, `label`, `sequence_a`, `sequence_b`, `sequence_a_raw`, `sequence_pair_text`, `sequence_key`, `group_feature_v`, `group_feature_j`, `group_feature_b_v`, `group_feature_b_j`, `group_feature_cdr3`, `group_feature_b_cdr3`, `metadata_target_region`, `metadata_origin`, `extra_col_10`, `extra_col_11`, `extra_col_12`, `extra_col_13`, `extra_col_14`, `extra_col_15`, `extra_col_16`, `extra_col_17`, `extra_col_18`, `extra_col_19`, `extra_col_20`, `extra_col_21`, `extra_col_22`, `extra_col_23`, `extra_col_24`, `row_id`, `heavy_sequence`, `light_sequence`, `has_light_bool`, `is_nanobody_like_bool`, `existing_cdrh3`, `existing_cdrl3`, `heavy_annotation_ok`, `light_annotation_ok`, `heavy_annotation_error`, `light_annotation_error`, `heavy_chain_type`, `light_chain_type`, `cdrh1_seq`, `cdrh2_seq`, `cdrh3_seq`, `cdrl1_seq`, `cdrl2_seq`, `cdrl3_seq`, `cdrh1_found`, `cdrh2_found`, `cdrh3_found`, `cdrl1_found`, `cdrl2_found`, `cdrl3_found`, `all_six_cdrs_found`, `marked_heavy_text`, `marked_light_text`, `heavy_marker_insertion_ok`, `light_marker_insertion_ok`, `heavy_marker_ambiguous_regions`, `light_marker_ambiguous_regions`, `marker_insertion_ok`, `whole_pair_kmer_text`, `all_cdr_kmer_text`, `cdrh3_cdrl3_kmer_text`, `heavy_length`, `light_length`, `cdrh1_length`, `cdrh2_length`, `cdrh3_length`, `cdrl1_length`, `cdrl2_length`, `cdrl3_length`, `hydrophobic_fraction_cdrh3`, `hydrophobic_fraction_cdrl3`, `cysteine_count_cdrh3`, `cysteine_count_cdrl3`, `n_glycosylation_motif_count_heavy`

### Paired/Light-Missing Counts

| Status | Count |
|---|---:|
| paired | 5092 |

### Target-Region Group Counts

| Group | Count |
|---|---:|
| RBD | 4386 |
| NTD | 423 |
| other | 272 |
| unknown | 10 |
| Spike/S | 1 |

### Domain/Region Columns

`group_feature_cdr3`, `group_feature_b_cdr3`, `metadata_target_region`, `existing_cdrh3`, `existing_cdrl3`, `heavy_annotation_ok`, `light_annotation_ok`, `heavy_annotation_error`, `light_annotation_error`, `heavy_chain_type`, `light_chain_type`, `cdrh1_seq`, `cdrh2_seq`, `cdrh3_seq`, `cdrl1_seq`, `cdrl2_seq`, `cdrl3_seq`, `cdrh1_found`, `cdrh2_found`, `cdrh3_found`, `cdrl1_found`, `cdrl2_found`, `cdrl3_found`, `all_six_cdrs_found`, `heavy_marker_insertion_ok`, `light_marker_insertion_ok`, `heavy_marker_ambiguous_regions`, `light_marker_ambiguous_regions`, `marker_insertion_ok`, `all_cdr_kmer_text`, `cdrh3_cdrl3_kmer_text`, `cdrh1_length`, `cdrh2_length`, `cdrh3_length`, `cdrl1_length`, `cdrl2_length`, `cdrl3_length`, `hydrophobic_fraction_cdrh3`, `hydrophobic_fraction_cdrl3`, `cysteine_count_cdrh3`, `cysteine_count_cdrl3`

### Missing-Value Counts

| Column | Missing count |
|---|---:|
| `sample_name` | 0 |
| `sample_type` | 0 |
| `label` | 0 |
| `sequence_a` | 0 |
| `sequence_b` | 0 |
| `sequence_a_raw` | 0 |
| `sequence_pair_text` | 0 |
| `sequence_key` | 0 |
| `group_feature_v` | 0 |
| `group_feature_j` | 0 |
| `group_feature_b_v` | 0 |
| `group_feature_b_j` | 0 |
| `group_feature_cdr3` | 0 |
| `group_feature_b_cdr3` | 5 |
| `metadata_target_region` | 10 |
| `metadata_origin` | 20 |
| `extra_col_10` | 0 |
| `extra_col_11` | 0 |
| `extra_col_12` | 0 |
| `extra_col_13` | 5 |
| `extra_col_14` | 0 |
| `extra_col_15` | 0 |
| `extra_col_16` | 0 |
| `extra_col_17` | 0 |
| `extra_col_18` | 0 |
| `extra_col_19` | 0 |
| `extra_col_20` | 0 |
| `extra_col_21` | 20 |
| `extra_col_22` | 0 |
| `extra_col_23` | 0 |
| `extra_col_24` | 0 |
| `row_id` | 0 |
| `heavy_sequence` | 0 |
| `light_sequence` | 0 |
| `has_light_bool` | 0 |
| `is_nanobody_like_bool` | 0 |
| `existing_cdrh3` | 0 |
| `existing_cdrl3` | 5 |
| `heavy_annotation_ok` | 0 |
| `light_annotation_ok` | 0 |
| `heavy_annotation_error` | 5092 |
| `light_annotation_error` | 5092 |
| `heavy_chain_type` | 0 |
| `light_chain_type` | 0 |
| `cdrh1_seq` | 0 |
| `cdrh2_seq` | 0 |
| `cdrh3_seq` | 0 |
| `cdrl1_seq` | 0 |
| `cdrl2_seq` | 0 |
| `cdrl3_seq` | 1 |
| `cdrh1_found` | 0 |
| `cdrh2_found` | 0 |
| `cdrh3_found` | 0 |
| `cdrl1_found` | 0 |
| `cdrl2_found` | 0 |
| `cdrl3_found` | 0 |
| `all_six_cdrs_found` | 0 |
| `marked_heavy_text` | 0 |
| `marked_light_text` | 0 |
| `heavy_marker_insertion_ok` | 0 |
| `light_marker_insertion_ok` | 0 |
| `heavy_marker_ambiguous_regions` | 5092 |
| `light_marker_ambiguous_regions` | 5085 |
| `marker_insertion_ok` | 0 |
| `whole_pair_kmer_text` | 0 |
| `all_cdr_kmer_text` | 0 |
| `cdrh3_cdrl3_kmer_text` | 0 |
| `heavy_length` | 0 |
| `light_length` | 0 |
| `cdrh1_length` | 0 |
| `cdrh2_length` | 0 |
| `cdrh3_length` | 0 |
| `cdrl1_length` | 0 |
| `cdrl2_length` | 0 |
| `cdrl3_length` | 0 |
| `hydrophobic_fraction_cdrh3` | 0 |
| `hydrophobic_fraction_cdrl3` | 0 |
| `cysteine_count_cdrh3` | 0 |
| `cysteine_count_cdrl3` | 0 |
| `n_glycosylation_motif_count_heavy` | 0 |

## Quality Gates

| Gate | Pass |
|---|---:|
| strict_labeled_row_count_reported | true |
| broader_row_count_reported | true |
| both_classes_counted | true |
| paired_and_light_missing_counted | true |
