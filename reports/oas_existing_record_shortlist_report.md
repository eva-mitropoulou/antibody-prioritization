# OAS Existing-Record Retrieval Shortlist

This shortlist contains existing OAS records that are sequence-similar to curated project-positive records.
OAS records are unknown-target background and, more specifically, unknown-target natural antibody background.
The output is an existing-record shortlist for expert review. The score is a computational prioritization score, not a binding probability.
The records are not validated binders or therapeutics, and this module does not generate or modify sequences.
Any downstream use requires independent expert review and appropriate experimental validation outside this repository.

## Inputs

- OAS standardized records: `data/processed/oas/oas_paired_standardized.csv`
- Project-positive records: `data/processed/neutral_sequence_classification_ml.csv`

## Method

Project-positive and OAS compact pair texts were used internally for hashing and k-mer TF-IDF features. Raw sequence strings were not saved to public outputs.
Ranking combined retrieval-model score, maximum nearest-neighbor similarity, top-10 neighbor similarity, and centroid similarity.
A greedy diversity filter selected records while avoiding OAS records with cosine similarity greater than 0.95 to already selected shortlist records.

## Summary

| Metric | Value |
|---|---:|
| OAS rows scored | 17882 |
| Project-positive reference rows | 3281 |
| Unique project-positive reference texts | 3281 |
| Top 25 shortlist size | 25 |
| Top 100 table size | 100 |
| Diversity clusters | 25 |
| Minimum score | 0.016377 |
| Maximum score | 0.910184 |
| Saved retrieval score matches | 3585 |

## Composite Score Weights

| Component | Normalized weight |
|---|---:|
| retrieval_score | 0.400 |
| max_positive_neighbor_similarity | 0.300 |
| top10_positive_neighbor_similarity | 0.200 |
| positive_centroid_similarity | 0.100 |

## Review Flags

Length, cysteine-count, glycosylation-motif, hydrophobic-fraction, and duplicate or near-duplicate flags are heuristic review flags only. They are not validation results.

## Artifacts

- `reports/oas_existing_record_shortlist_report.md`
- `reports/metrics/oas_existing_record_shortlist_metrics.json`
- `reports/oas_existing_record_shortlist_top25.csv`
- `reports/oas_existing_record_shortlist_top100.csv`
- `reports/oas_existing_record_scores_public.csv`
- `reports/figures/oas_existing_record_score_distribution.png`
- `reports/figures/oas_existing_record_similarity_vs_score.png`
- `reports/figures/oas_existing_record_diversity_map.png`

## Limitations

- OAS records are unknown-target natural antibody background records.
- Similarity to project-positive records does not establish binding or neutralisation.
- The retrieval score is for computational prioritization and expert review only.
- This module does not generate, mutate, design, optimize, or propose sequences.
- Wet-lab protocols and therapeutic claims are outside the scope of this repository.
