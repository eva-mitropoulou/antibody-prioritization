# Diversity-Aware Existing-Record Shortlist

This shortlist contains existing public dataset records only. It does not
create, alter, mutate, optimize, rank newly designed sequences, or propose
sequence changes.

## Method

Candidates were filtered to predicted probability >= 0.75, high confidence, and low or medium sequence-risk bins.
Known neutralising, missing-label, and conflict-label records were kept. Known non-neutralising records were included only when probability >= 0.90 and were marked as model-disagreement records.
One representative was selected per diversity group after sorting by predicted probability, confidence score, and sequence-risk score.

## Summary

| Metric | Value |
|---|---:|
| Input records | 11747 |
| Candidate pool size before diversity filtering | 1094 |
| Final shortlist size | 23 |
| Diversity groups | 23 |
| Missing-label records in shortlist | 17 |
| Conflict-label records in shortlist | 0 |
| Model-disagreement records in shortlist | 0 |

## Source Columns

- CDRH3 source: `missing`
- V-gene group source: `diversity_group_prefix`
- The broader input table did not contain a direct `group_feature_v` column; the script used only existing table fields and did not infer or assign new V-gene annotations.

## Tier Counts

| Tier | Count |
|---|---:|
| tier_1 | 13 |
| tier_2 | 10 |

## Record Category Counts

| Record category | Count |
|---|---:|
| missing_label | 17 |
| known_neutralising | 6 |

## Target Region Counts

| Target region | Count |
|---|---:|
| S; RBD | 3 |
| S; NTD | 3 |
| S; S2 | 3 |
| S; Unk | 3 |
| S; S2 (HR2 Peptide) | 2 |
| Nsp3; PLpro | 2 |
| S; S1 | 2 |
| Nsp9 | 1 |
| S; non-RBD | 1 |
| S; RBD/non-RBD | 1 |
| Mpro | 1 |
| unknown_target_region | 1 |

## Limitations

- Model score is used for existing-record prioritization.
- Diversity grouping is heuristic.
- Labels are heterogeneous literature-derived labels.
- This is retrospective prioritization of existing records only.
- No new sequences are generated, altered, proposed, or optimized.

## Artifacts

- `reports/diversity_aware_existing_record_shortlist.csv`
- `reports/metrics/diversity_aware_shortlist_summary.json`
- `reports/figures/shortlist_tier_counts.png`
- `reports/figures/shortlist_record_category_counts.png`
- `reports/figures/shortlist_target_region_counts.png`
- `reports/figures/shortlist_probability_distribution.png`
