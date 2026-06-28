# Final Consistency Audit

Overall status: **WARN**

## Checks

| Check | Pass |
|---|---:|
| expected_artifacts_exist | True |
| selected_model_consistent | True |
| oas_described_as_unknown_target_background | True |
| oas_not_described_as_assayed_negative_class | False |
| source_holdout_limitations_included | True |
| no_affirmative_prospective_or_efficacy_overclaim | True |
| public_score_csv_headers_safe | True |

## Selected Models

- Model registry primary model: `kmer_tfidf_logreg_pair_text`
- Source-robust selected model: `whole_pair_kmer`

## Missing Expected Artifacts

- None

## Notes

- OAS is treated as unknown-target background and kept separate from the neutralisation benchmark.
- Source-holdout limitations are preserved in project documentation.
- The audit checks paths, report wording, JSON summaries, and CSV headers only.
