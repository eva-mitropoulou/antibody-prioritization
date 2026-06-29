# Final Project Report

## Project Goal

Build a public-data antibody sequence-record ML workflow for neutralisation classification benchmarking, sequence-space analysis, retrospective selection simulation, and prioritization of existing records.

The workflow preserves source sequence fields and supports retrospective
benchmarking, review prioritization, and background retrieval analysis.

## Datasets

| Dataset | Rows | Columns | Label 0 | Label 1 |
|---|---:|---:|---:|---:|
| Strict labeled ML table | 5573 | 31 | 2292 | 3281 |
| Broader prepared table | 11748 | 27 | 2292 | 5646 |

## Cleaning And Labels

Labels were used as existing binary record metadata. Missing-label and conflict-label records were preserved for prioritization rather than discarded from the broader scoring table.

## Paired/Light-Missing Handling

Strict paired/light-missing counts: {'light_missing_or_single_chain': 481, 'paired': 5092}. Broader paired/light-missing counts: {'light_missing_or_single_chain': 1762, 'paired': 9986}.

## Domain-Region Annotation

Annotation status: available. Paired annotated rows: 5092. Single-chain/light-missing rows: 0.

## Target-Region Metadata Analysis

Target-region analysis status: available. Unknown broader count: 10. Useful for subgroup analysis: True.

## Matched Benchmark Results

Matched k-mer baselines used compact character strings, grouped splits, zero group overlap, and separate full strict versus paired annotated subsets.

Region features improved paired matched ROC-AUC: True; improved paired matched PR-AUC: True.

## Pretrained Sequence-Model Benchmarks

Pretrained and embedding models were treated as benchmark evidence, not automatic primary scorers. Same-row-count matched k-mer comparisons were used when available.

Pretrained models beat matched k-mer baselines on both primary metrics: False.

## Final Model Selection

Primary broad scorer: kmer_tfidf_logreg_pair_text on Full strict labeled dataset; whole-pair compact k-mer input. (ROC-AUC 0.7800, PR-AUC 0.8233).

Primary paired/region scorer: kmer_tfidf_logreg__paired_annotated_subset__region_only_compact_kmer on Paired annotated subset; whole-pair, region-only, and combined compact k-mer inputs. (ROC-AUC 0.6629, PR-AUC 0.6330).

## Skeptical Validation Controls

### Leave-Source/Leave-Study-Out Validation

Detected source groups: 84. Valid held-out source groups: 11. Macro source-holdout ROC-AUC: 0.5605. Macro source-holdout PR-AUC: 0.6454.

Source-grouped fallback ROC-AUC: 0.5247. Source-grouped fallback PR-AUC: 0.8189.

Leave-source-out validation is weaker than the matched grouped benchmark, so source/study effects may materially affect apparent model performance.

### Calibration And Threshold Analysis

Brier score: 0.2636. Expected calibration error: 0.3034. High-confidence review threshold: 0.7000 with precision 0.8187 and recall 0.2998.

Scores are more reliable for ranking than as absolute probabilities; thresholds should be treated as review cutoffs rather than calibrated risk estimates.

### Source-Robust Model Selection

Selected source-robust model: whole_pair_kmer. Meaningful improvement over previous source-holdout baseline: False.

Source-robust selection chose `whole_pair_kmer` and did not materially improve cross-source performance. CDR/region features were competitive for source robustness. Scores remain ranking and prioritization signals for existing records.

Selected weighted leave-source-out ROC-AUC: 0.6095. Selected weighted leave-source-out PR-AUC: 0.6363. High-confidence threshold: 0.7000 with precision 0.8266, recall 0.3062, and coverage 0.3051.

## Existing-Record Prioritization

Broader scored records: 11747. Missing-label records: 3810. Diversity groups: 73.

## Diversity-Aware Shortlist

Shortlist size: 23 from 1094 candidate records before diversity filtering.

## Unsupervised Landscape

Landscape status: available. Feature source: cached_pair_embeddings. Cluster count: 9.

## Background Retrieval Status

Background retrieval status: available. Background retrieval metrics were kept separate from the main classification task.

### Broad OAS Retrieval

Broad OAS retrieval used OAS paired rows as unknown-target background. Project rows: 11748. OAS rows after overlap removal: 17877. Exact overlaps removed: 5. ROC-AUC: 0.9921. PR-AUC: 0.9897.

### Hard Matched OAS Retrieval

Matched OAS retrieval used coarse heavy-length, light-length, total-length, and light-status bins. Matched project rows: 7192. Matched OAS rows: 7192. Skipped project rows: 1818. Exact overlaps removed: 5. ROC-AUC: 0.9911. PR-AUC: 0.9893.

### OAS Existing-Record Retrieval Shortlist

The OAS existing-record retrieval shortlist identifies existing OAS background records that are sequence-similar to curated project-positive records. OAS records are unknown-target natural antibody background, and the output is an existing-record shortlist for expert review.

OAS rows scored: 17882. Project-positive reference rows: 3281. Top-25 diverse shortlist size: 25. Top-100 ranked table size: 100. Diversity clusters: 25.

The shortlist score is a computational prioritization score, not a binding probability. Shortlisted records are not validated binders or therapeutics, and the module does not generate or modify sequences.

### Interpretation

OAS records are unknown-target natural antibody background. The OAS retrieval task is a background/enrichment diagnostic kept separate from the main neutralisation benchmark. High OAS retrieval separability likely reflects source/domain differences between project records and natural repertoire background.

## Retrospective Selection-Loop Simulation

Best strategy: highest_score. Best beats random mean: True.

## Structure Metadata

Structure metadata available in shortlist: 3. Docking was not run by default.

## Limitations

- Public source labels are heterogeneous and retrospective.
- Model probabilities are prioritization signals for existing-record review.
- Subset-specific metrics are not directly comparable across row subsets.
- Diversity and sequence-risk features are heuristic.
- Background retrieval is optional and local-data dependent.
- The OAS existing-record shortlist is an expert-review queue, not antibody design or therapeutic discovery.
- Docking remains a separate future validation workflow.

## Next Steps

- Curate clearer target-region and structure metadata where available.
- Add prospective-style validation only when new external records are available.
- Keep benchmark comparisons matched by row subset and split strategy.
- Use the shortlist as an inspection queue for existing records.
