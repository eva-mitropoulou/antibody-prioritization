# Data Card

## Local Dataset Artifacts

This repository uses local public-record artifacts only. Key files include:

- `data/processed/neutral_sequence_classification_ml.csv`
- `data/processed/neutral_prepared_sequences.csv`
- `data/processed/bioaware_paired_cdr_annotated.csv`
- `data/processed/oas/oas_paired_standardized.csv`
- `data/raw/oas/*.csv.gz`

Source rows and sequence values are not printed in reports.

## Strict Labeled Dataset

The strict labeled ML table is `data/processed/neutral_sequence_classification_ml.csv`. It contains 5,573 labeled rows with label balance 0=2,292 and 1=3,281. This table is used for matched broad benchmarking, source-holdout validation, calibration analysis, model selection, and unsupervised sequence-space analysis.

## Broader Record Table

The broader prepared table is `data/processed/neutral_prepared_sequences.csv`. It keeps existing project records beyond the strict supervised subset so prioritization can preserve missing-label and conflict-label records. The broader prioritization output is `reports/broader_existing_record_prioritization_table.csv`.

## Missing And Conflict Labels

Missing-label and conflict-label records are not converted into supervised negatives. They are preserved for scoring and review categorization in the broader table. Strict supervised metrics are computed only on rows with usable binary labels.

## Source/Study Caveats

Publication/source metadata are heterogeneous. Source fields may combine multiple identifiers, dates, or publication-like metadata. Validation modules sanitize source identifiers before reporting and use leave-source-out or source-grouped splits to measure whether model behavior survives study-level shifts.

## Label Heterogeneity Caveats

Neutralisation labels are public record labels, not a single harmonized assay. Label definitions, assay conditions, target-region annotations, and sequence completeness vary across records. Metrics should therefore be interpreted as retrospective record-classification evidence, not prospective biological certainty.

## OAS Background Semantics

OAS paired records are treated as unknown-target natural antibody background. They are not assayed negative-class labels and are not mixed with the main neutralisation benchmark. OAS retrieval reports measure whether project records are distinguishable from local natural background under broad and matched controls.

## Privacy And Safety

The workflow uses public records only. It does not generate, mutate, design, optimize, or propose biological sequences. Source sequence fields are not altered by the reporting workflow.
