# Bioaware CDR Annotation Report

This annotation step uses only existing paired antibody sequences.

status: `available`

## Coverage

| Metric | Value |
|---|---:|
| Raw rows | 5573 |
| Paired annotation candidates | 5092 |
| Heavy annotation OK | 5092 |
| Light annotation OK | 5092 |
| All six CDRs found | 5091 |
| All six CDRs found fraction | 99.98% |

## Existing CDR Metadata Comparison

| Region | Exact matches | Mismatches | Missing/comparison unavailable |
|---|---:|---:|---:|
| CDRH3 | 4986 | 106 | 0 |
| CDRL3 | 4460 | 627 | 5 |

## Outputs

- `data/processed/bioaware_paired_cdr_annotated.csv`
- `reports/metrics/bioaware_cdr_annotation_summary.json`
- `reports/figures/bioaware_cdr_coverage.png`
