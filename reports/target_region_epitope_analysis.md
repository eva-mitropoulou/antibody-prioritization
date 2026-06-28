# Target-Region Metadata Analysis

Target-region metadata was normalized to RBD, NTD, Spike/S, other, and
unknown groups. This report contains aggregate counts only.

| Metric | Value |
|---|---:|
| Broader records | 11748 |
| Strict labeled records | 5573 |
| Broader unknown target-region count | 10 |
| Strict unknown target-region count | 10 |
| Metadata useful for subgroup analysis | true |

## Broader Prepared Dataset

| Group | Rows | Labeled rows | Label 0 | Label 1 | Paired | Light-missing/single-chain | Structure available |
|---|---:|---:|---:|---:|---:|---:|---:|
| RBD | 8147 | 7111 | 1754 | 5357 | 7224 | 923 | 567 |
| NTD | 820 | 494 | 304 | 190 | 582 | 238 | 44 |
| Spike/S | 1 | 1 | 0 | 1 | 1 | 0 | 0 |
| other | 2770 | 322 | 232 | 90 | 2169 | 601 | 48 |
| unknown | 10 | 10 | 2 | 8 | 10 | 0 | 0 |

## Strict Labeled Dataset

| Group | Rows | Labeled rows | Label 0 | Label 1 | Paired | Light-missing/single-chain | Structure available |
|---|---:|---:|---:|---:|---:|---:|---:|
| RBD | 4820 | 4820 | 1754 | 3066 | 4386 | 434 | 4458 |
| NTD | 431 | 431 | 304 | 127 | 423 | 8 | 423 |
| Spike/S | 1 | 1 | 0 | 1 | 1 | 0 | 1 |
| other | 311 | 311 | 232 | 79 | 272 | 39 | 272 |
| unknown | 10 | 10 | 2 | 8 | 10 | 0 | 10 |

## Interpretation

Target-region metadata is useful for subgroup analysis when a normalized group has enough labeled rows and contains both label classes.

## Artifacts

- `reports/metrics/target_region_epitope_analysis.json`
- `reports/figures/target_region_group_counts.png`
