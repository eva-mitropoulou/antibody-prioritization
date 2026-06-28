# SARS-CoV-2 Neutralisation ML Dataset Report

Input file: `data/processed/covabdab_sarscov2_sequences.csv`
Output file: `data/processed/covabdab_neutralisation_ml.csv`

No machine learning is trained in this step.

## Filtering Summary

| Metric | Count |
|---|---:|
| Input row count | 11748 |
| Rows removed because label missing | 3810 |
| Rows removed because neutralisation_conflict true | 2363 |
| Rows removed because missing/invalid heavy sequence | 1 |
| Rows removed because present light sequence is invalid | 1 |
| Final ML row count | 5573 |

## Label Balance

| Label | Count | Percentage |
|---|---:|---:|
| 0: non-neutralising | 2292 | 41.13% |
| 1: neutralising | 3281 | 58.87% |

## Dataset Composition

| Metric | Count |
|---|---:|
| Paired heavy-light count | 5092 |
| Nanobody-like count | 313 |
| Rows with structure | 330 |
| Targets RBD | 4820 |
| Targets Spike | 5556 |
| Targets NTD | 432 |

## Length Summaries

### heavy_length

| Statistic | Value |
|---|---:|
| count | 5573 |
| mean | 122.85 |
| std | 4.43 |
| min | 88.00 |
| 25% | 120.00 |
| 50% | 123.00 |
| 75% | 126.00 |
| max | 226.00 |

### light_length

| Statistic | Value |
|---|---:|
| count | 5092 |
| mean | 108.54 |
| std | 2.54 |
| min | 86.00 |
| 25% | 107.00 |
| 50% | 108.00 |
| 75% | 110.00 |
| max | 215.00 |

### cdrh3_length

| Statistic | Value |
|---|---:|
| count | 5573 |
| mean | 15.95 |
| std | 3.95 |
| min | 3.00 |
| 25% | 13.00 |
| 50% | 16.00 |
| 75% | 19.00 |
| max | 63.00 |

### cdrl3_length

| Statistic | Value |
|---|---:|
| count | 5247 |
| mean | 9.74 |
| std | 1.45 |
| min | 4.00 |
| 25% | 9.00 |
| 50% | 10.00 |
| 75% | 11.00 |
| max | 71.00 |

## Figures

- `reports/figures/label_counts.png`
- `reports/figures/heavy_length_distribution.png`
- `reports/figures/cdrh3_length_distribution.png`
- `reports/figures/antibody_type_counts.png`
