# OAS Matched Background Audit

OAS rows are treated as unknown-target background for enrichment analysis.
This matching audit reports aggregate matching fields and public-safe identifiers.

| Metric | Value |
|---|---:|
| Project row count | 11748 |
| Raw OAS row count | 17882 |
| Matched project row count | 7192 |
| Matched OAS row count | 7192 |
| Skipped project rows due to no matched OAS bin | 1818 |
| Project rows not selected due to OAS bin shortage | 2738 |
| Exact overlap count removed | 5 |
| Matching bins | 52 |
| Non-empty matched bins | 28 |

## Class Balance

| Class | Count |
|---|---:|
| project_record | 7192 |
| oas_unknown_target_background | 7192 |

## Length-Bin Distributions

### project_heavy_length_bin

| Bin | Count |
|---|---:|
| 100 | 1 |
| 105 | 1 |
| 110 | 100 |
| 115 | 1760 |
| 120 | 4276 |
| 125 | 976 |
| 130 | 75 |
| 135 | 3 |

### oas_heavy_length_bin

| Bin | Count |
|---|---:|
| 100 | 1 |
| 105 | 1 |
| 110 | 100 |
| 115 | 1760 |
| 120 | 4276 |
| 125 | 976 |
| 130 | 75 |
| 135 | 3 |

### project_light_length_bin

| Bin | Count |
|---|---:|
| 100 | 38 |
| 105 | 4883 |
| 110 | 2257 |
| 115 | 11 |
| 120 | 2 |
| 90 | 1 |

### oas_light_length_bin

| Bin | Count |
|---|---:|
| 100 | 38 |
| 105 | 4883 |
| 110 | 2257 |
| 115 | 11 |
| 120 | 2 |
| 90 | 1 |

### project_total_length_bin

| Bin | Count |
|---|---:|
| 200 | 2 |
| 210 | 12 |
| 220 | 3397 |
| 230 | 3709 |
| 240 | 72 |

### oas_total_length_bin

| Bin | Count |
|---|---:|
| 200 | 2 |
| 210 | 12 |
| 220 | 3397 |
| 230 | 3709 |
| 240 | 72 |

### project_has_light

| Bin | Count |
|---|---:|
| True | 7192 |

### oas_has_light

| Bin | Count |
|---|---:|
| True | 7192 |
