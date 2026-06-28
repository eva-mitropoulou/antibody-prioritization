# Domain-Region Annotation Summary

This stage reuses existing annotation artifacts when present. It does not
rerun heavy annotation or print raw sequence values.

| Metric | Value |
|---|---:|
| Status | available |
| Annotated rows | 5092 |
| Paired rows | 5092 |
| Single-chain or light-missing rows | 0 |
| Existing annotation report available | true |
| Existing annotation metrics available | true |

## Six-Region Coverage

| Region flag | Found count |
|---|---:|
| `cdrh1_found` | 5092 |
| `cdrh2_found` | 5092 |
| `cdrh3_found` | 5092 |
| `cdrl1_found` | 5092 |
| `cdrl2_found` | 5092 |
| `cdrl3_found` | 5091 |
| `all_six_cdrs_found` | 5091 |

## Fallback Region-3 Columns

`existing_cdrh3`, `existing_cdrl3`, `group_feature_cdr3`, `group_feature_b_cdr3`

## Mismatch/Missing Counts

| Check | Count |
|---|---:|
| heavy_annotation_ok_false_or_missing | 0 |
| light_annotation_ok_false_or_missing | 0 |
| marker_insertion_ok_false_or_missing | 1 |
