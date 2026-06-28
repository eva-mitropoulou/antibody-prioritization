# CoV-AbDab SARS-CoV-2 Cleaning Report

Source file: `data/raw/covabdab.csv`
Output file: `data/processed/covabdab_sarscov2_sequences.csv`

Metrics are counted after filtering to SARS-CoV-2 binders unless noted.
Rows without a heavy/VHH sequence are counted, then excluded from the
final sequence-key table because no usable sequence key can be created.

## Required Metrics

| Metric | Count |
|---|---:|
| Raw row count | 12918 |
| SARS-CoV-2 filtered row count | 12479 |
| Rows with heavy/VHH sequence | 11923 |
| Rows with light sequence | 10137 |
| Paired heavy-light rows | 10137 |
| Nanobody-like rows | 766 |
| neutralising_label = 1 | 6038 |
| neutralising_label = 0 | 2499 |
| neutralising_label missing | 3942 |
| Neutralisation conflicts | 2446 |
| Invalid heavy/VHH sequences | 1 |
| Invalid light sequences | 1 |
| Duplicate sequence keys | 155 |
| Rows removed by missing sequence key | 556 |
| Rows removed by sequence-key deduplication | 175 |
| Final deduplicated row count | 11748 |

## Notes

- `neutralising_label = 1` means `Neutralising Vs` mentions SARS-CoV-2.
- `neutralising_label = 0` means `Not Neutralising Vs` mentions SARS-CoV-2 and the positive column does not.
- `neutralising_conflict = true` marks rows where both neutralising columns mention SARS-CoV-2.
- Sequence validation uses only canonical amino acids: `ACDEFGHIKLMNPQRSTVWY`.
