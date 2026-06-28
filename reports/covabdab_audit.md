# CoV-AbDab Data Audit

Source file: `data/raw/covabdab.csv`
Dataset shape: 12918 rows x 23 columns

## Key counts

- Number of rows: 12918
- Number of columns: 23
- Rows marked SARS-CoV-2: 12657
- Rows with heavy-chain sequence: 12346
- Rows with light-chain sequence: 10523
- Rows with both heavy and light-chain sequence: 10523
- Rows marked neutralising: 6373
- Rows marked non-neutralising: 5482

## Column names

- Name
- Ab or Nb
- Binds to
- Doesn't Bind to
- Neutralising Vs
- Not Neutralising Vs
- Protein + Epitope
- Origin
- VHorVHH
- VL
- Heavy V Gene
- Heavy J Gene
- Light V Gene
- Light J Gene
- CDRH3
- CDRL3
- Structures
- ABB Homology Model (if no structure)
- Sources
- Date Added
- Last Updated
- Update Description
- Notes/Following Up?

## Detected label and sequence columns

| Concept | Likely columns | Rows with relevant values |
|---|---|---:|
| antibody name | Name | 12918 |
| heavy chain sequence | VHorVHH | 12346 |
| light chain sequence | VL | 10523 |
| bind target antigen | Binds to, Doesn't Bind to, Protein + Epitope | 12918 |
| sars cov 2 | Binds to, Doesn't Bind to, Neutralising Vs, Not Neutralising Vs | 12657 |
| neutralisation | Neutralising Vs, Not Neutralising Vs | 8914 |
| pdb or structure | Structures, ABB Homology Model (if no structure) | 710 |
| epitope | Protein + Epitope | 12899 |

## Available labels

- Binding target / antigen: available (Binds to, Doesn't Bind to, Protein + Epitope)
- Neutralisation: available (Neutralising Vs, Not Neutralising Vs)
- Target / epitope: available (Protein + Epitope)
- PDB / structure: available (Structures, ABB Homology Model (if no structure))

## Example values from detected label columns

### ABB Homology Model (if no structure)

- No non-empty values found

### Binds to

- SARS-CoV2_WT;SARS-CoV2_Beta
- SARS-CoV2_WT
- SARS-CoV2_WT;SARS-CoV2_Alpha;SARS-CoV2_Beta;SARS-CoV2_Gamma;SARS-CoV2_Delta;SARS-CoV2_Epsilon;SARS-CoV2_Zeta;SARS-CoV2_Eta;SARS-CoV2_Iota;SARS-CoV2_Kappa;SARS-CoV2_Lambda;SARS-CoV2_Mu;SARS-CoV2_Omicron-BA1;SARS-CoV2_Omicron-BA2;SARS-CoV2_Omicron-BA2.12.1;SARS-CoV2_Omicron-BA3;SARS-CoV2_Omicron-BA4;SARS-CoV2_Omicron-BA5;SARS-CoV2_Omicron-XD;SARS-CoV2_Omicron-BA4.6;SARS-CoV2_Omicron-BF7
- SARS-CoV2_WT;SARS-CoV2_Alpha;SARS-CoV2_Beta;SARS-CoV2_Gamma;SARS-CoV2_Delta;SARS-CoV2_Omicron-BA1;SARS-CoV2_Omicron-BA1.1;SARS-CoV2_Omicron-BA2;SARS-CoV2_Omicron-BA2.12.1;SARS-CoV2_Omicron-BA2.75;SARS-CoV2_Omicron-BA4/5;SARS-CoV2_Omicron-BF7
- SARS-CoV2_WT;SARS-CoV2_Delta;SARS-CoV2_Omicron-BA1
- SARS-CoV2_WT;SARS-CoV2_Delta;SARS-CoV2_Omicron-BA1;SARS-CoV2_Omicron-BA2
- SARS-CoV2_Omicron-XBB1;SARS-CoV2_Omicron-XBB1.5;SARS-CoV2_Omicron-XBB2.3;SARS-CoV2_Omicron-EG5.1
- SARS-CoV2_WT;SARS-CoV2_Alpha;SARS-CoV2_Beta;SARS-CoV2_Gamma;SARS-CoV2_Delta;SARS-CoV2_Eta;SARS-CoV2_Omicron-BA3;SARS-CoV2_Omicron-BA2.75;SARS-CoV2_Omicron-BF7

### Doesn't Bind to

- SARS-CoV2_Omicron-BA1;HKU1
- SARS-CoV2_Beta;SARS-CoV2_Omicron-BA1;HKU1
- SARS-CoV2_Omicron-XBB1.5
- SARS-CoV2_Omicron_BA2
- SARS-CoV2_Omicron_BA4/5
- SARS-CoV2_Delta
- OC43;HKU1
- OC43

### Neutralising Vs

- SARS-CoV2_WT (weak)
- SARS-CoV2_WT;SARS-CoV2_Alpha;SARS-CoV2_Beta;SARS-CoV2_Gamma;SARS-CoV2_Delta;SARS-CoV2_Epsilon;SARS-CoV2_Zeta;SARS-CoV2_Eta;SARS-CoV2_Iota;SARS-CoV2_Kappa;SARS-CoV2_Lambda;SARS-CoV2_Mu;SARS-CoV2_Omicron-BA1 (weak);SARS-CoV2_Omicron-BA2 (weak);SARS-CoV2_Omicron-BA2.12.1;SARS-CoV2_Omicron-BA3;SARS-CoV2_Omicron-BA4;SARS-CoV2_Omicron-BA5;SARS-CoV2_Omicron-XD;SARS-CoV2_Omicron-BA4.6;SARS-CoV2_Omicron-BF7
- SARS-CoV2_WT;SARS-CoV2_Alpha;SARS-CoV2_Beta;SARS-CoV2_Gamma;SARS-CoV2_Delta;SARS-CoV2_Omicron-BA1;SARS-CoV2_Omicron-BA1.1;SARS-CoV2_Omicron-BA2;SARS-CoV2_Omicron-BA2.12.1;SARS-CoV2_Omicron-BA2.75;SARS-CoV2_Omicron-BA4/5;SARS-CoV2_Omicron-BF7
- SARS-CoV2_WT;SARS-CoV2_Delta;SARS-CoV2_Omicron-BA1
- SARS-CoV2_WT;SARS-CoV2_Delta;SARS-CoV2_Omicron-BA1;SARS-CoV2_Omicron-BA2
- SARS-CoV2_Omicron-XBB1;SARS-CoV2_Omicron-XBB1.5;SARS-CoV2_Omicron-XBB2.3;SARS-CoV2_Omicron-EG5.1
- SARS-CoV2_WT;SARS-CoV2_Alpha;SARS-CoV2_Beta;SARS-CoV2_Gamma;SARS-CoV2_Delta;SARS-CoV2_Eta;SARS-CoV2_Omicron-BA3;SARS-CoV2_Omicron-BA2.75;SARS-CoV2_Omicron-BF7 (weak)
- SARS-CoV2_WT;SARS-CoV2_Alpha;SARS-CoV2_Delta;SARS-CoV2_Eta;SARS-CoV2_Omicron-BA1;SARS-CoV2_Omicron-BA2;SARS-CoV2_Omicron-BA3;SARS-CoV2_Omicron-BA2.75

### Not Neutralising Vs

- SARS-CoV2_WT
- SARS-CoV2_Omicron-BQ1;SARS-CoV2_Omicron-BQ1.1;SARS-CoV2_Omicron-XBB
- SARS-CoV2_Omicron-BA1;SARS-CoV2_Omicron-BA2;SARS-CoV2_Omicron-BA2.12.1;SARS-CoV2_Omicron-BA4/5;SARS-CoV2_Omicron-XBB;SARS-CoV2_Omicron-XBB1.5;SARS-CoV2_Omicron-XBB1.16;SARS-CoV2_Omicron-XBB2.3.2;SARS-CoV2_Omicron-BQ1.1
- SARS-CoV2_Beta;SARS-CoV2_Gamma;SARS-CoV2_Eta;SARS-CoV2_Omicron-BA4/5;SARS-CoV2_Omicron-BA2.12.1;SARS-CoV2_Omicron-XBB;SARS-CoV2_Omicron-BF7;SARS-CoV2_Omicron-BQ1.1;SARS-CoV2_Omicron-XBB1.5;SARS-CoV2_Omicron-XBB1.16;SARS-CoV2_Omicron-XBB2.3.2
- SARS-CoV2_Omicron-XBB1.5;SARS-CoV2_Omicron-XBB1.16
- SARS-CoV2_Omicron-BA2;SARS-CoV2_Omicron-BA4/5;SARS-CoV2_Omicron-XBB1.5;SARS-CoV2_Omicron-XBB1.16
- SARS-CoV2_Alpha;SARS-CoV2_Omicron-BA1;SARS-CoV2_Omicron-BA2
- SARS-CoV2_Beta;SARS-CoV1

### Protein + Epitope

- S; RBD/non-RBD
- S; non-RBD
- S; iso-RBD
- S; RBD
- S; S1
- S; S2 Fusion Peptide
- S; Unk
- S; S2 (HR1 Peptide)

### Structures

- https://www.rcsb.org/structure/8J1T;https://www.rcsb.org/structure/8J1V
- https://www.rcsb.org/structure/8IX3
- https://www.rcsb.org/structure/8F4P
- https://www.rcsb.org/structure/7YH6;https://www.rcsb.org/structure/7YH7
- https://www.rcsb.org/structure/8HES
- https://www.rcsb.org/structure/8HGL;https://www.rcsb.org/structure/8HGM
- https://www.rcsb.org/structure/8PQ2
- https://www.rcsb.org/structure/8H91

## Missing values per column

| Column | Missing | Non-missing | Missing % |
|---|---:|---:|---:|
| Name | 0 | 12918 | 0.00 |
| Ab or Nb | 0 | 12918 | 0.00 |
| Binds to | 0 | 12918 | 0.00 |
| Doesn't Bind to | 9717 | 3201 | 75.22 |
| Neutralising Vs | 6545 | 6373 | 50.67 |
| Not Neutralising Vs | 7436 | 5482 | 57.56 |
| Protein + Epitope | 19 | 12899 | 0.15 |
| Origin | 61 | 12857 | 0.47 |
| VHorVHH | 572 | 12346 | 4.43 |
| VL | 2395 | 10523 | 18.54 |
| Heavy V Gene | 40 | 12878 | 0.31 |
| Heavy J Gene | 1496 | 11422 | 11.58 |
| Light V Gene | 818 | 12100 | 6.33 |
| Light J Gene | 2274 | 10644 | 17.60 |
| CDRH3 | 11 | 12907 | 0.09 |
| CDRL3 | 867 | 12051 | 6.71 |
| Structures | 12208 | 710 | 94.50 |
| ABB Homology Model (if no structure) | 12918 | 0 | 100.00 |
| Sources | 0 | 12918 | 0.00 |
| Date Added | 0 | 12918 | 0.00 |
| Last Updated | 0 | 12918 | 0.00 |
| Update Description | 10219 | 2699 | 79.11 |
| Notes/Following Up? | 0 | 12918 | 0.00 |
