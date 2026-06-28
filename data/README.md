# Local Data

Raw and processed sequence-record tables are intentionally not committed to the public repository.

The workflow expects local public-record artifacts under `data/raw/` and `data/processed/` when rebuilding analyses. Reports, metrics, figures, tests, and documentation are committed separately so the repository remains inspectable without publishing sequence-bearing source tables.

OAS paired records are treated as unknown-target natural antibody background, not as assayed negative-class labels.
