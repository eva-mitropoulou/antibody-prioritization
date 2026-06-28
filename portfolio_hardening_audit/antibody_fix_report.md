# Antibody Hardening Fix Report

Branch: `portfolio-hardening-final`

## Summary

- Corrected the primary paired-region scorer to `region_only_compact_kmer`.
- Kept the full strict whole-pair k-mer model as the primary broad scorer.
- Removed diagnostic error-analysis artifacts from pretrained-model selection.
- Clarified that pretrained antibody models are benchmark evidence and not selected primary models.
- Clarified OAS as unknown-target natural antibody background and a background/enrichment diagnostic.
- Added lightweight reproducibility files for non-LM checks.

## Checks Run

- `make reproduce-small PYTHON=/usr/bin/python3`: pass.
- `make test PYTHON=/usr/bin/python3`: pass, 10 tests.
- Public text sequence scan: pass, zero long sequence-like hits.
- Report CSV scan: pass, sequence-related fields are key/hash fields rather than raw sequence columns.
- `git diff --check`: pass.

## Remaining Manual Review

- Optional full LM reproduction still requires heavier optional dependencies and model artifacts.
