# Initial Hardening Audit: antibody-prioritization

- Default branch: `main`
- Hardening branch: `portfolio-hardening-final`
- Start commit: `1e90c04d2c92`
- Tracked files: 209
- Markdown files: 48
- Metrics JSON files: 36
- Reports: 44
- Tests detected: 1
- CI workflows: 0

## Public-Facing Surfaces

- README, docs, projects, portfolio assets, reports, model/data cards where present.
- Public claim scan hits for manual review: 0
- Raw sequence-like public files found: 0

## Immediate Fixes Needed

- Correct paired-region scorer to region-only compact k-mer.
- Remove diagnostic model_error_analysis from best pretrained selection.
- Document OAS as unknown-target natural background and add lightweight dependency split.
- Add or document CI/test status.

## Reproducibility Files

- Makefile present: False
- pyproject present: False
- requirements present: True
- environment.yml present: False
- CI workflows: none

## Notes

This audit records file and claim-scan metadata only. It does not include raw rows, raw sequences, or raw molecule tables.
