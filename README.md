# Antibody Prioritization

This repository tracks a sequence-record ML workflow for working with public SARS-CoV-2 antibody records. The main use case is practical: keep the data cleaning, validation checks, model comparisons, prioritization tables, and background controls in one reproducible place for retrospective review.

## Why This Exists

Public antibody tables are useful, but the labels and source metadata are messy. I wanted a workflow that could answer a few concrete questions:

- Which simple sequence model is a defensible baseline for existing records?
- How much do results change under grouped and source-holdout validation?
- Are pretrained antibody language-model representations actually better here?
- Which existing records are worth reviewing first, given confidence, diversity, and metadata?
- How separable are project records from OAS unknown-target natural antibody background?

## What It Does

- Builds strict labeled and broader existing-record tables from local processed data.
- Benchmarks compact character k-mer TF-IDF logistic regression against pretrained sequence-model runs.
- Separates full strict, paired annotated, missing-label, and conflict-label subsets.
- Runs grouped validation, source-holdout validation, calibration/threshold analysis, and source-robust model selection.
- Scores existing records and summarizes a diversity-aware review shortlist.
- Runs OAS background retrieval as an unknown-target natural antibody background control, separate from neutralisation classification.
- Builds unsupervised sequence-space summaries without using labels for clustering.

## Reproducing The Current Reports

The repository keeps generated reports and metrics. To regenerate the lightweight final reports from existing artifacts:

```bash
python -m pip install -r requirements.txt
make reproduce-small
make test
```

The script skips optional expensive stages when their outputs already exist. Raw and processed sequence tables are local artifacts kept outside the public repository.

For full pretrained language-model experiments, install the optional LM dependencies:

```bash
python -m pip install -r requirements-lm.txt
```

Those optional runs are benchmark evidence only. The selected public scorers remain the full strict whole-pair k-mer broad scorer and the region-only compact k-mer paired-region scorer.

## Current Results

| Check | Row subset / split | Result | Notes |
|---|---|---:|---|
| Broad grouped k-mer benchmark | Full strict labeled subset, V-gene grouped split, zero group overlap | ROC-AUC 0.7800, PR-AUC 0.8233 | Primary broad baseline. |
| Paired region benchmark | Paired annotated subset, V-gene grouped split, zero group overlap | Region-only ROC-AUC 0.6629, PR-AUC 0.6330 | Region features helped within this paired subset. |
| Source-robust selection | Leave-source-out over sanitized source groups | weighted ROC-AUC 0.6095, weighted PR-AUC 0.6363 | Source/study effects remain a real limitation. |
| Calibration threshold | Source-robust selected model | threshold 0.7 precision 0.8266, recall 0.3062, coverage 0.3051 | High-confidence review cutoff for existing records. |
| OAS background retrieval | Project records vs OAS unknown-target background | ROC-AUC 0.9921, PR-AUC 0.9897 | Background enrichment diagnostic only. |
| Matched OAS retrieval | Coarse length/status matched background | ROC-AUC 0.9911, PR-AUC 0.9893 | Enrichment persisted after coarse matching. |
| Diversity-aware shortlist | Broader prioritization table | 23 records | Existing-record review queue. |

## Selected Model

The broad scorer is `whole_pair_kmer`: compact heavy/light sequence-pair text represented with character k-mer TF-IDF and a balanced logistic-regression classifier.

The simpler k-mer model remained more defensible than the pretrained antibody language-model representations on this noisy public-label task. The source-holdout checks are weaker than the grouped benchmark, so the score is best used for retrospective prioritization and high-confidence review.

## Repository Layout

```text
data/          # Local data placeholder; raw/processed sequence tables stay outside the public repo
docs/          # Data and model cards
models/        # Small saved classical model artifacts
reports/       # Generated reports, metrics, and figures
scripts/       # Reproduction helpers
src/           # Data, model, and analysis code
tests/         # Lightweight integrity checks
```

## Useful Outputs

- `reports/final_project_report.md`
- `reports/model_registry.md`
- `reports/matched_kmer_benchmark_audit.md`
- `reports/source_robust_model_selection_report.md`
- `reports/calibration_threshold_report.md`
- `reports/oas_background_retrieval_report.md`
- `reports/oas_matched_background_retrieval_report.md`
- `reports/unsupervised_antibody_landscape_report.md`
- `docs/DATA_CARD.md`
- `docs/MODEL_CARD.md`

Machine-readable summaries are under `reports/metrics/`.

## Project Frame

- Public neutralisation labels are heterogeneous across studies and assays.
- Source-holdout validation is intentionally conservative and shows weaker generalization than V-gene grouped validation.
- OAS records are unknown-target natural antibody background for enrichment analysis.
- Existing-record prioritization and high-confidence review are the intended outputs.
- Sequence-design and optimization are outside the project scope.

## Dependency Notes

The default requirements support lightweight report checks, classical k-mer baselines, and tests. Pretrained antibody language-model experiments use the optional packages listed in `requirements-lm.txt`.
