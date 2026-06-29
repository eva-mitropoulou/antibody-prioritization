# Antibody Prioritization

This repository contains an antibody sequence classification workflow for public SARS-CoV-2 antibody sequence records. The goal is not to design new antibodies. The goal is to test how well sequence-based models work on existing public records, check where results are affected by source or study bias, and produce review tables for records that may deserve closer inspection.

## Questions

The project is organized around a few concrete questions:

- Can a simple character k-mer model classify strict labeled neutralisation records?
- How much do results change when validation is grouped by sequence metadata or held out by source/study?
- Do pretrained antibody embedding or language-model approaches improve on the k-mer baseline for these labels?
- How should records with missing or conflicting labels be scored without treating them as new training labels?
- How different are the project records from OAS unknown-target antibody background records?
- What clusters or similarity patterns appear without using labels for clustering?

## What The Workflow Does

- Builds strict labeled tables and broader existing-record tables from local processed data.
- Runs supervised neutralisation classification on strict labeled records.
- Compares compact character k-mer TF-IDF logistic regression with pretrained antibody embedding/model runs.
- Keeps full strict, paired annotated, missing-label, and conflict-label subsets separate.
- Runs grouped validation, source-holdout validation, calibration/threshold analysis, and source-robust model selection.
- Scores broader existing records and creates a small diversity-aware review table.
- Uses OAS only as unknown-target antibody background, not as negative neutralisation labels.
- Builds unsupervised clustering and similarity summaries without using labels during clustering.

## Reproducing The Reports

The repository includes generated reports and machine-readable metrics. Raw and processed sequence tables are local artifacts and are not committed.

For the lightweight report refresh and integrity checks:

```bash
python -m pip install -r requirements.txt
make report
```

Equivalent direct command:

```bash
bash scripts/reproduce_final_reports.sh
```

To run only the tests:

```bash
make test
```

Optional pretrained model scripts require the packages listed in `requirements-lm.txt`. Those experiments are included as model comparisons; they are not required to use the saved reports.

## Current Results

| Check | Row subset / split | Result | Notes |
|---|---|---:|---|
| Broad grouped k-mer benchmark | Full strict labeled subset, V-gene grouped split, zero group overlap | ROC-AUC 0.7800, PR-AUC 0.8233 | Main broad baseline. |
| Paired region benchmark | Paired annotated subset, V-gene grouped split, zero group overlap | Region-only ROC-AUC 0.6629, PR-AUC 0.6330 | Region features helped within this paired subset. |
| Source-robust selection | Leave-source-out over sanitized source groups | weighted ROC-AUC 0.6095, weighted PR-AUC 0.6363 | Source/study effects remain a real limitation. |
| Calibration threshold | Source-robust selected model | threshold 0.7 precision 0.8266, recall 0.3062, coverage 0.3051 | High-confidence cutoff for reviewing existing records. |
| OAS background retrieval | Project records vs OAS unknown-target antibody background | ROC-AUC 0.9921, PR-AUC 0.9897 | Dataset/background comparison, not a neutralisation benchmark. |
| Matched OAS retrieval | Coarse length/status matched OAS background | ROC-AUC 0.9911, PR-AUC 0.9893 | The project/OAS difference remained high after coarse matching. |
| Diversity-aware shortlist | Broader prioritization table | 23 records | Small review table for existing records. |

## Selected Model

The selected broad model is `whole_pair_kmer`: compact heavy/light sequence-pair text represented with character k-mer TF-IDF and a balanced logistic-regression classifier.

In these reports, pretrained antibody embedding and language-model approaches did not outperform the simpler k-mer baseline on the noisy public-label task. The grouped benchmark is stronger than the source-holdout benchmark, so the model scores should be read mainly as ranking and review signals, not as absolute biological claims.

## Repository Layout

```text
data/          # Placeholder for local data; raw and processed sequence tables are not committed
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

## Interpretation Notes

- Public neutralisation labels are heterogeneous across studies and assays.
- Source-holdout results are weaker than grouped validation results, which suggests source/study effects are important.
- Probabilities and scores are mainly useful for ranking and review of existing records, not for absolute claims about biological activity.
- OAS is used as unknown-target antibody background. It is not used as neutralisation label data.
- Records with missing or conflicting labels are preserved for review outputs, not used as new ground-truth training labels.

## Requirements

The default requirements support report checks, classical k-mer baselines, and tests. Pretrained antibody embedding/model scripts use the optional packages listed in `requirements-lm.txt`.
