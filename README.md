# Antibody Prioritization

I built this project to work through a messy but common problem in public antibody data: there are useful SARS-CoV-2 antibody sequence records, but the labels, assays, source metadata, and missing fields are uneven. The code here builds a cleaned set of strict labeled records for neutralisation classification, keeps a broader table of existing records for review, and checks how much the model results depend on the way the data is split.

The main model is intentionally simple: compact heavy/light sequence text, character k-mer TF-IDF features, and balanced logistic regression. I also compared pretrained antibody embedding and language-model approaches, because they are a reasonable thing to try on this kind of data. In this dataset, the simpler k-mer baseline was stronger.

## What Is In The Project

The workflow has three main pieces.

First, it prepares antibody sequence records for supervised neutralisation classification. The strict labeled subset is used for model training and validation. Records with missing or conflicting labels are kept in the broader table so they can still be scored and reviewed separately.

Second, it tests the models under several validation setups. The grouped validation gives the stronger headline result, while source-holdout validation is weaker and more cautious. That difference matters: it suggests that publication or study-specific effects are part of the problem.

Third, it produces review outputs for existing records. These include scored tables, a small diversity-aware shortlist, target-region summaries where metadata is available, and unsupervised clustering and similarity summaries built from sequence features.

OAS is used as unknown-target antibody background for dataset comparison. That analysis asks how separable the project records are from natural antibody background records, separate from the neutralisation classifier.

## Questions This Tries To Answer

- How well does a simple sequence model classify strict labeled neutralisation records?
- How different are grouped validation and source-holdout validation?
- Do pretrained antibody embeddings or language models improve on the k-mer baseline here?
- Which existing records look worth reviewing first when score, metadata, and diversity are considered together?
- How different are the project records from OAS unknown-target antibody background?
- What clusters or similarity patterns appear from sequence features alone?

## Reproducing The Reports

The repository includes generated reports and machine-readable metrics. Raw and processed sequence tables are local artifacts kept outside the public repository.

For the report refresh and integrity checks:

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

The optional pretrained model scripts use the packages listed in `requirements-lm.txt`. The saved reports can be read without rerunning those heavier experiments.

## Current Results

| Check | Row subset / split | Result | Notes |
|---|---|---:|---|
| Broad grouped k-mer benchmark | Full strict labeled subset, V-gene grouped split, zero group overlap | ROC-AUC 0.7800, PR-AUC 0.8233 | Main broad baseline. |
| Paired region benchmark | Paired annotated subset, V-gene grouped split, zero group overlap | Region-only ROC-AUC 0.6629, PR-AUC 0.6330 | Region features helped within this paired subset. |
| Source-robust selection | Leave-source-out over sanitized source groups | weighted ROC-AUC 0.6095, weighted PR-AUC 0.6363 | Source/study effects remain a real limitation. |
| Calibration threshold | Source-robust selected model | threshold 0.7 precision 0.8266, recall 0.3062, coverage 0.3051 | High-confidence cutoff for reviewing existing records. |
| OAS background retrieval | Project records vs OAS unknown-target antibody background | ROC-AUC 0.9921, PR-AUC 0.9897 | Dataset/background comparison. |
| Matched OAS retrieval | Coarse length/status matched OAS background | ROC-AUC 0.9911, PR-AUC 0.9893 | The project/OAS difference remained high after coarse matching. |
| Diversity-aware shortlist | Broader prioritization table | 23 records | Small review table for existing records. |

## How I Read These Results

The grouped k-mer result is the best broad classification benchmark in the project. It is useful, but it is also optimistic compared with the source-holdout result. The source-holdout checks are a reminder that public antibody datasets carry study effects, assay differences, and label noise.

The selected broad model is `whole_pair_kmer`: compact heavy/light sequence-pair text represented with character k-mer TF-IDF and a balanced logistic-regression classifier. I treat its probabilities as ranking and review scores for existing records. The threshold analysis is included to make that use more explicit: at threshold 0.7, the model selects fewer records with higher precision and lower recall.

The OAS retrieval result is best read as a dataset/background comparison. The project records are highly separable from OAS unknown-target antibody background, and that remains true after coarse matching on length and light-chain status.

## Repository Layout

```text
data/          # Placeholder for local data; raw and processed sequence tables stay local
docs/          # Data and model cards
models/        # Small saved classical model artifacts
reports/       # Generated reports, metrics, and figures
scripts/       # Reproduction helpers
src/           # Data, model, and analysis code
tests/         # Lightweight integrity checks
```

## Useful Files

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

## Requirements

The default requirements support report checks, classical k-mer baselines, and tests. Pretrained antibody embedding and model scripts use the optional packages listed in `requirements-lm.txt`.
