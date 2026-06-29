# Antibody Prioritization

ML workflow for public SARS-CoV-2 antibody sequence records. It builds a strict labeled neutralisation dataset, compares sequence models, checks source and study sensitivity, and creates review tables for existing records.

OAS is used as unknown-target antibody background for dataset comparison.

## What Is Here

- Strict labeled records for supervised neutralisation classification
- Broader review table with existing records, including missing or conflicting labels
- Whole-pair k-mer TF-IDF logistic regression
- Pretrained antibody embedding and model comparisons
- Grouped validation, source-holdout validation, calibration, and threshold checks
- OAS background retrieval and matched OAS retrieval
- Unsupervised clustering and similarity summaries from sequence features

## Selected Model

`whole_pair_kmer`: compact heavy/light sequence-pair text, character k-mer TF-IDF features, and balanced logistic regression.

Among the tested approaches, the k-mer baseline performed best on this public-label dataset.

## Results

| Area | Result | What It Means |
|---|---:|---|
| Broad k-mer, grouped split | ROC-AUC 0.7800, PR-AUC 0.8233 | Main strict-label classification result. |
| Paired region model | ROC-AUC 0.6629, PR-AUC 0.6330 | Region features helped inside the paired annotated subset. |
| Source-holdout | weighted ROC-AUC 0.6095, weighted PR-AUC 0.6363 | Source and study effects are visible. |
| Threshold 0.7 | precision 0.8266, recall 0.3062, coverage 0.3051 | More selective review cutoff for existing records. |
| OAS retrieval | ROC-AUC 0.9921, PR-AUC 0.9897 | Project records are separable from OAS unknown-target background. |
| Matched OAS retrieval | ROC-AUC 0.9911, PR-AUC 0.9893 | Separation stayed high after coarse length/status matching. |
| Diversity-aware shortlist | 23 records | Small review table from the broader record set. |

## How To Read This

The grouped k-mer result is the main classification benchmark. The lower source-holdout result is important because it shows that source and study structure affect the task.

The model scores are used for ranking and review of existing records. The threshold analysis shows the precision/recall tradeoff for a more selective cutoff.

OAS retrieval is a dataset/background comparison using unknown-target antibody background records.

## Reproduce

The repository includes generated reports and machine-readable metrics. Raw and processed sequence tables stay local.

```bash
python -m pip install -r requirements.txt
make report
make test
```

Direct script:

```bash
bash scripts/reproduce_final_reports.sh
```

Optional pretrained model scripts use `requirements-lm.txt`.

## Useful Files

- `reports/final_project_report.md`
- `reports/model_registry.md`
- `reports/source_robust_model_selection_report.md`
- `reports/calibration_threshold_report.md`
- `reports/oas_background_retrieval_report.md`
- `reports/oas_matched_background_retrieval_report.md`
- `reports/unsupervised_antibody_landscape_report.md`
- `docs/DATA_CARD.md`
- `docs/MODEL_CARD.md`

Machine-readable summaries are under `reports/metrics/`.
