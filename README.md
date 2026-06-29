# Antibody Prioritization

This project asks whether a simple classifier can use public SARS-CoV-2 antibody amino-acid sequences to predict neutralising vs non-neutralising labels better than random baselines.

Each row in the data is one antibody entry. A row can include an antibody name, heavy-chain sequence, light-chain sequence, source or study metadata, and a neutralising or non-neutralising label. The training set uses rows where that label is clean enough to treat as yes or no. Rows with missing or conflicting labels stay in the project as review items.

## At a Glance

| Question | Answer |
|---|---|
| What is the data? | Public SARS-CoV-2 antibody entries. |
| What is the model input? | Heavy-chain and light-chain amino-acid sequences. |
| What is the label? | Neutralising vs non-neutralising, when the row has a clean label. |
| Main baseline | k-mer TF-IDF features from antibody sequences, then logistic regression. |
| Other models checked | Pretrained antibody embeddings and antibody language-model runs. |
| Main validation checks | Grouped validation, source or study holdout, calibration, and threshold analysis. |
| What happens to unclear rows? | Rows with missing or conflicting labels are kept for scoring and review. |
| What is OAS doing? | OAS is unknown-target antibody background for dataset comparison. |

## Main Results

| Area | Result | What It Means |
|---|---:|---|
| Broad k-mer, grouped split | ROC-AUC 0.7800, PR-AUC 0.8233 | The baseline learns useful signal on clean labeled rows. |
| Paired region model | ROC-AUC 0.6629, PR-AUC 0.6330 | Region features helped inside the paired annotated subset. |
| Source or study holdout | weighted ROC-AUC 0.6095, weighted PR-AUC 0.6363 | Performance drops when whole sources are held out. |
| Threshold 0.7 | precision 0.8266, recall 0.3062, coverage 0.3051 | More selective cutoff for reviewing existing rows. |
| OAS retrieval | ROC-AUC 0.9921, PR-AUC 0.9897 | SARS-CoV-2 antibody rows are separable from OAS background. |
| Matched OAS retrieval | ROC-AUC 0.9911, PR-AUC 0.9893 | Separation stayed high after coarse length and light-chain matching. |
| Diversity-aware shortlist | 23 rows | Small review table from the broader row set. |

<p align="center">
  <img src="reports/figures/threshold_precision_recall.png" alt="Threshold precision and recall tradeoff" width="48%">
  <img src="reports/figures/oas_matched_retrieval_score_distribution.png" alt="Matched OAS retrieval score distribution" width="48%">
</p>

## Selected Model

`whole_pair_kmer` uses compact heavy/light sequence-pair text, character k-mer TF-IDF features, and balanced logistic regression.

Among the tested approaches, this k-mer baseline performed best on the public-label dataset. The score is used for ranking and review of existing antibody rows.

## How To Read This

The grouped split is the main classification benchmark. The source-holdout result is lower, which shows that source and study structure matter in this dataset.

The threshold analysis shows the precision and recall tradeoff for review. At threshold 0.7, the model covers about 31% of rows in the evaluated split.

OAS retrieval is a dataset comparison against unknown-target antibody background rows.

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
