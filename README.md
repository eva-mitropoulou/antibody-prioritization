# Antibody Prioritization

This project explores how different ML models and validation checks behave on public SARS-CoV-2 antibody records. It compares simple k-mer TF-IDF logistic-regression baselines, CDR/region feature variants, pretrained antibody representation runs, grouped validation, source/study holdout, calibration and threshold analysis, and OAS background retrieval.

The aim is retrospective and practical: use existing public records to benchmark sequence-based ranking signals, understand where those signals are stable or fragile, and build a small review shortlist for closer inspection. The project does not design antibodies or generate new sequences.

Most rows represent one public antibody entry, usually with a heavy-chain or VHH amino-acid sequence, sometimes a light-chain sequence, source metadata, target-region metadata, and, when available, a neutralising or non-neutralising label.

## Table of Contents

- [At a Glance](#at-a-glance)
- [Project Workflow](#project-workflow)
- [Main Results](#main-results)
- [Selected Model](#selected-model)
- [How To Read This](#how-to-read-this)
- [Figures](#figures)
- [Scope and Limits](#scope-and-limits)
- [Reproduce](#reproduce)
- [Useful Files](#useful-files)

## At a Glance

| Part | What it does |
|---|---|
| Data curation | Cleans public SARS-CoV-2 antibody entries and separates rows with clear yes/no neutralisation labels from rows with missing or conflicting labels. |
| Main classifier | Represents heavy/light-chain sequence text with k-mer TF-IDF features and trains balanced logistic regression. |
| Model comparison | Compares the k-mer baseline with pretrained antibody embedding and language-model runs. |
| CDR and region checks | Tests CDR/region feature views on the paired annotated subset. |
| Validation | Uses grouped validation, source/study holdout, calibration, and threshold analysis to check how the signal behaves. |
| Existing-row review | Scores the broader public-record table and builds a diversity-aware review shortlist. |
| OAS comparison | Uses OAS as unknown-target antibody background for a separate dataset-comparison task. |
| Unsupervised analysis | Summarizes clustering and similarity patterns from sequence features. |

<p align="center">
  <img src="docs/assets/project_workflow.png" alt="Project workflow from public antibody rows to model validation and review outputs" width="100%">
</p>

## Project Workflow

The first step is cleaning public CoV-AbDab records into project tables. Rows are filtered to public entries whose `Binds to` field mentions SARS-CoV-2. Sequence fields are normalized, common placeholder values are treated as missing, canonical amino-acid checks are applied, and a sequence key is built from the heavy/VHH chain plus the light chain when present.

Neutralisation labels come from the public record fields. A row is treated as label 1 when `Neutralising Vs` mentions SARS-CoV-2. A row is treated as label 0 when `Not Neutralising Vs` mentions SARS-CoV-2 and the positive field does not. Rows where both fields point to SARS-CoV-2 are marked as conflicts.

The strict labeled table contains rows with usable binary labels. It is used for supervised benchmarking, source/study validation, calibration checks, model selection, and sequence-space summaries. Rows with missing or conflicting labels are not used for strict supervised metrics, but they stay in the broader prepared table so they can be scored and reviewed later.

| Table | Rows | Used for |
|---|---:|---|
| Strict labeled ML table | 5,573; label 0 = 2,292, label 1 = 3,281 | Matched broad benchmarking, source/study holdout, calibration, model selection, and sequence-space summaries. |
| Broader prepared table | 11,748 | Existing-record scoring, missing/conflicting-label review categories, and shortlist construction. |
| Paired annotated subset | 5,092 | CDR and region feature checks on rows with paired-chain annotation. |

For ML, antibody entries are represented as sequence text in several ways: whole-pair, heavy-only, paired-only whole-pair, CDR/region, and whole-pair plus CDR/region. These views are not all available for the same rows, so full strict-table results and paired/region-subset results are reported separately.

In this repository, a k-mer TF-IDF logistic-regression model means that sequence text is split into overlapping amino-acid character k-mers, those k-mers are weighted with TF-IDF, and a balanced logistic-regression classifier is fit to the public binary labels.

Grouped validation holds out sequence groups with zero group overlap. Source/study holdout validation holds out source groups to ask whether the model still works when study-level structure changes. Calibration and threshold analysis then ask how model scores behave as review cutoffs rather than as calibrated biological probabilities.

OAS background retrieval is separate from the main binary-label benchmark. It compares project rows with OAS unknown-target antibody background rows. OAS is not treated as non-neutralising neutralisation data.

The review shortlist is built from the broader scored table. Existing rows with high model scores, high confidence, acceptable heuristic risk, and review-relevant record categories form a candidate pool; the final diversity-aware shortlist keeps one representative per diversity group and contains 23 records.

## Main Results

| Area | Result | What it means |
|---|---:|---|
| Broad whole-pair k-mer benchmark | ROC-AUC 0.7800, PR-AUC 0.8233 | The sequence baseline learns signal on the strict labeled table. |
| Paired/region benchmark | ROC-AUC 0.6629, PR-AUC 0.6330 | CDR/region features are evaluated on the paired annotated subset, not the full strict table. |
| Source-robust selected model | `whole_pair_kmer` | The selected model under source-robust model selection. |
| Source/study holdout | weighted ROC-AUC 0.6095, weighted PR-AUC 0.6363 | Performance is lower when whole sources are held out. |
| Threshold 0.7 | precision 0.8266, recall 0.3062, coverage 0.3051 | A more selective cutoff for review of existing rows. |
| Broad OAS retrieval | ROC-AUC 0.9921, PR-AUC 0.9897 | Project rows are separable from OAS unknown-target antibody background. |
| Matched OAS retrieval | ROC-AUC 0.9911, PR-AUC 0.9893 | Separation stays high after coarse length and light-chain matching. |
| Diversity-aware shortlist | 23 records | A small review queue from the broader row set. |

<p align="center">
  <img src="reports/figures/threshold_precision_recall.png" alt="Threshold precision and recall tradeoff" width="48%">
  <img src="reports/figures/oas_matched_retrieval_score_distribution.png" alt="Matched OAS retrieval score distribution" width="48%">
</p>

## Selected Model

The selected source-robust model is `whole_pair_kmer`. It uses compact heavy/light sequence-pair text, character k-mer TF-IDF features, and balanced logistic regression.

In model-card terms, the k-mer setup is `TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=2)` plus `LogisticRegression(max_iter=5000, class_weight="balanced")`. The workflow uses compact sequence strings for these k-mer inputs.

This model was kept as the broad scorer because it works on the full strict labeled table, has the best matched broad k-mer result, remains the selected source-robust model, and is simpler than the pretrained alternatives. Pretrained antibody representation runs are kept as benchmark evidence, but none reliably replaces the matched k-mer references on both primary metrics.

Its scores are used for ranking and review of existing records, not as biological proof.

## How To Read This

The broad whole-pair k-mer benchmark is the main strict-table classification result. It asks whether amino-acid sequence fields carry useful signal on rows with clear yes/no labels.

The paired/region benchmark answers a narrower question. It uses rows where paired-chain CDR/region annotation is available, so it should not be compared directly against full strict-table metrics.

The source/study holdout result is intentionally more skeptical. It tests whether the model still performs when whole source groups are held out. The lower value matters because public antibody records carry study-specific structure.

The threshold analysis turns model scores into possible review cutoffs. At threshold 0.7, the model covers about 30.5% of evaluated rows with higher precision and lower recall.

The OAS analyses are background comparisons. They show that project records are separable from OAS unknown-target antibody background, including after coarse matching, but this is not the same as proving neutralisation or using OAS as a negative class.

## Figures

The left plot, `threshold_precision_recall.png`, shows how precision and recall change as the review threshold moves. Higher thresholds select fewer rows; in this project, threshold 0.7 is reported as a selective review cutoff.

The right plot, `oas_matched_retrieval_score_distribution.png`, shows the matched OAS retrieval score distribution. It compares project records with OAS unknown-target antibody background after coarse matching by heavy-chain length, light-chain length, total length, and light-chain status. The strong separation is a background-retrieval diagnostic, not a neutralisation benchmark.

## Scope and Limits

This is a retrospective public-record ML project. It does not perform antibody design, sequence generation, sequence optimization, or prospective wet-lab validation.

Model scores are ranking signals for existing-record review. They are not calibrated biological truth and do not establish neutralisation, binding, developability, or therapeutic value.

Results are sensitive to source/study validation. Background comparison metrics are separate from the main binary-label benchmark.

OAS is used as unknown-target antibody background. It is not used as non-neutralising neutralisation data.

## Reproduce

The repository includes generated reports and machine-readable metrics. Some raw and processed sequence tables are local artifacts and may not be committed.

Lightweight report refresh plus tests:

```bash
python -m pip install -r requirements.txt
make reproduce-small
make test
```

Direct script:

```bash
RUN_TESTS=0 bash scripts/reproduce_final_reports.sh
```

`make report` runs the same report script with tests enabled. OAS retrieval steps are skipped if local standardized OAS data is missing. Optional pretrained model scripts use `requirements-lm.txt`.

## Useful Files

- `reports/final_project_report.md`
- `reports/model_registry.md`
- `docs/DATA_CARD.md`
- `docs/MODEL_CARD.md`
- `scripts/reproduce_final_reports.sh`
- `Makefile`

Machine-readable summaries are under `reports/metrics/`.
