# Antibody Prioritization

This project builds an antibody sequence ML pipeline using public SARS-CoV-2 antibody records. I curated labeled public records, trained ML models to learn patterns associated with neutralising versus non-neutralising sequences, and then used the trained scoring workflow to prioritize existing OAS antibody records that look most similar to known neutralizing antibodies. The goal is finding existing records that may be worth closer expert review.

## Table of Contents

- [Project Workflow](#project-workflow)
- [Model Benchmarking and Selection](#model-benchmarking-and-selection)
- [Main Results](#main-results)
- [Selected Model](#selected-model)
- [How To Read This](#how-to-read-this)
- [Figures](#figures)
- [Scope and Limits](#scope-and-limits)
- [Reproduce](#reproduce)
- [Useful Files](#useful-files)


## Project Workflow

<p align="center">
  <img src="docs/assets/project_workflow.png" alt="Project workflow from public antibody rows to model validation and review outputs" width="100%">
</p>

The project starts with CoV AbDab SARS CoV 2 entries: heavy/VHH and light chain sequences are cleaned, missing placeholders are standardised, amino acid strings are checked, and each record is linked to its source and target region metadata when available.

Neutralisation labels are taken directly from the public record fields. Records reported as neutralising against SARS CoV 2 form the positive class, records reported as not neutralising form the negative class, and conflicting records are kept separate rather than forced into the supervised benchmark.

After curation, the data is organised into working tables. The strict labelled table is used for model benchmarking, grouped validation, source holdout, calibration, and model selection. The broader prepared table keeps records with missing or conflicting labels so they can still be scored and reviewed. A paired annotated subset is used separately for CDR and region based comparisons.

| Table | Rows | Used for |
|---|---:|---|
| Strict labelled ML table | 5,573; label 0 = 2,292, label 1 = 3,281 | Supervised benchmarking, source/study holdout, calibration, model selection, and sequence space summaries. |
| Broader prepared table | 11,748 | Existing record scoring, missing/conflicting label review categories, and shortlist construction. |
| Paired annotated subset | 5,092 | CDR and region comparisons on rows with paired chain annotation. |

For modelling, each antibody record is represented as heavy/VHH sequence, paired heavy-light sequence when available, CDR/region sequence, or combined whole-pair plus region sequence. These representations are evaluated separately because not all records contain the same chain fields or region annotations.

The main baseline uses amino acid k mer TF IDF features with logistic regression and class weights. Pretrained antibody representations, including AbLang2 and IgBERT based experiments, are benchmarked as comparisons rather than assumed to be better.

The OAS analysis is kept separate from the neutralisation benchmark. OAS records are treated as unknown target antibody background, then existing OAS records are ranked using model score and similarity to curated positive CoV AbDab records. Final review lists use hashed outputs, sequence review flags, and diversity filtering to avoid near duplicate shortlists.

## Validation Strategy

The workflow is evaluated with several checks before any record shortlist is interpreted:

1. **Strict and broader label curation:**  
   Clear neutralising and non neutralising records are used for supervised evaluation. Missing or conflicting labels are kept separate for review instead of being forced into the benchmark.

1. **Grouped validation:**  
   Related antibody records are kept together during train/test splitting, so closely related sequence families do not appear on both sides of the split.

1. **Source and study holdout validation:**  
   Entire source or study groups are held out from training to test whether performance survives publication or dataset shifts.

1. **Calibration and threshold analysis:**  
   Calibration checks whether model scores behave like probabilities. Threshold analysis measures how precision, recall, and coverage change when only records above a selected score cutoff are reviewed.

1. **CDR and region comparisons:**  
   Whole sequence and CDR/region representations are compared to test whether the label signal is concentrated in antigen binding regions or distributed across the paired sequence.

1. **Pretrained antibody representation benchmarks:**  
   The k mer baseline is compared with pretrained antibody representation models, including AbLang2 and IgBERT based runs.

1. **OAS background controls:**  
   Broad and length matched OAS retrieval controls test how separable curated CoV AbDab records are from external unknown target antibody background.

1. **Nearest neighbour similarity checks:**  
   Existing OAS records are compared with curated positive CoV AbDab records to add local sequence neighbourhood context to the ranking.

1. **Diverse shortlist selection:**  
   Final review lists avoid returning many near duplicate records by preserving diversity across sequence and metadata groups.

## Model Benchmarking and Selection

The main supervised benchmark compared sequence models on the strict labelled CoV AbDab table. The simplest model used amino acid k mer TF IDF features with logistic regression, while the pretrained model experiments tested antibody language model representations, including AbLang2 embeddings and IgBERT fine tuning.

The first comparison showed that the whole pair k mer model and IgBERT fine tuning were close. The k mer model reached ROC AUC 0.7800 and PR AUC 0.8233, while the best single IgBERT fine tuning run reached ROC AUC 0.7695 and PR AUC 0.8317. IgBERT improved PR AUC slightly, but did not improve ROC AUC.

<p align="center">
  <img src="docs/assets/broad_model_benchmark.png" alt="Broad model benchmark on the full strict labelled dataset" width="100%">
</p>

This same-subset benchmark shows why the first comparison was close, but not a clear win for IgBERT on both primary metrics.

Because this was not a clear win, I ran additional checks instead of selecting the neural model from one strong run. A five seed IgBERT fine tuning check gave lower mean performance, with ROC AUC 0.7443 and PR AUC 0.8151. Later IgBERT variants also did not consistently improve over the k mer baseline.

<p align="center">
  <img src="docs/assets/kmer_vs_igbert_followup.png" alt="K-mer and IgBERT follow-up model comparison" width="100%">
</p>

The seed-averaged follow-up supports retaining the k mer model rather than selecting a neural model from one strong run.

The final broad scorer was therefore the whole pair k mer model. It was retained because it performed strongly on the full strict labelled dataset, remained simpler and easier to reproduce, and no same subset pretrained alternative clearly improved both primary metrics.

I then tested the selected model under stricter validation. Grouped validation reduced sequence family leakage, while source and study holdout tested whether performance survived publication level shifts. The source holdout result was lower, with weighted ROC AUC 0.6095 and weighted PR AUC 0.6363, so model scores are treated as ranking signals for review rather than final biological labels.

Calibration and threshold analysis were used after model selection. The threshold 0.7 setting selected fewer records but with higher precision, making it useful for focused review lists. This is the score cutoff used to discuss high confidence review behaviour, not a claim that the score is a calibrated probability.

<p align="center">
  <img src="docs/assets/selected_model_robustness.png" alt="Selected model robustness and threshold 0.7 review cutoff" width="100%">
</p>

The selected model is useful for review ranking, but the source/study holdout drop motivates conservative interpretation.

## Main Results

The simple k mer model performed well on the curated labelled benchmark, but source holdout showed a clear drop, so the final outputs are treated as review lists rather than final biological labels.

### 1. Supervised neutralisation benchmark

| Result | Value | Interpretation |
|---|---:|---|
| Strict labelled dataset | 5,573 records | 2,292 non neutralising, 3,281 neutralising |
| Selected broad model | `whole_pair_kmer` | Whole pair k mer TF IDF with logistic regression |
| Main k mer benchmark | ROC AUC 0.7800, PR AUC 0.8233 | On the strict labelled table, the whole pair k mer model ranks reported neutralising records above reported non neutralising records under grouped validation. |
| CDR/region subset benchmark | ROC AUC 0.6629, PR AUC 0.6330 | CDR and region based inputs were tested on the paired annotated subset only, so this result is reported separately from the full table benchmark. |
| IgBERT fine tuning benchmark | ROC AUC 0.7695, PR AUC 0.8317 | Fine tuning improved PR AUC slightly but did not improve ROC AUC, so it was kept as comparison evidence rather than replacing the k mer model. |

### 2. Robustness and score interpretation

| Check | Result | Interpretation |
|---|---:|---|
| Source/study holdout | weighted ROC AUC 0.6095, weighted PR AUC 0.6363 | Performance drops when whole sources are held out |
| Selected source robust model | `whole_pair_kmer` | The simpler broad model remained the selected scorer |
| Threshold 0.7 | precision 0.8266, recall 0.3062, coverage 0.3051 | Useful as a selective review cutoff |
| Calibration | imperfect | Scores are better used for ranking than as literal probabilities |

### 3. Existing record review outputs

| Output | Result | Interpretation |
|---|---:|---|
| Broader CoV AbDab table | 11,748 records | Includes missing/conflicting labels for review |
| Broader CoV AbDab shortlist | 23 records | Compact review list after filtering and diversity selection |
| Broad OAS retrieval control | ROC AUC 0.9921, PR AUC 0.9897 | CoV AbDab records are separable from broad OAS background |
| Matched OAS retrieval control | ROC AUC 0.9911, PR AUC 0.9893 | Separation remains after coarse length and light chain matching |
| OAS existing record scoring | 17,882 OAS rows scored | Existing OAS records ranked with model score and similarity to curated positives |
| OAS shortlist | top 25 diverse records | Public safe expert review queue |

## Selected Model

The selected source-robust model is `whole_pair_kmer`. It uses compact heavy/light sequence-pair text, character k-mer TF-IDF features, and balanced logistic regression.

In model-card terms, the k-mer setup is `TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=2)` plus `LogisticRegression(max_iter=5000, class_weight="balanced")`. The workflow uses compact sequence strings for these k-mer inputs.

This model was kept as the broad scorer because it works on the full strict labeled table, has the best matched broad k-mer result, remains the selected source-robust model, and is simpler than the pretrained alternatives. Pretrained antibody representation runs are kept as benchmark evidence, but none reliably replaces the matched k-mer references on both primary metrics.

Its scores are used for ranking and review of existing records, not as biological proof.

## How To Read This

The broad whole-pair k-mer benchmark is the main strict-table classification result. It asks whether amino-acid sequence fields carry useful signal on rows with clear yes/no labels.

The paired/region benchmark answers a narrower question. It uses rows where paired-chain CDR/region annotation is available, so it should not be compared directly against full strict-table metrics.

The source/study holdout result is the skeptical check. It tests whether the model still performs when whole source groups are held out. The lower value matters because public antibody records carry study-specific structure.

The threshold analysis turns model scores into possible review cutoffs. At threshold 0.7, the model covers about 30.5% of evaluated rows with higher precision and lower recall.

The OAS tasks should be read as background and review workflows. OAS rows are unknown-target natural antibody background, not assayed negative neutralisation data. Similarity to curated project-positive records can help organize records for review, but it does not establish binding, neutralisation, or therapeutic value.

## Figures

The left plot, `threshold_precision_recall.png`, shows how precision and recall change as the review threshold moves. Higher thresholds select fewer rows; in this project, threshold 0.7 is reported as a selective review cutoff.

The right plot, `oas_matched_retrieval_score_distribution.png`, shows the matched OAS retrieval score distribution. It compares project records with OAS unknown-target antibody background after coarse matching by heavy-chain length, light-chain length, total length, and light-chain status. The strong separation is a background-retrieval diagnostic, not a neutralisation benchmark.

## Scope and Limits

This is a retrospective public-record ML project. It does not perform antibody design, sequence generation, sequence optimization, or prospective wet-lab validation.

Model scores are ranking signals for existing-record review. They are not calibrated biological truth and do not establish neutralisation, binding, developability, or therapeutic value.

OAS is used as unknown-target antibody background. It is not used as non-neutralising neutralisation data.

The OAS existing-record shortlist is an expert-review queue. It is not antibody design, therapeutic discovery, or prospective validation.

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
- `reports/oas_existing_record_shortlist_report.md`
- `docs/DATA_CARD.md`
- `docs/MODEL_CARD.md`
- `scripts/reproduce_final_reports.sh`
- `Makefile`

Machine-readable summaries are under `reports/metrics/`.
