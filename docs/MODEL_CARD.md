# Model Card

## Selected Model

The selected model is `whole_pair_kmer`.

It represents compact heavy/light sequence-pair text with character k-mer TF-IDF and fits a balanced logistic-regression classifier. The primary broad saved artifact is `models/kmer_logreg_pair_text.joblib`.

## Features

- Input: compact whole-pair sequence text.
- Representation: `TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=2)`.
- Classifier: `LogisticRegression(max_iter=5000, class_weight="balanced")`.
- The workflow uses compact strings, not spaced model-token strings, for k-mer inputs.

## Validation Protocols

| Protocol | Purpose | Result |
|---|---|---:|
| V-gene grouped validation | Primary matched broad benchmark with zero group overlap | ROC-AUC 0.7800, PR-AUC 0.8233 |
| Source-holdout validation | Skeptical leave-source/leave-study-out check | macro ROC-AUC 0.5605, macro PR-AUC 0.6454 |
| Source-robust model selection | Compare conservative k-mer variants on the same source groups | selected `whole_pair_kmer`, weighted ROC-AUC 0.6095, weighted PR-AUC 0.6363 |
| Calibration/threshold analysis | Estimate score reliability for review thresholds | Brier 0.2637; threshold 0.7 precision 0.8266, recall 0.3062, coverage 0.3051 |
| OAS background retrieval | Unknown-target natural antibody background enrichment diagnostic | broad ROC-AUC 0.9921, PR-AUC 0.9897 |

Pretrained antibody language-model representations were benchmarked but did not outperform the simpler matched k-mer baselines on this noisy public-label task.

## Intended Use

- Retrospective public-record prioritization.
- Model-family comparison under matched row subsets and split strategies.
- Existing-record review queues with confidence, target-region, risk, diversity, and structure metadata.
- Reproducible comparison of sequence-record models and validation controls.

## Non-Intended Use

- Therapeutic efficacy prediction.
- Prospective wet-lab selection.
- Antibody generation, design, mutation, or optimization.
- Treating OAS records as assayed negative-class labels.
- Production clinical or laboratory deployment.

## Threshold 0.7 Interpretation

In source-robust analysis, threshold 0.7 selected about 30.5% of held-out records with precision 0.8266 and recall 0.3062. This is a high-confidence review cutoff for existing records, not a calibrated guarantee of future assay outcome.

## Limitations

- Source-holdout performance is weaker than V-gene grouped validation.
- Public labels and source metadata are heterogeneous.
- Probabilities are more useful for ranking than absolute risk estimation.
- Region/CDR variants did not clearly replace the whole-pair k-mer scorer.
- OAS retrieval is a background enrichment diagnostic and remains separate from the neutralisation benchmark.
