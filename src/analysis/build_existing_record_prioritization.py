"""Build an existing-record prioritization table from saved benchmark inputs.

This script scores and annotates existing public dataset rows only. It does not
generate, alter, mutate, optimize, or propose biological sequences.

Run from the project root:

    python src/analysis/build_existing_record_prioritization.py
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd


# Matplotlib needs a writable cache directory in this environment.
MPLCONFIGDIR = Path("/tmp") / "antibody_prioritization_mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "neutral_sequence_classification_ml.csv"
BROADER_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "neutral_prepared_sequences.csv"
KMER_MODEL_PATH = PROJECT_ROOT / "models" / "kmer_logreg_pair_text.joblib"
GROUPED_METRICS_PATH = PROJECT_ROOT / "reports" / "metrics" / "grouped_validation_metrics.json"
FINETUNE_SEED_METRICS_PATH = (
    PROJECT_ROOT / "reports" / "metrics" / "pretrained_finetune_seed_check_metrics.json"
)
MODEL_REGISTRY_PATH = PROJECT_ROOT / "reports" / "metrics" / "model_registry.json"

OUTPUT_TABLE_PATH = PROJECT_ROOT / "reports" / "existing_record_prioritization_table.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "existing_record_prioritization_report.md"
SUMMARY_PATH = (
    PROJECT_ROOT / "reports" / "metrics" / "existing_record_prioritization_summary.json"
)
BROADER_OUTPUT_TABLE_PATH = (
    PROJECT_ROOT / "reports" / "broader_existing_record_prioritization_table.csv"
)
BROADER_REPORT_PATH = PROJECT_ROOT / "reports" / "broader_existing_record_prioritization_report.md"
BROADER_SUMMARY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "metrics"
    / "broader_existing_record_prioritization_summary.json"
)
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
PROBABILITY_DISTRIBUTION_PATH = (
    FIGURE_DIR / "prioritization_probability_distribution.png"
)
RISK_COUNTS_PATH = FIGURE_DIR / "developability_risk_counts.png"
PRIORITY_COUNTS_PATH = FIGURE_DIR / "priority_category_counts.png"
PROBABILITY_RISK_PATH = FIGURE_DIR / "probability_vs_developability_risk.png"
BROADER_PRIORITY_COUNTS_PATH = FIGURE_DIR / "broader_priority_category_counts.png"
BROADER_RISK_COUNTS_PATH = FIGURE_DIR / "broader_developability_risk_counts.png"
BROADER_PROBABILITY_BY_RECORD_CATEGORY_PATH = (
    FIGURE_DIR / "broader_probability_by_record_category.png"
)
BROADER_PROBABILITY_RISK_PATH = FIGURE_DIR / "broader_probability_vs_risk_score.png"

RANDOM_STATE = 42
INPUT_COLUMN = "sequence_pair_text"
LABEL_COLUMN = "label"
GROUP_COLUMN = "group_feature_v"
TARGET_REGION_COLUMN = "metadata_target_region"
TOP_RECORDS_PER_TARGET = 3

HYDROPHOBIC_RESIDUES = set("AVILMFWY")
AROMATIC_RESIDUES = set("FWY")
CHARGED_RESIDUES = set("DEKRH")
ACIDIC_RESIDUES = set("DE")
BASIC_RESIDUES = set("KRH")
MISSING_TEXT_VALUES = {"", "na", "nan", "none", "null", "nd", "n/a", "unknown"}
SEQUENCE_VALUE_COLUMNS = {
    "sequence_a",
    "sequence_b",
    "sequence_a_raw",
    "sequence_pair_text",
    "sequence_heavy_only",
    "heavy_sequence",
    "light_sequence",
    "group_feature_cdr3",
    "group_feature_b_cdr3",
    "cdrh1_seq",
    "cdrh2_seq",
    "cdrh3_seq",
    "cdrl1_seq",
    "cdrl2_seq",
    "cdrl3_seq",
    "marked_heavy_text",
    "marked_light_text",
    "whole_pair_kmer_text",
    "all_cdr_kmer_text",
    "cdrh3_cdrl3_kmer_text",
}

KMER_GROUPED_ROC_AUC = 0.7810
KMER_GROUPED_PR_AUC = 0.8236
FROZEN_PAIR_MLP_GROUPED_ROC_AUC = 0.7541
FROZEN_PAIR_MLP_GROUPED_PR_AUC = 0.8078


def read_csv(path: Path) -> pd.DataFrame:
    """Read CSV text while preserving blank fields."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON file if available."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def optional_column(data: pd.DataFrame, column: str) -> pd.Series:
    """Return a text column or aligned blanks."""
    if column in data.columns:
        return data[column]
    return pd.Series([""] * len(data), index=data.index)


def first_available_column(data: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Return the first available column from a list, else aligned blanks."""
    for column in columns:
        if column in data.columns:
            return data[column]
    return pd.Series([""] * len(data), index=data.index)


def normalized_text(values: pd.Series) -> pd.Series:
    """Normalize text for grouping and display fallbacks."""
    return values.fillna("").astype(str).str.strip()


def is_missing_text(value: Any) -> bool:
    """Return true for common missing-value tokens in this project."""
    return str(value or "").strip().lower() in MISSING_TEXT_VALUES


def normalize_sequence(value: Any) -> str:
    """Normalize an existing sequence string for counting features only."""
    text = re.sub(r"\s+", "", str(value or "")).upper()
    return "" if text.lower() in MISSING_TEXT_VALUES else text


def target_region_group_value(value: Any) -> str:
    """Map free-text target metadata into broad aggregate groups."""
    text = str(value or "").strip().lower()
    if text in MISSING_TEXT_VALUES:
        return "unknown"
    if "rbd" in text or "receptor binding" in text:
        return "RBD"
    if "ntd" in text or "n-terminal" in text or "n terminal" in text:
        return "NTD"
    if text in {"s", "spike"} or "spike" in text or "s protein" in text:
        return "Spike/S"
    return "other"


def boolean_series(data: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Parse the first useful boolean-like value across candidate columns."""
    combined = pd.Series([np.nan] * len(data), index=data.index, dtype=object)
    for column in columns:
        if column not in data.columns:
            continue
        raw = data[column]
        text = normalized_text(raw).str.lower()
        mapped = text.map(
            {
                "true": True,
                "false": False,
                "yes": True,
                "no": False,
                "1": True,
                "0": False,
            }
        )
        numeric = pd.to_numeric(raw, errors="coerce")
        numeric_bool = numeric.map(
            lambda value: bool(value) if pd.notna(value) else np.nan
        )
        parsed = mapped.where(mapped.notna(), numeric_bool)
        fill_mask = combined.isna() & parsed.notna()
        combined.loc[fill_mask] = parsed.loc[fill_mask]
    return combined.fillna(False).astype(bool)


def label_series(data: pd.DataFrame) -> pd.Series:
    """Return label values from strict or broader neutral tables."""
    raw = first_available_column(data, ["label", "neutralising_label", "extra_col_07"])
    text = normalized_text(raw)
    numeric = pd.to_numeric(text.replace({"": np.nan}), errors="coerce")
    return numeric


def construct_pair_text_from_existing(data: pd.DataFrame) -> pd.Series:
    """Build model text from existing heavy/light strings, heavy-only if no light."""
    heavy = optional_column(data, "sequence_a").map(normalize_sequence)
    light = optional_column(data, "sequence_b").map(normalize_sequence)
    pair_text = []
    for heavy_value, light_value in zip(heavy, light):
        if light_value:
            pair_text.append(f"{heavy_value}[SEP]{light_value}")
        else:
            pair_text.append(heavy_value)
    return pd.Series(pair_text, index=data.index)


def ensure_scoring_columns(
    data: pd.DataFrame,
    dataset_name: str,
    require_usable_heavy: bool,
) -> pd.DataFrame:
    """Ensure label metadata and model input columns are explicit."""
    output = data.copy()
    output["label"] = label_series(output).astype("Int64")
    output["neutralising_conflict"] = boolean_series(
        output,
        ["neutralising_conflict", "extra_col_23"],
    )

    heavy = optional_column(output, "sequence_a").map(normalize_sequence)
    if require_usable_heavy:
        usable_heavy = heavy.ne("")
        if "extra_col_15" in output.columns:
            valid_heavy = boolean_series(output, ["valid_heavy_or_vhh_sequence", "extra_col_15"])
            usable_heavy = usable_heavy & valid_heavy
        output = output.loc[usable_heavy].copy()
        heavy = heavy.loc[output.index]

    if INPUT_COLUMN in output.columns and normalized_text(output[INPUT_COLUMN]).ne("").all():
        output[INPUT_COLUMN] = output[INPUT_COLUMN].fillna("").astype(str)
    else:
        output[INPUT_COLUMN] = construct_pair_text_from_existing(output)

    output["source_dataset"] = dataset_name
    return output.reset_index(drop=True)


def make_kmer_pipeline() -> Pipeline:
    """Create the reliable k-mer TF-IDF logistic-regression model."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=5000, class_weight="balanced"),
            ),
        ]
    )


def positive_scores(model: Any, values: pd.Series) -> np.ndarray:
    """Return label-1 probabilities from a fitted estimator."""
    if not hasattr(model, "predict_proba"):
        raise TypeError("K-mer estimator must expose predict_proba.")
    class_list = list(model.classes_)
    if 1 not in class_list:
        raise ValueError("K-mer estimator was not fitted with positive class label 1.")
    positive_index = class_list.index(1)
    return model.predict_proba(values)[:, positive_index]


def load_or_train_kmer_model(data: pd.DataFrame) -> tuple[Any, dict[str, Any]]:
    """Load the saved pair-text k-mer model, or fit the same model on labels."""
    if KMER_MODEL_PATH.exists():
        try:
            model = joblib.load(KMER_MODEL_PATH)
            # Validate compatibility before committing to the saved model.
            positive_scores(model, data[INPUT_COLUMN].head(min(5, len(data))))
            return model, {
                "source": "loaded_saved_model",
                "path": str(KMER_MODEL_PATH.relative_to(PROJECT_ROOT)),
                "fallback_used": False,
                "training_rows": None,
            }
        except Exception as exc:
            load_error = str(exc)
    else:
        load_error = "saved model path not found"

    labeled = data[data[LABEL_COLUMN].notna()].copy()
    labeled = labeled[normalized_text(labeled[LABEL_COLUMN]).ne("")]
    labeled[LABEL_COLUMN] = labeled[LABEL_COLUMN].astype(int)
    if labeled[LABEL_COLUMN].nunique() != 2:
        raise ValueError("Fallback k-mer training requires two label classes.")

    model = make_kmer_pipeline()
    with threadpool_limits(limits=1):
        model.fit(labeled[INPUT_COLUMN], labeled[LABEL_COLUMN])
    return model, {
        "source": "trained_fallback_on_strict_ml_rows",
        "path": None,
        "fallback_used": True,
        "load_error": load_error,
        "training_rows": int(len(labeled)),
    }


def n_glycosylation_motif_count(sequence: str) -> int:
    """Count N-X-S/T motifs where X is not P."""
    count = 0
    for index in range(max(0, len(sequence) - 2)):
        first, second, third = sequence[index : index + 3]
        if first == "N" and second != "P" and third in {"S", "T"}:
            count += 1
    return count


def fraction_of(sequence: str, residues: set[str]) -> float:
    """Return residue fraction, or NaN for blank sequences."""
    if not sequence:
        return float("nan")
    return float(sum(1 for residue in sequence if residue in residues) / len(sequence))


def add_prediction_columns(data: pd.DataFrame, model: Any) -> pd.DataFrame:
    """Append k-mer prediction, label, and confidence columns."""
    output = data.copy()
    probabilities = positive_scores(model, output[INPUT_COLUMN])
    registry = load_json(MODEL_REGISTRY_PATH) or {}
    primary_model = (
        registry.get("primary_broad_scorer", {}).get("model_id")
        if isinstance(registry.get("primary_broad_scorer"), dict)
        else None
    ) or "kmer_tfidf_logreg_pair_text"
    output["primary_model_name"] = primary_model
    output["primary_probability"] = probabilities.astype(float)
    output["kmer_probability"] = probabilities.astype(float)
    output["pretrained_model_probability"] = np.nan
    output["predicted_neutralisation_probability"] = probabilities.astype(float)
    output["predicted_label"] = (probabilities >= 0.5).astype(int)
    output["confidence_score"] = np.abs(probabilities - 0.5)
    output["confidence_bin"] = np.select(
        [
            output["confidence_score"].ge(0.30),
            output["confidence_score"].ge(0.15) & output["confidence_score"].lt(0.30),
        ],
        ["high", "medium"],
        default="low",
    )
    # Per-record LM probabilities are optional secondary evidence. Existing
    # project metric files summarize LM benchmarks but do not currently provide
    # row-level probabilities for the broader table, so keep explicit columns
    # and mark disagreement unavailable rather than blocking prioritization.
    output["igbert_probability"] = np.nan
    output["bioaware_lm_probability"] = np.nan
    output["model_disagreement_flag"] = False
    return output


def add_sequence_flags(data: pd.DataFrame) -> pd.DataFrame:
    """Append developability-style flags from existing sequences."""
    output = data.copy()
    heavy = optional_column(output, "sequence_a").map(normalize_sequence)
    light = optional_column(output, "sequence_b").map(normalize_sequence)
    cdrh3 = optional_column(output, "group_feature_cdr3").map(normalize_sequence)
    cdrl3 = optional_column(output, "group_feature_b_cdr3").map(normalize_sequence)

    output["heavy_length"] = heavy.map(len).astype(int)
    output["light_length"] = light.map(len).astype(int)
    output["cdrh3_length"] = cdrh3.map(len).astype(int)
    output["cdrl3_length"] = cdrl3.map(len).astype(int)
    output["cysteine_count_heavy"] = heavy.map(lambda value: value.count("C")).astype(int)
    output["cysteine_count_light"] = light.map(lambda value: value.count("C")).astype(int)
    output["cysteine_count_cdrh3"] = cdrh3.map(lambda value: value.count("C")).astype(int)
    output["aromatic_fraction_heavy"] = heavy.map(
        lambda value: fraction_of(value, AROMATIC_RESIDUES)
    )
    output["hydrophobic_fraction_heavy"] = heavy.map(
        lambda value: fraction_of(value, HYDROPHOBIC_RESIDUES)
    )
    output["hydrophobic_fraction_cdrh3"] = cdrh3.map(
        lambda value: fraction_of(value, HYDROPHOBIC_RESIDUES)
    )
    output["charged_fraction_heavy"] = heavy.map(
        lambda value: fraction_of(value, CHARGED_RESIDUES)
    )
    output["acidic_count_heavy"] = heavy.map(
        lambda value: sum(1 for residue in value if residue in ACIDIC_RESIDUES)
    ).astype(int)
    output["basic_count_heavy"] = heavy.map(
        lambda value: sum(1 for residue in value if residue in BASIC_RESIDUES)
    ).astype(int)
    output["net_charge_proxy_heavy"] = (
        output["basic_count_heavy"] - output["acidic_count_heavy"]
    ).astype(int)
    output["n_glycosylation_motif_count_heavy"] = heavy.map(
        n_glycosylation_motif_count
    ).astype(int)
    output["methionine_count_heavy"] = heavy.map(lambda value: value.count("M")).astype(int)
    output["tryptophan_count_heavy"] = heavy.map(lambda value: value.count("W")).astype(int)
    output["proline_fraction_cdrh3"] = cdrh3.map(lambda value: fraction_of(value, {"P"}))

    output["long_cdrh3_flag"] = output["cdrh3_length"].gt(20)
    output["very_long_cdrh3_flag"] = output["cdrh3_length"].gt(30)
    output["unusual_heavy_length_flag"] = output["heavy_length"].lt(105) | output[
        "heavy_length"
    ].gt(140)
    output["missing_light_flag"] = light.map(lambda value: value == "")
    output["paired_light_status"] = np.where(
        output["missing_light_flag"],
        "light_missing_or_single_chain",
        "paired",
    )

    type_text = normalized_text(optional_column(output, "sample_type")).str.lower()
    explicit_nanobody = boolean_series(
        output,
        ["is_nanobody_like", "extra_col_22", "extra_col_15"],
    )
    type_nanobody = type_text.str.contains(r"\b(?:nb|nanobody|vhh|sdab)\b", regex=True)
    output["nanobody_like_flag"] = explicit_nanobody | type_nanobody | output[
        "missing_light_flag"
    ]

    structure_from_bool = boolean_series(output, ["has_structure", "extra_col_08", "extra_col_16"])
    if "source_dataset" in output.columns and output["source_dataset"].eq(
        "broader_cleaned"
    ).all():
        structure_text_columns = ["metadata_structure", "extra_col_11"]
    else:
        structure_text_columns = ["metadata_structure"]
    structure_text = normalized_text(first_available_column(output, structure_text_columns))
    structure_from_text = structure_text.map(lambda value: not is_missing_text(value))
    output["has_structure"] = structure_from_bool | structure_from_text.astype(bool)
    return output


def add_risk_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Append simple developability risk score and bin."""
    output = data.copy()
    risk_components = pd.DataFrame(
        {
            "cdrh3_length_gt_20": output["cdrh3_length"].gt(20),
            "cdrh3_length_gt_30": output["cdrh3_length"].gt(30),
            "high_hydrophobic_fraction_cdrh3": output[
                "hydrophobic_fraction_cdrh3"
            ].fillna(0.0)
            > 0.45,
            "high_hydrophobic_fraction_heavy": output[
                "hydrophobic_fraction_heavy"
            ].fillna(0.0)
            > 0.42,
            "high_cysteine_count_cdrh3": output["cysteine_count_cdrh3"].gt(1),
            "high_cysteine_count_heavy": output["cysteine_count_heavy"].gt(4),
            "n_glycosylation_motif_heavy": output[
                "n_glycosylation_motif_count_heavy"
            ].gt(0),
            "large_abs_net_charge_proxy_heavy": output[
                "net_charge_proxy_heavy"
            ].abs()
            > 12,
            "unusual_heavy_length": output["unusual_heavy_length_flag"],
        },
        index=output.index,
    )
    output["developability_risk_score"] = risk_components.sum(axis=1).astype(int)
    output["developability_risk_bin"] = np.select(
        [
            output["developability_risk_score"].le(1),
            output["developability_risk_score"].between(2, 3),
        ],
        ["low", "medium"],
        default="high",
    )
    for column in risk_components.columns:
        output[f"risk_component_{column}"] = risk_components[column].astype(bool)
    return output


def add_diversity_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Append grouping columns for diversity/novelty inspection."""
    output = data.copy()
    cdrh3_length = output["cdrh3_length"]
    output["cdrh3_length_bin"] = np.select(
        [cdrh3_length.le(10), cdrh3_length.between(11, 20), cdrh3_length.gt(20)],
        ["short", "medium", "long"],
        default="unknown",
    )

    group_feature_v = normalized_text(optional_column(output, GROUP_COLUMN)).replace(
        {"": "unknown_v"}
    )
    target_region = normalized_text(optional_column(output, TARGET_REGION_COLUMN)).replace(
        {"": "unknown_target_region"}
    )
    output["target_region_group"] = target_region.map(target_region_group_value)
    output["diversity_group"] = (
        group_feature_v
        + " | "
        + target_region
        + " | "
        + output["cdrh3_length_bin"].astype(str)
    )
    output["diversity_group_size"] = output.groupby("diversity_group")[
        "diversity_group"
    ].transform("size")
    output["priority_rank_within_diversity_group"] = (
        output.groupby("diversity_group")["predicted_neutralisation_probability"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    output["priority_rank_within_target_region"] = (
        output.groupby(target_region)["predicted_neutralisation_probability"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return output


def add_record_category(data: pd.DataFrame) -> pd.DataFrame:
    """Classify existing records by observed label and conflict metadata."""
    output = data.copy()
    label_numeric = pd.to_numeric(output[LABEL_COLUMN], errors="coerce")
    conflict = output.get("neutralising_conflict")
    if conflict is None:
        conflict = pd.Series(False, index=output.index)
    else:
        conflict = conflict.astype(bool)

    output["record_category"] = "other"
    output.loc[label_numeric.isna(), "record_category"] = "missing_label"
    output.loc[label_numeric.eq(0), "record_category"] = "known_non_neutralising"
    output.loc[label_numeric.eq(1), "record_category"] = "known_neutralising"
    output.loc[conflict, "record_category"] = "conflict_label"
    return output


def add_priority_category(data: pd.DataFrame) -> pd.DataFrame:
    """Assign mutually exclusive existing-record priority categories."""
    output = data.copy()
    high_probability = output["predicted_neutralisation_probability"].ge(0.75)
    high_confidence = output["confidence_bin"].eq("high")
    acceptable_risk = output["developability_risk_bin"].isin(["low", "medium"])
    high_risk = output["developability_risk_bin"].eq("high")
    high_score_confident = high_probability & high_confidence

    output["priority_category"] = "lower_priority"
    output.loc[output["confidence_bin"].eq("low"), "priority_category"] = "uncertain_prediction"
    output.loc[high_score_confident & high_risk, "priority_category"] = (
        "high_score_but_sequence_risk"
    )
    output.loc[
        output["record_category"].eq("missing_label")
        & high_score_confident
        & acceptable_risk,
        "priority_category",
    ] = "high_score_missing_label"
    output.loc[
        output["record_category"].eq("conflict_label")
        & high_score_confident
        & acceptable_risk,
        "priority_category",
    ] = "high_score_conflict_label"
    output.loc[
        output["record_category"].eq("known_neutralising")
        & high_score_confident
        & acceptable_risk,
        "priority_category",
    ] = "high_confidence_known_positive"
    return output


def build_prioritization_table(raw_data: pd.DataFrame, model: Any) -> pd.DataFrame:
    """Build the full prioritization table."""
    data = raw_data.copy()
    data.insert(0, "row_id", np.arange(len(data), dtype=int))
    data = ensure_scoring_columns(
        data,
        dataset_name="strict_labeled_ml",
        require_usable_heavy=False,
    )
    if data[INPUT_COLUMN].str.len().eq(0).any():
        raise ValueError(f"{INPUT_COLUMN} contains blank inputs.")

    table = add_prediction_columns(data, model)
    table = add_sequence_flags(table)
    table = add_risk_columns(table)
    table = add_diversity_columns(table)
    table = add_record_category(table)
    table = add_priority_category(table)
    return order_output_columns(table)


def build_broader_prioritization_table(raw_data: pd.DataFrame, model: Any) -> pd.DataFrame:
    """Build prioritization over the broader cleaned existing-record table."""
    data = raw_data.copy()
    data.insert(0, "row_id", np.arange(len(data), dtype=int))
    data = ensure_scoring_columns(
        data,
        dataset_name="broader_cleaned",
        require_usable_heavy=True,
    )
    if data[INPUT_COLUMN].str.len().eq(0).any():
        raise ValueError(f"{INPUT_COLUMN} contains blank inputs after heavy filtering.")

    table = add_prediction_columns(data, model)
    table = add_sequence_flags(table)
    table = add_risk_columns(table)
    table = add_diversity_columns(table)
    table = add_record_category(table)
    table = add_priority_category(table)
    return order_output_columns(table)


def order_output_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Place high-signal prioritization columns first, then original metadata."""
    leading_columns = [
        "row_id",
        "sample_name",
        "sample_type",
        "label",
        "primary_model_name",
        "primary_probability",
        "kmer_probability",
        "pretrained_model_probability",
        "predicted_neutralisation_probability",
        "predicted_label",
        "igbert_probability",
        "bioaware_lm_probability",
        "model_disagreement_flag",
        "confidence_score",
        "confidence_bin",
        "record_category",
        "priority_category",
        "developability_risk_score",
        "developability_risk_bin",
        "diversity_group",
        "diversity_group_size",
        "priority_rank_within_diversity_group",
        "priority_rank_within_target_region",
        "group_feature_v",
        "metadata_target_region",
        "target_region_group",
        "neutralising_conflict",
        "has_structure",
        "missing_light_flag",
        "paired_light_status",
        "nanobody_like_flag",
        "heavy_length",
        "light_length",
        "cdrh3_length",
        "cdrl3_length",
        "cdrh3_length_bin",
        "cysteine_count_heavy",
        "cysteine_count_light",
        "cysteine_count_cdrh3",
        "aromatic_fraction_heavy",
        "hydrophobic_fraction_heavy",
        "hydrophobic_fraction_cdrh3",
        "charged_fraction_heavy",
        "acidic_count_heavy",
        "basic_count_heavy",
        "net_charge_proxy_heavy",
        "n_glycosylation_motif_count_heavy",
        "methionine_count_heavy",
        "tryptophan_count_heavy",
        "proline_fraction_cdrh3",
        "long_cdrh3_flag",
        "very_long_cdrh3_flag",
        "unusual_heavy_length_flag",
        "risk_component_cdrh3_length_gt_20",
        "risk_component_cdrh3_length_gt_30",
        "risk_component_high_hydrophobic_fraction_cdrh3",
        "risk_component_high_hydrophobic_fraction_heavy",
        "risk_component_high_cysteine_count_cdrh3",
        "risk_component_high_cysteine_count_heavy",
        "risk_component_n_glycosylation_motif_heavy",
        "risk_component_large_abs_net_charge_proxy_heavy",
        "risk_component_unusual_heavy_length",
        "metadata_origin",
    ]
    existing_leading = [column for column in leading_columns if column in table.columns]
    remaining = [
        column
        for column in table.columns
        if column not in existing_leading and column not in SEQUENCE_VALUE_COLUMNS
    ]
    return table[existing_leading + remaining]


def value_counts_dict(series: pd.Series) -> dict[str, int]:
    """Return stable value counts for JSON output."""
    counts = series.fillna("missing").astype(str).value_counts()
    return {str(key): int(value) for key, value in counts.items()}


def model_metrics_summary() -> dict[str, Any]:
    """Load saved benchmark metrics used to justify model choice."""
    grouped = load_json(GROUPED_METRICS_PATH)
    finetune = load_json(FINETUNE_SEED_METRICS_PATH)

    grouped_kmer = None
    if grouped:
        grouped_kmer = (
            grouped.get("results", {})
            .get(GROUP_COLUMN, {})
            .get("kmer_logreg", {})
            .get(INPUT_COLUMN)
        )
    finetune_seed_mean = None
    if finetune:
        finetune_seed_mean = finetune.get("comparison", {}).get("seed_mean")

    return {
        "grouped_kmer_pair_text": grouped_kmer
        or {"roc_auc": KMER_GROUPED_ROC_AUC, "average_precision": KMER_GROUPED_PR_AUC},
        "frozen_pretrained_pair_mlp": {
            "roc_auc": FROZEN_PAIR_MLP_GROUPED_ROC_AUC,
            "average_precision": FROZEN_PAIR_MLP_GROUPED_PR_AUC,
        },
        "finetuned_igbert_last_1_layer_seed_mean": finetune_seed_mean,
    }


def build_top_records(table: pd.DataFrame) -> pd.DataFrame:
    """Return top existing records by probability within each target region."""
    target = normalized_text(optional_column(table, TARGET_REGION_COLUMN)).replace(
        {"": "unknown_target_region"}
    )
    temp = table.copy()
    temp["_target_region_for_sort"] = target
    temp = temp.sort_values(
        ["_target_region_for_sort", "predicted_neutralisation_probability"],
        ascending=[True, False],
    )
    top = temp.groupby("_target_region_for_sort", sort=True).head(TOP_RECORDS_PER_TARGET)
    keep = [
        "_target_region_for_sort",
        "sample_name",
        "label",
        "predicted_neutralisation_probability",
        "confidence_bin",
        "record_category",
        "developability_risk_bin",
        "priority_category",
        "diversity_group",
        "has_structure",
    ]
    top = top[[column for column in keep if column in top.columns]].copy()
    top = top.rename(columns={"_target_region_for_sort": "metadata_target_region"})
    top.insert(
        1,
        "target_region_rank",
        top.groupby("metadata_target_region")[
            "predicted_neutralisation_probability"
        ].rank(method="first", ascending=False).astype(int),
    )
    return top


def summarize_table(
    table: pd.DataFrame,
    model_info: dict[str, Any],
    input_path: Path,
    strict_scored_count: int | None = None,
    artifacts: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build summary metrics for the prioritization output."""
    probability = table["predicted_neutralisation_probability"]
    label_numeric = pd.to_numeric(table[LABEL_COLUMN], errors="coerce")
    metrics = model_metrics_summary()
    high_score_high_confidence = table["predicted_neutralisation_probability"].ge(
        0.75
    ) & table["confidence_bin"].eq("high")
    if artifacts is None:
        artifacts = {
            "table_csv": str(OUTPUT_TABLE_PATH.relative_to(PROJECT_ROOT)),
            "report": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
            "summary_json": str(SUMMARY_PATH.relative_to(PROJECT_ROOT)),
            "probability_distribution": str(
                PROBABILITY_DISTRIBUTION_PATH.relative_to(PROJECT_ROOT)
            ),
            "developability_risk_counts": str(RISK_COUNTS_PATH.relative_to(PROJECT_ROOT)),
            "priority_category_counts": str(
                PRIORITY_COUNTS_PATH.relative_to(PROJECT_ROOT)
            ),
            "probability_vs_developability_risk": str(
                PROBABILITY_RISK_PATH.relative_to(PROJECT_ROOT)
            ),
        }
    return {
        "status": "available",
        "input_path": str(input_path.relative_to(PROJECT_ROOT)),
        "strict_labeled_scored_record_count": strict_scored_count,
        "scored_record_count": int(len(table)),
        "labeled_record_count": int(label_numeric.notna().sum()),
        "unlabeled_record_count": int(label_numeric.isna().sum()),
        "model": model_info,
        "baseline_context": metrics,
        "probability_summary": {
            "min": float(probability.min()),
            "mean": float(probability.mean()),
            "median": float(probability.median()),
            "max": float(probability.max()),
        },
        "predicted_label_counts": value_counts_dict(table["predicted_label"]),
        "confidence_bin_counts": value_counts_dict(table["confidence_bin"]),
        "record_category_counts": value_counts_dict(table["record_category"]),
        "priority_category_counts": value_counts_dict(table["priority_category"]),
        "developability_risk_counts": value_counts_dict(table["developability_risk_bin"]),
        "developability_risk_score_counts": value_counts_dict(
            table["developability_risk_score"]
        ),
        "high_score_high_confidence_counts": {
            "missing_label": int(
                (high_score_high_confidence & table["record_category"].eq("missing_label")).sum()
            ),
            "conflict_label": int(
                (high_score_high_confidence & table["record_category"].eq("conflict_label")).sum()
            ),
            "known_neutralising": int(
                (
                    high_score_high_confidence
                    & table["record_category"].eq("known_neutralising")
                ).sum()
            ),
            "known_non_neutralising": int(
                (
                    high_score_high_confidence
                    & table["record_category"].eq("known_non_neutralising")
                ).sum()
            ),
        },
        "target_region_counts": value_counts_dict(
            optional_column(table, TARGET_REGION_COLUMN).replace({"": "missing"})
        ),
        "target_region_group_counts": value_counts_dict(
            optional_column(table, "target_region_group").replace({"": "missing"})
        ),
        "diversity_group_count": int(table["diversity_group"].nunique()),
        "has_structure_count": int(table["has_structure"].sum()),
        "missing_light_count": int(table["missing_light_flag"].sum()),
        "paired_light_status_counts": value_counts_dict(table["paired_light_status"]),
        "nanobody_like_count": int(table["nanobody_like_flag"].sum()),
        "artifacts": artifacts,
    }


def save_probability_distribution(table: pd.DataFrame) -> None:
    """Save predicted probability distribution."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = pd.to_numeric(table[LABEL_COLUMN], errors="coerce")
    for label, color in [(0, "#4C78A8"), (1, "#F58518")]:
        values = table.loc[labels.eq(label), "predicted_neutralisation_probability"]
        if not values.empty:
            ax.hist(
                values,
                bins=30,
                alpha=0.65,
                label=f"label {label}",
                color=color,
                edgecolor="white",
            )
    if labels.isna().any():
        ax.hist(
            table.loc[labels.isna(), "predicted_neutralisation_probability"],
            bins=30,
            alpha=0.65,
            label="label missing",
            color="#BAB0AC",
            edgecolor="white",
        )
    ax.axvline(0.75, color="#E45756", linestyle="--", linewidth=1.5, label="0.75")
    ax.set_xlabel("Predicted neutralisation probability")
    ax.set_ylabel("Existing record count")
    ax.set_title("K-mer score distribution for existing records")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PROBABILITY_DISTRIBUTION_PATH, dpi=200)
    plt.close(fig)


def save_count_figure(
    counts: pd.Series,
    output_path: Path,
    title: str,
    xlabel: str,
    color: str,
) -> None:
    """Save a horizontal count bar chart."""
    counts = counts.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7, max(3.6, 0.35 * len(counts))))
    ax.barh(np.arange(len(counts)), counts.values, color=color)
    ax.set_yticks(np.arange(len(counts)))
    ax.set_yticklabels(counts.index.astype(str))
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    for index, value in enumerate(counts.values):
        ax.text(value + max(counts.max() * 0.01, 0.5), index, str(int(value)), va="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_probability_vs_risk(table: pd.DataFrame, output_path: Path = PROBABILITY_RISK_PATH) -> None:
    """Save probability versus risk score scatter."""
    fig, ax = plt.subplots(figsize=(7, 4.8))
    rng = np.random.default_rng(RANDOM_STATE)
    x = table["developability_risk_score"].astype(float).to_numpy()
    x = x + rng.uniform(-0.08, 0.08, size=len(table))
    colors = table["priority_category"].astype("category").cat.codes
    scatter = ax.scatter(
        x,
        table["predicted_neutralisation_probability"],
        c=colors,
        cmap="tab10",
        s=14,
        alpha=0.45,
        linewidths=0,
    )
    ax.axhline(0.75, color="#E45756", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Developability risk score")
    ax.set_ylabel("Predicted neutralisation probability")
    ax.set_title("Existing-record score versus sequence-heuristic risk")
    handles, _ = scatter.legend_elements(num=None)
    labels = list(table["priority_category"].astype("category").cat.categories)
    if handles and labels:
        ax.legend(handles, labels, title="Priority category", fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_probability_by_record_category(table: pd.DataFrame) -> None:
    """Save probability distributions by record category for the broader table."""
    categories = [
        "known_neutralising",
        "known_non_neutralising",
        "missing_label",
        "conflict_label",
        "other",
    ]
    values = [
        table.loc[
            table["record_category"].eq(category),
            "predicted_neutralisation_probability",
        ].to_numpy()
        for category in categories
        if table["record_category"].eq(category).any()
    ]
    labels = [
        category
        for category in categories
        if table["record_category"].eq(category).any()
    ]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.boxplot(values, tick_labels=labels, showfliers=False, patch_artist=True)
    ax.axhline(0.75, color="#E45756", linestyle="--", linewidth=1.5)
    ax.set_ylabel("Predicted neutralisation probability")
    ax.set_title("Broader existing records by label/conflict category")
    ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    fig.savefig(BROADER_PROBABILITY_BY_RECORD_CATEGORY_PATH, dpi=200)
    plt.close(fig)


def save_figures(table: pd.DataFrame) -> None:
    """Save all requested prioritization figures."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    save_probability_distribution(table)
    save_count_figure(
        table["developability_risk_bin"].value_counts(),
        RISK_COUNTS_PATH,
        "Developability risk bins",
        "Existing record count",
        "#4C78A8",
    )
    save_count_figure(
        table["priority_category"].value_counts(),
        PRIORITY_COUNTS_PATH,
        "Priority categories",
        "Existing record count",
        "#54A24B",
    )
    save_probability_vs_risk(table)


def save_broader_figures(table: pd.DataFrame) -> None:
    """Save all requested broader prioritization figures."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    save_count_figure(
        table["priority_category"].value_counts(),
        BROADER_PRIORITY_COUNTS_PATH,
        "Broader priority categories",
        "Existing record count",
        "#54A24B",
    )
    save_count_figure(
        table["developability_risk_bin"].value_counts(),
        BROADER_RISK_COUNTS_PATH,
        "Broader developability risk bins",
        "Existing record count",
        "#4C78A8",
    )
    save_probability_by_record_category(table)
    save_probability_vs_risk(table, BROADER_PROBABILITY_RISK_PATH)


def format_metric(value: Any) -> str:
    """Format numeric metrics compactly."""
    if value is None:
        return "n/a"
    try:
        if pd.isna(value):
            return "n/a"
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def format_counts_table(counts: dict[str, int], name: str) -> list[str]:
    """Format a JSON count dictionary as Markdown."""
    lines = [f"| {name} | Count |", "|---|---:|"]
    for key, value in counts.items():
        lines.append(f"| {key} | {value} |")
    return lines


def top_records_table(top_records: pd.DataFrame) -> list[str]:
    """Format top existing records by target region."""
    lines = [
        (
            "| Target region | Rank | Record | Label | Record category | Probability | "
            "Confidence | Risk | Priority category | Structure |"
        ),
        "|---|---:|---|---:|---|---:|---|---|---|---:|",
    ]
    for _, row in top_records.iterrows():
        label_value = row.get("label", "")
        label_text = "missing" if pd.isna(label_value) else str(label_value)
        lines.append(
            f"| {row.get('metadata_target_region', '')} | "
            f"{int(row.get('target_region_rank', 0))} | "
            f"{row.get('sample_name', '')} | {label_text} | "
            f"{row.get('record_category', '')} | "
            f"{float(row.get('predicted_neutralisation_probability', np.nan)):.4f} | "
            f"{row.get('confidence_bin', '')} | "
            f"{row.get('developability_risk_bin', '')} | "
            f"{row.get('priority_category', '')} | "
            f"{str(row.get('has_structure', '')).lower()} |"
        )
    return lines


def build_report(
    summary: dict[str, Any],
    top_records: pd.DataFrame,
) -> str:
    """Build Markdown prioritization report."""
    baseline = summary["baseline_context"]
    kmer = baseline.get("grouped_kmer_pair_text") or {}
    frozen = baseline.get("frozen_pretrained_pair_mlp") or {}
    finetuned = baseline.get("finetuned_igbert_last_1_layer_seed_mean") or {}
    lines = [
        "# Existing-Record Prioritization",
        "",
        "This analysis scores and annotates existing public dataset records only.",
        "It does not generate, alter, mutate, optimize, rank newly designed",
        "sequences, or propose sequence changes.",
        "",
        "## Model Context",
        "",
        "| Model | Grouped ROC-AUC | Grouped PR-AUC | Role |",
        "|---|---:|---:|---|",
        (
            f"| k-mer TF-IDF + logistic regression | "
            f"{format_metric(kmer.get('roc_auc'))} | "
            f"{format_metric(kmer.get('average_precision'))} | "
            "main scoring model |"
        ),
        (
            f"| frozen pretrained pair MLP | {FROZEN_PAIR_MLP_GROUPED_ROC_AUC:.4f} | "
            f"{FROZEN_PAIR_MLP_GROUPED_PR_AUC:.4f} | benchmark comparison |"
        ),
        (
            f"| IgBert last_1_layer seed mean | "
            f"{format_metric(finetuned.get('roc_auc'))} | "
            f"{format_metric(finetuned.get('pr_auc'))} | benchmark comparison |"
        ),
        "",
        f"Scoring model source: `{summary['model']['source']}`.",
    ]
    if summary["model"].get("path"):
        lines.append(f"Loaded model path: `{summary['model']['path']}`.")
    if summary["model"].get("fallback_used"):
        lines.append(
            f"Fallback retraining rows: {summary['model'].get('training_rows')}."
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Scored records | {summary['scored_record_count']} |",
            f"| Labeled records | {summary['labeled_record_count']} |",
            f"| Unlabeled records | {summary['unlabeled_record_count']} |",
            f"| Diversity groups | {summary['diversity_group_count']} |",
            f"| Records with structure | {summary['has_structure_count']} |",
            f"| Missing-light records | {summary['missing_light_count']} |",
            f"| Nanobody-like records | {summary['nanobody_like_count']} |",
            f"| Mean predicted probability | {summary['probability_summary']['mean']:.4f} |",
            f"| Median predicted probability | {summary['probability_summary']['median']:.4f} |",
            "",
            "## Priority Category Counts",
            "",
        ]
    )
    lines.extend(format_counts_table(summary["priority_category_counts"], "Priority category"))
    lines.extend(["", "## Target Region Group Counts", ""])
    lines.extend(format_counts_table(summary["target_region_group_counts"], "Target region group"))
    lines.extend(["", "## Paired/Light-Missing Counts", ""])
    lines.extend(format_counts_table(summary["paired_light_status_counts"], "Status"))
    lines.extend(["", "## Developability Risk Counts", ""])
    lines.extend(format_counts_table(summary["developability_risk_counts"], "Risk bin"))
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Model score is not therapeutic efficacy.",
            "- Sequence-risk flags are heuristic.",
            "- Labels are heterogeneous literature-derived labels.",
            "- This is retrospective scoring of existing records only.",
            "- No new sequences are generated, altered, proposed, or optimized.",
            "",
            "## Artifacts",
            "",
            f"- `{OUTPUT_TABLE_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{SUMMARY_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{PROBABILITY_DISTRIBUTION_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{RISK_COUNTS_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{PRIORITY_COUNTS_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{PROBABILITY_RISK_PATH.relative_to(PROJECT_ROOT)}`",
            "",
        ]
    )
    return "\n".join(lines)


def build_broader_report(
    summary: dict[str, Any],
    top_records: pd.DataFrame,
) -> str:
    """Build Markdown report for the broader cleaned existing-record table."""
    baseline = summary["baseline_context"]
    kmer = baseline.get("grouped_kmer_pair_text") or {}
    finetuned = baseline.get("finetuned_igbert_last_1_layer_seed_mean") or {}
    high_counts = summary["high_score_high_confidence_counts"]
    lines = [
        "# Broader Existing-Record Prioritization",
        "",
        "This analysis scores and annotates broader cleaned public dataset records",
        "with usable existing heavy sequences. It does not create, alter, mutate,",
        "optimize, or propose new sequences.",
        "",
        "## Model Context",
        "",
        "| Model | Grouped ROC-AUC | Grouped PR-AUC | Role |",
        "|---|---:|---:|---|",
        (
            f"| k-mer TF-IDF + logistic regression | "
            f"{format_metric(kmer.get('roc_auc'))} | "
            f"{format_metric(kmer.get('average_precision'))} | "
            "main scoring model |"
        ),
        (
            f"| IgBert last_1_layer seed mean | "
            f"{format_metric(finetuned.get('roc_auc'))} | "
            f"{format_metric(finetuned.get('pr_auc'))} | "
            "benchmark comparison |"
        ),
        "",
        f"Scoring model source: `{summary['model']['source']}`.",
    ]
    if summary["model"].get("path"):
        lines.append(f"Loaded model path: `{summary['model']['path']}`.")

    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Strict labeled records scored | {summary['strict_labeled_scored_record_count']} |",
            f"| Broader records scored | {summary['scored_record_count']} |",
            f"| Labeled broader records | {summary['labeled_record_count']} |",
            f"| Missing-label broader records | {summary['unlabeled_record_count']} |",
            f"| Diversity groups | {summary['diversity_group_count']} |",
            f"| Records with structure | {summary['has_structure_count']} |",
            f"| Missing-light records | {summary['missing_light_count']} |",
            f"| Nanobody-like records | {summary['nanobody_like_count']} |",
            f"| Mean predicted probability | {summary['probability_summary']['mean']:.4f} |",
            f"| Median predicted probability | {summary['probability_summary']['median']:.4f} |",
            f"| Missing-label high-score/high-confidence records | {high_counts['missing_label']} |",
            f"| Conflict-label high-score/high-confidence records | {high_counts['conflict_label']} |",
            "",
            "## Record Category Counts",
            "",
        ]
    )
    lines.extend(format_counts_table(summary["record_category_counts"], "Record category"))
    lines.extend(["", "## Priority Category Counts", ""])
    lines.extend(format_counts_table(summary["priority_category_counts"], "Priority category"))
    lines.extend(["", "## Target Region Group Counts", ""])
    lines.extend(format_counts_table(summary["target_region_group_counts"], "Target region group"))
    lines.extend(["", "## Paired/Light-Missing Counts", ""])
    lines.extend(format_counts_table(summary["paired_light_status_counts"], "Status"))
    lines.extend(["", "## Developability Risk Counts", ""])
    lines.extend(format_counts_table(summary["developability_risk_counts"], "Risk bin"))
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Model score is not therapeutic efficacy.",
            "- Sequence-risk flags are heuristic.",
            "- Labels are heterogeneous literature-derived labels.",
            "- This is retrospective scoring of existing records only.",
            "- No new sequences are generated, altered, proposed, or optimized.",
            "",
            "## Artifacts",
            "",
            f"- `{BROADER_OUTPUT_TABLE_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{BROADER_SUMMARY_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{BROADER_PRIORITY_COUNTS_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{BROADER_RISK_COUNTS_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{BROADER_PROBABILITY_BY_RECORD_CATEGORY_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{BROADER_PROBABILITY_RISK_PATH.relative_to(PROJECT_ROOT)}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    """Build and save existing-record prioritization artifacts."""
    OUTPUT_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BROADER_OUTPUT_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    BROADER_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    BROADER_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    strict_raw = read_csv(INPUT_PATH)
    broader_raw = read_csv(BROADER_INPUT_PATH)

    strict_for_model = ensure_scoring_columns(
        strict_raw.copy(),
        dataset_name="strict_labeled_ml",
        require_usable_heavy=False,
    )
    model, model_info = load_or_train_kmer_model(strict_for_model)

    strict_table = build_prioritization_table(strict_raw, model)
    broader_table = build_broader_prioritization_table(broader_raw, model)
    strict_top_records = build_top_records(strict_table)
    broader_top_records = build_top_records(broader_table)
    strict_summary = summarize_table(
        strict_table,
        model_info,
        input_path=INPUT_PATH,
        strict_scored_count=int(len(strict_table)),
    )
    broader_summary = summarize_table(
        broader_table,
        model_info,
        input_path=BROADER_INPUT_PATH,
        strict_scored_count=int(len(strict_table)),
        artifacts={
            "table_csv": str(BROADER_OUTPUT_TABLE_PATH.relative_to(PROJECT_ROOT)),
            "report": str(BROADER_REPORT_PATH.relative_to(PROJECT_ROOT)),
            "summary_json": str(BROADER_SUMMARY_PATH.relative_to(PROJECT_ROOT)),
            "priority_category_counts": str(
                BROADER_PRIORITY_COUNTS_PATH.relative_to(PROJECT_ROOT)
            ),
            "developability_risk_counts": str(
                BROADER_RISK_COUNTS_PATH.relative_to(PROJECT_ROOT)
            ),
            "probability_by_record_category": str(
                BROADER_PROBABILITY_BY_RECORD_CATEGORY_PATH.relative_to(PROJECT_ROOT)
            ),
            "probability_vs_risk_score": str(
                BROADER_PROBABILITY_RISK_PATH.relative_to(PROJECT_ROOT)
            ),
        },
    )

    strict_table.to_csv(OUTPUT_TABLE_PATH, index=False)
    broader_table.to_csv(BROADER_OUTPUT_TABLE_PATH, index=False)
    save_figures(strict_table)
    save_broader_figures(broader_table)
    SUMMARY_PATH.write_text(
        json.dumps(strict_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    BROADER_SUMMARY_PATH.write_text(
        json.dumps(broader_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        build_report(strict_summary, strict_top_records),
        encoding="utf-8",
    )
    BROADER_REPORT_PATH.write_text(
        build_broader_report(broader_summary, broader_top_records),
        encoding="utf-8",
    )

    print(
        "Existing-record prioritization complete: "
        f"strict_rows={len(strict_table)}, "
        f"broader_rows={len(broader_table)}, "
        f"model_source={model_info['source']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
