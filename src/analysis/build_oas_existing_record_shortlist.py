"""Build a public-safe OAS existing-record retrieval shortlist.

This module ranks already-existing OAS paired records by similarity to curated
project-positive records. It does not generate, mutate, optimize, or propose
new biological sequences. Public artifacts contain hashes, row indices, scores,
aggregate review flags, and metadata only.

Run from the project root:

    python src/analysis/build_oas_existing_record_shortlist.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

MPLCONFIGDIR = Path("/tmp") / "antibody_prioritization_mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from _safe_analysis_utils import PROJECT_ROOT, relpath, write_json, write_text


OAS_PATH = PROJECT_ROOT / "data" / "processed" / "oas" / "oas_paired_standardized.csv"
PROJECT_PREPARED_PATH = PROJECT_ROOT / "data" / "processed" / "neutral_prepared_sequences.csv"
PROJECT_ML_PATH = PROJECT_ROOT / "data" / "processed" / "neutral_sequence_classification_ml.csv"
BROAD_RETRIEVAL_SCORES_PATH = PROJECT_ROOT / "reports" / "oas_background_retrieval_scores.csv"
MATCHED_RETRIEVAL_SCORES_PATH = (
    PROJECT_ROOT / "reports" / "oas_matched_background_retrieval_scores.csv"
)

REPORT_PATH = PROJECT_ROOT / "reports" / "oas_existing_record_shortlist_report.md"
METRICS_PATH = (
    PROJECT_ROOT / "reports" / "metrics" / "oas_existing_record_shortlist_metrics.json"
)
TOP25_PATH = PROJECT_ROOT / "reports" / "oas_existing_record_shortlist_top25.csv"
TOP100_PATH = PROJECT_ROOT / "reports" / "oas_existing_record_shortlist_top100.csv"
FULL_SCORES_PATH = PROJECT_ROOT / "reports" / "oas_existing_record_scores_public.csv"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
SCORE_DISTRIBUTION_PATH = FIGURE_DIR / "oas_existing_record_score_distribution.png"
SIMILARITY_VS_SCORE_PATH = FIGURE_DIR / "oas_existing_record_similarity_vs_score.png"
DIVERSITY_MAP_PATH = FIGURE_DIR / "oas_existing_record_diversity_map.png"

RANDOM_STATE = 42
MAX_FEATURES = 50_000
NEIGHBOR_CHUNK_SIZE = 512
TOP_SHORTLIST_SIZE = 25
TOP_RANKED_SIZE = 100
DIVERSITY_SIMILARITY_THRESHOLD = 0.95
LABEL_CANDIDATES = ["label", "target", "neutralising_label", "is_positive"]
MISSING_TEXT_VALUES = {"", "na", "nan", "none", "null", "nd", "n/a", "unknown"}
HYDROPHOBIC_RESIDUES = set("AVILMFWY")
FORBIDDEN_PUBLIC_COLUMNS = {
    "sequence",
    "heavy_sequence",
    "light_sequence",
    "sequence_pair_text",
    "vhorvhh",
    "vl",
}

BASE_COMPOSITE_WEIGHTS = {
    "retrieval_score": 0.40,
    "max_positive_neighbor_similarity": 0.30,
    "top10_positive_neighbor_similarity": 0.20,
    "positive_centroid_similarity": 0.10,
}


def read_csv_text(path: Path) -> pd.DataFrame:
    """Read CSV as strings while preserving blanks."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {relpath(path)}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def normalize_sequence(value: Any) -> str:
    """Normalize an existing sequence-like field for internal feature use only."""
    text = re.sub(r"\s+", "", str(value or "")).upper()
    return "" if text.lower() in MISSING_TEXT_VALUES else text


def compact_existing_pair_text(value: Any) -> str:
    """Compact an existing sequence-pair field for character k-mer features."""
    text = str(value or "").replace("[SEP]", "|")
    return normalize_sequence(text)


def compact_pair_from_values(heavy: Any, light: Any = "") -> str:
    """Build compact heavy/light text internally without saving it."""
    heavy_text = normalize_sequence(heavy)
    light_text = normalize_sequence(light)
    return f"{heavy_text}|{light_text}" if light_text else heavy_text


def sha256_text(value: str) -> str:
    """Return a stable SHA256 digest for redacted public IDs."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def optional_column(data: pd.DataFrame, column: str) -> pd.Series:
    """Return an existing column or aligned blanks."""
    if column in data.columns:
        return data[column]
    return pd.Series([""] * len(data), index=data.index)


def detect_label_column(data: pd.DataFrame) -> str | None:
    """Return the first strict positive-label column present in the input."""
    for column in LABEL_CANDIDATES:
        if column in data.columns:
            return column
    return None


def numeric_label_values(values: pd.Series) -> pd.Series:
    """Convert label-like values to numeric labels."""
    text = values.fillna("").astype(str).str.strip()
    mapped = text.str.lower().map(
        {
            "positive": 1,
            "pos": 1,
            "true": 1,
            "yes": 1,
            "neutralising": 1,
            "neutralizing": 1,
            "negative": 0,
            "neg": 0,
            "false": 0,
            "no": 0,
            "non-neutralising": 0,
            "non-neutralizing": 0,
        }
    )
    numeric = pd.to_numeric(text.replace({"": np.nan}), errors="coerce")
    return numeric.where(numeric.notna(), mapped)


def project_pair_text(data: pd.DataFrame) -> pd.Series:
    """Build compact project pair text from an existing pair field or chain fields."""
    if "sequence_pair_text" in data.columns:
        pair_text = data["sequence_pair_text"].map(compact_existing_pair_text)
        if pair_text.str.len().gt(0).any():
            return pair_text
    heavy = optional_column(data, "sequence_a")
    if heavy.str.len().eq(0).all():
        heavy = optional_column(data, "heavy_sequence")
    light = optional_column(data, "sequence_b")
    if light.str.len().eq(0).all():
        light = optional_column(data, "light_sequence")
    return pd.Series(
        [compact_pair_from_values(h, l) for h, l in zip(heavy, light)],
        index=data.index,
    )


def load_project_positive_records() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load strict project-positive reference records."""
    searched_paths = [PROJECT_ML_PATH, PROJECT_PREPARED_PATH]
    load_attempts: list[dict[str, Any]] = []
    for path in searched_paths:
        if not path.exists():
            load_attempts.append({"path": relpath(path), "status": "missing"})
            continue
        data = read_csv_text(path)
        label_column = detect_label_column(data)
        if label_column is None:
            load_attempts.append(
                {
                    "path": relpath(path),
                    "status": "no_strict_label_column",
                    "columns": list(data.columns),
                }
            )
            continue
        labels = numeric_label_values(data[label_column])
        positive_mask = labels.eq(1)
        positives = data.loc[positive_mask].copy()
        positives["compact_pair_text"] = project_pair_text(positives)
        positives = positives[positives["compact_pair_text"].str.len().gt(0)].copy()
        positives["project_positive_hash"] = positives["compact_pair_text"].map(sha256_text)
        positive_unique = positives.drop_duplicates("project_positive_hash").reset_index(
            drop=True
        )
        if positive_unique.empty:
            load_attempts.append(
                {
                    "path": relpath(path),
                    "status": "no_usable_positive_pair_text",
                    "label_column": label_column,
                }
            )
            continue
        diagnostics = {
            "path": relpath(path),
            "status": "available",
            "label_column": label_column,
            "input_rows": int(len(data)),
            "strict_positive_rows": int(positive_mask.sum()),
            "usable_positive_reference_rows": int(len(positives)),
            "unique_positive_reference_texts": int(len(positive_unique)),
            "columns": list(data.columns),
            "load_attempts": load_attempts,
        }
        return positive_unique[["compact_pair_text", "project_positive_hash"]], diagnostics

    raise ValueError(
        "No project-positive reference set could be built from strict label columns."
    )


def load_oas_records() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load standardized OAS paired records and internal compact text."""
    data = read_csv_text(OAS_PATH)
    required = {"sequence_pair_text", "source_file"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"OAS standardized input missing required columns: {missing}")

    output = pd.DataFrame(
        {
            "source_file": optional_column(data, "source_file"),
            "background_source": optional_column(data, "background_source"),
            "source_row_index": np.arange(len(data), dtype=int),
            "compact_pair_text": data["sequence_pair_text"].map(compact_existing_pair_text),
            "compact_pair_text_sep_hash": data["sequence_pair_text"]
            .map(lambda value: re.sub(r"\s+", "", str(value or "")).upper())
            .map(sha256_text),
            "heavy_internal": optional_column(data, "heavy_sequence").map(normalize_sequence),
            "light_internal": optional_column(data, "light_sequence").map(normalize_sequence),
        }
    )
    blank_text = output["compact_pair_text"].str.len().eq(0)
    output = output.loc[~blank_text].copy()
    output["oas_record_hash"] = output["compact_pair_text"].map(sha256_text)
    diagnostics = {
        "path": relpath(OAS_PATH),
        "input_rows": int(len(data)),
        "usable_oas_rows": int(len(output)),
        "blank_pair_text_rows_dropped": int(blank_text.sum()),
        "columns": list(data.columns),
        "source_file_count": int(output["source_file"].nunique()),
    }
    return output.reset_index(drop=True), diagnostics


def build_vectorizer(
    positive_texts: pd.Series,
    oas_texts: pd.Series,
) -> tuple[TfidfVectorizer, sparse.csr_matrix, sparse.csr_matrix]:
    """Fit TF-IDF on project-positive plus OAS compact texts."""
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=MAX_FEATURES,
    )
    combined_texts = pd.concat([positive_texts, oas_texts], ignore_index=True)
    combined_vectors = vectorizer.fit_transform(combined_texts)
    positive_count = len(positive_texts)
    positive_vectors = combined_vectors[:positive_count].tocsr()
    oas_vectors = combined_vectors[positive_count:].tocsr()
    return vectorizer, positive_vectors, oas_vectors


def compute_neighbor_metrics(
    oas_vectors: sparse.csr_matrix,
    positive_vectors: sparse.csr_matrix,
    positive_hashes: pd.Series,
) -> pd.DataFrame:
    """Compute nearest project-positive cosine-similarity summaries for OAS rows."""
    oas_count = oas_vectors.shape[0]
    positive_count = positive_vectors.shape[0]
    top_k = min(10, positive_count)
    top5_k = min(5, positive_count)
    max_similarity = np.zeros(oas_count, dtype=float)
    top5_similarity = np.zeros(oas_count, dtype=float)
    top10_similarity = np.zeros(oas_count, dtype=float)
    nearest_positive_hash: list[str] = [""] * oas_count
    hash_values = positive_hashes.to_numpy(dtype=object)

    for start in range(0, oas_count, NEIGHBOR_CHUNK_SIZE):
        stop = min(start + NEIGHBOR_CHUNK_SIZE, oas_count)
        similarity = linear_kernel(
            oas_vectors[start:stop],
            positive_vectors,
            dense_output=True,
        )
        partition = np.argpartition(
            similarity,
            kth=positive_count - top_k,
            axis=1,
        )[:, positive_count - top_k :]
        top_values = np.take_along_axis(similarity, partition, axis=1)
        order = np.argsort(top_values, axis=1)[:, ::-1]
        sorted_values = np.take_along_axis(top_values, order, axis=1)
        sorted_indices = np.take_along_axis(partition, order, axis=1)
        chunk_rows = stop - start
        max_similarity[start:stop] = sorted_values[:, 0]
        top5_similarity[start:stop] = sorted_values[:, :top5_k].mean(axis=1)
        top10_similarity[start:stop] = sorted_values.mean(axis=1)
        nearest = sorted_indices[:, 0]
        for offset in range(chunk_rows):
            nearest_positive_hash[start + offset] = str(hash_values[nearest[offset]])

    return pd.DataFrame(
        {
            "max_positive_neighbor_similarity": max_similarity,
            "top5_positive_neighbor_similarity": top5_similarity,
            "top10_positive_neighbor_similarity": top10_similarity,
            "nearest_project_positive_hash": nearest_positive_hash,
        }
    )


def compute_centroid_similarity(
    oas_vectors: sparse.csr_matrix,
    positive_vectors: sparse.csr_matrix,
) -> np.ndarray:
    """Compute cosine similarity to the project-positive TF-IDF centroid."""
    centroid = positive_vectors.mean(axis=0)
    centroid = sparse.csr_matrix(centroid)
    centroid = normalize(centroid)
    return (oas_vectors @ centroid.T).toarray().ravel().astype(float)


def aggregate_existing_score_file(
    path: Path,
    hash_columns: list[str],
    score_columns: list[str],
) -> pd.DataFrame | None:
    """Read a saved retrieval score file and aggregate compatible hash scores."""
    if not path.exists():
        return None
    columns = list(pd.read_csv(path, nrows=0).columns)
    hash_column = next((column for column in hash_columns if column in columns), None)
    score_column = next((column for column in score_columns if column in columns), None)
    if hash_column is None or score_column is None:
        return None
    scores = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        usecols=[hash_column, score_column],
    )
    scores[score_column] = pd.to_numeric(scores[score_column], errors="coerce")
    scores = scores.dropna(subset=[score_column])
    if scores.empty:
        return None
    grouped = (
        scores.groupby(hash_column, as_index=False)[score_column]
        .mean()
        .rename(columns={hash_column: "score_hash", score_column: "saved_score"})
    )
    grouped["score_file"] = relpath(path)
    return grouped


def merge_saved_retrieval_scores(oas: pd.DataFrame) -> pd.DataFrame:
    """Merge compatible saved OAS retrieval scores by public or legacy hash."""
    lookup = oas[
        ["oas_record_hash", "compact_pair_text_sep_hash"]
    ].drop_duplicates()
    lookup_public = lookup[["oas_record_hash"]].rename(columns={"oas_record_hash": "score_hash"})
    lookup_public["oas_record_hash"] = lookup_public["score_hash"]
    lookup_sep = lookup[["compact_pair_text_sep_hash", "oas_record_hash"]].rename(
        columns={"compact_pair_text_sep_hash": "score_hash"}
    )
    score_parts = []
    for path in [BROAD_RETRIEVAL_SCORES_PATH, MATCHED_RETRIEVAL_SCORES_PATH]:
        part = aggregate_existing_score_file(
            path,
            hash_columns=["sequence_pair_hash", "hashed_sequence_key", "oas_record_hash"],
            score_columns=[
                "retrieval_score",
                "project_retrieval_probability",
                "oas_project_like_score",
            ],
        )
        if part is None:
            continue
        mapped = part.merge(lookup_public, on="score_hash", how="inner")
        mapped_alt = part.merge(lookup_sep, on="score_hash", how="inner")
        mapped = pd.concat([mapped, mapped_alt], ignore_index=True)
        if not mapped.empty:
            score_parts.append(mapped[["oas_record_hash", "saved_score", "score_file"]])
    if not score_parts:
        return pd.DataFrame(
            columns=["oas_record_hash", "saved_retrieval_score", "saved_score_file_count"]
        )
    saved = pd.concat(score_parts, ignore_index=True).drop_duplicates()
    aggregated = (
        saved.groupby("oas_record_hash")
        .agg(
            saved_retrieval_score=("saved_score", "mean"),
            saved_score_file_count=("score_file", "nunique"),
        )
        .reset_index()
    )
    return aggregated


def train_internal_retrieval_score(
    positive_vectors: sparse.csr_matrix,
    oas_vectors: sparse.csr_matrix,
) -> np.ndarray:
    """Train a project-vs-OAS retrieval scorer for ranking only."""
    x = sparse.vstack([positive_vectors, oas_vectors], format="csr")
    y = np.concatenate(
        [
            np.ones(positive_vectors.shape[0], dtype=int),
            np.zeros(oas_vectors.shape[0], dtype=int),
        ]
    )
    model = LogisticRegression(max_iter=3000, class_weight="balanced")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        with threadpool_limits(limits=1):
            model.fit(x, y)
    positive_index = list(model.classes_).index(1)
    return model.predict_proba(oas_vectors)[:, positive_index].astype(float)


def minmax_normalized(values: pd.Series) -> pd.Series:
    """Normalize a numeric score column to [0, 1] when possible."""
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        return pd.Series([np.nan] * len(numeric), index=numeric.index, dtype=float)
    min_value = float(finite.min())
    max_value = float(finite.max())
    if max_value == min_value:
        return numeric.map(lambda value: 0.5 if np.isfinite(value) else np.nan)
    return (numeric - min_value) / (max_value - min_value)


def add_composite_score(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Add weighted project-like composite score with weight renormalization."""
    output = data.copy()
    available_weights: dict[str, float] = {}
    normalized_components: dict[str, pd.Series] = {}
    for column, weight in BASE_COMPOSITE_WEIGHTS.items():
        normalized = minmax_normalized(output[column])
        if normalized.notna().any():
            normalized_components[column] = normalized
            available_weights[column] = weight

    weight_sum = sum(available_weights.values())
    if weight_sum <= 0:
        raise ValueError("No score components were available for composite scoring.")

    score = pd.Series(np.zeros(len(output), dtype=float), index=output.index)
    normalized_weights = {
        column: weight / weight_sum for column, weight in available_weights.items()
    }
    for column, weight in normalized_weights.items():
        score += normalized_components[column].fillna(0.0) * weight
    output["oas_project_like_score"] = score.astype(float)
    return output, normalized_weights


def n_glycosylation_motif_count(sequence: str) -> int:
    """Count N-X-S/T motifs where X is not P."""
    count = 0
    for index in range(max(0, len(sequence) - 2)):
        first, second, third = sequence[index : index + 3]
        if first == "N" and second != "P" and third in {"S", "T"}:
            count += 1
    return count


def hydrophobic_fraction(sequence: str) -> float:
    """Return hydrophobic residue fraction for aggregate review flags."""
    if not sequence:
        return float("nan")
    return float(sum(1 for residue in sequence if residue in HYDROPHOBIC_RESIDUES) / len(sequence))


def add_review_flags(data: pd.DataFrame) -> pd.DataFrame:
    """Append simple aggregate sequence-risk proxy flags without exposing sequences."""
    output = data.copy()
    heavy = output["heavy_internal"]
    light = output["light_internal"]
    combined = heavy + light
    output["heavy_length"] = heavy.map(len).astype(int)
    output["light_length"] = light.map(len).astype(int)
    output["total_length"] = output["heavy_length"] + output["light_length"]
    output["unusual_length_flag"] = (
        output["heavy_length"].lt(105)
        | output["heavy_length"].gt(140)
        | output["light_length"].lt(85)
        | output["light_length"].gt(130)
    )
    heavy_cysteines = heavy.map(lambda value: value.count("C")).astype(int)
    light_cysteines = light.map(lambda value: value.count("C")).astype(int)
    output["cysteine_count_flag"] = (
        heavy_cysteines.gt(4)
        | light_cysteines.gt(4)
        | (heavy_cysteines + light_cysteines).gt(10)
    )
    output["glycosylation_motif_count"] = combined.map(n_glycosylation_motif_count).astype(int)
    output["hydrophobic_fraction"] = combined.map(hydrophobic_fraction).astype(float)
    output["hydrophobic_fraction_flag"] = output["hydrophobic_fraction"].fillna(0.0).gt(0.45)
    duplicate_hash = output.groupby("oas_record_hash")["oas_record_hash"].transform("size").gt(1)
    output["exact_duplicate_hash_flag"] = duplicate_hash.astype(bool)
    return output


def select_diverse_shortlist(
    scored: pd.DataFrame,
    oas_vectors: sparse.csr_matrix,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Greedily select top diverse OAS records using cosine similarity."""
    output = scored.copy()
    order = np.lexsort(
        (
            output["source_row_index"].to_numpy(dtype=int),
            -output["oas_project_like_score"].to_numpy(dtype=float),
        )
    )
    selected_positions: list[int] = []
    selected_clusters: list[str] = []
    cluster_values = np.array(["outside_top25_similarity_clusters"] * len(output), dtype=object)
    selected_flags = np.zeros(len(output), dtype=bool)
    max_similarity_to_selected = np.zeros(len(output), dtype=float)

    for position in order:
        if selected_positions:
            similarities = linear_kernel(
                oas_vectors[position],
                oas_vectors[selected_positions],
                dense_output=True,
            ).ravel()
            nearest_selected = int(np.argmax(similarities))
            max_similarity = float(similarities[nearest_selected])
            max_similarity_to_selected[position] = max_similarity
            if max_similarity > DIVERSITY_SIMILARITY_THRESHOLD:
                cluster_values[position] = selected_clusters[nearest_selected]
                continue
        if len(selected_positions) < TOP_SHORTLIST_SIZE:
            cluster = f"cluster_{len(selected_positions) + 1:03d}"
            selected_positions.append(int(position))
            selected_clusters.append(cluster)
            selected_flags[position] = True
            cluster_values[position] = cluster
        elif selected_positions:
            similarities = linear_kernel(
                oas_vectors[position],
                oas_vectors[selected_positions],
                dense_output=True,
            ).ravel()
            nearest_selected = int(np.argmax(similarities))
            max_similarity = float(similarities[nearest_selected])
            max_similarity_to_selected[position] = max_similarity
            if max_similarity > DIVERSITY_SIMILARITY_THRESHOLD:
                cluster_values[position] = selected_clusters[nearest_selected]

    output["diversity_cluster"] = cluster_values
    output["selected_diverse_shortlist"] = selected_flags
    output["max_similarity_to_diverse_shortlist"] = max_similarity_to_selected
    output["duplicate_or_near_duplicate_flag"] = (
        output["exact_duplicate_hash_flag"]
        | (
            output["max_similarity_to_diverse_shortlist"].gt(DIVERSITY_SIMILARITY_THRESHOLD)
            & ~output["selected_diverse_shortlist"]
        )
    )
    output["risk_flag_count"] = output[
        [
            "unusual_length_flag",
            "cysteine_count_flag",
            "hydrophobic_fraction_flag",
            "duplicate_or_near_duplicate_flag",
        ]
    ].sum(axis=1).astype(int)
    output["review_notes"] = np.select(
        [
            output["selected_diverse_shortlist"],
            output["duplicate_or_near_duplicate_flag"],
            output["risk_flag_count"].gt(0),
        ],
        [
            "diverse existing-record shortlist for expert review",
            "similar to a higher-ranked shortlisted OAS record",
            "aggregate review flag present",
        ],
        default="ranked existing OAS background record",
    )
    summary = {
        "selected_diverse_shortlist_size": int(output["selected_diverse_shortlist"].sum()),
        "diversity_cluster_count": int(len(selected_clusters)),
        "near_duplicate_to_shortlist_count": int(
            (
                output["max_similarity_to_diverse_shortlist"].gt(
                    DIVERSITY_SIMILARITY_THRESHOLD
                )
                & ~output["selected_diverse_shortlist"]
            ).sum()
        ),
        "exact_duplicate_hash_count": int(output["exact_duplicate_hash_flag"].sum()),
        "diversity_similarity_threshold": DIVERSITY_SIMILARITY_THRESHOLD,
    }
    return output, summary


def public_output_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Return public-safe columns in a stable order."""
    columns = [
        "rank",
        "oas_record_hash",
        "source_file",
        "source_row_index",
        "background_source",
        "oas_project_like_score",
        "retrieval_score",
        "saved_retrieval_score_available",
        "max_positive_neighbor_similarity",
        "top5_positive_neighbor_similarity",
        "top10_positive_neighbor_similarity",
        "nearest_project_positive_hash",
        "positive_centroid_similarity",
        "diversity_cluster",
        "selected_diverse_shortlist",
        "heavy_length",
        "light_length",
        "total_length",
        "unusual_length_flag",
        "cysteine_count_flag",
        "glycosylation_motif_count",
        "hydrophobic_fraction",
        "hydrophobic_fraction_flag",
        "duplicate_or_near_duplicate_flag",
        "risk_flag_count",
        "review_notes",
    ]
    available = [column for column in columns if column in data.columns]
    public = data[available].copy()
    normalized_columns = {column.strip().lower() for column in public.columns}
    forbidden = sorted(normalized_columns & FORBIDDEN_PUBLIC_COLUMNS)
    if forbidden:
        raise ValueError(f"Public output contains forbidden columns: {forbidden}")
    return public


def save_score_distribution(table: pd.DataFrame) -> None:
    """Save score distribution figure."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        table["oas_project_like_score"],
        bins=40,
        color="#4C78A8",
        edgecolor="white",
        alpha=0.85,
    )
    ax.set_title("OAS existing-record project-like score distribution")
    ax.set_xlabel("Computational prioritization score")
    ax.set_ylabel("Existing OAS record count")
    fig.tight_layout()
    SCORE_DISTRIBUTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SCORE_DISTRIBUTION_PATH, dpi=200)
    plt.close(fig)


def save_similarity_vs_score(table: pd.DataFrame) -> None:
    """Save nearest-neighbor similarity versus composite score."""
    fig, ax = plt.subplots(figsize=(8, 5))
    selected = table["selected_diverse_shortlist"].astype(bool)
    ax.scatter(
        table.loc[~selected, "max_positive_neighbor_similarity"],
        table.loc[~selected, "oas_project_like_score"],
        s=10,
        alpha=0.30,
        color="#4C78A8",
        linewidths=0,
        label="OAS background",
    )
    ax.scatter(
        table.loc[selected, "max_positive_neighbor_similarity"],
        table.loc[selected, "oas_project_like_score"],
        s=42,
        alpha=0.90,
        color="#E45756",
        linewidths=0.4,
        edgecolors="white",
        label="top 25 diverse",
    )
    ax.set_title("OAS similarity to project-positive records")
    ax.set_xlabel("Nearest project-positive cosine similarity")
    ax.set_ylabel("Computational prioritization score")
    ax.legend(frameon=False)
    fig.tight_layout()
    SIMILARITY_VS_SCORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SIMILARITY_VS_SCORE_PATH, dpi=200)
    plt.close(fig)


def save_diversity_map(table: pd.DataFrame, oas_vectors: sparse.csr_matrix) -> None:
    """Save a 2D reduced-feature map for aggregate diversity inspection."""
    reducer = TruncatedSVD(n_components=2, random_state=RANDOM_STATE)
    coordinates = reducer.fit_transform(oas_vectors)
    selected = table["selected_diverse_shortlist"].astype(bool).to_numpy()
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=table["oas_project_like_score"],
        cmap="viridis",
        s=9,
        alpha=0.45,
        linewidths=0,
    )
    if selected.any():
        ax.scatter(
            coordinates[selected, 0],
            coordinates[selected, 1],
            s=48,
            color="#E45756",
            edgecolors="white",
            linewidths=0.5,
            label="top 25 diverse",
        )
        ax.legend(frameon=False)
    ax.set_title("OAS existing-record diversity map")
    ax.set_xlabel("Reduced TF-IDF component 1")
    ax.set_ylabel("Reduced TF-IDF component 2")
    fig.colorbar(scatter, ax=ax, label="Computational prioritization score")
    fig.tight_layout()
    DIVERSITY_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(DIVERSITY_MAP_PATH, dpi=200)
    plt.close(fig)


def build_report(summary: dict[str, Any]) -> str:
    """Build public-safe Markdown report."""
    artifacts = summary["artifacts"]
    score_summary = summary["score_summary"]
    return "\n".join(
        [
            "# OAS Existing-Record Retrieval Shortlist",
            "",
            (
                "This shortlist contains existing OAS records that are sequence-similar "
                "to curated project-positive records."
            ),
            (
                "OAS records are unknown-target background and, more specifically, "
                "unknown-target natural antibody background."
            ),
            (
                "The output is an existing-record shortlist for expert review. The score "
                "is a computational prioritization score, not a binding probability."
            ),
            (
                "The records are not validated binders or therapeutics, and this module "
                "does not generate or modify sequences."
            ),
            (
                "Any downstream use requires independent expert review and appropriate "
                "experimental validation outside this repository."
            ),
            "",
            "## Inputs",
            "",
            f"- OAS standardized records: `{summary['input_paths']['oas_records']}`",
            f"- Project-positive records: `{summary['input_paths']['project_positive_records']}`",
            "",
            "## Method",
            "",
            (
                "Project-positive and OAS compact pair texts were used internally for "
                "hashing and k-mer TF-IDF features. Raw sequence strings were not saved "
                "to public outputs."
            ),
            (
                "Ranking combined retrieval-model score, maximum nearest-neighbor "
                "similarity, top-10 neighbor similarity, and centroid similarity."
            ),
            (
                "A greedy diversity filter selected records while avoiding OAS records "
                f"with cosine similarity greater than {DIVERSITY_SIMILARITY_THRESHOLD:.2f} "
                "to already selected shortlist records."
            ),
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| OAS rows scored | {summary['oas_rows_scored']} |",
            (
                "| Project-positive reference rows | "
                f"{summary['project_positive_reference_rows']} |"
            ),
            f"| Unique project-positive reference texts | {summary['unique_positive_reference_texts']} |",
            f"| Top 25 shortlist size | {summary['top25_shortlist_size']} |",
            f"| Top 100 table size | {summary['top100_table_size']} |",
            f"| Diversity clusters | {summary['diversity_cluster_count']} |",
            f"| Minimum score | {score_summary['min']:.6f} |",
            f"| Maximum score | {score_summary['max']:.6f} |",
            f"| Saved retrieval score matches | {summary['saved_retrieval_score_match_count']} |",
            "",
            "## Composite Score Weights",
            "",
            "| Component | Normalized weight |",
            "|---|---:|",
            *[
                f"| {component} | {weight:.3f} |"
                for component, weight in summary["composite_weights"].items()
            ],
            "",
            "## Review Flags",
            "",
            (
                "Length, cysteine-count, glycosylation-motif, hydrophobic-fraction, "
                "and duplicate or near-duplicate flags are heuristic review flags only. "
                "They are not validation results."
            ),
            "",
            "## Artifacts",
            "",
            f"- `{artifacts['report']}`",
            f"- `{artifacts['metrics_json']}`",
            f"- `{artifacts['top25_csv']}`",
            f"- `{artifacts['top100_csv']}`",
            f"- `{artifacts['full_scores_csv']}`",
            f"- `{artifacts['score_distribution_figure']}`",
            f"- `{artifacts['similarity_vs_score_figure']}`",
            f"- `{artifacts['diversity_map_figure']}`",
            "",
            "## Limitations",
            "",
            "- OAS records are unknown-target natural antibody background records.",
            "- Similarity to project-positive records does not establish binding or neutralisation.",
            "- The retrieval score is for computational prioritization and expert review only.",
            "- This module does not generate, mutate, design, optimize, or propose sequences.",
            "- Wet-lab protocols and therapeutic claims are outside the scope of this repository.",
            "",
        ]
    )


def build_summary(
    oas_diagnostics: dict[str, Any],
    project_diagnostics: dict[str, Any],
    vectorizer: TfidfVectorizer,
    scored: pd.DataFrame,
    diversity_summary: dict[str, Any],
    composite_weights: dict[str, float],
    saved_score_match_count: int,
) -> dict[str, Any]:
    """Build machine-readable metrics for the shortlist."""
    score = scored["oas_project_like_score"]
    top25_count = int(scored["selected_diverse_shortlist"].sum())
    top100_count = int(min(TOP_RANKED_SIZE, len(scored)))
    return {
        "status": "available",
        "module": "OAS existing-record retrieval shortlist",
        "interpretation": {
            "background_semantics": "OAS records are unknown-target natural antibody background.",
            "output_semantics": "existing-record shortlist for expert review",
            "score_semantics": "computational prioritization score, not a binding probability",
            "sequence_generation": False,
            "therapeutic_efficacy_claim": False,
        },
        "input_paths": {
            "oas_records": relpath(OAS_PATH),
            "project_positive_records": project_diagnostics["path"],
            "broad_retrieval_scores": relpath(BROAD_RETRIEVAL_SCORES_PATH),
            "matched_retrieval_scores": relpath(MATCHED_RETRIEVAL_SCORES_PATH),
        },
        "input_diagnostics": {
            "oas": oas_diagnostics,
            "project_positive": project_diagnostics,
        },
        "oas_rows_scored": int(len(scored)),
        "project_positive_reference_rows": int(
            project_diagnostics["usable_positive_reference_rows"]
        ),
        "unique_positive_reference_texts": int(
            project_diagnostics["unique_positive_reference_texts"]
        ),
        "top25_shortlist_size": top25_count,
        "top100_table_size": top100_count,
        "diversity_cluster_count": int(diversity_summary["diversity_cluster_count"]),
        "diversity_summary": diversity_summary,
        "saved_retrieval_score_match_count": int(saved_score_match_count),
        "retrieval_score_strategy": (
            "internal logistic-regression retrieval score for all OAS rows; compatible "
            "saved OAS retrieval scores merged and averaged where available"
        ),
        "feature_model": {
            "vectorizer": (
                'TfidfVectorizer(analyzer="char", ngram_range=(3,5), '
                f"min_df=2, max_features={MAX_FEATURES})"
            ),
            "vocabulary_size": int(len(vectorizer.vocabulary_)),
            "internal_retrieval_classifier": (
                'LogisticRegression(max_iter=3000, class_weight="balanced")'
            ),
        },
        "composite_weights": composite_weights,
        "score_summary": {
            "min": float(score.min()),
            "mean": float(score.mean()),
            "median": float(score.median()),
            "max": float(score.max()),
        },
        "risk_flag_counts": {
            "unusual_length_flag": int(scored["unusual_length_flag"].sum()),
            "cysteine_count_flag": int(scored["cysteine_count_flag"].sum()),
            "hydrophobic_fraction_flag": int(scored["hydrophobic_fraction_flag"].sum()),
            "duplicate_or_near_duplicate_flag": int(
                scored["duplicate_or_near_duplicate_flag"].sum()
            ),
        },
        "public_output_forbidden_columns": sorted(FORBIDDEN_PUBLIC_COLUMNS),
        "artifacts": {
            "report": relpath(REPORT_PATH),
            "metrics_json": relpath(METRICS_PATH),
            "top25_csv": relpath(TOP25_PATH),
            "top100_csv": relpath(TOP100_PATH),
            "full_scores_csv": relpath(FULL_SCORES_PATH),
            "score_distribution_figure": relpath(SCORE_DISTRIBUTION_PATH),
            "similarity_vs_score_figure": relpath(SIMILARITY_VS_SCORE_PATH),
            "diversity_map_figure": relpath(DIVERSITY_MAP_PATH),
        },
    }


def build_shortlist() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build all shortlist tables and metrics."""
    oas, oas_diagnostics = load_oas_records()
    positives, project_diagnostics = load_project_positive_records()
    vectorizer, positive_vectors, oas_vectors = build_vectorizer(
        positives["compact_pair_text"],
        oas["compact_pair_text"],
    )
    neighbor_metrics = compute_neighbor_metrics(
        oas_vectors,
        positive_vectors,
        positives["project_positive_hash"],
    )
    scored = pd.concat([oas.reset_index(drop=True), neighbor_metrics], axis=1)
    scored["positive_centroid_similarity"] = compute_centroid_similarity(
        oas_vectors,
        positive_vectors,
    )

    saved_scores = merge_saved_retrieval_scores(scored)
    scored = scored.merge(saved_scores, on="oas_record_hash", how="left")
    internal_score = train_internal_retrieval_score(positive_vectors, oas_vectors)
    scored["internal_retrieval_score"] = internal_score
    scored["saved_retrieval_score_available"] = scored["saved_retrieval_score"].notna()
    scored["retrieval_score"] = scored[
        ["internal_retrieval_score", "saved_retrieval_score"]
    ].mean(axis=1, skipna=True)

    scored = add_review_flags(scored)
    scored, composite_weights = add_composite_score(scored)
    scored, diversity_summary = select_diverse_shortlist(scored, oas_vectors)
    scored = scored.sort_values(
        ["oas_project_like_score", "source_row_index"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    scored.insert(0, "rank", np.arange(1, len(scored) + 1, dtype=int))

    public_full = public_output_columns(scored)
    top25 = public_full.loc[public_full["selected_diverse_shortlist"].astype(bool)].copy()
    top25 = top25.sort_values("rank", kind="mergesort").reset_index(drop=True)
    top100 = public_full.iloc[: min(TOP_RANKED_SIZE, len(public_full))].copy()

    summary = build_summary(
        oas_diagnostics=oas_diagnostics,
        project_diagnostics=project_diagnostics,
        vectorizer=vectorizer,
        scored=scored,
        diversity_summary=diversity_summary,
        composite_weights=composite_weights,
        saved_score_match_count=int(scored["saved_retrieval_score_available"].sum()),
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    save_score_distribution(scored)
    save_similarity_vs_score(scored)
    save_diversity_map(scored, oas_vectors)
    return public_full, top25, top100, summary


def main() -> int:
    """Build and save public-safe OAS existing-record shortlist artifacts."""
    public_full, top25, top100, summary = build_shortlist()

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOP25_PATH.parent.mkdir(parents=True, exist_ok=True)

    public_full.to_csv(FULL_SCORES_PATH, index=False)
    top25.to_csv(TOP25_PATH, index=False)
    top100.to_csv(TOP100_PATH, index=False)
    write_json(METRICS_PATH, summary)
    write_text(REPORT_PATH, build_report(summary))

    score_summary = summary["score_summary"]
    output_paths = [
        summary["artifacts"]["report"],
        summary["artifacts"]["metrics_json"],
        summary["artifacts"]["top25_csv"],
        summary["artifacts"]["top100_csv"],
        summary["artifacts"]["full_scores_csv"],
        summary["artifacts"]["score_distribution_figure"],
        summary["artifacts"]["similarity_vs_score_figure"],
        summary["artifacts"]["diversity_map_figure"],
    ]
    print(f"OAS rows scored: {summary['oas_rows_scored']}")
    print(f"project-positive reference rows: {summary['project_positive_reference_rows']}")
    print(f"top25 shortlist size: {summary['top25_shortlist_size']}")
    print(f"top100 table size: {summary['top100_table_size']}")
    print(f"number of diversity clusters: {summary['diversity_cluster_count']}")
    print(f"score range: {score_summary['min']:.6f} to {score_summary['max']:.6f}")
    print("output file paths:")
    for path in output_paths:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
