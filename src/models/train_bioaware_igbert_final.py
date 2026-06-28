"""Final bioaware IgBert benchmark with all-six-CDR annotation.

This script uses only existing rows and existing sequences from the strict
neutral ML dataset. It does not create, mutate, optimize, rank, or propose new
sequence records.

Run from the project root:

    python src/models/train_bioaware_igbert_final.py

Optionally choose a Hugging Face model:

    PRETRAINED_SEQUENCE_MODEL=Exscientia/IgBert \
        python src/models/train_bioaware_igbert_final.py
"""

from __future__ import annotations

import copy
import gc
import json
import os
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd


MPLCONFIGDIR = Path("/tmp") / "antibody_prioritization_mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import train_pretrained_finetune as ft
from src.models.embed_with_pretrained_sequence_model import (
    ensure_pad_token,
    last_hidden_state,
    max_length_from_model,
    normalize_sequence_text,
    read_input,
    run_model_forward,
)


INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "neutral_sequence_classification_ml.csv"
ANNOTATED_PATH = PROJECT_ROOT / "data" / "processed" / "bioaware_paired_cdr_annotated.csv"
KMER_MODEL_PATH = PROJECT_ROOT / "models" / "kmer_logreg_pair_text.joblib"

ANNOTATION_REPORT_PATH = PROJECT_ROOT / "reports" / "bioaware_cdr_annotation_report.md"
ANNOTATION_SUMMARY_PATH = (
    PROJECT_ROOT / "reports" / "metrics" / "bioaware_cdr_annotation_summary.json"
)
REPORT_PATH = PROJECT_ROOT / "reports" / "bioaware_igbert_final_report.md"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics" / "bioaware_igbert_final_metrics.json"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
SEED_SUMMARY_FIGURE_PATH = FIGURE_DIR / "bioaware_igbert_final_seed_summary.png"
TRAINING_CURVE_PATH = FIGURE_DIR / "bioaware_igbert_final_training_curves.png"
SUBGROUP_FIGURE_PATH = FIGURE_DIR / "bioaware_igbert_final_subgroup_metrics.png"
CDR_COVERAGE_FIGURE_PATH = FIGURE_DIR / "bioaware_cdr_coverage.png"
MODEL_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODEL_DIR / "bioaware_igbert_final_best.pt"

DEFAULT_MODEL_NAME = "Exscientia/IgBert"
MODEL_ENV_VAR = "PRETRAINED_SEQUENCE_MODEL"
MODEL_NAME = os.environ.get(MODEL_ENV_VAR, DEFAULT_MODEL_NAME)

RANDOM_STATE = 42
TEST_SIZE = 0.2
INNER_VALIDATION_SIZE = 0.15
MAX_GROUP_SPLIT_ATTEMPTS = 50
GROUP_COLUMN = "group_feature_v"

TRAINING_SEEDS = [1, 7, 42]
STD_DDOF = 1
BATCH_SIZE = int(os.environ.get("BIOAWARE_IGBERT_BATCH_SIZE", "8"))
MAX_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 4
MIN_DELTA = 1e-4
MAX_GRAD_NORM = 1.0
WARMUP_RATIO = 0.10

HEAD_HIDDEN_SIZE = 256
HEAD_DROPOUT = 0.5
HEAD_LEARNING_RATE = 5e-4
LORA_LEARNING_RATE = 5e-5
MARKER_EMBEDDING_LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05

LORA_R = 4
LORA_ALPHA = 8
LORA_DROPOUT = 0.20

LABEL_SMOOTHING_POSITIVE = 0.95
LABEL_SMOOTHING_NEGATIVE = 0.05

FULL_DATASET_KMER_GROUPED_ROC_AUC = 0.7810
FULL_DATASET_KMER_GROUPED_PR_AUC = 0.8236
PREVIOUS_DIRECT_FINETUNE_ROC_AUC = 0.7443
PREVIOUS_DIRECT_FINETUNE_PR_AUC = 0.8151
DIRECT_FINETUNE_METRICS_PATH = (
    PROJECT_ROOT / "reports" / "metrics" / "pretrained_finetune_seed_check_metrics.json"
)

CDR_MARKER_TOKENS = [
    "<CDRH1_START>",
    "<CDRH1_END>",
    "<CDRH2_START>",
    "<CDRH2_END>",
    "<CDRH3_START>",
    "<CDRH3_END>",
    "<CDRL1_START>",
    "<CDRL1_END>",
    "<CDRL2_START>",
    "<CDRL2_END>",
    "<CDRL3_START>",
    "<CDRL3_END>",
]
REGION_START_TOKENS = {
    "cdrh1": "<CDRH1_START>",
    "cdrh2": "<CDRH2_START>",
    "cdrh3": "<CDRH3_START>",
    "cdrl1": "<CDRL1_START>",
    "cdrl2": "<CDRL2_START>",
    "cdrl3": "<CDRL3_START>",
}
REGION_END_TOKENS = {
    "cdrh1": "<CDRH1_END>",
    "cdrh2": "<CDRH2_END>",
    "cdrh3": "<CDRH3_END>",
    "cdrl1": "<CDRL1_END>",
    "cdrl2": "<CDRL2_END>",
    "cdrl3": "<CDRL3_END>",
}
POOL_REGIONS = ["cdrh1", "cdrh2", "cdrh3", "cdrl1", "cdrl2", "cdrl3"]
MODEL_INPUT_KEYS = {"input_ids", "attention_mask", "token_type_ids", "position_ids"}
COMMON_LORA_TARGET_NAMES = [
    "query",
    "key",
    "value",
    "dense",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "out_proj",
]
NUMERIC_FEATURE_COLUMNS = [
    "heavy_length",
    "light_length",
    "cdrh1_length",
    "cdrh2_length",
    "cdrh3_length",
    "cdrl1_length",
    "cdrl2_length",
    "cdrl3_length",
    "hydrophobic_fraction_cdrh3",
    "hydrophobic_fraction_cdrl3",
    "cysteine_count_cdrh3",
    "cysteine_count_cdrl3",
    "n_glycosylation_motif_count_heavy",
]
HYDROPHOBIC_RESIDUES = set("AVILMFWY")
SUBGROUP_MIN_ROWS = 20


class BioawareDataset(Dataset):
    """Dataset for existing CDR-marked paired antibodies."""

    def __init__(self, rows: pd.DataFrame, numeric_features: np.ndarray) -> None:
        self.marked_heavy_text = rows["marked_heavy_text"].astype(str).tolist()
        self.marked_light_text = rows["marked_light_text"].astype(str).tolist()
        self.labels = rows["label"].to_numpy(dtype=np.float32)
        self.numeric_features = numeric_features.astype(np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "marked_heavy_text": self.marked_heavy_text[index],
            "marked_light_text": self.marked_light_text[index],
            "label": np.float32(self.labels[index]),
            "numeric_features": self.numeric_features[index],
        }


class BioawareIgBertClassifier(nn.Module):
    """IgBert token encoder with whole-pair and all-six-CDR pooling."""

    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        numeric_feature_count: int,
        include_cls: bool,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.include_cls = include_cls
        pooled_vector_count = 11 if include_cls else 10
        classifier_input_size = hidden_size * pooled_vector_count + numeric_feature_count
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_size),
            nn.Dropout(HEAD_DROPOUT),
            nn.Linear(classifier_input_size, HEAD_HIDDEN_SIZE),
            nn.GELU(),
            nn.Dropout(HEAD_DROPOUT),
            nn.Linear(HEAD_HIDDEN_SIZE, 1),
        )

    @staticmethod
    def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean-pool token states over true mask positions."""
        mask_f = mask.to(hidden.device).unsqueeze(-1).type_as(hidden)
        summed = (hidden * mask_f).sum(dim=1)
        counts = mask_f.sum(dim=1).clamp(min=1.0)
        return summed / counts

    @staticmethod
    def max_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Max-pool token states over true mask positions."""
        mask_b = mask.to(hidden.device).unsqueeze(-1).bool()
        counts = mask_b.sum(dim=1)
        masked = hidden.masked_fill(~mask_b, torch.finfo(hidden.dtype).min)
        pooled = masked.max(dim=1).values
        return torch.where(counts.gt(0), pooled, torch.zeros_like(pooled))

    def forward(
        self,
        encoded: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
        numeric_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return one logit per CDR-marked pair input."""
        model_inputs = {
            key: value for key, value in encoded.items() if key in MODEL_INPUT_KEYS
        }
        outputs = run_model_forward(self.backbone, model_inputs)
        hidden = last_hidden_state(outputs)

        pooled = []
        if self.include_cls:
            pooled.append(hidden[:, 0, :])
        pooled.append(self.mean_pool(hidden, masks["whole"]))
        pooled.append(self.max_pool(hidden, masks["whole"]))
        for region in POOL_REGIONS:
            pooled.append(self.mean_pool(hidden, masks[region]))
        pooled.append(self.max_pool(hidden, masks["cdrh3"]))
        pooled.append(self.max_pool(hidden, masks["cdrl3"]))
        combined = torch.cat([*pooled, numeric_features.to(hidden.device)], dim=1)
        return self.classifier(combined).squeeze(-1)


def set_seed(seed: int) -> None:
    """Set deterministic seeds for numpy, Python, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_batch_size() -> None:
    """Keep the requested batch size in the small-GPU range."""
    if BATCH_SIZE not in {8, 16}:
        raise ValueError("BIOAWARE_IGBERT_BATCH_SIZE must be 8 or 16.")


def normalize_bool(value: Any) -> bool:
    """Parse common boolean encodings."""
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return False


def optional_column(data: pd.DataFrame, candidates: list[str]) -> pd.Series:
    """Return the first available text column or aligned blanks."""
    for column in candidates:
        if column in data.columns:
            return data[column]
    return pd.Series([""] * len(data), index=data.index)


def optional_column_name(data: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first available column name from candidates."""
    for column in candidates:
        if column in data.columns:
            return column
    return None


def load_abnumber_chain() -> tuple[Any | None, dict[str, Any]]:
    """Import AbNumber's Chain class, returning clear dependency status."""
    try:
        from abnumber import Chain
    except Exception as exc:
        return None, {
            "available": False,
            "dependency": "abnumber",
            "error_type": exc.__class__.__name__,
            "error": str(exc),
            "message": (
                "AbNumber is required for all-six-CDR annotation. Model training "
                "was not run because the final benchmark must not fall back to "
                "weak CDRH3/CDRL3-only heuristics."
            ),
        }
    hmmscan_path = shutil.which("hmmscan")
    if hmmscan_path is None:
        return None, {
            "available": False,
            "dependency": "abnumber+hmmer",
            "error_type": "MissingExecutable",
            "error": "hmmscan was not found on PATH",
            "message": (
                "AbNumber imported, but ANARCI numbering requires the HMMER "
                "`hmmscan` executable on PATH. Model training was not run "
                "because all-six-CDR annotation is required for this benchmark."
            ),
        }
    return Chain, {
        "available": True,
        "dependency": "abnumber",
        "hmmscan_path": hmmscan_path,
        "error": None,
    }


def make_kmer_pipeline() -> Pipeline:
    """Construct the controlled k-mer TF-IDF logistic-regression baseline."""
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2)),
            ("classifier", LogisticRegression(max_iter=5000, class_weight="balanced")),
        ]
    )


def positive_scores(model: Any, values: pd.Series) -> np.ndarray:
    """Return positive-class probabilities from a fitted classifier."""
    probabilities = model.predict_proba(values.astype(str))
    classes = list(getattr(model, "classes_", [0, 1]))
    positive_index = classes.index(1) if 1 in classes else 1
    return probabilities[:, positive_index].astype(np.float32)


def prepare_raw_table() -> pd.DataFrame:
    """Normalize strict input columns for annotation and filtering only."""
    raw = read_input(INPUT_PATH)
    heavy = optional_column(raw, ["sequence_heavy_only", "sequence_a"]).map(
        normalize_sequence_text
    )
    light = optional_column(raw, ["sequence_b", "light_sequence", "sequence_light_only"]).map(
        normalize_sequence_text
    )
    if "sequence_pair_text" in raw.columns:
        pair = raw["sequence_pair_text"].fillna("").astype(str)
    else:
        pair = [
            f"{h}[SEP]{l}" if l else h
            for h, l in zip(heavy.astype(str), light.astype(str))
        ]

    if "has_light" in raw.columns:
        has_light = raw["has_light"].map(normalize_bool)
    else:
        has_light = light.ne("")

    sample_type = optional_column(raw, ["sample_type"]).astype(str).str.strip().str.upper()
    if "is_nanobody_like" in raw.columns:
        is_nanobody_like = raw["is_nanobody_like"].map(normalize_bool)
    else:
        is_nanobody_like = sample_type.isin({"NB", "NANOBODY"}) | ~has_light

    label = pd.to_numeric(optional_column(raw, ["label"]), errors="coerce")
    output = raw.copy()
    output["row_id"] = np.arange(len(output), dtype=int)
    output["heavy_sequence"] = heavy
    output["light_sequence"] = light
    output["sequence_pair_text"] = pd.Series(pair, index=raw.index).astype(str)
    output["label"] = label.astype("Int64")
    output["has_light_bool"] = has_light.astype(bool)
    output["is_nanobody_like_bool"] = is_nanobody_like.astype(bool)
    output["existing_cdrh3"] = optional_column(output, ["cdrh3", "group_feature_cdr3"]).map(
        normalize_sequence_text
    )
    output["existing_cdrl3"] = optional_column(output, ["cdrl3", "group_feature_b_cdr3"]).map(
        normalize_sequence_text
    )
    if GROUP_COLUMN not in output.columns:
        output[GROUP_COLUMN] = ""
    return output


def chain_cdrs(sequence: str, Chain: Any) -> dict[str, Any]:
    """Annotate one variable domain with AbNumber/IMGT and extract CDRs."""
    try:
        chain = Chain(sequence, scheme="imgt")
        return {
            "ok": True,
            "error": "",
            "chain_type": str(getattr(chain, "chain_type", "")),
            "cdr1": normalize_sequence_text(getattr(chain, "cdr1_seq", "")),
            "cdr2": normalize_sequence_text(getattr(chain, "cdr2_seq", "")),
            "cdr3": normalize_sequence_text(getattr(chain, "cdr3_seq", "")),
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{exc.__class__.__name__}: {exc}",
            "chain_type": "",
            "cdr1": "",
            "cdr2": "",
            "cdr3": "",
        }


def annotate_paired_rows(raw: pd.DataFrame, Chain: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run AbNumber annotation on paired antibody rows."""
    paired_mask = (
        raw["has_light_bool"]
        & ~raw["is_nanobody_like_bool"]
        & raw["heavy_sequence"].astype(str).str.len().gt(0)
        & raw["light_sequence"].astype(str).str.len().gt(0)
    )
    paired = raw.loc[paired_mask].copy().reset_index(drop=True)
    records = []
    for row in tqdm(paired.itertuples(index=False), total=len(paired), desc="Annotating CDRs"):
        heavy = chain_cdrs(str(row.heavy_sequence), Chain)
        light = chain_cdrs(str(row.light_sequence), Chain)
        records.append(
            {
                "heavy_annotation_ok": heavy["ok"],
                "light_annotation_ok": light["ok"],
                "heavy_annotation_error": heavy["error"],
                "light_annotation_error": light["error"],
                "heavy_chain_type": heavy["chain_type"],
                "light_chain_type": light["chain_type"],
                "cdrh1_seq": heavy["cdr1"],
                "cdrh2_seq": heavy["cdr2"],
                "cdrh3_seq": heavy["cdr3"],
                "cdrl1_seq": light["cdr1"],
                "cdrl2_seq": light["cdr2"],
                "cdrl3_seq": light["cdr3"],
            }
        )

    annotated = pd.concat([paired, pd.DataFrame.from_records(records)], axis=1)
    for column in [
        "cdrh1_seq",
        "cdrh2_seq",
        "cdrh3_seq",
        "cdrl1_seq",
        "cdrl2_seq",
        "cdrl3_seq",
    ]:
        annotated[f"{column[:-4]}_found"] = annotated[column].astype(str).str.len().gt(0)
    found_columns = [
        "cdrh1_found",
        "cdrh2_found",
        "cdrh3_found",
        "cdrl1_found",
        "cdrl2_found",
        "cdrl3_found",
    ]
    annotated["all_six_cdrs_found"] = annotated[found_columns].all(axis=1)

    summary = {
        "raw_row_count": int(len(raw)),
        "paired_annotation_candidate_count": int(len(annotated)),
        "nanobody_like_or_missing_light_count": int((~paired_mask).sum()),
        "heavy_annotation_ok_count": int(annotated["heavy_annotation_ok"].sum()),
        "light_annotation_ok_count": int(annotated["light_annotation_ok"].sum()),
        "all_six_cdrs_found_count": int(annotated["all_six_cdrs_found"].sum()),
        "all_six_cdrs_found_fraction": (
            float(annotated["all_six_cdrs_found"].mean()) if len(annotated) else 0.0
        ),
        "cdr_match_summary": {
            "cdrh3": compare_existing_cdr(annotated, "existing_cdrh3", "cdrh3_seq"),
            "cdrl3": compare_existing_cdr(annotated, "existing_cdrl3", "cdrl3_seq"),
        },
    }
    return annotated, summary


def compare_existing_cdr(
    data: pd.DataFrame,
    existing_column: str,
    annotated_column: str,
) -> dict[str, int]:
    """Compare AbNumber CDRs against existing CDR metadata."""
    existing = data[existing_column].fillna("").astype(str)
    annotated = data[annotated_column].fillna("").astype(str)
    comparable = existing.ne("") & annotated.ne("")
    return {
        "exact_match_count": int((existing.eq(annotated) & comparable).sum()),
        "mismatch_count": int((existing.ne(annotated) & comparable).sum()),
        "missing_count": int((~comparable).sum()),
        "existing_nonempty_count": int(existing.ne("").sum()),
        "annotated_nonempty_count": int(annotated.ne("").sum()),
    }


def fraction_of(sequence: str, residues: set[str]) -> float:
    """Compute residue fraction for a sequence."""
    if not sequence:
        return 0.0
    return float(sum(1 for residue in sequence if residue in residues) / len(sequence))


def n_glycosylation_motifs(sequence: str) -> int:
    """Count N-X-S/T motifs where X is not P."""
    return len(re.findall(r"N[^P][ST]", sequence))


def mark_chain_tokens(
    sequence: str,
    cdr_regions: list[tuple[str, str]],
) -> tuple[list[str], dict[str, Any]]:
    """Insert CDR marker tokens by ordered substring matching."""
    spans = []
    cursor = 0
    failures = []
    ambiguous = []
    for region, cdr in cdr_regions:
        if not cdr:
            failures.append(region)
            continue
        remaining = sequence[cursor:]
        count = remaining.count(cdr)
        if count == 0:
            failures.append(region)
            continue
        if count > 1:
            ambiguous.append(region)
        start = sequence.find(cdr, cursor)
        end = start + len(cdr)
        spans.append((start, end, region))
        cursor = end

    tokens: list[str] = []
    position = 0
    for start, end, region in spans:
        tokens.extend(list(sequence[position:start]))
        tokens.append(REGION_START_TOKENS[region])
        tokens.extend(list(sequence[start:end]))
        tokens.append(REGION_END_TOKENS[region])
        position = end
    tokens.extend(list(sequence[position:]))
    return tokens, {
        "ok": not failures,
        "failed_regions": failures,
        "ambiguous_regions": ambiguous,
        "method": "ordered_substring_matching_from_abnumber_cdr_sequences",
    }


def add_marked_inputs_and_features(annotated: pd.DataFrame) -> pd.DataFrame:
    """Create marked pair text, k-mer baseline text, and numeric features."""
    data = annotated.copy()
    heavy_statuses = []
    light_statuses = []
    marked_heavy = []
    marked_light = []
    for row in data.itertuples(index=False):
        heavy_tokens, heavy_status = mark_chain_tokens(
            str(row.heavy_sequence),
            [
                ("cdrh1", str(row.cdrh1_seq)),
                ("cdrh2", str(row.cdrh2_seq)),
                ("cdrh3", str(row.cdrh3_seq)),
            ],
        )
        light_tokens, light_status = mark_chain_tokens(
            str(row.light_sequence),
            [
                ("cdrl1", str(row.cdrl1_seq)),
                ("cdrl2", str(row.cdrl2_seq)),
                ("cdrl3", str(row.cdrl3_seq)),
            ],
        )
        heavy_statuses.append(heavy_status)
        light_statuses.append(light_status)
        marked_heavy.append(" ".join(heavy_tokens))
        marked_light.append(" ".join(light_tokens))

    data["marked_heavy_text"] = marked_heavy
    data["marked_light_text"] = marked_light
    data["heavy_marker_insertion_ok"] = [item["ok"] for item in heavy_statuses]
    data["light_marker_insertion_ok"] = [item["ok"] for item in light_statuses]
    data["heavy_marker_ambiguous_regions"] = [
        ",".join(item["ambiguous_regions"]) for item in heavy_statuses
    ]
    data["light_marker_ambiguous_regions"] = [
        ",".join(item["ambiguous_regions"]) for item in light_statuses
    ]
    data["marker_insertion_ok"] = (
        data["heavy_marker_insertion_ok"] & data["light_marker_insertion_ok"]
    )

    data["whole_pair_kmer_text"] = (
        data["heavy_sequence"].astype(str) + "[SEP]" + data["light_sequence"].astype(str)
    )
    data["all_cdr_kmer_text"] = (
        data["cdrh1_seq"].astype(str)
        + data["cdrh2_seq"].astype(str)
        + data["cdrh3_seq"].astype(str)
        + "[SEP]"
        + data["cdrl1_seq"].astype(str)
        + data["cdrl2_seq"].astype(str)
        + data["cdrl3_seq"].astype(str)
    )
    data["cdrh3_cdrl3_kmer_text"] = (
        data["cdrh3_seq"].astype(str) + "[SEP]" + data["cdrl3_seq"].astype(str)
    )

    data["heavy_length"] = data["heavy_sequence"].astype(str).str.len().astype(float)
    data["light_length"] = data["light_sequence"].astype(str).str.len().astype(float)
    for cdr in ["cdrh1", "cdrh2", "cdrh3", "cdrl1", "cdrl2", "cdrl3"]:
        data[f"{cdr}_length"] = data[f"{cdr}_seq"].astype(str).str.len().astype(float)
    data["hydrophobic_fraction_cdrh3"] = data["cdrh3_seq"].map(
        lambda value: fraction_of(str(value), HYDROPHOBIC_RESIDUES)
    )
    data["hydrophobic_fraction_cdrl3"] = data["cdrl3_seq"].map(
        lambda value: fraction_of(str(value), HYDROPHOBIC_RESIDUES)
    )
    data["cysteine_count_cdrh3"] = data["cdrh3_seq"].map(lambda value: str(value).count("C"))
    data["cysteine_count_cdrl3"] = data["cdrl3_seq"].map(lambda value: str(value).count("C"))
    data["n_glycosylation_motif_count_heavy"] = data["heavy_sequence"].map(
        lambda value: n_glycosylation_motifs(str(value))
    )
    return data


def primary_training_data(annotated: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Filter to the primary paired-antibody training dataset."""
    label_is_binary = annotated["label"].isin([0, 1])
    mask = (
        annotated["has_light_bool"].astype(bool)
        & ~annotated["is_nanobody_like_bool"].astype(bool)
        & label_is_binary
        & annotated["all_six_cdrs_found"].astype(bool)
        & annotated["marker_insertion_ok"].astype(bool)
        & annotated["heavy_sequence"].astype(str).str.len().gt(0)
        & annotated["light_sequence"].astype(str).str.len().gt(0)
    )
    primary = annotated.loc[mask].copy().reset_index(drop=True)
    primary["label"] = primary["label"].astype(int)
    summary = {
        "primary_row_count": int(len(primary)),
        "annotation_failure_count": int((~annotated["all_six_cdrs_found"]).sum()),
        "marker_insertion_failure_count": int((~annotated["marker_insertion_ok"]).sum()),
        "ambiguous_heavy_marker_count": int(
            annotated["heavy_marker_ambiguous_regions"].astype(str).str.len().gt(0).sum()
        ),
        "ambiguous_light_marker_count": int(
            annotated["light_marker_ambiguous_regions"].astype(str).str.len().gt(0).sum()
        ),
    }
    return primary, summary


def summarize_nanobody_rows(raw: pd.DataFrame) -> dict[str, Any]:
    """Summarize nanobody-like or light-missing rows excluded from paired training."""
    mask = raw["is_nanobody_like_bool"] | ~raw["has_light_bool"]
    subset = raw.loc[mask].copy()
    summary: dict[str, Any] = {
        "count": int(len(subset)),
        "label_counts": ft.label_counts(subset["label"].dropna().astype(int).to_numpy())
        if len(subset["label"].dropna())
        else {"0": 0, "1": 0},
        "kmer_score_available": False,
    }
    if KMER_MODEL_PATH.exists() and len(subset):
        try:
            model = joblib.load(KMER_MODEL_PATH)
            scores = positive_scores(model, subset["sequence_pair_text"].astype(str))
            summary.update(
                {
                    "kmer_score_available": True,
                    "kmer_score_mean": float(np.mean(scores)),
                    "kmer_score_median": float(np.median(scores)),
                    "kmer_score_min": float(np.min(scores)),
                    "kmer_score_max": float(np.max(scores)),
                }
            )
        except Exception as exc:
            summary["kmer_score_error"] = str(exc)
    return summary


def group_column_status(values: pd.Series) -> dict[str, Any]:
    """Decide whether a grouping series supports meaningful grouped validation."""
    row_count = int(len(values))
    text = values.fillna("").astype(str).str.strip()
    non_missing = text[text.ne("")]
    missing_count = int(row_count - len(non_missing))
    non_missing_count = int(len(non_missing))
    unique_non_missing_count = int(non_missing.nunique(dropna=True))
    if row_count == 0 or non_missing_count == 0:
        return {
            "useful_for_grouping": False,
            "reason": "empty",
            "missing_count": missing_count,
            "non_missing_count": non_missing_count,
            "unique_non_missing_count": unique_non_missing_count,
        }
    missing_ratio = missing_count / row_count
    unique_row_ratio = unique_non_missing_count / row_count
    unique_non_missing_ratio = unique_non_missing_count / non_missing_count
    has_repeated_values = unique_non_missing_count < non_missing_count
    if missing_ratio >= 0.50:
        reason = "mostly_empty"
        useful = False
    elif not has_repeated_values:
        reason = "no_repeated_values"
        useful = False
    elif unique_row_ratio >= 0.80 or unique_non_missing_ratio >= 0.90:
        reason = "near_row_unique"
        useful = False
    else:
        reason = "ok"
        useful = True
    return {
        "useful_for_grouping": useful,
        "reason": reason,
        "missing_count": missing_count,
        "non_missing_count": non_missing_count,
        "unique_non_missing_count": unique_non_missing_count,
    }


def cdrh3_length_bin(length: int) -> str:
    """Return a coarse CDRH3 length bin."""
    if length <= 10:
        return "short"
    if length <= 20:
        return "medium"
    return "long"


def add_clonotype_like_group(data: pd.DataFrame) -> pd.Series:
    """Create a clonotype-like leakage diagnostic group."""
    cdrh3 = data["cdrh3_seq"].fillna("").astype(str)
    length_bins = cdrh3.map(lambda value: cdrh3_length_bin(len(value)))
    return (
        data[GROUP_COLUMN].fillna("").astype(str)
        + "|"
        + length_bins.astype(str)
        + "|"
        + cdrh3.map(lambda value: value[:5])
        + "|"
        + cdrh3.map(lambda value: value[-5:])
    )


def split_group_values(data: pd.DataFrame) -> pd.Series:
    """Return group_feature_v values without row-level fallback groups."""
    groups = data[GROUP_COLUMN].fillna("").astype(str).str.strip().copy()
    groups.loc[groups.eq("")] = "__missing_group__"
    return groups


def grouped_train_test_split(data: pd.DataFrame) -> dict[str, Any]:
    """Create the controlled group_feature_v train/test split."""
    status = group_column_status(data[GROUP_COLUMN])
    if not status["useful_for_grouping"]:
        raise ValueError(f"group_feature_v is not useful for grouping: {status['reason']}")
    groups = split_group_values(data)
    last_error = "no_valid_grouped_split"
    for offset in range(MAX_GROUP_SPLIT_ATTEMPTS):
        split_random_state = RANDOM_STATE + offset
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=TEST_SIZE,
            random_state=split_random_state,
        )
        try:
            train_idx, test_idx = next(splitter.split(data, data["label"], groups=groups))
        except ValueError as exc:
            last_error = str(exc)
            break
        train_labels = data.iloc[train_idx]["label"]
        test_labels = data.iloc[test_idx]["label"]
        if train_labels.nunique() != 2:
            last_error = "train_split_single_label"
            continue
        if test_labels.nunique() != 2:
            last_error = "test_split_single_label"
            continue
        if set(groups.iloc[train_idx]) & set(groups.iloc[test_idx]):
            last_error = "group_overlap"
            continue
        return {
            "train_idx": np.asarray(train_idx),
            "test_idx": np.asarray(test_idx),
            "groups": groups,
            "split_random_state": split_random_state,
            "group_column_status": status,
        }
    raise ValueError(last_error)


def inner_validation_split(
    data: pd.DataFrame,
    train_idx: np.ndarray,
    groups: pd.Series,
) -> dict[str, Any]:
    """Create a grouped validation split inside the outer training set."""
    train_data = data.iloc[train_idx].reset_index(drop=True)
    train_groups = groups.iloc[train_idx].reset_index(drop=True)
    last_error = "no_valid_inner_grouped_split"
    for offset in range(MAX_GROUP_SPLIT_ATTEMPTS):
        split_random_state = RANDOM_STATE + 100 + offset
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=INNER_VALIDATION_SIZE,
            random_state=split_random_state,
        )
        try:
            inner_train_local, val_local = next(
                splitter.split(train_data, train_data["label"], groups=train_groups)
            )
        except ValueError as exc:
            last_error = str(exc)
            break
        if train_data.iloc[inner_train_local]["label"].nunique() != 2:
            last_error = "inner_train_split_single_label"
            continue
        if train_data.iloc[val_local]["label"].nunique() != 2:
            last_error = "validation_split_single_label"
            continue
        if set(train_groups.iloc[inner_train_local]) & set(train_groups.iloc[val_local]):
            last_error = "inner_group_overlap"
            continue
        return {
            "method": "grouped",
            "reason": "ok",
            "train_core_idx": train_idx[np.asarray(inner_train_local)],
            "val_idx": train_idx[np.asarray(val_local)],
            "split_random_state": split_random_state,
            "group_overlap_count": 0,
        }

    return {
        **ft.inner_validation_split(data, train_idx, groups),
        "reason": last_error,
    }


def compute_numeric_features(
    data: pd.DataFrame,
    train_core_idx: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Standardize numeric features using the training-core rows only."""
    raw = data[NUMERIC_FEATURE_COLUMNS].astype(float)
    means = raw.iloc[train_core_idx].mean(axis=0)
    stds = raw.iloc[train_core_idx].std(axis=0, ddof=0).replace(0, 1).fillna(1)
    standardized = ((raw - means) / stds).fillna(0).to_numpy(dtype=np.float32)
    return standardized, {
        "columns": NUMERIC_FEATURE_COLUMNS,
        "mean": {key: float(value) for key, value in means.items()},
        "std": {key: float(value) for key, value in stds.items()},
    }


def add_cdr_marker_tokens(tokenizer: Any) -> dict[str, Any]:
    """Add all CDR marker tokens to a tokenizer."""
    before = int(len(tokenizer))
    added = tokenizer.add_special_tokens({"additional_special_tokens": CDR_MARKER_TOKENS})
    token_ids = {
        token: int(tokenizer.convert_tokens_to_ids(token)) for token in CDR_MARKER_TOKENS
    }
    after = int(len(tokenizer))
    return {
        "vocab_size_before": before,
        "vocab_size_after": after,
        "new_token_count": int(added),
        "all_marker_tokens_available": all(token_id >= 0 for token_id in token_ids.values()),
        "marker_token_ids": token_ids,
    }


def tokenize_marked_pairs(
    tokenizer: Any,
    heavy_texts: list[str],
    light_texts: list[str],
    max_length: int | None,
) -> Any:
    """Tokenize marked heavy/light texts as true paired inputs."""
    kwargs: dict[str, Any] = {
        "padding": True,
        "return_tensors": "pt",
        "return_special_tokens_mask": True,
    }
    if max_length is not None:
        kwargs.update({"truncation": True, "max_length": max_length})
    return tokenizer(heavy_texts, text_pair=light_texts, **kwargs)


def special_mask(encoded: Any, tokenizer: Any) -> torch.Tensor:
    """Return a boolean special-token mask."""
    if "special_tokens_mask" in encoded:
        return encoded["special_tokens_mask"].bool()
    input_ids = encoded["input_ids"]
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for token_id in getattr(tokenizer, "all_special_ids", []) or []:
        mask |= input_ids.eq(int(token_id))
    return mask


def build_region_masks(encoded: Any, tokenizer: Any, marker_token_ids: dict[str, int]) -> dict[str, torch.Tensor]:
    """Build whole-pair and CDR masks from explicit marker tokens."""
    input_ids = encoded["input_ids"]
    attention = encoded["attention_mask"].bool()
    special = special_mask(encoded, tokenizer)
    whole = attention & ~special
    masks: dict[str, torch.Tensor] = {"whole": whole}
    success: dict[str, torch.Tensor] = {}
    for region in POOL_REGIONS:
        start_id = marker_token_ids[REGION_START_TOKENS[region]]
        end_id = marker_token_ids[REGION_END_TOKENS[region]]
        region_mask = torch.zeros_like(whole)
        region_success = torch.zeros(input_ids.shape[0], dtype=torch.bool)
        for row_index in range(input_ids.shape[0]):
            positions = input_ids[row_index]
            starts = torch.where(positions.eq(start_id))[0]
            ends = torch.where(positions.eq(end_id))[0]
            if len(starts) != 1 or len(ends) != 1:
                continue
            start = int(starts[0].item())
            end_candidates = ends[ends > start]
            if len(end_candidates) != 1:
                continue
            end = int(end_candidates[0].item())
            if end <= start + 1:
                continue
            region_mask[row_index, start + 1 : end] = whole[row_index, start + 1 : end]
            region_success[row_index] = bool(region_mask[row_index].sum().item() > 0)
        masks[region] = region_mask
        success[f"{region}_mask_success"] = region_success
    masks.update(success)
    return masks


def make_collate_fn(tokenizer: Any, max_length: int | None, marker_token_ids: dict[str, int]) -> Any:
    """Create a collate function that tokenizes and builds marker-derived masks."""

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        heavy = [item["marked_heavy_text"] for item in batch]
        light = [item["marked_light_text"] for item in batch]
        encoded = tokenize_marked_pairs(tokenizer, heavy, light, max_length)
        masks = build_region_masks(encoded, tokenizer, marker_token_ids)
        labels = torch.tensor([float(item["label"]) for item in batch], dtype=torch.float32)
        numeric = torch.tensor(
            np.stack([item["numeric_features"] for item in batch]),
            dtype=torch.float32,
        )
        return {
            "encoded": encoded,
            "masks": masks,
            "labels": labels,
            "numeric_features": numeric,
        }

    return collate


def make_loader(
    data: pd.DataFrame,
    indices: np.ndarray,
    numeric_features: np.ndarray,
    tokenizer: Any,
    max_length: int | None,
    marker_token_ids: dict[str, int],
    shuffle: bool,
) -> DataLoader:
    """Build a DataLoader for CDR-marked paired inputs."""
    selected = data.iloc[indices].reset_index(drop=True)
    selected_features = numeric_features[np.asarray(indices)]
    dataset = BioawareDataset(selected, selected_features)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        collate_fn=make_collate_fn(tokenizer, max_length, marker_token_ids),
    )


def move_encoded_to_device(encoded: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tokenizer tensor outputs to device."""
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in encoded.items()
    }


def move_tensor_dict(values: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Move tensor dictionary values to device."""
    return {key: value.to(device) for key, value in values.items()}


def load_peft_optional() -> tuple[dict[str, Any] | None, str | None]:
    """Import PEFT when available; otherwise the model uses frozen backbone."""
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except Exception as exc:
        return None, f"{exc.__class__.__name__}: {exc}"
    return {
        "LoraConfig": LoraConfig,
        "TaskType": TaskType,
        "get_peft_model": get_peft_model,
    }, None


def terminal_module_name(module_name: str) -> str:
    """Return the terminal component of a dotted module name."""
    return module_name.rsplit(".", 1)[-1]


def find_lora_target_modules(backbone: nn.Module) -> dict[str, Any]:
    """Find common projection module names present as linear layers."""
    found: dict[str, list[str]] = {}
    for module_name, module in backbone.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        terminal = terminal_module_name(module_name)
        if terminal in COMMON_LORA_TARGET_NAMES:
            found.setdefault(terminal, []).append(module_name)
    target_modules = [name for name in COMMON_LORA_TARGET_NAMES if name in found]
    return {
        "target_modules": target_modules,
        "matched_module_count": int(sum(len(values) for values in found.values())),
        "matched_modules_by_target": found,
    }


def apply_lora(backbone: nn.Module, peft_api: dict[str, Any], targets: list[str]) -> nn.Module:
    """Freeze the base model and attach LoRA adapters."""
    if not targets:
        raise RuntimeError("No common LoRA target modules were found.")
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    config = peft_api["LoraConfig"](
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=targets,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=peft_api["TaskType"].FEATURE_EXTRACTION,
    )
    return peft_api["get_peft_model"](backbone, config)


def configure_marker_embedding_training(
    backbone: nn.Module,
    marker_token_ids: dict[str, int],
) -> dict[str, Any]:
    """Train marker-token embedding rows while masking non-marker gradients."""
    embedding = backbone.get_input_embeddings()
    if embedding is None or not hasattr(embedding, "weight"):
        return {"available": False, "reason": "no_input_embedding_weight"}
    marker_ids = sorted(set(marker_token_ids.values()))
    embedding.weight.requires_grad_(True)
    mask = torch.zeros_like(embedding.weight, dtype=torch.float32)
    mask[torch.tensor(marker_ids, dtype=torch.long)] = 1.0

    def mask_grad(grad: torch.Tensor) -> torch.Tensor:
        return grad * mask.to(grad.device)

    embedding.weight.register_hook(mask_grad)
    return {
        "available": True,
        "marker_token_ids": marker_ids,
        "embedding_parameter_id": id(embedding.weight),
        "trained_marker_row_count": int(len(marker_ids)),
    }


def trainable_parameter_summary(model: nn.Module) -> dict[str, Any]:
    """Count trainable and total parameters."""
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return {
        "trainable_parameter_count": int(trainable),
        "total_parameter_count": int(total),
        "trainable_parameter_fraction": float(trainable / total) if total else 0.0,
    }


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Copy trainable parameters only for compact checkpointing."""
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    state = model.state_dict()
    return {
        name: state[name].detach().cpu().clone()
        for name in sorted(names)
        if name in state
    }


def restore_trainable_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Restore trainable parameters captured by trainable_state_dict."""
    current = model.state_dict()
    for name, tensor in state.items():
        current[name].copy_(tensor.to(current[name].device))
    model.load_state_dict(current)


def smoothed_labels(labels: torch.Tensor) -> torch.Tensor:
    """Apply binary label smoothing y * 0.9 + 0.05."""
    return labels * (LABEL_SMOOTHING_POSITIVE - LABEL_SMOOTHING_NEGATIVE) + LABEL_SMOOTHING_NEGATIVE


def build_optimizer_and_scheduler(
    model: BioawareIgBertClassifier,
    train_loader: DataLoader,
    use_lora: bool,
    marker_embedding_info: dict[str, Any],
) -> tuple[torch.optim.Optimizer, Any]:
    """Create AdamW with requested learning-rate groups and warmup."""
    marker_parameter_id = marker_embedding_info.get("embedding_parameter_id")
    head_params = list(model.classifier.parameters())
    marker_params = []
    lora_params = []
    for parameter in model.backbone.parameters():
        if not parameter.requires_grad:
            continue
        if marker_parameter_id is not None and id(parameter) == marker_parameter_id:
            marker_params.append(parameter)
        else:
            lora_params.append(parameter)
    groups = [
        {
            "params": head_params,
            "lr": HEAD_LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
        }
    ]
    if use_lora and lora_params:
        groups.append(
            {
                "params": lora_params,
                "lr": LORA_LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
            }
        )
    if marker_params:
        groups.append(
            {
                "params": marker_params,
                "lr": MARKER_EMBEDDING_LEARNING_RATE,
                "weight_decay": 0.0,
            }
        )
    optimizer = torch.optim.AdamW(groups)
    total_steps = max(1, MAX_EPOCHS * len(train_loader))
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler


def evaluate_model(
    model: BioawareIgBertClassifier,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[dict[str, Any], float, np.ndarray, np.ndarray]:
    """Evaluate the model and return metrics, loss, labels, and scores."""
    model.eval()
    losses = []
    labels = []
    scores = []
    with torch.no_grad():
        for batch in loader:
            encoded = move_encoded_to_device(batch["encoded"], device)
            masks = move_tensor_dict(batch["masks"], device)
            numeric = batch["numeric_features"].to(device)
            y = batch["labels"].to(device)
            logits = model(encoded, masks, numeric)
            loss = loss_fn(logits, y)
            losses.append(float(loss.detach().cpu().item()) * len(y))
            scores.append(torch.sigmoid(logits).detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())
    y_true = np.concatenate(labels).astype(int)
    y_score = np.concatenate(scores)
    y_pred = (y_score >= 0.5).astype(int)
    return ft.metric_dict(y_true, y_pred, y_score), float(sum(losses) / len(y_true)), y_true, y_score


def configure_model_for_seed(
    seed: int,
    tokenizer_length: int,
    marker_token_ids: dict[str, int],
    peft_api: dict[str, Any] | None,
) -> tuple[BioawareIgBertClassifier, dict[str, Any]]:
    """Load IgBert, resize marker embeddings, and configure PEFT or frozen mode."""
    set_seed(seed)
    backbone = AutoModel.from_pretrained(MODEL_NAME)
    backbone.resize_token_embeddings(tokenizer_length)
    hidden_size = int(getattr(backbone.config, "hidden_size"))
    lora_info = find_lora_target_modules(backbone)
    use_lora = bool(peft_api is not None and lora_info["target_modules"])

    if use_lora:
        backbone = apply_lora(backbone, peft_api, lora_info["target_modules"])
        marker_embedding_info = configure_marker_embedding_training(backbone, marker_token_ids)
    else:
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        marker_embedding_info = {"available": False, "reason": "frozen_backbone_fallback"}

    include_cls = True
    model = BioawareIgBertClassifier(
        backbone=backbone,
        hidden_size=hidden_size,
        numeric_feature_count=len(NUMERIC_FEATURE_COLUMNS),
        include_cls=include_cls,
    )
    for parameter in model.classifier.parameters():
        parameter.requires_grad_(True)
    trainability = trainable_parameter_summary(model)
    trainability.update(
        {
            "training_mode": "lora_cdr_marked" if use_lora else "frozen_backbone_head_only",
            "peft_available": peft_api is not None,
            "lora": lora_info,
            "marker_embedding_training": marker_embedding_info,
        }
    )
    return model, trainability


def train_one_seed(
    seed: int,
    data: pd.DataFrame,
    split_info: dict[str, Any],
    validation_info: dict[str, Any],
    numeric_features: np.ndarray,
    numeric_feature_info: dict[str, Any],
    tokenizer: Any,
    max_length: int | None,
    marker_token_ids: dict[str, int],
    peft_api: dict[str, Any] | None,
    device_name: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Train one seed and return seed metrics plus checkpoint."""
    device = torch.device(device_name)
    model, trainability = configure_model_for_seed(
        seed,
        tokenizer_length=len(tokenizer),
        marker_token_ids=marker_token_ids,
        peft_api=peft_api,
    )
    model.to(device)

    train_core_idx = validation_info["train_core_idx"]
    val_idx = validation_info["val_idx"]
    test_idx = split_info["test_idx"]
    train_loader = make_loader(
        data,
        train_core_idx,
        numeric_features,
        tokenizer,
        max_length,
        marker_token_ids,
        shuffle=True,
    )
    val_loader = make_loader(
        data,
        val_idx,
        numeric_features,
        tokenizer,
        max_length,
        marker_token_ids,
        shuffle=False,
    )
    test_loader = make_loader(
        data,
        test_idx,
        numeric_features,
        tokenizer,
        max_length,
        marker_token_ids,
        shuffle=False,
    )
    y_train = data.iloc[train_core_idx]["label"].to_numpy(dtype=np.float32)
    positives = float(np.sum(y_train == 1))
    negatives = float(np.sum(y_train == 0))
    pos_weight_value = negatives / positives if positives > 0 else 1.0
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    )
    use_lora = trainability["training_mode"] == "lora_cdr_marked"
    optimizer, scheduler = build_optimizer_and_scheduler(
        model,
        train_loader,
        use_lora=use_lora,
        marker_embedding_info=trainability["marker_embedding_training"],
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_record: dict[str, Any] | None = None
    best_score = -np.inf
    stale_epochs = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        losses = []
        progress = tqdm(
            train_loader,
            desc=f"bioaware seed {seed} epoch {epoch}",
            leave=False,
            disable=os.environ.get("BIOAWARE_IGBERT_TQDM", "0") != "1",
        )
        for batch in progress:
            encoded = move_encoded_to_device(batch["encoded"], device)
            masks = move_tensor_dict(batch["masks"], device)
            numeric = batch["numeric_features"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(encoded, masks, numeric)
            loss = loss_fn(logits, smoothed_labels(labels))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                MAX_GRAD_NORM,
            )
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu().item()))

        val_metrics, val_loss, _, _ = evaluate_model(model, val_loader, loss_fn, device)
        val_roc = -np.inf if val_metrics["roc_auc"] is None else float(val_metrics["roc_auc"])
        val_pr = (
            -np.inf
            if val_metrics["average_precision"] is None
            else float(val_metrics["average_precision"])
        )
        train_loss = float(np.mean(losses))
        record = {
            "epoch": int(epoch),
            "train_loss": train_loss,
            "validation_loss": float(val_loss),
            "validation_roc_auc": None if val_roc == -np.inf else val_roc,
            "validation_average_precision": None if val_pr == -np.inf else val_pr,
            "validation_f1": val_metrics["f1"],
        }
        history.append(record)
        print(
            f"seed={seed} epoch {epoch}: train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_roc_auc={val_roc:.4f} val_pr_auc={val_pr:.4f}",
            flush=True,
        )

        current_score = 0.5 * val_roc + 0.5 * val_pr
        if current_score > best_score + MIN_DELTA:
            best_score = current_score
            best_record = dict(record)
            best_state = trainable_state_dict(model)
            stale_epochs = 0
        else:
            stale_epochs += 1

        overfit_stop = False
        if len(history) >= 4:
            recent = history[-3:]
            train_improves = recent[-1]["train_loss"] < recent[0]["train_loss"]
            val_loss_worse = recent[-1]["validation_loss"] > recent[0]["validation_loss"] + 0.02
            overfit_stop = bool(train_improves and val_loss_worse)
        if stale_epochs >= EARLY_STOPPING_PATIENCE or overfit_stop:
            break

    if best_state is not None:
        restore_trainable_state(model, best_state)

    test_metrics, test_loss, y_true, y_score = evaluate_model(
        model,
        test_loader,
        loss_fn,
        device,
    )
    test_idx = np.asarray(test_idx)
    y_pred = (y_score >= 0.5).astype(int)
    prediction_records = [
        {
            "data_index": int(index),
            "row_id": int(data.iloc[index]["row_id"]),
            "true_label": int(label),
            "predicted_probability": float(score),
            "predicted_label": int(pred),
        }
        for index, label, score, pred in zip(test_idx, y_true, y_score, y_pred)
    ]
    split_diag = ft.split_diagnostics(
        data,
        split_info["train_idx"],
        split_info["test_idx"],
        split_info["groups"],
    )
    val_diag = ft.split_diagnostics(
        data,
        train_core_idx,
        val_idx,
        split_info["groups"],
    )
    overfit = ft.overfitting_summary(history)
    checkpoint = {
        "model_name": MODEL_NAME,
        "seed": int(seed),
        "training_mode": trainability["training_mode"],
        "max_length": max_length,
        "marker_token_ids": marker_token_ids,
        "numeric_feature_info": numeric_feature_info,
        "head_hidden_size": HEAD_HIDDEN_SIZE,
        "head_dropout": HEAD_DROPOUT,
        "trainability": trainability,
        "trainable_state_dict": trainable_state_dict(model),
        "best_epoch": None if best_record is None else int(best_record["epoch"]),
        "best_validation_score": None if best_record is None else float(best_score),
        "test_metrics": test_metrics,
    }
    result = {
        "seed": int(seed),
        "valid": True,
        "reason": "ok",
        "model_name": MODEL_NAME,
        "device": device_name,
        "training_mode": trainability["training_mode"],
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "epochs_trained": int(len(history)),
        "best_epoch": None if best_record is None else int(best_record["epoch"]),
        "best_epoch_train_loss": None
        if best_record is None
        else float(best_record["train_loss"]),
        "best_epoch_validation_loss": None
        if best_record is None
        else float(best_record["validation_loss"]),
        "best_epoch_validation_roc_auc": None
        if best_record is None
        else best_record["validation_roc_auc"],
        "best_epoch_validation_average_precision": None
        if best_record is None
        else best_record["validation_average_precision"],
        "pos_weight": float(pos_weight_value),
        "history": history,
        "split": split_diag,
        "validation_split": {
            **val_diag,
            "method": validation_info["method"],
            "reason": validation_info["reason"],
            "split_random_state": validation_info["split_random_state"],
        },
        "test_loss": float(test_loss),
        "metrics": {**split_diag, **test_metrics},
        "overfitting": overfit,
        "trainability": trainability,
        "optimizer": {
            "type": "AdamW",
            "classifier_learning_rate": HEAD_LEARNING_RATE,
            "lora_learning_rate": LORA_LEARNING_RATE if use_lora else None,
            "marker_embedding_learning_rate": MARKER_EMBEDDING_LEARNING_RATE
            if trainability["marker_embedding_training"].get("available")
            else None,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "max_grad_norm": MAX_GRAD_NORM,
        },
        "prediction_records": prediction_records,
    }
    return result, checkpoint


def run_kmer_baseline(
    data: pd.DataFrame,
    split_info: dict[str, Any],
    text_column: str,
) -> dict[str, Any]:
    """Train/evaluate one k-mer baseline on the paired primary subset."""
    train_idx = split_info["train_idx"]
    test_idx = split_info["test_idx"]
    model = make_kmer_pipeline()
    model.fit(data.iloc[train_idx][text_column].astype(str), data.iloc[train_idx]["label"].astype(int))
    y_true = data.iloc[test_idx]["label"].astype(int).to_numpy()
    y_score = positive_scores(model, data.iloc[test_idx][text_column].astype(str))
    y_pred = (y_score >= 0.5).astype(int)
    return {
        "text_column": text_column,
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "metrics": ft.metric_dict(y_true, y_pred, y_score),
    }


def run_subset_kmer_baselines(data: pd.DataFrame, split_info: dict[str, Any]) -> dict[str, Any]:
    """Run whole-pair and CDR-local k-mer baselines on the same subset."""
    baselines = {}
    for name, column in {
        "whole_pair_kmer": "whole_pair_kmer_text",
        "all_cdr_kmer": "all_cdr_kmer_text",
        "cdrh3_cdrl3_kmer": "cdrh3_cdrl3_kmer_text",
    }.items():
        try:
            baselines[name] = run_kmer_baseline(data, split_info, column)
        except Exception as exc:
            baselines[name] = {"valid": False, "reason": str(exc), "text_column": column}
    return baselines


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute mean/std across seeds."""
    valid = [result for result in results if result.get("valid")]
    aggregate: dict[str, Any] = {"valid_seed_count": int(len(valid)), "std_ddof": STD_DDOF}
    metric_map = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "f1": "f1",
        "balanced_accuracy": "balanced_accuracy",
    }
    for output_name, metric_name in metric_map.items():
        values = [
            float(result["metrics"][metric_name])
            for result in valid
            if result.get("metrics", {}).get(metric_name) is not None
        ]
        aggregate[output_name] = {
            "values": values,
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values, ddof=STD_DDOF)) if len(values) > 1 else 0.0,
            "min": float(np.min(values)) if values else None,
            "max": float(np.max(values)) if values else None,
        }
    aggregate["overfit_seed_count"] = int(
        sum(1 for result in valid if result.get("overfitting", {}).get("evidence"))
    )
    return aggregate


def load_previous_direct_context() -> dict[str, Any]:
    """Load previous direct fine-tuning context when present."""
    context = {
        "roc_auc": PREVIOUS_DIRECT_FINETUNE_ROC_AUC,
        "pr_auc": PREVIOUS_DIRECT_FINETUNE_PR_AUC,
        "overfit_seed_count": None,
        "source": "hardcoded_current_fact",
    }
    if DIRECT_FINETUNE_METRICS_PATH.exists():
        try:
            payload = json.loads(DIRECT_FINETUNE_METRICS_PATH.read_text())
            seed_mean = payload.get("comparison", {}).get("seed_mean", {})
            context.update(
                {
                    "roc_auc": float(seed_mean.get("roc_auc", context["roc_auc"])),
                    "pr_auc": float(seed_mean.get("pr_auc", context["pr_auc"])),
                    "overfit_seed_count": payload.get("comparison", {}).get(
                        "overfit_seed_count"
                    ),
                    "source": str(DIRECT_FINETUNE_METRICS_PATH.relative_to(PROJECT_ROOT)),
                }
            )
        except Exception as exc:
            context["load_error"] = str(exc)
    return context


def build_comparison(
    aggregate: dict[str, Any],
    baselines: dict[str, Any],
) -> dict[str, Any]:
    """Compare IgBert final model against same-subset and full-dataset baselines."""
    direct = load_previous_direct_context()
    model_roc = aggregate["roc_auc"]["mean"]
    model_pr = aggregate["pr_auc"]["mean"]
    whole_pair = baselines.get("whole_pair_kmer", {}).get("metrics", {})
    all_cdr = baselines.get("all_cdr_kmer", {}).get("metrics", {})
    whole_roc = whole_pair.get("roc_auc")
    whole_pr = whole_pair.get("average_precision")
    all_cdr_roc = all_cdr.get("roc_auc")
    all_cdr_pr = all_cdr.get("average_precision")
    direct_overfit = direct.get("overfit_seed_count")
    return {
        "baselines": {
            "full_dataset_kmer_current_fact": {
                "roc_auc": FULL_DATASET_KMER_GROUPED_ROC_AUC,
                "pr_auc": FULL_DATASET_KMER_GROUPED_PR_AUC,
            },
            "previous_direct_finetuning_seed_mean": direct,
            "paired_subset_kmer_baselines": baselines,
        },
        "model_seed_mean": {
            "roc_auc": model_roc,
            "pr_auc": model_pr,
            "delta_roc_auc_vs_paired_whole_pair_kmer": None
            if model_roc is None or whole_roc is None
            else float(model_roc - whole_roc),
            "delta_pr_auc_vs_paired_whole_pair_kmer": None
            if model_pr is None or whole_pr is None
            else float(model_pr - whole_pr),
        },
        "all_cdr_kmer_beats_whole_pair_kmer_roc_auc": bool(
            all_cdr_roc is not None and whole_roc is not None and all_cdr_roc > whole_roc
        ),
        "all_cdr_kmer_beats_whole_pair_kmer_pr_auc": bool(
            all_cdr_pr is not None and whole_pr is not None and all_cdr_pr > whole_pr
        ),
        "bioaware_model_beats_paired_whole_pair_kmer_roc_auc": bool(
            model_roc is not None and whole_roc is not None and model_roc > whole_roc
        ),
        "bioaware_model_beats_paired_whole_pair_kmer_pr_auc": bool(
            model_pr is not None and whole_pr is not None and model_pr > whole_pr
        ),
        "overfitting_reduced_vs_previous_direct_finetuning": (
            None
            if direct_overfit is None
            else bool(aggregate["overfit_seed_count"] < int(direct_overfit))
        ),
    }


def best_result_by_validation(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Select best seed result by validation ROC/PR average."""
    valid = [result for result in results if result.get("valid")]
    if not valid:
        return None
    return max(
        valid,
        key=lambda result: (
            float(result.get("best_epoch_validation_roc_auc") or -np.inf)
            + float(result.get("best_epoch_validation_average_precision") or -np.inf)
        )
        / 2,
    )


def subgroup_metrics(
    data: pd.DataFrame,
    best_result: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Compute target-region and structure subgroup metrics on test predictions."""
    if best_result is None:
        return []
    predictions = pd.DataFrame.from_records(best_result["prediction_records"])
    if predictions.empty:
        return []
    merged = predictions.merge(
        data.reset_index(names="data_index"),
        on="data_index",
        how="left",
        suffixes=("", "_row"),
    )
    subgroups: list[tuple[str, str, pd.Series]] = []
    for column, label in [
        ("targets_rbd", "RBD rows"),
        ("targets_ntd", "NTD rows"),
        ("targets_spike", "Spike rows"),
    ]:
        if column in merged.columns:
            mask = merged[column].astype(str).str.lower().isin({"true", "1", "yes"})
            subgroups.append((column, label, mask))
    if "metadata_target_region" in merged.columns:
        for value, group in merged.groupby(merged["metadata_target_region"].fillna("").astype(str)):
            if value.strip():
                subgroups.append(("metadata_target_region", value, merged.index.isin(group.index)))
    if "has_structure" in merged.columns:
        has_structure = merged["has_structure"].astype(str).str.lower().isin({"true", "1", "yes"})
    elif "metadata_structure" in merged.columns:
        has_structure = merged["metadata_structure"].fillna("").astype(str).str.strip().ne("")
    else:
        has_structure = pd.Series([False] * len(merged), index=merged.index)
    subgroups.append(("structure", "has_structure", has_structure))
    subgroups.append(("structure", "without_structure", ~has_structure))

    records = []
    for subgroup_type, subgroup_name, mask in subgroups:
        subset = merged.loc[mask].copy()
        label_counts = ft.label_counts(subset["true_label"].astype(int).to_numpy()) if len(subset) else {"0": 0, "1": 0}
        record: dict[str, Any] = {
            "subgroup_type": subgroup_type,
            "subgroup_name": subgroup_name,
            "row_count": int(len(subset)),
            "label_counts": label_counts,
            "positive_fraction": float(subset["true_label"].mean()) if len(subset) else None,
            "metrics_available": False,
            "reason": None,
        }
        if len(subset) < SUBGROUP_MIN_ROWS:
            record["reason"] = "too_few_rows"
        elif subset["true_label"].nunique() < 2:
            record["reason"] = "single_label"
        else:
            y_true = subset["true_label"].astype(int).to_numpy()
            y_score = subset["predicted_probability"].astype(float).to_numpy()
            y_pred = subset["predicted_label"].astype(int).to_numpy()
            record.update(ft.metric_dict(y_true, y_pred, y_score))
            record["metrics_available"] = True
        records.append(record)
    return records


def select_and_save_best_checkpoint(
    checkpoints: list[dict[str, Any]],
) -> tuple[int | None, str | None]:
    """Save best checkpoint by validation score."""
    valid = [
        checkpoint
        for checkpoint in checkpoints
        if checkpoint.get("best_validation_score") is not None
    ]
    if not valid:
        return None, None
    best = max(valid, key=lambda item: float(item["best_validation_score"]))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best, BEST_MODEL_PATH)
    return int(best["seed"]), str(BEST_MODEL_PATH.relative_to(PROJECT_ROOT))


def metric_or_na(value: Any) -> str:
    """Format optional numeric metrics."""
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def format_seed_results_table(results: list[dict[str, Any]]) -> list[str]:
    """Format seed-wise metrics as Markdown."""
    lines = [
        (
            "| Seed | Best epoch | Train loss | Val loss | Val ROC-AUC | Val PR-AUC | "
            "Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC | "
            "PR-AUC | Confusion matrix |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        if not result.get("valid"):
            lines.append(
                f"| {result.get('seed')} | skipped | {result.get('reason', 'unknown')} | "
                "n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
            )
            continue
        metrics = result["metrics"]
        lines.append(
            f"| {result['seed']} | {result['best_epoch']} | "
            f"{metric_or_na(result['best_epoch_train_loss'])} | "
            f"{metric_or_na(result['best_epoch_validation_loss'])} | "
            f"{metric_or_na(result['best_epoch_validation_roc_auc'])} | "
            f"{metric_or_na(result['best_epoch_validation_average_precision'])} | "
            f"{metrics['accuracy']:.4f} | {metrics['balanced_accuracy']:.4f} | "
            f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | "
            f"{metrics['f1']:.4f} | {metric_or_na(metrics['roc_auc'])} | "
            f"{metric_or_na(metrics['average_precision'])} | "
            f"{metrics['confusion_matrix']} |"
        )
    return lines


def format_aggregate_table(aggregate: dict[str, Any]) -> list[str]:
    """Format aggregate metrics as Markdown."""
    lines = ["| Metric | Mean | Std | Min | Max |", "|---|---:|---:|---:|---:|"]
    for metric in ["roc_auc", "pr_auc", "f1", "balanced_accuracy"]:
        item = aggregate[metric]
        lines.append(
            f"| {metric} | {metric_or_na(item['mean'])} | {metric_or_na(item['std'])} | "
            f"{metric_or_na(item['min'])} | {metric_or_na(item['max'])} |"
        )
    return lines


def save_cdr_coverage_figure(annotation_summary: dict[str, Any], annotated: pd.DataFrame | None) -> None:
    """Save a CDR coverage figure."""
    if annotated is None or annotated.empty:
        counts = {"AbNumber available": 0}
    else:
        counts = {
            "CDRH1": int(annotated["cdrh1_found"].sum()),
            "CDRH2": int(annotated["cdrh2_found"].sum()),
            "CDRH3": int(annotated["cdrh3_found"].sum()),
            "CDRL1": int(annotated["cdrl1_found"].sum()),
            "CDRL2": int(annotated["cdrl2_found"].sum()),
            "CDRL3": int(annotated["cdrl3_found"].sum()),
            "All six": int(annotated["all_six_cdrs_found"].sum()),
        }
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(list(counts.keys()), list(counts.values()), color="#4C78A8")
    ax.set_ylabel("Rows")
    ax.set_title("Bioaware CDR annotation coverage")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    CDR_COVERAGE_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CDR_COVERAGE_FIGURE_PATH, dpi=200)
    plt.close(fig)


def save_training_curves(results: list[dict[str, Any]]) -> None:
    """Save training and validation curves."""
    records = []
    for result in results:
        if not result.get("valid"):
            continue
        for item in result.get("history", []):
            records.append({"seed": result["seed"], **item})
    if not records:
        return
    table = pd.DataFrame.from_records(records)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    specs = [
        ("train_loss", "Train loss", axes[0, 0]),
        ("validation_loss", "Validation loss", axes[0, 1]),
        ("validation_roc_auc", "Validation ROC-AUC", axes[1, 0]),
        ("validation_average_precision", "Validation PR-AUC", axes[1, 1]),
    ]
    for seed, group in table.groupby("seed"):
        for column, title, axis in specs:
            axis.plot(group["epoch"], group[column], marker="o", label=f"seed {seed}")
            axis.set_title(title)
            axis.set_xlabel("Epoch")
            axis.grid(alpha=0.25)
    axes[0, 0].set_ylabel("Loss")
    axes[0, 1].set_ylabel("Loss")
    axes[1, 0].set_ylabel("ROC-AUC")
    axes[1, 1].set_ylabel("PR-AUC")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].legend(fontsize=8)
    fig.tight_layout()
    TRAINING_CURVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(TRAINING_CURVE_PATH, dpi=200)
    plt.close(fig)


def save_seed_summary_figure(
    results: list[dict[str, Any]],
    aggregate: dict[str, Any],
    baselines: dict[str, Any],
) -> None:
    """Save seed metric summary against same-subset baselines."""
    valid = [result for result in results if result.get("valid")]
    if not valid:
        return
    labels = [str(result["seed"]) for result in valid]
    roc_values = [result["metrics"]["roc_auc"] for result in valid]
    pr_values = [result["metrics"]["average_precision"] for result in valid]
    x = np.arange(len(valid))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, roc_values, width, label="IgBert ROC-AUC", color="#4C78A8")
    ax.bar(x + width / 2, pr_values, width, label="IgBert PR-AUC", color="#54A24B")
    whole = baselines.get("whole_pair_kmer", {}).get("metrics", {})
    if whole.get("roc_auc") is not None:
        ax.axhline(float(whole["roc_auc"]), color="#4C78A8", linestyle="--", label="paired k-mer ROC-AUC")
    if whole.get("average_precision") is not None:
        ax.axhline(float(whole["average_precision"]), color="#54A24B", linestyle="--", label="paired k-mer PR-AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Grouped held-out test metric")
    ax.set_title("Bioaware IgBert final seed summary")
    ax.legend(fontsize=8)
    summary = (
        f"mean ROC={metric_or_na(aggregate['roc_auc']['mean'])}, "
        f"mean PR={metric_or_na(aggregate['pr_auc']['mean'])}"
    )
    ax.text(0.01, 0.02, summary, transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    SEED_SUMMARY_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SEED_SUMMARY_FIGURE_PATH, dpi=200)
    plt.close(fig)


def save_subgroup_figure(subgroups: list[dict[str, Any]]) -> None:
    """Save subgroup ROC/PR metrics when available."""
    available = [item for item in subgroups if item.get("metrics_available")]
    if not available:
        return
    labels = [f"{item['subgroup_type']}:{item['subgroup_name']}" for item in available]
    roc = [item.get("roc_auc") for item in available]
    pr = [item.get("average_precision") for item in available]
    x = np.arange(len(available))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.5), 5))
    ax.bar(x - width / 2, roc, width, label="ROC-AUC", color="#4C78A8")
    ax.bar(x + width / 2, pr, width, label="PR-AUC", color="#54A24B")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Bioaware IgBert subgroup metrics")
    ax.set_ylabel("Metric")
    ax.legend(fontsize=8)
    fig.tight_layout()
    SUBGROUP_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SUBGROUP_FIGURE_PATH, dpi=200)
    plt.close(fig)


def save_figures(
    annotation_summary: dict[str, Any],
    annotated: pd.DataFrame | None,
    results: list[dict[str, Any]] | None = None,
    aggregate: dict[str, Any] | None = None,
    baselines: dict[str, Any] | None = None,
    subgroups: list[dict[str, Any]] | None = None,
) -> None:
    """Save requested figures when inputs exist."""
    save_cdr_coverage_figure(annotation_summary, annotated)
    if results is not None and aggregate is not None and baselines is not None:
        save_training_curves(results)
        save_seed_summary_figure(results, aggregate, baselines)
    if subgroups is not None:
        save_subgroup_figure(subgroups)


def build_annotation_report(summary: dict[str, Any]) -> str:
    """Build the all-six-CDR annotation report."""
    lines = [
        "# Bioaware CDR Annotation Report",
        "",
        "This annotation step uses only existing paired antibody sequences.",
        "",
        f"status: `{summary['status']}`",
        "",
    ]
    if summary["status"] != "available":
        dep = summary.get("dependency_status", {})
        lines.extend(
            [
                "## Dependency Gate",
                "",
                f"AbNumber available: `{dep.get('available')}`",
                f"Reason: `{dep.get('message') or dep.get('error')}`",
                "",
                "Model training was not run.",
                "",
            ]
        )
        return "\n".join(lines)

    match = summary["cdr_match_summary"]
    lines.extend(
        [
            "## Coverage",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Raw rows | {summary['raw_row_count']} |",
            f"| Paired annotation candidates | {summary['paired_annotation_candidate_count']} |",
            f"| Heavy annotation OK | {summary['heavy_annotation_ok_count']} |",
            f"| Light annotation OK | {summary['light_annotation_ok_count']} |",
            f"| All six CDRs found | {summary['all_six_cdrs_found_count']} |",
            f"| All six CDRs found fraction | {summary['all_six_cdrs_found_fraction']:.2%} |",
            "",
            "## Existing CDR Metadata Comparison",
            "",
            "| Region | Exact matches | Mismatches | Missing/comparison unavailable |",
            "|---|---:|---:|---:|",
            f"| CDRH3 | {match['cdrh3']['exact_match_count']} | {match['cdrh3']['mismatch_count']} | {match['cdrh3']['missing_count']} |",
            f"| CDRL3 | {match['cdrl3']['exact_match_count']} | {match['cdrl3']['mismatch_count']} | {match['cdrl3']['missing_count']} |",
            "",
            "## Outputs",
            "",
            f"- `{ANNOTATED_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{ANNOTATION_SUMMARY_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{CDR_COVERAGE_FIGURE_PATH.relative_to(PROJECT_ROOT)}`",
            "",
        ]
    )
    return "\n".join(lines)


def build_final_report(payload: dict[str, Any]) -> str:
    """Build the final model Markdown report."""
    if payload["status"] != "available":
        return "\n".join(
            [
                "# Bioaware IgBert Final Report",
                "",
                f"status: `{payload['status']}`",
                f"reason: `{payload.get('reason', 'unknown')}`",
                "",
                "All-six-CDR annotation is required for this final benchmark.",
                "No model training or weak fallback was run.",
                "",
            ]
        )

    annotation = payload["annotation_summary"]
    comparison = payload["comparison"]
    split = payload["split"]
    validation = payload["validation_split"]
    tokenizer_info = payload["tokenizer"]
    nanobody = payload["nanobody_summary"]
    primary = payload["primary_summary"]
    clonotype = payload["clonotype_like_grouping_diagnostic"]
    lines = [
        "# Bioaware IgBert Final Report",
        "",
        "This final benchmark trains a biologically informed IgBert classifier on",
        "paired antibodies only, with explicit all-six-CDR marker tokens and CDR",
        "token pooling. Target-region metadata is used only for analysis.",
        "",
        "## Setup",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Model name | `{payload['model_name']}` |",
        f"| Tokenizer class | `{tokenizer_info['tokenizer_class']}` |",
        f"| Model class | `{payload['model_class']}` |",
        f"| Device | `{payload['device']}` |",
        f"| Training mode | `{payload['training_mode']}` |",
        f"| Marker tokens available | `{tokenizer_info['all_marker_tokens_available']}` |",
        f"| New marker tokens added | `{tokenizer_info['new_token_count']}` |",
        f"| Batch size | `{payload['batch_size']}` |",
        f"| Max epochs | `{payload['max_epochs']}` |",
        f"| Seeds | `{payload['seeds']}` |",
        "",
        "## Annotation And Dataset",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| All-six-CDR annotation worked | {annotation['status'] == 'available'} |",
        f"| All six CDRs found | {annotation['all_six_cdrs_found_count']} |",
        f"| All-six-CDR coverage | {annotation['all_six_cdrs_found_fraction']:.2%} |",
        f"| Paired-antibody rows retained | {primary['primary_row_count']} |",
        f"| Nanobody-like/light-missing rows excluded | {nanobody['count']} |",
        f"| Annotation failure count | {primary['annotation_failure_count']} |",
        f"| Marker insertion failure count | {primary['marker_insertion_failure_count']} |",
        "",
        "Nanobody-like rows were excluded because the primary model is a paired",
        "heavy-light model and nanobodies lack a paired light chain. They require a",
        "separate heavy-only/VHH model.",
        "",
        "## Tokenization Verification",
        "",
        f"Special token IDs: `{tokenizer_info['marker_token_ids']}`",
        "",
        "Example marked heavy sequence:",
        "",
        "```text",
        payload["examples"]["marked_heavy"][:1000],
        "```",
        "",
        "Example marked light sequence:",
        "",
        "```text",
        payload["examples"]["marked_light"][:1000],
        "```",
        "",
        "Tokenized example snippet:",
        "",
        "```text",
        " ".join(payload["examples"]["tokenized_tokens"][:120]),
        "```",
        "",
        "## Leakage Diagnostics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Outer train rows | {split['train_size']} |",
        f"| Outer test rows | {split['test_size']} |",
        f"| Train/test group overlap | {split['group_overlap_count']} |",
        f"| Inner validation method | {validation['method']} |",
        f"| Inner train/validation group overlap | {validation['group_overlap_count']} |",
        f"| Clonotype-like grouping usable | {clonotype['useful_for_grouping']} |",
        f"| Clonotype-like grouping reason | `{clonotype['reason']}` |",
        "",
        "## Seed-Wise Results",
        "",
    ]
    lines.extend(format_seed_results_table(payload["results"]))
    lines.extend(["", "## Mean +/- Std Across Seeds", ""])
    lines.extend(format_aggregate_table(payload["aggregate"]))

    lines.extend(
        [
            "",
            "## Same-Subset K-mer Baselines",
            "",
            "| Baseline | ROC-AUC | PR-AUC | F1 | Balanced accuracy | Confusion matrix |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for name, item in payload["same_subset_kmer_baselines"].items():
        metrics = item.get("metrics", {})
        lines.append(
            f"| {name} | {metric_or_na(metrics.get('roc_auc'))} | "
            f"{metric_or_na(metrics.get('average_precision'))} | "
            f"{metric_or_na(metrics.get('f1'))} | "
            f"{metric_or_na(metrics.get('balanced_accuracy'))} | "
            f"{metrics.get('confusion_matrix', 'n/a')} |"
        )

    lines.extend(
        [
            "",
            "## Subgroup Analysis",
            "",
            "| Subgroup | Rows | Positive fraction | ROC-AUC | PR-AUC | Reason if unavailable |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for item in payload["subgroup_metrics"]:
        lines.append(
            f"| {item['subgroup_type']}:{item['subgroup_name']} | {item['row_count']} | "
            f"{metric_or_na(item.get('positive_fraction'))} | "
            f"{metric_or_na(item.get('roc_auc'))} | "
            f"{metric_or_na(item.get('average_precision'))} | "
            f"{item.get('reason') or ''} |"
        )

    lines.extend(
        [
            "",
            "## Required Conclusions",
            "",
            f"1. Did all-six-CDR annotation work? {'Yes' if annotation['status'] == 'available' else 'No'}.",
            f"2. Paired-antibody rows retained: {primary['primary_row_count']}.",
            f"3. Nanobody-like/light-missing rows excluded: {nanobody['count']}; they require a separate heavy-only/VHH model.",
            f"4. Did all-CDR k-mer beat whole-pair k-mer? ROC-AUC: {comparison['all_cdr_kmer_beats_whole_pair_kmer_roc_auc']}; PR-AUC: {comparison['all_cdr_kmer_beats_whole_pair_kmer_pr_auc']}.",
            f"5. Did CDR-aware IgBert beat paired-subset k-mer on ROC-AUC? {comparison['bioaware_model_beats_paired_whole_pair_kmer_roc_auc']}.",
            f"6. Did CDR-aware IgBert beat paired-subset k-mer on PR-AUC? {comparison['bioaware_model_beats_paired_whole_pair_kmer_pr_auc']}.",
            f"7. Did explicit CDR markers and CDR pooling reduce overfitting? {comparison['overfitting_reduced_vs_previous_direct_finetuning']}.",
            f"8. Did target-region subgroup analysis show biological heterogeneity? See subgroup metrics above; heterogeneous class balance or metrics indicate target-region dependence.",
            "9. Honest final conclusion: if k-mers remain stronger on the same paired subset, the sequence-local CDR motif signal is still better captured by sparse k-mer features than by this parameter-efficient IgBert setup under grouped validation.",
            "",
            "## Artifacts",
            "",
            f"- `{ANNOTATED_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{ANNOTATION_REPORT_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{METRICS_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{SEED_SUMMARY_FIGURE_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{TRAINING_CURVE_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{SUBGROUP_FIGURE_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{CDR_COVERAGE_FIGURE_PATH.relative_to(PROJECT_ROOT)}`",
        ]
    )
    if payload.get("best_checkpoint_path"):
        lines.append(f"- `{payload['best_checkpoint_path']}`")
    lines.append("")
    return "\n".join(lines)


def write_dependency_unavailable(dependency_status: dict[str, Any]) -> None:
    """Write clear reports when AbNumber is unavailable."""
    ANNOTATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ANNOTATION_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "unavailable",
        "model_training_run": False,
        "reason": dependency_status.get("message") or dependency_status.get("error"),
        "dependency_status": dependency_status,
        "input_path": str(INPUT_PATH.relative_to(PROJECT_ROOT)),
        "annotated_output_created": False,
    }
    ANNOTATION_SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    ANNOTATION_REPORT_PATH.write_text(build_annotation_report(summary), encoding="utf-8")
    final_payload = {
        "status": "unavailable",
        "reason": summary["reason"],
        "model_name": MODEL_NAME,
        "model_training_run": False,
        "annotation_summary": summary,
        "artifacts": {
            "annotation_report": str(ANNOTATION_REPORT_PATH.relative_to(PROJECT_ROOT)),
            "annotation_summary": str(ANNOTATION_SUMMARY_PATH.relative_to(PROJECT_ROOT)),
            "final_report": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
            "metrics_json": str(METRICS_PATH.relative_to(PROJECT_ROOT)),
        },
    }
    METRICS_PATH.write_text(json.dumps(final_payload, indent=2, sort_keys=True), encoding="utf-8")
    REPORT_PATH.write_text(build_final_report(final_payload), encoding="utf-8")
    save_figures(summary, None)


def create_tokenization_examples(
    tokenizer: Any,
    primary: pd.DataFrame,
    max_length: int | None,
) -> dict[str, Any]:
    """Build example marked text and tokenized-marker verification."""
    example = primary.iloc[0]
    encoded = tokenize_marked_pairs(
        tokenizer,
        [str(example["marked_heavy_text"])],
        [str(example["marked_light_text"])],
        max_length,
    )
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0].tolist())
    return {
        "row_id": int(example["row_id"]),
        "marked_heavy": str(example["marked_heavy_text"]),
        "marked_light": str(example["marked_light_text"]),
        "tokenized_tokens": tokens,
        "contains_all_cdr_marker_tokens": all(token in tokens for token in CDR_MARKER_TOKENS),
    }


def main() -> int:
    """Run the final bioaware IgBert benchmark."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ANNOTATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ANNOTATION_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    require_batch_size()
    Chain, dependency_status = load_abnumber_chain()
    if Chain is None:
        write_dependency_unavailable(dependency_status)
        print(dependency_status["message"], flush=True)
        return 1

    raw = prepare_raw_table()
    nanobody_summary = summarize_nanobody_rows(raw)
    annotated, annotation_summary = annotate_paired_rows(raw, Chain)
    annotated = add_marked_inputs_and_features(annotated)
    primary, primary_summary = primary_training_data(annotated)
    annotation_summary.update(
        {
            "status": "available",
            "dependency_status": dependency_status,
            "primary_summary": primary_summary,
            "nanobody_summary": nanobody_summary,
            "annotated_output": str(ANNOTATED_PATH.relative_to(PROJECT_ROOT)),
        }
    )
    ANNOTATED_PATH.parent.mkdir(parents=True, exist_ok=True)
    annotated.to_csv(ANNOTATED_PATH, index=False)
    ANNOTATION_SUMMARY_PATH.write_text(
        json.dumps(annotation_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    ANNOTATION_REPORT_PATH.write_text(build_annotation_report(annotation_summary), encoding="utf-8")
    save_figures(annotation_summary, annotated)

    if len(primary) < 20 or primary["label"].nunique() != 2:
        final_payload = {
            "status": "unavailable",
            "reason": "primary paired all-six-CDR dataset is too small or single-label",
            "annotation_summary": annotation_summary,
            "primary_summary": primary_summary,
        }
        METRICS_PATH.write_text(json.dumps(final_payload, indent=2, sort_keys=True), encoding="utf-8")
        REPORT_PATH.write_text(build_final_report(final_payload), encoding="utf-8")
        return 1

    set_seed(RANDOM_STATE)
    split_info = grouped_train_test_split(primary)
    validation_info = inner_validation_split(primary, split_info["train_idx"], split_info["groups"])
    numeric_features, numeric_feature_info = compute_numeric_features(
        primary,
        validation_info["train_core_idx"],
    )
    subset_kmer_baselines = run_subset_kmer_baselines(primary, split_info)
    clonotype_group = add_clonotype_like_group(primary)
    clonotype_diagnostic = group_column_status(clonotype_group)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ensure_pad_token(tokenizer)
    tokenizer_info = add_cdr_marker_tokens(tokenizer)
    probe_model = AutoModel.from_pretrained(MODEL_NAME)
    probe_model.resize_token_embeddings(len(tokenizer))
    max_length = max_length_from_model(tokenizer, probe_model)
    model_class = probe_model.__class__.__name__
    del probe_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    examples = create_tokenization_examples(tokenizer, primary, max_length)
    peft_api, peft_error = load_peft_optional()
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device_name}", flush=True)
    if peft_api is None:
        print(f"PEFT unavailable; using frozen backbone head-only mode: {peft_error}", flush=True)

    results: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    for seed in TRAINING_SEEDS:
        print(f"Starting bioaware IgBert seed: {seed}", flush=True)
        try:
            result, checkpoint = train_one_seed(
                seed=seed,
                data=primary,
                split_info=split_info,
                validation_info=validation_info,
                numeric_features=numeric_features,
                numeric_feature_info=numeric_feature_info,
                tokenizer=tokenizer,
                max_length=max_length,
                marker_token_ids=tokenizer_info["marker_token_ids"],
                peft_api=peft_api,
                device_name=device_name,
            )
            results.append(result)
            if checkpoint is not None:
                checkpoints.append(checkpoint)
        except Exception as exc:
            results.append(
                {
                    "seed": int(seed),
                    "valid": False,
                    "reason": str(exc),
                }
            )
            print(f"Seed {seed} failed: {exc}", flush=True)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    aggregate = aggregate_results(results)
    comparison = build_comparison(aggregate, subset_kmer_baselines)
    best_result = best_result_by_validation(results)
    subgroup_records = subgroup_metrics(primary, best_result)
    best_seed, best_checkpoint_path = select_and_save_best_checkpoint(checkpoints)
    split_diag = ft.split_diagnostics(
        primary,
        split_info["train_idx"],
        split_info["test_idx"],
        split_info["groups"],
    )
    val_diag = ft.split_diagnostics(
        primary,
        validation_info["train_core_idx"],
        validation_info["val_idx"],
        split_info["groups"],
    )
    validation_summary = {
        **val_diag,
        "method": validation_info["method"],
        "reason": validation_info["reason"],
        "split_random_state": validation_info["split_random_state"],
    }
    training_mode = (
        results[0].get("training_mode")
        if results and results[0].get("valid")
        else ("lora_cdr_marked" if peft_api else "frozen_backbone_head_only")
    )
    payload = {
        "status": "available" if aggregate["valid_seed_count"] else "unavailable",
        "model_name": MODEL_NAME,
        "environment_variable": MODEL_ENV_VAR,
        "model_class": model_class,
        "input_path": str(INPUT_PATH.relative_to(PROJECT_ROOT)),
        "device": device_name,
        "training_mode": training_mode,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "seeds": TRAINING_SEEDS,
        "tokenizer": {
            "tokenizer_class": tokenizer.__class__.__name__,
            **tokenizer_info,
        },
        "examples": examples,
        "annotation_summary": annotation_summary,
        "primary_summary": primary_summary,
        "nanobody_summary": nanobody_summary,
        "same_subset_kmer_baselines": subset_kmer_baselines,
        "clonotype_like_grouping_diagnostic": clonotype_diagnostic,
        "split": split_diag,
        "validation_split": validation_summary,
        "numeric_feature_info": numeric_feature_info,
        "peft_available": peft_api is not None,
        "peft_error": peft_error,
        "lora_config": {
            "r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "target_module_candidates": COMMON_LORA_TARGET_NAMES,
        },
        "results": results,
        "aggregate": aggregate,
        "comparison": comparison,
        "subgroup_metrics": subgroup_records,
        "best_checkpoint_seed": best_seed,
        "best_checkpoint_path": best_checkpoint_path,
        "artifacts": {
            "annotated_csv": str(ANNOTATED_PATH.relative_to(PROJECT_ROOT)),
            "annotation_report": str(ANNOTATION_REPORT_PATH.relative_to(PROJECT_ROOT)),
            "final_report": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
            "metrics_json": str(METRICS_PATH.relative_to(PROJECT_ROOT)),
            "seed_summary_figure": str(SEED_SUMMARY_FIGURE_PATH.relative_to(PROJECT_ROOT)),
            "training_curves": str(TRAINING_CURVE_PATH.relative_to(PROJECT_ROOT)),
            "subgroup_figure": str(SUBGROUP_FIGURE_PATH.relative_to(PROJECT_ROOT)),
            "cdr_coverage_figure": str(CDR_COVERAGE_FIGURE_PATH.relative_to(PROJECT_ROOT)),
            "best_checkpoint": best_checkpoint_path,
        },
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    REPORT_PATH.write_text(build_final_report(payload), encoding="utf-8")
    save_figures(
        annotation_summary,
        annotated,
        results=results,
        aggregate=aggregate,
        baselines=subset_kmer_baselines,
        subgroups=subgroup_records,
    )

    print("\nBioaware IgBert seed metrics")
    for line in format_seed_results_table(results):
        print(line)
    print("\nAggregate")
    for line in format_aggregate_table(aggregate):
        print(line)
    print(f"\nBest checkpoint seed: {best_seed}")
    return 0 if aggregate["valid_seed_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
