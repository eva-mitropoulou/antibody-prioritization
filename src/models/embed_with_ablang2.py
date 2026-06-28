"""Embed existing labeled sequences with pretrained AbLang2.

This script uses only existing rows from the neutral input table. It does not
generate, design, mutate, optimize, rank, or propose biological sequences.

Run from the project root:

    python src/models/embed_with_ablang2.py
"""

from __future__ import annotations

import importlib.metadata
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "neutral_sequence_classification_ml.csv"
HEAVY_EMBEDDING_PATH = PROJECT_ROOT / "data" / "processed" / "embeddings_ablang2_heavy.npy"
PAIR_EMBEDDING_PATH = PROJECT_ROOT / "data" / "processed" / "embeddings_ablang2_pair.npy"
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "embeddings_ablang2_metadata.csv"
STATUS_PATH = PROJECT_ROOT / "reports" / "embedding_dependency_status.md"

BATCH_SIZE = 128
MODEL_NAME = "ablang2-paired"
DEVICE = "cpu"
VALID_SEQUENCE_RE = re.compile(r"^[A-Z*]+$")


def read_csv(path: Path) -> pd.DataFrame:
    """Read the neutral CSV as text while preserving blank fields as blanks."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def text_column(data: pd.DataFrame, preferred: str, fallback: str | None = None) -> pd.Series:
    """Return a preferred neutral column with a legacy fallback if needed."""
    if preferred in data.columns:
        return data[preferred]
    if fallback and fallback in data.columns:
        return data[fallback]
    raise KeyError(f"Missing required column: {preferred}")


def optional_text_column(
    data: pd.DataFrame,
    preferred: str,
    fallback: str | None = None,
) -> pd.Series:
    """Return an optional text column or blanks aligned to the input table."""
    if preferred in data.columns:
        return data[preferred]
    if fallback and fallback in data.columns:
        return data[fallback]
    return pd.Series([""] * len(data), index=data.index)


def normalize_sequence_text(values: pd.Series) -> pd.Series:
    """Normalize existing sequence strings for model input."""
    return (
        values.fillna("")
        .astype(str)
        .str.replace(r"\s+", "", regex=True)
        .str.upper()
    )


def validate_sequences(heavy: pd.Series, light: pd.Series) -> None:
    """Fail clearly if rows cannot be sent to the pretrained tokenizer."""
    empty_heavy_count = int(heavy.eq("").sum())
    invalid_heavy_count = int((~heavy.map(is_valid_sequence)).sum())
    invalid_light_count = int((light.ne("") & ~light.map(is_valid_sequence)).sum())

    errors = []
    if empty_heavy_count:
        errors.append(f"{empty_heavy_count} rows have empty sequence_a")
    if invalid_heavy_count:
        errors.append(f"{invalid_heavy_count} rows have unsupported sequence_a tokens")
    if invalid_light_count:
        errors.append(f"{invalid_light_count} rows have unsupported sequence_b tokens")
    if errors:
        raise ValueError("; ".join(errors))


def is_valid_sequence(value: str) -> bool:
    """Check whether a sequence contains tokenizer-compatible symbols."""
    return bool(value) and VALID_SEQUENCE_RE.match(value) is not None


def write_status(lines: list[str]) -> None:
    """Write the embedding dependency/status report."""
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def dependency_failure_status(message: str) -> None:
    """Print and persist a clear dependency failure."""
    lines = [
        "# Embedding Dependency Status",
        "",
        "status: unavailable",
        f"reason: {message}",
        "",
        "Install help:",
        "",
        "```bash",
        "python -m pip install ablang2",
        "```",
        "",
        "No random or placeholder embeddings were created.",
    ]
    write_status(lines)
    print("embedding dependency status: unavailable")
    print(message)
    print(f"Wrote {STATUS_PATH.relative_to(PROJECT_ROOT)}")


def success_status(payload: dict[str, Any]) -> None:
    """Write a successful embedding status report."""
    lines = [
        "# Embedding Dependency Status",
        "",
        "status: available",
        f"model: {payload['model_name']}",
        f"device: {payload['device']}",
        f"ablang2_version: {payload['ablang2_version']}",
        f"torch_version: {payload['torch_version']}",
        f"rows: {payload['rows']}",
        f"heavy_embedding_shape: {payload['heavy_embedding_shape']}",
        f"pair_embedding_shape: {payload['pair_embedding_shape']}",
        "",
        "Artifacts:",
        "",
        f"- `{HEAVY_EMBEDDING_PATH.relative_to(PROJECT_ROOT)}`",
        f"- `{PAIR_EMBEDDING_PATH.relative_to(PROJECT_ROOT)}`",
        f"- `{METADATA_PATH.relative_to(PROJECT_ROOT)}`",
    ]
    write_status(lines)


def import_ablang2() -> tuple[Any, str, str]:
    """Import AbLang2 and torch, returning modules plus versions."""
    try:
        import ablang2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("AbLang2 is not installed in this environment.") from exc

    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed in this environment.") from exc

    return (
        ablang2,
        importlib.metadata.version("ablang2"),
        str(torch.__version__),
    )


def load_ablang2_model() -> tuple[Any, str, str]:
    """Load the pretrained AbLang2 paired model."""
    ablang2, ablang2_version, torch_version = import_ablang2()
    model = ablang2.pretrained(model_to_use=MODEL_NAME, device=DEVICE)
    return model, ablang2_version, torch_version


def as_2d_array(values: Any) -> np.ndarray:
    """Coerce AbLang2 output to a two-dimensional numpy array."""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D embedding array, got shape {array.shape}")
    return array


def embed_pairs(
    model: Any,
    heavy_values: list[str],
    light_values: list[str],
    description: str,
) -> np.ndarray:
    """Embed batches of heavy/light pairs with AbLang2 seqcoding."""
    chunks: list[np.ndarray] = []
    row_count = len(heavy_values)

    for start in tqdm(range(0, row_count, BATCH_SIZE), desc=description):
        stop = min(start + BATCH_SIZE, row_count)
        batch_pairs = list(zip(heavy_values[start:stop], light_values[start:stop]))
        encoded = model(batch_pairs, mode="seqcoding", batch_size=len(batch_pairs))
        chunks.append(as_2d_array(encoded))

    return np.vstack(chunks).astype(np.float32)


def build_metadata(data: pd.DataFrame, heavy: pd.Series, light: pd.Series) -> pd.DataFrame:
    """Build row metadata aligned to the cached embeddings."""
    label = text_column(data, "label")
    group_feature_v = optional_text_column(data, "group_feature_v")
    sample_name = optional_text_column(data, "sample_name", fallback="antibody_name")
    has_light = light.ne("")

    return pd.DataFrame(
        {
            "row_id": np.arange(len(data), dtype=int),
            "antibody_name": sample_name.fillna("").astype(str),
            "label": label.fillna("").astype(str),
            "group_feature_v": group_feature_v.fillna("").astype(str),
            "has_light": has_light.astype(bool),
            "is_nanobody_like": (~has_light).astype(bool),
        }
    )


def main() -> int:
    """Create and cache AbLang2 embeddings."""
    try:
        model, ablang2_version, torch_version = load_ablang2_model()
    except Exception as exc:  # dependency/download errors need a clear report
        dependency_failure_status(str(exc))
        return 1

    data = read_csv(INPUT_PATH)
    heavy = normalize_sequence_text(text_column(data, "sequence_a", "sequence_heavy_only"))
    light = normalize_sequence_text(optional_text_column(data, "sequence_b", "light_sequence"))
    validate_sequences(heavy, light)

    heavy_values = heavy.tolist()
    blank_light_values = [""] * len(heavy_values)
    light_values = light.tolist()
    has_light = light.ne("").to_numpy()

    heavy_embeddings = embed_pairs(
        model=model,
        heavy_values=heavy_values,
        light_values=blank_light_values,
        description="Embedding heavy sequences",
    )

    light_embeddings = np.zeros_like(heavy_embeddings, dtype=np.float32)
    if bool(has_light.any()):
        light_indices = np.flatnonzero(has_light)
        light_only_heavy = [""] * len(light_indices)
        light_only_light = [light_values[index] for index in light_indices]
        encoded_light = embed_pairs(
            model=model,
            heavy_values=light_only_heavy,
            light_values=light_only_light,
            description="Embedding light sequences",
        )
        light_embeddings[light_indices] = encoded_light

    pair_embeddings = np.hstack([heavy_embeddings, light_embeddings]).astype(np.float32)
    metadata = build_metadata(data, heavy, light)

    HEAVY_EMBEDDING_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(HEAVY_EMBEDDING_PATH, heavy_embeddings)
    np.save(PAIR_EMBEDDING_PATH, pair_embeddings)
    metadata.to_csv(METADATA_PATH, index=False)

    payload = {
        "model_name": MODEL_NAME,
        "device": DEVICE,
        "ablang2_version": ablang2_version,
        "torch_version": torch_version,
        "rows": int(len(data)),
        "heavy_embedding_shape": list(heavy_embeddings.shape),
        "pair_embedding_shape": list(pair_embeddings.shape),
    }
    success_status(payload)

    print("embedding dependency status: available")
    print(f"heavy embedding shape: {heavy_embeddings.shape}")
    print(f"pair embedding shape: {pair_embeddings.shape}")
    print(f"Wrote {HEAVY_EMBEDDING_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {PAIR_EMBEDDING_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {METADATA_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {STATUS_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
