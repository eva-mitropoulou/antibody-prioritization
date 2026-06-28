"""Embed existing labeled sequences with a frozen Hugging Face sequence model.

This script uses only existing rows and labels from the neutral ML table. It
does not generate, mutate, optimize, rank, or propose biological sequences.

Run from the project root:

    python src/models/embed_with_pretrained_sequence_model.py

Optionally choose a Hugging Face model:

    PRETRAINED_SEQUENCE_MODEL=Exscientia/IgBert \
        python src/models/embed_with_pretrained_sequence_model.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "neutral_sequence_classification_ml.csv"
EMBEDDING_DIR = PROJECT_ROOT / "data" / "processed" / "pretrained_embeddings"
HEAVY_EMBEDDING_PATH = EMBEDDING_DIR / "heavy.npy"
PAIR_EMBEDDING_PATH = EMBEDDING_DIR / "pair.npy"
METADATA_PATH = EMBEDDING_DIR / "metadata.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "pretrained_embedding_report.md"

DEFAULT_MODEL_NAME = "Exscientia/IgBert"
MODEL_ENV_VAR = "PRETRAINED_SEQUENCE_MODEL"
MODEL_NAME = os.environ.get(MODEL_ENV_VAR, DEFAULT_MODEL_NAME)
BATCH_SIZE = int(os.environ.get("PRETRAINED_SEQUENCE_BATCH_SIZE", "64"))
TORCH_THREADS = int(
    os.environ.get("PRETRAINED_SEQUENCE_TORCH_THREADS", str(os.cpu_count() or 1))
)
SORT_BY_LENGTH = os.environ.get("PRETRAINED_SEQUENCE_SORT_BY_LENGTH", "1").strip() != "0"

SEPARATOR_TEXT = "[SEP]"
STYLE_CANDIDATES = ("spaced", "raw")
HUGE_MODEL_MAX_LENGTH = 1_000_000


def read_input(path: Path) -> pd.DataFrame:
    """Read the neutral ML table as text while preserving blank fields."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def text_column(data: pd.DataFrame, preferred: str, fallbacks: list[str]) -> pd.Series:
    """Return the first available text column from preferred names."""
    for column in [preferred, *fallbacks]:
        if column in data.columns:
            return data[column]
    raise KeyError(f"Missing required column: {preferred}")


def optional_text_column(
    data: pd.DataFrame,
    preferred: str,
    fallbacks: list[str] | None = None,
) -> pd.Series:
    """Return an optional text column or blanks aligned to the input table."""
    for column in [preferred, *(fallbacks or [])]:
        if column in data.columns:
            return data[column]
    return pd.Series([""] * len(data), index=data.index)


def normalize_sequence_text(value: Any) -> str:
    """Normalize sequence text for tokenizer input without changing residues."""
    return re.sub(r"\s+", "", str(value or "")).upper()


def format_sequence(sequence: str, style: str) -> str:
    """Format an existing sequence for tokenizers that expect residue tokens."""
    normalized = normalize_sequence_text(sequence)
    if style == "spaced":
        return " ".join(normalized)
    if style == "raw":
        return normalized
    raise ValueError(f"Unknown tokenization style: {style}")


def split_pair_text(pair_text: str) -> tuple[str, str]:
    """Split the existing heavy-light text representation around [SEP]."""
    if SEPARATOR_TEXT not in pair_text:
        return normalize_sequence_text(pair_text), ""
    heavy, light = pair_text.split(SEPARATOR_TEXT, 1)
    return normalize_sequence_text(heavy), normalize_sequence_text(light)


def construct_pair_text(heavy: pd.Series, light: pd.Series) -> pd.Series:
    """Construct pair text from existing heavy/light columns if needed."""
    values = []
    for heavy_value, light_value in zip(heavy, light):
        heavy_text = normalize_sequence_text(heavy_value)
        light_text = normalize_sequence_text(light_value)
        if light_text:
            values.append(f"{heavy_text}{SEPARATOR_TEXT}{light_text}")
        else:
            values.append(heavy_text)
    return pd.Series(values, index=heavy.index)


def load_transformers() -> tuple[Any, Any]:
    """Import Hugging Face classes lazily so failures can be reported."""
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is not installed. Install dependencies from requirements.txt."
        ) from exc
    return AutoTokenizer, AutoModel


def load_torch() -> Any:
    """Import torch lazily so failures can be reported."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed in this environment.") from exc
    return torch


def ensure_pad_token(tokenizer: Any) -> str | None:
    """Ensure batching with padding is possible when the tokenizer supports it."""
    if tokenizer.pad_token is not None:
        return None
    for token_name in ["eos_token", "sep_token", "cls_token", "unk_token"]:
        token = getattr(tokenizer, token_name, None)
        if token is not None:
            tokenizer.pad_token = token
            return f"pad token was unset; using {token_name}={token!r}"
    raise ValueError("Tokenizer has no pad/eos/sep/cls/unk token for padded batches.")


def max_length_from_model(tokenizer: Any, model: Any) -> int | None:
    """Return a usable max length if tokenizer/model config exposes one."""
    tokenizer_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_max, int) and 0 < tokenizer_max < HUGE_MODEL_MAX_LENGTH:
        return tokenizer_max

    config_max = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if isinstance(config_max, int) and config_max > 0:
        return config_max
    return None


def tokenizer_kwargs(max_length: int | None) -> dict[str, Any]:
    """Build tokenizer kwargs for padded batches."""
    kwargs: dict[str, Any] = {
        "padding": True,
        "return_tensors": "pt",
    }
    if max_length is not None:
        kwargs.update({"truncation": True, "max_length": max_length})
    return kwargs


def tokenize_heavy(
    tokenizer: Any,
    sequences: list[str],
    style: str,
    max_length: int | None,
) -> Any:
    """Tokenize heavy-chain-only inputs."""
    texts = [format_sequence(sequence, style) for sequence in sequences]
    return tokenizer(texts, **tokenizer_kwargs(max_length))


def tokenize_pairs(
    tokenizer: Any,
    pair_texts: list[str],
    style: str,
    max_length: int | None,
) -> Any:
    """Tokenize existing heavy-light pair text as tokenizer text pairs."""
    first_sequences = []
    second_sequences = []
    for pair_text in pair_texts:
        heavy, light = split_pair_text(pair_text)
        first_sequences.append(format_sequence(heavy, style))
        second_sequences.append(format_sequence(light, style))

    if any(second_sequences):
        return tokenizer(
            first_sequences,
            text_pair=second_sequences,
            **tokenizer_kwargs(max_length),
        )
    return tokenizer(first_sequences, **tokenizer_kwargs(max_length))


def tokenization_quality(tokenizer: Any, batch: Any) -> dict[str, Any]:
    """Score a tokenization by unknown-token fraction and content length."""
    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask")
    if attention_mask is None:
        active_mask = input_ids.new_ones(input_ids.shape, dtype=bool)
    else:
        active_mask = attention_mask.bool()

    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    if unk_token_id is not None:
        special_ids.discard(int(unk_token_id))
    special_mask = input_ids.new_zeros(input_ids.shape, dtype=bool)
    for special_id in special_ids:
        special_mask |= input_ids.eq(int(special_id))

    content_mask = active_mask & ~special_mask
    content_count = int(content_mask.sum().item())
    unknown_count = 0
    if unk_token_id is not None:
        unknown_count = int((input_ids.eq(int(unk_token_id)) & content_mask).sum().item())

    unknown_fraction = unknown_count / content_count if content_count else 1.0
    return {
        "content_token_count": content_count,
        "unknown_token_count": unknown_count,
        "unknown_fraction": float(unknown_fraction),
    }


def choose_tokenization_style(
    tokenizer: Any,
    heavy_examples: list[str],
    pair_examples: list[str],
    max_length: int | None,
) -> tuple[str, list[dict[str, Any]]]:
    """Choose raw or spaced residue formatting using real example rows."""
    attempts = []
    for style in STYLE_CANDIDATES:
        try:
            heavy_batch = tokenize_heavy(tokenizer, heavy_examples, style, max_length)
            pair_batch = tokenize_pairs(tokenizer, pair_examples, style, max_length)
            heavy_quality = tokenization_quality(tokenizer, heavy_batch)
            pair_quality = tokenization_quality(tokenizer, pair_batch)
            score = (
                heavy_quality["unknown_token_count"]
                + pair_quality["unknown_token_count"]
            )
            content_count = (
                heavy_quality["content_token_count"] + pair_quality["content_token_count"]
            )
            attempts.append(
                {
                    "style": style,
                    "ok": True,
                    "score": int(score),
                    "content_token_count": int(content_count),
                    "heavy_quality": heavy_quality,
                    "pair_quality": pair_quality,
                    "error": None,
                }
            )
        except Exception as exc:
            attempts.append(
                {
                    "style": style,
                    "ok": False,
                    "score": None,
                    "content_token_count": 0,
                    "heavy_quality": None,
                    "pair_quality": None,
                    "error": str(exc),
                }
            )

    valid_attempts = [attempt for attempt in attempts if attempt["ok"]]
    if not valid_attempts:
        raise ValueError(f"No tokenization style worked: {attempts}")

    valid_attempts.sort(key=lambda item: (item["score"], -item["content_token_count"]))
    return str(valid_attempts[0]["style"]), attempts


def move_batch_to_device(batch: Any, device: Any) -> dict[str, Any]:
    """Move tensor inputs onto the selected torch device."""
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def run_model_forward(model: Any, batch: dict[str, Any]) -> Any:
    """Run a forward pass, retrying without token_type_ids if needed."""
    try:
        return model(**batch)
    except TypeError:
        if "token_type_ids" not in batch:
            raise
        reduced = {key: value for key, value in batch.items() if key != "token_type_ids"}
        return model(**reduced)


def last_hidden_state(outputs: Any) -> Any:
    """Extract the final token hidden states from a Hugging Face output."""
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is not None:
        return hidden
    if isinstance(outputs, tuple) and outputs and hasattr(outputs[0], "shape"):
        return outputs[0]
    raise ValueError("Model output did not include last_hidden_state.")


def mean_pool(hidden: Any, attention_mask: Any | None) -> Any:
    """Mean-pool token states over non-padding positions only."""
    if attention_mask is None:
        return hidden.mean(dim=1)
    mask = attention_mask.to(hidden.device).unsqueeze(-1).type_as(hidden)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def input_length(value: str, mode: str) -> int:
    """Estimate tokenizer content length for padding-efficient batching."""
    if mode == "heavy":
        return len(normalize_sequence_text(value))
    if mode == "pair":
        heavy, light = split_pair_text(value)
        return len(heavy) + len(light)
    raise ValueError(f"Unknown embedding mode: {mode}")


def embed_batches(
    model: Any,
    tokenizer: Any,
    values: list[str],
    mode: str,
    style: str,
    max_length: int | None,
    device: Any,
    torch: Any,
) -> np.ndarray:
    """Create frozen mean-pooled embeddings in batches."""
    if SORT_BY_LENGTH:
        order = sorted(range(len(values)), key=lambda index: input_length(values[index], mode))
    else:
        order = list(range(len(values)))

    chunks: list[tuple[list[int], np.ndarray]] = []
    for start in tqdm(range(0, len(order), BATCH_SIZE), desc=f"Embedding {mode}"):
        stop = min(start + BATCH_SIZE, len(order))
        batch_indices = order[start:stop]
        batch_values = [values[index] for index in batch_indices]
        if mode == "heavy":
            batch = tokenize_heavy(tokenizer, batch_values, style, max_length)
        elif mode == "pair":
            batch = tokenize_pairs(tokenizer, batch_values, style, max_length)
        else:
            raise ValueError(f"Unknown embedding mode: {mode}")

        batch = move_batch_to_device(batch, device)
        with torch.no_grad():
            outputs = run_model_forward(model, batch)
            hidden = last_hidden_state(outputs)
            pooled = mean_pool(hidden, batch.get("attention_mask"))
        chunks.append((batch_indices, pooled.detach().cpu().numpy().astype(np.float32)))

    if not chunks:
        return np.empty((0, 0), dtype=np.float32)

    embedding_dim = int(chunks[0][1].shape[1])
    output = np.empty((len(values), embedding_dim), dtype=np.float32)
    for batch_indices, batch_embeddings in chunks:
        output[np.asarray(batch_indices, dtype=int)] = batch_embeddings
    return output




def can_embed_pair(
    model: Any,
    tokenizer: Any,
    pair_values: list[str],
    style: str,
    max_length: int | None,
    device: Any,
    torch: Any,
) -> tuple[bool, str | None]:
    """Check whether pair tokenization and one tiny forward pass work."""
    try:
        sample_values = pair_values[: min(4, len(pair_values))]
        if not sample_values:
            return False, "no sequence_pair_text values"
        batch = tokenize_pairs(tokenizer, sample_values, style, max_length)
        batch = move_batch_to_device(batch, device)
        with torch.no_grad():
            outputs = run_model_forward(model, batch)
            _ = last_hidden_state(outputs)
        return True, None
    except Exception as exc:
        return False, str(exc)


def bool_text(value: bool) -> str:
    """Return CSV-friendly boolean text."""
    return "True" if value else "False"


def build_metadata(
    data: pd.DataFrame,
    heavy: pd.Series,
    pair: pd.Series,
    model_name: str,
    tokenizer: Any,
    model: Any,
    tokenization_style: str,
    pair_embedding_available: bool,
    pair_skip_reason: str | None,
) -> pd.DataFrame:
    """Build row metadata aligned to cached embedding arrays."""
    label = text_column(data, "label", [])
    group_feature_v = optional_text_column(data, "group_feature_v")
    sample_name = optional_text_column(data, "sample_name", ["antibody_name"])
    sample_type = optional_text_column(data, "sample_type")
    light = optional_text_column(data, "sequence_b", ["sequence_light_only", "light_sequence"])

    parsed_light = pair.map(lambda value: split_pair_text(str(value))[1])
    has_light = light.map(normalize_sequence_text).ne("") | parsed_light.ne("")
    if "has_light" in data.columns:
        has_light = data["has_light"].astype(str).str.lower().isin({"true", "1", "yes"})

    if "is_nanobody_like" in data.columns:
        is_nanobody_like = data["is_nanobody_like"].astype(str).str.lower().isin(
            {"true", "1", "yes"}
        )
    else:
        type_is_nb = sample_type.astype(str).str.strip().str.upper().isin(
            {"NB", "NANOBODY"}
        )
        is_nanobody_like = type_is_nb | ~has_light

    return pd.DataFrame(
        {
            "row_id": np.arange(len(data), dtype=int),
            "sample_name": sample_name.fillna("").astype(str),
            "sequence_heavy_only": heavy.fillna("").astype(str),
            "sequence_pair_text": pair.fillna("").astype(str),
            "label": label.fillna("").astype(str),
            "group_feature_v": group_feature_v.fillna("").astype(str),
            "has_light": [bool_text(bool(value)) for value in has_light],
            "is_nanobody_like": [bool_text(bool(value)) for value in is_nanobody_like],
            "pretrained_model_name": model_name,
            "tokenizer_class": tokenizer.__class__.__name__,
            "model_class": model.__class__.__name__,
            "tokenization_style": tokenization_style,
            "pair_embedding_available": bool_text(pair_embedding_available),
            "pair_skip_reason": pair_skip_reason or "",
        }
    )


def build_report(payload: dict[str, Any]) -> str:
    """Build the Markdown embedding report."""
    lines = [
        "# Pretrained Sequence Embedding Report",
        "",
        f"status: `{payload['status']}`",
        f"model_name: `{payload['model_name']}`",
        "",
    ]
    if payload["status"] != "available":
        lines.extend(
            [
                f"stage: `{payload.get('stage', 'unknown')}`",
                f"reason: `{payload.get('error', 'unknown')}`",
                "",
                "No random or placeholder embeddings were created.",
                "",
            ]
        )
        return "\n".join(lines)

    lines.extend(
        [
            "This report describes frozen embeddings generated from existing labeled",
            "rows only. Model parameters were not updated.",
            "",
            "## Model",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| Tokenizer class | `{payload['tokenizer_class']}` |",
            f"| Model class | `{payload['model_class']}` |",
            f"| Device | `{payload['device']}` |",
            f"| Batch size | `{payload['batch_size']}` |",
            f"| Torch CPU threads | `{payload.get('torch_threads', 'n/a')}` |",
            f"| Sort batches by length | `{payload.get('sort_by_length', 'n/a')}` |",
            f"| Tokenization style | `{payload['tokenization_style']}` |",
            f"| Max sequence length | `{payload['max_sequence_length']}` |",
            f"| Pair embeddings available | `{payload['pair_embedding_available']}` |",
            f"| Pair skip reason | `{payload.get('pair_skip_reason') or 'n/a'}` |",
            "",
            "## Tokenization Probe",
            "",
            "```json",
            json.dumps(payload["tokenization_attempts"], indent=2),
            "```",
            "",
            "## Outputs",
            "",
            "| Artifact | Shape / Status |",
            "|---|---|",
            f"| `{HEAVY_EMBEDDING_PATH.relative_to(PROJECT_ROOT)}` | `{payload['heavy_embedding_shape']}` |",
            f"| `{PAIR_EMBEDDING_PATH.relative_to(PROJECT_ROOT)}` | `{payload.get('pair_embedding_shape', 'not created')}` |",
            f"| `{METADATA_PATH.relative_to(PROJECT_ROOT)}` | `{payload['metadata_shape']}` |",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(payload: dict[str, Any]) -> None:
    """Persist the embedding Markdown report."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(payload), encoding="utf-8")


def failure_payload(stage: str, error: Exception) -> dict[str, Any]:
    """Create a stable failure payload."""
    return {
        "status": "unavailable",
        "stage": stage,
        "error": str(error),
        "model_name": MODEL_NAME,
        "environment_variable": MODEL_ENV_VAR,
        "input_path": str(INPUT_PATH.relative_to(PROJECT_ROOT)),
        "embedding_dir": str(EMBEDDING_DIR.relative_to(PROJECT_ROOT)),
    }


def main() -> int:
    """Generate frozen pretrained sequence embeddings."""
    try:
        torch = load_torch()
        AutoTokenizer, AutoModel = load_transformers()
    except Exception as exc:
        payload = failure_payload("dependency_import", exc)
        write_report(payload)
        print(f"pretrained embedding unavailable: {exc}")
        return 1

    try:
        data = read_input(INPUT_PATH)
        heavy = text_column(data, "sequence_heavy_only", ["sequence_a"]).map(
            normalize_sequence_text
        )
        light = optional_text_column(data, "sequence_b", ["sequence_light_only", "light_sequence"])
        if "sequence_pair_text" in data.columns:
            pair = data["sequence_pair_text"].fillna("").astype(str)
        else:
            pair = construct_pair_text(heavy, light)
    except Exception as exc:
        payload = failure_payload("input_loading", exc)
        write_report(payload)
        print(f"pretrained embedding unavailable: {exc}")
        return 1

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
        pad_note = ensure_pad_token(tokenizer)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cpu":
            torch.set_num_threads(max(1, TORCH_THREADS))
            torch.set_num_interop_threads(1)
        model.to(device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        max_length = max_length_from_model(tokenizer, model)
        style, attempts = choose_tokenization_style(
            tokenizer=tokenizer,
            heavy_examples=heavy.head(5).tolist(),
            pair_examples=pair.head(5).tolist(),
            max_length=max_length,
        )

        pair_available, pair_skip_reason = can_embed_pair(
            model=model,
            tokenizer=tokenizer,
            pair_values=pair.tolist(),
            style=style,
            max_length=max_length,
            device=device,
            torch=torch,
        )

        heavy_embeddings = embed_batches(
            model=model,
            tokenizer=tokenizer,
            values=heavy.tolist(),
            mode="heavy",
            style=style,
            max_length=max_length,
            device=device,
            torch=torch,
        )

        pair_embeddings = None
        if pair_available:
            pair_embeddings = embed_batches(
                model=model,
                tokenizer=tokenizer,
                values=pair.tolist(),
                mode="pair",
                style=style,
                max_length=max_length,
                device=device,
                torch=torch,
            )

        metadata = build_metadata(
            data=data,
            heavy=heavy,
            pair=pair,
            model_name=MODEL_NAME,
            tokenizer=tokenizer,
            model=model,
            tokenization_style=style,
            pair_embedding_available=pair_available,
            pair_skip_reason=pair_skip_reason,
        )

        EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
        np.save(HEAVY_EMBEDDING_PATH, heavy_embeddings.astype(np.float32))
        if pair_embeddings is not None:
            np.save(PAIR_EMBEDDING_PATH, pair_embeddings.astype(np.float32))
        metadata.to_csv(METADATA_PATH, index=False)

        payload = {
            "status": "available",
            "model_name": MODEL_NAME,
            "environment_variable": MODEL_ENV_VAR,
            "input_path": str(INPUT_PATH.relative_to(PROJECT_ROOT)),
            "tokenizer_class": tokenizer.__class__.__name__,
            "model_class": model.__class__.__name__,
            "device": str(device),
            "batch_size": BATCH_SIZE,
            "torch_threads": torch.get_num_threads(),
            "sort_by_length": SORT_BY_LENGTH,
            "tokenization_style": style,
            "tokenization_attempts": attempts,
            "max_sequence_length": max_length,
            "pad_token_note": pad_note,
            "rows": int(len(metadata)),
            "heavy_embedding_shape": list(heavy_embeddings.shape),
            "pair_embedding_available": pair_available,
            "pair_skip_reason": pair_skip_reason,
            "pair_embedding_shape": (
                list(pair_embeddings.shape) if pair_embeddings is not None else None
            ),
            "metadata_shape": list(metadata.shape),
            "artifacts": {
                "heavy_embeddings": str(HEAVY_EMBEDDING_PATH.relative_to(PROJECT_ROOT)),
                "pair_embeddings": (
                    str(PAIR_EMBEDDING_PATH.relative_to(PROJECT_ROOT))
                    if pair_embeddings is not None
                    else None
                ),
                "metadata": str(METADATA_PATH.relative_to(PROJECT_ROOT)),
                "report": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
            },
        }
        write_report(payload)

        print(f"model: {MODEL_NAME}")
        print(f"device: {device}")
        print(f"tokenization style: {style}")
        print(f"heavy embedding shape: {heavy_embeddings.shape}")
        if pair_embeddings is not None:
            print(f"pair embedding shape: {pair_embeddings.shape}")
        else:
            print(f"pair embeddings unavailable: {pair_skip_reason}")
        print(f"Wrote {HEAVY_EMBEDDING_PATH.relative_to(PROJECT_ROOT)}")
        if pair_embeddings is not None:
            print(f"Wrote {PAIR_EMBEDDING_PATH.relative_to(PROJECT_ROOT)}")
        print(f"Wrote {METADATA_PATH.relative_to(PROJECT_ROOT)}")
        print(f"Wrote {REPORT_PATH.relative_to(PROJECT_ROOT)}")
        return 0
    except Exception as exc:
        payload = failure_payload("embedding", exc)
        write_report(payload)
        print(f"pretrained embedding unavailable: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
