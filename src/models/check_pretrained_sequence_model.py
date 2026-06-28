"""Check a frozen Hugging Face sequence model on existing antibody rows.

This script only reads existing labeled rows. It does not generate, mutate,
optimize, rank, or propose biological sequences.

Run from the project root:

    python src/models/check_pretrained_sequence_model.py

Optionally choose a Hugging Face model:

    PRETRAINED_SEQUENCE_MODEL=Exscientia/IgBert \
        python src/models/check_pretrained_sequence_model.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "neutral_sequence_classification_ml.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "pretrained_sequence_model_check.md"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics" / "pretrained_sequence_model_check.json"

DEFAULT_MODEL_NAME = "Exscientia/IgBert"
MODEL_ENV_VAR = "PRETRAINED_SEQUENCE_MODEL"
MODEL_NAME = os.environ.get(MODEL_ENV_VAR, DEFAULT_MODEL_NAME)

EXAMPLE_ROWS = 5
TOKEN_PREVIEW = 80
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


def output_shapes(outputs: Any) -> dict[str, list[int]]:
    """Collect tensor shapes from common Hugging Face model outputs."""
    shapes: dict[str, list[int]] = {}
    for name in ["last_hidden_state", "pooler_output"]:
        value = getattr(outputs, name, None)
        if value is not None and hasattr(value, "shape"):
            shapes[name] = [int(dim) for dim in value.shape]
    if isinstance(outputs, tuple):
        for index, value in enumerate(outputs):
            if hasattr(value, "shape"):
                shapes[f"tuple_{index}"] = [int(dim) for dim in value.shape]
    return shapes


def summarize_tokenization(tokenizer: Any, batch: Any, row_index: int = 0) -> dict[str, Any]:
    """Return a compact tokenization preview for Markdown and JSON."""
    input_ids = batch["input_ids"][row_index].detach().cpu().tolist()
    attention = batch.get("attention_mask")
    if attention is None:
        attention_values = [1] * len(input_ids)
    else:
        attention_values = attention[row_index].detach().cpu().tolist()

    active_token_count = int(sum(int(value) for value in attention_values))
    preview_ids = input_ids[:TOKEN_PREVIEW]
    preview_attention = attention_values[:TOKEN_PREVIEW]
    preview_tokens = tokenizer.convert_ids_to_tokens(preview_ids)
    return {
        "active_token_count": active_token_count,
        "sequence_length_with_padding": int(len(input_ids)),
        "input_ids_preview": [int(value) for value in preview_ids],
        "tokens_preview": [str(value) for value in preview_tokens],
        "attention_mask_preview": [int(value) for value in preview_attention],
        "preview_truncated": len(input_ids) > TOKEN_PREVIEW,
    }


def build_report(payload: dict[str, Any]) -> str:
    """Build the Markdown model-check report."""
    lines = [
        "# Pretrained Sequence Model Check",
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
                "No placeholder outputs were created.",
                "",
            ]
        )
        return "\n".join(lines)

    lines.extend(
        [
            "## Loaded Components",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| Tokenizer class | `{payload['tokenizer_class']}` |",
            f"| Model class | `{payload['model_class']}` |",
            f"| Vocabulary size | `{payload['vocabulary_size']}` |",
            f"| Max sequence length | `{payload['max_sequence_length']}` |",
            f"| Pad token | `{payload['pad_token']}` |",
            f"| Separator token | `{payload['separator_token']}` |",
            f"| Tokenization style | `{payload['tokenization_style']}` |",
            f"| Device | `{payload['device']}` |",
            "",
            "## Tokenization Probe",
            "",
            "The script tried raw amino-acid strings and space-separated residue tokens,",
            "then selected the style with fewer unknown content tokens.",
            "",
            "```json",
            json.dumps(payload["tokenization_attempts"], indent=2),
            "```",
            "",
            "## Example Tokenization",
            "",
            "### sequence_heavy_only",
            "",
            "```json",
            json.dumps(payload["example_tokenization"]["sequence_heavy_only"], indent=2),
            "```",
            "",
            "### sequence_pair_text",
            "",
            "```json",
            json.dumps(payload["example_tokenization"]["sequence_pair_text"], indent=2),
            "```",
            "",
            "## Forward Pass",
            "",
            f"Tiny batch size: `{payload['forward_pass']['batch_size']}`",
            "",
            "Output tensor shapes:",
            "",
            "```json",
            json.dumps(payload["forward_pass"]["output_shapes"], indent=2),
            "```",
            "",
            "Artifacts:",
            "",
            f"- `{REPORT_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{METRICS_PATH.relative_to(PROJECT_ROOT)}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_payload(payload: dict[str, Any]) -> None:
    """Persist JSON and Markdown check outputs."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
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
        "artifacts": {
            "report": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
            "metrics_json": str(METRICS_PATH.relative_to(PROJECT_ROOT)),
        },
    }


def main() -> int:
    """Run the model/tokenizer availability check."""
    try:
        torch = load_torch()
        AutoTokenizer, AutoModel = load_transformers()
    except Exception as exc:
        payload = failure_payload("dependency_import", exc)
        write_payload(payload)
        print(f"pretrained model check unavailable: {exc}")
        return 1

    try:
        data = read_input(INPUT_PATH).head(EXAMPLE_ROWS).copy()
        heavy = text_column(data, "sequence_heavy_only", ["sequence_a"]).map(
            normalize_sequence_text
        )
        pair = text_column(data, "sequence_pair_text", []).astype(str)
    except Exception as exc:
        payload = failure_payload("input_loading", exc)
        write_payload(payload)
        print(f"pretrained model check unavailable: {exc}")
        return 1

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
        pad_note = ensure_pad_token(tokenizer)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    except Exception as exc:
        payload = failure_payload("model_loading", exc)
        write_payload(payload)
        print(f"pretrained model check unavailable: {exc}")
        return 1

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        max_length = max_length_from_model(tokenizer, model)
        style, attempts = choose_tokenization_style(
            tokenizer=tokenizer,
            heavy_examples=heavy.tolist(),
            pair_examples=pair.tolist(),
            max_length=max_length,
        )

        heavy_batch = tokenize_heavy(tokenizer, heavy.tolist(), style, max_length)
        pair_batch = tokenize_pairs(tokenizer, pair.tolist(), style, max_length)
        tiny_batch = tokenize_heavy(tokenizer, heavy.tolist()[:2], style, max_length)

        with torch.no_grad():
            outputs = run_model_forward(model, move_batch_to_device(tiny_batch, device))

        payload = {
            "status": "available",
            "model_name": MODEL_NAME,
            "environment_variable": MODEL_ENV_VAR,
            "input_path": str(INPUT_PATH.relative_to(PROJECT_ROOT)),
            "example_row_count": int(len(data)),
            "tokenizer_class": tokenizer.__class__.__name__,
            "model_class": model.__class__.__name__,
            "vocabulary_size": int(len(tokenizer)),
            "max_sequence_length": max_length,
            "raw_tokenizer_model_max_length": getattr(tokenizer, "model_max_length", None),
            "pad_token": tokenizer.pad_token,
            "pad_token_note": pad_note,
            "separator_token": tokenizer.sep_token,
            "tokenization_style": style,
            "tokenization_attempts": attempts,
            "device": str(device),
            "example_tokenization": {
                "sequence_heavy_only": summarize_tokenization(tokenizer, heavy_batch),
                "sequence_pair_text": summarize_tokenization(tokenizer, pair_batch),
            },
            "forward_pass": {
                "batch_size": 2,
                "output_shapes": output_shapes(outputs),
            },
            "artifacts": {
                "report": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
                "metrics_json": str(METRICS_PATH.relative_to(PROJECT_ROOT)),
            },
        }
        write_payload(payload)
        print(f"model: {MODEL_NAME}")
        print(f"tokenizer: {payload['tokenizer_class']}")
        print(f"model class: {payload['model_class']}")
        print(f"tokenization style: {style}")
        print(f"output shapes: {payload['forward_pass']['output_shapes']}")
        print(f"Wrote {REPORT_PATH.relative_to(PROJECT_ROOT)}")
        print(f"Wrote {METRICS_PATH.relative_to(PROJECT_ROOT)}")
        return 0
    except Exception as exc:
        payload = failure_payload("tokenization_or_forward", exc)
        write_payload(payload)
        print(f"pretrained model check unavailable: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
