# Pretrained Sequence Embedding Report

status: `available`
model_name: `Exscientia/IgBert`

This report describes frozen embeddings generated from existing labeled
rows only. Model parameters were not updated.

## Model

| Field | Value |
|---|---|
| Tokenizer class | `BertTokenizer` |
| Model class | `BertModel` |
| Device | `cpu` |
| Batch size | `64` |
| Torch CPU threads | `12` |
| Sort batches by length | `True` |
| Tokenization style | `spaced` |
| Max sequence length | `40000` |
| Pair embeddings available | `True` |
| Pair skip reason | `n/a` |

## Tokenization Probe

```json
[
  {
    "style": "spaced",
    "ok": true,
    "score": 0,
    "content_token_count": 1840,
    "heavy_quality": {
      "content_token_count": 648,
      "unknown_token_count": 0,
      "unknown_fraction": 0.0
    },
    "pair_quality": {
      "content_token_count": 1192,
      "unknown_token_count": 0,
      "unknown_fraction": 0.0
    },
    "error": null
  },
  {
    "style": "raw",
    "ok": true,
    "score": 15,
    "content_token_count": 15,
    "heavy_quality": {
      "content_token_count": 5,
      "unknown_token_count": 5,
      "unknown_fraction": 1.0
    },
    "pair_quality": {
      "content_token_count": 10,
      "unknown_token_count": 10,
      "unknown_fraction": 1.0
    },
    "error": null
  }
]
```

## Outputs

| Artifact | Shape / Status |
|---|---|
| `data/processed/pretrained_embeddings/heavy.npy` | `[5573, 1024]` |
| `data/processed/pretrained_embeddings/pair.npy` | `[5573, 1024]` |
| `data/processed/pretrained_embeddings/metadata.csv` | `[5573, 14]` |
