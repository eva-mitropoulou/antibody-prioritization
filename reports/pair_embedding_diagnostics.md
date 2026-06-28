# Pair Embedding Diagnostics

Question: are heavy-light pair embeddings true joint paired embeddings, or
concatenated heavy-only plus light-only embeddings?

Short answer:

- AbLang2 `pair` embeddings are concatenated heavy-only plus light-only
  embeddings.
- Hugging Face / IgBert `pair` embeddings are true joint text-pair embeddings
  from one tokenized heavy/light forward pass.

## Summary Table

| Model | Input variant | Embedding construction | Shape | Joint or concatenated? |
| ----- | ------------- | ---------------------- | ----- | ---------------------- |
| AbLang2 | heavy | Heavy sequence embedded with blank light chain via `embed_pairs(model, heavy_values, blank_light_values, ...)` | `(5573, 480)` | Heavy-only |
| AbLang2 | pair | Heavy-only embedding horizontally stacked with separately computed light-only embedding via `np.hstack([heavy_embeddings, light_embeddings])` | `(5573, 960)` | Concatenated |
| Hugging Face / IgBert | heavy | Heavy sequence tokenized alone, passed through `AutoModel`, attention-mask mean pooled | `(5573, 1024)` | Heavy-only |
| Hugging Face / IgBert | pair | Existing pair text split into heavy and light, tokenized as `text_pair`, passed through one `AutoModel` forward, attention-mask mean pooled | `(5573, 1024)` | Joint |

## AbLang2

Relevant file: `src/models/embed_with_ablang2.py`

The shared embedding helper sends a list of `(heavy, light)` tuples into the
AbLang2 model:

```python
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
```

Heavy-only embeddings are created by passing real heavy sequences and blank
light strings:

```python
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
```

Light-chain embeddings are created separately by passing blank heavy strings
and real light sequences only for rows with light chains:

```python
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
```

The pair embedding is then a horizontal concatenation:

```python
pair_embeddings = np.hstack([heavy_embeddings, light_embeddings]).astype(np.float32)
```

The script saves those arrays directly:

```python
np.save(HEAVY_EMBEDDING_PATH, heavy_embeddings)
np.save(PAIR_EMBEDDING_PATH, pair_embeddings)
```

Conclusion for AbLang2: despite the helper function accepting `(heavy, light)`
tuples, this script creates AbLang2 `pair` arrays by concatenating separately
embedded heavy and light chains. The saved AbLang2 `pair` array is
concatenated heavy-only plus light-only.

Saved shapes:

- `data/processed/embeddings_ablang2_heavy.npy`: `(5573, 480)`
- `data/processed/embeddings_ablang2_pair.npy`: `(5573, 960)`

## Hugging Face / IgBert

Relevant files:

- `src/models/embed_with_pretrained_sequence_model.py`
- `src/models/check_pretrained_sequence_model.py`

Heavy-only tokenization sends each heavy sequence as a single text input:

```python
def tokenize_heavy(
    tokenizer: Any,
    sequences: list[str],
    style: str,
    max_length: int | None,
) -> Any:
    """Tokenize heavy-chain-only inputs."""
    texts = [format_sequence(sequence, style) for sequence in sequences]
    return tokenizer(texts, **tokenizer_kwargs(max_length))
```

Pair tokenization splits the existing `sequence_pair_text` around `[SEP]` and
passes heavy and light as a Hugging Face `text_pair` in one tokenizer call:

```python
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
```

Both heavy and pair modes are passed through the same frozen model forward and
attention-mask-aware mean pooling:

```python
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
```

The main embedding script creates heavy embeddings from `mode="heavy"` and
pair embeddings from `mode="pair"`:

```python
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
```

The checker uses the same pair-tokenization path:

```python
pair_batch = tokenize_pairs(tokenizer, pair_examples, style, max_length)
```

Conclusion for Hugging Face / IgBert: the saved `pair` array is not a
concatenation of separate heavy and light embeddings. It is one pooled model
representation from a single tokenized heavy/light text-pair input. The
tokenizer inserts the model separator tokens through the `text_pair` API.

Saved shapes:

- `data/processed/pretrained_embeddings/heavy.npy`: `(5573, 1024)`
- `data/processed/pretrained_embeddings/pair.npy`: `(5573, 1024)`

## What The Classifiers Receive

### AbLang2 classifiers

The logistic-regression baseline loads the cached heavy and pair matrices:

```python
embeddings = {"heavy": np.load(HEAVY_EMBEDDING_PATH)}

if PAIR_EMBEDDING_PATH.exists():
    embeddings["pair"] = np.load(PAIR_EMBEDDING_PATH)
```

The PyTorch MLP baseline also loads the cached heavy and pair matrices:

```python
embeddings = {
    "heavy": np.load(HEAVY_EMBEDDING_PATH).astype(np.float32),
    "pair": np.load(PAIR_EMBEDDING_PATH).astype(np.float32),
}
```

Therefore:

- heavy-only classifier input: one 480-dimensional AbLang2 heavy-only vector.
- pair classifier input: one 960-dimensional concatenated vector:
  `[heavy-only embedding, light-only embedding]`.
- classifier interaction capacity: the pretrained AbLang2 representation itself
  represents separately embedded heavy and light chains in the saved `pair`
  array. A downstream MLP can learn
  interactions after concatenation; logistic regression cannot learn nonlinear
  cross-chain interactions unless they are already encoded or explicitly
  featurized.

### Hugging Face / IgBert classifiers

The frozen pretrained baseline loads the cached heavy matrix and optionally the
cached pair matrix:

```python
embeddings: dict[str, np.ndarray] = {
    "heavy": np.load(HEAVY_EMBEDDING_PATH).astype(np.float32)
}

if PAIR_EMBEDDING_PATH.exists() and pair_metadata_available:
    embeddings["pair"] = np.load(PAIR_EMBEDDING_PATH).astype(np.float32)
```

The logistic regression classifier receives one selected embedding matrix:

```python
model.fit(embeddings[train_idx], y_all[train_idx])
```

The MLP receives standardized rows from the same selected matrix:

```python
x_train = scaler.fit_transform(embeddings[train_core_idx]).astype(np.float32)
x_val = scaler.transform(embeddings[val_idx]).astype(np.float32)
x_test = scaler.transform(embeddings[test_idx]).astype(np.float32)
```

Therefore:

- heavy-only classifier input: one 1024-dimensional pooled IgBert
  heavy-sequence representation.
- pair classifier input: one 1024-dimensional pooled IgBert joint heavy/light
  representation.
- classifier interaction capacity: the pair classifier can receive
  heavy-light interactions already represented by the transformer forward pass,
  because heavy and light tokens are processed together before pooling. The
  downstream MLP can further transform that joint representation; logistic
  regression remains linear on the pooled joint vector.

## Recommendation

Current naming:

- `pair` is accurate for Hugging Face / IgBert because the embedding is a true
  joint pair-text model representation.
- `pair` is potentially misleading for AbLang2 because the saved array is
  concatenated heavy-only plus light-only rather than a joint heavy-light model
  embedding.

Rename recommendation:

- Consider renaming AbLang2 artifacts and report labels from `pair` to
  `concat_pair` or `heavy_light_concat` in a future cleanup.
- Do not rename the Hugging Face / IgBert `pair` artifacts; those are joint
  pair embeddings.

Benchmark recommendation:

- Add a true AbLang2 joint-pair benchmark before any fine-tuning if the AbLang2
  API supports passing real `(heavy, light)` pairs through one model call and
  returning a single paired representation.
- The Hugging Face / IgBert benchmark already covers a true joint-pair frozen
  representation.
