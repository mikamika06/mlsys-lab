## Context

**Global unstructured pruning** (e.g. PyTorch's
`torch.nn.utils.prune.global_unstructured` with an L1/magnitude
criterion) picks *one* threshold for an entire model: pool the absolute
values of every parameter from every layer into one flat list, and prune
the globally-smallest fraction `r` of them, wherever they happen to live.

This is very different from **per-tensor pruning**, which would prune the
smallest `r` fraction *independently within each layer*. If two layers
have very different overall weight magnitude — one layer's weights
cluster around $0.01$, another's around $2.0$ — a single global threshold
is not "small" or "large" in any layer-relative sense: it is one absolute
number. The low-magnitude layer's *entire* weight distribution can fall
below it, while the high-magnitude layer loses almost nothing. Global
pruning silently starves low-scale layers.

For $L$ layers with weight tensors $W_0,\dots,W_{L-1}$ (arbitrary shapes)
and a target global prune ratio $r\in(0,1)$:

1. Pool $|W_{\ell,i}|$ over **every** element of **every** layer into one
   flat array of length $N=\sum_\ell |W_\ell|$.
2. Let $k=\operatorname{round}(rN)$ (Python's round-half-to-even). Prune
   the $k$ globally-smallest-magnitude elements (ties broken toward the
   lower flat index, i.e. a stable ascending sort of the pooled absolute
   values) — this exactly mirrors `topk(..., largest=False)` selection,
   avoiding threshold/tie ambiguity.
3. Layer $\ell$'s realized sparsity is the fraction of *its own* elements
   that ended up in that globally-pruned set of $k$.

## Task

Implement `global_threshold_layer_sparsity(weights, prune_ratio)`:

```python

def global_threshold_layer_sparsity(weights: list[list[float]], prune_ratio: float) -> dict:
    ...
```

- `weights`: a Python list of `L` list (any shapes), the model's
  per-layer weight tensors, in a fixed order.
- `prune_ratio`: float in `(0, 1)`, the target global prune fraction $r$.

Return a dict:

- `"sparsity"`: `(L,)` float64 array — layer $\ell$'s realized sparsity
  fraction under the single shared global threshold, as defined above.
- `"most_pruned_layer"`: Python `int` — the index of the layer with the
  **highest** realized sparsity (the one global pruning starved the most;
  ties broken toward the lower index, i.e. plain `argmax`).

## Example

```python
w_small = [0.01] * 4   # tiny-magnitude layer
w_big   = [2.0] * 4    # large-magnitude layer
out = global_threshold_layer_sparsity([w_small, w_big], prune_ratio=0.5)
# Pooled: [0.01]*4 + [2.0]*4, the 4 globally-smallest are all in w_small.
# out["sparsity"]         -> array([1.0, 0.0])
# out["most_pruned_layer"] -> 0
```

## What the gate checks

The grader loads four committed fixture layers (`layer0.npy` .. `layer3.npy`,
shapes `(40,40)`, `(30,50)`, `(64,20)`, `(48,48)`, with deliberately
different per-layer magnitude scales spanning `0.01` to `2.0`) and a fixed
`prune_ratio = 0.5`, then recomputes the pooled-threshold selection and
per-layer sparsity independently in Python (stable ascending sort of the
pooled absolute values, `k = round(0.5 * N)`).

- `sparsity_exact`: `1.0` if your `"sparsity"` array matches the oracle's
  exactly element-for-element, else `0.0`.
- `most_pruned_layer_exact`: `1.0` if your `"most_pruned_layer"` equals
  the oracle's `argmax`, else `0.0`.

Computing each layer's threshold independently (per-tensor pruning)
instead of pooling first, using a mismatched rounding rule for $k$, or
mixing up which fraction counts as "pruned" vs "kept" will all shift the
sparsity numbers and typically also flip which layer is identified as
most-pruned.
