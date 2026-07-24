## Context

In a transformer feed‑forward network (FFN) the hidden representation is first projected to a higher dimensional space by a weight matrix \(W_{\text{down}}\in\mathbb{R}^{d_{\text{in}}\times d_{\text{hidden}}}\), then transformed back to the model dimension by \(W_{\text{up}}\in\mathbb{R}^{d_{\text{hidden}}\times d_{\text{out}}}\).  
Pruning neurons means selecting a subset of the hidden units and removing all associated rows from \(W_{\text{up}}\) and columns from \(W_{\text{down}}\).  The resulting matrices still compose to a valid linear map, but with fewer parameters.

A simple importance score for neuron \(i\) is

$$
s_i = \sum_{j} |(W_{\text{up}})_{ij}| + \sum_{k} |(W_{\text{down}})_{ki}|.
$$

Keeping the top‑\(k\) neurons according to this score yields a consistent pruning of both projections.

## Task

Implement `prune_ffn_neurons(up_proj, down_proj, target_width)`:

```python
def prune_ffn_neurons(
    up_proj: np.ndarray,
    down_proj: np.ndarray,
    target_width: int
) -> tuple[list[int], np.ndarray, np.ndarray]:
```

`up_proj` has shape `(hidden_dim, out_dim)`, `down_proj` has shape `(in_dim, hidden_dim)` and `target_width <= hidden_dim`.  
Return a tuple containing

1. the sorted list of kept neuron indices,
2. the sliced `up_proj` with only those rows,
3. the sliced `down_proj` with only those columns.

All operations must use NumPy; no Python loops are required.

## Example

```python
import numpy as np
np.random.seed(0)
in_dim, hidden_dim, out_dim = 4, 6, 5
up_proj   = np.random.randn(hidden_dim, out_dim)
down_proj = np.random.randn(in_dim, hidden_dim)

indices, up_sliced, down_sliced = prune_ffn_neurons(up_proj, down_proj, target_width=3)

print(indices)          # e.g. [0, 2, 5]
print(up_sliced.shape)  # (3, 5)
print(down_sliced.shape)# (4, 3)
```

## What the gate checks

The grader computes a reference solution that ranks neurons by the sum of absolute weights in both projections and keeps the top‑\(k\).  
Your implementation must return exactly the same index list and produce slices with matching shapes.  The metric `exact_match` is used; any deviation causes failure.
