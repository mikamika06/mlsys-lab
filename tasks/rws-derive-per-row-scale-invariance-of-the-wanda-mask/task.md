## Context

Wanda ("Pruning by Weights and Activations") scores each weight by how much it
contributes to the layer's output, without needing any gradients: it multiplies the
weight's magnitude by the L2 norm of the activations flowing through that input
channel, measured on a small calibration set. For weight matrix $W \in
\mathbb{R}^{\text{rows}\times\text{cols}}$ (output features $\times$ input features)
and per-input-channel norms $n_j = \lVert X_{:,j} \rVert_2$:

$$
\mathrm{score}_{ij} = |W_{ij}| \cdot n_j .
$$

Pruning is done **per output row**: within row $i$, keep the `keep_ratio` fraction of
columns with the highest score, zero out (mask out) the rest.

Now suppose every weight in row $i$ is scaled by some positive constant $c_i$ — for
instance because the row was folded with a per-output-channel calibration scale, as
happens in some quantization pipelines applied before pruning. The scaled score is

$$
\mathrm{score}'_{ij} = |c_i W_{ij}| \cdot n_j = c_i \cdot |W_{ij}| \cdot n_j = c_i \cdot \mathrm{score}_{ij},
$$

since $c_i > 0$ is a *positive constant within row $i$*, it multiplies every score in
that row by the same factor and therefore **cannot change the relative ranking** of
columns within the row. The Wanda mask of $\mathrm{diag}(c) \, W$ (any positive
per-row scale $c$) must therefore be identical, row by row, to the mask of $W$ itself.

## Task

Implement `wanda_mask`:

```python
def wanda_mask(W: np.ndarray, col_norms: np.ndarray, keep_ratio: float) -> np.ndarray:
    ...
```

- `W`: `float64` array of shape `(rows, cols)`.
- `col_norms`: `float64` array of shape `(cols,)`, the per-input-channel activation
  norm $n_j$ (already computed — you don't need to touch raw activations here).
- `keep_ratio`: fraction of each row's columns to keep, `k = round(cols * keep_ratio)`
  (at least 1).

For each row, rank columns by $\mathrm{score}_{ij} = |W_{ij}| \cdot n_j$ (descending,
ties broken by the smaller column index — i.e. a stable sort of the negated scores) and
return a boolean array of the same shape as `W`, `True` at the `k` highest-scoring
columns of each row and `False` elsewhere.

## Example

```python
import numpy as np

W = np.array([[1.0, -4.0, 2.0, 0.5]])
col_norms = np.array([1.0, 0.5, 2.0, 1.0])
wanda_mask(W, col_norms, keep_ratio=0.5)
# scores = |W| * col_norms = [1.0, 2.0, 4.0, 0.5] -> keep top 2 -> columns 1, 2
# array([[False,  True,  True, False]])
```

## What the gate checks

The gate builds a NumPy oracle computing the Wanda mask directly from a fixed test
weight matrix and fixed column norms, at several `keep_ratio` values. For each ratio, it
scales every row of `W` by an independent positive random constant (drawn across several
orders of magnitude, plus the identity scale) and calls **your** `wanda_mask` on the
*scaled* matrix with the *same* `col_norms`. `exact_match` requires your mask, computed
on the scaled matrix, to equal the oracle's mask of the **original, unscaled** `W` in
every case (must be `1.0`) — confirming both that your ranking is implemented correctly
and that it is invariant to positive per-row rescaling.
