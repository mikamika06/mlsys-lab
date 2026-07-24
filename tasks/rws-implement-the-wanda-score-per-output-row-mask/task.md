## Context

Wanda ("Pruning by Weights and Activations") is a gradient-free pruning criterion: it
scores each weight by combining its own magnitude with how much signal flows through its
input channel, measured on a small calibration batch. For weight matrix
$W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ and calibration activations
$X \in \mathbb{R}^{n \times d_{\text{in}}}$ (rows are samples, columns are input
channels), the per-input-channel norm is

$$
\lVert X_{:,j} \rVert_2 = \sqrt{\sum_{t=1}^{n} X_{t,j}^2},
$$

and the Wanda score of weight $W_{ij}$ is

$$
S_{ij} = |W_{ij}| \cdot \lVert X_{:,j} \rVert_2 .
$$

Pruning is applied **per output row** $i$: within that row, keep only the columns with
the highest score, dropping a `sparsity` fraction of them.

## Task

Implement `wanda_score_mask`:

```python
def wanda_score_mask(W: np.ndarray, X: np.ndarray, sparsity: float) -> np.ndarray:
    ...
```

- `W`: `float64` array of shape `(d_out, d_in)`.
- `X`: `float64` array of shape `(n_samples, d_in)`.
- `sparsity`: fraction of each row's `d_in` columns to prune, in $[0, 1)$.

Compute $S = |W| \cdot \lVert X \rVert_{2,\text{axis}=0}$ (broadcast the per-column norm
across every row), then for every row keep the
$k = \max(1, \mathrm{round}((1 - \texttt{sparsity}) \cdot d_{\text{in}}))$
highest-scoring columns (ties broken by the smaller column index). Return a boolean
array of shape `(d_out, d_in)`, `True` where the weight is kept, `False` where it is
pruned.

## Example

```python
import numpy as np

W = np.array([[1.0, -4.0, 2.0, 0.5]])
X = np.array([[1.0, 0.5, 2.0, 1.0],
              [1.0, 0.5, 2.0, 1.0]])  # column norms: sqrt(2)*[1, 0.5, 2, 1]
wanda_score_mask(W, X, sparsity=0.5)
# scores ~ |W| * col_norm -> column 2 (2.0*2*sqrt2) and column 1 (4*0.5*sqrt2) score highest
# array([[False,  True,  True, False]])
```

## What the gate checks

The gate builds a NumPy oracle computing $S = |W| \cdot \lVert X \rVert_{2,\text{axis}=0}$
and the per-row top-$k$ mask directly, on a fixed test `W`/`X` pair (with input channels
at deliberately different activation scales) at several `sparsity` levels. For each
level it computes the intersection-over-union (`iou`) between your mask and the
oracle's; the reported `iou` is the minimum across all tested sparsity levels and must
be exactly `1.0` — i.e. your mask must match the oracle's elementwise, every time.
