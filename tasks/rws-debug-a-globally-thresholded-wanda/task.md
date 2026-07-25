## Context

Wanda pruning scores each weight using both the magnitude of the weight and an input activation scale. For a weight matrix $W \in \mathbb{R}^{m \times n}$, let the input column norm vector be $s \in \mathbb{R}^{n}$, where $s_j$ measures the importance of input feature $j$.

The Wanda score matrix is

$$
S_{ij} = |W_{ij}| \cdot s_j .
$$

A pruning decision is made independently for each output row. For row $i$, the largest scores are kept and the remaining weights are removed. This avoids a single global threshold causing some output rows to become much denser or much sparser than intended.

If the target keep ratio is $r$, each row keeps

$$
k = \max(1, \operatorname{round}(n r))
$$

weights with the largest values of $S_i$. The result is a binary mask $M$ where $M_{ij}=1$ means the weight is kept.

## Task

Implement `wanda_mask(W, col_norms, keep_ratio)`:

```python
def wanda_mask(
    W: np.ndarray,
    col_norms: np.ndarray,
    keep_ratio: float
) -> np.ndarray:
    ...
```

The inputs are:

- `W`: a 2-D NumPy array of shape $(m,n)$ containing weights.
- `col_norms`: a 1-D NumPy array of length $n$ containing input column norms.
- `keep_ratio`: a value in $(0,1]` specifying the fraction of weights to keep in every output row.

Return a boolean NumPy array of shape $(m,n)$. For each row, exactly $k$ positions should be `True`, where $k=\max(1,\operatorname{round}(n\,\texttt{keep\_ratio}))$.

Use the Wanda score formula and perform the selection independently for every row. Ties may be resolved by NumPy's deterministic ordering.

## Example

```python
import numpy as np

W = np.array([
    [1.0, -4.0, 2.0, 0.5],
    [3.0,  1.0, 1.5, 2.0],
])
col_norms = np.array([1.0, 0.5, 2.0, 1.0])

mask = wanda_mask(W, col_norms, 0.5)

# scores:
# row 0: [1.0, 2.0, 4.0, 0.5]
# row 1: [3.0, 0.5, 3.0, 2.0]
#
# one valid result:
# [[False, True, True, False],
#  [True, False, True, False]]
```

## What the gate checks

The gate computes a NumPy oracle for the Wanda scores and the row-wise top-$k$ selection. The returned mask must match the oracle exactly.

Solutions that select the largest values from the whole matrix at once, or that rank using row means instead of the L2-derived Wanda scores, produce incorrect per-row sparsity patterns and fail the gate.
