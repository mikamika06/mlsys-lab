## Context

The Cholesky decomposition factors a symmetric positive-definite matrix
$A \in \mathbb{R}^{n\times n}$ into

$$
A = L L^\top,
$$

where $L$ is lower-triangular with strictly positive diagonal entries. The
classic scalar algorithm fills $L$ column by column, row by row:

$$
L_{jj} = \sqrt{A_{jj} - \sum_{k=0}^{j-1} L_{jk}^2}, \qquad
L_{ij} = \frac{1}{L_{jj}}\left(A_{ij} - \sum_{k=0}^{j-1} L_{ik}L_{jk}\right), \quad i > j.
$$

## Task

The function below is supposed to implement `cholesky`, but it has a bug:
the diagonal entry $L_{jj}$ is assigned the raw residual $A_{jj} - s$
instead of its square root. Because every off-diagonal entry in that
column is later divided by $L_{jj}$, the bug corrupts the entire factor,
not just the diagonal.

```python
def cholesky(A: np.ndarray) -> np.ndarray:
    ...
```

* `A` — a 2-D NumPy array of shape $(n, n)$, symmetric positive-definite.
* Returns `L`, a 2-D NumPy array of shape $(n, n)$: lower-triangular with
  `L @ L.T` reconstructing `A`.

Find and fix the bug.

## Example

```python
import numpy as np
A = np.array([[4.0, 2.0], [2.0, 5.0]])
L = cholesky(A)
print(L)
# [[2.  0.]
#  [1.  2.]]
print(L @ L.T)   # -> A, exactly
```

## What the gate checks

Two gates, both computed against real oracles on several deterministic
SPD test matrices (sizes 4, 7, 12, 25):

* **max_abs_err** — the largest entrywise error in the reconstruction
  `L @ L.T - A`. Must be below `1e-10`.
* **oracle_max_abs_err** — the largest entrywise difference between your
  `L` and `np.linalg.cholesky(A)`, which is the unique lower-triangular
  factor with positive diagonal for an SPD matrix. Must be below `1e-10`.
