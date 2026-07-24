## Context

The rank of a matrix describes the number of independent directions represented by
the matrix. The singular value decomposition (SVD) writes a matrix
$A \in \mathbb{R}^{m \times n}$ as

$$
A = U \Sigma V^\top ,
$$

where $\Sigma$ contains the singular values
$\sigma_1, \sigma_2, \dots, \sigma_k$ on its diagonal.

When a numerical threshold is used, only singular values larger than the threshold
are counted as meaningful dimensions:

$$
\mathrm{rank}(A) = \left|\{i : \sigma_i > \mathrm{tol}\}\right|.
$$

This lets an implementation ignore singular values that are at or below the
chosen noise level.

## Task

Implement `svd_rank(A, tol)`:

```python
def svd_rank(A: np.ndarray, tol: float) -> int:
    ...
```

The function receives a 2-D NumPy array and a threshold value. Return the number
of singular values strictly greater than `tol`.

Use singular values from an SVD computation. The comparison must be strict:
a singular value equal to `tol` does not contribute to the rank.

## Example

```python
import numpy as np

A = np.array([
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [1.0, 1.0, 2.0],
])

print(svd_rank(A, 1e-10))
# 2
```

## What the gate checks

The gate compares the result with a NumPy reference computed using
`numpy.linalg.matrix_rank` with the provided tolerance.

The tests include full-rank matrices, rank-deficient matrices, and a boundary case
where a singular value is exactly equal to the threshold. A correct solution must
match the NumPy oracle and use the strict $>$ threshold rule.
