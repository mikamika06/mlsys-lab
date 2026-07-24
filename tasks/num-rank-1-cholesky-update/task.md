## Context

For a symmetric positive definite matrix $A \in \mathbb{R}^{n \times n}$, the Cholesky decomposition writes

$$
A = L L^\top,
$$

where $L$ is lower triangular with positive diagonal entries.

When a matrix receives a rank-1 update

$$
A' = A + x x^\top,
$$

recomputing the entire factorization with Cholesky decomposition works but ignores the special structure of the update. A rank-1 Cholesky update modifies the existing factor $L$ directly in $O(n^2)$ operations.

For each column $k$, the update maintains the invariant that the leading part of $L$ represents the Cholesky factor of the updated leading submatrix. The scalar rotation values are

$$
r = \sqrt{L_{kk}^2 + x_k^2}, \qquad c = \frac{r}{L_{kk}}, \qquad s = \frac{x_k}{L_{kk}} .
$$

The remaining entries are updated using

$$
L_{ik} \leftarrow \frac{L_{ik} + s x_i}{c},
$$

and

$$
x_i \leftarrow c x_i - s L_{ik}.
$$

## Task

Implement `rank1_cholesky_update(L, x)`:

```python
def rank1_cholesky_update(L: np.ndarray, x: np.ndarray) -> np.ndarray:
    ...
```

The function receives a lower-triangular Cholesky factor `L` such that $A = LL^\top$ and a vector $x$. Return a new lower-triangular matrix containing the Cholesky factor of

$$
A + x x^\top .
$$

The input arrays may not be modified. Use NumPy operations for the arithmetic. The returned array must have dtype `float64`.

## Example

```python
import numpy as np

L = np.array([[2.0, 0.0],
              [1.0, 1.0]])
x = np.array([1.0, 2.0])

updated = rank1_cholesky_update(L, x)

# updated @ updated.T equals:
# [[5.0, 4.0],
#  [4.0, 6.0]]
```

## What the gate checks

The gate computes the oracle answer by forming $A = LL^\top$, applying the rank-1 update $A + xx^\top$, and using NumPy's Cholesky implementation.

The returned factor is compared with the oracle using the maximum absolute error

$$
\max_{i,j} |L_{ij}^{candidate} - L_{ij}^{oracle}|.
$$

The `max_abs_err` value must be less than $10^{-9}$.
