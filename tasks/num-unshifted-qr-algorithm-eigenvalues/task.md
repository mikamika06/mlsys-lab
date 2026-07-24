## Context

For a real symmetric matrix $A \in \mathbb{R}^{n \times n}$, the eigenvalues are the
values $\lambda$ satisfying

$$
A v = \lambda v
$$

for some nonzero vector $v$. Symmetric matrices have real eigenvalues and can be
diagonalized by an orthogonal basis.

The unshifted QR algorithm repeatedly factors the current matrix as

$$
A_k = Q_k R_k
$$

where $Q_k$ is orthogonal and $R_k$ is upper triangular, then updates

$$
A_{k+1} = R_k Q_k .
$$

For symmetric input, the matrices converge toward diagonal form. The diagonal
entries of the converged matrix approximate the eigenvalues of the original
matrix.

## Task

Implement `qr_eigenvalues(A, max_iter=1000, tol=1e-12)`:

```python
def qr_eigenvalues(A: np.ndarray, max_iter: int = 1000, tol: float = 1e-12) -> np.ndarray:
    ...
```

The function receives a real symmetric 2-D NumPy array and returns a
one-dimensional NumPy array containing its eigenvalues.

Use the unshifted QR iteration:
1. Compute the QR factorization of the current matrix.
2. Replace the matrix with $R Q$.
3. Stop when the off-diagonal entries are sufficiently small.

The returned eigenvalues may be in any order. The output must be `float64`.

## Example

```python
import numpy as np

A = np.array([
    [2.0, 1.0],
    [1.0, 2.0],
])

values = qr_eigenvalues(A)
# values are approximately [3.0, 1.0]
```

## What the gate checks

The gate compares the sorted eigenvalues from `qr_eigenvalues` against the
NumPy symmetric eigensolver reference. The relative error

$$
\mathrm{rel\_err} =
\frac{\lVert x - y \rVert_2}{\lVert y \rVert_2 + 10^{-12}}
$$

must satisfy $\mathrm{rel\_err} < 10^{-6}$ on several symmetric matrices.
