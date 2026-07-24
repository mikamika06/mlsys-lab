## Context

For a square matrix $A \in \mathbb{R}^{n \times n}$, an LU decomposition with partial pivoting writes

$$
P A = L U ,
$$

where $P$ is a permutation matrix, $L$ is lower triangular, and $U$ is upper triangular.

The determinant follows from the decomposition:

$$
\det(A) = \det(P) \det(L) \det(U).
$$

Because $L$ has ones on its diagonal, $\det(L)=1$. The permutation contributes only a sign, so the computation reduces to

$$
\det(A) = \operatorname{sign}(P) \prod_{i=1}^{n} U_{ii}.
$$

Partial pivoting swaps rows during elimination. Every row swap changes the sign of the permutation, so the implementation must track the parity of swaps while constructing the LU factors.

## Task

Implement `det_from_lu(A)`:

```python
def det_from_lu(A: np.ndarray) -> float:
    ...
```

The function receives a square NumPy array and returns the determinant computed from an LU decomposition with partial pivoting. Do not call `np.linalg.det` inside the implementation.

Perform Gaussian elimination with row swaps, track the number of swaps, and return the sign of the permutation multiplied by the product of the diagonal entries of the resulting upper triangular matrix. The returned value must be a Python float.

## Example

```python
import numpy as np

A = np.array([[2.0, 1.0], [4.0, 3.0]])

det = det_from_lu(A)
# det == 2.0
```

## What the gate checks

The gate compares the implementation against NumPy's determinant oracle on several matrices. The relative error

$$
\mathrm{rel\_err} =
\frac{\lVert x-y\rVert_2}{\lVert y\rVert_2 + 10^{-12}}
$$

must be at most $10^{-8}$.
