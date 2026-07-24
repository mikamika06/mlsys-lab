## Context

For a real symmetric matrix $A \in \mathbb{R}^{n \times n}$, the eigendecomposition has the form

$$
A = V \Lambda V^\top ,
$$

where the columns of $V$ are orthonormal eigenvectors and $\Lambda$ is a diagonal matrix containing the corresponding eigenvalues.

A symmetric eigensolver returns eigenvalues $w$ and eigenvectors $V$ such that

$$
A V = V \Lambda ,
$$

with $\Lambda = \operatorname{diag}(w)$. Reconstructing the original matrix requires placing the eigenvalues on the diagonal and multiplying the three factors in the correct order:

$$
\hat{A} = V \operatorname{diag}(w) V^\top .
$$

Because eigenvectors can have arbitrary sign choices, the individual values in $V$ do not need to match another eigensolver exactly. The reconstructed matrix is the important result.

## Task

Implement `reconstruct_from_eigh(A)`:

```python
def reconstruct_from_eigh(A: np.ndarray) -> np.ndarray:
    ...
```

The input is a real symmetric NumPy matrix. Compute its symmetric eigendecomposition and return the reconstructed matrix

$$
V \operatorname{diag}(w) V^\top .
$$

The returned array must be a floating point NumPy array with the same square shape as `A`.

## Example

```python
import numpy as np

A = np.array([[2.0, 1.0], [1.0, 2.0]])
B = reconstruct_from_eigh(A)

# B is approximately:
# [[2.0, 1.0],
#  [1.0, 2.0]]
```

## What the gate checks

The gate computes the oracle reconstruction using NumPy's symmetric eigendecomposition and measures

$$
\max_{i,j} |\hat{A}_{ij} - A_{ij}|.
$$

The `max_abs_err` value must be less than $10^{-8}$. A solution that returns the eigendecomposition factors incorrectly, does not accumulate the eigenvector matrix, or uses an incorrect multiplication order will fail.
