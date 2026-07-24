## Context

Solving a linear system $Ax=b$ can amplify numerical errors when the matrix $A$ is ill-conditioned. The condition number of a matrix measures how sensitive the solution is to small perturbations.

For a matrix $A$, the 2-norm condition number is

$$
\kappa(A) = \lVert A \rVert_2 \lVert A^{-1} \rVert_2 .
$$

A small condition number indicates a stable system, while a large condition number indicates that small input errors can produce large output errors.

Given several matrices, comparing their condition numbers allows us to rank them by numerical stability. NumPy computes the reference condition number using singular values:

$$
\kappa(A) = \frac{\sigma_{\max}(A)}{\sigma_{\min}(A)} .
$$

## Task

Implement `rank_by_condition(matrices)`:

```python
def rank_by_condition(matrices: list[np.ndarray]) -> list[int]:
    ...
```

The function receives a list of square NumPy arrays. Return a list of the original indices sorted from the best-conditioned matrix to the worst-conditioned matrix.

The ordering must be ascending by the matrix 2-norm condition number. Use NumPy numerical routines for the computation.

## Example

```python
import numpy as np

mats = [
    np.eye(2),
    np.array([[1, 1], [1, 1.000001]]),
    np.array([[2, 0], [0, 0.5]])
]

order = rank_by_condition(mats)
# The identity matrix should appear before the nearly singular matrix.
```

## What the gate checks

The gate computes the condition number of every test matrix with NumPy's condition number implementation. It compares the returned index ordering with the oracle ordering using exact list equality.

A solution passes only if it ranks all matrices in the same order as the numerical oracle.
