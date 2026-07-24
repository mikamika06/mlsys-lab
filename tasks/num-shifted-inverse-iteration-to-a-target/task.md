## Context

Power iteration repeatedly applies a matrix to a vector and converges toward the
eigenvector with the largest magnitude eigenvalue. Shifted inverse iteration changes
the target by applying the inverse of a shifted matrix.

For a symmetric matrix $A \in \mathbb{R}^{n \times n}$ and shift $\mu$, each step
solves

$$
(A-\mu I)y_k=x_k
$$

and normalizes the result:

$$
x_{k+1}=\frac{y_k}{\lVert y_k\rVert_2}.
$$

The shift changes the convergence target. After enough iterations, the vector
approximates the eigenvector whose eigenvalue is closest to $\mu$. The eigenvalue
can be estimated with the Rayleigh quotient:

$$
\lambda =
\frac{x^\top A x}{x^\top x}.
$$

## Task

Implement `shifted_inverse_iteration(A, mu, x0, iters)`:

```python
def shifted_inverse_iteration(
    A: np.ndarray,
    mu: float,
    x0: np.ndarray,
    iters: int
) -> float:
    ...
```

The input `A` is a real symmetric square matrix. Perform shifted inverse iteration
for exactly `iters` iterations starting from `x0` and return the final Rayleigh
quotient as a Python `float`.

Use NumPy operations. Solving the linear system with `np.linalg.solve` is allowed.

## Example

```python
import numpy as np

A = np.array([[4.0, 1.0], [1.0, 3.0]])
value = shifted_inverse_iteration(A, 3.2, np.array([1.0, 0.5]), 20)

# value is close to the eigenvalue nearest to 3.2
```

## What the gate checks

The gate computes the oracle answer with NumPy's eigensolver. It selects the
eigenvalue closest to the supplied shift $\mu$ and compares the implementation
output against that value.

The relative error is

$$
\mathrm{rel\_err} =
\frac{|x-y|}{|x|+10^{-12}}.
$$

The maximum relative error across deterministic test matrices must be less than
$10^{-6}$.
