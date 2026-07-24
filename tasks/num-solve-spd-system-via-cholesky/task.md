## Context

A symmetric positive definite (SPD) matrix $A \in \mathbb{R}^{n \times n}$ has a Cholesky factorization

$$
A = L L^\top,
$$

where $L$ is a lower triangular matrix with positive diagonal entries.

To solve the linear system

$$
Ax = b,
$$

the factorization turns the problem into two triangular solves:

$$
Ly = b,
$$

followed by

$$
L^\top x = y.
$$

This avoids computing an explicit inverse and uses the structure of SPD matrices for an efficient and numerically stable solve.

## Task

Implement `solve_spd(A, b)`:

```python
def solve_spd(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    ...
```

The function receives a square SPD matrix `A` and a one-dimensional right-hand side vector `b`. Return the solution vector `x` to $Ax=b$.

Implement the algorithm using Cholesky decomposition and two triangular solves. Do not call `np.linalg.solve` or other direct linear-system solvers. The result must be `float64`.

## Example

```python
import numpy as np

A = np.array([
    [4.0, 2.0],
    [2.0, 3.0],
])
b = np.array([8.0, 7.0])

x = solve_spd(A, b)
# [1. 2.]
```

## What the gate checks

The gate computes a reference solution using NumPy's linear solver on the same SPD inputs. The submitted implementation is compared using the relative error

$$
\mathrm{rel\_err} =
\frac{\lVert x_{\mathrm{candidate}}-x_{\mathrm{reference}}\rVert_2}
{\lVert x_{\mathrm{reference}}\rVert_2 + 10^{-12}}.
$$

The solution passes when $\mathrm{rel\_err} < 10^{-9}$ across all test cases.
