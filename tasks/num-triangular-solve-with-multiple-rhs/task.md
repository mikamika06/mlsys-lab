## Context

A triangular system has a coefficient matrix that is either lower or upper triangular. This structure allows solving the system without computing a matrix inverse.

For a lower triangular matrix $L \in \mathbb{R}^{n \times n}$ and multiple right-hand sides $B \in \mathbb{R}^{n \times k}$, the goal is to find $X \in \mathbb{R}^{n \times k}$ such that

$$LX = B.$$

Forward substitution computes each row of $X$ using the rows that have already been solved:

$$
x_i = \frac{1}{L_{ii}}\left(b_i - \sum_{j=0}^{i-1} L_{ij}x_j\right).
$$

When there are multiple right-hand sides, each $x_i$ is a row vector and the same substitution logic is applied to all columns of $B$ at once.

## Task

Implement `solve_lower_multi_rhs(L, B)`:

```python
def solve_lower_multi_rhs(L: np.ndarray, B: np.ndarray) -> np.ndarray:
    ...
```

The function receives:
- `L`: a 2-D NumPy array with shape $(n, n)$ containing a nonsingular lower triangular matrix.
- `B`: a 2-D NumPy array with shape $(n, k)$ containing $k$ right-hand sides.

Return a `float64` NumPy array `X` with shape $(n, k)$ satisfying $LX \approx B`.

Use forward substitution. Do not call a high-level linear solver such as `np.linalg.solve`.

## Example

```python
import numpy as np

L = np.array([
    [2.0, 0.0],
    [3.0, 1.0],
])

B = np.array([
    [4.0, 2.0],
    [7.0, 5.0],
])

X = solve_lower_multi_rhs(L, B)

# X is approximately:
# [[2.0, 1.0],
#  [1.0, 2.0]]
```

## What the gate checks

The gate builds several lower triangular systems and computes the reference solution using NumPy's linear algebra implementation.

The returned matrix is compared with the NumPy oracle using the relative error

$$
\mathrm{rel\_err} =
\frac{\lVert X_{\mathrm{candidate}} - X_{\mathrm{oracle}}\rVert_2}
{\lVert X_{\mathrm{oracle}}\rVert_2 + 10^{-12}}.
$$

The score must satisfy $\mathrm{rel\_err} < 10^{-10}$. Implementations that solve only the first right-hand side or use incorrect substitution order will fail.
