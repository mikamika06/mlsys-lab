## Context

For a square linear system $Ax=b$ with $A \in \mathbb{R}^{n \times n}$, LU decomposition with partial pivoting factors the matrix as

$$
PA = LU,
$$

where $P$ is a permutation matrix, $L$ is lower triangular with unit diagonal entries, and $U$ is upper triangular.

After factorization, solving the system requires two triangular solves. First compute $y$ from

$$
Ly = Pb,
$$

using forward substitution. Then compute $x$ from

$$
Ux = y,
$$

using back substitution.

Partial pivoting chooses the largest magnitude pivot in the current column:

$$
|U_{kk}| = \max_{i \geq k}|U_{ik}|.
$$

This row exchange avoids unstable divisions by small pivots and allows the method to solve a wider range of systems.

## Task

Implement `solve_lu(A, b)`:

```python
def solve_lu(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    ...
```

The function receives a square real-valued matrix `A` and vector `b`, and returns the solution vector $x$ satisfying $Ax=b$.

Implement:
- LU decomposition with partial pivoting.
- Forward substitution.
- Back substitution.

Do not call `numpy.linalg.solve` or another direct linear-system solver.

Return a one-dimensional `float64` NumPy array.

## Example

```python
import numpy as np

A = np.array([[3.0, 1.0], [1.0, 2.0]])
b = np.array([9.0, 8.0])

x = solve_lu(A, b)
# approximately [2.0, 3.0]
```

## What the gate checks

The gate uses NumPy's `np.linalg.solve` as the numeric oracle.

For each system, it computes

$$
\mathrm{rel\_err} =
\frac{\lVert x_{\mathrm{candidate}}-x_{\mathrm{oracle}}\rVert_2}
{\lVert x_{\mathrm{oracle}}\rVert_2+10^{-12}}.
$$

The largest error across all systems must satisfy $\mathrm{rel\_err}\leq 10^{-9}$.
