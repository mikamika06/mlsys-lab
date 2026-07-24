## Context

Forward substitution solves a lower-triangular linear system $Lx = b$ where
$L \in \mathbb{R}^{n \times n}$:

$$
\begin{pmatrix}
l_{00} & 0      & \cdots & 0 \\
l_{10} & l_{11} & \cdots & 0 \\
\vdots &        & \ddots & \vdots \\
l_{n-1,0} & l_{n-1,1} & \cdots & l_{n-1,n-1}
\end{pmatrix}
\begin{pmatrix} x_0 \\ x_1 \\ \vdots \\ x_{n-1} \end{pmatrix}
=
\begin{pmatrix} b_0 \\ b_1 \\ \vdots \\ b_{n-1} \end{pmatrix}
$$

Starting from the top row and working downward, each unknown is determined
from already-computed values:

$$
x_0 = \frac{b_0}{l_{00}}, \qquad
x_i = \frac{1}{l_{ii}}\!\left(b_i - \sum_{j=0}^{i-1} l_{ij}\, x_j\right)
\;\; \text{for } i = 1, \dots, n-1.
$$

## Task

Implement `forward_sub`:

```python
def forward_sub(L, b):
    """Solve Lx = b where L is lower triangular (n×n) with a nonzero
    diagonal, by rows. Returns a float64 NumPy array of length n."""
```

`L` is a square lower-triangular NumPy array with nonzero diagonal entries;
`b` is a 1-D NumPy array of the same length. Work through the rows from
$i=0$ to $n-1$, accumulating the already-known terms, exactly as in the
formula above. Do **not** call a general or triangular linear-system solver
(`numpy.linalg.solve`, `scipy.linalg.solve`, `scipy.linalg.solve_triangular`,
...) — the point of the exercise is to implement the row-by-row recurrence
yourself.

## Example

```python
import numpy as np
L = np.array([[2.0, 0.0, 0.0],
              [1.0, 3.0, 0.0],
              [4.0, 2.0, 5.0]])
b = np.array([4.0, 5.0, 20.0])

x = forward_sub(L, b)
# x ≈ [2.0, 1.0, 2.0]   (check: L @ x == b)
```

## What the gate checks

Two gates must both pass:

* **rel_err** — the relative $L_2$ error
  $$
  \mathrm{rel\_err} = \frac{\lVert x_{\text{student}} - x_{\text{ref}} \rVert_2}{\lVert x_{\text{ref}} \rVert_2}
  $$
  against `scipy.linalg.solve_triangular(L, b, lower=True)` on several random
  well-conditioned lower-triangular systems, must satisfy
  $\mathrm{rel\_err} \le 10^{-10}$.
* **no_solver_shortcut** — while your function runs, the grader temporarily
  wraps `numpy.linalg.solve`, `scipy.linalg.solve`, and
  `scipy.linalg.solve_triangular` to detect whether any of them were called.
  This metric must equal `1.0` (no shortcut call detected); calling any of
  them to sidestep the row-by-row implementation makes it `0.0` and fails
  the gate even if the numeric answer is correct.
