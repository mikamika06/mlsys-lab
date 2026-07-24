## Context

Back-substitution solves an upper-triangular linear system $Ux = b$ where
$U \in \mathbb{R}^{n \times n}$. The system looks like:

$$\begin{pmatrix} u_{00} & u_{01} & \cdots & u_{0,n-1} \\ 0 & u_{11} & \cdots & u_{1,n-1} \\ \vdots & & \ddots & \vdots \\ 0 & 0 & \cdots & u_{n-1,n-1} \end{pmatrix} \begin{pmatrix} x_0 \\ x_1 \\ \vdots \\ x_{n-1} \end{pmatrix} = \begin{pmatrix} b_0 \\ b_1 \\ \vdots \\ b_{n-1} \end{pmatrix}$$

Starting from the bottom row and working upward, each unknown is determined by
substituting all already-computed values. The last unknown is

$$x_{n-1} = \frac{b_{n-1}}{u_{n-1,n-1}}$$

and for $i = n-2, n-3, \ldots, 0$:

$$x_i = \frac{1}{u_{ii}}\!\left(b_i - \sum_{j=i+1}^{n-1} u_{ij}\, x_j\right)$$

The loop must therefore iterate $i$ from $n-1$ down to $0$ inclusive.

## Task

A colleague left you a Python implementation of back-substitution in the file
`starter.py`. It contains an off-by-one error in the loop range: it skips the
last row, leaving the final element of $x$ at zero. Find the bug, fix it, and
make the function pass all test cases.

The function signature is:

```python
def back_sub(U, b):
    """Solve Ux = b where U is upper triangular (n×n). Returns array x of length n."""
```

Both `U` and `b` are NumPy arrays. `U` is square with nonzero diagonal entries.
Return a `float64` NumPy array of length $n$.

## Example

```python
import numpy as np
U = np.array([[2.0, 1.0, 3.0],
              [0.0, 4.0, 5.0],
              [0.0, 0.0, 6.0]])
b = np.array([9.0, 23.0, 12.0])
x = back_sub(U, b)
# x ≈ [1.0, 2.0, 2.0]   (check: U @ x == b)
```

## What the gate checks

The gate computes the relative $L_2$ error

$$\mathrm{rel\_err} = \frac{\lVert x_{\text{student}} - x_{\text{ref}} \rVert_2}{\lVert x_{\text{ref}} \rVert_2}$$

against a NumPy reference (`np.linalg.solve`) on four random upper-triangular
systems of sizes $n \in \{4, 8, 15, 20\}$. The maximum relative error across all
cases must satisfy $\mathrm{rel\_err} \le 10^{-10}$.

The buggy starter skips row $n-1$, leaving $x_{n-1} = 0$, which produces a
relative error well above this threshold on every non-trivial test case.
