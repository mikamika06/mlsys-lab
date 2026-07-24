## Context

The squared Euclidean distance between two vectors $a,b\in\mathbb{R}^d$ is

$$
\lVert a-b\rVert^2 = \sum_{k=1}^{d}(a_k-b_k)^2.
$$

Computing all pairwise distances between $n$ rows of a matrix $A\in\mathbb{R}^{n\times d}$ with nested Python loops performs $O(n^2d)$ explicit operations in Python. NumPy can move the computation into optimized array operations by using the expansion identity

$$
\lVert a-b\rVert^2 = \lVert a\rVert^2+\lVert b\rVert^2-2a^\top b.
$$

For a whole matrix, define

$$
g_i=\lVert A_i\rVert^2.
$$

The pairwise distance matrix can then be computed as

$$
D = g\mathbf{1}^{\top}+\mathbf{1}g^{\top}-2AA^{\top}.
$$

This approach avoids Python-level loops over rows and keeps the expensive work inside NumPy.

## Task

Implement `pairwise_sq_dists(A)`:

```python
def pairwise_sq_dists(A: np.ndarray) -> np.ndarray:
    ...
```

The function receives a two-dimensional NumPy array of shape $(n,d)$ and returns an $(n,n)$ NumPy array where entry $D_{ij}$ is the squared Euclidean distance between row $i$ and row $j$.

Requirements:

- Use vectorized NumPy operations only.
- Do not use Python `for` or `while` loops over the rows or columns.
- Return a `float64` array.
- The result should be numerically equivalent to the expansion-trick formula.

## Example

```python
import numpy as np

A = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 2.0],
])

D = pairwise_sq_dists(A)

# [[0. 1. 4.]
#  [1. 0. 5.]
#  [4. 5. 0.]]
```

## What the gate checks

The gate computes the reference result using NumPy operations from the mathematical expansion formula.

The mean squared error

$$
\mathrm{MSE}=\frac{1}{n^2}\sum_{i,j}(D_{ij}-D^{ref}_{ij})^2
$$

must be at most $10^{-8}$.

The gate also records Python line events executed inside `pairwise_sq_dists`. The count must be at most $60$. A nested Python implementation creates many line events as $n$ grows and fails this check, while a vectorized implementation keeps the count approximately constant.
