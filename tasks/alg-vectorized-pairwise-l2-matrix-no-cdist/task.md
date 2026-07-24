## Context

The squared Euclidean distance between two vectors $a,b\in \mathbb{R}^d$ is

$$\lVert a-b\rVert^2 = \sum_{i=1}^{d}(a_i-b_i)^2.$$

Given two 2‑D NumPy arrays $X\in\mathbb{R}^{n\times d}$ and $Y\in\mathbb{R}^{m\times d}$, the pairwise squared distance matrix $D\in\mathbb{R}^{n\times m}$ has entries

$$D_{ij}=\lVert X_i-Y_j\rVert^2.$$

A naive implementation loops over all pairs $(i,j)$ and costs $O(nmd)$ Python operations.  
Using the algebraic identity

$$\lVert a-b\rVert^2 = \lVert a\rVert^2 + \lVert b\rVert^2 - 2\,a^\top b,$$

the whole matrix can be computed with only NumPy vectorised operations:

$$D = X_{\text{norm}}\mathbf{1}^\top + \mathbf{1}\,Y_{\text{norm}}^\top - 2\,X Y^\top,$$

where $X_{\text{norm}}$ and $Y_{\text{norm}}$ are column vectors of squared norms.

## Task

Implement the function `pairwise_l2_matrix(X, Y)`:

```python
def pairwise_l2_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    ...
```

It must accept two 2‑D NumPy arrays of shape `(n,d)` and `(m,d)` respectively, compute the matrix of squared Euclidean distances between every row of `X` and every row of `Y`, and return a NumPy array of shape `(n,m)` with dtype `float64`. No explicit Python loops are allowed; use only vectorised NumPy operations.

## Example

```python
import numpy as np
from pairwise_l2_matrix import pairwise_l2_matrix

X = np.array([[0, 0], [1, 0], [0, 2]])
Y = np.array([[1, 1], [3, 4]])

D = pairwise_l2_matrix(X, Y)
print(D)
# [[ 2. 13.]
#  [ 1. 20.]
#  [ 5. 29.]]
```

## What the gate checks

The grader computes a reference matrix using NumPy’s vectorised formula and compares it to your output with mean‑squared error:

$$\text{mse} = \frac{1}{nm}\sum_{i,j}(D_{\text{cand}}-D_{\text{ref}})^2.$$

Your solution must achieve $\text{mse}\le 10^{-12}$ on all test cases. The output must also be of dtype `float64`. A correct, fully vectorised implementation will satisfy this gate; a loop‑based or mathematically incorrect version will fail.
