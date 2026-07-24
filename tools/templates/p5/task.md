## Context

The squared Euclidean distance between two vectors $a, b \in \mathbb{R}^d$ is

$$\lVert a - b \rVert^2 = \sum_{i=1}^{d} (a_i - b_i)^2 = (a_1 - b_1)^2 + \dots + (a_d - b_d)^2 .$$

Given a matrix $A \in \mathbb{R}^{n \times d}$ whose rows are $n$ points, the pairwise
squared-distance matrix $D \in \mathbb{R}^{n \times n}$ has entries
$D_{ij} = \lVert A_i - A_j \rVert^2$.

A naive implementation loops over every pair $(i, j)$ in Python and costs
$O(n^2 d)$ interpreted operations. The vectorized identity removes the loop:

$$\lVert a - b \rVert^2 = \lVert a \rVert^2 + \lVert b \rVert^2 - 2\, a^\top b ,$$

so the whole matrix is $D = g\,\mathbf{1}^\top + \mathbf{1}\,g^\top - 2\,A A^\top$,
where $g_i = \lVert A_i \rVert^2$.

## Task

Implement `pairwise_sq_dists(A)`:

```python
def pairwise_sq_dists(A: np.ndarray) -> np.ndarray:
    ...
```

It takes a 2-D NumPy array of shape $(n, d)$ and returns the $(n, n)$ matrix of
squared Euclidean distances between every pair of rows. Use vectorized NumPy
only — no Python `for` loops. The result must be `float64`.

## Example

```python
import numpy as np
A = np.array([[0, 0], [1, 0], [0, 2]])
D = pairwise_sq_dists(A)
# [[0. 1. 4.]
#  [1. 0. 5.]
#  [4. 5. 0.]]
```

## What the gate checks

Two gates. The relative error $\mathrm{rel\_err}$ against a NumPy reference must
satisfy $\mathrm{rel\_err} \le 10^{-9}$, and the operation count
$\mathrm{op\_count}$ (Python line events recorded by the tracer) must stay below
$50$. A Python double loop emits thousands of line events, so only a fully
vectorized solution passes.
