## Context

The squared Euclidean distance between two vectors $a, b \in \mathbb{R}^d$ is

$$\lVert a - b \rVert^2 = \sum_{i=1}^{d} (a_i - b_i)^2.$$

Cosine similarity measures the angle between two vectors:

$$\operatorname{cos}(a,b) = \frac{a^\top b}{\lVert a\rVert\,\lVert b\rVert}.$$

Given a dataset $X \in \mathbb{R}^{n\times d}$, we can rank each point by either of these metrics. For a query point $x_i$ the *neighbor set* under metric $\mu$ is the ordered list of indices of its $k$ nearest (or most similar) points according to $\mu$, excluding $i$ itself.

The **divergence** between two neighbor sets for the same query is simply whether the two sets contain exactly the same indices. If they differ, we say the metrics disagree on that point’s local neighborhood.

## Task

Implement the function

```python
def l2_vs_cosine_neighbor_set_divergence(X: np.ndarray, k: int) -> np.ndarray:
    ...
```

It receives a 2‑D NumPy array `X` of shape `(n,d)` and an integer `k < n`.  
Return a boolean array of length `n`; the element at position `i` is `True`
iff the top‑`k` neighbor set of row `i` under L2 distance differs from that
under cosine similarity.

The implementation must use only NumPy operations; no explicit Python loops
over rows are required but may be used if you wish. The result should have
dtype `bool`.

## Example

```python
import numpy as np
X = np.array([[0, 0],
              [1, 0],
              [0, 2]])
k = 1
div = l2_vs_cosine_neighbor_set_divergence(X, k)
print(div)          # array([False, False, False])
```

For this tiny dataset the nearest neighbor under both metrics is the same for every point.

## What the gate checks

The grader generates several random datasets and compares your output to a
reference implementation that uses NumPy’s broadcasting and matrix
multiplication. The metric `exact_match` requires that the boolean arrays be
identical element‑wise; any discrepancy causes the gate to fail.
