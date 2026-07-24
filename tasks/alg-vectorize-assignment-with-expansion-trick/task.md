## Context

In k‑means clustering each data point $x_i \in \mathbb{R}^d$ is assigned to the nearest centroid $c_j$. The squared Euclidean distance between two vectors is

$$\lVert x - c\rVert^2 = \sum_{k=1}^{d}(x_k-c_k)^2.$$

A naive implementation loops over all pairs $(i,j)$ and costs $O(nk d)$. Using the algebraic identity

$$\lVert x - c\rVert^2 = \lVert x\rVert^2 + \lVert c\rVert^2 - 2\,x^\top c,$$

the entire distance matrix can be built with a handful of NumPy operations.

## Task

Implement `assign_clusters(X, centroids)`:

```python
def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    ...
```

`X` is an $(n,d)$ array of data points and `centroids` is a $(k,d)$ array. Return a 1‑D integer array of length $n$ containing the index of the nearest centroid for each point. The implementation must use only vectorised NumPy; no explicit Python loops are allowed.

## Example

```python
import numpy as np
X = np.array([[0, 0], [1, 0], [0, 2]])
C = np.array([[0, 0], [1, 1]])
idx = assign_clusters(X, C)
# array([0, 1, 0])
```

## What the gate checks

Two metrics are evaluated:

* **exact_match** – the returned assignments must be identical to a NumPy reference for several random test cases.
* **op_count** – the number of Python line events executed while calling `assign_clusters` on a small test case must not exceed 80. This ensures that the solution is fully vectorised and contains no explicit loops.

A correct implementation will pass both gates.
