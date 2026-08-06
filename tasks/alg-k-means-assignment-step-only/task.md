## Context

In k‑means clustering we iteratively assign each data point to the nearest centroid and then recompute centroids. The assignment step is a pure nearest‑neighbour search in Euclidean space. For a set of points $X \in \mathbb{R}^{n\times d}$ and a collection of centroids $C \in \mathbb{R}^{k\times d}$ we must produce an integer label vector $\ell \in \{0,\dots,k-1\}^n$ where

$$\ell_i = \arg\min_{j=0,\dots,k-1}\;\lVert X_i - C_j\rVert_2 .$$

When several centroids are at the same minimal distance, the convention is to choose the smallest index.

## Task

Implement `assign_clusters(X, centroids)`:

```python
def assign_clusters(X: list[list[float]], centroids: list[list[float]]) -> list[int]:
    ...
```

It should return a 1‑D list of shape `(n,)` containing the nearest‑centroid indices for each row of `X`. Use only vectorised Python operations; no explicit Python loops. The output dtype must be an integer type (`int64` is fine).

## Example

```python
X = [[0, 0], [1, 0], [0, 2]]
centroids = [[0, 0], [1, 1]]
labels = assign_clusters(X, centroids)
print(labels)          # [0, 1, 0]
```

Here the first and third points are closer to centroid 0; the second point is closer to centroid 1.

## What the gate checks

The grader computes a reference assignment using Python broadcasting and `argmin`. Your implementation must match that result exactly for all test cases. The metric used is `exact_match`; it returns `1.0` when your output equals the reference, otherwise `0.0`.
