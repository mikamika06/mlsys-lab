## Context

The k-means algorithm partitions points $X \in \mathbb{R}^{n \times d}$ into $k$ clusters.
Each iteration alternates between assigning points to the nearest centroid and updating
each centroid using the mean of its assigned points.

For a cluster $C_j$, the centroid update is

$$
\mu_j = \frac{1}{|C_j|}\sum_{x_i \in C_j} x_i .
$$

This formula is undefined when a cluster becomes empty because $|C_j| = 0$. A direct
implementation can produce a division-by-zero warning, a NaN centroid, or a later
failure during distance comparisons.

For this task, an empty cluster must be repaired with the following deterministic rule:
choose the point with the largest squared distance to its currently assigned centroid and
use that point as the new centroid for the empty cluster.

The squared distance between a point and a centroid is

$$
d(x,\mu) = \sum_{r=1}^{d}(x_r-\mu_r)^2 .
$$

## Task

Implement `kmeans_labels(X, k, centers, iterations)`:

```python
def kmeans_labels(
    X: list[list[float]],
    k: int,
    centers: list[list[float]],
    iterations: int
) -> list[int]:
    ...
```

The function receives a matrix of points `X` with shape $(n,d)$, an initial centroid
matrix `centers` with shape $(k,d)$, and a fixed number of iterations.

For each iteration:

1. Assign every row of `X` to the closest centroid using squared Euclidean distance.
2. Update each centroid to the mean of its assigned points.
3. If a cluster has no assigned points, replace its centroid using the empty-cluster
   reseed rule described in the context.

Return the final integer label array of shape $(n,)$.

Use deterministic Python behavior. The returned labels must use integer dtype.

## Example

```python

X = [
    [0.0, 0.0],
    [0.1, 0.0],
    [5.0, 5.0],
]

centers = [
    [0.0, 0.0],
    [0.2, 0.0],
    [10.0, 10.0],
]

labels = kmeans_labels(X, 3, centers, 3)
```

The third cluster starts with no nearby points, so the implementation must avoid creating
a NaN centroid and must apply the reseeding rule.

## What the gate checks

The gate runs the candidate implementation on clustering cases designed to trigger an
empty cluster. It computes the expected labels using an independent reference
implementation of the same k-means update and reseed algorithm.

The `exact_match` metric is `1.0` only when the returned labels exactly match the
reference labels for every case. A solution that leaves empty centroids as NaN or skips
the reseed rule will fail.
