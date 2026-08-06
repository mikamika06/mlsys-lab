## Context

k‑Means clustering seeks a partition of $n$ data points $\{x_i\}_{i=1}^n \subset \mathbb R^d$ into $k$ clusters by minimizing the within‑cluster sum of squared Euclidean distances

$$
J(\mathbf C, \mathbf z) = \sum_{i=1}^{n}\lVert x_i - c_{z_i}\rVert^2,
$$

where $\mathbf C=\{c_1,\dots,c_k\}$ are the cluster centroids and $z_i\in\{1,\dots,k\}$ assigns each point to a centroid.  
The classical Lloyd algorithm alternates two steps until convergence:

1. **Assignment** – assign every point to its nearest centroid.
2. **Update** – recompute each centroid as the mean of all points assigned to it.

When the initial centroids are supplied by the caller, the algorithm is called *full* Lloyd with fixed initialization.

## Task

Implement `lloyd_fixed_init`:

```python
def lloyd_fixed_init(
    X: list[list[float]],
    init_centroids: list[list[float]],
    max_iter: int = 300,
    tol: float = 1e-4
) -> tuple[list[int], int]:
    ...
```

* `X` – an $(n,d)$ array of data points.  
* `init_centroids` – a $(k,d)$ array giving the starting centroids.  
* The function must return a tuple `(labels, n_iter)` where  
  * `labels` is a length‑$n$ integer array with cluster indices in $\{0,\dots,k-1\}$,  
  * `n_iter` is the number of iterations actually performed (the last iteration that produced an assignment change).

The implementation must be fully vectorised: no explicit Python loops over data points. The algorithm should stop early if the assignments do not change between two consecutive iterations, or after `max_iter` iterations.

## Example

```python

X = [[0., 0.],
              [1., 0.],
              [0., 2.],
              [5., 5.]]
init_centroids = [[0., 0.],   # cluster 0
                           [5., 5.]]  # cluster 1

labels, n_iter = lloyd_fixed_init(X, init_centroids)
print(labels)   # array([0, 0, 0, 1])
print(n_iter)   # 1
```

The first three points are closer to the origin centroid; the last point is assigned to the far centroid. The algorithm converges after a single iteration.

## What the gate checks

Two aspects are verified:

* **Exact match** – the returned `labels` array must be identical to that produced by a reference implementation using Python’s broadcasting and vectorised operations.
* **Correct shape & type** – `labels` must have shape `(n,)`, dtype `int64`; `n_iter` must be an integer.

The grader runs several random test cases; any deviation from the reference labels causes the gate to fail. No timing or line‑count checks are performed for this task.
