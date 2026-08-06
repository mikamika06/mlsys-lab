## Context

K‑Means++ is a popular initialization scheme for the $k$‑means clustering algorithm.  
Given a dataset $X \in \mathbb{R}^{n\times d}$ and a desired number of clusters $k$, the method selects $k$ initial centers by repeatedly sampling points with probability proportional to the squared Euclidean distance $D^2$ from each point to its nearest already chosen center.

Let $x_i$ denote the $i$‑th row of $X$.  
After selecting a set $\mathcal{C}$ of centers, define for every data point

$$
d_i = \min_{c\in\mathcal{C}} \lVert x_i - c\rVert^2 .
$$

The next center is chosen by sampling an index $j$ with probability

$$
p_j = \frac{d_j}{\sum_{\ell=1}^n d_\ell}.
$$

This weighted sampling ensures that points far from existing centers are more likely to become new centers, leading to a better spread of initial seeds.

In this task we are given an explicit random stream $u \in [0,1)^m$ (a 1‑D list) and must consume its values in order to perform the weighted sampling deterministically. No additional randomness is allowed; the algorithm must be fully reproducible from $X$, $k$, and $u$.

## Task

Implement a function with the following signature:

```python
def kmeans_pp_seed(X: list[list[float]], n_clusters: int, rng_stream: list[float]) -> list[int]:
    ...
```

* `X` – a 2‑D list of shape `(n_samples, n_features)` containing the data points.
* `n_clusters` – the number of centers to select (`k`).
* `rng_stream` – a 1‑D list of uniformly distributed floats in `[0,1)`.  
  The first element is used for selecting the very first center (uniform over all points).  
  Subsequent elements are consumed one per additional center and are interpreted as uniform random numbers to perform weighted sampling via the cumulative distribution function.

The function must return a list of shape `(n_clusters,)` containing the indices of the chosen centers. Indices should be `int64`. The algorithm must **not** use any other source of randomness; it must rely solely on `rng_stream`.

## Example

```python

X = [[0, 0],
              [1, 0],
              [0, 1],
              [10, 10]]
rng_stream = [0.25, 0.6, 0.9]   # three values for k=3

indices = kmeans_pp_seed(X, 3, rng_stream)
print(indices)          # e.g., array([2, 1, 3])
```

The exact output depends on the deterministic sampling procedure described above.

## What the gate checks

* **Exact match** – The returned indices must be identical to those produced by a reference implementation that follows the same algorithm and consumes `rng_stream` in order.  
  The grader computes the reference using Python operations; no hard‑coded expected values are used.
* **Determinism** – Because only `rng_stream` is allowed for randomness, any deviation (e.g., reusing the stream incorrectly or ignoring it) will cause a mismatch and fail the gate.

The task is considered solved when your implementation passes the exact‑match check on all provided test cases.
