## Context

The $k$‑means algorithm partitions a set of points $\{x_i\}_{i=1}^n \subseteq \mathbb{R}^d$ into $k$ clusters by iteratively updating cluster centroids.  
A *mini‑batch* variant processes only a small random subset (the batch) at each iteration, which reduces the cost from $O(nkd)$ to $O(bkd)$ per step where $b \ll n$.  Determinism is achieved by fixing the random seed and using the same initial centroids for every run.

## Task

Implement `mini_batch_kmeans`:

```python
def mini_batch_kmeans(
    X: np.ndarray,
    k: int,
    batch_size: int,
    n_iter: int,
    seed: int = 0
) -> np.ndarray:
    ...
```

* `X` is a 2‑D NumPy array of shape $(n, d)$ containing the data points.  
* The function must return an array of shape $(k, d)$ with the final centroids after `n_iter` iterations.  
* Use the first $k$ rows of `X` as the initial centroids.  
* At each iteration:
  1. Sample `batch_size` indices from $\{0,\dots,n-1\}$ **with replacement** using a NumPy random generator seeded by `seed`.  
  2. Assign every point in the batch to its nearest centroid (Euclidean distance).  
  3. Compute the mean of each assigned subset; if a centroid receives no points, keep its previous value.  
  4. Update centroids with an incremental average: after iteration $t$,
     $$c^{(t)} = \frac{(t-1)c^{(t-1)} + m^{(t)}}{t},$$
     where $m^{(t)}$ is the batch‑mean vector for that iteration.  
* All computations must be performed with NumPy only; no explicit Python loops over samples are required.

## Example

```python
import numpy as np
X = np.array([[0, 0], [1, 0], [0, 2], [5, 5]])
centroids = mini_batch_kmeans(X, k=2, batch_size=2, n_iter=10, seed=42)
print(centroids)
# [[0. 0.]
#  [5. 5.]]
```

## What the gate checks

The grader computes a reference implementation of the same algorithm and compares your output to it using the relative L2 error:

$$\mathrm{rel\_err} = \frac{\|C_{\text{your}} - C_{\text{ref}}\|}{\|C_{\text{ref}}\|}.$$

Your solution must satisfy $\mathrm{rel\_err}\le 10^{-6}$ for all provided test cases.  The implementation is required to be deterministic: the same input and seed must always produce identical centroids.
