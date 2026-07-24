## Context

k‑Means clustering partitions a set of $n$ points $\{x_i\}_{i=1}^n \subseteq \mathbb R^d$ into $k$ disjoint groups by minimizing the within‑cluster sum of squared distances (WCSS)

$$J(\{\mu_j\}, \{c_i\}) = \sum_{j=1}^{k}\;\sum_{\substack{i\\ c_i=j}}\!\!\|x_i-\mu_j\|^2,$$

where $\mu_j$ is the centroid of cluster $j$ and $c_i\in\{1,\dots,k\}$ denotes the assignment of point $i$.  
The classic Lloyd’s algorithm alternates two steps until convergence:

1. **Assignment** – each point is assigned to its nearest centroid.
2. **Update** – centroids are recomputed as the mean of all points assigned to them.

Convergence is declared when the change in every centroid falls below a tolerance $\varepsilon$ or after a maximum number of iterations.

## Task

Implement `predict_kmeans_convergence`:

```python
def predict_kmeans_convergence(
    X: np.ndarray,
    k: int,
    max_iter: int = 300,
    tol: float = 1e-4
) -> tuple[int, np.ndarray]:
    ...
```

The function receives a 2‑D NumPy array `X` of shape $(n,d)$ and the desired number of clusters `k`.  
It must run Lloyd’s algorithm with a deterministic random seed (`seed=0`) for centroid initialization.  
Return a pair `(iters, labels)` where:

* **iters** – the number of iterations executed until convergence (including the final iteration that satisfies the tolerance). If the algorithm reaches `max_iter` without converging, return `max_iter`.
* **labels** – a 1‑D NumPy array of length $n$ containing integer cluster assignments in the range `[0, k-1]`.

The implementation must use only vectorised NumPy operations; no explicit Python loops over points or clusters.

## Example

```python
import numpy as np
X = np.array([[0, 0], [1, 0], [0, 2], [10, 10]])
iters, labels = predict_kmeans_convergence(X, k=2)
print(iters)   # e.g. 3
print(labels)  # array([0, 0, 0, 1])
```

## What the gate checks

The grader computes a reference solution using NumPy and compares your output with an **exact match** metric.  
Your function must return exactly the same iteration count and label array as the oracle for all test cases.
