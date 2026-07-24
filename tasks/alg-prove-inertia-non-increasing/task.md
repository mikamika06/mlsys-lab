## Context

In k‑means clustering the *inertia* (also called within‑cluster sum of squares) is defined as

$$\text{inertia} = \sum_{i=1}^{n}\lVert x_i - c_{\sigma(i)}\rVert^2,$$

where $x_i$ are the data points, $\sigma(i)$ assigns each point to a cluster and $c_k$ denotes the mean of all points in cluster $k$.  
During the standard Lloyd’s algorithm the centroids are recomputed as the means of their assigned points.  It can be shown that this update step never increases the inertia; formally

$$\text{inertia}_{t+1}\;\leq\;\text{inertia}_t,$$

for every iteration $t$.  Proving this property is a classic exercise in convexity and projection theory.

## Task

Implement the function `inertia_sequence` that runs Lloyd’s algorithm on a given dataset and returns the sequence of inertia values after each iteration.

```python
def inertia_sequence(X: np.ndarray, n_clusters: int, max_iter: int = 10) -> list[float]:
    ...
```

* `X` is an $(n_{\text{samples}},\,d)$ NumPy array of type ``float64``.
* The initial centroids must be chosen deterministically by sampling without replacement from the rows of `X` using a random number generator seeded with `0`.  In practice this can be achieved with `np.random.default_rng(0)`.
* After each assignment step compute the inertia as defined above and append it to the result list.
* Stop early if the centroids do not change (within machine precision).
* Return a Python list of floats; the length may be less than `max_iter` if convergence occurs earlier.

The function must use only NumPy operations—no explicit Python loops over samples or clusters.

## Example

```python
import numpy as np
X = np.array([[0., 0.], [1., 0.], [0., 2.], [3., 4.], [5., 6.]])
seq = inertia_sequence(X, n_clusters=2, max_iter=10)
print(seq)
# e.g. [34.0, 12.25, 8.75]
```

## What the gate checks

Two quantitative gates are applied:

1. **Monotonicity** – The returned sequence must be non‑increasing up to a tolerance of $10^{-9}$.  The grader sets a flag `monotonic` to `1` if all successive differences satisfy  
   $$\text{seq}_{t+1}-\text{seq}_t \leq 10^{-9},$$  
   otherwise the flag is `0`.

2. **Mean‑squared error** – The sequence produced by your implementation is compared against a reference sequence computed by an oracle implementation in the grader.  The metric `mse` is the global relative L² error  
   $$\text{mse} = \frac{\lVert \text{seq}_{\text{ref}}-\text{seq}\rVert_2}{\lVert \text{seq}_{\text{ref}}\rVert_2},$$  
   and must not exceed $10^{-8}$.

Both gates must pass for the solution to be accepted.
