## Context

The $k$‑nearest‑neighbour ($k$‑NN) problem asks, for each query point $q$, to return the indices of the $k$ data points that are closest in Euclidean distance:
$$
\operatorname{nn}_k(q)=\argmin_{i=1,\dots,n}^{(k)} \lVert q-A_i\rVert_2 .
$$

Exact search requires $\mathcal O(n)$ distance evaluations per query, which is expensive for large $n$.  Random‑hyperplane locality‑sensitive hashing (LSH) offers a fast approximation: each data point and query is projected onto many random hyperplanes; the sign pattern of these projections serves as a hash key.  Points that fall into the same bucket are considered candidates for being near a query.

The recall@$k$ metric measures how well an approximate algorithm recovers the true $k$ nearest neighbours:
$$
\operatorname{recall}@k=\frac{1}{m}\sum_{j=1}^{m}
  \frac{\lvert\,\widehat{\operatorname{nn}}_k(q_j)\cap
          \operatorname{nn}_k(q_j)\,\rvert}{k},
$$
where $m$ is the number of queries and $\widehat{\operatorname{nn}}_k$ denotes the approximate result.

## Task

Implement a function that, given a data matrix $A\in\mathbb R^{n\times d}$, a query matrix $Q\in\mathbb R^{m\times d}$, an integer $k$, a number of hash tables $t$, and a random seed, returns the average recall@$k$ over all queries using random‑hyperplane LSH.

```python
def lsh_recall(A: np.ndarray,
               Q: np.ndarray,
               k: int,
               t: int,
               seed: int) -> float:
    ...
```

The function must:

1. Generate $t$ independent random hyperplanes (rows of a matrix $\mathbf{R}\in\mathbb R^{t\times d}$).
2. Compute the sign pattern for every point in $A$ and $Q$:  `sign(x) = x @ R.T >= 0`.
3. For each query, collect all data points that share the same sign bit in any of the $t$ tables.
4. From this candidate set compute exact Euclidean distances to the query and pick the $k$ closest.
5. Compute recall@$k$ against the true nearest neighbours (obtained by brute force) and return the average over all queries.

The implementation should be fully vectorised where possible but may use Python loops for small auxiliary structures.  The returned value must be a `float` in the interval $[0,1]$.

## Example

```python
import numpy as np
A = np.random.randn(100, 20)
Q = np.random.randn(5, 20)
recall = lsh_recall(A, Q, k=3, t=10, seed=42)
print(f"Recall@3: {recall:.4f}")
```

With a reasonable number of hash tables the recall will typically be close to $1$ for this toy data.

## What the gate checks

The grader evaluates your implementation on random data and compares the returned average recall against a threshold.  The function must return a value $\ge 0.75$.  Any lower value fails the gate.
