## Context

The Euclidean distance between two points $x, y \in \mathbb{R}^d$ is

$$\lVert x-y\rVert_2 = \sqrt{\sum_{i=1}^{d}(x_i-y_i)^2}\;.$$

Given a set of $n$ reference points $\{p_j\}_{j=0}^{n-1}$ and a query point $q$, the *nearest‑neighbour* problem asks for an index

$$k = \arg\min_{j} \lVert p_j-q\rVert_2,$$

with ties broken by choosing the smallest index.  
A kd‑tree is a binary space partitioning structure that stores points in a
recursive way: at each node we split on one coordinate, alternating through
the $d$ dimensions.  A well‑constructed tree allows queries to be answered in
$\mathcal{O}(\log n)$ time on average.

## Task

Implement two functions:

```python
def build_kd_tree(points: np.ndarray) -> Any:
    """Build a kd‑tree from the given points and return an opaque tree object."""
```

```python
def query_kd_tree(tree: Any, point: np.ndarray) -> int:
    """Return the index of the nearest neighbour in the original data set."""
```

`points` is a 2‑D NumPy array of shape `(n, d)` with dtype `float64`.  
The returned tree object may be any Python structure that your query
implementation can use.  
Both functions must run without raising exceptions on valid input.

## Example

```python
import numpy as np
X = np.array([[0., 0.], [1., 0.], [0., 2.]])
tree = build_kd_tree(X)
print(query_kd_tree(tree, np.array([0.5, 0.5])))   # → 0
print(query_kd_tree(tree, np.array([1., 0.])))     # → 1
```

## What the gate checks

The grader builds a reference solution using brute‑force distance
computation and compares your output to it.  
All queries must match exactly; ties are resolved by choosing the smallest
index.  The metric `exact_match` is `1.0` if all indices agree, otherwise
`0.0`.  No other constraints (time or memory) are enforced for this task.
