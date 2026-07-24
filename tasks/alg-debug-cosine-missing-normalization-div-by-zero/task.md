## Context

The cosine similarity between two non‑zero vectors $x, y \in \mathbb{R}^d$ is defined as

$$\operatorname{cos}(x,y) = \frac{x^\top y}{\lVert x\rVert\,\lVert y\rVert}\;.$$

It measures the angle between the two directions and is widely used in information retrieval, clustering and recommendation systems.  
If either vector has zero norm, the denominator becomes $0$ and the expression is undefined.  A naïve implementation that simply computes the dot product

$$x^\top y$$

misses this normalization step and will produce incorrect rankings of similarity scores.  Moreover, if a query vector is exactly $\mathbf{0}$, dividing by its norm would raise a runtime error.

## Task

Implement `cosine_similarity(A, B)`:

```python
def cosine_similarity(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    ...
```

`A` and `B` are 2‑D NumPy arrays of shapes `(n_q, d)` and `(n_c, d)` respectively.  
The function must return an `(n_q, n_c)` matrix where the entry in row *i*, column *j* is the cosine similarity between `A[i]` and `B[j]`.  The implementation must

1. perform the computation using vectorised NumPy operations only (no Python loops),
2. handle zero‑norm rows gracefully by returning a similarity of `0.0` for any pair involving a zero vector,
3. produce results with dtype `float64`.

## Example

```python
import numpy as np
A = np.array([[1, 1], [0, 0]])
B = np.array([[100, 0], [1, 1]])

D = cosine_similarity(A, B)
# D ≈ [[0.70710678, 1.        ],
#      [0.,          0.        ]]
```

The first query `[1,1]` is most similar to the second candidate `[1,1]`; the zero query has similarity `0.0` with all candidates.

## What the gate checks

Two conditions are enforced:

* **Correct ranking** – The index of the maximum similarity for each query must match that produced by a reference implementation that normalises vectors and safely handles zeros.  This is measured by the scorer `argmax_agreement`, which returns the fraction of rows whose argmax matches the reference.  The gate requires this value to be exactly `1.0`.

* **No division‑by‑zero** – The function must not raise an exception when a query vector has zero norm; it should simply return zeros for that row.

The grader constructs a small test set containing both a normal query and a zero query, computes the reference similarities with NumPy, and compares argmax indices.  A broken implementation that omits normalization will fail on the first query, yielding an agreement of `0.5`.  Only a correct implementation passes the gate.
