## Context

The Hamming distance between two binary vectors $x, y \in \{0,1\}^d$ is the number of coordinates where they differ,

$$
H(x,y) = \sum_{i=1}^{d} (x_i \oplus y_i),
$$

where $\oplus$ denotes exclusive‑or. The Jaccard similarity measures how many coordinates are simultaneously 1 relative to all coordinates that are 1 in at least one vector:

$$
J(x,y)=\frac{|\,x \land y\,|}{|\,x \lor y\,|}
=\frac{\sum_{i=1}^{d} (x_i \wedge y_i)}{\sum_{i=1}^{d} (x_i \vee y_i)},
$$

with the convention $J(0,0)=1$.

Given a binary matrix $B\in\{0,1\}^{n\times d}$ whose rows are $n$ vectors, we want to compute two $n\times n$ matrices: the Hamming distance matrix $H$ and the Jaccard similarity matrix $J$.

## Task

Implement `hamming_and_jaccard(B)`:

```python
def hamming_and_jaccard(B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ...
```

It takes a 2‑D NumPy array of shape $(n,d)$ with entries 0 or 1 and returns a tuple `(H,J)` where

* `H` is an integer matrix of shape $(n,n)$ containing the Hamming distances,
* `J` is a float64 matrix of shape $(n,n)$ containing the Jaccard similarities.

The implementation must use only NumPy vectorised operations; no explicit Python loops are allowed. The returned matrices should be exactly as described: `H.dtype` integer (any signed type), `J.dtype` `float64`.

## Example

```python
import numpy as np
B = np.array([[0,1,0],
              [1,1,0],
              [0,0,1]], dtype=np.uint8)
H,J = hamming_and_jaccard(B)
print(H)
# [[0 1 2]
#  [1 0 3]
#  [2 3 0]]
print(J)
# [[1.   0.5  0. ]
#  [0.5  1.   0. ]
#  [0.   0.   1. ]]
```

## What the gate checks

The grader computes a reference implementation with NumPy and compares your output.

* The Hamming matrix must match exactly (`np.array_equal`).
* The Jaccard matrix must have a global relative L2 error `rel_err(J, J_ref) ≤ 10⁻¹²`.

If either condition fails the gate returns `exact_match = 0.0`; otherwise it is `1.0`. No other metrics are used.
