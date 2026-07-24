## Context

The singular value decomposition of a matrix $A \in \mathbb{R}^{m \times n}$ is

$$
A = U \Sigma V^\top ,
$$

where the columns of $U$ and $V$ are orthonormal and $\Sigma$ contains the singular values.

Randomized SVD approximates the dominant singular directions without decomposing the full input matrix. The Halko randomized range finder samples

$$
\Omega \in \mathbb{R}^{n \times (k+p)},
$$

where $p$ is a small oversampling parameter, and computes

$$
Y = A\Omega .
$$

An orthonormal basis $Q$ for the range of $Y$ is found with QR factorization:

$$
Y = QR .
$$

The smaller matrix

$$
B = Q^\top A
$$

is decomposed:

$$
B = \widetilde{U}\Sigma V^\top .
$$

The approximate left singular vectors are

$$
U = Q\widetilde{U}.
$$

This captures the leading singular subspace while avoiding a full SVD of $A$.

## Task

Implement `randomized_svd(A, k, seed)`:

```python
def randomized_svd(A: np.ndarray, k: int, seed: int):
    ...
```

The function receives a real-valued 2-D NumPy array $A$, a target rank $k$, and a random seed. Return `(U, S, Vt)`:

- `U` has shape `(m, k)`.
- `S` has shape `(k,)` and contains approximate top-$k$ singular values in descending order.
- `Vt` has shape `(k, n)`.

Use a Halko-style randomized range finder with deterministic randomness from `seed`. The implementation should use an oversampled random matrix internally and should not call `np.linalg.svd(A)` on the original input matrix.

## Example

```python
import numpy as np

A = np.array([
    [3.0, 0.0],
    [0.0, 2.0],
    [0.0, 0.5],
])

U, S, Vt = randomized_svd(A, 2, 7)

# S is close to [3.0, 2.0]
# U @ np.diag(S) @ Vt approximates A
```

## What the gate checks

The grader uses NumPy's exact SVD as the numerical oracle. It compares the returned singular values and dominant left singular subspace against the oracle.

The singular-value error is

$$
\mathrm{rel\_err} =
\frac{\lVert S_{\mathrm{candidate}}-S_{\mathrm{reference}}\rVert_2}
{\lVert S_{\mathrm{reference}}\rVert_2+10^{-12}} .
$$

The subspace metric is the largest principal angle between the returned $U$ space and the exact top-$k$ left singular vector space. Both metrics must remain below their thresholds.
