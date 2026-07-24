## Context

Low-rank factorization is the cheapest structural compression you can apply to a dense
linear layer. A weight matrix $W \in \mathbb{R}^{m \times n}$ costs $mn$ parameters; if the
spectrum of $W$ decays, most of that is redundant.

The thin singular value decomposition writes

$$
W = U \Sigma V^{\top}, \qquad
U \in \mathbb{R}^{m \times r},\;
\Sigma = \mathrm{diag}(\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r),\;
V \in \mathbb{R}^{n \times r},
$$

with $r = \min(m, n)$. Truncating to the $k$ largest singular triplets gives

$$
W_k \;=\; U_{:,:k}\, \Sigma_{:k,:k}\, V_{:,:k}^{\top}
\;=\; \sum_{i=1}^{k} \sigma_i\, u_i v_i^{\top}.
$$

By the Eckart-Young-Mirsky theorem $W_k$ is the *optimal* rank-$k$ approximation of $W$ in
both the Frobenius and the spectral norm:

$$
\lVert W - W_k \rVert_F^2 \;=\; \sum_{i=k+1}^{r} \sigma_i^2 .
$$

In practice you never store $W_k$ itself — you store the two thin factors

$$
A = U_{:,:k}\,\Sigma_{:k,:k} \in \mathbb{R}^{m \times k},
\qquad
B = V_{:,:k}^{\top} \in \mathbb{R}^{k \times n},
$$

so the layer costs $k(m+n)$ parameters instead of $mn$ and the forward pass becomes two
small matmuls, $x B^{\top} A^{\top}$ instead of $x W^{\top}$. The *round trip* is
compress $W \to (A, B)$, then decompress $(A, B) \to AB \approx W$.

## Task

Implement two functions.

```python
def low_rank_factors(W: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    ...

def low_rank_reconstruct(W: np.ndarray, k: int) -> np.ndarray:
    ...
```

`low_rank_factors(W, k)` returns the pair `(A, B)` where `A` has shape `(m, k)` and `B` has
shape `(k, n)`, taken from the thin SVD of `W` with the singular values folded into `A` as
described above. Both must be `float64`.

`low_rank_reconstruct(W, k)` returns the rank-$k$ approximation $W_k$, a `float64` array of
shape `(m, n)`. It must be consistent with the factors, i.e. $AB = W_k$.

`k` satisfies $1 \le k \le \min(m, n)$. Singular values are ordered from largest to smallest,
which is what `np.linalg.svd` already gives you.

## Example

```python
import numpy as np

W = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0],
              [7.0, 8.0, 9.0]])

A, B = low_rank_factors(W, 1)
A.shape, B.shape          # -> ((3, 1), (1, 3))

W1 = low_rank_reconstruct(W, 1)
np.allclose(A @ B, W1)    # -> True

# W has rank 2, so k = 3 is a lossless round trip:
np.allclose(low_rank_reconstruct(W, 3), W)   # -> True
```

## What the gate checks

The grader recomputes the truncated SVD reconstruction with `numpy.linalg.svd` for several
shapes (tall, wide, square) and several values of $k$, including $k = \min(m, n)$, and
reports:

- `max_abs_err` — worst $\max_{ij} |\hat{W}_{ij} - (W_k)_{ij}|$ over all cases for the value
  returned by `low_rank_reconstruct`. Must be $\le 10^{-8}$.
- `factor_max_abs_err` — the same quantity for the product $AB$ of the returned factors.
  Must be $\le 10^{-8}$. This catches a factorization that reconstructs correctly but stores
  the wrong pieces, for example splitting $\Sigma$ as $\sqrt{\Sigma}$ into both factors while
  reporting a different reconstruction.
- `shape_ok` — `1.0` only if every returned `A` is exactly $(m, k)$ and every `B` is exactly
  $(k, n)$. Must equal `1.0`. Returning full-width factors defeats the compression.

The sign convention of the SVD is irrelevant here: flipping the sign of a $(u_i, v_i)$ pair
leaves both $AB$ and $W_k$ unchanged.
