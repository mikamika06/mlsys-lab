## Context

PCA-style rank-$k$ reconstruction projects data onto a $k$-dimensional
subspace and back, and by the Eckart-Young theorem the *minimum possible*
reconstruction error is achieved by projecting onto the $k$ eigenvectors
of $C = X^\top X/n$ with the **largest** eigenvalues (the directions of
greatest variance). `np.linalg.eigh` returns eigenvalues/eigenvectors in
**ascending** order — a common off-by-direction bug is to slice the
*first* $k$ columns of the returned eigenvector matrix, which are the
**smallest**-eigenvalue directions: this doesn't just fail to minimize
reconstruction error, it actively **maximizes** it among all rank-$k$
orthogonal projections.

The code below has exactly this bug. Fix it.

$$
C = \frac{1}{n}X^\top X \in \mathbb{R}^{d\times d}, \qquad
C = V\Lambda V^\top \ \ (\Lambda \text{ ascending})
$$

Correct rank-$k$ reconstruction: let $V_k \in \mathbb{R}^{d\times k}$ be
the $k$ columns of $V$ with the **largest** eigenvalues (i.e. the *last*
$k$ columns, since $\Lambda$ is ascending):

$$
\hat X = X V_k V_k^\top
$$

## Task

Fix `pca_reconstruct`:

```python
def pca_reconstruct(X: np.ndarray, k: int) -> np.ndarray:
    ...
```

* `X` — `(n, d)` data matrix.
* `k` — target rank, `2 <= k <= d-1`.

Return `X_hat` — the `(n, d)` rank-$k$ reconstruction using the
**largest**-eigenvalue directions of $C = X^\top X/n$, as defined above.

## Example

```python
import numpy as np
rng = np.random.default_rng(0)
X = rng.normal(size=(50, 8)) @ np.diag(rng.uniform(0.3, 3.0, size=8))
X_hat = pca_reconstruct(X, k=3)
# mean((X_hat - X)**2) should be the SMALLEST achievable over all rank-3
# orthogonal projections, not the largest.
```

## What the gate checks

* **recon_max_abs_err** — your `X_hat` must match, element-wise, a NumPy
  oracle that projects onto the `k` largest-eigenvalue directions of
  `C = X^T X / n`, over several random data matrices. (Reconstruction is
  invariant to eigenvector sign, so a correct fix matches exactly up to
  float noise — the buggy version keeps the *smallest*-eigenvalue
  directions and fails this badly.)
