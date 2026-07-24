## Context

For a data matrix $X \in \mathbb{R}^{n\times d}$ whose rows are $n$ activation vectors, the empirical covariance matrix is

$$C = \frac{1}{n}\, X^\top X,$$

a symmetric positive‑semidefinite $d\times d$ matrix.  The eigen decomposition of $C$ gives a set of orthonormal eigenvectors $Q \in \mathbb{R}^{d\times d}$ and corresponding eigenvalues $\lambda_1\geqslant\lambda_2\geqslant\dots\geqslant\lambda_d$.  The columns of $Q$ form an orthogonal basis that diagonalises the covariance: $C = Q\, \operatorname{diag}(\lambda)\, Q^\top$.

The eigenvectors are defined only up to sign; multiplying any column by $-1$ leaves the decomposition unchanged.  In practice we sort the eigenvalues in descending order and return the associated eigenvectors in that same order.

## Task

Implement a function `cov_and_eig(X)` that:

```python
def cov_and_eig(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...
```

* `X` is a 2‑D NumPy array of shape `(n, d)` with arbitrary numeric dtype.
* The function returns a tuple `(C, eigvals, eigvecs)` where
  * `C` is the covariance matrix as a float64 array of shape `(d, d)`;
  * `eigvals` is a 1‑D float64 array of length `d`, sorted in **descending** order;
  * `eigvecs` is a 2‑D float64 array of shape `(d, d)` whose columns are the corresponding eigenvectors, also sorted to match `eigvals`.

The implementation must use only NumPy operations; no explicit Python loops.

## Example

```python
import numpy as np
X = np.array([[1, 0], [0, 2], [3, 4]], dtype=np.float64)
C, eigvals, eigvecs = cov_and_eig(X)

# C ≈ [[10.6667, 13.3333],
#      [13.3333, 20.6667]]

# eigvals ≈ [30.0, 1.3333]
# eigvecs columns are orthonormal eigenvectors (sign may differ)
```

## What the gate checks

* **Relative error**: The Euclidean relative error between your returned covariance matrix and the reference computed with NumPy (`C_ref = X.T @ X / n`) must satisfy `rel_err ≤ 1e-9`.  
  The same metric is applied to the eigenvalues; the larger of the two errors is used.

* **Eigenvector consistency**: For each column, your returned eigenvector must match the reference up to sign.  All columns must satisfy this condition for the gate to pass (`eigvec_match == 1.0`).

The grader computes a NumPy oracle on the fly; no hard‑coded expected values are used.
